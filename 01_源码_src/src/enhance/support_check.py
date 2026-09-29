# -*- coding: utf-8 -*-
"""模块1 + 模块6：论断支持度校验 + 证据等级自动判定（默认关闭）。

审查员原话
----------
「把编号存在性校验改进为支持度校验加回检闭环。对正文每个论断句和表格每个单元格，
计算其与所引原文块的蕴含得分。按阈值自动判定证据等级，**替代目前由模型自报的
模型推断标签**。」

本模块怎么落地
--------------
两级判据，**都不问模型**：

  Stage A1（纯确定性，零模型，先跑）
      数字/单位一致性硬门：论断里出现的每个数值必须在所引原文块里出现。
      这正是"引用了一个**存在、但不支持该论断**的条目"最常见的形态——
      模型把 A 论文的数字挂到 B 条目上。存在性校验拦不住，这里能拦。

  Stage A2（本地交叉编码器）
      用 `src.engine.reranker.Reranker`（bge-reranker-v2-m3，本地 ONNX、无网络）
      给 (论断, 原文块) 打相关性分，作为**蕴含得分的代理**。
      取该论断所引各块的最大分。

  合成 support_score = 0.0（数字硬门不过）或 A2 分。
  低于阈值 → 判"不支持" → 交给回检闭环（`recheck.py`）；回检仍失败则降级为
  "原文未记载"并同步编号映射。

Stage A2 是**相关性**而非严格的 NLI 蕴含——这里如实说明：它衡量"原文块是否在讲
这件事"，对"把结论挂到同对象另一条目上"这类错误敏感，对"原文说了相反的话"不敏感。
要更强的判据需换 NLI 模型（需联网下载），当前环境离线，故先用交叉编码器 + 数字硬门。

证据等级（模块6）由 (得分, 原文块里的方法学词) 自动判定，取值与原稿一致：
直接实验 / 遗传证据 / 关联 / 推测 / 原文未定级。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ── 论断切分 ────────────────────────────────────────────────
_REF_RE = re.compile(r"\[(W?\d+|H\d+)\]", re.IGNORECASE)
# 句子边界：中文句号/问号/叹号/分号 + 英文句点（后接空白或行尾）
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？；!?;])\s*|(?<=\.)\s+(?=[A-Z\u4e00-\u9fff])")
# 数字（含百分数、小数、千分位、范围、p 值）
_NUM_RE = re.compile(r"\d+(?:[.,]\d+)*\s*(?:%|‰|mg|g|kg|μg|µg|mL|L|mm|cm|m|nm|"
                     r"d|h|min|s|°C|℃|倍|个|株|份|条|年|月|天|次)?")
# 表格行
_TABLE_ROW_RE = re.compile(r"^\s*\|(.+)\|\s*$")

# ── 方法学词表 → 证据等级（模块6）────────────────────────────
_METHOD_WORDS: list[tuple[str, tuple[str, ...]]] = [
    ("直接实验", ("y2h", "酵母双杂交", "pull-down", "pulldown", "co-ip", "coip",
                  "bifc", "lci", "免疫共沉淀", "互作", "western", "qrt-pcr",
                  "转录组", "酶活", "定点突变", "瞬时表达", "接种", "侵染实验",
                  "knockdown", "过表达", "crispr", "rna-seq", "lc-ms", "hplc")),
    ("遗传证据", ("遗传", "qtl", "gwas", "连锁", "分离比", "杂交后代", "突变体",
                  "转基因", "snp", "多态性", "共线性", "进化")),
    ("关联", ("相关", "关联", "正相关", "负相关", "correlat", "associat",
              "p<0.05", "p < 0.05", "显著差异")),
    ("推测", ("推测", "可能", "很可能", "提示", "暗示", "suggest", "may",
              "might", "putative", "presumably", "需以原文为准")),
]
_LEVEL_ORDER = ("直接实验", "遗传证据", "关联", "推测", "原文未定级")


@dataclass
class Claim:
    """一条待校验的论断。"""
    text: str
    kind: str                                  # "sentence" | "cell"
    ref_ids: list[str] = field(default_factory=list)
    locator: str = ""                          # 便于回写：句子序号 / 表格坐标


@dataclass
class ClaimVerdict:
    claim: Claim
    support_score: float
    supported: bool
    level: str
    reason: str
    best_ref: Optional[str] = None


def split_claims(answer: str) -> list[Claim]:
    """把回答切成论断：表格单元格各算一条，正文按句切、只留**带引用标记**的句子。

    只校验带引用的论断——没有引用的句子属于"缺口/常识"，由既有的时效与口径检查管，
    不在本模块职责内。
    """
    claims: list[Claim] = []
    in_table = False
    for raw in (answer or "").splitlines():
        line = raw.rstrip()
        row = _TABLE_ROW_RE.match(line)
        if row:
            cells = [c.strip() for c in row.group(1).split("|")]
            if all(set(c) <= set("-: ") for c in cells if c):
                continue                                    # 分隔行
            for ci, cell in enumerate(cells):
                if not cell or not _REF_RE.search(cell):
                    continue
                claims.append(Claim(text=cell, kind="cell",
                                    ref_ids=_refs_of(cell),
                                    locator=f"table:r?c{ci}"))
            in_table = True
            continue
        if in_table and not line.strip():
            in_table = False
        if not line.strip() or _REF_RE.search(line) is None:
            continue
        for si, sent in enumerate(s for s in _SENT_SPLIT_RE.split(line) if s.strip()):
            if not _REF_RE.search(sent):
                continue
            claims.append(Claim(text=sent.strip(), kind="sentence",
                                ref_ids=_refs_of(sent),
                                locator=f"line:{si}"))
    return claims


def _refs_of(text: str) -> list[str]:
    seen, out = set(), []
    for m in _REF_RE.finditer(text or ""):
        k = m.group(1).upper()
        if k not in seen:
            seen.add(k)
            out.append(k)
    return out


def _numbers(text: str) -> list[str]:
    """抽出可比对的数值（去掉千分位、统一小写单位）。

    先剥掉引用标记 `[1]` / `[W2]` / `[H1]`——否则引用编号本身会被当成数字，
    导致"原文块里没有 1 这个数"的假报警（自检 EP-4 实测踩到）。
    """
    text = _REF_RE.sub(" ", text or "")
    out = []
    for m in _NUM_RE.finditer(text):
        s = m.group(0).strip().lower().replace(",", "")
        s = re.sub(r"\s+", "", s)
        if s and any(ch.isdigit() for ch in s):
            out.append(s)
    return out


def number_gate(claim_text: str, evidence_text: str) -> tuple[bool, list[str]]:
    """Stage A1 数字硬门：论断里的每个数值都要在所引原文块里出现。

    返回 (是否通过, 缺失的数值)。论断里没有数字则直接通过。
    """
    nums = _numbers(claim_text)
    if not nums:
        return True, []
    hay = (evidence_text or "").lower().replace(",", "")
    hay = re.sub(r"\s+", "", hay)
    missing = [n for n in nums if n not in hay]
    return (not missing), missing


def _reranker_score(claim_text: str, evidence_text: str) -> Optional[float]:
    """Stage A2：本地交叉编码器相关性（作为蕴含代理）。失败返回 None（fail-open）。"""
    try:
        from src.engine.reranker import Reranker
        r = Reranker()
        got = r.rerank(claim_text, [{"text": evidence_text}], top_k=1)
        if not got:
            return None
        v = got[0].get("rerank_score", got[0].get("score"))
        return float(v) if v is not None else None
    except Exception as e:                                     # noqa: BLE001
        logger.debug("[SupportCheck] reranker 不可用，退回纯数字硬门: %s", e)
        return None


def assign_level(score: float, evidence_text: str) -> str:
    """模块6：证据等级自动判定——**由得分与原文里的方法学词决定，不问模型**。"""
    hay = (evidence_text or "").lower()
    for level, words in _METHOD_WORDS:
        if any(w in hay for w in words):
            # 原文只给推测语气时不得升级为"直接实验"（原稿权利要求5 的约束）
            if level == "直接实验" and any(w in hay for w in _METHOD_WORDS[3][1]):
                return "关联" if score >= 0.35 else "推测"
            return level
    return "原文未定级"


def check_claims(claims: list[Claim], text_by_ref: dict[str, str], *,
                 threshold: float = 0.35,
                 require_number_match: bool = True,
                 scorer=None) -> list[ClaimVerdict]:
    """逐条论断算支持度。`scorer(claim, evidence) -> float|None` 可注入（测试用）。"""
    score_fn = scorer or _reranker_score
    out: list[ClaimVerdict] = []
    for c in claims:
        ev = {rid: (text_by_ref.get(rid) or "") for rid in c.ref_ids}
        if not ev:
            out.append(ClaimVerdict(c, 0.0, False, "原文未定级",
                                    "所引编号在证据池中无对应原文块"))
            continue
        best_ref, best_score, reason = None, -1.0, ""
        gate_failed: list[str] = []
        for rid, txt in ev.items():
            ok, missing = (number_gate(c.text, txt) if require_number_match
                           else (True, []))
            if not ok:
                gate_failed.extend(missing)
                continue
            s = score_fn(c.text, txt)
            if s is None:
                # 交叉编码器不可用：数字硬门通过即视为弱支持
                s = 1.0 if require_number_match else 0.0
                reason = reason or "仅数字硬门（交叉编码器不可用）"
            if s > best_score:
                best_ref, best_score = rid, float(s)
        if best_ref is None:
            out.append(ClaimVerdict(
                c, 0.0, False, "原文未定级",
                f"数字硬门不过：所引块未出现 {sorted(set(gate_failed))[:3]}"))
            continue
        supported = best_score >= threshold
        lvl = assign_level(best_score, ev[best_ref]) if supported else "原文未定级"
        out.append(ClaimVerdict(c, best_score, supported, lvl,
                                reason or f"支持度 {best_score:.2f}"
                                f"{'≥' if supported else '<'} 阈值 {threshold}",
                                best_ref))
    return out


def run_support_check(answer: str, cited_refs: list, *,
                      threshold: float = 0.35,
                      require_number_match: bool = True,
                      scorer=None) -> dict:
    """入口：返回可直接落日志/前端事件的结构化报告。

    关闭时**根本不该被调用**（调用方先看 `FLAGS.support_check`）。
    """
    text_by_ref = {}
    for it in cited_refs or []:
        rid = str(it.get("ref_id", "")).upper()
        if not rid:
            continue
        txt = (it.get("text_preview") or it.get("text") or it.get("snippet")
               or it.get("abstract") or "")
        text_by_ref[rid] = str(txt)
    claims = split_claims(answer)
    verdicts = check_claims(claims, text_by_ref, threshold=threshold,
                            require_number_match=require_number_match,
                            scorer=scorer)
    unsupported = [v for v in verdicts if not v.supported]
    levels: dict[str, int] = {k: 0 for k in _LEVEL_ORDER}
    for v in verdicts:
        levels[v.level] = levels.get(v.level, 0) + 1
    return {
        "n_claims": len(verdicts),
        "n_unsupported": len(unsupported),
        "threshold": threshold,
        "level_counts": levels,
        "claims": [{"text": v.claim.text[:200], "kind": v.claim.kind,
                    "ref_ids": v.claim.ref_ids, "locator": v.claim.locator,
                    "score": round(v.support_score, 4), "supported": v.supported,
                    "level": v.level, "reason": v.reason, "best_ref": v.best_ref}
                   for v in verdicts],
        "unsupported": [{"text": v.claim.text[:200], "ref_ids": v.claim.ref_ids,
                         "locator": v.claim.locator,
                         "score": round(v.support_score, 4), "reason": v.reason,
                         "level": v.level, "best_ref": v.best_ref}
                        for v in unsupported],
    }
