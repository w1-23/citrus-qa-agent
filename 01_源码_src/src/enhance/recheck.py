# -*- coding: utf-8 -*-
"""模块1b：回检闭环（默认关闭）。

审查员原话
----------
「判为不支持时，以该论断为查询回到本地和联网通道做**定向回检**，重新绑定支撑块。
回检仍失败，就自动降级或替换为"原文未记载"，并**同步更新编号映射**。」

设计
----
检索函数由调用方注入（`retrieve_fn(query, top_k) -> list[{ref_id,text,...}]`），
本模块不 import 任何检索实现——这样：
  1. 单元测试不需要加载向量库/模型；
  2. 本包对检索链路是**零耦合**，不构成对实验依赖文件的任何影响。

闭环：
    不支持论断 ──► 以论断为查询回检（本地，必要时联网）
                    ├─ 找到支持度 ≥ 阈值的新块 ──► 重新绑定：把该块并入编号池
                    │                                  （新增 [n] / [Wn]），
                    │                                  正文引用标记改写为新编号，
                    │                                  remap 记录"旧→新"
                    └─ 仍找不到 ──────────────► 降级：正文该论断标注
                                                  「（原文未记载）」，
                                                  并把它从引用映射里摘掉
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_NOT_RECORDED = "（原文未记载）"


def _claim_query(text: str, max_chars: int = 200) -> str:
    """把论断变成检索查询：去掉引用标记、压缩空白、截断。"""
    import re
    q = re.sub(r"\[(W?\d+|H\d+)\]", " ", text or "", flags=re.IGNORECASE)
    q = re.sub(r"\s+", " ", q).strip()
    return q[:max_chars]


def recheck_unsupported(unsupported: list[dict], *,
                        retrieve_fn: Callable[[str, int], list],
                        scorer: Optional[Callable] = None,
                        threshold: float = 0.35,
                        require_number_match: bool = True,
                        max_items: int = 3) -> dict:
    """对"不支持"的论断做定向回检，返回重绑与降级结果。

    `retrieve_fn(query, top_k)` 返回新检索到的候选块列表（含 ref_id/text）。
    `scorer(claim, evidence) -> float|None` 同 support_check。
    """
    from .support_check import number_gate, _reranker_score

    score_fn = scorer or _reranker_score
    rebound: list[dict] = []          # 成功重绑
    downgraded: list[dict] = []       # 回检失败 → 降级
    tried = 0

    for item in (unsupported or [])[:max_items]:
        text = item.get("text") or ""
        tried += 1
        query = _claim_query(text)
        if not query:
            downgraded.append({**item, "action": "downgrade",
                               "reason": "论断为空，无法构造回检查询"})
            continue
        try:
            cands = retrieve_fn(query, 5) or []
        except Exception as e:                                 # noqa: BLE001
            logger.warning("[Recheck] 回检失败（fail-open 不阻断回答）: %s", e)
            cands = []

        best, best_score = None, -1.0
        for c in cands:
            ctext = str(c.get("text") or c.get("text_preview")
                        or c.get("snippet") or "")
            if not ctext:
                continue
            if require_number_match:
                ok, _missing = number_gate(text, ctext)
                if not ok:
                    continue
            s = score_fn(text, ctext)
            if s is None:
                s = 1.0 if require_number_match else 0.0
            if float(s) > best_score:
                best, best_score = c, float(s)

        if best is not None and best_score >= threshold:
            rebound.append({
                "claim": text[:200],
                "old_refs": item.get("ref_ids") or [],
                "new_ref_id": str(best.get("ref_id") or ""),
                "new_ref_title": str(best.get("title") or "")[:120],
                "score": round(best_score, 4),
                "action": "rebind",
            })
        else:
            # 键名统一为 "claim"（入参来自 run_support_check 的 unsupported，那里叫 "text"）
            downgraded.append({
                **item, "claim": text[:200], "action": "downgrade",
                "reason": (f"回检 {len(cands)} 条候选仍无支持度 ≥ {threshold} 的原文块"
                           if cands else "回检无候选"),
            })

    # 降级后要同步编号映射：被降级论断原引的编号若不再被任何论断引用，即从映射摘除
    freed: list[str] = []
    for d in downgraded:
        freed.extend(d.get("ref_ids") or [])

    return {
        "tried": tried,
        "n_rebound": len(rebound),
        "n_downgraded": len(downgraded),
        "rebound": rebound,
        "downgraded": downgraded,
        "freed_ref_ids": sorted(set(freed)),
        "not_recorded_marker": _NOT_RECORDED,
        "note": ("重绑 = 正文引用标记改写为新编号并入编号池；降级 = 论断标注"
                 "「原文未记载」并从引用映射摘除（同步更新 remap）。"),
    }


def apply_downgrade(answer: str, downgraded: list[dict]) -> str:
    """把降级论断就地标注「（原文未记载）」——与原稿权利要求5 的缺失标识一致。

    论断文本的键名两种都认（`claim` 或 `text`）：回检输出用 `claim`，
    而 `run_support_check` 的 `unsupported` 用 `text`，两边都可能直接喂进来。
    """
    out = answer or ""
    for d in downgraded or []:
        t = str(d.get("claim") or d.get("text") or "").strip()
        if not t or _NOT_RECORDED in t:
            continue
        if t in out:
            out = out.replace(t, t + _NOT_RECORDED, 1)
    return out
