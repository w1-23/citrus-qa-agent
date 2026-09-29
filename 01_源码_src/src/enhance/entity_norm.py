# -*- coding: utf-8 -*-
"""模块3：柑橘领域实体归一（默认关闭）。

审查员原话
----------
「引入柑橘领域实体归一，使领域限定有技术意义。建立 CLas 基因座号、效应子别名、
柑橘宿主基因同源命名的映射，例如 **CLIBASIA_02470、SDE2470、SDE5 指向同一对象**。
将其用于三处：查询扩展、跨来源证据的对象对齐（作为表格行主键）、冲突检测。
冲突检测的做法是：从各来源抽取效应子-靶标-作用关系-验证手段结构化关系，
**同一对象对出现矛盾时自动识别**，而不是依赖提示词让模型发现。」

本模块做什么
------------
1. `normalize(text)`      —— 把任意写法归到规范对象 ID（CLas 效应子 / 柑橘宿主基因 / 品种）
2. `expand_query(q)`      —— 查询扩展：一个别名命中，把同对象的全部别名一并加入检索
3. `object_key(text)`     —— 表格行主键：跨来源的同一对象落到同一行
4. `detect_conflicts()`   —— 冲突检测：抽取 (效应子, 靶标, 作用, 验证手段) 四元组，
                             同一对象对出现矛盾即自动识别（**不靠提示词**）

映射表来源
----------
`agent/enhance_entities.yaml`（可由维护者增补；本模块 import 时自动合并进内置表）。

**当前规模：26 个规范对象 / 93 条别名**（内置 21/85 + 语料验证的 13 个 CLas 效应子）。
效应子条目逐条带 DOI 出处与 NCBI GeneID，由三步法构建（脚本在
`experiment/mechanism/scripts/`）：
    ① `entity_extract_candidates.py`  正则抽取候选（187,441 块 → 1,797 个候选）
    ② `entity_pairs.py`               语料同位语/别名句式取证（→ 134 个候选对）
       `entity_cluster_llm.py`        依据出处句子做证据接地判定（含负对照校验）
    ③ `entity_validate.py`            NCBI 标识符存在性核验 → 正式表 / 待确认表
       `entity_merge_table.py`        与内置表合并 → 本 YAML

**一条重要事实**：SDE 编号与基因座号的对应关系**不在任何公开数据库里**
（实测 NCBI gene/protein 与 UniProt 查询均 0 条，柑橘基因组库无公开 API），
所以映射证据来自**已发表文献的明确陈述**（如 "SDE5 (CLIBASIA_02470, designated
as SDE2470 in this study)"），数据库只用于核验标识符真实存在。

**这是领域技术特征**——换一个领域，这张表与四元组抽取规则都不成立。
"""
from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

# v6（B1b）：enhance_entities.yaml 已搬到 02_配置_config/
AGENT_DIR = Path(__file__).resolve().parents[3]          # 仓库根
YAML_PATH = AGENT_DIR / "02_配置_config" / "enhance_entities.yaml"

# ── 内置最小映射（可被 enhance_entities.yaml 覆盖/扩充）────────────
# 结构：规范 ID → {"type":..., "aliases":[...], "locus":..., "host":...}
BUILTIN: dict[str, dict] = {
    # CLas Sec 依赖效应子：基因座号 / SDE 编号 / 文献别名 → 同一对象
    "CLas.SDE2470": {"type": "effector", "locus": "CLIBASIA_02470",
                     "aliases": ["CLIBASIA_02470", "SDE2470", "SDE5", "SDE 2470",
                                 "sde2470", "SDE-2470"],
                     "host": ["CsVOZ2", "MYC2"], "note": "PUB21/26S 降解 MYC2；BRUTUS 促 CsVOZ2 降解"},
    "CLas.SDE1": {"type": "effector", "locus": "CLIBASIA_05315",
                  "aliases": ["CLIBASIA_05315", "SDE1", "SDE 1"],
                  "host": ["CsPLCP", "AALP", "XBCP3", "SAG12"]},
    "CLas.SDE3": {"type": "effector", "locus": "CLIBASIA_00420",
                  "aliases": ["CLIBASIA_00420", "SDE3", "SDE 3"],
                  "host": ["CsGAPC", "GAPDH"]},
    "CLas.SDE15": {"type": "effector", "locus": "CLIBASIA_04025",
                   "aliases": ["CLIBASIA_04025", "SDE15", "SDE 15"],
                   "host": ["ACD2"]},
    "CLas.SDE19": {"type": "effector", "locus": "CLIBASIA_05320",
                   "aliases": ["CLIBASIA_05320", "SDE19", "SDE 19"],
                   "host": ["Sec12", "Sar1"], "note": "ER 干扰 COPII 囊泡转运"},
    "CLas.SDE4405": {"type": "effector", "locus": "CLIBASIA_04405",
                     "aliases": ["CLIBASIA_04405", "SDE4405", "SDE 4405"]},
    "CLas.SDE525": {"type": "effector", "aliases": ["SDE525", "SDE 525"],
                    "host": ["NACα", "NACa"]},
    "CLas.SDE4250": {"type": "effector", "aliases": ["SDE4250", "SDE 4250"],
                     "host": ["CsRPN10"]},
    "CLas.SECP8": {"type": "effector", "aliases": ["SECP8", "mSECP8", "SECP 8"],
                   "host": ["CsTCP15"]},
    "CLas.AGH17488": {"type": "effector", "aliases": ["AGH17488"],
                      "host": ["APX6"], "note": "前噬菌体区编码"},
    "CLas.FlgI": {"type": "effector", "aliases": ["FlgI", "FLGI"]},
    "CLas.LasP235": {"type": "effector", "aliases": ["LasP235", "LasP 235"]},
    # 宿主基因（柑橘）
    "Cs.Sec12": {"type": "host", "aliases": ["Sec12", "CsSec12", "甜橙Sec12"],
                 "ortholog": {"Arabidopsis": "AtSec12", "yeast": "Sec12p"}},
    "Cs.VOZ2": {"type": "host", "aliases": ["CsVOZ2", "VOZ2"],
                "ortholog": {"Arabidopsis": "AtVOZ2"}},
    "Cs.MYC2": {"type": "host", "aliases": ["MYC2", "CsMYC2"],
                "ortholog": {"Arabidopsis": "AtMYC2", "tomato": "SlMYC2"}},
    "Cs.TCP15": {"type": "host", "aliases": ["CsTCP15", "TCP15"]},
    "Cs.RPN10": {"type": "host", "aliases": ["CsRPN10", "RPN10"]},
    "Cs.APX6": {"type": "host", "aliases": ["APX6", "抗坏血酸过氧化物酶6"]},
    "Cs.GAPC": {"type": "host", "aliases": ["CsGAPC", "GAPC", "GAPDH"]},
    "Cs.ACD2": {"type": "host", "aliases": ["ACD2", "accelerated cell death 2"]},
    "Cs.NACa": {"type": "host", "aliases": ["NACα", "NACa", "NAC-alpha"]},
}

# 物种别名（用于跨来源对象对齐时的范围判定）
SPECIES_ALIASES = {
    "CLas": ["Candidatus Liberibacter asiaticus", "CLas", "Las", "亚洲韧皮部杆菌"],
    "Cs": ["Citrus sinensis", "甜橙", "脐橙", "Citrus aurantium", "酸橙"],
    "Xcc": ["Xanthomonas citri", "柑橘溃疡病菌"],
    "Pst": ["Pseudomonas syringae"],
}

_ALIAS2ID: dict[str, str] = {}


def _build_index() -> None:
    _ALIAS2ID.clear()
    for cid, spec in BUILTIN.items():
        _ALIAS2ID[cid.lower()] = cid
        for a in spec.get("aliases", []) or []:
            _ALIAS2ID[str(a).strip().lower()] = cid
    for cid, aliases in SPECIES_ALIASES.items():
        for a in aliases:
            _ALIAS2ID.setdefault(str(a).strip().lower(), cid)


def _load_yaml() -> None:
    """可选扩充：`agent/enhance_entities.yaml` 的 entities 段合并进 BUILTIN。"""
    if not YAML_PATH.exists():
        return
    try:
        import yaml
        d = yaml.safe_load(YAML_PATH.read_text(encoding="utf-8")) or {}
        for cid, spec in (d.get("entities") or {}).items():
            BUILTIN.setdefault(cid, {}).update(spec or {})
    except Exception as e:                                     # noqa: BLE001
        logger.warning("[EntityNorm] enhance_entities.yaml 读取失败（用内置表）: %s", e)


_build_index()
_load_yaml()
_build_index()


def normalize(text: str) -> list[str]:
    """把一段文本里出现的所有别名归一到规范 ID（去重、保序）。"""
    t = (text or "").lower()
    hits: list[str] = []
    for alias, cid in _ALIAS2ID.items():
        if not alias:
            continue
        if re.search(r"(?<![a-z0-9])" + re.escape(alias) + r"(?![a-z0-9])", t):
            if cid not in hits:
                hits.append(cid)
    return hits


def expand_query(query: str, max_terms: int = 8) -> dict:
    """查询扩展：命中的对象 → 把同对象的**其他别名**一并加入检索词。

    例：查询里只有 `SDE5` → 自动补 `CLIBASIA_02470` / `SDE2470`，
    使只写基因座号的文献也能被召回。
    """
    ids = normalize(query)
    added: list[str] = []
    for cid in ids:
        spec = BUILTIN.get(cid, {})
        for a in list(spec.get("aliases") or []) + list(spec.get("host") or []):
            a = str(a)
            if a.lower() not in (query or "").lower() and a not in added:
                added.append(a)
    added = added[:max_terms]
    return {"query": query, "objects": ids, "added_terms": added,
            "expanded": (query + " " + " ".join(added)).strip() if added else query}


def object_key(text: str) -> Optional[str]:
    """表格行主键：把同一对象的不同写法收敛到同一 key（跨来源对齐）。"""
    ids = [i for i in normalize(text) if not i.startswith(("CLas", "Cs")) is False]
    ids = normalize(text)
    for cid in ids:
        if BUILTIN.get(cid, {}).get("type") == "effector":
            return cid
    return ids[0] if ids else None


# ── 冲突检测：四元组抽取 + 同对象对矛盾识别 ──────────────────────
_REL_PATTERNS: list[tuple[str, str]] = [
    ("抑制", r"(抑制|负调控|阻断|干扰|降低|下调|repress|inhibit|suppress)"),
    ("促进", r"(促进|正调控|增强|提高|上调|激活|promote|enhance|activate)"),
    ("互作", r"(互作|结合|相互作用|interact|bind)"),
    ("降解", r"(降解|泛素化|degrada|ubiquitin)"),
]
_METHOD_HINTS = ("y2h", "pull-down", "co-ip", "bifc", "lci", "biFC".lower(),
                 "酵母", "酶活", "转录组", "western", "qrt-pcr", "定点突变")


def extract_relations(text: str) -> list[dict]:
    """从一段原文/论断里抽 (效应子, 宿主靶标, 作用, 验证手段) 四元组。"""
    ids = normalize(text)
    eff = [i for i in ids if BUILTIN.get(i, {}).get("type") == "effector"]
    host = [i for i in ids if BUILTIN.get(i, {}).get("type") == "host"]
    if not eff or not host:
        return []
    low = (text or "").lower()
    action = next((name for name, pat in _REL_PATTERNS if re.search(pat, low)), "未明")
    methods = sorted({m for m in _METHOD_HINTS if m in low})
    return [{"effector": e, "host": h, "action": action,
             "methods": methods, "text": (text or "")[:200]}
            for e in eff for h in host]


def detect_conflicts(claims: Iterable[dict]) -> list[dict]:
    """同一 (效应子, 宿主) 对出现**方向相反**的作用时自动识别为冲突。

    `claims` 每项形如 {"text": "...", "source": "rag|web", "ref_ids": [...]}。
    **不依赖提示词**——纯查表 + 正则，确定性。
    """
    by_pair: dict[tuple, list[dict]] = {}
    for c in claims or []:
        for rel in extract_relations(c.get("text") or ""):
            by_pair.setdefault((rel["effector"], rel["host"]), []).append(
                {**rel, "source": c.get("source", ""),
                 "ref_ids": c.get("ref_ids") or []})
    out: list[dict] = []
    for (e, h), rels in by_pair.items():
        acts = {r["action"] for r in rels}
        if {"抑制", "促进"} <= acts:
            out.append({
                "object": f"{e} → {h}",
                "kind": "action_conflict",
                "detail": "同一对象对在不同来源/论断里被描述为既抑制又促进",
                "items": rels,
            })
        methods = {m for r in rels for m in r["methods"]}
        if len(rels) > 1 and not methods:
            out.append({
                "object": f"{e} → {h}", "kind": "method_missing",
                "detail": "多处提及该对象对但均未标注实验方法（无法定级）",
                "items": rels,
            })
    return out


def stats() -> dict:
    return {"n_objects": len(BUILTIN), "n_aliases": len(_ALIAS2ID),
            "yaml": str(YAML_PATH) if YAML_PATH.exists() else None}
