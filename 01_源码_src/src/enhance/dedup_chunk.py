# -*- coding: utf-8 -*-
"""模块4：块级去重（默认关闭；默认仍按 DOI，与实验口径一致）。

审查员原话
----------
「去重粒度与展示粒度冲突。本地证据是文本块级，去重却按 DOI 进行。同一论文的多个
文本块会被合并为一条，侧栏展示模型实际依据的文本块原文就无法确定是哪一块。
碰撞时保留正文更丰富条目又与保持首次出现顺序在**位置归属**上未作交代。」

两件事
------
1. **块级去重**（本模块）：`CITRUS_DEDUP_GRANULARITY=chunk` 时，去重键用
   `paper_id:chunk_index`（缺省退化为 `(doi, text 前 N 字哈希)`）——
   同一论文的不同块各自成条，侧栏"展示的就是模型依据的那一块"这一不变量才成立。
   默认 `doi` 时**行为与 v9.6.3 完全一致**（不改实验口径）。

2. **位置归属交代**（代码里本来就有的语义，只是原稿没写）：
   碰撞时保留**首次出现的位置**、内容替换为**正文更丰富**的那一条
   （`evidence.dedup_evidence_items` 的 `out[i] = r`）。本模块把它显式化为
   `dedup_with_positions()` 并在返回值里标注 `position_from`，便于写进专利说明书。
"""
from __future__ import annotations

import hashlib
import logging
from typing import Iterable

logger = logging.getLogger(__name__)


def _chunk_key(r: dict) -> tuple:
    """块级去重键：优先 (paper_id, chunk_index)，退化到 (doi, 文本哈希)。"""
    pid = str(r.get("paper_id") or "").strip()
    ci = r.get("chunk_index")
    if pid and ci is not None:
        return ("pc", pid, str(ci))
    doi = str(r.get("doi") or "").strip().lower()
    txt = str(r.get("text") or r.get("abstract") or r.get("snippet") or "")
    h = hashlib.sha1(txt[:2000].encode("utf-8", "ignore")).hexdigest()[:16]
    return ("dh", doi, h)


def _doi_key(r: dict) -> tuple:
    """DOI 级去重键（v9.6.3 原口径：DOI → 标题）。"""
    doi = str(r.get("doi") or "").strip().lower()
    if doi:
        return ("d", doi)
    return ("t", str(r.get("title") or "").strip().lower()[:80])


def _richness(r: dict) -> tuple:
    """正文丰富度（与原实现同口径：有 text > 只有摘要/片段 > 长度）。"""
    text = str(r.get("text") or "").strip()
    fallback = str(r.get("abstract") or r.get("snippet") or "").strip()
    return (bool(text), bool(fallback), len(text or fallback))


def dedup_with_positions(items: Iterable[dict], granularity: str = "chunk") -> tuple:
    """按指定粒度去重，并显式记录"位置来自谁、内容来自谁"。

    返回 (去重后的列表, 交代信息列表)。
    交代信息每条形如：
        {"key": "...", "position_from": 首个出现的下标, "content_from": 被采纳的下标,
         "replaced": bool}
    —— 这正是原稿"位置归属未作交代"缺的那句话。
    """
    keyfn = _chunk_key if granularity == "chunk" else _doi_key
    out: list[dict] = []
    seen: dict[tuple, int] = {}
    notes: list[dict] = []
    for r in items or []:
        if not isinstance(r, dict):
            continue
        k = keyfn(r)
        if not any(k[1:]):
            continue
        if k in seen:
            i = seen[k]
            prev = out[i]
            if _richness(r) > _richness(prev):
                out[i] = r
                notes.append({"key": ":".join(map(str, k)), "position_from": i,
                              "content_from": i, "replaced": True,
                              "rule": "位置=首次出现；内容=正文更丰富者"})
            continue
        seen[k] = len(out)
        out.append(r)
        notes.append({"key": ":".join(map(str, k)), "position_from": len(out) - 1,
                      "content_from": len(out) - 1, "replaced": False,
                      "rule": "首次出现"})
    return out, notes


def dedup_granularity_note(granularity: str) -> str:
    return ("块级去重（paper_id:chunk_index）：同一论文的不同块各自成条，"
            "侧栏展示的即模型实际依据的那一块"
            if granularity == "chunk" else
            "文献级去重（DOI → 标题）：与 v9.6.3 / 三个机制实验口径一致")
