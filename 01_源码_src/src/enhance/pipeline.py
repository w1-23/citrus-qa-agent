# -*- coding: utf-8 -*-
"""增强包总入口：生成后处理编排（默认关闭）。

`run_postprocess()` 把六个模块串成一条链，**只在开关打开时**才做任何事：

    回答 + cited_refs
      │
      ├─ 模块2 联网原文块   web_fulltext.build_web_blocks()   → 联网条目升级为块（可选）
      ├─ 模块3 实体归一     entity_norm.expand_query()        → 查询扩展（可选）
      │        └ 冲突检测   entity_norm.detect_conflicts()    → 同对象对矛盾自动识别
      ├─ 模块4 块级去重     dedup_chunk.dedup_with_positions() → 粒度可选（默认 doi 不变）
      ├─ 模块1 支持度校验   support_check.run_support_check()  → 逐论断打分 + 等级
      │        └ 模块6      （等级由得分判定，替代模型自报）
      └─ 模块1b 回检闭环    recheck.recheck_unsupported()      → 重绑 / 降级「原文未记载」

返回值直接挂进 `audit_report["enhance"]`，随既有审计链路落日志与前端事件。

**默认关闭时的成本 = 一次 `FLAGS.any_on()` 布尔判断**，不做任何 IO、不加载任何模型。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

from .flags import FLAGS

logger = logging.getLogger(__name__)


def enrich_with_full_text(cited_refs: list,
                          main_results: Optional[list]) -> tuple[list, dict]:
    """用 `deduped_main` 的**全文**补进 cited_refs，产出 `text_full` 字段。

    为什么必须补（2026-09-22 实测缺陷）
    ----------------------------------
    `cited_refs.text_preview` 被截断到 **300 字符**（`agent_loop.py:89`）。
    实测两块真实语料：

        | 截断      | 块A 四元组 | 块B 四元组 |
        |----------|-----------|-----------|
        | 前 300 字 | 0         | 0         |   ← 真实链路用的
        | 前 1500 字| 1         | 10        |

    效应子首次出现在第 **372 / 975** 字符处，远超 300 字窗口 →
    `extract_relations` 抽不到四元组 → **冲突检测在真实链路上空转**。

    对应关系：重排后的 cited_refs 是 `deduped_main[:20]` 的**子集**且 ref_id 已变，
    故按 `(doi, title)` 匹配，**不能按下标对应**。

    返回 (增强后的 cited_refs, 统计)。
    """
    if not main_results:
        return cited_refs, {"enriched": 0, "reason": "未提供 main_results"}
    idx: dict[tuple, str] = {}
    for r in main_results:
        if not isinstance(r, dict):
            continue
        full = str(r.get("text") or r.get("abstract") or r.get("snippet") or "")
        if not full:
            continue
        doi = str(r.get("doi") or "").strip().lower()
        title = str(r.get("title") or r.get("name") or "").strip()
        if doi:
            idx.setdefault(("d", doi), full)
        if title:
            idx.setdefault(("t", title), full)
    out, n = [], 0
    for c in (cited_refs or []):
        d = dict(c)
        doi = str(c.get("doi") or "").strip().lower()
        title = str(c.get("title") or "").strip()
        full = (idx.get(("d", doi)) if doi else None) or (idx.get(("t", title)) if title else None)
        if full:
            d["text_full"] = full
            n += 1
        out.append(d)
    return out, {"enriched": n, "n_refs": len(out or []),
                 "note": "text_full 来自 deduped_main 的 text 字段（非 300 字预览）"}


def run_postprocess(answer: str, cited_refs: list, *,
                    web_items: Optional[list] = None,
                    retrieve_fn: Optional[Callable] = None,
                    fetch_fn: Optional[Callable] = None,
                    scorer: Optional[Callable] = None,
                    query: str = "",
                    main_results: Optional[list] = None) -> dict:
    """生成后处理。关闭时立刻返回 `{"enabled": False}`，零副作用。

    `main_results` = `deduped_main`（含 `text` 全文字段）。提供时会把全文补进
    cited_refs 的 `text_full`，供实体归一等模块使用——否则它们只能看到
    300 字预览，实体对常被截断（详见 `enrich_with_full_text` 的实测表）。
    """
    if not FLAGS.any_on():
        return {"enabled": False}

    rep: dict = {"enabled": True, "flags": FLAGS.as_dict()}

    # ── 前置：补全文（修 300 字预览导致的实体对截断）──
    if main_results:
        cited_refs, _st = enrich_with_full_text(cited_refs, main_results)
        rep["full_text"] = _st

    # ── 模块2：联网原文块 ──
    if FLAGS.web_fulltext and web_items:
        try:
            from .web_fulltext import build_web_blocks, to_cited_ref
            blocks, stat = build_web_blocks(web_items,
                                            max_items=int(FLAGS.web_fulltext_max),
                                            fetch_fn=fetch_fn)
            rep["web_fulltext"] = stat
            # 用块级条目替换原联网条目（侧栏预览因此变成"模型依据的那一块"）
            if stat.get("n_blocks"):
                keep = [c for c in (cited_refs or [])
                        if str(c.get("type")) != "web"]
                keep += [to_cited_ref(b) for b in blocks]
                cited_refs = keep
        except Exception as e:                                 # noqa: BLE001
            logger.warning("[Enhance] 模块2 失败（fail-open，保留原联网条目）: %s", e)
            rep["web_fulltext"] = {"error": repr(e)}

    # ── 模块3：实体归一（**默认只做对齐 + 冲突检测，不做查询扩展**）──
    # 依据：机制实验 exp3 证明查询扩展会把 gold 挤出，故 query_expansion 默认关闭。
    if FLAGS.entity_norm:
        try:
            from . import entity_norm as E
            _objs = E.normalize(query) if query else []

            def _full(c: dict) -> str:
                """优先用全文（text_full），退化到 300 字预览。"""
                return str(c.get("text_full") or c.get("text_preview") or "")

            # 跨来源对象对齐：每条证据 → 规范对象 ID（表格行主键）
            align = []
            for c in (cited_refs or []):
                txt = _full(c)
                oid = E.object_key(txt) if txt else None
                if oid:
                    align.append({"ref_id": str(c.get("ref_id")), "object": oid,
                                  "source": c.get("source", "")})
            rep["entity_norm"] = {
                "objects_in_query": _objs,
                "alignment": align if FLAGS.entity_alignment else [],
                "n_aligned": len(align),
                "conflicts": (E.detect_conflicts(
                    [{"text": _full(c), "source": c.get("source", ""),
                      "ref_ids": [str(c.get("ref_id"))]}
                     for c in (cited_refs or [])])
                    if FLAGS.entity_conflict else []),
                "stats": E.stats(),
                "query_expansion": False,
                "used_full_text": sum(1 for c in (cited_refs or [])
                                      if c.get("text_full")),
            }
            if FLAGS.entity_query_expansion:
                rep["entity_norm"]["expand"] = E.expand_query(query)
                rep["entity_norm"]["query_expansion"] = True
        except Exception as e:                                 # noqa: BLE001
            logger.warning("[Enhance] 模块3 失败（fail-open）: %s", e)
            rep["entity_norm"] = {"error": repr(e)}

    # ── 模块4：块级去重（默认 doi，不改口径）──
    gran = str(FLAGS.dedup_granularity or "doi").lower()
    if gran == "chunk":
        try:
            from .dedup_chunk import dedup_with_positions, dedup_granularity_note
            cited_refs, notes = dedup_with_positions(cited_refs, "chunk")
            rep["dedup"] = {"granularity": "chunk", "n_after": len(cited_refs),
                            "positions": notes[:20],
                            "note": dedup_granularity_note("chunk")}
        except Exception as e:                                 # noqa: BLE001
            logger.warning("[Enhance] 模块4 失败（fail-open）: %s", e)
            rep["dedup"] = {"error": repr(e)}

    # ── 模块1 + 6：支持度校验 + 证据等级自动判定 ──
    if FLAGS.support_check:
        try:
            from .support_check import run_support_check
            sc = run_support_check(answer, cited_refs,
                                   threshold=float(FLAGS.support_threshold),
                                   require_number_match=bool(FLAGS.require_number_match),
                                   scorer=scorer)
            rep["support_check"] = sc

            # ── 模块1b：回检闭环 ──
            if FLAGS.recheck and sc.get("unsupported") and retrieve_fn is not None:
                from .recheck import recheck_unsupported, apply_downgrade
                rc = recheck_unsupported(
                    sc["unsupported"], retrieve_fn=retrieve_fn,
                    scorer=scorer, threshold=float(FLAGS.support_threshold),
                    require_number_match=bool(FLAGS.require_number_match),
                    max_items=int(FLAGS.recheck_max))
                rep["recheck"] = rc
                if rc.get("downgraded"):
                    answer = apply_downgrade(answer, rc["downgraded"])
                    rep["answer_after_downgrade"] = True
        except Exception as e:                                 # noqa: BLE001
            logger.warning("[Enhance] 模块1/1b 失败（fail-open，不阻断回答）: %s", e)
            rep["support_check"] = {"error": repr(e)}

    rep["answer"] = answer          # 可能被降级标注改写
    return rep
