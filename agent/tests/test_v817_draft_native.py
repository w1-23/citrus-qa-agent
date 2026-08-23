# -*- coding: utf-8 -*-
"""v8.17 原生回答草稿 + UCR 优先回归：

  VF-39  品种意图检测（_is_variety_intent）+ UCR 优先检索接线（search_multi ucr_boost）
  VF-40  draft_worker 联网路径（原生联网回答草稿 + 二级提取 + [Wn]/落仓）与非联网路径
  VF-41  回执「原生回答参考」段 + 提示词 UCR 优先标记 + snapshot 新增结构

全部离线、无模型、无网络（调用打桩；接线断言读源码）。
"""
import sys
import os
import asyncio
import json
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings

passed, failed = [], []
ROOT = Path(__file__).resolve().parents[1]


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ── VF-39 品种意图 + UCR 优先检索 ─────────────────────────────────
def test_v817_variety_intent():
    print("[VF-39] 品种意图检测 + UCR 优先接线")
    import src.tools.search as st
    from src.retrieval import multi_retriever as mr

    check("中文'品种'命中", st._is_variety_intent("柑橘品种特性与来源") is True)
    check("UCR 命中", st._is_variety_intent("UCR 库有哪些柑橘品种") is True)
    check("cultivar 命中", st._is_variety_intent("Kinnow cultivar traits") is True)
    check("CRC 命中", st._is_variety_intent("CRC 3585 Kishu Ponkan") is True)
    check("非品种（机制）不命中", st._is_variety_intent("柑橘黄龙病防控机制") is False)
    check("非品种（病害）不命中", st._is_variety_intent("citrus canker control") is False)
    check("空查询不命中", st._is_variety_intent("") is False)

    se = (ROOT / "src/tools/search.py").read_text(encoding="utf-8")
    check("citrus_rag_search 接线 ucr_boost（结构化路径）",
          "_ucr_boost = _is_variety_intent(query)" in se
          and "rag.search_multi(queries, ucr_boost=_ucr_boost)" in se)
    mr_src = (ROOT / "src/retrieval/multi_retriever.py").read_text(encoding="utf-8")
    check("search_multi 支持 ucr_boost 参数", "def search_multi(self, queries: List[str], ucr_boost: bool = False)" in mr_src)
    check("_fuse 来源加权挂点（boost_src）", "boost_src" in mr_src and '"ucr" if ucr_boost' in mr_src)
    check("加权置前排序（_src == boost_src）", 'c.get("_src") == boost_src' in mr_src)


# ── VF-40 draft_worker 联网/非联网路径 ────────────────────────────
def test_v817_draft_worker_paths():
    print("[VF-40] draft_worker 联网路径（原生联网草稿 + 提取 + [Wn]）")
    import src.tools.deepseek_web as dw
    from src.core.progress_bus import set_request_queue, clear_request_queue
    from src.core.draft_store import draft_store
    from src.core import tracing

    # ── 联网路径：_responses_web_search 打桩（原生联网回答）──
    q = asyncio.Queue()
    set_request_queue(q)
    draft_store.clear()
    real_resp = dw._responses_web_search
    real_extract = dw._call_extract_from_answer
    real_search = dw._draft_search_multi
    real_blog = dw._draft_blog
    records = []

    def recorder(event, **fields):
        records.append((event, fields))

    try:
        dw._responses_web_search = lambda inp: (
            "柑橘品种资源：UCR 库是世界最重要种质资源库之一，保存栽培品种、野生种与"
            "杂交后代；中国原产如温州蜜柑、椪柑，日本品种如宫川、清见。……",
            [{"url": "https://ucr.example/variety", "title": "UCR variety collection"}])
        dw._call_extract_from_answer = lambda qry, ans: (
            "===STRUCTURED_START===\n"
            "MULTI_QUERY: UCR citrus germplasm accessions|Chinese origin citrus cultivars|Japanese citrus mandarin varieties\n"
            "SUMMARY: UCR germplasm collection|Chinese origin mandarins|Japanese satsuma varieties\n"
            "===STRUCTURED_END===\n")
        dw._draft_search_multi = lambda queries: (
            [{"title": "draft-web-1", "doi": "10.d/w", "year": 2026, "text": "body"}]
            if len(queries) > 1 else [])
        dw._draft_blog = recorder

        tracing.set_web_search_enabled(True)   # 模拟前端「联网」开关开启（conext copied into asyncio.run）
        try:
            asyncio.run(dw.draft_worker("UCR 库有哪些品种", "sess-web"))
        finally:
            tracing.set_web_search_enabled(False)
    finally:
        dw._responses_web_search = real_resp
        dw._call_extract_from_answer = real_extract
        dw._draft_search_multi = real_search
        dw._draft_blog = real_blog

    items = []
    while not q.empty():
        items.append(q.get_nowait())
    clear_request_queue()
    draft_ev = next((it for it in items if it.get("event") == "draft"), {})
    ddata = json.loads(draft_ev["data"]) if draft_ev.get("data") else {}
    check("联网路径发出 draft 事件（原生联网回答）", bool(ddata.get("content"))
          and ddata["content"].startswith("柑橘品种资源"), str(ddata)[:120])

    payload = draft_store.pop("sess-web", "-")
    check("联网路径落仓（原生回答 + web_items + 检索结果）",
          payload is not None and len(payload.get("web_items") or []) == 1
          and len(payload.get("results") or []) == 1
          and payload.get("web_mode") is True, str(payload)[:200])
    check("联网路径 web_items 带 url", (payload or {}).get("web_items", [{}])[0].get("url", "").startswith("https"))
    draft_store.clear()

    # ── 联网路径失败 → draft_skipped（call_exception + mode=web）──
    q2 = asyncio.Queue()
    set_request_queue(q2)
    records.clear()
    try:
        dw._responses_web_search = lambda inp: (_ for _ in ()).throw(RuntimeError("web down"))
        dw._draft_blog = recorder
        tracing.set_web_search_enabled(True)
        try:
            asyncio.run(dw.draft_worker("UCR 库", "sess-web2"))
        finally:
            tracing.set_web_search_enabled(False)
    finally:
        dw._responses_web_search = real_resp
        dw._draft_blog = real_blog
        while not q2.empty():
            q2.get_nowait()
        clear_request_queue()
    ev, fields = records[-1]
    check("联网失败 → draft_skipped + call_exception:RuntimeError + mode=web",
          ev == "draft_skipped" and fields.get("reason", "").startswith("call_exception:RuntimeError:")
          and fields.get("mode") == "web", str(records[-1]))

    # ── 非联网路径默认（web 关）：queries 来自 ANSWER 一体提取 ──
    q3 = asyncio.Queue()
    set_request_queue(q3)
    draft_store.clear()
    real_call = dw._call_structured_draft
    try:
        dw._call_structured_draft = lambda qry: (
            "===STRUCTURED_START===\n"
            "ANSWER: 本地品种库检索建议：涉及品种特性问题时优先命中 UCR 品种库资料，"
            "再以文献补充机制背景。……\n"
            "MULTI_QUERY: a1|a2|a3\nSUMMARY: s1|s2|s3\n"
            "===STRUCTURED_END===\n")
        dw._draft_search_multi = lambda queries: (
            [{"title": "local-1", "doi": "10.d/l", "year": 2026, "text": "b"}] if len(queries) > 1 else [])
        asyncio.run(dw.draft_worker("柑橘品种特性", "sess-local"))
    finally:
        dw._call_structured_draft = real_call
        dw._draft_search_multi = real_search
        while not q3.empty():
            q3.get_nowait()
        clear_request_queue()
    payload2 = draft_store.pop("sess-local", "-")
    check("非联网路径：web_mode=False + 检索并入（MQ/SUMMARY 一体提取）",
          payload2 is not None and payload2.get("web_mode") is False
          and len(payload2.get("results") or []) == 1, str(payload2)[:160])
    draft_store.clear()


# ── VF-41 原生回答段 + 提示词标记 + snapshot ──────────────────────
def test_v817_fusion_and_prompts():
    print("[VF-41] 原生回答融合段 + UCR 提示词 + snapshot")
    from src.core.agent_runner import build_evidence_report

    arts = {"main_results": [{"title": "P1", "doi": "10.1/x", "year": 2023, "text": "body"}],
            "web_results": []}
    rep = build_evidence_report(arts, "variety question", 1,
                                draft_extra_count=3, draft_answer="原生预答：温州蜜柑源自中国。")
    check("回执含原生回答参考段（融合素材）",
          "原生回答参考（草稿预答，非检索证据）" in rep and "温州蜜柑源自中国" in rep)
    check("回执含草稿检索补充行", "草稿多路检索补充: 3 条证据" in rep)

    ra = (ROOT / "src/prompts/agents/retrieve-agent.md").read_text(encoding="utf-8")
    check("retrieve-agent.md 含 UCR 优先行", "优先本地 UCR 品种库" in ra)
    dg = (ROOT / "src/prompts/system/decision_guide.md").read_text(encoding="utf-8")
    check("decision_guide.md 含 UCR 优先策略", "品种/品种特性类问题优先 UCR 品种库（v8.17）" in dg)

    snap = (ROOT / "src/prompts/snapshot.py").read_text(encoding="utf-8")
    check("snapshot 渲染 structured_extract.txt", '"structured_extract.txt"' in snap)
    out_snap = (ROOT / "src/prompts/snapshots/structured_output.txt")
    ext_snap = (ROOT / "src/prompts/snapshots/structured_extract.txt")
    if out_snap.exists():
        txt = out_snap.read_text(encoding="utf-8")
        check("snapshot structured_output 已同步新模板（含 ANSWER）", "ANSWER:" in txt)
    else:
        check("snapshot structured_output 已同步新模板（含 ANSWER）", False, "缺快照")
    if ext_snap.exists():
        txt2 = ext_snap.read_text(encoding="utf-8")
        check("snapshot structured_extract 已生成", "MULTI_QUERY:" in txt2)
    else:
        check("snapshot structured_extract 已生成", False, "缺快照")


if __name__ == "__main__":
    import types as _types
    _funcs = [v for v in list(globals().values())
              if isinstance(v, _types.FunctionType)
              and v.__name__.startswith("test_")]
    for _f in sorted(_funcs, key=lambda x: x.__name__):
        try:
            _f()
        except Exception as _e:
            if os.environ.get("PYTEST_CURRENT_TEST"):
                raise
            print(f"  {_e}")
    ok = len(failed) == 0
    print(f"\n[VF-17] PASS {len(passed)} / FAIL {len(failed)}"
          f"  ({len(passed) + len(failed)} total)")
    for f in failed:
        print(f"  [FAIL] {f}")
    sys.exit(0 if ok else 1)