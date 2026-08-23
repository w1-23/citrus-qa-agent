# -*- coding: utf-8 -*-
"""v8.17 原生回答草稿 + UCR 优先回归（含 v8.17.1 联网一次化修订）：

  VF-39  品种意图检测（_is_variety_intent）+ UCR 优先接线（回执聚拢置前）
  VF-40  draft_worker 联网/非联网路径（v8.17.1: 联网失败→回退快速调用，草稿恒在）
  VF-41  回执「原生回答参考」段 + 提示词 UCR 优先标记 + snapshot
  VF-42  v8.17.1 联网一次化接线（白名单无联网 + prompt 限制 + decision_guide 规则）

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
    # v8.17 修订（修正1）：检索层路由已移除——keyword 检测函数保留（修正3），
    # 但不再驱动 ucr_boost / 来源加权（混合库无物理分区，路由在架构上不可行）
    check("检索层无 ucr_boost 接线（保持批次融合公平）",
          "ucr_boost" not in se and "rag.search_multi(queries)" in se)
    mr_src = (ROOT / "src/retrieval/multi_retriever.py").read_text(encoding="utf-8")
    check("multi_retriever 无 boost 参数、无来源加权",
          "def search_multi(self, queries: List[str])" in mr_src
          and "boost_src" not in mr_src and "boost_factor" not in mr_src)
    ar = (ROOT / "src/core/agent_runner.py").read_text(encoding="utf-8")
    check("回执层聚拢置前接线（_is_variety_intent → ucr_first）",
          "_is_variety_intent" in ar and "ucr_first=_ucr_first" in ar
          and "def build_evidence_report" in ar and "ucr_first: bool = False" in ar)

    # 元数据标记（修正1）：UCR 条目 _src 贯通检索→证据（src_of 判定）
    from src.core.evidence import src_of
    check("UCR chunk _src 标记贯通 src_of 判定",
          src_of({"_src": "ucr", "title": "x"}) == "ucr"
          and src_of({"_src": "rag", "title": "x"}) == "rag")
    check("source_type 兜底判定（无 _src 时按 source_type）",
          src_of({"source_type": "UCR citrus variety"}) == "ucr")

    # 聚拢置前（展示层，稳定分区保序）
    from src.core.agent_runner import build_evidence_report
    mixed = {"main_results": [
        {"title": "A文献", "_src": "rag", "year": 2023, "text": "a"},
        {"title": "B品种", "_src": "ucr", "year": 2024, "text": "b"},
        {"title": "C文献", "_src": "rag", "year": 2022, "text": "c"},
    ], "web_results": []}
    rep_plain = build_evidence_report(mixed, "机制问题", 1)
    check("非品种意图不回执重排（保持公平性）", rep_plain.index("[1][RAG] A文献") < rep_plain.index("[2][UCR] B品种"))
    rep_ucr = build_evidence_report(mixed, "Kinnow cultivar traits", 1, ucr_first=True)
    check("品种意图 → 回执 [UCR] 聚拢置前",
          rep_ucr.index("[1][UCR] B品种") < rep_ucr.index("[2][RAG] A文献"), rep_ucr[:160])
    check("聚拢置前组内保序（两个 RAG 相对顺序不变）",
          rep_ucr.index("[2][RAG] A文献") < rep_ucr.index("[3][RAG] C文献"))


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
    check("draft 事件带 source=web（修正4 前端来源标识）",
          ddata.get("source") == "web", str(ddata)[:120])

    payload = draft_store.pop("sess-web", "-")
    check("联网路径落仓（原生回答 + web_items + 检索结果）",
          payload is not None and len(payload.get("web_items") or []) == 1
          and len(payload.get("results") or []) == 1
          and payload.get("web_mode") is True, str(payload)[:200])
    check("联网路径 web_items 带 url", (payload or {}).get("web_items", [{}])[0].get("url", "").startswith("https"))
    draft_store.clear()

    # ── 联网路径失败 → v8.17.1 回退快速调用（草稿恒在，不跳过）──
    q2 = asyncio.Queue()
    set_request_queue(q2)
    records.clear()
    draft_store.clear()
    real_call2 = dw._call_structured_draft
    try:
        dw._responses_web_search = lambda inp: (_ for _ in ()).throw(RuntimeError("web down"))
        dw._call_structured_draft = lambda qry: (
            "===STRUCTURED_START===\n"
            "ANSWER: 回退快速调用产出的原生回答草稿，覆盖台州本地品种与课题。\n"
            "MULTI_QUERY: a1|a2|a3\nSUMMARY: s1|s2|s3\n"
            "===STRUCTURED_END===\n")
        dw._draft_search_multi = lambda queries: (
            [{"title": "fb-1", "doi": "10.d/f", "year": 2026, "text": "b"}] if len(queries) > 1 else [])
        dw._draft_blog = recorder
        tracing.set_web_search_enabled(True)
        try:
            asyncio.run(dw.draft_worker("UCR 库", "sess-web2"))
        finally:
            tracing.set_web_search_enabled(False)
    finally:
        dw._responses_web_search = real_resp
        dw._call_structured_draft = real_call2
        dw._draft_search_multi = real_search
        dw._draft_blog = real_blog
        evs2 = []
        while not q2.empty():
            evs2.append(q2.get_nowait())
        clear_request_queue()
    # 回退后：draft 事件仍发出（source=local）+ draft_web_fallback 日志 + 落仓 web_mode=False
    d_ev2 = next((it for it in evs2 if it.get("event") == "draft"), {})
    d2 = json.loads(d_ev2["data"]) if d_ev2.get("data") else {}
    check("v8.17.1 联网失败 → 回退快速调用，draft 事件仍发出",
          bool(d2.get("content")) and d2["content"].startswith("回退快速调用"), str(d2)[:120])
    check("回退草稿 source=local（来源标识如实降级）", d2.get("source") == "local")
    blog_evs = {e for e, _ in records}
    check("draft_web_fallback 业务日志已记", "draft_web_fallback" in blog_evs, str(records))
    pay2 = draft_store.pop("sess-web2", "-")
    check("回退落仓 web_mode=False（快速调用而非联网）",
          pay2 is not None and pay2.get("web_mode") is False
          and len(pay2.get("results") or []) == 1, str(pay2)[:160])
    check("回退落仓无 web_items（无 [Wn]）", pay2 is not None and not (pay2.get("web_items") or []))
    draft_store.clear()

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
        items3 = []
        while not q3.empty():
            items3.append(q3.get_nowait())
        clear_request_queue()
    payload2 = draft_store.pop("sess-local", "-")
    check("非联网路径：web_mode=False + 检索并入（MQ/SUMMARY 一体提取）",
          payload2 is not None and payload2.get("web_mode") is False
          and len(payload2.get("results") or []) == 1, str(payload2)[:160])
    draft_ev3 = next((it for it in items3 if it.get("event") == "draft"), {})
    ddata3 = json.loads(draft_ev3["data"]) if draft_ev3.get("data") else {}
    check("草稿内容=完整原生回答（不截断，修正2）",
          bool(ddata3.get("content")) and ddata3["content"].startswith("本地品种库检索建议"),
          str(ddata3)[:120])
    check("draft 事件带 source=local（修正4）", ddata3.get("source") == "local")
    draft_store.clear()


# ── VF-43 v8.17.6 无包裹标记容错提取（提示词+解析器双容错）──────────
def test_v8176_tolerant_extract():
    print("[VF-43] 无包裹标记容错提取：模板改为标签行 + 解析器兜底 + worker 并入")
    import asyncio
    import src.tools.deepseek_web as dw
    from src.core.progress_bus import (
        set_request_queue, clear_request_queue)
    from src.core.draft_store import draft_store
    from src.core import tracing

    ep = (ROOT / "src/prompts/structured_extract.md").read_text(encoding="utf-8")
    check("提取模板改为独立标签行（不再要求 ===STRUCTURED===）",
          "不依赖包裹标记" in ep and "MULTI_QUERY:" in ep and "SUMMARY:" in ep)
    check("提取模板示例用标签行形态", "\nMULTI_QUERY: citrus HLB integrated control 2025" in ep)

    # 无包裹标签行 → 容错解析（直接解析器级验证）
    parsed = dw._parse_structured_response(
        "素材：\nMULTI_QUERY: citrus HLB 2025|ACP monitoring trap|dsRNA biopesticide\n"
        "SUMMARY: ACP trap decline|Cq 37.73 near threshold\n", require_answer=False)
    check("解析器无包裹标签行 → MQ/SUMMARY 提取", len(parsed["multi_query"]) == 3
          and len(parsed["summary"]) == 2, str(parsed))

    # worker 级：web 路径提取返回无包裹标签行 → 检索素材仍并入（草稿证据 >0）
    real_resp = dw._responses_web_search
    real_extract = dw._call_extract_from_answer
    real_search = dw._draft_search_multi
    q = asyncio.Queue()
    set_request_queue(q)
    draft_store.clear()
    try:
        dw._responses_web_search = lambda inp: (
            "原生联网回答：2025 年柑橘黄龙病防控从媒介监测与精准施药…",
            [{"url": "https://w.example/hlb", "title": "HLB 2025"}])
        dw._call_extract_from_answer = lambda qry, ans: (
            "检索素材：\n"
            "MULTI_QUERY: citrus HLB integrated management 2025|ACP monitoring Florida|dsRNA biopesticide trial\n"
            "SUMMARY: ACP trap decline 9%|Cq 37.73 near threshold|dsRNA field validation\n")
        dw._draft_search_multi = lambda queries: (
            [{"title": "draft-tol-1", "doi": "10.t/1", "year": 2026, "text": "body"}]
            if len(queries) > 1 else [])
        tracing.set_web_search_enabled(True)
        try:
            asyncio.run(dw.draft_worker("HLB 2025 防控进展", "sess-tol"))
        finally:
            tracing.set_web_search_enabled(False)
    finally:
        dw._responses_web_search = real_resp
        dw._call_extract_from_answer = real_extract
        dw._draft_search_multi = real_search
        items = []
        while not q.empty():
            items.append(q.get_nowait())
        clear_request_queue()
    payload = draft_store.pop("sess-tol", "-")
    check("无包裹提取 → 草稿多路检索并入（queries>1）",
          payload is not None and len(payload.get("results") or []) == 1
          and payload.get("web_mode") is True, str(payload)[:200])
    draft_store.clear()


# ── VF-41 原生回答段 + 提示词标记 + snapshot ──────────────────────
def test_v817_fusion_and_prompts():
    print("[VF-41] 原生回答融合段 + UCR 提示词 + snapshot")
    from src.core.agent_runner import build_evidence_report

    arts = {"main_results": [{"title": "P1", "doi": "10.1/x", "year": 2023, "text": "body"}],
            "web_results": []}
    rep = build_evidence_report(arts, "variety question", 1,
                                draft_extra_count=3, draft_answer="原生预答：温州蜜柑源自中国。")
    check("回执含原生回答参考段（v8.17.4 默认可信）",
          "原生回答参考（草稿预答）" in rep and "温州蜜柑源自中国" in rep
          and "默认可信" in rep)
    check("回执含草稿检索补充行", "草稿多路检索补充: 3 条证据" in rep)

    ra = (ROOT / "src/prompts/agents/retrieve-agent.md").read_text(encoding="utf-8")
    check("retrieve-agent.md 含 UCR 优先行", "优先本地 UCR 品种库" in ra)
    check("retrieve-agent.md 品种意图规则（Prompt 层）", "品种意图识别（v8.17）" in ra
          and "检索方向**优先覆盖 UCR 品种库相关内容**" in ra)
    check("retrieve-agent.md 不残留'来源加权置前'（检索层已撤）", "UCR 来源加权置前" not in ra)
    dg = (ROOT / "src/prompts/system/decision_guide.md").read_text(encoding="utf-8")
    check("decision_guide.md 含 UCR 优先策略", "品种/品种特性类问题优先 UCR 品种库（v8.17）" in dg)
    check("decision_guide.md 不残留'来源加权'", "UCR 来源加权" not in dg)
    check("decision_guide.md 规则⑨ v8.17.4 默认可信（不再'仅供参考'）",
          "9. **原生回答参考（v8.17，v8.17.4 升级为默认可信）**" in dg
          and "原生联网回答默认可信" in dg and "仅供融合参考、不是检索证据" not in dg)

    idx = (ROOT / "index.html").read_text(encoding="utf-8")
    check("前端 draft.source 徽标接线（修正4）",
          ".draft-source" in idx and "data.source === 'web'" in idx
          and "原生联网" in idx and "快速回答" in idx)
    # v8.17.2: 草稿手风琴——完成默认收缩 + 不强制擦除
    check("前端草稿手风琴（draft-head 可点击头部）",
          "draft-head" in idx and "classList.remove('open')" in idx
          and "_dpEl.classList.remove('open')" in idx)
    check("前端草稿不强制擦除（text 事件保留面板）",
          "面板保留" in idx and "draftShown = true;" in idx
          and "_dpEl.remove()" not in idx.split("case 'text'")[1].split("case 'draft'")[0])
    # v8.17.3: 联网引用编号紧凑重排（[W1][W3][W7] → [W1][W2][W3]）
    check("前端引用紧凑重排函数存在", "_buildCompactRefMap" in idx
          and "_rewriteWRefsInText" in idx and "仅压缩 [Wn]（联网网址）" in idx)
    check("citations 事件接入重排（重写全文 + roundHistory 同步）",
          "_wmap" in idx and "_rewriteWRefsInText(fullText, _wmap)" in idx
          and "renderAnswer()" in idx)
    check("渲染保护不抹草稿面板（renderAnswer/done）",
          "面板保护" in idx and "_dpKeep" in idx and "_dpD" in idx)

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


# ── VF-42 v8.17.1 联网一次化接线（草稿层唯一联网，ReAct 不再联网）────────
def test_v8171_web_once_wiring():
    print("[VF-42] 联网一次化：白名单 + prompt 限制 + 规则")
    ar = (ROOT / "src/core/agent_runner.py").read_text(encoding="utf-8")
    ra = (ROOT / "src/prompts/agents/retrieve-agent.md").read_text(encoding="utf-8")
    dg = (ROOT / "src/prompts/system/decision_guide.md").read_text(encoding="utf-8")
    dw = (ROOT / "src/tools/deepseek_web.py").read_text(encoding="utf-8")

    # 代码级兜底：白名单不含 deepseek_web_search
    names = None
    from src.core import agent_runner as _ar
    names = _ar._resolve_tool_names("retrieve-agent")
    check("白名单不含 deepseek_web_search", "deepseek_web_search" not in names, str(names))
    check("白名单仍注册本地检索", "citrus_rag_search" in names)

    # prompt 限制
    check("retrieve-agent.md 禁止联网调用", "禁止调用 `deepseek_web_search`" in ra
          and "联网检索限制（v8.17.1，强制）" in ra)
    check("retrieve-agent.md 仅禁止块提及工具名（数据源表/自审区已移除）",
          ra.count("deepseek_web_search") == 1)  # 唯一次数=「禁止调用」块

    # web_search_status 提示块改为"仅告知，不授予联网"
    check("web_search_status 提示不再教调联网", "你无需也**不能**调用联网工具" in ar
          or "仅做本地检索（citrus_rag_search）" in ar)

    # decision_guide 规则
    check("decision_guide 联网仅草稿层唯一一次", "联网仅由草稿层唯一一次执行" in dg
          and "不要指示检索子代理联网补" in dg)
    check("decision_guide 原生回答段使用规则（v8.17.1 / v8.17.4）",
          "原生回答参考段的使用规则" in dg
          and "作为回答骨架" in dg and "填充/修正" in dg)
    check("decision_guide 检索子代理不再联网（无停止条件）",
          "检索子代理不再承担联网职责" in dg and "无需联网停止条件" in dg)

    # v8.17.4: 模型自述不再被确定性回执覆盖（合并输出，[Wn] 裁决可达 supervisor）
    check("agent_runner 模型自述与确定性回执合并（不覆盖）",
          "检索员判定（模型总结论）" in ar
          and "与确定性回执" in ar and "v8.17.4 不再覆盖" in ar)
    check("回执头文案 v8.17.4 默认可信",
          "v8.17.4：默认可信" in ar and "原生回答参考（草稿预答）" in ar)
    check("decision_guide 规则⑨ v8.17.4 默认可信 + [Wn] 直接引用",
          "原生联网回答默认可信" in dg and "无需本地向量库交叉验证" in dg
          and "检索员判定段（v8.17.4）" in dg)
    check("decision_guide 本地[ n ]有则以本地为准",
          "本地 [n] 已覆盖的同一事实 → 以本地 [n] 为准" in dg)
    # v8.17.4: 引用统计认 [Wn]/[Hn]（双轨引用，不误报 mismatch）
    eg = (ROOT / "src/graph/expert_graph.py").read_text(encoding="utf-8")
    check("citation 统计双轨（[Wn]/[Hn] 计入）",
          "cited_wn" in eg and "web_citation_count" in eg
          and "v8.17.4: 失配判定" in eg)

    # 草稿层回退
    check("draft_worker 联网失败回退快速调用",
          "draft_web_fallback" in dw
          and "回退到快速非联网调用（草稿恒在）" in dw)


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