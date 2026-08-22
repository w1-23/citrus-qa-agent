# -*- coding: utf-8 -*-
"""v8.16.1 草稿先行特性回归：分隔符结构化解析 / 草稿事件 / 草稿仓 / 证据并入回执。

全部离线、无模型、无网络（解析/事件/仓/回执均为纯逻辑；草稿调用与多路检索
以打桩验证 fail-soft 与接线）。约定见 test_batch1.py / test_v8153_fixes.py。
覆盖：
  VF-19  分隔符结构化解析（正常/缺字段/缺区块/围栏容错/大小写/空值）
  VF-20  草稿模板加载 + config 默认值
  VF-21  emit_draft 事件 + draft_worker 全链路（打桩调用与检索，fail-soft）
  VF-22  draft_store 请求级仓（put/pop/TTL/淘汰）
  VF-23  build_evidence_report 草稿补充行（draft_extra_count）
  VF-24  源码接线断言（expert/light 启动、agent_runner 并入、前端 draft 事件）
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


# ── VF-19 分隔符结构化解析（计划 v8.16.1 §5.2）────────────────────
def test_v8161_parse_structured():
    print("[VF-19] _parse_structured_response 分隔符解析")
    import src.tools.deepseek_web as dw

    raw = (
        "以下是检索结果摘要。\n"
        "===STRUCTURED_START===\n"
        "DRAFT_ZH: 2026年柑橘产业政策聚焦品种改良、品牌建设。\n"
        "DRAFT_EN: In 2026, citrus policies focus on variety improvement.\n"
        "MULTI_QUERY: national citrus policy 2026|citrus HLB control measures|citrus quality standards\n"
        "SUMMARY: Variety improvement|HLB control|Brand value\n"
        "===STRUCTURED_END===\n"
        "trailing"
    )
    p = dw._parse_structured_response(raw)
    check("回答正文 = 区块前文本", p["answer"] == "以下是检索结果摘要。", p["answer"])
    check("DRAFT_ZH 提取", p["draft_zh"] == "2026年柑橘产业政策聚焦品种改良、品牌建设。", p["draft_zh"])
    check("DRAFT_EN 提取", p["draft_en"] == "In 2026, citrus policies focus on variety improvement.", p["draft_en"])
    check("MULTI_QUERY 竖线拆分 3 条", p["multi_query"] == [
        "national citrus policy 2026", "citrus HLB control measures",
        "citrus quality standards"], str(p["multi_query"]))
    check("SUMMARY 竖线拆分 3 条", p["summary"] == ["Variety improvement", "HLB control", "Brand value"],
          str(p["summary"]))

    # 字段名大小写容错（提示词要求大写，防御小写）
    raw_lower = ("===STRUCTURED_START===\n"
                 "draft_zh: 中文草稿\n"
                 "draft_en: English draft\n"
                 "multi_query: a|b|c\n"
                 "summary: x|y|z\n"
                 "===STRUCTURED_END===\n")
    p2 = dw._parse_structured_response(raw_lower)
    check("字段名大小写容错", p2["draft_zh"] == "中文草稿" and p2["summary"] == ["x", "y", "z"], str(p2))

    # Markdown 围栏容错（提示词已约束，防御模型仍包 ```）
    raw_fence = ("```text\n===STRUCTURED_START===\n"
                 "DRAFT_ZH: 中文草稿\nDRAFT_EN: English draft\n"
                 "MULTI_QUERY: a|b|c\nSUMMARY: x|y|z\n"
                 "===STRUCTURED_END===\n```")
    p3 = dw._parse_structured_response(raw_fence)
    check("代码围栏剥离后解析成功", p3["draft_zh"] == "中文草稿", str(p3))

    # 续行容错：模型把 DRAFT_EN 折行（无字段名前缀的行归属上一字段）
    raw_wrap = ("===STRUCTURED_START===\n"
                "DRAFT_ZH: 中文草稿\n"
                "DRAFT_EN: In 2026, citrus policies focus on variety improvement\n"
                "and brand building for the citrus industry.\n"
                "MULTI_QUERY: a|b|c\nSUMMARY: x|y|z\n"
                "===STRUCTURED_END===\n")
    p4 = dw._parse_structured_response(raw_wrap)
    check("DRAFT_EN 折行续行合并", p4["draft_en"] == (
        "In 2026, citrus policies focus on variety improvement\n"
        "and brand building for the citrus industry."), repr(p4["draft_en"]))
    check("续行不污染其他字段", p4["multi_query"] == ["a", "b", "c"], str(p4["multi_query"]))

    # 异常用例
    try:
        dw._parse_structured_response("没有区块的文本")
        check("缺区块 → 抛 StructuredParseError", False)
    except dw.StructuredParseError:
        check("缺区块 → 抛 StructuredParseError", True)
    raw_missing = ("===STRUCTURED_START===\nDRAFT_ZH: 中文草稿\nDRAFT_EN: English draft\n"
                   "MULTI_QUERY: a|b|c\n===STRUCTURED_END===\n")
    try:
        dw._parse_structured_response(raw_missing)
        check("缺 SUMMARY → 抛错", False)
    except dw.StructuredParseError as e:
        check("缺 SUMMARY → 抛错", "SUMMARY" in str(e), str(e))
    raw_empty = ("===STRUCTURED_START===\nDRAFT_ZH: 中文草稿\nDRAFT_EN: English draft\n"
                 "MULTI_QUERY: a|b|c\nSUMMARY: \n===STRUCTURED_END===\n")
    try:
        dw._parse_structured_response(raw_empty)
        check("空值字段 → 抛错", False)
    except dw.StructuredParseError:
        check("空值字段 → 抛错", True)
    try:
        dw._parse_structured_response("")
        check("空响应 → 抛错", False)
    except dw.StructuredParseError:
        check("空响应 → 抛错", True)


# ── VF-20 草稿模板 + config 默认值 ────────────────────────────────
def test_v8161_prompt_and_config():
    print("[VF-20] structured_output 模板 + config")
    from src.prompts.loader import assemble_structured_output_prompt

    prompt = assemble_structured_output_prompt()
    check("模板非空", bool(prompt))
    check("模板含 START 分隔符", "===STRUCTURED_START===" in prompt)
    check("模板含 END 分隔符", "===STRUCTURED_END===" in prompt)
    check("模板含四个字段", all(f in prompt for f in ("DRAFT_ZH", "DRAFT_EN", "MULTI_QUERY", "SUMMARY")))
    check("模板禁止 JSON/围栏", "不要输出 JSON" in prompt and "不要输出 Markdown 代码块" in prompt)
    check("模板 DRAFT_ZH 100-150 字约束", "100-150" in prompt)

    check("DRAFT_ENABLED 默认开", settings.DRAFT_ENABLED is True)
    check("DRAFT_LABEL 默认文案", settings.DRAFT_LABEL == "预检索草稿·验证中", settings.DRAFT_LABEL)
    check("DRAFT_MAX_CHARS 默认 300", settings.DRAFT_MAX_CHARS == 300)
    check("DRAFT_MAX_ANGLES 默认 3", settings.DRAFT_MAX_ANGLES == 3)
    check("DRAFT_SUMMARY_POINTS 默认 3（已确认上限）", settings.DRAFT_SUMMARY_POINTS == 3)
    check("DRAFT_EXTRA_RETRIEVAL 默认开", settings.DRAFT_EXTRA_RETRIEVAL is True)


# ── VF-21 emit_draft + draft_worker 全链路（打桩，fail-soft）──────
def test_v8161_draft_event_and_worker():
    print("[VF-21] emit_draft 事件 + draft_worker 链路")
    import asyncio
    import json
    import src.tools.deepseek_web as dw
    from src.core.progress_bus import (
        set_request_queue, clear_request_queue, emit_draft)
    from src.core.draft_store import draft_store

    # 1) emit_draft 事件形态
    q = asyncio.Queue()
    set_request_queue(q)
    try:
        emit_draft("2026年柑橘政策初判…", "预检索草稿·验证中")
        items = [q.get_nowait() for _ in range(1)]
    finally:
        clear_request_queue()
    check("draft 事件入队", items[0]["event"] == "draft", str(items[0]))
    ev = json.loads(items[0]["data"]) if isinstance(items[0].get("data"), str) else {}
    check("draft 载荷 content+label",
          ev.get("content") == "2026年柑橘政策初判…"
          and ev.get("label") == "预检索草稿·验证中", str(ev))

    # 2) draft_worker 全链路（打桩调用与检索；验证 emit + 仓暂存 + 幂等幂等安全）
    q2 = asyncio.Queue()
    set_request_queue(q2)
    draft_store.clear()

    real_call = dw._call_structured_draft

    class _FakeCall:
        def __call__(self, query):
            return ("===STRUCTURED_START===\n"
                    "DRAFT_ZH: 测试草稿内容。\n"
                    "DRAFT_EN: Test draft content.\n"
                    "MULTI_QUERY: citrus test query a|citrus test query b|citrus test query c\n"
                    "SUMMARY: point one|point two|point three\n"
                    "===STRUCTURED_END===\n")

    class _FakeRetriever:
        def __init__(self):
            pass

        def search_multi(self, queries):
            return [{"title": "DraftPaper", "doi": "10.999/draft",
                     "year": 2026, "text": "draft hit", "score": 0.9}]

    import src.retrieval.multi_retriever as mr
    real_retriever = mr.MultiBatchRetriever
    try:
        dw._call_structured_draft = _FakeCall()
        mr.MultiBatchRetriever = _FakeRetriever
        asyncio.run(dw.draft_worker("2026柑橘政策", "sess-1"))
    finally:
        dw._call_structured_draft = real_call
        mr.MultiBatchRetriever = real_retriever
        items2 = []
        while not q2.empty():
            items2.append(q2.get_nowait())
        clear_request_queue()

    check("worker 发出 draft 事件", any(it["event"] == "draft" for it in items2), str(items2)[:200])
    draft_ev = next((it for it in items2 if it["event"] == "draft"), {})
    ddata = json.loads(draft_ev["data"]) if draft_ev.get("data") else {}
    check("草稿内容 = DRAFT_ZH", ddata.get("content") == "测试草稿内容。", str(ddata))
    check("草稿标签 = 预检索草稿·验证中", ddata.get("label") == "预检索草稿·验证中", str(ddata))

    payload = draft_store.pop("sess-1", "-")
    check("多路检索结果已暂存草稿仓", bool(payload and payload.get("results")), str(payload)[:200])
    check("draft_en 在首路查询", payload and payload["queries"][0] == "Test draft content.",
          str(payload and payload["queries"]))
    check("查询路由 = 1+3+3", payload and len(payload["queries"]) == 7,
          str(payload and len(payload["queries"])))
    draft_store.clear()

    # 3) fail-soft：解析失败不抛异常、不落仓
    q3 = asyncio.Queue()
    set_request_queue(q3)

    class _FakeBad:
        def __call__(self, query):
            return "没有结构化区块的输出"

    draft_store.clear()
    try:
        dw._call_structured_draft = _FakeBad()
        asyncio.run(dw.draft_worker("2026柑橘政策", "sess-2"))
    finally:
        dw._call_structured_draft = real_call
        while not q3.empty():
            q3.get_nowait()
        clear_request_queue()
    check("解析失败 → 不落仓", draft_store.pop("sess-2", "-") is None)
    draft_store.clear()


# ── VF-22 draft_store 请求级仓 ───────────────────────────────────
def test_v8161_draft_store():
    print("[VF-22] draft_store TTL/淘汰/键隔离")
    from src.core.draft_store import DraftStore

    st = DraftStore(max_entries=4, ttl_sec=100)
    st.put("s1", "r1", {"results": [1]})
    st.put("s2", "r2", {"results": [2]})
    check("同 session 不同 request 键隔离", st.pop("s1", "r2") is None)
    check("pop 取回", st.pop("s1", "r1") == {"results": [1]})
    check("pop 后删键", st.pop("s1", "r1") is None)

    # TTL 过期
    st2 = DraftStore(max_entries=4, ttl_sec=10)
    st2.put("s", "r", {"results": [1]})
    st2._entries[st2._key("s", "r")]["ts"] -= 30  # 人为过期
    check("TTL 过期 → pop 返回 None", st2.pop("s", "r") is None)
    check("过期键被清理", st2._key("s", "r") not in st2._entries)

    # 超限淘汰最旧
    st3 = DraftStore(max_entries=3, ttl_sec=100)
    for i in range(5):
        st3.put("s", f"r{i}", {"results": [i]})
    check("超过上限 → 只留最近 3 条",
          len(st3._entries) == 3 and st3.pop("s", "r4") == {"results": [4]}
          and st3.pop("s", "r0") is None, str(len(st3._entries)))


# ── VF-23 证据回执草稿补充行 ─────────────────────────────────────
def test_v8161_evidence_report_draft_line():
    print("[VF-23] build_evidence_report 草稿补充行")
    from src.core.agent_runner import build_evidence_report

    arts = {"main_results": [{"title": "P1", "doi": "10.1/x", "year": 2023, "text": "body"}],
            "web_results": []}
    base = build_evidence_report(arts, "citrus hlb", 2)
    check("默认无草稿行", "草稿多路检索补充" not in base)
    with_draft = build_evidence_report(arts, "citrus hlb", 2, draft_extra_count=4)
    check("draft_extra_count=4 → 回执明示补充 4 条",
          "草稿多路检索补充: 4 条证据" in with_draft, with_draft[:300])
    check("补充行含三路来源说明", "DRAFT_EN + MULTI_QUERY + SUMMARY" in with_draft)


# ── VF-24 源码接线断言（防接线回归，VF-17 同款模式）────────────────
def test_v8161_wiring_assertions():
    print("[VF-24] 源码接线断言")
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]  # agent/
    ex = (root / "src/graph/expert_graph.py").read_text(encoding="utf-8")
    lg = (root / "src/graph/light_graph.py").read_text(encoding="utf-8")
    ar = (root / "src/core/agent_runner.py").read_text(encoding="utf-8")
    dw = (root / "src/tools/deepseek_web.py").read_text(encoding="utf-8")
    idx = (root / "index.html").read_text(encoding="utf-8")
    check("expert load 启动 draft_worker",
          "draft_worker(query, session_id)" in ex and "草稿先行" in ex)
    check("light load 启动 draft_worker（全场景草稿）",
          "draft_worker(query, session_id)" in lg)
    check("agent_runner 并入草稿证据", "draft_store.pop(session_id" in ar
          and "草稿多路检索补充" in ar)
    check("deepseek_web 含分隔符解析与 worker",
          "def _parse_structured_response" in dw and "async def draft_worker" in dw)
    check("前端 draft 事件分支", "case 'draft':" in idx)
    check("前端草稿标签文案", "预检索草稿·验证中" in idx)
    check("前端首 text 清空草稿", "draftShown = false" in idx)


# ── 汇总 ──────────────────────────────────────────────────────────
def _summary():
    print(f"\n[VF-16.1] PASS {len(passed)} / FAIL {len(failed)}"
          f"  ({len(passed) + len(failed)} total)")
    for f in failed:
        print(f"  [FAIL] {f}")
    return len(failed) == 0


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
    ok = _summary()
    sys.exit(0 if ok else 1)