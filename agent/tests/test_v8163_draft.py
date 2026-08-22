# -*- coding: utf-8 -*-
"""v8.16.3 草稿通道与引用密度回归：异常/空输出区分、降级草稿、web[:14]、P2+软性引用。

全部离线、无模型、无网络（草稿调用以打桩验证 fail-soft 与 reason 区分）。
覆盖：
  VF-28  call_exception/call_empty 区分（异常不再被吞成 call_empty）+ max_tokens=2048
  VF-29  解析失败降级草稿（非空文本→raw[:200] 展示、不落仓、模板保持严格）
  VF-30  回执 web[:10]→[:14]（联网 14 条结果全部可见）
  VF-31  decision_guide 规则⑦⑧（[Hn] 历史引用 + 软性引用覆盖指令）
  VF-32  源码接线断言
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


_STRUCT_OK = ("===STRUCTURED_START===\n"
              "DRAFT_ZH: 测试草稿。\n"
              "DRAFT_EN: Test draft.\n"
              "MULTI_QUERY: a q1|a q2|a q3\n"
              "SUMMARY: p1|p2|p3\n"
              "===STRUCTURED_END===\n")


# ── VF-28 call_exception / call_empty 区分 ────────────────────────
def test_v8163_reason_split():
    print("[VF-28] call_exception/call_empty 区分 + max_tokens")
    import asyncio
    import src.tools.deepseek_web as dw
    from src.core.progress_bus import set_request_queue, clear_request_queue

    check("DRAFT_MAX_TOKENS 默认 2048（思维链吃预算根因防线的余量）",
          settings.DRAFT_MAX_TOKENS == 2048, str(settings.DRAFT_MAX_TOKENS))

    real_call = dw._call_structured_draft
    real_blog = dw._draft_blog
    records = []

    def recorder(event, **fields):
        records.append((event, fields))

    class _FakeRaise:
        def __call__(self, query):
            raise RuntimeError("boom api err")

    class _FakeNone:
        def __call__(self, query):
            return None

    try:
        dw._draft_blog = recorder

        # 异常 → call_exception（带异常类型与文本）
        dw._call_structured_draft = _FakeRaise()
        q = asyncio.Queue()
        set_request_queue(q)
        try:
            asyncio.run(dw.draft_worker("2026柑橘政策", "sess-x"))
        finally:
            clear_request_queue()
        ev, fields = records[-1]
        check("异常 → draft_skipped",
              ev == "draft_skipped", str(records[-1]))
        check("reason=call_exception:RuntimeError + 异常文本",
              fields.get("reason", "").startswith("call_exception:RuntimeError:")
              and "boom api err" in fields.get("reason", ""), str(fields))

        # 两次空返回 → call_empty（不再与异常混淆）
        dw._call_structured_draft = _FakeNone()
        q2 = asyncio.Queue()
        set_request_queue(q2)
        try:
            asyncio.run(dw.draft_worker("2026柑橘政策", "sess-y"))
        finally:
            clear_request_queue()
        ev2, fields2 = records[-1]
        check("真空输出 → reason=call_empty（与异常区分开）",
              ev2 == "draft_skipped" and fields2.get("reason") == "call_empty",
              str(records[-1]))

        # 异常不得再被吞成 call_empty（标签区分已生效）
        reasons = {f.get("reason", "") for _, f in records}
        check("call_exception 与 call_empty 都出现（无混同）",
              any(r.startswith("call_exception:") for r in reasons)
              and "call_empty" in reasons, str(reasons))
    finally:
        dw._call_structured_draft = real_call
        dw._draft_blog = real_blog


# ── VF-29 解析失败降级草稿 ────────────────────────────────────────
def test_v8163_fallback_draft():
    print("[VF-29] 解析失败降级草稿（worker 侧，模板保持严格）")
    import asyncio
    import json
    import src.tools.deepseek_web as dw
    from src.core.progress_bus import set_request_queue, clear_request_queue
    from src.core.draft_store import draft_store

    # 1) 纯函数降级提取
    check("fallback: 取区块前正文", dw._fallback_draft_zh(
        "开头回答正文。===STRUCTURED_START===\nDRAFT_ZH: x\n") == "开头回答正文。")
    check("fallback: 去代码围栏",
          dw._fallback_draft_zh("```text\n你好世界柑橘\n```") == "你好世界柑橘")
    check("fallback: 空输入", dw._fallback_draft_zh("") == "")
    check("fallback: 上限 200 字", len(dw._fallback_draft_zh("字" * 500)) == 200)

    # 2) worker 全链路：非空但无分隔符 → 降级草稿事件 + draft_fallback 日志 + 不落仓
    real_call = dw._call_structured_draft
    real_blog = dw._draft_blog
    records = []

    def recorder(event, **fields):
        records.append((event, fields))

    class _FakeGarbage:
        def __call__(self, query):
            return "模型没有按模板输出,这是纯文本回答。柑橘产业政策…"

    draft_store.clear()
    try:
        dw._call_structured_draft = _FakeGarbage()
        dw._draft_blog = recorder
        q = asyncio.Queue()
        set_request_queue(q)
        try:
            asyncio.run(dw.draft_worker("2026柑橘政策", "sess-fb"))
        finally:
            items = []
            while not q.empty():
                items.append(q.get_nowait())
            clear_request_queue()
    finally:
        dw._call_structured_draft = real_call
        dw._draft_blog = real_blog

    draft_evs = [it for it in items if it.get("event") == "draft"]
    check("降级草稿事件已发出", len(draft_evs) == 1, str(items)[:200])
    if draft_evs:
        data = json.loads(draft_evs[0]["data"]) if isinstance(draft_evs[0].get("data"), str) else {}
        check("降级草稿内容 = raw 前 200 字", data.get("content", "").startswith("模型没有按模板输出"), str(data))
        check("降级草稿标签不变", data.get("label") == "预检索草稿·验证中", str(data))
    blog_events = {e for e, _ in records}
    check("draft_fallback 业务日志已记", "draft_fallback" in blog_events, str(blog_events))
    check("降级路径不落仓（检索并入与结构化产出绑定）",
          draft_store.pop("sess-fb", "-") is None)
    draft_store.clear()

    # 3) 模板保持严格：structured_output.md 无"降级/纯文本"指令（弱化契约的回归护栏）
    prompt = __import__("src.prompts.loader", fromlist=["assemble_structured_output_prompt"]).assemble_structured_output_prompt()
    check("结构化模板不含降级/纯文本兜底指令",
          "降级" not in prompt and "纯文本" not in prompt)


# ── VF-30 回执 web[:14] ───────────────────────────────────────────
def test_v8163_web14():
    print("[VF-30] 回执 web[:10]→[:14]")
    from src.core.agent_runner import build_evidence_report

    web = [{"title": f"W{i}", "url": f"https://x/{i}", "abstract": "a" * 60}
           for i in range(15)]
    rep = build_evidence_report({"main_results": [], "web_results": web,
                                 "web_summaries": []}, "q", 1)
    check("14 条联网条目全渲染", "[W14]" in rep, "[W14] 缺失")
    check("第 15 条不渲染", "[W15]" not in rep)
    check("条目编号递增到 14", "[W13]" in rep and "[W12]" in rep)


# ── VF-31 decision_guide 规则⑦⑧ ─────────────────────────────────
def test_v8163_prompt_rules():
    print("[VF-31] decision_guide 历史引用 + 软性覆盖")
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(src, "src", "prompts", "system", "decision_guide.md"),
              encoding="utf-8") as f:
        txt = f.read()
    check("规则⑦ 历史证据 H1..Hn 引用", "H1..Hn" in txt and "历史证据引用（v8.16.3）" in txt)
    check("规则⑧ 软性引用覆盖（不为凑数）",
          "引用覆盖（v8.16.3，软性）" in txt and "不为凑数而引用" in txt)


# ── VF-32 源码接线 ────────────────────────────────────────────────
def test_v8163_wiring():
    print("[VF-32] 接线断言")
    src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    with open(os.path.join(src, "src", "core", "agent_runner.py"),
              encoding="utf-8") as f:
        ar = f.read()
    check("agent_runner web[:14]", "web[:14]" in ar)

    with open(os.path.join(src, "src", "tools", "deepseek_web.py"),
              encoding="utf-8") as f:
        dw = f.read()
    check("draft_worker 区分 call_exception", '"call_exception:{type(e).__name__}' in dw)
    check("_fallback_draft_zh 存在", "def _fallback_draft_zh" in dw)
    check("draft_fallback 业务日志", '"draft_fallback"' in dw)

    with open(os.path.join(src, "src", "config.py"), encoding="utf-8") as f:
        cfg = f.read()
    check("config DRAFT_MAX_TOKENS 字段", "DRAFT_MAX_TOKENS" in cfg)


# ── 汇总 ──────────────────────────────────────────────────────────
def _summary():
    print(f"\n[VF-16.3] PASS {len(passed)} / FAIL {len(failed)}"
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