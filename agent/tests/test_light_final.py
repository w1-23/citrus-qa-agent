# -*- coding: utf-8 -*-
"""light supervisor 收尾行为 A/B 回归测试（v8.13-b5a AgentLoop 第二步前置安全网）。

与 test_supervisor_final.py 对齐的不变量（light 此前无行为覆盖）：
  - 自然完成（无 tool_calls → break）→ LLM 仅调用 1 次，answer 原样保留
  - 跑满轮次 → 统一 _light_force_final（详尽 prompt + 临时列表不入史）
  - turn_trace 绝不出现合成收尾消息
"""
import asyncio
import os

from langchain_core.messages import AIMessage, ToolMessage

from src.core.agent_loop import FINAL_ANSWER_PROMPT

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


class FakeLLM:
    def __init__(self, script):
        self.script = script
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        idx = min(self.calls - 1, len(self.script) - 1)
        return self.script[idx]


class FakeChat:
    def __init__(self, *a, **k):
        self.fake = _current_fake()

    def bind_tools(self, tools=None):
        return self

    async def ainvoke(self, messages):
        return await self.fake.ainvoke(messages)

    async def astream(self, messages):
        import json as _json
        from langchain_core.messages import AIMessageChunk
        from langchain_core.messages.tool import tool_call_chunk
        resp = await self.fake.ainvoke(messages)
        tcs = getattr(resp, "tool_calls", None) or []
        tcc = [
            tool_call_chunk(name=tc.get("name"),
                            args=_json.dumps(tc.get("args") or {}, ensure_ascii=False),
                            id=tc.get("id", f"t{i}"), index=i)
            for i, tc in enumerate(tcs)
        ]
        yield AIMessageChunk(content=str(resp.content or ""), tool_call_chunks=tcc)


_current_fake_holder = {}


def _current_fake():
    return _current_fake_holder["fake"]


def _patch_llm_pool(script):
    import src.core.llm_pool as pool
    _current_fake_holder["fake"] = FakeLLM(script)
    orig = pool.get_llm
    pool.get_llm = lambda **kw: FakeChat(**kw)
    return orig


def _restore_llm_pool(orig):
    import src.core.llm_pool as pool
    pool.get_llm = orig


def _min_state():
    return {
        "query": "测试问题",
        "session_id": "test-light-final",
        "messages": [],
        "answer": "",
        "idempotency_key": "k",
        "format_hint": None,
        "history_summary": None,
        "long_term_memory": None,
        "resident_cards": None,
        "search_suggestions": [],
        "retrieval_context": "",
        "main_results": [],
        "web_results": [],
        "history_evidence_block": None,
    }


def _no_synthetic(trace) -> bool:
    for m in trace:
        content = str(getattr(m, "content", "") or "")
        if content.startswith("You have reached the maximum number of turns"):
            return False
        if content.startswith(FINAL_ANSWER_PROMPT[:20]):
            return False
    return True


def test_light_natural_completion_not_overridden():
    print("[LF-1] light 自然完成 → 不覆盖、不额外调用")
    from src.graph import light_graph as lg
    orig = _patch_llm_pool([AIMessage(content="详尽的第一版回答，包含完整证据链")])
    try:
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(lg.light_supervisor_node(_min_state()))
        finally:
            loop.close()
        fake = _current_fake_holder["fake"]
        check("LLM 仅调用 1 次", fake.calls == 1, f"calls={fake.calls}")
        check("answer 原样保留（不被覆盖）",
              r.get("answer") == "详尽的第一版回答，包含完整证据链",
              f"got={r.get('answer', '')[:40]!r}")
        check("turn_trace 无合成收尾消息", _no_synthetic(r.get("turn_trace", [])))
    finally:
        _restore_llm_pool(orig)


def test_light_max_turns_forces_final():
    print("[LF-2] light 跑满轮次 → 统一收尾（详尽 prompt，临时列表不入史）")
    from src.graph import light_graph as lg
    import src.tools.registry as reg

    tool_calls_msg = AIMessage(content="", tool_calls=[
        {"id": "c1", "name": "citrus_rag_search",
         "args": {"query": "citrus anthocyanin"}}])
    orig = _patch_llm_pool([tool_calls_msg, tool_calls_msg,
                            AIMessage(content="跑满轮次后的详尽最终回答")])
    orig_turns = lg.LIGHT_MAX_TURNS
    lg.LIGHT_MAX_TURNS = 2

    orig_exec = reg.PartitionedToolNode.execute_tools

    async def fake_exec(self, tool_calls):
        msgs = []
        for tc in tool_calls:
            msgs.append(ToolMessage(
                content="ok", tool_call_id=tc["id"],
                name=tc.get("name"),
                artifact={"main_results": [], "web_results": []}))
        return msgs

    reg.PartitionedToolNode.execute_tools = fake_exec
    try:
        loop = asyncio.new_event_loop()
        try:
            r = loop.run_until_complete(lg.light_supervisor_node(_min_state()))
        finally:
            loop.close()
        fake = _current_fake_holder["fake"]
        check("跑满后触发收尾调用（共 3 次）", fake.calls == 3, f"calls={fake.calls}")
        check("收尾回答为详尽版", r.get("answer") == "跑满轮次后的详尽最终回答",
              f"got={r.get('answer', '')[:40]!r}")
        check("turn_trace 无合成收尾消息", _no_synthetic(r.get("turn_trace", [])))
    finally:
        lg.LIGHT_MAX_TURNS = orig_turns
        reg.PartitionedToolNode.execute_tools = orig_exec
        _restore_llm_pool(orig)


if __name__ == "__main__":
    test_light_natural_completion_not_overridden()
    test_light_max_turns_forces_final()
    print(f"light final tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        raise SystemExit(1)