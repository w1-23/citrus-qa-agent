"""supervisor 收尾根治回归测试（v8.4.2）。

核心不变量: answer 一旦赋值，任何路径不得覆盖（结构保证）——
  - 自然完成（无 tool_calls → break）→ LLM 仅调用 1 次，answer 原样保留
  - 熔断/预算/跑满轮次 → 统一走 _force_final_answer（详尽 prompt + 临时列表不入史）
  - turn_trace 中绝不出现合成收尾消息
  - 历史读时过滤旧版伪造的"max turns"伪用户指令
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

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


_current_fake_holder = {}


def _current_fake():
    return _current_fake_holder["fake"]


def _patch_llm_pool(script):
    import src.core.llm_pool as pool
    _current_fake_holder["fake"] = FakeLLM(script)
    orig = pool.get_llm
    pool.get_llm = lambda **kw: FakeChat(**kw)
    return orig


def _min_state():
    return {
        "query": "测试问题",
        "session_id": "test-supervisor-final",
        "mode": "expert",
        "messages": [],
        "answer": "",
        "idempotency_key": "k",
        "format_hint": None,
        "history_summary": None,
        "long_term_memory": None,
        "resident_cards": None,
        "search_suggestions": [],
    }


def _no_synthetic(trace) -> bool:
    from src.graph.expert_graph import _FINAL_PROMPT
    for m in trace:
        content = str(getattr(m, "content", "") or "")
        if content.startswith("You have reached the maximum number of turns"):
            return False
        if content.startswith(_FINAL_PROMPT[:20]):
            return False
    return True


def test_natural_completion_not_overridden():
    print("[SF-1] 自然完成 → 不覆盖、不额外调用")
    from src.graph import expert_graph as eg
    orig = _patch_llm_pool([AIMessage(content="详尽的第一版回答，包含完整证据链")])
    try:
        state = _min_state()
        r = asyncio.new_event_loop().run_until_complete(eg.supervisor_node(state))
        fake = _current_fake_holder["fake"]
        check("LLM 仅调用 1 次", fake.calls == 1, f"calls={fake.calls}")
        check("answer 原样保留（不被覆盖）",
              r.get("answer") == "详尽的第一版回答，包含完整证据链",
              f"got={r.get('answer', '')[:40]!r}")
        check("turn_trace 无合成收尾消息", _no_synthetic(r.get("turn_trace", [])))
    finally:
        _restore_llm_pool(orig)


def test_max_turns_forces_detailed_final():
    print("[SF-2] 跑满轮次 → 统一收尾（详尽 prompt，临时列表不入史）")
    from src.graph import expert_graph as eg
    tool_calls_msg = AIMessage(content="", tool_calls=[
        {"id": "c1", "name": "call_retrieve_agent",
         "args": {"query": "citrus anthocyanin", "goal": "g"}}])
    orig = _patch_llm_pool([tool_calls_msg, tool_calls_msg,
                            AIMessage(content="跑满轮次后的详尽最终回答")])
    orig_turns = eg.SUPERVISOR_MAX_TURNS
    eg.SUPERVISOR_MAX_TURNS = 2

    async def fake_execute_tool_call(tc, tc_id="", material_pack=None, session_id=""):
        return {"agent": "retrieve-agent", "result": "检索报告：6 篇关键文献",
                "artifacts": {}}

    orig_exec = eg._execute_tool_call
    eg._execute_tool_call = fake_execute_tool_call
    try:
        state = _min_state()
        r = asyncio.new_event_loop().run_until_complete(eg.supervisor_node(state))
        fake = _current_fake_holder["fake"]
        check("跑满后触发收尾调用（共 3 次）", fake.calls == 3, f"calls={fake.calls}")
        check("收尾回答为详尽版", r.get("answer") == "跑满轮次后的详尽最终回答",
              f"got={r.get('answer', '')[:40]!r}")
        check("turn_trace 无合成收尾消息", _no_synthetic(r.get("turn_trace", [])))
    finally:
        eg.SUPERVISOR_MAX_TURNS = orig_turns
        eg._execute_tool_call = orig_exec
        _restore_llm_pool(orig)


def test_budget_hard_threshold_forces_final():
    print("[SF-3] 预算硬阈值 → 循环内就地收尾（不再走循环后块）")
    from src.graph import expert_graph as eg
    from src.config import settings
    orig = _patch_llm_pool([AIMessage(content="预算收尾详尽回答")])
    orig_max = settings.CONTEXT_BUDGET_MAX_TOKENS
    settings.CONTEXT_BUDGET_MAX_TOKENS = 1000  # system 提示词远超 930 → 必触发硬阈值
    try:
        state = _min_state()
        r = asyncio.new_event_loop().run_until_complete(eg.supervisor_node(state))
        fake = _current_fake_holder["fake"]
        check("预算触发后仅 1 次收尾调用", fake.calls == 1, f"calls={fake.calls}")
        check("收尾回答生效", r.get("answer") == "预算收尾详尽回答",
              f"got={r.get('answer', '')[:40]!r}")
        check("turn_trace 无合成收尾消息", _no_synthetic(r.get("turn_trace", [])))
    finally:
        settings.CONTEXT_BUDGET_MAX_TOKENS = orig_max
        _restore_llm_pool(orig)


def test_history_filter_synth_instruction():
    print("[SF-4] 历史读时过滤伪造'用户指令'")
    from src.session.manager import SessionManager, _SYNTH_FORCE_FINAL_MARK
    import sqlite3, tempfile
    from pathlib import Path

    sm = SessionManager()
    tmp = Path(tempfile.mkdtemp()) / "sessions.db"
    sm.db_path = str(tmp)
    sm._init_db_sync()
    sm._create_session_sync("sf4")
    with sqlite3.connect(str(tmp)) as conn:
        conn.execute(
            "INSERT INTO messages (session_id, msg_type, content) VALUES (?, 'human', ?)",
            ("sf4", "正常用户问题"))
        conn.execute(
            "INSERT INTO messages (session_id, msg_type, content) VALUES (?, 'human', ?)",
            ("sf4", _SYNTH_FORCE_FINAL_MARK + ". Do NOT call any more tools."))
        conn.execute(
            "INSERT INTO messages (session_id, msg_type, content) VALUES (?, 'ai', ?)",
            ("sf4", "正常回答"))
        conn.commit()
    msgs = sm._get_messages_sync("sf4")
    contents = [str(getattr(m, "content", "")) for m in msgs]
    check("伪造指令被过滤", not any(c.startswith(_SYNTH_FORCE_FINAL_MARK) for c in contents))
    check("真实消息保留", "正常用户问题" in contents and "正常回答" in contents,
          str(contents))
    sm._clear_session_sync("sf4")


def test_hyde_english_validation():
    print("[SF-5] HyDE 强制英文校验（确定性兜底）")
    from src.tools.search import _is_english_answer
    check("纯英文 → 通过", _is_english_answer(
        "Citrus anthocyanin accumulation is regulated by the Ruby MYB transcription factor."))
    check("中文 → 拒绝", not _is_english_answer(
        "柑橘花青素积累受Ruby转录因子调控，血橙果肉着色与转座子插入有关。"))
    check("英中混合 → 拒绝", not _is_english_answer(
        "Ruby MYB regulates anthocyanin 柑橘花青素 accumulation in blood orange."))
    check("空串 → 拒绝", not _is_english_answer(""))
    check("英文含少量符号 → 通过", _is_english_answer(
        "The Copia retrotransposon insertion into the Ruby promoter (TSS -1,500 bp) drives fruit coloration."))


def test_permission_grant_flow():
    print("[SF-6] 结构化权限（auto_workspace / ask 授权闭环）")
    import asyncio as _asyncio
    from src.tools.registry import _check_tool_sandbox, _is_workspace_output_path
    from src.config import settings
    from src.session.manager import SessionManager

    check("workspace/output 路径识别", _is_workspace_output_path("a.md"))
    check("越界路径不识别", not _is_workspace_output_path("E:/outside/x.md"))
    check("上级路径不识别", not _is_workspace_output_path("../x.md"))

    loop = _asyncio.new_event_loop()

    # auto_workspace（默认）: workspace 内写文件放行，越界拒绝
    settings.PERMISSION_MODE = "auto_workspace"
    r1 = loop.run_until_complete(_check_tool_sandbox(
        "write_local_file", {"path": "test_perm.md"}))
    check("auto_workspace 内放行", r1 is None, repr(r1))
    r2 = loop.run_until_complete(_check_tool_sandbox(
        "write_local_file", {"path": "E:/outside/x.md"}))
    check("auto_workspace 越界拒绝", r2 is not None and "ERR_HITL_REJECT" in r2, repr(r2))
    r3 = loop.run_until_complete(_check_tool_sandbox("citrus_rag_search", {"query": "x"}))
    check("只读工具放行", r3 is None)

    # ask: 无授权拒绝；workspace 授权后放行；once 授权消费
    settings.PERMISSION_MODE = "ask"
    from src.session.manager import session_manager
    import sqlite3, tempfile
    from pathlib import Path
    sm = SessionManager()
    tmp = Path(tempfile.mkdtemp()) / "sessions.db"
    sm.db_path = str(tmp)
    sm._init_db_sync()
    sm._create_session_sync("perm-test")

    r4 = loop.run_until_complete(_check_tool_sandbox(
        "write_local_file", {"path": "a.md"}))
    check("ask 无授权拒绝", r4 is not None and "权限未授权" in r4, repr(r4))

    sm.grant_permission("perm-test", "write_local_file", "once")
    r6 = loop.run_until_complete(_check_tool_sandbox(
        "write_local_file", {"path": "a.md"}))
    check("ask+once 授权放行", r6 is None, repr(r6))
    r7 = loop.run_until_complete(_check_tool_sandbox(
        "write_local_file", {"path": "a.md"}))
    check("once 消费后再次拒绝", r7 is not None, repr(r7))

    sm.grant_permission("perm-test", "write_local_file", "workspace")
    r5 = loop.run_until_complete(_check_tool_sandbox(
        "write_local_file", {"path": "a.md"}))
    check("ask+workspace 授权放行", r5 is None, repr(r5))
    r5b = loop.run_until_complete(_check_tool_sandbox(
        "write_local_file", {"path": "a.md"}))
    check("workspace 授权可复用", r5b is None, repr(r5b))

    sm.grant_permission("other-session", "write_local_file", "session")
    r8 = loop.run_until_complete(_check_tool_sandbox(
        "write_local_file", {"path": "E:/outside/x.md"}))
    check("session 范围不跨会话", r8 is not None, repr(r8))
    sm._clear_session_sync("perm-test")
    loop.close()


def test_offload_cleanup():
    print("[SF-7] offload 临时文件清理（v8.4.4 接线）")
    from src.tools.registry import (
        _offload_large_result, get_offload_file_list, cleanup_offload_files)
    p1 = _offload_large_result("x" * 20000, "test_tool")
    p2 = _offload_large_result("y" * 20000, "test_tool")
    check("offload 产生引用文本", "已自动卸载" in p1 and "test_tool" in p1, p1[:60])
    files = get_offload_file_list()
    check("列表记录 2 个文件", len(files) == 2, str(len(files)))
    check("文件真实存在", all(f.exists() for f in files))
    n = cleanup_offload_files()
    check("清理计数正确", n == 2, f"n={n}")
    check("清理后列表清空", len(get_offload_file_list()) == 0)
    check("清理后文件不存在", all(not f.exists() for f in files))


def test_fast_guard():
    print("[SF-8] Fast Guard 问候语命中（v8.4.4 去长度门槛）")
    from src.api.main import _is_fast_guard_hit
    check("纯问候命中", _is_fast_guard_hit("你好"))
    check("英文问候命中", _is_fast_guard_hit("hello"))
    check("超 12 字符问候命中（旧 bug: what can you do）",
          _is_fast_guard_hit("what can you do"))
    check("问候+语气词命中", _is_fast_guard_hit("你好呀"))
    check("科研问题不命中", not _is_fast_guard_hit("柑橘黄龙病的致病机制是什么"))
    check("问候+科研不命中", not _is_fast_guard_hit("你好，帮我查一下黄龙病"))
    check("身份询问命中", _is_fast_guard_hit("你是谁"))
    check("空白不命中", not _is_fast_guard_hit(""))


def _restore_llm_pool(orig):
    import src.core.llm_pool as pool
    pool.get_llm = orig


print()
if __name__ == "__main__":
    test_natural_completion_not_overridden()
    test_max_turns_forces_detailed_final()
    test_budget_hard_threshold_forces_final()
    test_history_filter_synth_instruction()
    test_hyde_english_validation()
    test_permission_grant_flow()
    test_offload_cleanup()
    test_fast_guard()
    print(f"supervisor final tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
