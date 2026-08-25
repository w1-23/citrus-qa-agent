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
from _tmpenv import tmp_path

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

    async def astream(self, messages):
        # v8.4.13: 真流式——测试假模型也走 astream（supervisor 主循环已流式化）。
        # 把 ainvoke 结果转为 AIMessageChunk 单 chunk 流（含 tool_call_chunks）；
        # args 必须是 JSON 字符串（str(dict) 是 Python 字面量，转换会失败）。
        import json as _json
        from langchain_core.messages import AIMessageChunk
        from langchain_core.messages.tool import tool_call_chunk
        resp = await self.fake.ainvoke(messages)
        tcs = getattr(resp, "tool_calls", None) or []
        tcc = []
        for i, tc in enumerate(tcs):
            tcc.append(tool_call_chunk(
                name=tc.get("name"),
                args=_json.dumps(tc.get("args") or {}, ensure_ascii=False),
                id=tc.get("id", f"t{i}"), index=i))
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
    from src.core.agent_loop import FINAL_ANSWER_PROMPT
    for m in trace:
        content = str(getattr(m, "content", "") or "")
        if content.startswith("You have reached the maximum number of turns"):
            return False
        if content.startswith(FINAL_ANSWER_PROMPT[:20]):
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
        {"id": "c1", "name": "call_search_both",
         "args": {"local_goal": "citrus anthocyanin", "web_goal": "citrus anthocyanin 2025"}}])
    orig = _patch_llm_pool([tool_calls_msg, tool_calls_msg,
                            AIMessage(content="跑满轮次后的详尽最终回答")])
    orig_turns = eg.SUPERVISOR_MAX_TURNS
    eg.SUPERVISOR_MAX_TURNS = 2

    async def fake_execute_tool_call(tc, tc_id="", material_pack=None,
                                     session_id="", seen_queries=None):
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
    tmp = tmp_path("db")
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
    # v8.4.5: ask 模式默认等待授权 90s——测试缩短等待窗口，避免用例挂起
    _orig_wait = settings.PERMISSION_WAIT_SEC
    settings.PERMISSION_WAIT_SEC = 0.2
    from src.session.manager import session_manager
    import sqlite3, tempfile
    from pathlib import Path
    sm = SessionManager()
    tmp = tmp_path("db")
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
    settings.PERMISSION_MODE = "auto_workspace"
    settings.PERMISSION_WAIT_SEC = _orig_wait
    loop.close()


def test_permission_wait_resume():
    """v8.4.5: ask 模式授权等待——授权到达后同一执行内恢复（无需整轮重跑）。"""
    print("[SF-6b] ask 授权等待/唤醒（同执行内恢复）")
    import asyncio as _asyncio
    from src.tools.registry import _check_tool_sandbox, signal_permission_granted
    from src.config import settings
    from src.session.manager import SessionManager
    from src.core import tracing

    _orig_mode = settings.PERMISSION_MODE
    _orig_wait = settings.PERMISSION_WAIT_SEC
    try:
        settings.PERMISSION_MODE = "ask"
        settings.PERMISSION_WAIT_SEC = 5
        import tempfile
        from pathlib import Path
        sm = SessionManager()
        tmp = tmp_path("db")
        sm.db_path = str(tmp)
        sm._init_db_sync()
        sm._create_session_sync("perm-wait-test")

        loop = _asyncio.new_event_loop()
        try:
            async def _grant_later():
                await _asyncio.sleep(0.3)
                sm.grant_permission("perm-wait-test", "write_local_file", "once")
                signal_permission_granted("perm-wait-test", "write_local_file")

            async def _main():
                tracing.set_session_id("perm-wait-test")
                waiter = _asyncio.create_task(
                    _check_tool_sandbox("write_local_file", {"path": "a.md"}))
                granter = _asyncio.create_task(_grant_later())
                result = await waiter
                await granter
                return result

            r = loop.run_until_complete(_main())
        finally:
            loop.close()
        check("授权到达后同执行内放行", r is None, repr(r))
        sm._clear_session_sync("perm-wait-test")
    finally:
        settings.PERMISSION_MODE = _orig_mode
        settings.PERMISSION_WAIT_SEC = _orig_wait
        tracing.set_session_id("")


def test_offload_cleanup():
    print("[SF-7] offload 临时文件清理（v8.4.4 接线）")
    from src.tools.registry import (
        _offload_large_result, get_offload_file_list, cleanup_offload_files)
    # v8.4.6: 前置清场——其它用例（如 retrieve-agent 证据暂存）可能已注册文件
    cleanup_offload_files()
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


def test_query_dedup():
    print("[SF-8b] 检索角度代码级去重（v8.4.6）")
    from src.core.agent_runner import check_query_redundant
    seen = ["citrus anthocyanin regulation transcription factor MYB WRKY"]
    check("完全相同角度 → 去重", bool(check_query_redundant(
        "citrus anthocyanin regulation transcription factor MYB WRKY", seen)))
    check("token 集合相同(乱序/大小写) → 去重", bool(check_query_redundant(
        "WRKY MYB citrus regulation transcription factor ANTHOCYANIN", seen)))
    check("高度重叠(Jaccard≥0.85) → 去重", bool(check_query_redundant(
        "citrus anthocyanin MYB WRKY regulation factor", seen)))
    check("实质不同角度 → 放行", not check_query_redundant(
        "blood orange Ruby1 retrotransposon cold induced", seen))
    check("空查询 → 放行", not check_query_redundant("", seen))


def test_evidence_report_builder():
    print("[SF-9] 确定性证据回执（v8.4.6）")
    from src.core.agent_runner import build_evidence_report
    arts = {"main_results": [
        {"doi": "10.1/a", "title": "Paper A", "year": "2024", "score": 0.9,
         "text": "Ruby1 受低温诱导，Corky 转座子插入启动子。"},
        {"doi": "10.1/a", "title": "Paper A dup", "year": "2024", "score": 0.8,
         "text": "重复条目"},
        {"title": "Paper B", "year": "2023", "score": 0.7,
         "text": "WRKY75 结合启动子。"},
    ], "web_results": [
        {"title": "W1", "doi": "10.2/x", "abstract": "非柑橘对照"},
    ]}
    rep = build_evidence_report(arts, "test query", 3)
    check("summary 含检索次数", "检索执行: 3 次" in rep)
    check("DOI 去重(2 条 main)", "[1]" in rep and "[2]" in rep and "[3]" not in rep)
    check("文献细节含标题", "Paper A" in rep and "Paper B" in rep)
    check("chunk 全文直接进上下文",
          "Ruby1 受低温诱导，Corky 转座子插入启动子。" in rep
          and "WRKY75 结合启动子。" in rep)
    check("web 补充条目", "[W1]" in rep)
    # v8.16.2: 引用导引前置（截尾防护——旧文末位置会被 cap 截断吃掉）
    check("引用编号指引（前置,在 [W1] 之前）",
          "引用编号请使用下列" in rep
          and rep.find("引用编号请使用下列") < rep.find("[W1]"))


def test_draft_publish():
    print("[SF-10] 写作草稿-发布（v8.4.6）")
    import uuid
    from src.core.write_pipeline import _draft_path, _publish_draft, _WORKSPACE_ROOT
    from src.tools.file_ops import write_local_file
    name = f"test_draft_{uuid.uuid4().hex[:6]}.md"
    draft = _draft_path(name)
    msg = write_local_file.func(draft, "# 测试\n内容", "write")
    check("草稿已写入", not msg.startswith("Error"), msg[:60])
    ok = _publish_draft(name)
    check("发布成功", ok)
    final = _WORKSPACE_ROOT / name
    check("正式文件存在且内容正确",
          final.exists() and "测试" in final.read_text(encoding="utf-8"))
    check("草稿已移除", not (_WORKSPACE_ROOT / draft).exists())
    if final.exists():
        final.unlink()


def test_output_profile():
    print("[SF-11] 输出画像（指标决定回答，v8.4.6）")
    from src.graph.expert_graph import _build_output_profile
    p8 = _build_output_profile(10, "fact")
    check("证据≥8 → 深度档", "1200~2000" in p8 and "至少覆盖 6 条" in p8, p8[:80])
    p3 = _build_output_profile(5, "method")
    check("3~7 → 标准档", "600~1200" in p3 and "覆盖全部 5 条" in p3, p3[:80])
    p0 = _build_output_profile(0, "fact")
    check("0 → 薄档", "300 字以内" in p0 and "模型知识" in p0, p0[:80])
    check("综述类不注入", _build_output_profile(10, "review") == "")


def test_pii_mask():
    print("[SF-12] 日志脱敏（v8.4.6 B6）")
    from src.core.pii_mask import mask_sensitive
    s = "联系 zhang@test.com 或 13800138000，身份证 11010119900307123X，key=sk-abc1234567890123456789"
    m = mask_sensitive(s)
    check("邮箱脱敏", "zhang@test.com" not in m and "<email>" in m)
    check("手机号脱敏", "13800138000" not in m and "<phone>" in m)
    check("身份证脱敏", "11010119900307123X" not in m and "<idcard>" in m)
    check("API Key 脱敏", "sk-abc1234567890123456789" not in m and "<api_key>" in m)
    check("正常文本不受影响", mask_sensitive("柑橘黄龙病研究") == "柑橘黄龙病研究")


def test_hard_trim_identifiers():
    print("[SF-13] 硬截断保留标识符（v8.4.6 B2）")
    from src.core.context_budget import ContextBudget, ContextBudgetConfig
    from langchain_core.messages import HumanMessage, ToolMessage
    cfg = ContextBudgetConfig(max_tokens=10, hard_threshold=0.9,
                              keep_recent_turns=1, protect_recent_turns=1)
    budget = ContextBudget(cfg)
    msgs = [
        HumanMessage(content="q1"),
        ToolMessage(content="[1] DOI: 10.1000/abc 证据内容", tool_call_id="a", name="t"),
        HumanMessage(content="q2"),
        ToolMessage(content="[2] DOI: 10.2000/xyz 证据内容", tool_call_id="b", name="t"),
        HumanMessage(content="q3"),
        HumanMessage(content="q4"),
    ]
    out = budget._hard_trim(msgs, cfg)
    text = "\n".join(str(getattr(m, "content", "")) for m in out)
    check("被丢轮次的 DOI 已保留", "10.1000/abc" in text and "10.2000/xyz" in text)
    check("保留标识符标记存在", "硬截断保留标识符" in text)
    check("最近轮次保留(keep=1 仅保留 q4)", "q4" in text and "q3" not in text)


def test_tool_meta_envelope():
    print("[SF-14] 工具结果结构化 envelope（v8.4.6 B8）")
    import asyncio as _asyncio
    from langchain_core.tools import tool
    from src.tools.registry import PartitionedToolNode

    @tool
    def _ok_tool(x: str) -> str:
        """返回 ok 前缀字符串。"""
        return f"ok:{x}"

    @tool
    def _err_tool(x: str) -> str:
        """返回错误标记字符串。"""
        return "[ERR_PARSE] 解析失败"

    node = PartitionedToolNode([_ok_tool, _err_tool])
    loop = _asyncio.new_event_loop()
    try:
        rs = loop.run_until_complete(node.execute_tools([
            {"id": "1", "name": "_ok_tool", "args": {"x": "a"}},
            {"id": "2", "name": "_err_tool", "args": {"x": "b"}},
        ]))
    finally:
        loop.close()
    meta0 = (getattr(rs[0], "artifact", {}) or {}).get("_meta", {})
    meta1 = (getattr(rs[1], "artifact", {}) or {}).get("_meta", {})
    check("成功工具 meta=ok/OK", meta0.get("status") == "ok" and meta0.get("code") == "OK")
    check("失败工具 meta=error/ERR_PARSE",
          meta1.get("status") == "error" and meta1.get("code") == "ERR_PARSE")


def test_retrieve_budget_and_convergence():
    print("[SF-15] 检索预算与代码收敛（v8.4.8）")
    import asyncio as _asyncio
    from langchain_core.messages import AIMessage, ToolMessage
    from src.core import agent_runner as ar

    state = {"rag": 0, "aca": 0, "turns": 0, "same_doi": False}

    class FakeLLM2:
        def __init__(self):
            self._turn = 0

        async def ainvoke(self, messages):
            # 每轮换新检索角度：查询 token 差异显著（同轮 Jaccard≈0.56、
            # 跨轮≈0.56，均 <0.85 去重阈值），确保测到的是预算上限而非去重
            self._turn += 1
            return AIMessage(content="", tool_calls=[
                {"id": f"r{i}", "name": "citrus_rag_search",
                 "args": {"query": f"retrieval angle {i} round {self._turn} "
                                   f"specific {self._turn * 10 + i}"}}
                for i in range(3)
            ])

    class FakeChat2:
        def __init__(self, *a, **k):
            self.fake = FakeLLM2()

        def bind_tools(self, tools=None):
            return self

        async def ainvoke(self, messages):
            return await self.fake.ainvoke(messages)

    orig_exec = ar.PartitionedToolNode.execute_tools

    async def fake_exec(self, tool_calls):
        msgs = []
        for tc in tool_calls:
            name = tc.get("name")
            if name == "citrus_rag_search":
                state["rag"] += 1
                # same_doi=True: 恒返回同一批 8 个 DOI（≥6 触发收敛判定），
                # 第 2 轮起新增 0 → 代码收敛提前结束（预期 rag=4）
                n_items = 8 if state["same_doi"] else 3
                n = 1 if state["same_doi"] else state["rag"]
                msgs.append(ToolMessage(
                    content="ok", tool_call_id=tc["id"], name=name,
                    artifact={"main_results": [
                        {"doi": f"10.{i}/{n}", "title": f"p{i}", "text": "x"}
                        for i in range(1, n_items + 1)
                    ]}))
            else:
                msgs.append(ToolMessage(content="ok", tool_call_id=tc["id"],
                                        name=name, artifact={}))
        return msgs

    import src.core.llm_pool as pool
    orig_get_llm = pool.get_llm

    def _run():
        loop = _asyncio.new_event_loop()
        try:
            return loop.run_until_complete(ar.run_agent(
                "retrieve-agent", {"goal": "g", "query": "q"}))
        finally:
            loop.close()

    try:
        ar.PartitionedToolNode.execute_tools = fake_exec
        pool.get_llm = lambda **kw: FakeChat2(**kw)

        # A) 预算: 每轮 3 rag 请求 → 只执行 2
        state.update(rag=0, aca=0, turns=0, same_doi=False)
        r = _run()
        check("每轮 rag 上限 2（3 轮共 6）", state["rag"] == 6, f"rag={state['rag']}")
        check("请求级 rag 预算 ≤6", state["rag"] <= 6, f"rag={state['rag']}")
        check("回执含预算拦截说明", "SEARCH_BUDGET" in r.get("result", ""),
              "result 未含预算说明")

        # B) 收敛: 从第 2 轮起无新证据（同 DOI）→ 代码提前结束（2 轮 4 次 rag）
        state.update(rag=0, aca=0, turns=0, same_doi=True)
        r2 = _run()
        check("边际收益过低 → 提前收敛（rag=4）", state["rag"] == 4,
              f"rag={state['rag']}")
    finally:
        ar.PartitionedToolNode.execute_tools = orig_exec
        pool.get_llm = orig_get_llm


def test_chat_cancel_endpoint():
    """v8.4.11 停止功能：cancel 端点取消会话所有 running job（安全点取消）。

    书中 §4.7.6 取消式处理：向运行任务发取消信号（task.cancel() →
    LLM/tool await 处抛 CancelledError），不在任意时刻强行掐断。
    """
    print("[SF-17] 用户中断：cancel 端点取消 running job")
    import asyncio as _asyncio
    import src.api.main as api_main
    import src.core.jobs as jobs_mod

    sid = "cancel-test-session"

    async def _run():
        # 1) 创建 job + 注册表挂一个"运行中"任务（模拟 graph 任务）
        job_id = jobs_mod.create_job(sid, "rid-cancel", "chat")
        ran = {"cancelled": False}

        async def fake_graph_task():
            try:
                await _asyncio.sleep(60)
            except _asyncio.CancelledError:
                ran["cancelled"] = True
                raise

        loop = _asyncio.get_event_loop()
        task = loop.create_task(fake_graph_task())
        api_main._running_graph_tasks[job_id] = task
        await _asyncio.sleep(0.05)  # 让任务进入 sleep

        # 2) cancel 端点（无运行任务注册时仅置状态；有则发取消信号）
        req = api_main.CancelRequest(session_id=sid)
        r = await api_main.cancel_chat(req)
        await _asyncio.sleep(0.1)  # 让取消传播

        # 3) 断言
        job = jobs_mod.get_job(job_id)
        check("cancel 返回 job_id", job_id in r.get("cancelled", []),
              str(r.get("cancelled")))
        check("运行任务收到取消信号", ran["cancelled"], f"cancelled={ran['cancelled']}")
        check("job 状态置 cancelled", (job or {}).get("status") == "cancelled",
              str((job or {}).get("status")))
        api_main._running_graph_tasks.pop(job_id, None)

        # 4) 无 running job 时：幂等返回 count=0
        r2 = await api_main.cancel_chat(req)
        check("无 running job 幂等", r2.get("count") == 0, f"count={r2.get('count')}")

    _asyncio.run(_run())


def test_stream_llm_aggregation():
    """v8.4.13 真流式：stream_llm_response 聚合与回调。

    断言：content/reasoning 逐 chunk 回调收到全部分片；聚合消息与 ainvoke
    同构（content 拼接、tool_calls 合并、usage_metadata 保留）；CitrusChatOpenAI
    的 delta 钩子把 reasoning_content 透传到 additional_kwargs。
    """
    print("[SF-18] 真流式：stream_llm_response 聚合与 reasoning 透传")
    import asyncio as _asyncio
    from langchain_core.messages import AIMessageChunk
    from langchain_core.outputs import ChatGenerationChunk
    from langchain_core.messages.ai import UsageMetadata
    from src.core.stream_llm import stream_llm_response
    from src.core.llm_pool import _install_reasoning_passthrough
    _install_reasoning_passthrough()

    chunks = [
        AIMessageChunk(content="你", additional_kwargs={"reasoning_content": "先"},
                       usage_metadata=UsageMetadata(
                           input_tokens=10, output_tokens=1, total_tokens=11)),
        AIMessageChunk(content="好", additional_kwargs={"reasoning_content": "思考"}),
        AIMessageChunk(content="，我来回答", additional_kwargs={"reasoning_content": "再回答"}),
    ]
    class FakeStreamLLM:
        async def astream(self, messages):
            for c in chunks:
                yield c

    got_text, got_rc = [], []
    resp = _asyncio.run(stream_llm_response(
        FakeStreamLLM(), [SystemMessage(content="x")],
        on_text=got_text.append, on_reasoning=got_rc.append))
    check("content 逐 chunk 回调", got_text == ["你", "好", "，我来回答"], str(got_text))
    check("reasoning 逐 chunk 回调", got_rc == ["先", "思考", "再回答"], str(got_rc))
    check("聚合 content 完整", resp.content == "你好，我来回答", resp.content)
    check("聚合消息与 ainvoke 同构", hasattr(resp, "tool_calls"), type(resp).__name__)

    # 带 tool_calls 的 chunk 聚合（工具轮）
    from langchain_core.messages.tool import tool_call_chunk
    tc_chunks = [
        AIMessageChunk(content="", tool_call_chunks=[
            tool_call_chunk(name="citrus_rag_search", args='{"query": "', id="t1", index=0)]),
        # 真实流式：name 仅首 chunk 携带，后续为 None（args 分片继续）
        AIMessageChunk(content="", tool_call_chunks=[
            tool_call_chunk(name=None, args='HITL"}', id="t1", index=0)]),
    ]
    class FakeStreamLLM2:
        async def astream(self, messages):
            for c in tc_chunks:
                yield c
    resp2 = _asyncio.run(stream_llm_response(FakeStreamLLM2(), []))
    tcs = resp2.tool_calls
    check("tool_calls 聚合", len(tcs) == 1 and tcs[0]["name"] == "citrus_rag_search"
          and tcs[0]["args"] == {"query": "HITL"}, str(tcs))

    # reasoning 透传：llm_pool 加载时 monkeypatch 的模块级转换函数
    import langchain_openai.chat_models.base as _lcb
    check("delta 转换透传 reasoning",
          _lcb._convert_delta_to_message_chunk(
              {"role": "assistant", "content": "hi",
               "reasoning_content": "think..."}, AIMessageChunk)
          .additional_kwargs.get("reasoning_content") == "think...",
          "delta passthrough")
    check("非流式转换透传 reasoning",
          _lcb._convert_dict_to_message(
              {"role": "assistant", "content": "hi",
               "reasoning_content": "think2"})
          .additional_kwargs.get("reasoning_content") == "think2",
          "dict passthrough")


def _restore_llm_pool(orig):
    import src.core.llm_pool as pool
    pool.get_llm = orig


def test_session_history_restore():
    """v8.4.9 会话持久化：历史对话读取端点（前端刷新/关闭重开后恢复渲染）。

    数据在 sessions.db 本就持久化（save_messages 全量入库），本测试验证读取
    端点只回用户可见的 Human/AI 轮次、过滤工具/系统消息、limit 生效。
    DB 放工作区内路径（沙箱可写；TEMP 下的 sqlite 在本环境被拒）。
    """
    print("[SF-16] 会话持久化：历史对话读取端点")
    import asyncio as _asyncio
    from pathlib import Path
    from src.session.manager import SessionManager
    import src.api.main as api_main
    from langchain_core.messages import ToolMessage, SystemMessage

    db_file = Path(__file__).resolve().parent / ".tmp_session_hist.db"
    if db_file.exists():
        db_file.unlink()
    sm = SessionManager()
    sm.db_path = str(db_file)
    sm._init_db_sync()
    sm._create_session_sync("hist-test")

    msgs = [
        HumanMessage(content="什么是 HITL？"),
        # v8.8: 协议合法结构——AI 带 tool_calls 后紧跟配对 ToolMessage
        # （此前是无 tool_calls 的 AI + 孤立 ToolMessage，被加载端 INV-01 防御丢弃）
        AIMessage(content="先检索 HITL 相关资料", tool_calls=[
            {"id": "t1", "name": "citrus_rag_search", "args": {}}]),
        ToolMessage(content="ok", tool_call_id="t1", name="citrus_rag_search"),
        SystemMessage(content="system noise"),
        # v8.7: 真实链路 user 消息为完整上下文 HumanMessage——显示层只回 <user_query> 原文
        HumanMessage(content=(
            "<long_term_memory>\n## 跨会话记忆\n- 内部记忆块</long_term_memory>\n\n"
            "<format_guide>\n内部格式指南</format_guide>\n\n"
            "<user_query>\n怎么验证？\n</user_query>")),
        AIMessage(content="通过审批卡片闭环验证。"),
    ]
    _asyncio.run(sm.save_messages("hist-test", msgs, "idem-hist-1"))

    orig_sm = api_main.session_manager
    api_main.session_manager = sm
    try:
        r = _asyncio.run(api_main.session_messages("hist-test"))
        roles = [m["role"] for m in r["messages"]]
        # v8.10l: 工具消息随历史返回（前端恢复工具链折叠块）——5 条而非 4 条
        check("端点返回 5 条（含工具消息）", r["count"] == 5, f"count={r['count']}")
        check("工具消息随历史返回", "tool" in roles, str(roles))
        check("系统消息被过滤", "system" not in roles, str(roles))
        check("顺序保持", roles == ["user", "assistant", "tool", "user", "assistant"], str(roles))
        tmsg = [m for m in r["messages"] if m["role"] == "tool"][0]
        check("工具消息含名称与截断标记",
              tmsg.get("name") == "citrus_rag_search" and "truncated" in tmsg, str(tmsg))
        check("内容完整", r["messages"][1]["content"] == "先检索 HITL 相关资料",
              r["messages"][1]["content"][:30] if r["messages"] else "none")
        # v8.7: 内部上下文块（记忆/格式指南）不显示，只回原始问题
        # v8.10l: 顺序为 user/assistant/tool/user/assistant——第二条 user 在 index 3
        u2 = r["messages"][3]["content"]
        check("user 消息裁剪为原始问题", u2 == "怎么验证？",
              u2[:80] if u2 else "none")
        check("内部块不泄露", "<long_term_memory>" not in u2
              and "<format_guide>" not in u2 and "<user_query>" not in u2, u2[:80])
        # 无 <user_query> 标签的旧数据回退原文
        check("无标签回退原文",
              api_main._user_display_text("普通旧消息") == "普通旧消息")

        r2 = _asyncio.run(api_main.session_messages("hist-test", limit=2))
        check("limit=2 生效", r2["count"] == 2, f"count={r2['count']}")

        r3 = _asyncio.run(api_main.session_messages("no-such-session"))
        check("不存在的会话返回空", r3["count"] == 0, f"count={r3['count']}")

        # v8.4.10: 上下文快照随历史返回（刷新后面板即时恢复，无需等下次提问）
        c = r.get("context") or {}
        check("快照含 estimated_tokens>0", (c.get("estimated_tokens") or 0) > 0,
              f"est={c.get('estimated_tokens')}")
        check("快照 history_msgs 与全量一致", c.get("history_msgs") == 6,
              f"history_msgs={c.get('history_msgs')}")
        check("快照含阈值与预算", c.get("max_tokens") == 1000000
              and c.get("soft_threshold") == 0.75, str({k: c.get(k) for k in
                                                        ("max_tokens", "soft_threshold")}))
        c3 = r3.get("context") or {}
        check("空会话快照 est=0", (c3.get("estimated_tokens") or 0) == 0,
              f"est={c3.get('estimated_tokens')}")
    finally:
        api_main.session_manager = orig_sm
        try:
            db_file.unlink()
        except Exception:
            pass


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
    test_query_dedup()
    test_evidence_report_builder()
    test_draft_publish()
    test_output_profile()
    test_retrieve_budget_and_convergence()
    test_session_history_restore()
    test_chat_cancel_endpoint()
    test_stream_llm_aggregation()
    test_pii_mask()
    test_hard_trim_identifiers()
    test_tool_meta_envelope()
    test_retrieve_budget_and_convergence()
    print(f"supervisor final tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
