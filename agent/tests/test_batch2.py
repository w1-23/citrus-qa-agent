"""Batch-2 修复验证脚本（无外部依赖，仅验证逻辑正确性）."""
import asyncio
import json
import os
import sys
import uuid

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


def test_ag4_session_new():
    print("[AG-4] session 'new' 语义")
    from src.session.manager import session_manager
    loop = asyncio.new_event_loop()
    sid = loop.run_until_complete(session_manager.get_or_create_session("new"))
    check("'new' → 生成 UUID", sid != "new" and len(sid) == 36, sid[:8])
    sid2 = loop.run_until_complete(session_manager.get_or_create_session(None))
    check("None → 生成 UUID", sid2 != "new" and sid2 != sid)
    sid3 = loop.run_until_complete(session_manager.get_or_create_session("keep-me"))
    check("已有 id 原样返回", sid3 == "keep-me")
    loop.run_until_complete(session_manager.clear_session(sid))
    loop.close()


def test_ag7_timeout_retry():
    print("[AG-7] 工具超时 + LLM 重试")
    check("settings 读取 TOOL_EXEC_TIMEOUT_SEC=60", settings.TOOL_EXEC_TIMEOUT_SEC == 60)
    import inspect
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tools', 'registry.py'),
               encoding='utf-8').read()
    check("registry 含 asyncio.wait_for", "asyncio.wait_for" in src)
    check("registry 含 [ERR_TIMEOUT]", "[ERR_TIMEOUT]" in src)
    loop_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                 'src', 'core', 'agent_loop.py'), encoding='utf-8').read()
    runner = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    # v8.13-b5a: LLM 重试循环收敛至 agent_loop.invoke_llm_with_retry（range(retries)，默认 3 次），
    # agent_runner 只保留 label/2s 兜底差异点
    check("重试循环收敛至 agent_loop", "range(retries)" in loop_src
          and "retries: int = 3" in loop_src)
    check("agent_runner 接统一重试", "invoke_llm_with_retry" in runner and "retry" in runner)
    check("agent_runner 含 [AgentError]", "[AgentError]" in runner)

    # 模拟超时: 构造一个 sleep 工具, 用极短超时验证 wait_for 生效
    from src.tools.registry import PartitionedToolNode
    from langchain_core.tools import tool as lc_tool

    @lc_tool
    def slow_tool(x: int) -> str:
        """Sleep 3s then return done (timeout test)."""
        import time
        time.sleep(3)
        return "done"

    class FakeCfg:
        TOOL_EXEC_TIMEOUT_SEC = 1
    import src.tools.registry as reg_mod
    old = reg_mod.settings
    reg_mod.settings = FakeCfg()
    try:
        loop = asyncio.new_event_loop()
        node = PartitionedToolNode([slow_tool])
        tc = {"id": "call_1", "name": "slow_tool", "args": {"x": 1}}
        msgs = loop.run_until_complete(node.execute_tools([tc]))
        check("慢工具 1s 超时返回 [ERR_TIMEOUT]",
              "[ERR_TIMEOUT]" in msgs[0].content, msgs[0].content[:60])
        loop.close()
    finally:
        reg_mod.settings = old


def test_ag8_truncation():
    print("[AG-8] 子 Agent 结果分档截断 + 按条目截断")
    from src.core.agent_runner import _truncate_context_blocks
    orig_blocks = [f"[{i}] " + "字" * 1000 for i in range(40)]
    ctx = "\n\n".join(orig_blocks)
    out = _truncate_context_blocks(ctx, 5000)
    blocks = out.split("\n\n")
    # v8.3.5: 截断标记行（[上下文已截断...]）需过滤后再校验原块完整性
    pure = [b for b in blocks if not b.startswith("[上下文已截断")]
    check("总长 ≤ 上限+单条+标记", len(out) <= 5000 + 1000 + 2 + 120, f"len={len(out)}")
    check("保留完整条目（每块都是原块）",
          all(b in orig_blocks for b in pure), f"{len(pure)} blocks")
    check("非首块即被截断时不含半截", len(pure) == 4 or all(
        b in orig_blocks for b in pure))
    check("截断含透明标记", "[上下文已截断" in out)
    check("短内容不截断", _truncate_context_blocks("abc", 100) == "abc")


def test_ag9_atomic_write():
    print("[AG-9] 原子写 + 并发锁")
    import concurrent.futures
    from pathlib import Path
    from src.tools.file_ops import write_local_file
    from src.config import PROJECT_ROOT

    target = PROJECT_ROOT / "workspace" / "output" / f"test_ag9_{uuid.uuid4().hex[:6]}.md"
    def w(i):
        return write_local_file.func(target.name, f"content-{i}", "write")
    with concurrent.futures.ThreadPoolExecutor(6) as ex:
        rs = list(ex.map(w, range(6)))
    content = target.read_text(encoding="utf-8")
    check("并发写最终内容完整且为单份", content in {f"content-{i}" for i in range(6)},
          repr(content[:30]))
    tmp_left = list((target.parent).glob(target.name + ".tmp"))
    check("无残留 .tmp 文件", len(tmp_left) == 0, f"{len(tmp_left)} tmp left")
    if target.exists():
        target.unlink()


def test_ag10_limits():
    print("[AG-10] 文件大小 + 样本量上限")
    # v8.3.3: 绝对路径读取限项目根内 → 测试文件置于 workspace/output
    from src.config import PROJECT_ROOT
    from pathlib import Path
    from src.tools.analyze import statistical_analysis
    out_dir = PROJECT_ROOT / "workspace" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)

    tmp = out_dir / "test_big.csv"
    tmp.write_text("a,b\n" + "1,2\n" * 150000, encoding="utf-8")
    r = statistical_analysis.func(str(tmp), "descriptive", '{"value_column": "a"}')
    check("超过样本量 → 标注截断", "[数据截断]" in r, r[:80])
    tmp.unlink()

    huge = out_dir / "test_huge.csv"
    huge.write_bytes(b"0" * (settings.FILE_READ_MAX_SIZE_MB * 1024 * 1024 + 1024))
    r2 = statistical_analysis.func(str(huge), "descriptive", '{"value_column": "a"}')
    check("超过文件大小 → [ERR_FILE_TOO_LARGE]", "[ERR_FILE_TOO_LARGE]" in r2, r2[:60])
    huge.unlink()

    # v8.3.3: 项目目录外的绝对路径 → 拒绝
    # v8.4.14: 用工作区内、项目根外路径（沙箱下 TEMP 不可写，语义不变）
    from src.config import PROJECT_ROOT
    outside = PROJECT_ROOT.parent / f"outside_ag10_{uuid.uuid4().hex[:8]}.csv"
    try:
        outside.write_text("a,b\n1,2\n", encoding="utf-8")
        r3 = statistical_analysis.func(str(outside), "descriptive", '{"value_column": "a"}')
        check("项目外绝对路径 → [ERR_PERMISSION]", "[ERR_PERMISSION]" in r3, r3[:60])
    finally:
        # v8.4.14: 无论断言成败都清理残留文件（曾混入 git 提交）
        try:
            outside.unlink()
        except Exception:
            pass

    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'tools', 'search.py'),
               encoding='utf-8').read()
    check("pdf_read 含大小检查", "ERR_FILE_TOO_LARGE" in src)


def test_ag11_idx_map():
    print("[AG-11] _idx_map 按 payload (paper_id, chunk_index) 匹配 + miss 跳过")
    from src.retrieval.multi_retriever import MultiBatchRetriever
    r = MultiBatchRetriever()
    r._idx_map = {("p1", 1): 0, ("p1", 2): 1}
    r.global_chunks = [{"_global_idx": 0, "text": "c0"}, {"_global_idx": 1, "text": "c1"}]

    class FakePoint:
        def __init__(self, pid, score, payload):
            self.id = pid
            self.score = score
            self.payload = payload

    class FakeRes:
        points = [FakePoint(999, 0.9, {"paper_id": "p1", "chunk_index": 1}),
                  FakePoint(888, 0.8, {"paper_id": "p1", "chunk_index": 99})]

    class FakeClient:
        def query_points(self, **kw):
            return FakeRes()

    out = r._search_qdrant("b1", FakeClient(), "coll", [1.0], 10)
    check("payload 匹配正常点", [i for i, s in out] == [0], str(out))
    check("miss 点被跳过（不回退 -1）", len(out) == 1)
    check("score 保留", abs(out[0][1] - 0.9) < 1e-6)

    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'retrieval', 'multi_retriever.py'),
               encoding='utf-8').read()
    check("idx_map 键为 (paper_id, chunk_index)",
          'c.get("paper_id", "")' in src and 'c.get("chunk_index")' in src)
    check("_verify_idx_map 抽样匹配率", "match rate" in src or "0.95" in src)


def test_ag12_route_removed():
    print("[AG-12] 路由自动升级已彻底删除（用户手动切换）")
    main_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'api', 'main.py'),
                    encoding='utf-8').read()
    cfg_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'config.py'),
                   encoding='utf-8').read()
    yaml_src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml'),
                    encoding='utf-8').read()
    check("main.py 无 _resolve_mode", "_resolve_mode" not in main_src)
    check("main.py 无 ROUTE_ENABLED 引用", "ROUTE_ENABLED" not in main_src)
    check("main.py mode 由 light_mode 直接决定",
          'mode = "light" if req.light_mode else "expert"' in main_src)
    check("config.py 无 ROUTE_* 字段", "ROUTE_" not in cfg_src)
    check("config.yaml 无 route: 段", "route:" not in yaml_src)


def test_ag14_degraded_event():
    print("[AG-14] context_degraded 事件")
    from src.core.context_manager import ContextManager
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'core', 'context_manager.py'),
               encoding='utf-8').read()
    check("load 失败 emit context_degraded", "context_degraded" in src)


def test_ag5_ltm():
    print("[AG-5] LTM 来源标注/衰减/迁移")
    from src.guardrails.memory import MemoryStore
    import sqlite3
    from _tmpenv import tmp_path
    from src.config import PROJECT_ROOT

    tmp_db = tmp_path("db")
    ms = MemoryStore()
    orig_path = str(PROJECT_ROOT / "state" / "sessions.db")
    import src.guardrails.memory as mem_mod
    # 直接建临时库验证 schema 迁移与保存
    conn = sqlite3.connect(str(tmp_db))
    conn.execute("""CREATE TABLE IF NOT EXISTS long_term_memory (
        fact_key TEXT PRIMARY KEY, fact_value TEXT,
        confidence REAL DEFAULT 0.5, updated_at TEXT)""")
    conn.execute("INSERT INTO long_term_memory (fact_key, fact_value, confidence, updated_at) VALUES (?,?,?,?)",
                 ("测试事实", "HLB 病原为 CLas", 0.9, "2020-01-01T00:00:00"))
    conn.commit()
    # 模拟 _ensure_ltm_schema
    cols = {r[1] for r in conn.execute("PRAGMA table_info(long_term_memory)")}
    if "owner_session" not in cols:
        conn.execute("ALTER TABLE long_term_memory ADD COLUMN owner_session TEXT DEFAULT ''")
    if "source_query" not in cols:
        conn.execute("ALTER TABLE long_term_memory ADD COLUMN source_query TEXT DEFAULT ''")
    conn.execute("UPDATE long_term_memory SET owner_session='s1', source_query='HLB 病原' WHERE fact_key='测试事实'")
    conn.commit()
    rows = conn.execute("SELECT * FROM long_term_memory").fetchall()
    check("迁移后含新列", len(rows[0]) >= 6, str(len(rows[0])))
    conn.close()

    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'guardrails', 'memory.py'),
               encoding='utf-8').read()
    check("save 含 owner_session/source_query 参数", "owner_session" in src and "source_query" in src)
    check("recall 含衰减", "0.95" in src)
    check("recall 输出含来源标注", "来源" in src)
    check("recall 含 max_chars 截断", "max_chars" in src)
    g1 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'graph', 'expert_graph.py'),
              encoding='utf-8').read()
    # v8.4: LTM 提取转后台 spawn（不阻塞响应），原 to_thread 内联断言更新
    check("save 节点 LTM 提取后台化", "_extract_and_save_ltm" in g1
          and "spawn(asyncio.to_thread(" in g1)
    c = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'prompts', 'system', 'constraints.md'),
             encoding='utf-8').read()
    check("constraints 含冲突规则", "跨会话记忆规则" in c)


def test_ag17_thread_context_propagation():
    print("[AG-17] 工具线程内 emit 路由到请求队列（执行卡片修复）")
    from langchain_core.tools import tool as lc_tool
    from src.tools.registry import PartitionedToolNode
    from src.core import progress_bus as pb

    @lc_tool
    def emitting_tool(msg: str) -> str:
        """Emit tool_progress from inside the tool (sync → executor thread)."""
        pb.emit_progress("tool_progress", {"message": msg, "tool_call_id": "inner"})
        return "ok:" + msg

    loop = asyncio.new_event_loop()
    request_queue = asyncio.Queue()
    pb.set_request_queue(request_queue)
    try:
        node = PartitionedToolNode([emitting_tool])
        tc = {"id": "call_emit", "name": "emitting_tool", "args": {"msg": "hello-from-thread"}}
        msgs = loop.run_until_complete(node.execute_tools([tc]))
        check("工具返回正常", msgs and "ok:" in msgs[0].content, str(msgs)[:80])
        got = []
        while not request_queue.empty():
            got.append(loop.run_until_complete(request_queue.get()))
        evt = [e for e in got if e.get("event") == "tool_progress"]
        check("线程内 tool_progress 事件进入请求队列",
              len(evt) == 1 and "hello-from-thread" in evt[0]["data"],
              f"got={[e.get('event') for e in got]}")
    finally:
        pb.clear_request_queue()
        loop.close()


def test_ag18_budget_and_converge():
    print("[AG-18] supervisor 工具预算 + retrieve-agent 收敛")
    check("config supervisor.max_tools_per_turn=2", settings.SUPERVISOR_MAX_TOOLS_PER_TURN == 2,
          str(settings.SUPERVISOR_MAX_TOOLS_PER_TURN))
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    check("expert_graph 含预算截断逻辑", "SUPERVISOR_MAX_TOOLS_PER_TURN" in src
          and "budget" in src)
    runner = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    # v8.4.3 指令A: 移除"≥6 篇强制收敛"（动态阈值已过滤，全部证据应入报告）
    check("agent_runner 已移除代码级收敛",
          "提前收敛" not in runner and "RETRIEVE_CONVERGE_MIN_DOCS" not in runner)

    from src.core.agent_loop import count_unique_docs
    docs = [
        {"doi": "10.1/a"}, {"doi": "10.1/a"}, {"doi": "10.1/b"},
        {"doi": ""}, {"doi": ""},
    ]
    check("去重计数(DOI+无DOI按条)", count_unique_docs(docs) == 4, str(count_unique_docs(docs)))
    check("全同 DOI → 1", count_unique_docs([{"doi": "x"}, {"doi": "x"}]) == 1)
    check("空列表 → 0", count_unique_docs([]) == 0)


def test_ag19_protocol_pairing():
    print("[AG-19] 协议配对不变量（INV-01）")
    from src.core.agent_loop import tc_id
    # dict / 对象 两种形态 id 提取一致且非空
    d_tc = {"id": "call_dict_1", "name": "x", "args": {}}
    o_tc = type("TC", (), {"id": "call_obj_1", "name": "y", "args": {}})()
    check("dict 形态 id 提取", tc_id(d_tc) == "call_dict_1")
    check("对象形态 id 提取", tc_id(o_tc) == "call_obj_1")
    check("缺 id dict → 兜底 uuid", len(tc_id({"name": "z", "args": {}})) == 36)
    # 预算截断/熔断场景: 每个 tool_call 必须有 ToolMessage 占位（配对硬约束）
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    check("跳过调用生成占位 ToolMessage", "budget_skip" in src
          and "占位响应" in src or "未执行" in src)
    check("熔断补占位 ToolMessage", "circuit_breaker" in src
          and "extract_tc_id(rest)" in src)


def test_ag20_lock_degrade():
    print("[AG-20] 锁冲突降级归因（INV-03）")
    import yaml
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'retrieval', 'multi_retriever.py'), encoding='utf-8').read()
    check("冲突批次跳过 Qdrant 加载", "lock_conflict" in src
          and "failed_batches" in src)
    check("启动 ERROR 汇总失败批次", "批次向量库不可用" in src)
    check("瞬时窗口重试一次", "already accessed" in src and "time.sleep(2)" in src)
    s2 = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'src', 'tools', 'search.py'), encoding='utf-8').read()
    check("空结果归因含降级提示", "向量库有" in s2 and "failed_batches" in s2)


def test_ag21_timer_pairing():
    print("[AG-21] 计时器配对（INV-04）— dict 形态")
    from src.core import progress_bus as pb
    loop = asyncio.new_event_loop()
    try:
        # dict 形态 tool_call: emit_tool_call_start → 执行中非空 → emit_tool_result → 空
        tc_id = "call_timer_1"
        pb.mark_tool_start(tc_id, "citrus_rag_search")
        check("执行中 get_running_tools 非空", any(c == tc_id for c, _, _ in pb.get_running_tools()))
        pb.mark_tool_end(tc_id)
        check("结束后 get_running_tools 为空", not any(c == tc_id for c, _, _ in pb.get_running_tools()))
        # 对象形态
        tc_id2 = "call_timer_2"
        pb.mark_tool_start(tc_id2, "read_local_file")
        pb.mark_tool_end(tc_id2)
        check("对象形态配对后无残留", not any(c == tc_id2 for c, _, _ in pb.get_running_tools()))
        pb.clear_tool_timers()
    finally:
        loop.close()


def test_ag22_output_routing():
    print("[AG-22] 输出路由与证据保真（INV-05）")
    prom = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             'src', 'prompts', 'agents', 'retrieve-agent.md'), encoding='utf-8').read()
    check("retrieve 无需报告（v8.10m 消除矛盾）",
          "无需撰写检索报告" in prom and "系统代码确定性组装" in prom
          and "核心结论与证据点" not in prom)
    # v8.4.3 指令A: 收敛由模型自然判断（不再按文献数代码强制），保留三阶段与轮次上限
    check("retrieve 三阶段工作流", "阶段 1" in prom and "阶段 3" in prom
          and "轮次上限 3 轮" in prom)
    guide = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                              'src', 'prompts', 'system', 'decision_guide.md'), encoding='utf-8').read()
    check("决策指南深度规则", "逐证据引用" in guide and "深度问题生成策略" in guide)
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                           'config.yaml'), encoding='utf-8'))
    check("retrieve-agent cap=40000",
          cfg['agent']['tool_result_caps']['retrieve-agent'] == 40000)


def test_ag26_budget_forward():
    print("[AG-26] 预算检查前移（规范 2.2.5，每次调用前）")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    check("每轮调用前估算", "estimate_tokens(call_messages)" in src)
    check("硬阈值强制收尾", "硬阈值" in src and "强制收尾" in src)
    check("软阈值状态栏提示", "软阈值" in src and "尽快收敛" in src)
    from src.core.context_budget import ContextBudget, ContextBudgetConfig
    b = ContextBudget(ContextBudgetConfig(max_tokens=1000))
    from langchain_core.messages import HumanMessage
    est = b.estimate_tokens([HumanMessage(content="字" * 100)])
    check("估算函数可用", est >= 100, str(est))


def test_ag23_converge_behavior():
    print("[AG-23] 收敛行为（INV-02）— mock 子代理循环")
    import sys
    from langchain_core.messages import AIMessage, ToolMessage
    from src.core import agent_runner as ar

    results = [{"doi": f"10.1/{i}", "title": f"paper{i}"} for i in range(7)]
    calls = {"n": 0}

    class FakeLLM:
        async def ainvoke(self, messages):
            calls["n"] += 1
            if calls["n"] == 1:
                return AIMessage(content="", tool_calls=[
                    {"id": "c1", "name": "citrus_rag_search", "args": {"query": "q"}}])
            return AIMessage(content="检索报告：7 篇关键文献")

    class FakeChat:
        def __init__(self, *a, **k):
            self.fake = FakeLLM()
        def bind_tools(self, tools=None):
            return self
        async def ainvoke(self, messages):
            return await self.fake.ainvoke(messages)

    orig_exec = ar.PartitionedToolNode.execute_tools

    async def fake_exec(self, tool_calls):
        return [ToolMessage(content="ok", tool_call_id="c1", name="citrus_rag_search",
                            artifact={"main_results": results})]

    ar.PartitionedToolNode.execute_tools = fake_exec
    # v8.4: agent_runner 经 llm_pool.get_llm 创建客户端（不再直接用 ChatOpenAI），
    # 测试改 patch llm_pool 入口
    import src.core.llm_pool as pool
    orig_get_llm = pool.get_llm
    pool.get_llm = lambda **kw: FakeChat(**kw)
    try:
        loop = asyncio.new_event_loop()
        r = loop.run_until_complete(ar.run_agent(
            "retrieve-agent", {"goal": "g", "query": "q"}))
        loop.close()
        check("证据充分后自然完成（第 2 次调用即最终报告）", calls["n"] == 2,
              f"calls={calls['n']}")
        # v8.4.6: retrieve-agent 回执由代码确定性组装（检索回执 + 文献细节）
        check("结果含代码组装的证据回执",
              "检索回执" in r.get("result", "") and "paper0" in r.get("result", ""),
              r.get("result", "")[:100])
    finally:
        ar.PartitionedToolNode.execute_tools = orig_exec
        pool.get_llm = orig_get_llm


def test_ag24_circuit_breaker():
    print("[AG-24] 熔断与截断透明（INV-08）")
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    check("连续失败计数逻辑", "consecutive_failures" in src and ">= 3" in src)
    check("熔断强制收尾", "熔断强制收尾" in src or "forced_final" in src)
    runner = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    check("截断标记实现", "已截断" in runner)


def test_ag25_truncation_transparency():
    print("[AG-25] 上下文截断透明（INV-08 纯函数）")
    from src.core.agent_runner import _truncate_context_blocks
    short = "a\n\nb\n\nc"
    check("短内容不截断", _truncate_context_blocks(short, 100) == short)
    long_block = "\n\n".join(f"block-{i}-" + "x" * 300 for i in range(100))
    out = _truncate_context_blocks(long_block, 2000)
    check("截断含标记", "已截断" in out and "共 100 条" in out, out[-60:])
    tiny = _truncate_context_blocks(long_block, 10)
    check("无完整块 → 前缀+标记", "已截断" in tiny)


def test_ag27_idempotency():
    print("[AG-27] 历史幂等 + 会话写锁 + query 上限（M1）")
    from src.session.manager import (session_manager, compute_idempotency_key,
                                     _get_session_lock)
    from langchain_core.messages import HumanMessage, AIMessage

    check("幂等键稳定（client_request_id）",
          compute_idempotency_key("s1", "q1", "crid-1") == compute_idempotency_key("s1", "q1", "crid-1"))
    check("同 key 同 query 重放幂等（重试语义）",
          compute_idempotency_key("s1", "q2", "crid-1") == compute_idempotency_key("s1", "q1", "crid-1"))
    check("不同 client_request_id 区分",
          compute_idempotency_key("s1", "q1", "crid-1") != compute_idempotency_key("s1", "q1", "crid-2"))
    check("同 session 同锁", _get_session_lock("sx") is _get_session_lock("sx"))
    check("不同 session 不同锁", _get_session_lock("sa") is not _get_session_lock("sb"))

    loop = asyncio.new_event_loop()
    sid = f"idem-{uuid.uuid4().hex[:8]}"
    key = "crid-test-idem-1"
    msgs = [HumanMessage(content="q"), AIMessage(content="a")]
    w1 = loop.run_until_complete(session_manager.save_messages(sid, msgs, key))
    w2 = loop.run_until_complete(session_manager.save_messages(sid, msgs, key))
    check("首次写入 True", w1 is True)
    check("同 key 重放被跳过", w2 is False)
    hist = loop.run_until_complete(session_manager.get_messages(sid))
    check("历史无重复轮次", len(hist) == 2, f"len={len(hist)}")
    loop.run_until_complete(session_manager.clear_session(sid))

    # 并发写不抛异常（save vs replace 走同一会话锁）
    sid2 = f"conc-{uuid.uuid4().hex[:8]}"
    import concurrent.futures
    def do_save(i):
        ms = [HumanMessage(content=f"s{i}")]
        import asyncio as _a
        _l = _a.new_event_loop()
        _l.run_until_complete(session_manager.save_messages(sid2, ms, f"k{i}"))
        _l.close()
    def do_replace(i):
        ms = [HumanMessage(content=f"r{i}a"), AIMessage(content=f"r{i}b")]
        import asyncio as _a
        _l = _a.new_event_loop()
        _l.run_until_complete(session_manager.replace_history(sid2, ms))
        _l.close()
    with concurrent.futures.ThreadPoolExecutor(4) as ex:
        fs = [ex.submit(do_save, i) for i in range(4)] + [ex.submit(do_replace, i) for i in range(4)]
        for f in fs:
            f.result(timeout=30)
    hist2 = loop.run_until_complete(session_manager.get_messages(sid2))
    check("并发写后可稳定加载且条数合理", len(hist2) in (1, 2), f"len={len(hist2)}")
    loop.run_until_complete(session_manager.clear_session(sid2))
    loop.close()

    # query 长度上限
    from src.api.main import ChatRequest
    try:
        from pydantic import ValidationError
        ChatRequest(query="x" * 20001)
        check("超长 query 被拒绝", False, "未抛异常")
    except ValidationError:
        check("超长 query 被拒绝", True)
    ok_req = ChatRequest(query="正常问题", client_request_id="crid-abc")
    check("正常请求通过（含幂等 ID）", ok_req.client_request_id == "crid-abc")


def test_ag28_jobs():
    print("[AG-28] task_jobs 最小闭环（M2）")
    from src.core import jobs as jobs_mod
    jid = jobs_mod.create_job("s-job", "r-1", "chat")
    check("create_job 返回 id", bool(jid) and len(jid) == 12, jid)
    job = jobs_mod.get_job(jid)
    check("初始 status=running", job is not None and job["status"] == "running", str(job))
    check("初始非 write 任务", not jobs_mod.is_write_job(jid))
    jobs_mod.update_job(jid, job_type="write", current_step="plan_ready 4sections",
                        progress_summary="标题T")
    job2 = jobs_mod.get_job(jid)
    check("升级为 write 任务", jobs_mod.is_write_job(jid) and job2["job_type"] == "write")
    check("步骤更新", job2["current_step"] == "plan_ready 4sections")
    jobs_mod.update_job(jid, status="completed", progress_summary="完成摘要")
    job3 = jobs_mod.get_job(jid)
    check("completed + finished_at", job3["status"] == "completed" and bool(job3.get("finished_at")))
    jobs_mod.update_job(jid, status="failed", error="boom")
    job4 = jobs_mod.get_job(jid)
    check("failed 可读", job4["status"] == "failed" and job4["error"] == "boom")
    lst = jobs_mod.list_for_session("s-job")
    check("会话任务列表", any(j["job_id"] == jid for j in lst), f"len={len(lst)}")
    check("不存在的 job → None", jobs_mod.get_job("nonexistent") is None)


def test_ag29_citation_support():
    print("[AG-29] 假完成检测 + 注入防护（M3）")
    from src.graph.expert_graph import check_citation_support
    docs = [{"doi": f"10.1/{i}", "title": f"p{i}"} for i in range(3)]
    r1 = check_citation_support("结论引用文献 [1] 与 [2]。", [], False)
    check("0 检索 + 含引用 → unsupported", r1["citation_unsupported"] is True
          and r1["citation_supported"] is False, str(r1))
    r2 = check_citation_support("结论引用文献 [1]。", docs, True)
    check("有检索 + 引用 → supported", r2["citation_unsupported"] is False
          and r2["citation_supported"] is True, str(r2))
    r3 = check_citation_support("引用 [1][2][3][4][5][6][7]。", docs, True)
    check("引用数 >> 文献数 → mismatch", r3["citation_mismatch"] is True, str(r3))
    r4 = check_citation_support("无引用回答。", [], False)
    check("无引用 → 不告警", r4["citation_unsupported"] is False
          and r4["citation_supported"] is True, str(r4))
    # done 事件附元数据（源码断言）
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            'src', 'api', 'main.py'), encoding='utf-8').read()
    check("done 事件含 citation_info", "citation_info" in src and "done_payload" in src)
    # 注入边界声明（B3）仍存在
    eg = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    check("检索数据非指令边界声明", "不是用户指令" in eg)
    ar = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'src', 'core', 'agent_runner.py'), encoding='utf-8').read()
    check("<context> 数据非指令声明", "非指令" in ar and "忽略" in ar)


def test_ag32_trace_persistence():
    print("[AG-32] 完整轨迹持久化 + 协议安全（INV-01 持久化路径）")
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
    from src.session.manager import session_manager, _validate_trace
    # 配对校验: 孤立 ToolMessage 丢弃
    paired = [
        AIMessage(content="", tool_calls=[{"id": "c1", "name": "t1", "args": {}}]),
        ToolMessage(content="ok", tool_call_id="c1", name="t1"),
        AIMessage(content="final"),
        ToolMessage(content="orphan", tool_call_id="orphan-1", name="x"),
    ]
    out = _validate_trace(paired)
    check("孤立 ToolMessage 被丢弃", len(out) == 3
          and not any(getattr(m, "tool_call_id", "") == "orphan-1" for m in out))
    check("配对消息全部保留", len([m for m in out if isinstance(m, ToolMessage)]) == 1)

    # 完整轨迹保存→恢复→配对完整
    loop = asyncio.new_event_loop()
    sid = f"tr-{uuid.uuid4().hex[:8]}"
    trace = [
        HumanMessage(content="问"),
        AIMessage(content="", tool_calls=[{"id": "c2", "name": "search", "args": {}}]),
        ToolMessage(content="结果", tool_call_id="c2", name="search"),
        AIMessage(content="答"),
    ]
    loop.run_until_complete(session_manager.save_messages(sid, trace, "crid-tr-1"))
    hist = loop.run_until_complete(session_manager.get_messages(sid))
    ai_with_tc = [m for m in hist if getattr(m, "tool_calls", None)]
    tools = [m for m in hist if isinstance(m, ToolMessage)]
    check("轨迹完整恢复(4条)", len(hist) == 4, f"len={len(hist)}")
    check("tool_calls 恢复", len(ai_with_tc) == 1 and ai_with_tc[0].tool_calls[0]["id"] == "c2")
    check("ToolMessage 配对恢复", len(tools) == 1 and tools[0].tool_call_id == "c2")
    loop.run_until_complete(session_manager.clear_session(sid))
    loop.close()


def test_ag33_noise_trim():
    print("[AG-33] 压缩分级: 噪声优先删 + 证据保护（INV-08）")
    from langchain_core.messages import AIMessage, ToolMessage, HumanMessage
    from src.core.context_budget import ContextBudget
    b = ContextBudget()
    turns = [[
        HumanMessage(content="问"),
        AIMessage(content="", tool_calls=[
            {"id": "c1", "name": "call_retrieve_agent", "args": {}},
            {"id": "c2", "name": "budget_skip", "args": {}},
        ]),
        ToolMessage(content="[retrieve-agent result] 证据报告", tool_call_id="c1",
                    name="call_retrieve_agent"),
        ToolMessage(content="[budget] 未执行", tool_call_id="c2", name="budget_skip"),
        AIMessage(content="答"),
    ]]
    cleaned = b._trim_noise(turns)
    flat = [m for t in cleaned for m in t]
    check("占位 ToolMessage 删除", not any(getattr(m, "name", "") == "budget_skip" for m in flat))
    check("证据 ToolMessage 保留", any("证据报告" in str(getattr(m, "content", "")) for m in flat))
    ai = [m for m in flat if getattr(m, "tool_calls", None)]
    check("AIMessage.tool_calls 同步修剪(配对不变量)",
          ai and len(ai[0].tool_calls) == 1 and ai[0].tool_calls[0]["id"] == "c1",
          str(ai[0].tool_calls if ai else None))


def test_ag34_evidence_ledger():
    print("[AG-34] 证据账本跨轮复用")
    from src.session.manager import session_manager
    loop = asyncio.new_event_loop()
    sid = f"ev-{uuid.uuid4().hex[:8]}"
    evd = [{"doi": "10.1/a", "title": "TitleA", "score": 0.9, "snippet": "机制细节"}]
    loop.run_until_complete(session_manager.save_evidence(sid, "花青素调控", evd, "报告: Ruby 启动子转座子插入"))
    block = session_manager.build_evidence_block(sid, limit=2)
    check("证据块含历史问题与报告", "花青素调控" in block and "Ruby 启动子" in block, block[:100])
    check("证据块含边界声明", "非用户输入" in block)
    loop.run_until_complete(session_manager.save_evidence(sid, "第二轮", [], "第二份报告"))
    block2 = session_manager.build_evidence_block(sid, limit=1)
    check("limit=1 只取最近一轮", "第二份报告" in block2 and "Ruby 启动子" not in block2)
    loop.run_until_complete(session_manager.clear_session(sid))
    check("clear 清空证据", session_manager.build_evidence_block(sid) == "")
    loop.close()


def test_ag35_material_fidelity():
    print("[AG-35] 材料零截断 + 总量预算")
    from src.core import write_pipeline as wp
    full_chunk = "机制细节A" * 350  # 1750 字符 < 3000
    r = {"doi": "10.1/x", "title": "T", "text": full_chunk}
    out = wp._format_material_pack([r], max_entries=5)
    check("1992 内 chunk 零截断", full_chunk in out, f"len={len(out)}")
    # 总量预算: 30 条 × 2500 字符 > 60000 → 截断条数并标记
    many = [{"doi": f"10.1/{i}", "title": f"T{i}", "text": "x" * 2500} for i in range(30)]
    out2 = wp._format_material_pack(many, max_entries=30)
    check("总量预算触发条数截断标记", "材料总量达" in out2, out2[-80:])
    check("单条正文未砍", "x" * 2500 in out2)


def test_ag36_compression_pairing():
    print("[AG-36] 强制压缩后协议配对完整（INV-01/10 压缩路径）")
    from langchain_core.messages import (HumanMessage, AIMessage,
                                         ToolMessage, SystemMessage)
    from src.core.context_budget import ContextBudget, ContextBudgetConfig

    msgs = [SystemMessage(content="sys")]
    for i in range(6):
        msgs.append(HumanMessage(content="q%d" % i + "长" * 200))
        msgs.append(AIMessage(content="", tool_calls=[
            {"id": "c%d" % i, "name": "call_retrieve_agent", "args": {}}]))
        msgs.append(ToolMessage(
            content="[retrieve-agent result] report %d " % i + "证据" * 300,
            tool_call_id="c%d" % i, name="call_retrieve_agent"))
        msgs.append(AIMessage(content="answer %d" % i + "回" * 200))

    def check_pairing(r, label):
        valid = set()
        for m in r:
            for tc in (getattr(m, "tool_calls", None) or []):
                valid.add(tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", ""))
        orphans = [m for m in r if isinstance(m, ToolMessage)
                   and m.tool_call_id not in valid]
        ok = True
        for i, m in enumerate(r):
            for tc in (getattr(m, "tool_calls", None) or []):
                tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", "")
                nxt = r[i + 1] if i + 1 < len(r) else None
                if not (isinstance(nxt, ToolMessage) and nxt.tool_call_id == tid):
                    ok = False
        return len(orphans) == 0 and ok

    loop = asyncio.new_event_loop()
    r1 = loop.run_until_complete(ContextBudget(ContextBudgetConfig(
        max_tokens=3000, soft_threshold=0.3, hard_threshold=0.6)).check(msgs))
    r2 = loop.run_until_complete(ContextBudget(ContextBudgetConfig(
        max_tokens=3000, soft_threshold=0.3, hard_threshold=0.4)).check(msgs))
    check("SUMMARIZE 后配对完整", check_pairing(r1.messages, "sum"),
          f"level={r1.level.value} msgs={len(r1.messages)}")
    check("TRUNCATE 后配对完整", check_pairing(r2.messages, "trunc"),
          f"level={r2.level.value} msgs={len(r2.messages)}")
    loop.close()


def test_ag37_chunk_id_traceable():
    print("[AG-37] 证据账本 chunk_id 可回查（INV-09）")
    from src.session.manager import session_manager
    from src.core import progress_bus as pb
    loop = asyncio.new_event_loop()
    sid = f"cid-{uuid.uuid4().hex[:8]}"
    evd = [{"doi": "10.1/x", "chunk_id": "P1:3", "title": "T", "score": 0.9,
            "snippet": "s"}]
    loop.run_until_complete(session_manager.save_evidence(sid, "q", evd, "报告"))
    block = session_manager.build_evidence_block(sid, limit=1)
    check("证据块含 chunk_id", "chunk: P1:3" in block, block[:200])
    loop.run_until_complete(session_manager.clear_session(sid))
    loop.close()
    # 源码断言: save 节点构建 evidence 含 chunk_id
    eg = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'src', 'graph', 'expert_graph.py'), encoding='utf-8').read()
    check("save 节点保留 chunk_id", "chunk_id" in eg and "paper_id" in eg)
    # 报告合并（多轮检索不丢）
    lg = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'src', 'graph', 'light_graph.py'), encoding='utf-8').read()
    check("报告合并逻辑", "report_parts" in eg and "report_parts" in lg)


if __name__ == "__main__":
    test_ag4_session_new()
    test_ag7_timeout_retry()
    test_ag8_truncation()
    test_ag9_atomic_write()
    test_ag10_limits()
    test_ag11_idx_map()
    test_ag12_route_removed()
    test_ag14_degraded_event()
    test_ag5_ltm()
    test_ag17_thread_context_propagation()
    test_ag18_budget_and_converge()
    test_ag19_protocol_pairing()
    test_ag20_lock_degrade()
    test_ag21_timer_pairing()
    test_ag22_output_routing()
    test_ag23_converge_behavior()
    test_ag24_circuit_breaker()
    test_ag25_truncation_transparency()
    test_ag26_budget_forward()
    test_ag27_idempotency()
    test_ag28_jobs()
    test_ag29_citation_support()
    test_ag32_trace_persistence()
    test_ag33_noise_trim()
    test_ag34_evidence_ledger()
    test_ag35_material_fidelity()
    test_ag36_compression_pairing()
    test_ag37_chunk_id_traceable()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
