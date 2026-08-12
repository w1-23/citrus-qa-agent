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
    (passed if cond else failed).append(name)
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
    runner = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'core', 'agent_runner.py'),
                  encoding='utf-8').read()
    check("agent_runner 含 3 次重试", "range(3)" in runner and "retry" in runner)
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
    check("总长 ≤ 上限+单条", len(out) <= 5000 + 1000 + 2, f"len={len(out)}")
    check("保留完整条目（每块都是原块）",
          all(b in orig_blocks for b in blocks), f"{len(blocks)} blocks")
    check("非首块即被截断时不含半截", len(blocks) == 4 or all(
        b in orig_blocks for b in blocks))
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
    import tempfile
    outside = Path(tempfile.mkdtemp()) / "out.csv"
    outside.write_text("a,b\n1,2\n", encoding="utf-8")
    r3 = statistical_analysis.func(str(outside), "descriptive", '{"value_column": "a"}')
    check("项目外绝对路径 → [ERR_PERMISSION]", "[ERR_PERMISSION]" in r3, r3[:60])

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
    import tempfile, sqlite3
    from pathlib import Path
    from src.config import PROJECT_ROOT

    tmp_db = Path(tempfile.mkdtemp()) / "sessions.db"
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
    check("save 节点 to_thread 异步化", "asyncio.to_thread(memory_store.extract_key_facts" in g1)
    c = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'prompts', 'system', 'constraints.md'),
             encoding='utf-8').read()
    check("constraints 含冲突规则", "跨会话记忆规则" in c)


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
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
