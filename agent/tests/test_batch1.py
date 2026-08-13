"""Batch-1 修复验证脚本（无外部依赖，仅验证逻辑正确性）."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import settings
from src.core.context_budget import ContextBudget, ContextBudgetConfig, ContextBudgetLevel
from langchain_core.messages import HumanMessage, AIMessage

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def build_history(n_turns, tokens_per_msg=1500):
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"问{i}: " + "字" * tokens_per_msg))
        msgs.append(AIMessage(content=f"答{i}: " + "字" * tokens_per_msg))
    return msgs


def test_ag3_config_single_source():
    print("[AG-3] 配置单向化")
    check("settings 读取 max_tokens=512K (v8.4 发送视图)",
          settings.CONTEXT_BUDGET_MAX_TOKENS == 512000,
          f"got {settings.CONTEXT_BUDGET_MAX_TOKENS}")
    check("settings 读取 soft=0.75", abs(settings.CONTEXT_BUDGET_SOFT_THRESHOLD - 0.75) < 1e-6,
          f"got {settings.CONTEXT_BUDGET_SOFT_THRESHOLD}")
    check("settings 读取 hard=0.93", abs(settings.CONTEXT_BUDGET_HARD_THRESHOLD - 0.93) < 1e-6,
          f"got {settings.CONTEXT_BUDGET_HARD_THRESHOLD}")
    import yaml
    cfg = yaml.safe_load(open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'config.yaml'), encoding='utf-8'))
    cb = cfg['context_budget']
    check("config.yaml soft=0.75", abs(cb['soft_threshold'] - 0.75) < 1e-6)
    check("config.yaml hard=0.93", abs(cb['hard_threshold'] - 0.93) < 1e-6)
    check("circuit_breaker 重复阈值已删",
          'context_soft_threshold' not in cfg.get('circuit_breaker', {}),
          f"remaining keys: {list(cfg.get('circuit_breaker', {}).keys())}")


def test_ag3_early_compaction():
    print("[AG-3] 早期压缩触发 (512K 视图)")
    cfg = ContextBudgetConfig(max_tokens=512000, soft_threshold=0.75, hard_threshold=0.93)
    budget = ContextBudget(cfg)

    short = build_history(20)   # ~36K tokens → ratio ~7% → NORMAL
    r = asyncio.get_event_loop().run_until_complete(budget.check(short))
    check("20 轮不触发", r.level == ContextBudgetLevel.NORMAL, f"level={r.level.value}")

    long = build_history(110)   # 110 轮 ≈ 397K tokens → ratio ~77% → SUMMARIZE
    r2 = asyncio.get_event_loop().run_until_complete(budget.check(long))
    check("110 轮触发 SUMMARIZE", r2.level == ContextBudgetLevel.SUMMARIZE,
          f"level={r2.level.value}, tokens={budget.estimate_tokens(long)}")
    check("压缩后消息数显著下降", len(r2.messages) < len(long) * 0.35,
          f"{len(r2.messages)} < {len(long)*0.35:.0f}")

    huge = build_history(500)   # 500 轮 ≈ 1.8M tokens → ratio > 93% → TRUNCATE
    r3 = asyncio.get_event_loop().run_until_complete(budget.check(huge))
    check("500 轮触发 TRUNCATE", r3.level == ContextBudgetLevel.TRUNCATE,
          f"level={r3.level.value}")


def test_ag6_persistence_schema():
    print("[AG-6] replace_history 持久化接口")
    from src.session.manager import SessionManager
    import tempfile, sqlite3, uuid
    from pathlib import Path

    sid = "test_" + uuid.uuid4().hex[:8]
    tmp = Path(tempfile.mkdtemp()) / "sessions.db"
    sm = SessionManager()
    sm.db_path = str(tmp)  # 指向临时库
    sm._init_db_sync()

    loop = asyncio.new_event_loop()
    loop.run_until_complete(sm.save_messages(sid, build_history(5, 100)))
    n_before = sqlite3.connect(str(tmp)).execute(
        "SELECT COUNT(*) FROM messages WHERE session_id=?", (sid,)).fetchone()[0]
    check("先保存 10 条", n_before == 10, f"n={n_before}")

    compacted = build_history(2, 100)  # 模拟压缩结果: 4 条
    loop.run_until_complete(sm.replace_history(sid, compacted))
    rows = sqlite3.connect(str(tmp)).execute(
        "SELECT msg_type, content FROM messages WHERE session_id=? ORDER BY id", (sid,)).fetchall()
    check("替换后剩 4 条", len(rows) == 4, f"n={len(rows)}")
    check("替换内容为新消息", rows[0][1].startswith("问0:"), rows[0][1][:20])

    loop.run_until_complete(sm.clear_session(sid))
    loop.close()


def test_ag2_file_semantics():
    print("[AG-2] write=覆盖 / append=追加 语义")
    import uuid
    from pathlib import Path
    from src.tools.file_ops import write_local_file
    from src.config import PROJECT_ROOT

    work_dir = PROJECT_ROOT / "workspace" / "output"
    target = work_dir / f"test_ag2_{uuid.uuid4().hex[:6]}.md"

    r1 = write_local_file.func(target.name, "AAA", "write")
    check("首次 write 创建", "Success" in r1 and "write" in r1, r1[:60])
    r2 = write_local_file.func(target.name, "BBB", "write")
    content = target.read_text(encoding="utf-8")
    check("再次 write 覆盖（不叠加）", content == "BBB", repr(content))
    r3 = write_local_file.func(target.name, "CCC", "append")
    content = target.read_text(encoding="utf-8")
    check("append 追加", content == "BBB\n\nCCC", repr(content))
    r4 = write_local_file.func("E:/fake/outside.md", "X", "write")
    check("越权路径拒绝", "Access denied" in r4, r4[:40])
    if target.exists():
        target.unlink()


def test_ag1_truncation_block():
    print("[AG-1] 截断分支逻辑（无 task 引用）")
    import ast
    src = open(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src', 'graph', 'expert_graph.py'),
               encoding='utf-8').read()
    check("源码不再含 task.get('output_path')",
          "task.get('output_path'" not in src)
    check("源码含截断 warning 日志", "truncated" in src and "logger.warning" in src)
    ast.parse(src)
    check("语法解析通过", True)


if __name__ == "__main__":
    test_ag3_config_single_source()
    test_ag3_early_compaction()
    test_ag6_persistence_schema()
    test_ag2_file_semantics()
    test_ag1_truncation_block()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
