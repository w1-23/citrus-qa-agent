"""压缩架构 v8.4（存储全量·发送裁剪）回归测试."""
import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage, ToolMessage

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


def test_checkpoint_persistence_append_only():
    print("[CP-1] checkpoint 持久化 + 轨迹 append-only")
    from src.session.manager import SessionManager
    from src.core.context_budget import ContextBudget, ContextBudgetConfig
    import tempfile, uuid
    from pathlib import Path

    sid = "cp_" + uuid.uuid4().hex[:8]
    tmp = Path(tempfile.mkdtemp()) / "sessions.db"
    sm = SessionManager()
    sm.db_path = str(tmp)
    sm._init_db_sync()
    sm._create_session_sync(sid)

    # 3 轮历史写入（模拟原始轨迹）
    loop = asyncio.new_event_loop()
    hist = build_history(3, 500)
    loop.run_until_complete(sm.save_messages(sid, hist))

    cfg = ContextBudgetConfig(max_tokens=512000, soft_threshold=0.0, hard_threshold=0.93)
    budget = ContextBudget(cfg)
    budget.set_compact_fn(None)

    # 直接驱动 checkpoint（模拟 ContextManager 压缩后的持久化）
    sm.set_checkpoint(sid, msg_id=4, summary="SUMMARY-TEXT")
    cp = sm.get_checkpoint(sid)
    check("checkpoint 可读回", cp["msg_id"] == 4 and cp["summary"] == "SUMMARY-TEXT", str(cp))

    raw = sm._get_messages_sync(sid)
    check("原始轨迹未被删除（append-only）", len(raw) == 6, f"got {len(raw)}")

    # 追加新消息后 checkpoint 仍有效
    loop.run_until_complete(sm.save_messages(sid, build_history(1, 100)))
    raw2 = sm._get_messages_sync(sid)
    check("追加后轨迹增长", len(raw2) == 8, f"got {len(raw2)}")

    sm._clear_session_sync(sid)
    cp2 = sm.get_checkpoint(sid)
    check("清会话重置 checkpoint", cp2["msg_id"] == 0 and cp2["summary"] == "", str(cp2))
    loop.close()


def test_batch_compression_and_protection():
    print("[CP-2] 批量压缩 + 保护名单")
    from src.core.context_budget import ContextBudget, ContextBudgetConfig, ContextBudgetLevel

    cfg = ContextBudgetConfig(
        max_tokens=512000, soft_threshold=0.75, hard_threshold=0.93,
        protect_recent_turns=3, target_ratio=0.50)
    budget = ContextBudget(cfg)
    budget.set_compact_fn(None)   # 规则式摘要（无 LLM 依赖）

    msgs = build_history(110)   # ~397K → 77% → SUMMARIZE
    r = asyncio.new_event_loop().run_until_complete(budget.check(msgs, query="黄龙病"))
    check("110 轮批量压缩触发", r.level == ContextBudgetLevel.SUMMARIZE, f"{r.level.value}")
    check("视图 = 锚点 + 摘要 + 最近3轮", len(r.messages) <= 1 + 1 + 6,
          f"got {len(r.messages)}")
    check("摘要嵌入 <conversation_summary>",
          any("<conversation_summary>" in str(m.content) for m in r.messages))
    last_three_turns_flat = msgs[-6:]
    check("最近 3 轮原样保留（保护名单）",
          all(any(getattr(m, "content", "") == getattr(k, "content", "")
                  for m in r.messages[-6:]) for k in last_three_turns_flat))
    check("checkpoint cutoff 指向被压缩消息",
          r.cutoff_id is not None and r.cutoff_id == len(msgs) - 6 - 1 + 1 - 1,
          f"cutoff_id={r.cutoff_id}")

    # 增量: checkpoint 视图上再追加一轮 → 应基于摘要继续，而非全量重压
    prefix = r.messages + [HumanMessage(content="追问: " + "字" * 500),
                           AIMessage(content="回答: " + "字" * 500)]
    ids = list(range(len(prefix)))
    r2 = asyncio.new_event_loop().run_until_complete(
        budget.check(prefix, query="追问", ids=ids))
    check("checkpoint 视图未超软阈值 → NORMAL（防摘要套摘要）",
          r2.level == ContextBudgetLevel.NORMAL, f"{r2.level.value}")


def test_noise_trim_and_identifiers():
    print("[CP-3] 噪声删除 + 标识符保留")
    from src.core.context_budget import _keep_identifiers

    out = _keep_identifiers("过程性内容\nDOI: 10.1000/xyz123\nevidence_id: ev-9\n" + "噪声" * 5000, max_chars=200)
    check("截断透明标记", "[TRUNCATED" in out)
    check("DOI 保留", "10.1000/xyz123" in out)
    check("evidence_id 保留", "ev-9" in out)


def test_circuit_breaker():
    print("[CP-4] 压缩熔断器（连续失败 ≥3 → 规则式）")
    from src.core.context_budget import (
        ContextBudget, ContextBudgetConfig, _compaction_failures,
        _reset_compaction_failures, _record_compaction_failure)

    _reset_compaction_failures("cb_test")
    async def failing_fn(msgs, query=""):
        raise RuntimeError("boom")

    budget = ContextBudget(ContextBudgetConfig(max_tokens=1000, soft_threshold=0.0,
                                               hard_threshold=0.99))
    budget.set_compact_fn(failing_fn)
    msgs = build_history(5, 100)

    async def run():
        r1 = await budget.check(msgs, query="q", session_id="cb_test")
        r2 = await budget.check(msgs, query="q", session_id="cb_test")
        r3 = await budget.check(msgs, query="q", session_id="cb_test")
        return r1, r2, r3

    r1, r2, r3 = asyncio.new_event_loop().run_until_complete(run())
    check("前 2 次失败降级为规则式摘要", bool(r1.summary) and bool(r2.summary))
    check("熔断计数达 3", _compaction_failures.get("cb_test", 0) == 3,
          f"got {_compaction_failures.get('cb_test', 0)}")
    _reset_compaction_failures("cb_test")


print()
print(f"compaction v2 tests: {len(passed)} passed, {len(failed)} failed")
if failed:
    print("FAILED:", failed)
    sys.exit(1)
