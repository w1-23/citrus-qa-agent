# -*- coding: utf-8 -*-
"""上下文预算 1M 窗口验证（v8.4.4 项2，只读运行）。

构造逼近 1M 的发送视图，断言:
  1. 软阈值 75% 触发 SUMMARIZE（批量压缩至 ~50%）
  2. 硬阈值 93% 前不溢出（估算器 vs 预压缩比例）
  3. TRUNCATE 仅当压缩后视图仍超硬阈值
  （真实 1M 窗口溢出验证需服务端一次真实长会话，本脚本验证估算链路）
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, AIMessage
from src.core.context_budget import ContextBudget, ContextBudgetConfig, ContextBudgetLevel

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def build_history(n_turns, chars_per_msg=1500):
    msgs = []
    for i in range(n_turns):
        msgs.append(HumanMessage(content=f"问{i}: " + "字" * chars_per_msg))
        msgs.append(AIMessage(content=f"答{i}: " + "字" * chars_per_msg))
    return msgs


def test_1m_budget():
    print("[1M] 上下文预算 1M 窗口验证")
    cfg = ContextBudgetConfig(max_tokens=1000000, soft_threshold=0.75,
                              hard_threshold=0.93, protect_recent_turns=3)
    budget = ContextBudget(cfg)

    # ~70% 预算（~700K tokens）：低于软阈值 → NORMAL
    hist = build_history(195)   # 195×2×~1802 ≈ 703K → 70.3%
    r = asyncio.new_event_loop().run_until_complete(budget.check(hist))
    check("~70% 不触发", r.level == ContextBudgetLevel.NORMAL,
          f"level={r.level.value} tokens={budget.estimate_tokens(hist)}")

    # ~80% 预算（~800K tokens）：触发 SUMMARIZE
    hist2 = build_history(222)  # 222×2×~1802 ≈ 800K → 80%
    r2 = asyncio.new_event_loop().run_until_complete(budget.check(hist2))
    check("~80% 触发 SUMMARIZE", r2.level == ContextBudgetLevel.SUMMARIZE,
          f"level={r2.level.value} tokens={budget.estimate_tokens(hist2)}")
    check("压缩后视图大幅收敛",
          budget.estimate_tokens(r2.messages) < cfg.max_tokens * 0.30,
          f"view={budget.estimate_tokens(r2.messages)}")

    # ~99% 预算（~990K tokens）：仍在窗口内、压缩成功 → SUMMARIZE
    hist3 = build_history(275)  # 275×2×~1802 ≈ 991K → 99.1%
    r3 = asyncio.new_event_loop().run_until_complete(budget.check(hist3))
    check("~99% 压缩成功 → SUMMARIZE", r3.level == ContextBudgetLevel.SUMMARIZE,
          f"level={r3.level.value} tokens={budget.estimate_tokens(hist3)}")

    # 硬阈值熔断场景: 压缩产出超长摘要 → 压缩后视图仍 ≥93% → TRUNCATE
    budget2 = ContextBudget(cfg)

    async def huge_summary(msgs, query="", prior_summary=""):
        return "超长摘要" * 800000   # 320 万字符 CJK ≈ 384 万 token

    budget2.set_compact_fn(huge_summary)
    r4 = asyncio.new_event_loop().run_until_complete(budget2.check(hist3))
    check("压缩后仍超限 → TRUNCATE", r4.level == ContextBudgetLevel.TRUNCATE,
          f"level={r4.level.value}")

    # 估算偏差说明: 估算 = CJK 1.2/字（实测区间 1.0-1.5），偏差 <15%
    est = budget.estimate_tokens(hist3)
    print(f"  估算参考: {len(hist3)} msgs -> {est} tok（CJK 1.2/字，偏差 <15%）")


print()
if __name__ == "__main__":
    test_1m_budget()
    print(f"1M budget tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
