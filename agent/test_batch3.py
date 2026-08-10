# -*- coding: utf-8 -*-
"""Batch-3 验证：AG-15 估算精度 / AG-16 卡片文案 / 回归不破坏"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.context_budget import _estimate_chars_tokens

passed, failed = [], []


def check(name, cond, detail=""):
    (passed if cond else failed).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_ag15_estimate():
    print("[AG-15] 混合语言估算精度")
    # 基准: DeepSeek 实测近似 — 中文 1 字≈1.0-1.5 token, 英文 1 词≈1.3 token(≈4.6字/token)
    cases = [
        ("纯中文 500 字", "柑橘黄龙病病原致病机制抗病基因代谢通路调控" * 20, 480, 600, 720),
        ("纯英文", "citrus huanglongbing pathogen resistance gene " * 30, 186, 248, 372),
        ("中英混合", "柑橘黄龙病病原 " * 30 + "citrus HLB pathogen " * 15, 195, 285, 420),
    ]
    for name, text, lo, mid, hi in cases:
        est = _estimate_chars_tokens(text)
        ok = lo <= est <= hi
        check(f"估算在基准区间内 [{lo},{hi}]", ok, f"est={est}")

    # 旧系数对比: 旧 len/1.5 对纯英文高估 (500 字符 → 333), 新系数 ≈125
    en = "a" * 500
    old = 500 // 1.5
    new = _estimate_chars_tokens(en)
    check("英文不再高估 (新 < 旧)", new < old, f"old={old} new={new}")
    check("英文 ≈ 4字符/token", 100 <= new <= 150, f"new={new}")

    zh = "字" * 500
    check("中文 ≈ 1.2/字", 550 <= _estimate_chars_tokens(zh) <= 650, f"new={_estimate_chars_tokens(zh)}")
    check("空串返回 0", _estimate_chars_tokens("") == 0)


def test_ag16_cards():
    print("[AG-16] 策略卡片文案")
    src = open(os.path.join(os.path.dirname(__file__), 'src', 'core', 'strategy_cards.py'),
               encoding='utf-8').read()
    check("无 web_search/multi_search 残留", "web_search" not in src and "multi_search" not in src)
    check("含 academic_search 指引", "academic_search" in src)


def test_n2_tools():
    print("[N2] 工具 description 评审")
    src = open(os.path.join(os.path.dirname(__file__), 'src', 'graph', 'expert_graph.py'),
               encoding='utf-8').read()
    check("retrieve 含同义词重试指引", "synonyms" in src)
    check("write 含成文/素材说明", "finished document" in src and "raw material" in src)


def test_regression_budget():
    print("[回归] 预算触发不破坏 (AG-3)")
    from src.core.context_budget import ContextBudget, ContextBudgetConfig, ContextBudgetLevel
    import asyncio
    from langchain_core.messages import HumanMessage, AIMessage

    def build(n, per_msg=30):
        msgs = []
        for i in range(n):
            msgs.append(HumanMessage(content=f"问{i}:" + "柑橘黄龙病机制基因调控" * per_msg))
            msgs.append(AIMessage(content=f"答{i}:" + "代谢通路与抗病基因分析" * per_msg))
        return msgs

    cfg = ContextBudgetConfig(max_tokens=1000000, soft_threshold=0.60, hard_threshold=0.93)
    budget = ContextBudget(cfg)
    loop = asyncio.new_event_loop()
    r = loop.run_until_complete(budget.check(build(20)))
    check("20 轮 NORMAL", r.level == ContextBudgetLevel.NORMAL)
    # 每消息 30×9≈270 字 ×1.2 ≈ 324 token, 500 轮 ×2 条 = 32.4 万 → ratio 32% < 60% → NORMAL
    r_med = loop.run_until_complete(budget.check(build(500)))
    check("500 轮仍 NORMAL（新估算更准，触发点后移）", r_med.level == ContextBudgetLevel.NORMAL)
    # 1000 轮 ×2 条 ×324 ≈ 64.8 万 > 60 万 → SUMMARIZE
    r2 = loop.run_until_complete(budget.check(build(1000)))
    check("1000 轮 SUMMARIZE", r2.level == ContextBudgetLevel.SUMMARIZE, f"level={r2.level.value}")
    loop.close()


if __name__ == "__main__":
    test_ag15_estimate()
    test_ag16_cards()
    test_n2_tools()
    test_regression_budget()
    print(f"\n结果: {len(passed)} passed / {len(failed)} failed")
    if failed:
        print("失败项:", failed)
        sys.exit(1)
