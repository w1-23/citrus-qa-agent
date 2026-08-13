"""状态栏正确性单测（书 2.6: 状态栏准确率是一线生产指标，代码维护、不依赖 LLM）."""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def test_status_bar_content():
    from src.graph.expert_graph import _build_status_content

    # 首轮: 无工具调用
    s0 = _build_status_content(
        turn=1, max_turns=8, tool_call_count=0, max_tools_per_turn=2,
        unique_docs=0, used_queries=[], consecutive_failures=0,
        tool_names_called=[])
    check("首轮 TODO 未完成", "[ ] 评估需求" in s0 and "[ ] 输出最终回答" in s0)

    # 检索后: 检索项勾选
    s1 = _build_status_content(
        turn=2, max_turns=8, tool_call_count=2, max_tools_per_turn=2,
        unique_docs=12, used_queries=["salt tolerance", "citrus"], consecutive_failures=0,
        tool_names_called=["call_retrieve_agent"])
    check("检索后 TODO: 检索项勾选", "[✓] 检索文献" in s1)
    check("计数与真实一致: 工具调用 2 次", "已执行工具调用: 2 次" in s1)
    check("计数与真实一致: 文献 12 篇", "已检索去重文献: 12 篇" in s1)
    check("关键词上限 5 条", "; ".join(["salt tolerance", "citrus"]) in s1)

    # 写作后: 撰写项勾选
    s2 = _build_status_content(
        turn=3, max_turns=8, tool_call_count=4, max_tools_per_turn=2,
        unique_docs=12, used_queries=[], consecutive_failures=0,
        tool_names_called=["call_retrieve_agent", "call_write_agent"],
        budget_ratio=0.63)
    check("写作后 TODO: 撰写项勾选", "[✓] 撰写/保存内容" in s2)
    check("预算占用率入状态栏", "上下文占用: 63.0%" in s2)
    check("状态栏包在 <agent_status> 内",
          s2.strip().startswith("<agent_status>") and "</agent_status>" in s2)

    # 连续失败 ≥3 提示
    s3 = _build_status_content(
        turn=4, max_turns=8, tool_call_count=6, max_tools_per_turn=2,
        unique_docs=3, used_queries=[], consecutive_failures=3,
        tool_names_called=["call_retrieve_agent"])
    check("失败计数入状态栏", "连续工具失败: 3 次" in s3)


print()
if __name__ == "__main__":
    test_status_bar_content()
    print(f"status bar tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
