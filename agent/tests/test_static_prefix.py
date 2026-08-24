"""静态前缀（阶段1）回归测试 — context.static_prefix 开关行为验证.

v9.0 起提示词为"启动时固定拼接"架构：无论开关如何，system prompt 都是
字节级稳定的固定字符串（不再按 format_hint/query 动态注入格式模板）。
本测试改为验证固定语义。
"""
import sys
import os

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


def test_static_prefix_system_stability():
    print("[SP-1] SystemMessage 字节级稳定（v9.0 固定提示词）")
    # v8.6: 保存并还原原始开关值（此前 finally 硬编码 False，污染后续测试全局）
    _orig = settings.CONTEXT_STATIC_PREFIX
    settings.CONTEXT_STATIC_PREFIX = True
    try:
        from src.prompts.loader import assemble_system_prompt, build_dynamic_blocks

        s1 = assemble_system_prompt(mode="expert", format_hint="fact", query="salt tolerance")
        s2 = assemble_system_prompt(mode="expert", format_hint="review", query="huanglongbing")
        s3 = assemble_system_prompt(mode="expert", format_hint=None, query=None)
        check("format_hint/query 变化不影响 SystemMessage", s1 == s2 == s3)
        check("SystemMessage 非空", len(s1) > 100, f"len={len(s1)}")
        # v9.0: 格式模板已固定进 Supervisor system prompt——动态块恒为空
        d1 = build_dynamic_blocks(format_hint="fact", query="salt tolerance")
        d2 = build_dynamic_blocks(format_hint="review", query="huanglongbing")
        check("动态块恒为空（模板固定在前缀内）", d1 == d2 == "")
        # 固定前缀内已包含全部格式模板（模型自行选择，不再动态注入）
        check("Supervisor 前缀含格式模板",
              "输出格式：事实查询" in s1 and "输出格式：对比查询" in s1)
    finally:
        settings.CONTEXT_STATIC_PREFIX = _orig


def test_static_prefix_legacy_behavior():
    print("[SP-2] v9.0 固定提示词不依赖开关（灰度开关不再影响内容）")
    from src.prompts.loader import assemble_system_prompt, build_dynamic_blocks

    _orig = settings.CONTEXT_STATIC_PREFIX
    settings.CONTEXT_STATIC_PREFIX = False
    try:
        s1 = assemble_system_prompt(mode="expert", format_hint="fact", query="salt tolerance")
        s2 = assemble_system_prompt(mode="expert", format_hint="review", query="huanglongbing")
        check("开关关闭时 system prompt 仍固定（s1==s2）", s1 == s2)
        check("开关关闭时动态块仍为空", build_dynamic_blocks(format_hint="fact", query="x") == "")
        sl = assemble_system_prompt(mode="light", format_hint="review", query="x")
        check("light 与 expert 使用不同固定提示词", sl != s1 and "轻量模式" in sl)
    finally:
        settings.CONTEXT_STATIC_PREFIX = _orig


def test_static_prefix_agent_prompt():
    print("[SP-3] 子 Agent 前缀稳定")
    _orig = settings.CONTEXT_STATIC_PREFIX
    settings.CONTEXT_STATIC_PREFIX = True
    try:
        from src.prompts.loader import assemble_agent_prompt, build_agent_extra_block

        a1 = assemble_agent_prompt("write-agent", skills=["citrus-review-writer"])
        a2 = assemble_agent_prompt("write-agent")
        check("skills 不进入子 Agent SystemMessage", a1 == a2)

        # v8.10d: skills/*.md 按名读取——不存在时优雅降级（只透传 extra）；
        # 生产路径 run_agent 不传 skills（写作技能经 skill_prompt/skill_map 注入）
        e = build_agent_extra_block(
            skills=["citrus-review-writer"], system_prompt_extra="EXTRA_INSTR")
        check("skills 文件缺失时优雅降级（extra 仍透传）", "EXTRA_INSTR" in e)
    finally:
        settings.CONTEXT_STATIC_PREFIX = _orig

    e2 = build_agent_extra_block(system_prompt_extra="EXTRA_INSTR")
    check("extra 块透传", e2 == "EXTRA_INSTR")


def test_static_prefix_human_message():
    print("[SP-4] 当前轮 HumanMessage 只含用户问句（无动态格式块）")
    _orig = settings.CONTEXT_STATIC_PREFIX
    settings.CONTEXT_STATIC_PREFIX = True
    try:
        from src.core.context_manager import LoadedContext, build_human_message

        ctx = LoadedContext(
            session_id="s", mode="expert", query="test query",
            format_hint="fact", search_suggestions=["a b c"],
        )
        msg = build_human_message(ctx)
        check("包含 user_query 原文",
              "<user_query>" in msg.content and "test query" in msg.content)
        check("无格式指南动态块（v9.0 已固定进前缀）",
              "<format_guide>" not in msg.content)
    finally:
        settings.CONTEXT_STATIC_PREFIX = _orig


def test_supervisor_tool_schemas_single_source():
    print("[SP-5] supervisor 工具 schema 单一来源")
    from src.tools.supervisor_tools import get_supervisor_tool_schemas, get_supervisor_tool_names
    from src.graph.expert_graph import _AGENT_TOOLS

    schemas = get_supervisor_tool_schemas()
    check("expert_graph 引用单一来源", _AGENT_TOOLS is schemas)
    check("schema 顺序固定", get_supervisor_tool_schemas() == schemas)
    names = get_supervisor_tool_names()
    check("6 个 supervisor 工具", len(names) == 6, f"got {names}")
    check("含 call_search_both + 另两个 call_*_agent", all(n in names for n in
                                     ("call_search_both", "call_write_agent", "call_analyze_agent")))
    check("call_retrieve_agent 已移除", "call_retrieve_agent" not in names)


def test_supervisor_tool_schemas_single_source():
    print("[SP-5] supervisor 工具 schema 单一来源")
    from src.tools.supervisor_tools import get_supervisor_tool_schemas, get_supervisor_tool_names
    from src.graph.expert_graph import _AGENT_TOOLS

    schemas = get_supervisor_tool_schemas()
    check("expert_graph 引用单一来源", _AGENT_TOOLS is schemas)
    check("schema 顺序固定", get_supervisor_tool_schemas() == schemas)
    names = get_supervisor_tool_names()
    check("6 个 supervisor 工具", len(names) == 6, f"got {names}")
    check("含 call_search_both + 另两个 call_*_agent", all(n in names for n in
                                     ("call_search_both", "call_write_agent", "call_analyze_agent")))
    check("call_retrieve_agent 已移除", "call_retrieve_agent" not in names)


print()
if __name__ == "__main__":
    test_static_prefix_system_stability()
    test_static_prefix_legacy_behavior()
    test_static_prefix_agent_prompt()
    test_static_prefix_human_message()
    test_supervisor_tool_schemas_single_source()
    print(f"static_prefix tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
