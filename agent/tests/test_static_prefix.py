"""静态前缀（阶段1）回归测试 — context.static_prefix 开关行为验证."""
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
    print("[SP-1] SystemMessage 字节级稳定")
    settings.CONTEXT_STATIC_PREFIX = True
    try:
        from src.prompts.loader import assemble_system_prompt, build_dynamic_blocks

        s1 = assemble_system_prompt(mode="expert", format_hint="fact", query="salt tolerance")
        s2 = assemble_system_prompt(mode="expert", format_hint="review", query="huanglongbing")
        s3 = assemble_system_prompt(mode="expert", format_hint=None, query=None)
        check("format_hint/query 变化不影响 SystemMessage", s1 == s2 == s3)
        check("SystemMessage 非空", len(s1) > 100, f"len={len(s1)}")

        d1 = build_dynamic_blocks(format_hint="fact", query="salt tolerance")
        d2 = build_dynamic_blocks(format_hint="review", query="huanglongbing")
        check("动态块随请求变化", d1 != d2)
        check("动态块含 format_guide 标签", "<format_guide>" in d1)
        check("动态块含 output_guide 标签", "<output_guide>" in d1)
    finally:
        settings.CONTEXT_STATIC_PREFIX = False


def test_static_prefix_legacy_behavior():
    print("[SP-2] 旧模式行为不变（灰度开关默认 off）")
    from src.prompts.loader import assemble_system_prompt, build_dynamic_blocks

    s1 = assemble_system_prompt(mode="expert", format_hint="fact", query="salt tolerance")
    s2 = assemble_system_prompt(mode="expert", format_hint="review", query="huanglongbing")
    check("旧模式 SystemMessage 仍含动态内容", s1 != s2)
    check("旧模式动态块为空", build_dynamic_blocks(format_hint="fact", query="x") == "")


def test_static_prefix_agent_prompt():
    print("[SP-3] 子 Agent 前缀稳定")
    settings.CONTEXT_STATIC_PREFIX = True
    try:
        from src.prompts.loader import assemble_agent_prompt, build_agent_extra_block

        a1 = assemble_agent_prompt("write-agent", skills=["citrus-review-writer"])
        a2 = assemble_agent_prompt("write-agent")
        check("skills 不进入子 Agent SystemMessage", a1 == a2)

        e = build_agent_extra_block(
            skills=["citrus-review-writer"], system_prompt_extra="EXTRA_INSTR")
        check("skills+extra 进独立块", "EXTRA_INSTR" in e and "写作技能" in e)
    finally:
        settings.CONTEXT_STATIC_PREFIX = False

    e2 = build_agent_extra_block(system_prompt_extra="EXTRA_INSTR")
    check("旧模式 extra 块透传（拼进 system）", e2 == "EXTRA_INSTR")


def test_static_prefix_human_message():
    print("[SP-4] 当前轮 HumanMessage 尾部追加动态块")
    settings.CONTEXT_STATIC_PREFIX = True
    try:
        from src.core.context_manager import LoadedContext, build_human_message

        ctx = LoadedContext(
            session_id="s", mode="expert", query="test query",
            format_hint="fact", search_suggestions=["a b c"],
        )
        msg = build_human_message(ctx)
        check("动态块位于 user_query 之后",
              msg.content.find("<user_query>") < msg.content.find("<format_guide>"))
    finally:
        settings.CONTEXT_STATIC_PREFIX = False


def test_supervisor_tool_schemas_single_source():
    print("[SP-5] supervisor 工具 schema 单一来源")
    from src.tools.supervisor_tools import get_supervisor_tool_schemas, get_supervisor_tool_names
    from src.graph.expert_graph import _AGENT_TOOLS

    schemas = get_supervisor_tool_schemas()
    check("expert_graph 引用单一来源", _AGENT_TOOLS is schemas)
    check("schema 顺序固定", get_supervisor_tool_schemas() == schemas)
    names = get_supervisor_tool_names()
    check("6 个 supervisor 工具", len(names) == 6, f"got {names}")
    check("含三个 call_*_agent", all(n in names for n in
                                     ("call_retrieve_agent", "call_write_agent", "call_analyze_agent")))


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
