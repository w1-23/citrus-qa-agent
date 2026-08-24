"""Prompt Loader — 启动时固定拼接的 system prompt 装配入口.

v9.0（用户决策改造，2026-09）: 固定 system prompt 架构——"检索 + 回答" 业务流中
所有角色规则在一轮流程内都会被用到，因此不再按需动态加载，改为启动时一次性拼接：

  - source/ 20 份源文件是唯一维护入口（改哪块只动哪块）。
  - builds/ 由 build_fixed_prompts() 从 source/ 按角色固定映射拼接生成并落盘，
    之后每次请求使用同一个字符串，不再变化（KV cache 前缀字节级复用）。
  - 每个角色（Supervisor / Retrieve / Write，以及 Lite / Analyze 兼容角色）
    拥有独立的固定 system prompt，互不影响。
  - 输出格式模板全部固定于 Supervisor 提示词中（09~15），由模型按问题类型
    自行选择，不再按 format_hint / query 动态插入模板。

兼容层（对外签名不变，内部改走固定提示词）:
  - assemble_system_prompt(mode=...) → supervisor / lite 固定提示词
  - assemble_agent_prompt(agent_name) → retrieve / write / analyze 固定提示词
  - build_dynamic_blocks(...) → 恒为空串（无动态格式块）
  - build_agent_extra_block(...) → 不变（skills/系统附加指令仍追加到首条
    HumanMessage 尾部，属于"追加"而非"修改前缀"，不破坏固定前缀缓存）
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

# ─────────────────────────────────────────────────────────────
# 角色 → 固定源文件映射（唯一维护入口，改这里即改拼接结果）
# ─────────────────────────────────────────────────────────────
ROLE_SOURCE_FILES: dict[str, list[str]] = {
    # Supervisor（主控 Agent：路由 + 证据仲裁 + 最终回答，含全部输出格式模板）
    "supervisor": [
        "01_global_role_domain.md",
        "02_global_data_fidelity_citation.md",
        "03_supervisor_routing_fusion.md",
        "09_format_fact.md",
        "10_format_mechanism.md",
        "11_format_compare.md",
        "12_format_review.md",
        "13_format_experiment.md",
        "14_format_task.md",
        "15_format_default.md",
        "16_tool_usage_file_rules.md",
        "17_data_source_boundaries.md",
        "18_evidence_arbitration_citation.md",
        "20_terminology_domain.md",
    ],
    # Retrieve-Agent（检索子代理：只检索 + 一句话总结论，无格式模板）
    "retrieve": [
        "01_global_role_domain.md",
        "02_global_data_fidelity_citation.md",
        "04_retrieve_agent_search.md",
        "17_data_source_boundaries.md",
        "20_terminology_domain.md",
    ],
    # Write-Agent（写作子代理：Plan-Execute 长文生成，不负责检索与证据仲裁）
    "write": [
        "01_global_role_domain.md",
        "02_global_data_fidelity_citation.md",
        "05_academic_writing_common.md",
        "06_review_planner.md",
        "07_review_chapter_writer.md",
        "20_terminology_domain.md",
    ],
    # 轻量模式（01 + 02 + 19 + 20，固定）
    "lite": [
        "01_global_role_domain.md",
        "02_global_data_fidelity_citation.md",
        "19_lite_mode.md",
        "20_terminology_domain.md",
    ],
    # analyze-agent 兼容角色（保留位：01 + 02 + 08 + 20，固定）
    "analyze": [
        "01_global_role_domain.md",
        "02_global_data_fidelity_citation.md",
        "08_data_analysis_experiment.md",
        "20_terminology_domain.md",
    ],
}

# 角色 → builds/ 中构建文件名
BUILD_FILE_NAMES: dict[str, str] = {
    "supervisor": "supervisor_system.md",
    "retrieve": "retrieve_agent_system.md",
    "write": "write_agent_system.md",
    "lite": "lite_system.md",
    "analyze": "analyze_agent_system.md",
}

# 子 Agent 名 → 角色（assemble_agent_prompt 兼容层）
AGENT_ROLE_MAP: dict[str, str] = {
    "retrieve-agent": "retrieve",
    "write-agent": "write",
    "analyze-agent": "analyze",
}

# 兼容常量（供 snapshot/测试消费，含义同旧版）
VALID_AGENTS = set(AGENT_ROLE_MAP.keys())
VALID_FORMATS = {
    "fact", "compare", "method", "design", "review", "task", "fallback",
}

PROMPT_DIR = Path(__file__).resolve().parent


def _read_prompt(relative_path: str) -> str:
    path = PROMPT_DIR / relative_path
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception:
        return ""


@lru_cache(maxsize=256)
def _read_prompt_cached(relative_path: str) -> str:
    return _read_prompt(relative_path)


def _join_parts(parts: Iterable[str]) -> str:
    """固定拼接：源文件内容逐字保留，以 \n\n 分隔（与源文件间空行一致）。"""
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n\n".join(cleaned)


# ─────────────────────────────────────────────────────────────
# 固定 system prompt：构建 / 缓存 / 读取
# ─────────────────────────────────────────────────────────────

_FIXED_PROMPTS: dict[str, str] | None = None


def _concat_source_files(filenames: list[str]) -> str:
    parts = [_read_prompt_cached(f"source/{fn}") for fn in filenames]
    return _join_parts(parts)


def build_fixed_prompts() -> dict[str, str]:
    """启动时执行一次：按角色固定映射拼接完整 system prompt 并写入 builds/。

    返回 {role: prompt}；随后每次请求复用缓存字符串，不再重新拼接。
    """
    out: dict[str, str] = {}
    for role, filenames in ROLE_SOURCE_FILES.items():
        out[role] = _concat_source_files(filenames)

    # 落盘 builds/（best-effort：只读环境失败不影响内存缓存）
    try:
        builds_dir = PROMPT_DIR / "builds"
        builds_dir.mkdir(parents=True, exist_ok=True)
        for role, prompt in out.items():
            target = builds_dir / BUILD_FILE_NAMES[role]
            try:
                if target.exists() and target.read_text(encoding="utf-8").strip() == prompt:
                    continue  # 已存在且内容一致 → 跳过写入（幂等）
            except Exception:
                pass
            target.write_text(prompt + "\n", encoding="utf-8")
    except Exception:
        pass
    return out


def ensure_fixed_prompts(reload: bool = False) -> dict[str, str]:
    """进程级缓存：首次调用构建一次，之后保持不变（除非显式 reload）。"""
    global _FIXED_PROMPTS
    if _FIXED_PROMPTS is None or reload:
        _FIXED_PROMPTS = build_fixed_prompts()
    return _FIXED_PROMPTS


def get_system_prompt(role: str = "supervisor") -> str:
    """按角色取固定 system prompt（supervisor/retrieve/write/lite/analyze）。"""
    return ensure_fixed_prompts().get(role, "")


# ─────────────────────────────────────────────────────────────
# 兼容函数（对外签名不变，内部改走固定提示词）
# ─────────────────────────────────────────────────────────────

def assemble_system_prompt(
    *,
    mode: str,
    format_hint: str | None = None,
    query: str | None = None,
    include_strategy_cards: bool = True,
) -> str:
    """主 Agent system prompt——v9.0 起为固定字符串。

    format_hint/query/include_strategy_cards 参数保留仅作签名兼容，不再影响
    内容：所有格式模板固定于 Supervisor 提示词中，KV cache 前缀字节级稳定。
    """
    mode_normalized = (mode or "expert").strip().lower()
    role = "lite" if mode_normalized == "light" else "supervisor"
    return get_system_prompt(role)


def build_dynamic_blocks(
    *,
    format_hint: str | None = None,
    query: str | None = None,
    include_strategy_cards: bool = True,
) -> str:
    """v9.0 起恒为空串：格式模板已固定进 Supervisor system prompt，
    不再随请求动态注入（动态注入会改变每请求前缀，破坏 KV cache 复用）。"""
    return ""


def assemble_agent_prompt(
    agent_name: str,
    *,
    skills: list[str] | None = None,
    task_type: str | None = None,
) -> str:
    """子 Agent system prompt——按 agent_name 返回对应角色的固定字符串。

    skills/task_type 参数保留仅作签名兼容（v8.4.3 起即被忽略），动态内容
    经 build_agent_extra_block 追加到首条 HumanMessage，不动固定前缀。
    """
    agent_normalized = (agent_name or "").strip().lower()
    if agent_normalized not in VALID_AGENTS:
        raise ValueError(
            f"Unknown agent name: {agent_name}. Valid: {sorted(VALID_AGENTS)}"
        )
    return get_system_prompt(AGENT_ROLE_MAP[agent_normalized])


def build_agent_extra_block(
    *,
    skills: list[str] | None = None,
    task_type: str | None = None,
    system_prompt_extra: str = "",
) -> str:
    """子 Agent 动态指令块——不变（v8.4.3）。

    由 agent_runner 追加到首条 HumanMessage（<instructions> 块）。
    追加语义 = 新消息 append，SystemMessage 前缀字节级稳定，KV/Prompt Cache
    不受 skill 加载影响。
    """
    blocks: list[str] = []
    if system_prompt_extra:
        blocks.append(system_prompt_extra)
    if skills:
        skill_parts: list[str] = []
        for skill in skills:
            skill_content = _read_prompt_cached(f"skills/{skill}.md")
            if skill_content:
                skill_parts.append(skill_content)
        if skill_parts:
            blocks.append("## 写作技能\n\n" + "\n\n".join(skill_parts))
    if task_type:
        task_prompt = _read_prompt_cached(f"strategies/planning/{task_type}.md")
        if task_prompt:
            blocks.append(f"## 任务策略\n\n{task_prompt}")
    return _join_parts(blocks)


def get_pipeline_prompt(kind: str) -> str:
    """写作流水线（Write-Pipeline）提示词：Plan 阶段读 source/06，章节撰写读 source/07。

    kind: "plan" → 06_review_planner.md；"section" → 07_review_chapter_writer.md
    """
    filename = {
        "plan": "06_review_planner.md",
        "section": "07_review_chapter_writer.md",
    }.get(kind)
    if not filename:
        return ""
    return _read_prompt_cached(f"source/{filename}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="按角色固定拼接 system prompt 并写入 builds/")
    ap.add_argument("--list", action="store_true", help="只打印各角色拼接长度，不写盘")
    args = ap.parse_args()

    # 测试桩：允许关闭写盘（--list 模拟）
    if args.list:
        prompts = {}
        for role, filenames in ROLE_SOURCE_FILES.items():
            prompts[role] = _concat_source_files(filenames)
            print(f"{role:10s} {len(prompts[role]):6d} chars  "
                  f"({len(filenames)} files)")
        for role, prompt in prompts.items():
            heads = [ln for ln in prompt.splitlines() if ln.startswith("#")]
            print(f"--- {role} headings ---")
            for h in heads:
                print("  " + h)
    else:
        built = build_fixed_prompts()
        for role, prompt in built.items():
            target = PROMPT_DIR / "builds" / BUILD_FILE_NAMES[role]
            print(f"built: {target} ({len(prompt)} chars)")