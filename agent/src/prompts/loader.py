"""Prompt Loader — 统一 SystemMessage 组装入口.

v8.1.1: 单一 SystemMessage 架构.
  - 1 个 SystemMessage: 角色 + 约束 + 决策原则 + 输出格式 + 策略卡片
  - 工具参数 schema 不在此处, 由 bind_tools 提供
  - format_hint 由 FAST_MODEL 轻量预测, LLM 可覆盖

v8.4 (阶段1 静态前缀): context.static_prefix=true 时——
  - SystemMessage 只含不随 query 变化的静态部分（role/constraints/decision_guide）
  - format 指南与策略卡片经 build_dynamic_blocks() 生成，追加到当前轮
    HumanMessage 尾部（KV Cache 铁律: 前缀字节级稳定，动态内容一律追加末尾）
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable


def _static_prefix_enabled() -> bool:
    try:
        from src.config import settings
        return bool(settings.CONTEXT_STATIC_PREFIX)
    except Exception:
        return False

PROMPT_DIR = Path(__file__).resolve().parent

FORMAT_FILES = {
    "fact": "formats/fact.md",
    "compare": "formats/compare.md",
    "method": "formats/method.md",
    "design": "formats/design.md",
    "review": "formats/review.md",
    "task": "formats/task.md",
    "fallback": "formats/fallback.md",
}

AGENT_FILES = {
    "retrieve-agent": "agents/retrieve-agent.md",
    "write-agent": "agents/write-agent.md",
    "analyze-agent": "agents/analyze-agent.md",
}

# v8.16.1: 草稿先行——分隔符结构化输出模板（独立于系统提示词装配，
# 仅供草稿 fast 调用使用；不进入 assemble_system_prompt，不动静态前缀）
STRUCTURED_OUTPUT_FILE = "structured_output.md"

VALID_FORMATS = set(FORMAT_FILES.keys())
VALID_AGENTS = set(AGENT_FILES.keys())


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
    cleaned = [part.strip() for part in parts if part and part.strip()]
    return "\n\n---\n\n".join(cleaned)


def normalize_format_hint(format_hint: str | None) -> str:
    if not format_hint:
        return "fallback"
    key = format_hint.strip().lower()
    if key not in VALID_FORMATS:
        return "fallback"
    return key


def _load_format(format_hint: str | None) -> str:
    key = normalize_format_hint(format_hint)
    return _read_prompt_cached(FORMAT_FILES[key])


def _load_skill_cards(query: str | None, card_type: str, top_k: int = 3) -> str:
    if not query:
        return ""
    try:
        from src.core.skill_tree import SkillTree
        st = SkillTree()
        cards = st.search_strategy_cards(query=query, card_type=card_type, top_k=top_k)
        if not cards:
            return ""
        if isinstance(cards, list):
            return "\n\n".join(str(card) for card in cards if card)
        return str(cards)
    except Exception:
        return ""


def assemble_system_prompt(
    *,
    mode: str,
    format_hint: str | None = None,
    query: str | None = None,
    include_strategy_cards: bool = True,
) -> str:
    mode_normalized = (mode or "expert").strip().lower()
    parts: list[str] = []

    parts.append(_read_prompt_cached("system/role.md"))
    parts.append(_read_prompt_cached("system/constraints.md"))

    if mode_normalized == "light":
        parts.append(_read_prompt_cached("system/light_rules.md"))
    else:
        parts.append(_read_prompt_cached("system/decision_guide.md"))

    if _static_prefix_enabled():
        # 阶段1: 静态前缀——format 指南/策略卡片移出 SystemMessage
        # (见 build_dynamic_blocks)，前缀字节级稳定，缓存跨请求命中
        return _join_parts(parts)

    parts.append(_load_format(format_hint))

    if include_strategy_cards and query:
        output_cards = _load_skill_cards(query=query, card_type="output", top_k=3)
        if output_cards:
            parts.append(f"## 输出指导\n\n{output_cards}")

    return _join_parts(parts)


def build_dynamic_blocks(
    *,
    format_hint: str | None = None,
    query: str | None = None,
    include_strategy_cards: bool = True,
) -> str:
    """阶段1: 静态前缀模式下，随请求变化的动态内容作为独立块返回。

    由 build_human_message 追加到当前轮 HumanMessage 尾部（<user_query> 之后），
    不进入 SystemMessage —— 动态信息一律追加末尾，前缀缓存不受影响。
    """
    if not _static_prefix_enabled():
        return ""
    blocks: list[str] = []
    fmt = _load_format(format_hint)
    if fmt:
        blocks.append(f"<format_guide>\n{fmt}\n</format_guide>")
    if include_strategy_cards and query:
        cards = _load_skill_cards(query=query, card_type="output", top_k=3)
        if cards:
            blocks.append(f"<output_guide>\n{cards}\n</output_guide>")
    return "\n\n".join(blocks)


def assemble_structured_output_prompt() -> str:
    """v8.16.1: 草稿先行结构化输出模板（===STRUCTURED_START=== 分隔符格式，非 JSON）。

    仅草稿 fast 调用使用（deepseek_web.draft_worker）；模板缺失时返回空串，
    调用方据此跳过草稿特性（fail-soft，不阻塞主链路）。
    """
    return _read_prompt_cached(STRUCTURED_OUTPUT_FILE)


def assemble_agent_prompt(
    agent_name: str,
    *,
    skills: list[str] | None = None,
    task_type: str | None = None,
) -> str:
    """子 Agent 系统提示——只含 agent 基础文件，按 agent_name 字节级稳定。

    v8.4.3: skills/task_type 一律不进入 SystemMessage（参数保留签名兼容，
    实际被忽略）——动态内容经 build_agent_extra_block 追加到首条 HumanMessage，
    属于"追加"而非"修改前缀"，静态缓存不受影响。
    """
    agent_normalized = (agent_name or "").strip().lower()
    if agent_normalized not in VALID_AGENTS:
        raise ValueError(
            f"Unknown agent name: {agent_name}. Valid: {sorted(VALID_AGENTS)}"
        )
    parts: list[str] = []
    parts.append(_read_prompt_cached(AGENT_FILES[agent_normalized]))
    return _join_parts(parts)


def build_agent_extra_block(
    *,
    skills: list[str] | None = None,
    task_type: str | None = None,
    system_prompt_extra: str = "",
) -> str:
    """子 Agent 动态指令块——无条件生效（v8.4.3）。

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



