"""Prompt Loader — 统一 SystemMessage 组装入口.

v8.1.1: 单一 SystemMessage 架构.
  - 1 个 SystemMessage: 角色 + 约束 + 决策原则 + 输出格式 + 策略卡片
  - 工具参数 schema 不在此处, 由 bind_tools 提供
  - format_hint 由 FAST_MODEL 轻量预测, LLM 可覆盖
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Iterable

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

    parts.append(_load_format(format_hint))

    if include_strategy_cards and query:
        output_cards = _load_skill_cards(query=query, card_type="output", top_k=3)
        if output_cards:
            parts.append(f"## 输出指导\n\n{output_cards}")

    return _join_parts(parts)


def assemble_agent_prompt(
    agent_name: str,
    *,
    skills: list[str] | None = None,
    task_type: str | None = None,
) -> str:
    agent_normalized = (agent_name or "").strip().lower()
    if agent_normalized not in VALID_AGENTS:
        raise ValueError(
            f"Unknown agent name: {agent_name}. Valid: {sorted(VALID_AGENTS)}"
        )
    parts: list[str] = []
    parts.append(_read_prompt_cached(AGENT_FILES[agent_normalized]))

    if skills:
        skill_parts: list[str] = []
        for skill in skills:
            skill_content = _read_prompt_cached(f"skills/{skill}.md")
            if skill_content:
                skill_parts.append(skill_content)
        if skill_parts:
            parts.append("## 写作技能\n\n" + "\n\n".join(skill_parts))

    if task_type:
        task_prompt = _read_prompt_cached(f"strategies/planning/{task_type}.md")
        if task_prompt:
            parts.append(f"## 任务策略\n\n{task_prompt}")

    return _join_parts(parts)


def available_formats() -> list[str]:
    return sorted(VALID_FORMATS)


def available_agents() -> list[str]:
    return sorted(VALID_AGENTS)
