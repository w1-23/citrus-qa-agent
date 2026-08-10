"""Context Budget — 实时 token 估算与按需压缩.

v8.1.1: 每次 LLM 调用前检查 token 用量.
  - < 90%: NORMAL, 不处理
  - 90-95%: SUMMARIZE, LLM 压缩最早消息为摘要
  - >= 95%: TRUNCATE, 硬截断只保留摘要+最近轮次
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Callable

logger = logging.getLogger(__name__)


class ContextBudgetLevel(Enum):
    NORMAL = "normal"
    SUMMARIZE = "summarize"
    TRUNCATE = "truncate"


@dataclass
class ContextBudgetConfig:
    max_tokens: int = 1000000
    soft_threshold: float = 0.60
    hard_threshold: float = 0.93
    summarize_ratio: float = 0.70
    keep_recent_turns: int = 2
    compact_max_tokens: int = 800
    enabled: bool = True


@dataclass
class BudgetResult:
    level: ContextBudgetLevel
    messages: list
    summary: str | None = None


def _estimate_chars_tokens(text: str) -> int:
    """混合语言 token 估算 (AG-15)。

    对齐 opencode 的字符估算思路（len/4，面向英文代码），本项目适配中英混合：
      - 中文/日文/韩文 CJK: ~1.2 token/字（DeepSeek BPE 实测区间 1.0-1.5）
      - 其他（英文/数字/符号）: 4 字符/token（与 opencode 一致）
    无外部 tokenizer 依赖，偏差目标 <15%。
    """
    if not text:
        return 0
    cjk = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    other = len(text) - cjk
    return max(int(cjk * 1.2 + other / 4), 1)


class ContextBudget:
    def __init__(self, config: ContextBudgetConfig | None = None) -> None:
        self.config = config or ContextBudgetConfig()
        self._compact_fn: Callable | None = None

    def set_compact_fn(self, fn: Callable) -> None:
        self._compact_fn = fn

    def estimate_tokens(self, messages: list) -> int:
        total = 0
        for msg in messages:
            content = ""
            if hasattr(msg, "content"):
                c = msg.content
                if isinstance(c, str):
                    content = c
                elif isinstance(c, list):
                    for block in c:
                        if isinstance(block, dict):
                            content += str(block.get("text", ""))
                else:
                    content = str(c) if c else ""
            total += _estimate_chars_tokens(content)

            tcs = getattr(msg, "tool_calls", None)
            if tcs:
                for tc in tcs:
                    args = getattr(tc, "args", {}) or {}
                    args_str = str(args) if args else ""
                    total += 30 + max(len(args_str) // 2, 1)
        return int(total)

    def _split_turns(self, messages: list) -> list[list]:
        turns = []
        current: list = []
        for msg in messages:
            current.append(msg)
            if hasattr(msg, "content") and msg.content:
                tcs = getattr(msg, "tool_calls", None)
                if not tcs:
                    if getattr(msg, "type", None) == "ai" or (hasattr(msg, "role") and msg.role == "assistant"):
                        pass
            role = getattr(msg, "type", None) or getattr(msg, "role", "")
            if role in ("human", "user"):
                if current and len(current) > 1:
                    turns.append(current)
                    current = [msg]
                else:
                    current = [msg]
            if role in ("ai", "assistant") and not getattr(msg, "tool_calls", None):
                if current and any(
                    getattr(m, "type", "") in ("human", "user")
                    or getattr(m, "role", "") in ("human", "user")
                    for m in current
                ):
                    turns.append(current)
                    current = []
                else:
                    current = []
        if current:
            turns.append(current)
        return turns

    async def check(self, messages: list) -> BudgetResult:
        if not self.config.enabled:
            return BudgetResult(level=ContextBudgetLevel.NORMAL, messages=list(messages))

        tokens = self.estimate_tokens(messages)
        max_t = self.config.max_tokens
        ratio = tokens / max_t if max_t > 0 else 0

        logger.debug(
            f"[ContextBudget] tokens={tokens}/{max_t} ({ratio:.1%})"
        )

        if ratio < self.config.soft_threshold:
            return BudgetResult(level=ContextBudgetLevel.NORMAL, messages=list(messages))

        if ratio >= self.config.hard_threshold:
            return await self._truncate(messages)

        return await self._summarize(messages)

    async def _summarize(self, messages: list) -> BudgetResult:
        turns = self._split_turns(messages)
        if len(turns) <= 3:
            return BudgetResult(level=ContextBudgetLevel.NORMAL, messages=list(messages))

        cutoff = max(1, int(len(turns) * self.config.summarize_ratio))
        old_turns_flat = [msg for turn in turns[:cutoff] for msg in turn]
        recent_turns_flat = [msg for turn in turns[cutoff:] for msg in turn]

        summary = ""
        if self._compact_fn and old_turns_flat:
            try:
                summary = await self._compact_fn(old_turns_flat)
            except Exception as e:
                logger.warning(f"[ContextBudget] summary failed: {e}")

        from langchain_core.messages import HumanMessage
        summary_msg = HumanMessage(
            content=f"<conversation_summary>\n{summary}\n</conversation_summary>"
        ) if summary else None

        new_messages = [messages[0]] if messages else []
        if summary_msg:
            new_messages.append(summary_msg)
        new_messages.extend(recent_turns_flat)

        logger.info(
            f"[ContextBudget] SUMMARIZE: {len(old_turns_flat)} msgs -> {len(summary)} chars summary, "
            f"keeping {len(recent_turns_flat)} recent msgs"
        )
        return BudgetResult(
            level=ContextBudgetLevel.SUMMARIZE,
            messages=new_messages,
            summary=summary,
        )

    async def _truncate(self, messages: list) -> BudgetResult:
        turns = self._split_turns(messages)
        keep = self.config.keep_recent_turns
        if len(turns) <= keep + 2:
            return BudgetResult(level=ContextBudgetLevel.NORMAL, messages=list(messages))

        old_turns_flat = [msg for turn in turns[:-keep] for msg in turn]
        recent_turns_flat = [msg for turn in turns[-keep:] for msg in turn]

        summary = ""
        if self._compact_fn and old_turns_flat:
            try:
                summary = await self._compact_fn(old_turns_flat)
            except Exception as e:
                logger.warning(f"[ContextBudget] truncate summary failed: {e}")

        from langchain_core.messages import HumanMessage
        new_messages = [messages[0]] if messages else []
        if summary:
            new_messages.append(HumanMessage(
                content=f"<conversation_summary>\n{summary}\n</conversation_summary>"
            ))
        new_messages.extend(recent_turns_flat)

        logger.info(
            f"[ContextBudget] TRUNCATE: keeping {keep} turns + summary, "
            f"total={len(new_messages)} msgs"
        )
        return BudgetResult(
            level=ContextBudgetLevel.TRUNCATE,
            messages=new_messages,
            summary=summary,
        )
