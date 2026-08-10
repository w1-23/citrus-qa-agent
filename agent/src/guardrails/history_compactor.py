"""History Compactor — 按需压缩.

v8.1.1: 退化为工厂函数，由 ContextBudget 按需调用.
不再自动初始化 LLM 客户端.
"""
from __future__ import annotations

import logging
import re
from typing import List

logger = logging.getLogger(__name__)

_CITE_PATTERN = re.compile(r'\[[^\]]+\]')


def _messages_to_text(messages: list, max_per_msg: int = 400) -> str:
    lines = []
    for m in messages:
        role = ""
        content = ""
        if hasattr(m, "type"):
            role = m.type
        elif hasattr(m, "role"):
            role = m.role
        if hasattr(m, "content") and m.content:
            content = str(m.content) if not isinstance(m.content, str) else m.content
        clean = _CITE_PATTERN.sub("", content)[:max_per_msg]
        lines.append(f"[{role}]: {clean}")
    return "\n".join(lines)


def _fallback_summary(messages: list) -> str:
    if not messages:
        return "(no conversation history)"
    recent = messages[-6:]
    text = _messages_to_text(recent, max_per_msg=300)
    return f"最近几轮对话摘要:\n{text[:2000]}"


async def compact_messages(
    messages: list,
    fast_llm=None,
    max_tokens: int = 800,
) -> str:
    """压缩一组消息为摘要字符串.

    Args:
        messages: 需要压缩的消息列表
        fast_llm: ChatOpenAI 客户端实例 (可选). 不提供时用 fallback.
        max_tokens: 摘要输出的最大 token

    Returns:
        压缩摘要文本
    """
    if not messages:
        return "(empty)"

    dialogue = _messages_to_text(messages, max_per_msg=600)

    if fast_llm is None:
        return _fallback_summary(messages)

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        prompt = (
            "你是对话摘要专家。将以下多轮对话压缩为一段摘要。\n\n"
            "要求:\n"
            "1. 保留所有关键科研实体(基因名、病害名、品种名、化合物名等)\n"
            "2. 保留所有具体数值(浓度、倍数、p值、百分比等)\n"
            "3. 保留用户的核心研究意图和已达成的共识\n"
            "4. 丢弃寒暄、重复确认、无关闲聊\n"
            "5. 仅输出摘要内容, 不要前缀说明\n\n"
            f"对话历史:\n{dialogue[:8000]}"
        )

        resp = await fast_llm.ainvoke([
            SystemMessage(content="你是专业对话摘要专家。"),
            HumanMessage(content=prompt),
        ])

        result = (resp.content or "").strip()
        if not result:
            return _fallback_summary(messages)

        original_len = sum(
            len(str(m.content)) if hasattr(m, "content") else 0
            for m in messages
        )
        logger.info(
            f"[Compactor] {len(messages)} msgs/{original_len} chars "
            f"-> {len(result)} chars"
        )
        return result

    except Exception as e:
        logger.warning(f"[Compactor] LLM summary failed: {e}")
        return _fallback_summary(messages)
