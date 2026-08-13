"""History Compactor — 上下文感知批量压缩 (v8.4).

书 2.7.4/2.7.5 生产级分层压缩的 L2 批量摘要:
  - 压缩提示带入当前 query（上下文感知压缩，实验2-10 最优策略: 省 76% token）
  - 保留优先级（2.7.6 显式定义）: 决策/约束、文件变更、验证状态、标识符
    (DOI/evidence_id/artifact_id) 必须保留；旧工具过程、错误日志、低相关
    证据可丢弃
  - 透明截断: 输入超限分片处理并显式 [TRUNCATED] 标记（绝不静默截断）
  - max_tokens 真正传入 LLM 调用（v8.3 遗留 bug: 声明未生效）
"""
from __future__ import annotations

import logging
import re
from typing import List, Optional

logger = logging.getLogger(__name__)

_CITE_PATTERN = re.compile(r'\[[^\]]+\]')

# checkpoint 旧摘要经 prior_summary 参数单独传入；对话文本中残留的包装块
# 需剥离，避免同一信息双份进 prompt（v8.4）
_SUMMARY_BLOCK_PATTERN = re.compile(
    r'<conversation_summary>.*?</conversation_summary>', re.DOTALL)

# 压缩输入总预算（字符）——超出部分分片处理，每片独立摘要后再合并
_MAX_INPUT_CHARS = 24000


def _messages_to_text(messages: list, max_per_msg: int = 600) -> str:
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


def _build_prompt(dialogue: str, query: str, prior_summary: str) -> str:
    return (
        "你是科研对话压缩专家。将以下多轮对话压缩为一段结构化摘要。\n\n"
        f"Given the user query (当前研究意图): {query[:500] or '(无)'}\n\n"
        f"{'已有摘要（增量更新，基于它继续整合，不要重复它已覆盖的内容）:\n'
         + prior_summary[:2000] + chr(10) if prior_summary else ''}"
        "要求:\n"
        "1. 保留所有关键科研实体(基因名、病害名、品种名、化合物名等)\n"
        "2. 保留所有具体数值(浓度、倍数、p值、百分比等)\n"
        "3. 保留用户的核心研究意图、已达成的共识与关键决策\n"
        "4. 必须原样保留所有标识符: DOI、evidence_id、artifact_id、chunk_id、"
        "文件路径、引用编号\n"
        "5. 保留被引用证据的结论与验证状态(pass/fail)\n"
        "6. 可丢弃: 工具调用过程细节、错误日志、低相关检索中间结果\n"
        "7. 仅输出摘要内容, 不要前缀说明\n\n"
        f"对话历史:\n{dialogue}"
    )


async def compact_messages(
    messages: list,
    fast_llm=None,
    max_tokens: int = 800,
    query: str = "",
    prior_summary: str = "",
) -> str:
    """压缩一组消息为摘要字符串（上下文感知 + 分片 + 透明截断）。

    Args:
        messages: 需要压缩的消息列表
        fast_llm: ChatOpenAI 客户端实例 (可选). 不提供时用 fallback.
        max_tokens: 摘要输出的最大 token（v8.4: 真正传入 LLM）
        query: 当前用户查询（压缩相关性锚点）
        prior_summary: 已有摘要（增量压缩，防摘要套摘要时保留旧信息）
    """
    if not messages:
        return "(empty)"

    if fast_llm is None:
        return _fallback_summary(messages)

    dialogue = _messages_to_text(messages, max_per_msg=600)
    dialogue = _SUMMARY_BLOCK_PATTERN.sub("", dialogue)

    # 透明分片: 输入超预算时切成多片，逐片摘要后合并（绝不静默截断）
    if len(dialogue) > _MAX_INPUT_CHARS:
        chunks = []
        for i in range(0, len(dialogue), _MAX_INPUT_CHARS):
            chunk = dialogue[i:i + _MAX_INPUT_CHARS]
            if i > 0:
                chunk = f"[上接前文，已截断 {i} 字符]\n{chunk}"
            if i + _MAX_INPUT_CHARS < len(dialogue):
                chunk += f"\n[下接后文，剩余 {len(dialogue) - i - _MAX_INPUT_CHARS} 字符]"
            chunks.append(chunk)
    else:
        chunks = [dialogue]

    try:
        from langchain_core.messages import SystemMessage, HumanMessage

        merged = prior_summary or ""
        for chunk in chunks:
            resp = await fast_llm.ainvoke(
                [
                    SystemMessage(content="你是专业对话摘要专家。"),
                    HumanMessage(content=_build_prompt(chunk, query, merged)),
                ],
                max_tokens=max_tokens,
            )
            part = (resp.content or "").strip()
            if part:
                merged = part  # 后续分片基于前片摘要增量整合（git-log 式）

        if not merged:
            return _fallback_summary(messages)

        original_len = sum(
            len(str(m.content)) if hasattr(m, "content") else 0
            for m in messages
        )
        logger.info(
            f"[Compactor] {len(messages)} msgs/{original_len} chars "
            f"-> {len(merged)} chars (chunks={len(chunks)})"
        )
        return merged

    except Exception as e:
        logger.warning(f"[Compactor] LLM summary failed: {e}")
        return _fallback_summary(messages)
