"""Stream user-facing answer text and aggregate a complete model response.

替换用户可见回答生成点的 ainvoke（expert/light 的 supervisor 主循环与
统一收尾 _force_final_answer）：
  - astream 逐 chunk 聚合：AIMessageChunk 合并 → 完整 AIMessage
    （tool_calls / usage_metadata 完整，cache_metrics 计量路径不变）
  - on_text: content 增量回调（前端 text 事件 → 回答区逐 token 渲染）
  - Private reasoning_content is deliberately discarded: it is not user-facing
    progress and must not be streamed or persisted in conversation history.
  - 失败语义：异常上抛，由调用方按原有重试逻辑处理。流式已上屏内容
    在重试时会轻微重复——权衡后接受（生产网络稳定、重试概率低）。
"""
from __future__ import annotations

import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


async def stream_llm_response(
    llm,
    messages: list,
    on_text: Optional[Callable[[str], None]] = None,
    on_reasoning: Optional[Callable[[str], None]] = None,
):
    """流式调用 LLM 并聚合完整响应，返回 AIMessage（与 ainvoke 同构）。

    Args:
        llm: 支持 astream 的 langchain chat model（CitrusChatOpenAI）
        messages: langchain 消息列表
        on_text: content 增量回调（回答区逐 token）
        on_reasoning: deprecated compatibility argument; never invoked
    """
    full = None
    async for chunk in llm.astream(messages):
        if full is None:
            full = chunk
        else:
            full = full + chunk

        # content 增量（str 形态；多模态 list 形态罕见，跳过展示）
        c = chunk.content
        if isinstance(c, str) and c and on_text is not None:
            try:
                on_text(c)
            except Exception:
                pass

    if full is None:
        raise RuntimeError("stream_llm_response: empty stream")
    # Some provider adapters copy vendor-specific fields into message metadata.
    # Strip that field before the message can enter persisted tool-call history.
    try:
        full.additional_kwargs.pop("reasoning_content", None)
    except Exception:
        pass
    return full
