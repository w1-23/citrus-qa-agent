"""v8.4.13 真流式 LLM 调用：回答逐 token 上屏 + 思维链（reasoning_content）提取。

替换用户可见回答生成点的 ainvoke（expert/light 的 supervisor 主循环与
统一收尾 _force_final_answer）：
  - astream 逐 chunk 聚合：AIMessageChunk 合并 → 完整 AIMessage
    （tool_calls / usage_metadata 完整，cache_metrics 计量路径不变）
  - on_text: content 增量回调（前端 text 事件 → 回答区逐 token 渲染）
  - on_reasoning: reasoning_content 增量回调（前端 reasoning 事件 →
    「深度思考」折叠块；经 CitrusChatOpenAI 子类透传到 additional_kwargs）
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
        on_reasoning: reasoning_content 增量回调（深度思考折叠块）
    """
    full = None
    reasoning_parts: list[str] = []   # v8.10l: 显式收集 reasoning（chunk 合并不可靠）
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

        # 思维链增量（CitrusChatOpenAI 透传到 additional_kwargs）
        rc = (getattr(chunk, "additional_kwargs", None) or {}).get("reasoning_content")
        if rc:
            reasoning_parts.append(rc)
            if on_reasoning is not None:
                try:
                    on_reasoning(rc)
                except Exception:
                    pass

    if full is None:
        raise RuntimeError("stream_llm_response: empty stream")
    # v8.10l: 聚合后的完整消息显式携带 reasoning（历史持久化用——
    # 退出会话再回来可恢复深度思考折叠块）
    if reasoning_parts:
        try:
            full.additional_kwargs["reasoning_content"] = "".join(reasoning_parts)
        except Exception:
            pass
    return full
