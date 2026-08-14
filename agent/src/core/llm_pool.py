"""LLM 客户端进程级复用 (v8.4).

此前每请求新建 2~N 个 ChatOpenAI 实例（supervisor/子Agent/压缩/hint 各建），
每个实例独立 httpx 连接池，初始化与连接开销重复。现按
(model, api_key, base_url, temperature, timeout, max_tokens) 缓存复用；
bind_tools 仍由调用方按需包装（轻量本地操作，不属于客户端创建开销）。

注意: 运行时切换模型（switch_model）后 get_deepseek_model() 返回新值，
自然产生新缓存键，旧实例保留至进程退出（成本可忽略）。
"""
from __future__ import annotations

import logging
import threading

from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


def _install_reasoning_passthrough() -> None:
    """v8.4.13 透传 DeepSeek reasoning_content（思维链流式展示）。

    langchain-openai 的 delta/消息转换是模块级函数，明确不保留第三方厂商的
    非标准响应字段（reasoning_content）。monkeypatch 两个转换函数一次
    （幂等）：把 reasoning_content 追加进 additional_kwargs——仅多存一个
    字段，无副作用；流式聚合后即可取思维链。
    """
    try:
        import langchain_openai.chat_models.base as _lcb
        if getattr(_lcb, "_citrus_reasoning_patched", False):
            return
        _orig_delta = _lcb._convert_delta_to_message_chunk
        _orig_dict = _lcb._convert_dict_to_message

        def _delta(_dict, default_class):
            chunk = _orig_delta(_dict, default_class)
            try:
                rc = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
                if rc:
                    chunk.additional_kwargs["reasoning_content"] = rc
            except Exception:
                pass
            return chunk

        def _dict_conv(_dict):
            msg = _orig_dict(_dict)
            try:
                rc = _dict.get("reasoning_content") if isinstance(_dict, dict) else None
                if rc:
                    msg.additional_kwargs["reasoning_content"] = rc
            except Exception:
                pass
            return msg

        _lcb._convert_delta_to_message_chunk = _delta
        _lcb._convert_dict_to_message = _dict_conv
        _lcb._citrus_reasoning_patched = True
        logger.debug("[LLMPool] reasoning_content passthrough installed")
    except Exception as e:
        logger.warning(f"[LLMPool] reasoning passthrough unavailable: {e}")


_install_reasoning_passthrough()


_cache: dict = {}
_lock = threading.Lock()


def get_llm(
    model: str,
    api_key: str,
    base_url: str,
    *,
    temperature: float = 0.0,
    timeout: float = 120,
    max_tokens: int | None = None,
) -> ChatOpenAI:
    key = (model, api_key, base_url, temperature, timeout, max_tokens)
    with _lock:
        inst = _cache.get(key)
        if inst is None:
            inst = ChatOpenAI(
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                timeout=timeout,
                max_tokens=max_tokens,
            )
            _cache[key] = inst
        return inst


