"""LLM 客户端进程级复用 (v8.4).

此前每请求新建 2~N 个 ChatOpenAI 实例（supervisor/子Agent/压缩/hint 各建），
每个实例独立 httpx 连接池，初始化与连接开销重复。现按
(model, api_key, base_url, temperature, timeout, max_tokens) 缓存复用；
bind_tools 仍由调用方按需包装（轻量本地操作，不属于客户端创建开销）。

注意: 运行时切换模型（switch_model）后 get_deepseek_model() 返回新值，
自然产生新缓存键，旧实例保留至进程退出（成本可忽略）。
"""
from __future__ import annotations

import threading

from langchain_openai import ChatOpenAI

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


def clear_llm_pool() -> None:
    """清空缓存（测试/热更新场景）。"""
    with _lock:
        _cache.clear()
