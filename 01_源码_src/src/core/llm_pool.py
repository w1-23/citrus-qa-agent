"""LLM 客户端进程级复用 (v8.4).

此前每请求新建 2~N 个 ChatOpenAI 实例（supervisor/子Agent/压缩/hint 各建），
每个实例独立 httpx 连接池，初始化与连接开销重复。现按
(model, api_key, base_url, temperature, timeout, max_tokens, thinking_off) 缓存复用；
bind_tools 仍由调用方按需包装（轻量本地操作，不属于客户端创建开销）。

注意: 前端设置面板切换模型后，get_deepseek_model() 返回新值（RESOLVED_MAIN_MODEL），
自然产生新缓存键，旧实例保留至进程退出（成本可忽略）。
"""
from __future__ import annotations

import logging
import threading

from langchain_openai import ChatOpenAI

from src.config import settings

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


def is_thinking_rejected(exc: Exception) -> bool:
    """v8.17.19: 判定 LLM 异常是否因"关闭思维链字段被网关拒绝"。

    命中即应去参重试一次（fail-soft）。判定：HTTP 400/404/422 状态码，
    或错误文本包含 thinking/reasoning/enable_thinking 字样（网关自定义错误兜底）。
    """
    status = getattr(exc, "status_code", None)
    if status in (400, 404, 422):
        return True
    s = str(exc).lower()
    return any(h in s for h in ("thinking", "reasoning", "enable_thinking"))


class _ThinkingOffWrapper:
    """v8.17.19: thinking_off 客户端级 fail-soft 包装。

    primary = 带关闭字段（extra_body）的实例；fallback = 无字段实例。
    网关拒绝关闭字段（400/422）时自动用 fallback 重试一次并记日志——
    保证该调用点不因参数被拒而挂（仅多一次快速失败请求）；真机日志据此
    判断字段是否被接受（连续出现"去参回退重试"警告 = 字段无效，需换
    config model.reasoning_off_body 或模型档）。

    只转发实际使用的接口（ainvoke / astream / bind_tools / model_kwargs 只读）。
    """

    def __init__(self, primary, fallback):
        self.primary = primary
        self.fallback = fallback

    @property
    def model_kwargs(self) -> dict:
        return self.primary.model_kwargs

    @property
    def extra_body(self) -> dict | None:
        return self.primary.extra_body

    def bind_tools(self, tools, **kw):
        return _ThinkingOffWrapper(
            self.primary.bind_tools(tools, **kw),
            self.fallback.bind_tools(tools, **kw))

    async def ainvoke(self, messages, **kw):
        try:
            return await self.primary.ainvoke(messages, **kw)
        except Exception as e:
            if is_thinking_rejected(e):
                logger.warning(
                    f"[LLMPool] thinking 关闭字段被网关拒绝，去参回退重试: {e}")
                return await self.fallback.ainvoke(messages, **kw)
            raise

    async def astream(self, messages, **kw):
        try:
            async for c in self.primary.astream(messages, **kw):
                yield c
        except Exception as e:
            if is_thinking_rejected(e):
                logger.warning(
                    f"[LLMPool] thinking 关闭字段被网关拒绝，去参回退重试: {e}")
                async for c in self.fallback.astream(messages, **kw):
                    yield c
            else:
                raise


def _new_client(model, api_key, base_url, temperature, timeout, max_tokens,
                extra_body: dict | None):
    kw: dict = {}
    if extra_body:
        # v8.17.19: 关闭字段必须经 extra_body 原样透传网关——langchain-openai
        # 把 model_kwargs 里的 extra_body 提取为显式构造参数（UserWarning 实证：
        # "should be specified explicitly"），故直接传构造参数；SDK 将其并入
        # 请求 body 由网关裁决。v8.17.18 曾放 model_kwargs 顶层 thinking，
        # langchain 展开为 create() 关键字参数 → TypeError（retrieve-agent 全挂根因）。
        kw["extra_body"] = dict(extra_body)
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        timeout=timeout,
        max_tokens=max_tokens,
        **kw,
    )


def get_llm(
    model: str,
    api_key: str,
    base_url: str,
    *,
    temperature: float = 0.0,
    timeout: float = 120,
    max_tokens: int | None = None,
    thinking_off: bool = False,
):
    """进程级客户端复用；v8.17.19: thinking_off 恢复有效逻辑。

    thinking_off=True 时经 extra_body 下发关闭字段（形态见
    settings.MODEL_REASONING_OFF_BODY，默认官方推荐 {"thinking":
    {"type": "disabled"}}），返回 _ThinkingOffWrapper（fail-soft 自动去参重试）；
    thinking_off=False/不传时不发送任何思维链字段（默认保持思维链开启——
    supervisor 融合判断路径即此形态）。
    """
    extra_body = None
    if thinking_off:
        cfg = getattr(settings, "MODEL_REASONING_OFF_BODY", None) or {}
        if isinstance(cfg, dict) and cfg:
            extra_body = dict(cfg)
    key = (model, api_key, base_url, temperature, timeout, max_tokens, bool(extra_body))
    with _lock:
        inst = _cache.get(key)
        if inst is None:
            inst = _new_client(model, api_key, base_url, temperature, timeout,
                               max_tokens, extra_body)
            _cache[key] = inst
        if extra_body:
            fkey = (model, api_key, base_url, temperature, timeout, max_tokens, False)
            fallback = _cache.get(fkey)
            if fallback is None:
                fallback = _new_client(model, api_key, base_url, temperature,
                                       timeout, max_tokens, None)
                _cache[fkey] = fallback
            return _ThinkingOffWrapper(inst, fallback)
        return inst