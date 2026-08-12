"""Request tracing — per-request correlation id (v8.3.3).

SSE 请求入口生成 request_id，经 contextvar 贯穿 graph/agent/tool 全链路，
日志过滤器注入 %(request_id)s 字段，多用户并发时可串线排查。
"""
from __future__ import annotations

import contextvars

_request_id: contextvars.ContextVar = contextvars.ContextVar(
    "request_id", default="-")


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str:
    return _request_id.get()


def new_request_id() -> str:
    import uuid
    rid = uuid.uuid4().hex[:8]
    set_request_id(rid)
    return rid
