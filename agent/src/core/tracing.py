"""Request tracing — per-request correlation id (v8.3.3).

SSE 请求入口生成 request_id，经 contextvar 贯穿 graph/agent/tool 全链路，
日志过滤器注入 %(request_id)s 字段，多用户并发时可串线排查。
"""
from __future__ import annotations

import contextvars

_request_id: contextvars.ContextVar = contextvars.ContextVar(
    "request_id", default="-")

# v8.3.7: 任务 job_id（write 流水线挂钩任务状态更新）
_job_id: contextvars.ContextVar = contextvars.ContextVar(
    "job_id", default="")

# v8.4.3: 会话 session_id（工具执行/权限判定需要，registry 无法从请求体取得）
_session_id: contextvars.ContextVar = contextvars.ContextVar(
    "session_id", default="")


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


def get_request_id() -> str:
    return _request_id.get()


def set_job_id(job_id: str) -> None:
    _job_id.set(job_id)


def get_job_id() -> str:
    return _job_id.get()


def set_session_id(session_id: str) -> None:
    _session_id.set(session_id)


def get_session_id() -> str:
    return _session_id.get()


def new_request_id() -> str:
    import uuid
    rid = uuid.uuid4().hex[:8]
    set_request_id(rid)
    return rid
