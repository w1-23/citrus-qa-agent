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

# v8.15: 本次请求是否开启联网搜索（前端开关逐请求下发；工具执行层据此短路）
# 与 session_id 同机制：chat_v2 显式写入，经 copy_context 贯穿到工具协程/线程。
_web_search_enabled: contextvars.ContextVar = contextvars.ContextVar(
    "web_search_enabled", default=False)

# v9.1（用户决策：每个用户请求最多调用一次 deepseek_web_search）: 联网调用
# 请求级预算。v8.17.17 的 _web_used 在 agent_runner 实例内（每次 run_agent 重置），
# v9.1 联网移出 retrieve-agent 到独立 web-agent（无 LLM、直接调工具）后，预算必须
# 上移到请求级 contextvar——chat_v2 每请求 reset，deepseek_web_search 入口消费。
_web_budget_left: contextvars.ContextVar = contextvars.ContextVar(
    "web_budget_left", default=1)

# v8.15.3d: 本次请求的用户原始问题——联网工具把原始问题直传 DeepSeek 原生联网，
# 让 output_text 围绕原始问题作答（模型给的检索词仅作"搜索参考关键词"）。
_original_query: contextvars.ContextVar = contextvars.ContextVar(
    "original_query", default="")


def set_web_search_enabled(enabled: bool) -> None:
    _web_search_enabled.set(bool(enabled))


def web_search_enabled() -> bool:
    return _web_search_enabled.get()


def reset_web_budget(quota: int = 1) -> None:
    """v9.1: 每请求开始重置联网预算（chat_v2 入口调用，与 set_web_search_enabled 同处）。"""
    _web_budget_left.set(max(int(quota), 0))


def consume_web_budget() -> bool:
    """消费一次联网预算。返回 False = 本次请求联网预算已用尽（上层应短路）。"""
    v = _web_budget_left.get()
    if v <= 0:
        return False
    _web_budget_left.set(v - 1)
    return True


def set_original_query(query: str) -> None:
    _original_query.set(str(query or ""))


def original_query() -> str:
    return _original_query.get()


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
