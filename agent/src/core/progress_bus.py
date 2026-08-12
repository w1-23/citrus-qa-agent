"""Progress bus — shared between graph nodes and SSE generator.

v8.3.3: request-scoped progress queue (contextvars, 消除跨请求串扰) +
unified JSON-string data format + usage delta emitter + heartbeat/progress
tracking for long tool executions + SSE raw-frame debug logging.
"""
from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 全局兜底队列（非 SSE 上下文，如 CLI/测试）；SSE 请求使用 per-request 队列
_progress_queue: asyncio.Queue | None = None
_log_queue: asyncio.Queue | None = None

# 请求级队列: 每个 SSE 请求 set_request_queue() 绑定自己的队列，emit 只进本请求
_current_queue: contextvars.ContextVar = contextvars.ContextVar(
    "progress_queue", default=None)


def get_progress_queue() -> asyncio.Queue:
    q = _current_queue.get()
    if q is not None:
        return q
    global _progress_queue
    if _progress_queue is None:
        _progress_queue = asyncio.Queue()
    return _progress_queue


def set_request_queue(queue: asyncio.Queue) -> None:
    """绑定当前请求的进度队列（SSE 事件生成器内调用）。"""
    _current_queue.set(queue)


def clear_request_queue() -> None:
    _current_queue.set(None)


def get_log_queue() -> asyncio.Queue:
    global _log_queue
    if _log_queue is None:
        _log_queue = asyncio.Queue(maxsize=500)
    return _log_queue


def reset_progress_queue() -> None:
    global _progress_queue, _log_queue
    _progress_queue = asyncio.Queue()
    if _log_queue is not None:
        while not _log_queue.empty():
            try:
                _log_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
    _log_queue = asyncio.Queue(maxsize=500)


def _encode_event(event_type: str, data: dict) -> dict:
    """Encode an SSE event dict with JSON-string data field (canonical format)."""
    return {"event": event_type, "data": json.dumps(data, ensure_ascii=False)}


def emit_progress(event_type: str, data: dict) -> None:
    """Push a progress event to the current request's SSE queue (non-blocking)."""
    q = get_progress_queue()
    try:
        q.put_nowait(_encode_event(event_type, data))
    except asyncio.QueueFull:
        pass


def emit_encoded(event_type: str, data: dict) -> None:
    """Alias for emit_progress — canonical event emission."""
    emit_progress(event_type, data)


# ═══════════════════════════════════════════════════════════════════════
# Token usage delta emitter — context_usage 增量推送
# ═══════════════════════════════════════════════════════════════════════

# 每次 LLM 调用的 usage.total_tokens 是该次请求的累计值（含重放历史），
# 前端若直接累加会多轮重复计数 → 按 session+source 发增量。
_usage_last: dict = {}


def emit_usage_delta(session_id: str, source: str, input_tokens: int = 0,
                     output_tokens: int = 0, total: int = 0) -> None:
    """context_usage 增量推送 (v8.3.3): total 为单次调用累计，delta 为相对上次的增量。"""
    if not total:
        return
    key = f"{session_id}:{source}"
    last = _usage_last.get(key, 0)
    if last == 0:
        delta = total  # 首次调用: 增量 = 本次总量
    else:
        delta = max(total - last, 0)  # 后续: 相对上次的增量（重放历史不重复计）
    if delta == 0:
        return  # 同值重复（如调用未消耗新 token），不发零增量
    _usage_last[key] = total
    emit_encoded("context_usage", {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total": total,
        "delta": delta,
        "source": source,
    })


# ═══════════════════════════════════════════════════════════════════════
# SSE Raw-Frame Debug Logger (prints every outgoing SSE chunk)
# ═══════════════════════════════════════════════════════════════════════

_sse_debug_enabled: bool = True


def enable_sse_debug(enabled: bool = True) -> None:
    global _sse_debug_enabled
    _sse_debug_enabled = enabled


def log_sse_frame(event_dict: dict) -> None:
    """Print an outgoing SSE frame for debugging purposes."""
    if not _sse_debug_enabled:
        return
    event_type = event_dict.get("event", "?")
    data_str = event_dict.get("data", "")
    preview = data_str[:200] if len(data_str) > 200 else data_str
    logger.debug(f"[SSE→] event={event_type} | data={preview}")


# ═══════════════════════════════════════════════════════════════════════
# Tool Execution Heartbeat / Progress tracker
# ═══════════════════════════════════════════════════════════════════════

_tool_start_times: dict[str, tuple[float, str]] = {}


def mark_tool_start(tool_call_id: str, tool_name: str = "") -> None:
    _tool_start_times[tool_call_id] = (time.perf_counter(), tool_name)


def mark_tool_end(tool_call_id: str) -> None:
    _tool_start_times.pop(tool_call_id, None)


def get_tool_elapsed(tool_call_id: str) -> float:
    entry = _tool_start_times.get(tool_call_id)
    if entry is None:
        return 0.0
    return time.perf_counter() - entry[0]


def get_tool_name(tool_call_id: str) -> str:
    entry = _tool_start_times.get(tool_call_id)
    return entry[1] if entry else ""


def get_running_tools() -> list[tuple[str, str, float]]:
    """Returns list of (call_id, tool_name, elapsed_seconds)."""
    now = time.perf_counter()
    return [(cid, name, now - ts) for cid, (ts, name) in _tool_start_times.items()]


def clear_tool_timers() -> None:
    _tool_start_times.clear()


# ═══════════════════════════════════════════════════════════════════════
# Structured event emitters (canonical type constants)
# ═══════════════════════════════════════════════════════════════════════


def emit_thinking(content: str) -> None:
    """Emit a THINKING event (LLM internal reasoning)."""
    emit_encoded("thinking", {"content": content})


def emit_tool_call_start(tool_name: str, args: dict, tool_call_id: str = "") -> None:
    """Emit a TOOL_CALL_START event with fully assembled arguments."""
    emit_encoded("tool_call_start", {
        "tool_name": tool_name,
        "args": args,
        "tool_call_id": tool_call_id,
    })
    if tool_call_id:
        mark_tool_start(tool_call_id, tool_name)


def emit_tool_executing(message: str, tool_name: str = "", tool_call_id: str = "") -> None:
    """Emit a TOOL_EXECUTING event (progress / heartbeat during tool run)."""
    emit_encoded("tool_executing", {
        "message": message,
        "tool_name": tool_name,
        "tool_call_id": tool_call_id,
    })


def emit_tool_result(tool_name: str, output: str, tool_call_id: str = "",
                     is_error: bool = False, summary: str = "") -> None:
    """Emit a TOOL_RESULT event with output content and optional summary."""
    if tool_call_id:
        mark_tool_end(tool_call_id)
    emit_encoded("tool_result", {
        "tool_name": tool_name,
        "output": output[:100000],
        "summary": summary[:500],
        "is_error": is_error,
        "tool_call_id": tool_call_id,
    })


def emit_text(content: str) -> None:
    """Emit a FINAL_ANSWER text chunk."""
    emit_encoded("text", {"content": content})


def emit_status(stage: str, **kwargs) -> None:
    """Emit a generic status event (backward compat)."""
    payload = {"stage": stage}
    payload.update(kwargs)
    emit_encoded("status", payload)


# ═══════════════════════════════════════════════════════════════════════
# SSELogHandler — forwards Python log lines to SSE log queue
# ═══════════════════════════════════════════════════════════════════════


class SSELogHandler(logging.Handler):
    """Custom logging handler that emits log lines to the SSE log queue."""

    def __init__(self):
        super().__init__(logging.INFO)
        self._start_time = time.perf_counter()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            elapsed = time.perf_counter() - self._start_time
            line = (
                f"[{elapsed:.1f}s] {record.levelname[:1]} "
                f"{record.name.split('.')[-1]}: {record.getMessage()}"
            )
            q = get_log_queue()
            q.put_nowait(
                _encode_event("log_line", {"line": line, "level": record.levelname})
            )
        except Exception:
            pass
