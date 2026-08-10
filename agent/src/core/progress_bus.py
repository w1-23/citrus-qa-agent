"""Progress bus — shared between graph nodes and SSE generator.

v8.3.0: unified JSON-string data format, ToolCallAccumulator for fragmented
tool_call args assembly, heartbeat/progress tracking for long tool executions,
SSE raw-frame debug logging middleware.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

_progress_queue: asyncio.Queue | None = None
_log_queue: asyncio.Queue | None = None


def get_progress_queue() -> asyncio.Queue:
    global _progress_queue
    if _progress_queue is None:
        _progress_queue = asyncio.Queue()
    return _progress_queue


def get_log_queue() -> asyncio.Queue:
    global _log_queue
    if _log_queue is None:
        _log_queue = asyncio.Queue(maxsize=500)
    return _log_queue


def reset_progress_queue() -> None:
    global _progress_queue, _log_queue
    if _progress_queue is not None:
        while not _progress_queue.empty():
            try:
                _progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
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
    """Push a progress event to the SSE queue (synchronous, non-blocking)."""
    q = get_progress_queue()
    try:
        q.put_nowait(_encode_event(event_type, data))
    except asyncio.QueueFull:
        pass


def emit_encoded(event_type: str, data: dict) -> None:
    """Alias for emit_progress — canonical event emission."""
    emit_progress(event_type, data)


# ═══════════════════════════════════════════════════════════════════════
# Tool Call State Accumulator — assembles fragmented tool_call arguments
# ═══════════════════════════════════════════════════════════════════════


# NOTE: ToolCallAccumulator is reserved for future streaming tool_call support.
# Currently unused (all LLM calls use ainvoke, not streaming).
class ToolCallAccumulator:
    """Assembles delta.tool_calls chunks into complete tool call events.

    LLM providers stream tool_calls as incremental deltas:
      - chunk 0: id="call_abc", name="get_weather"
      - chunk 1: id="call_abc", arguments='{"cit'
      - chunk 2: id="call_abc", arguments='y":"\\u'
      - chunk 3: id="call_abc", arguments='5357\\u'

    This accumulator buffers fragments per call_id and emits
    tool_call_start + tool_executing + tool_result events when ready.
    """

    def __init__(self):
        self._slots: dict[str, dict] = {}

    def feed(self, call_id: str, name: str | None, args_delta: str | None) -> Optional[str]:
        """Feed a delta chunk. Returns None normally, or the call_id
        when arguments are complete and the slot is finalized."""
        if call_id not in self._slots:
            self._slots[call_id] = {"name": name or "", "args_buf": ""}
        slot = self._slots[call_id]
        if name:
            slot["name"] = name
        if args_delta:
            slot["args_buf"] += args_delta
        return None

    def is_complete(self, call_id: str) -> bool:
        """Check if the call has a name and its args_buf is valid JSON."""
        slot = self._slots.get(call_id)
        if not slot:
            return False
        if not slot.get("name"):
            return False
        if not slot.get("args_buf"):
            return False
        try:
            json.loads(slot["args_buf"])
            return True
        except json.JSONDecodeError:
            return False

    def get_call(self, call_id: str) -> Optional[dict]:
        """Return the assembled call as {name, args} or None if incomplete."""
        slot = self._slots.get(call_id)
        if not slot or not slot.get("name"):
            return None
        try:
            args = json.loads(slot["args_buf"]) if slot["args_buf"] else {}
        except json.JSONDecodeError:
            return None
        return {"name": slot["name"], "args": args}

    def get_args_raw(self, call_id: str) -> str:
        slot = self._slots.get(call_id)
        return slot["args_buf"] if slot else ""

    def remove(self, call_id: str) -> None:
        self._slots.pop(call_id, None)

    def clear(self) -> None:
        self._slots.clear()


# Singleton instance shared across the request lifecycle
_tool_call_acc: Optional[ToolCallAccumulator] = None


def get_tool_call_accumulator() -> ToolCallAccumulator:
    global _tool_call_acc
    if _tool_call_acc is None:
        _tool_call_acc = ToolCallAccumulator()
    return _tool_call_acc


def reset_tool_call_accumulator() -> None:
    global _tool_call_acc
    if _tool_call_acc is not None:
        _tool_call_acc.clear()
    _tool_call_acc = ToolCallAccumulator()


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
