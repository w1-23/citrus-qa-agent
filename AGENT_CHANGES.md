# Citrus QA Agent v8.3 鈥?鍏ㄩ噺浠ｇ爜鍙樻洿瀵圭収

> 鏃ユ湡锛?026-07-26  
> 淇敼鏂囦欢锛? 涓紙Python 脳 6锛孒TML 脳 1锛? 

---

## 1. `agent/src/core/progress_bus.py` 鈥?杩涘害浜嬩欢鎬荤嚎

### 鍙樻洿鎽樿
- `emit_progress` 缁熶竴 `data` 涓?JSON 瀛楃涓诧紙淇 SSE 鏍煎紡涓嶄竴鑷达級
- 鏂板 `ToolCallAccumulator` 绫伙紙棰勭暀锛氭祦寮?tool_call 鍙傛暟鎷兼帴锛?- 鏂板 SSE 鍘熷甯ц皟璇曟棩蹇?`log_sse_frame()`
- `mark_tool_start` 璁板綍 `tool_name`锛堜慨澶嶅績璺冲悕绉颁涪澶憋級
- `get_running_tools()` 杩斿洖 `(call_id, tool_name, elapsed)` 鍏冪粍鍒楄〃
- 鏂板 5 涓粨鏋勫寲浜嬩欢鍙戝皠鍣細`emit_thinking`銆乣emit_tool_call_start`銆乣emit_tool_executing`銆乣emit_tool_result`銆乣emit_text`

### 瀹屾暣褰撳墠浠ｇ爜
```python
"""Progress bus 鈥?shared between graph nodes and SSE generator.

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
    """Alias for emit_progress 鈥?canonical event emission."""
    emit_progress(event_type, data)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# NOTE: ToolCallAccumulator is reserved for future streaming tool_call support.
# Currently unused (all LLM calls use ainvoke, not streaming).
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
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
        if call_id not in self._slots:
            self._slots[call_id] = {"name": name or "", "args_buf": ""}
        slot = self._slots[call_id]
        if name:
            slot["name"] = name
        if args_delta:
            slot["args_buf"] += args_delta
        return None

    def is_complete(self, call_id: str) -> bool:
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


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# SSE Raw-Frame Debug Logger
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
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
    logger.debug(f"[SSE鈫抅 event={event_type} | data={preview}")


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Tool Execution Heartbeat / Progress tracker
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
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


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# Structured event emitters (canonical type constants)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?

def emit_thinking(content: str) -> None:
    emit_encoded("thinking", {"content": content})


def emit_tool_call_start(tool_name: str, args: dict, tool_call_id: str = "") -> None:
    emit_encoded("tool_call_start", {
        "tool_name": tool_name,
        "args": args,
        "tool_call_id": tool_call_id,
    })
    if tool_call_id:
        mark_tool_start(tool_call_id, tool_name)


def emit_tool_executing(message: str, tool_name: str = "") -> None:
    emit_encoded("tool_executing", {
        "message": message,
        "tool_name": tool_name,
    })


def emit_tool_result(tool_name: str, output: str, tool_call_id: str = "",
                     is_error: bool = False, summary: str = "") -> None:
    if tool_call_id:
        mark_tool_end(tool_call_id)
    emit_encoded("tool_result", {
        "tool_name": tool_name,
        "output": output[:8000],
        "summary": summary[:500],
        "is_error": is_error,
        "tool_call_id": tool_call_id,
    })


def emit_text(content: str) -> None:
    emit_encoded("text", {"content": content})


def emit_status(stage: str, **kwargs) -> None:
    payload = {"stage": stage}
    payload.update(kwargs)
    emit_encoded("status", payload)


# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?# SSELogHandler 鈥?log lines 鈫?SSE log queue (optional, not used by default)
# 鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺愨晲鈺?
class SSELogHandler(logging.Handler):
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
```

---

## 2. `agent/src/api/main.py` 鈥?SSE 绔偣

### 鍙樻洿鎽樿
- 缁熶竴鎵€鏈?event_queue 浣跨敤 `_encode_event()`锛圝SON 瀛楃涓插寲锛?- 鏂板 `tool_heartbeat` 鍗忕▼锛堟瘡 2s 鍙戦€?tool_executing锛屽惈鐪熷疄 tool_name锛?- `process_graph` finally 鎺掔┖闃熷垪锛坢ax 1s锛夊悗鍙戦€?None 鍝ㄥ叺
- SSE yield 鍓嶈皟鐢?`log_sse_frame()` 璋冭瘯鎵撳嵃
- **涓嶆敞鍐?SSELogHandler**锛堜笉鍐嶅悜鍓嶇鍙戦€佸悗绔棩蹇楋級
- heartbeat 浣跨敤 `get_running_tools()` 鐨?`(id, name, elapsed)` 鏍煎紡

### 鍏抽敭浠ｇ爜娈?```python
async def event_generator():
    from src.core.progress_bus import (
        get_progress_queue, get_log_queue,
        reset_progress_queue, SSELogHandler,
        _encode_event, log_sse_frame,
        reset_tool_call_accumulator, get_tool_call_accumulator,
        get_running_tools, get_tool_elapsed, clear_tool_timers,
    )

    reset_progress_queue()
    reset_tool_call_accumulator()
    clear_tool_timers()
    progress_queue = get_progress_queue()
    log_queue = get_log_queue()
    event_queue: asyncio.Queue = asyncio.Queue()

    async def bridge_progress():
        while True:
            try:
                evt = await asyncio.wait_for(progress_queue.get(), timeout=0.3)
                await event_queue.put(evt)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def bridge_logs():
        while True:
            try:
                evt = await asyncio.wait_for(log_queue.get(), timeout=0.3)
                await event_queue.put(evt)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception:
                break

    async def tool_heartbeat():
        """Send progress events for tools that have been running > 2s."""
        try:
            while True:
                try:
                    await asyncio.sleep(2.0)
                except asyncio.CancelledError:
                    break
                running = get_running_tools()
                for call_id, tool_name, elapsed in running:
                    if elapsed > 2.0:
                        evt = _encode_event("tool_executing", {
                            "message": f"{tool_name or '宸ュ叿'} 鎵ц涓?.. ({elapsed:.0f}s)",
                            "tool_call_id": call_id,
                            "tool_name": tool_name,
                        })
                        await event_queue.put(evt)
        except asyncio.CancelledError:
            pass

    progress_task = asyncio.create_task(bridge_progress())
    log_task = asyncio.create_task(bridge_logs())
    heartbeat_task = asyncio.create_task(tool_heartbeat())

    async def process_graph():
        try:
            async for node_output in graph.astream(initial_state, stream_mode="updates"):
                for node_name, output in node_output.items():
                    if "_trace" in output:
                        trace = output["_trace"]
                        if trace.get("node") in ("load_context", "expert_load"):
                            await event_queue.put(_encode_event("status", {
                                "stage": "loading",
                                "summary": trace.get("summary", ""),
                            }))

                    if node_name in ("load_context", "expert_load", "save_context", "expert_save"):
                        continue

                    if node_name in ("light_retrieve",):
                        await event_queue.put(_encode_event("status", {
                            "stage": "retrieval_done",
                            "main_count": len(output.get("main_results", [])),
                        }))

                    if node_name in ("supervisor", "light_synthesize", "light_react"):
                        ans = output.get("answer", "")
                        if ans:
                            await asyncio.sleep(0.5)
                            await event_queue.put(_encode_event("done", {
                                "session_id": sid,
                                "answer": ans,
                                "gen_time_ms": int((time.perf_counter() - t0) * 1000),
                            }))
                    elif node_name not in ("light_retrieve",):
                        ans = output.get("answer", "")
                        if ans:
                            await event_queue.put(_encode_event("done", {
                                "session_id": sid,
                                "answer": ans,
                                "gen_time_ms": int((time.perf_counter() - t0) * 1000),
                            }))

                    ref_data = output.get("references_data")
                    if ref_data:
                        await event_queue.put(_encode_event("citations", ref_data))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"[SSE v2] graph processing error: {e}")
            try:
                await event_queue.put(_encode_event("error", {"message": str(e)}))
            except Exception:
                pass
        finally:
            # Drain pending bridge/log events before sending sentinel
            for _ in range(10):
                if progress_queue.empty() and log_queue.empty():
                    break
                await asyncio.sleep(0.1)
            heartbeat_task.cancel()
            await event_queue.put(None)

    graph_task = asyncio.create_task(process_graph())

    try:
        while True:
            try:
                event = await asyncio.wait_for(event_queue.get(), timeout=5.0)
            except asyncio.TimeoutError:
                hb = _encode_event("heartbeat", {})
                log_sse_frame(hb)
                yield hb
                continue

            if event is None:
                break
            log_sse_frame(event)
            yield event
    except asyncio.CancelledError:
        logger.warning(f"[SSE v2] Client disconnected (session={sid[:8]})")
        graph_task.cancel()
        try:
            await graph_task
        except Exception:
            pass
        raise
    finally:
        if not graph_task.done():
            graph_task.cancel()
        if not progress_task.done():
            progress_task.cancel()
        if not log_task.done():
            log_task.cancel()
        if not heartbeat_task.done():
            heartbeat_task.cancel()

    return EventSourceResponse(event_generator())
```

---

## 3. `agent/src/graph/light_graph.py` 鈥?Light 妯″紡 Graph

### 鍙樻洿鎽樿
- **鍒犻櫎** `emit_encoded("plan_start", {...})`锛?3-59 琛岋級
- `citrus_rag_search.func()` 鈫?`await asyncio.to_thread(citrus_rag_search.func, ...)`锛堥潪闃诲锛?- 浣跨敤缁撴瀯鍖栧彂灏勫櫒 `emit_tool_call_start`銆乣emit_tool_executing`銆乣emit_tool_result`銆乣emit_text`銆乣emit_thinking`
- `max_tokens=1000000`锛堝師 64000锛?
### 鍏抽敭浠ｇ爜娈?鈥?load_context_node
```python
async def load_context_node(state: AgentState) -> dict:
    query = state.get("query", "")
    session_id = state.get("session_id", "default")
    mode = state.get("mode", "light")
    logger.info(f"[LightGraph:load] session={session_id[:8]}...")

    try:
        emit_status("step_active", step_id="load", message="鍔犺浇浼氳瘽涓婁笅鏂?..")
    except Exception:
        pass
    # ... context loading code ...

    budget_config = ContextBudgetConfig(
        max_tokens=1000000,
        soft_threshold=0.85,
        hard_threshold=0.93,
    )
    # ...
```

### 鍏抽敭浠ｇ爜娈?鈥?light_retrieve_node
```python
async def light_retrieve_node(state: AgentState) -> dict:
    # ...
    tool_call_id = str(uuid.uuid4())
    try:
        from src.tools.search import citrus_rag_search
        try:
            emit_tool_call_start("citrus_rag_search", {"query": best_q}, tool_call_id)
            emit_tool_executing("姝ｅ湪鏌ヨ鏈湴鏌戞鏂囩尞鏁版嵁搴?..", "citrus_rag_search")
        except Exception:
            pass

        t_tool = time.perf_counter()
        content, artifact = await asyncio.to_thread(citrus_rag_search.func, best_q)  # 鈫?闈為樆濉?        dt_tool = (time.perf_counter() - t_tool) * 1000

        # ... process results ...
        emit_tool_result("citrus_rag_search", str(content)[:2000], tool_call_id,
                         summary=f"妫€绱㈠埌 {len(main_results)} 绡囨枃鐚? {len(web_results)} 涓綉椤?({dt_tool:.0f}ms)")
```

### 鍏抽敭浠ｇ爜娈?鈥?light_synthesize_node
```python
async def light_synthesize_node(state: AgentState) -> dict:
    # ...
    llm = _make_llm(timeout=120)
    async for chunk in llm.astream(messages):
        reasoning = ...
        if reasoning:
            emit_thinking(reasoning)
        if chunk.content:
            answer += chunk.content
            emit_text(chunk.content)  # 鈫?鏇夸唬鏃х殑 emit_progress("token", ...)
```

---

## 4. `agent/src/graph/expert_graph.py` 鈥?Expert 妯″紡 Graph

### 鍙樻洿鎽樿
- **鍒犻櫎** `emit_encoded("plan_start", {...})`锛?70-278 琛岋級锛屾敼涓哄湪 `expert_load_node` 涓彂閫?- `max_tokens=1000000`
- 缁撴瀯鍖栧伐鍏蜂簨浠讹細`emit_tool_call_start`銆乣emit_tool_executing`銆乣emit_tool_result`
- `mark_tool_start(tc_id, "read_local_file")`锛坮ead_local_file 鍒嗘敮浼犲叆宸ュ叿鍚嶏級
- `emit_text(chunk)` 鏇夸唬鏃?`emit_progress("token", ...)`

### 鍏抽敭浠ｇ爜娈?鈥?expert_load_node锛堟柊澧?plan_start锛?```python
async def expert_load_node(state: AgentState) -> dict:
    # ...
    result: dict = {}
    try:
        emit_encoded("plan_start", {
            "mode": "expert",
            "steps": [
                {"id": "load", "goal": "鍔犺浇涓婁笅鏂囦笌璁板繂", "icon": "load"},
                {"id": "retrieve", "goal": "妫€绱㈡煈姗樻枃鐚?, "icon": "search"},
                {"id": "supervise", "goal": "Supervisor 鍒嗘瀽涓庤皟鐢ㄥ瓙Agent", "icon": "brain"},
                {"id": "answer", "goal": "鐢熸垚鏈€缁堢瓟妗?, "icon": "write"},
            ],
        })
        from src.session.manager import session_manager
        # ...
    budget_config = ContextBudgetConfig(
        max_tokens=1000000,
        soft_threshold=0.85,
        hard_threshold=0.93,
    )
```

### 鍏抽敭浠ｇ爜娈?鈥?supervisor_node 宸ュ叿璋冪敤寰幆
```python
for tc in response.tool_calls:
    tc_dict = _make_tool_call(tc)
    tc_id = getattr(tc, "id", str(uuid.uuid4()))
    # ...
    try:
        emit_tool_call_start(tc_dict["name"], tc_dict.get("args", {}), tc_id)
        emit_tool_executing(f"姝ｅ湪鎵ц {tc_dict['name']}...", tc_dict["name"])
        if tc_dict["name"] in ("call_retrieve_agent", "call_write_agent", "call_analyze_agent"):
            # ... agent_switch ...
    except Exception:
        pass

    if tc_dict["name"] == "read_local_file":
        mark_tool_start(tc_id, "read_local_file")
        content = await read_local_file.ainvoke(tc_dict["args"])
        sub_result = {"agent": "read_local_file", "result": content or "", "artifacts": {}}
        emit_tool_result("read_local_file", str(content)[:2000], tc_id,
                         summary=f"璇诲彇瀹屾垚 ({len(str(content))} 瀛楃)")
    else:
        sub_result = await _execute_tool_call(tc_dict)
        emit_tool_result(tc_dict["name"], sub_result.get("result", "")[:2000], tc_id,
                         summary=(f"瀛怉gent {sub_result.get('agent', '?')} 瀹屾垚, "
                                  f"{sub_result.get('tools_called', 0)} 娆″伐鍏疯皟鐢?))
    # ...
```

### 鍏抽敭浠ｇ爜娈?鈥?鏈€缁堢瓟妗堝彂灏?```python
if answer:
    emit_status("step_done", step_id="retrieve")
    emit_status("step_done", step_id="supervise")
    emit_status("step_active", step_id="answer")
    emit_status("final_answer", message="鐢熸垚鏈€缁堢瓟妗?..")

    chunk_size = 8
    for i in range(0, len(answer), chunk_size):
        chunk = answer[i:i + chunk_size]
        emit_text(chunk)  # 鈫?鏇夸唬鏃?emit_progress("token", ...)
        await asyncio.sleep(0.012)
```

---

## 5. `agent/src/core/agent_runner.py` 鈥?瀛?Agent 鎵ц鍣?
### 鍙樻洿鎽樿
- `import uuid` 鏂板
- 浣跨敤缁撴瀯鍖栧彂灏勫櫒鏇夸唬鏃?`emit_progress("status", ...)` 鍜?`emit_progress("token", ...)`
- **3 澶?tc_id 淇濆簳淇**锛歚getattr(tc, "id", None) or str(uuid.uuid4())`
- 姣忎釜宸ュ叿璋冪敤鍙戦€佸畬鏁寸敓鍛藉懆鏈燂細`emit_tool_call_start` 鈫?`emit_tool_executing` 鈫?`emit_tool_result`
- `emit_agent_summary` 鏇夸唬鏃?`emit_progress("agent_summary", ...)`

### 鍏抽敭浠ｇ爜娈?鈥?宸ュ叿璋冪敤寰幆
```python
# Emit structured tool events for each tool call
for tc in response.tool_calls:
    tc_dict = _make_tool_call_dict(tc)
    tc_id = getattr(tc, "id", None) or str(uuid.uuid4())  # 鈫?淇濆簳闈炵┖
    tc_name = tc_dict.get("name", "?")
    try:
        emit_tool_call_start(tc_name[:30], tc_dict.get("args", {}), tc_id)
        emit_tool_executing(f"瀛怉gent {agent_name} 璋冪敤 {tc_name}...", tc_name[:30])
    except Exception:
        pass

t_tool = time.perf_counter()
try:
    tn = PartitionedToolNode(tools)
    tool_results = await tn.execute_tools(list(response.tool_calls))
except Exception as e:
    logger.error(f"[AgentRunner] {agent_name} tool exec failed: {e}")
    for tc in response.tool_calls:
        tc_id = getattr(tc, "id", None) or str(uuid.uuid4())  # 鈫?淇濆簳闈炵┖
        tc_name = getattr(tc, "name", "")
        try:
            emit_tool_result(tc_name, str(e)[:500], tc_id,
                             is_error=True, summary="宸ュ叿鎵ц澶辫触")
        except Exception:
            pass
    break

dt_tool = (time.perf_counter() - t_tool) * 1000

for idx, tr in enumerate(tool_results):
    # ...
    tc = response.tool_calls[idx] if idx < len(response.tool_calls) else None
    tc_id = getattr(tc, "id", None) or str(uuid.uuid4()) if tc else str(uuid.uuid4())  # 鈫?淇濆簳闈炵┖
    tc_name = getattr(tc, "name", "") if tc else "tool"
    try:
        result_text = str(getattr(tr, "content", ""))[:3000]
        result_count = len(collected_artifacts["main_results"]) + len(collected_artifacts["web_results"])
        emit_tool_result(tc_name[:30], result_text, tc_id,
                         summary=f"瀹屾垚 ({dt_tool:.0f}ms, {result_count} 鏉＄粨鏋?")
    except Exception:
        pass
```

---

## 6. `agent/src/guardrails/memory.py` 鈥?璁板繂瀛樺偍

### 鍙樻洿鎽樿
- `clear_session` 鍜?`_save_store` 涓や釜鏂规硶涓紝鍦ㄦ搷浣?`memory_store` 琛ㄤ箣鍓嶆柊澧?`CREATE TABLE IF NOT EXISTS memory_store (...)` 寤鸿〃璇彞
- 淇 `no such table: memory_store` 閿欒

### 鍏抽敭浠ｇ爜娈?鈥?clear_session
```python
def clear_session(self, session_id: str) -> None:
    try:
        import sqlite3
        from pathlib import Path
        from src.config import PROJECT_ROOT
        db_path = PROJECT_ROOT / "state" / "sessions.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_store ("
                "session_id TEXT NOT NULL, "
                "key TEXT NOT NULL, "
                "value TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, "
                "PRIMARY KEY (session_id, key))"
            )
            conn.execute("DELETE FROM memory_store WHERE session_id = ?", (session_id,))
            conn.commit()
        logger.info(f"[Memory] 浼氳瘽 {session_id[:8]}... 璁板繂宸叉竻绌?)
    except Exception as e:
        logger.error(f"[Memory] 娓呯┖璁板繂澶辫触: {e}")
```

### 鍏抽敭浠ｇ爜娈?鈥?_save_store
```python
def _save_store(self, session_id: str, store_name: str, data: dict) -> None:
    try:
        import sqlite3
        from pathlib import Path
        from src.config import PROJECT_ROOT
        db_path = PROJECT_ROOT / "state" / "sessions.db"
        with sqlite3.connect(str(db_path)) as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_store ("
                "session_id TEXT NOT NULL, "
                "key TEXT NOT NULL, "
                "value TEXT NOT NULL, "
                "updated_at TEXT NOT NULL, "
                "PRIMARY KEY (session_id, key))"
            )
            conn.execute(
                """INSERT OR REPLACE INTO memory_store ...""",
                (...),
            )
            conn.commit()
    except Exception as e:
        logger.error(f"[Memory] 淇濆瓨 {store_name} 澶辫触: {e}")
```

---

## 7. `agent/index.html` 鈥?鍓嶇 SPA

### 鍙樻洿鎽樿锛堝叡淇敼绾?30 澶勶級

| 鍖哄煙 | 鍙樻洿 |
|------|------|
| CSS `.process-panel` | 鍒犻櫎 `display: none` |
| CSS `.step-tracker` | `display: flex` 鈫?`display: none` |
| CSS `.msg-part.nested-tool` | **鏂板** 缂╄繘 + `鈹擿 杩炴帴绾挎牱寮?|
| JS 鍏ㄥ眬鍙橀噺 | `_toolCardMap`銆乣_agentStack`锛堝祵濂楁爤锛?|
| JS `createMsgPart()` | 绠€鍖栧伐鍏峰崱鐗囧垱寤猴紙鍒?`tool_executing`/`tool_result` 鐙珛鍗★級锛宍tool_call_start` 娉ㄥ唽鍒?`_toolCardMap`锛屾牴鎹?`_agentStack.length` 鍔?`nested-tool` class |
| JS `_updateToolCard()` | `if (!card) return false`锛堜慨澶?BUG锛氬師杩斿洖 `true` 楠楄繃璋冪敤鏂癸級 |
| JS `_updateToolProgress()` | 鍙洿鏂板凡瀛樺湪鐨勫崱鐗囷紝涓嶅垱寤烘柊鍗＄墖锛堥槻姝㈠績璺冲鍎垮崱锛?|
| JS `_finalizeToolCard()` | 鏇存柊宸插瓨鍦ㄥ崱鐗囦负"瀹屾垚"锛涙壘涓嶅埌鏃?fallback 鍒涘缓鏂板崱 |
| JS `handleEvent()` | `tool_command_start/tool_executing/tool_result` 璧扮敓鍛藉懆鏈燂紱**棣?text token 鎶樺彔鍏ㄩ儴宸ュ叿鍗?*锛沗done` 娓呯悊鍍靛案 + 閲嶇疆 `_agentStack` / `_toolCardMap` |
| JS `handleStatusEvent()` | 鍒犻櫎 `tool_start`/`tool_end` 姝讳唬鐮侊紱鍏ㄩ儴 `addLogEntry` 鈫?`appendLogLine`锛堟棩蹇楄繘鎶樺彔闈㈡澘锛夛紱`agent_done` 寮规爤 |
| JS SSE 瑙ｆ瀽鍣?| 鍙屽眰 `try/catch`锛坔andleEvent 寮傚父涓嶇牬鍧忔祦锛?|
| JS `addProcessCard()` | 淇閲嶅 CSS class 鎷兼帴 |

### 鍏抽敭浠ｇ爜 鈥?handleEvent锛堝畬鏁达級
```javascript
function handleEvent(type, data) {
    switch (type) {
      case 'thinking':
        if (!isAnswerStarted) {
          if (!_thinkingCardId) {
            _thinkingCardId = createMsgPart('thinking', data);
          } else {
            _appendThinking(data.content || '');
          }
        }
        break;

      case 'tool_call_start':
        createMsgPart('tool_call_start', data);
        break;

      case 'tool_executing':
        _updateToolProgress(data.tool_call_id || '', data.message || '鎵ц涓?..', data.tool_name);
        break;

      case 'tool_result':
        _finalizeToolCard(data.tool_call_id || '', data);
        break;

      case 'text':
        if (!isAnswerStarted) {
          isAnswerStarted = true;
          clearTimeout(spinnerTimeout);
          statusDiv.innerHTML = '';
          answerDiv.style.display = '';
          /* collapse all tool cards when answer starts */
          var allCards = processCards.querySelectorAll('.msg-part.tool-start-part.expanded');
          for (var ci = 0; ci < allCards.length; ci++) {
            allCards[ci].classList.remove('expanded');
          }
        }
        _answerStreamText += (data.content || '');
        fullText = _answerStreamText;
        renderAnswer();
        break;

      case 'context_status':
        renderContextPanel(data);
        break;

      case 'status':
        handleStatusEvent(data);
        break;

      case 'agent_switch':
        var aname = (data.agent_name || '').substring(0, 20);
        if (aname) _agentStack.push(aname);
        addProcessCard('agent_switch', data);
        break;

      case 'agent_summary':
        var doiCount = (data.doi_list || []).length;
        appendLogLine('[Agent] ' + (data.agent_name || '') + ': ' + (data.doc_count || '?') + ' 鏉＄粨鏋? ' + doiCount + ' DOI', 'INFO');
        break;

      case 'compression':
        if (_ctxData) {
          _ctxData.compressed = true;
          _ctxData.compression_len = (data.len || 0);
          renderContextPanel(_ctxData);
        }
        break;

      case 'context_usage':
        updateContextTokens(data.estimated_tokens || 0);
        break;

      case 'token':
        /* backward compat: legacy token events from post-hoc chunking */
        if (!isAnswerStarted) {
          isAnswerStarted = true;
          clearTimeout(spinnerTimeout);
          statusDiv.innerHTML = '';
          answerDiv.style.display = '';
        }
        fullText += data.content;
        renderAnswer();
        break;

      case 'citation_warning':
        appendLogLine('[寮曠敤] ' + (data.message || '寮曠敤寮傚父'), 'WARNING');
        break;

      case 'citations':
        roundHistory.push({
          roundId: currentRoundId, query: currentQuery,
          cited: data.cited || [], uncited: data.uncited || [], historical: data.historical || []
        });
        renderRoundSidebar();
        renderUrlSidebar(data.cited || [], data.uncited || []);
        citationsDiv.style.display = '';
        var cited = data.cited || [];
        if (cited.length > 0) {
          var c = document.createElement('div');
          c.style.cssText = 'margin-top:10px;padding-top:10px;border-top:1px solid var(--border-hairline)';
          c.innerHTML = '<div style="font-size:11px;color:var(--text-muted);margin-bottom:5px">寮曠敤 (' + cited.length + '绡?</div>';
          for (var k = 0; k < cited.length; k++) {
            c.innerHTML += '<span style="display:inline-block;font-family:var(--font-mono);font-size:10px;background:var(--success-glow);color:var(--success);padding:1px 5px;border-radius:3px;margin:2px;font-weight:600">[' + cited[k].ref_id + ']</span>';
          }
          citationsDiv.appendChild(c);
        }
        break;

      case 'error':
        clearTimeout(spinnerTimeout);
        isCancelled = true;
        setSendDisabled(false);
        userInput.focus();
        if (!fullText && !_answerStreamText) {
          statusDiv.innerHTML = '';
          answerDiv.style.display = '';
          answerDiv.innerHTML = '<div style="color:var(--error);font-weight:500">' +
            (data.code === 402
              ? '鈿?浣欓涓嶈冻锛?02锛?br><span style="font-size:13px;font-weight:400;color:var(--text-secondary)">璇锋鏌?API Key 棰濆害鎴栫◢鍚庨噸璇曘€?/span>'
              : '鈿?璇锋眰鍑洪敊: ' + (data.message || '鏈煡閿欒')) +
            '</div>';
        }
        break;

      case 'done':
        clearTimeout(spinnerTimeout);
        isCancelled = true;
        answerDiv.style.display = '';
        statusDiv.innerHTML = '';
        var hasContent = fullText || _answerStreamText || (data.answer || '');
        if (hasContent) {
          for (var si = 0; si < _processSteps.length; si++) {
            markStep(_processSteps[si].id, 'done');
          }
        }
        for (var wi = 0; wi < workflowNodeData.length; wi++) {
          if (workflowNodeData[wi].status !== 'done') setWorkflowNodeStatus(wi, 'done');
        }
        if (!fullText && !_answerStreamText && data.answer) {
          fullText = data.answer;
          _answerStreamText = data.answer;
        }
        var finalAnswer = fullText || _answerStreamText;
        if (finalAnswer && answerDiv) {
          answerDiv.innerHTML = marked.parse(finalAnswer.replace(/__NEED_COMPLEX__/g, ''));
        }
        /* clean up any lingering tool cards */
        var runningBadges = processCards.querySelectorAll('.msg-part-badge.running');
        for (var rb = 0; rb < runningBadges.length; rb++) {
          runningBadges[rb].textContent = '瀹屾垚';
          runningBadges[rb].className = 'msg-part-badge done';
        }
        _agentStack = [];
        _toolCardMap = {};
        /* 鈹€鈹€ [NEED_COMPLEX] escape */
        var needComplex = (fullText || _answerStreamText || '').indexOf('__NEED_COMPLEX__') >= 0;
        // ...
        break;

      case 'heartbeat':
        break;

      default:
        if (data && typeof data === 'object') {
          createMsgPart(type, data);
        } else {
          console.warn('[SSE] unknown event type:', type, data);
          createMsgPart('fallback', { output: JSON.stringify(data) });
        }
        break;
    }
}
```

### 鍏抽敭浠ｇ爜 鈥?handleStatusEvent锛堝畬鏁达級
```javascript
function handleStatusEvent(data) {
    var stage = data.stage;
    if (!stage) { appendLogLine('鏈煡鐘舵€? ' + JSON.stringify(data), 'WARNING'); return; }

    switch (stage) {
      case 'step_active':
        markStep(data.step_id, 'active');
        break;
      case 'step_done':
        markStep(data.step_id, 'done');
        break;
      case 'retrieving':
        statusDiv.innerHTML = '<div class="status"><div class="spinner"></div> 妫€绱腑...</div>';
        break;
      case 'retrieval_done':
        addProcessCard('tool_end', {tool: 'citrus_rag_search', result_len: data.main_count || 0});
        break;
      case 'react_fallback':
        appendLogLine('[ReAct] ' + (data.message || '鍚姩 ReAct 鎼滅储'), 'INFO');
        break;
      case 'react_thinking':
        appendLogLine('[ReAct] 绗? + (data.turn||'?') + '杞帹鐞?, 'INFO');
        break;
      case 'agent_start':
        appendLogLine('[璋冨害] 鍚姩 ' + (data.agent || '?'), 'INFO');
        break;
      case 'agent_done':
        if (_agentStack.length > 0) _agentStack.pop();
        addProcessCard('agent_done', {agent_name: data.agent || '', tools_called: data.tools_called || 0});
        break;
      case 'final_answer':
        appendLogLine('[杈撳嚭] 姝ｅ湪鐢熸垚鏈€缁堢瓟妗?..', 'INFO');
        break;
      case 'loading':
        appendLogLine('[涓婁笅鏂嘳 ' + (data.summary || ''), 'DEBUG');
        break;
      default:
        appendLogLine('[' + (stage || '') + '] ' + (data.message || ''), 'INFO');
        break;
    }
}
```

### 鍏抽敭浠ｇ爜 鈥?SSE 瑙ｆ瀽鍣紙鍙屽眰 try/catch锛?```javascript
    var reader = response.body.getReader();
    var decoder = new TextDecoder();
    var buffer = '';
    var currentEvent = 'message';
    while (true) {
      var chunk = await reader.read();
      if (chunk.done) break;
      buffer += decoder.decode(chunk.value, { stream: true });
      var lines = buffer.split('\n');
      buffer = lines.pop();
      for (var j = 0; j < lines.length; j++) {
        var line = lines[j];
        if (line.startsWith('event: ')) {
          currentEvent = line.substring(7).trim();
        } else if (line.startsWith('data: ')) {
          var rawData = line.substring(6);
          try {
            var parsedData = JSON.parse(rawData);
            try {
              handleEvent(currentEvent, parsedData);
            } catch (hErr) {
              console.error('handleEvent error:', currentEvent, hErr);
              try { handleEvent('fallback', { event: currentEvent, error: String(hErr) }); } catch (_) {}
            }
          } catch (e) {
            console.error('SSE parse error:', rawData.substring(0, 200));
            try { handleEvent('fallback', { raw: rawData.substring(0, 1000), error: String(e) }); } catch (_) {}
          }
        } else if (line.trim() === '') {
          currentEvent = 'message';
        }
      }
    }
```

### 鍏抽敭 CSS 鈥?宓屽缂╄繘
```css
/* 鈹€鈹€ Nested tool cards under agent 鈹€鈹€ */
.msg-part.nested-tool {
  margin-left: 20px;
  border-left: 2px solid var(--border-hover);
}
.msg-part.nested-tool::before {
  content: '\2514';
  position: absolute; left: -16px; top: 6px;
  color: var(--text-muted); font-size: 14px; font-family: monospace;
}

/* 鈹€鈹€ Step tracker hidden 鈹€鈹€ */
.step-tracker {
  display: none; ...
}

/* 鈹€鈹€ Process panel always visible 鈹€鈹€ */
.process-panel {
  margin: 8px 0;
  border: 1px solid var(--border-hairline);
  border-radius: 8px;
  overflow: hidden;
  background: var(--bg-elevated);
  /* display: none 宸插垹闄?*/
}
```

---

## 楠岃瘉娓呭崟

| 妫€鏌ラ」 | 鐘舵€?|
|--------|:---:|
| Python 缂栬瘧 | 鉁?6/6 鏂囦欢閫氳繃 |
| JS 鎷彿骞宠　 | 鉁?256/256 |
| JS 鍦嗘嫭鍙峰钩琛?| 鉁?747/747 |
| CSS 璇硶 | 鉁?鏃犲啿绐?|

### 閲嶅惎鍛戒护
```bash
cd E:\codex_WORKSPACES\Citrus_QA_Agent\agent
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 娴忚鍣ㄩ獙璇?1. 鎵撳紑 `http://localhost:8000`
2. **Ctrl+Shift+R** 寮哄埗鍒锋柊锛堝繀椤伙紒锛?3. 鍙戦€佷竴涓渶瑕佹绱㈢殑闂
4. 瑙傚療锛氬伐鍏峰崱鐗囨樉绀虹湡瀹炲悕绉般€佹墽琛岃繃绋嬪彲瑙併€佺瓟妗堢敓鎴愭椂鑷姩鎶樺彔銆佸畬鎴愬悗鏃犳畫鐣欏崱鐗?5. 瑙傚療渚ф爮锛氭枃鐚紩鐢ㄥ拰缃戠粶鏉ユ簮姝ｅ父鏄剧ず
6. 瑙傚療涓婁笅鏂囬潰鏉匡細涓婇檺 1,000,000 tokens
