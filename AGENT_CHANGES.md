# Citrus QA Agent v8.3 — 全量代码变更对照

> 日期：2026-07-26  
> 修改文件：7 个（Python × 6，HTML × 1）  

---

## 1. `agent/src/core/progress_bus.py` — 进度事件总线

### 变更摘要
- `emit_progress` 统一 `data` 为 JSON 字符串（修复 SSE 格式不一致）
- 新增 `ToolCallAccumulator` 类（预留：流式 tool_call 参数拼接）
- 新增 SSE 原始帧调试日志 `log_sse_frame()`
- `mark_tool_start` 记录 `tool_name`（修复心跳名称丢失）
- `get_running_tools()` 返回 `(call_id, tool_name, elapsed)` 元组列表
- 新增 5 个结构化事件发射器：`emit_thinking`、`emit_tool_call_start`、`emit_tool_executing`、`emit_tool_result`、`emit_text`

### 完整当前代码
```python
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
# NOTE: ToolCallAccumulator is reserved for future streaming tool_call support.
# Currently unused (all LLM calls use ainvoke, not streaming).
# ═══════════════════════════════════════════════════════════════════════

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


# ═══════════════════════════════════════════════════════════════════════
# SSE Raw-Frame Debug Logger
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


# ═══════════════════════════════════════════════════════════════════════
# SSELogHandler — log lines → SSE log queue (optional, not used by default)
# ═══════════════════════════════════════════════════════════════════════

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

## 2. `agent/src/api/main.py` — SSE 端点

### 变更摘要
- 统一所有 event_queue 使用 `_encode_event()`（JSON 字符串化）
- 新增 `tool_heartbeat` 协程（每 2s 发送 tool_executing，含真实 tool_name）
- `process_graph` finally 排空队列（max 1s）后发送 None 哨兵
- SSE yield 前调用 `log_sse_frame()` 调试打印
- **不注册 SSELogHandler**（不再向前端发送后端日志）
- heartbeat 使用 `get_running_tools()` 的 `(id, name, elapsed)` 格式

### 关键代码段
```python
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
                            "message": f"{tool_name or '工具'} 执行中... ({elapsed:.0f}s)",
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

## 3. `agent/src/graph/light_graph.py` — Light 模式 Graph

### 变更摘要
- **删除** `emit_encoded("plan_start", {...})`（53-59 行）
- `citrus_rag_search.func()` → `await asyncio.to_thread(citrus_rag_search.func, ...)`（非阻塞）
- 使用结构化发射器 `emit_tool_call_start`、`emit_tool_executing`、`emit_tool_result`、`emit_text`、`emit_thinking`
- `max_tokens=1000000`（原 64000）

### 关键代码段 — load_context_node
```python
async def load_context_node(state: AgentState) -> dict:
    query = state.get("query", "")
    session_id = state.get("session_id", "default")
    mode = state.get("mode", "light")
    logger.info(f"[LightGraph:load] session={session_id[:8]}...")

    try:
        emit_status("step_active", step_id="load", message="加载会话上下文...")
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

### 关键代码段 — light_retrieve_node
```python
async def light_retrieve_node(state: AgentState) -> dict:
    # ...
    tool_call_id = str(uuid.uuid4())
    try:
        from src.tools.search import citrus_rag_search
        try:
            emit_tool_call_start("citrus_rag_search", {"query": best_q}, tool_call_id)
            emit_tool_executing("正在查询本地柑橘文献数据库...", "citrus_rag_search")
        except Exception:
            pass

        t_tool = time.perf_counter()
        content, artifact = await asyncio.to_thread(citrus_rag_search.func, best_q)  # ← 非阻塞
        dt_tool = (time.perf_counter() - t_tool) * 1000

        # ... process results ...
        emit_tool_result("citrus_rag_search", str(content)[:2000], tool_call_id,
                         summary=f"检索到 {len(main_results)} 篇文献, {len(web_results)} 个网页 ({dt_tool:.0f}ms)")
```

### 关键代码段 — light_synthesize_node
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
            emit_text(chunk.content)  # ← 替代旧的 emit_progress("token", ...)
```

---

## 4. `agent/src/graph/expert_graph.py` — Expert 模式 Graph

### 变更摘要
- **删除** `emit_encoded("plan_start", {...})`（270-278 行），改为在 `expert_load_node` 中发送
- `max_tokens=1000000`
- 结构化工具事件：`emit_tool_call_start`、`emit_tool_executing`、`emit_tool_result`
- `mark_tool_start(tc_id, "read_local_file")`（read_local_file 分支传入工具名）
- `emit_text(chunk)` 替代旧 `emit_progress("token", ...)`

### 关键代码段 — expert_load_node（新增 plan_start）
```python
async def expert_load_node(state: AgentState) -> dict:
    # ...
    result: dict = {}
    try:
        emit_encoded("plan_start", {
            "mode": "expert",
            "steps": [
                {"id": "load", "goal": "加载上下文与记忆", "icon": "load"},
                {"id": "retrieve", "goal": "检索柑橘文献", "icon": "search"},
                {"id": "supervise", "goal": "Supervisor 分析与调用子Agent", "icon": "brain"},
                {"id": "answer", "goal": "生成最终答案", "icon": "write"},
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

### 关键代码段 — supervisor_node 工具调用循环
```python
for tc in response.tool_calls:
    tc_dict = _make_tool_call(tc)
    tc_id = getattr(tc, "id", str(uuid.uuid4()))
    # ...
    try:
        emit_tool_call_start(tc_dict["name"], tc_dict.get("args", {}), tc_id)
        emit_tool_executing(f"正在执行 {tc_dict['name']}...", tc_dict["name"])
        if tc_dict["name"] in ("call_retrieve_agent", "call_write_agent", "call_analyze_agent"):
            # ... agent_switch ...
    except Exception:
        pass

    if tc_dict["name"] == "read_local_file":
        mark_tool_start(tc_id, "read_local_file")
        content = await read_local_file.ainvoke(tc_dict["args"])
        sub_result = {"agent": "read_local_file", "result": content or "", "artifacts": {}}
        emit_tool_result("read_local_file", str(content)[:2000], tc_id,
                         summary=f"读取完成 ({len(str(content))} 字符)")
    else:
        sub_result = await _execute_tool_call(tc_dict)
        emit_tool_result(tc_dict["name"], sub_result.get("result", "")[:2000], tc_id,
                         summary=(f"子Agent {sub_result.get('agent', '?')} 完成, "
                                  f"{sub_result.get('tools_called', 0)} 次工具调用"))
    # ...
```

### 关键代码段 — 最终答案发射
```python
if answer:
    emit_status("step_done", step_id="retrieve")
    emit_status("step_done", step_id="supervise")
    emit_status("step_active", step_id="answer")
    emit_status("final_answer", message="生成最终答案...")

    chunk_size = 8
    for i in range(0, len(answer), chunk_size):
        chunk = answer[i:i + chunk_size]
        emit_text(chunk)  # ← 替代旧 emit_progress("token", ...)
        await asyncio.sleep(0.012)
```

---

## 5. `agent/src/core/agent_runner.py` — 子 Agent 执行器

### 变更摘要
- `import uuid` 新增
- 使用结构化发射器替代旧 `emit_progress("status", ...)` 和 `emit_progress("token", ...)`
- **3 处 tc_id 保底修复**：`getattr(tc, "id", None) or str(uuid.uuid4())`
- 每个工具调用发送完整生命周期：`emit_tool_call_start` → `emit_tool_executing` → `emit_tool_result`
- `emit_agent_summary` 替代旧 `emit_progress("agent_summary", ...)`

### 关键代码段 — 工具调用循环
```python
# Emit structured tool events for each tool call
for tc in response.tool_calls:
    tc_dict = _make_tool_call_dict(tc)
    tc_id = getattr(tc, "id", None) or str(uuid.uuid4())  # ← 保底非空
    tc_name = tc_dict.get("name", "?")
    try:
        emit_tool_call_start(tc_name[:30], tc_dict.get("args", {}), tc_id)
        emit_tool_executing(f"子Agent {agent_name} 调用 {tc_name}...", tc_name[:30])
    except Exception:
        pass

t_tool = time.perf_counter()
try:
    tn = PartitionedToolNode(tools)
    tool_results = await tn.execute_tools(list(response.tool_calls))
except Exception as e:
    logger.error(f"[AgentRunner] {agent_name} tool exec failed: {e}")
    for tc in response.tool_calls:
        tc_id = getattr(tc, "id", None) or str(uuid.uuid4())  # ← 保底非空
        tc_name = getattr(tc, "name", "")
        try:
            emit_tool_result(tc_name, str(e)[:500], tc_id,
                             is_error=True, summary="工具执行失败")
        except Exception:
            pass
    break

dt_tool = (time.perf_counter() - t_tool) * 1000

for idx, tr in enumerate(tool_results):
    # ...
    tc = response.tool_calls[idx] if idx < len(response.tool_calls) else None
    tc_id = getattr(tc, "id", None) or str(uuid.uuid4()) if tc else str(uuid.uuid4())  # ← 保底非空
    tc_name = getattr(tc, "name", "") if tc else "tool"
    try:
        result_text = str(getattr(tr, "content", ""))[:3000]
        result_count = len(collected_artifacts["main_results"]) + len(collected_artifacts["web_results"])
        emit_tool_result(tc_name[:30], result_text, tc_id,
                         summary=f"完成 ({dt_tool:.0f}ms, {result_count} 条结果)")
    except Exception:
        pass
```

---

## 6. `agent/src/guardrails/memory.py` — 记忆存储

### 变更摘要
- `clear_session` 和 `_save_store` 两个方法中，在操作 `memory_store` 表之前新增 `CREATE TABLE IF NOT EXISTS memory_store (...)` 建表语句
- 修复 `no such table: memory_store` 错误

### 关键代码段 — clear_session
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
        logger.info(f"[Memory] 会话 {session_id[:8]}... 记忆已清空")
    except Exception as e:
        logger.error(f"[Memory] 清空记忆失败: {e}")
```

### 关键代码段 — _save_store
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
        logger.error(f"[Memory] 保存 {store_name} 失败: {e}")
```

---

## 7. `agent/index.html` — 前端 SPA

### 变更摘要（共修改约 30 处）

| 区域 | 变更 |
|------|------|
| CSS `.process-panel` | 删除 `display: none` |
| CSS `.step-tracker` | `display: flex` → `display: none` |
| CSS `.msg-part.nested-tool` | **新增** 缩进 + `└` 连接线样式 |
| JS 全局变量 | `_toolCardMap`、`_agentStack`（嵌套栈） |
| JS `createMsgPart()` | 简化工具卡片创建（删 `tool_executing`/`tool_result` 独立卡），`tool_call_start` 注册到 `_toolCardMap`，根据 `_agentStack.length` 加 `nested-tool` class |
| JS `_updateToolCard()` | `if (!card) return false`（修复 BUG：原返回 `true` 骗过调用方） |
| JS `_updateToolProgress()` | 只更新已存在的卡片，不创建新卡片（防止心跳孤儿卡） |
| JS `_finalizeToolCard()` | 更新已存在卡片为"完成"；找不到时 fallback 创建新卡 |
| JS `handleEvent()` | `tool_command_start/tool_executing/tool_result` 走生命周期；**首 text token 折叠全部工具卡**；`done` 清理僵尸 + 重置 `_agentStack` / `_toolCardMap` |
| JS `handleStatusEvent()` | 删除 `tool_start`/`tool_end` 死代码；全部 `addLogEntry` → `appendLogLine`（日志进折叠面板）；`agent_done` 弹栈 |
| JS SSE 解析器 | 双层 `try/catch`（handleEvent 异常不破坏流） |
| JS `addProcessCard()` | 修复重复 CSS class 拼接 |

### 关键代码 — handleEvent（完整）
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
        _updateToolProgress(data.tool_call_id || '', data.message || '执行中...', data.tool_name);
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
        appendLogLine('[Agent] ' + (data.agent_name || '') + ': ' + (data.doc_count || '?') + ' 条结果, ' + doiCount + ' DOI', 'INFO');
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
        appendLogLine('[引用] ' + (data.message || '引用异常'), 'WARNING');
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
          c.innerHTML = '<div style="font-size:11px;color:var(--text-muted);margin-bottom:5px">引用 (' + cited.length + '篇)</div>';
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
              ? '⚠ 余额不足（402）<br><span style="font-size:13px;font-weight:400;color:var(--text-secondary)">请检查 API Key 额度或稍后重试。</span>'
              : '⚠ 请求出错: ' + (data.message || '未知错误')) +
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
          runningBadges[rb].textContent = '完成';
          runningBadges[rb].className = 'msg-part-badge done';
        }
        _agentStack = [];
        _toolCardMap = {};
        /* ── [NEED_COMPLEX] escape */
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

### 关键代码 — handleStatusEvent（完整）
```javascript
function handleStatusEvent(data) {
    var stage = data.stage;
    if (!stage) { appendLogLine('未知状态: ' + JSON.stringify(data), 'WARNING'); return; }

    switch (stage) {
      case 'step_active':
        markStep(data.step_id, 'active');
        break;
      case 'step_done':
        markStep(data.step_id, 'done');
        break;
      case 'retrieving':
        statusDiv.innerHTML = '<div class="status"><div class="spinner"></div> 检索中...</div>';
        break;
      case 'retrieval_done':
        addProcessCard('tool_end', {tool: 'citrus_rag_search', result_len: data.main_count || 0});
        break;
      case 'react_fallback':
        appendLogLine('[ReAct] ' + (data.message || '启动 ReAct 搜索'), 'INFO');
        break;
      case 'react_thinking':
        appendLogLine('[ReAct] 第' + (data.turn||'?') + '轮推理', 'INFO');
        break;
      case 'agent_start':
        appendLogLine('[调度] 启动 ' + (data.agent || '?'), 'INFO');
        break;
      case 'agent_done':
        if (_agentStack.length > 0) _agentStack.pop();
        addProcessCard('agent_done', {agent_name: data.agent || '', tools_called: data.tools_called || 0});
        break;
      case 'final_answer':
        appendLogLine('[输出] 正在生成最终答案...', 'INFO');
        break;
      case 'loading':
        appendLogLine('[上下文] ' + (data.summary || ''), 'DEBUG');
        break;
      default:
        appendLogLine('[' + (stage || '') + '] ' + (data.message || ''), 'INFO');
        break;
    }
}
```

### 关键代码 — SSE 解析器（双层 try/catch）
```javascript
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

### 关键 CSS — 嵌套缩进
```css
/* ── Nested tool cards under agent ── */
.msg-part.nested-tool {
  margin-left: 20px;
  border-left: 2px solid var(--border-hover);
}
.msg-part.nested-tool::before {
  content: '\2514';
  position: absolute; left: -16px; top: 6px;
  color: var(--text-muted); font-size: 14px; font-family: monospace;
}

/* ── Step tracker hidden ── */
.step-tracker {
  display: none; ...
}

/* ── Process panel always visible ── */
.process-panel {
  margin: 8px 0;
  border: 1px solid var(--border-hairline);
  border-radius: 8px;
  overflow: hidden;
  background: var(--bg-elevated);
  /* display: none 已删除 */
}
```

---

## 验证清单

| 检查项 | 状态 |
|--------|:---:|
| Python 编译 | ✅ 6/6 文件通过 |
| JS 括号平衡 | ✅ 256/256 |
| JS 圆括号平衡 | ✅ 747/747 |
| CSS 语法 | ✅ 无冲突 |

### 重启命令
```bash
cd E:\codex_WORKSPACES\Citrus_QA_Agent\agent
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

### 浏览器验证
1. 打开 `http://localhost:8000`
2. **Ctrl+Shift+R** 强制刷新（必须！）
3. 发送一个需要检索的问题
4. 观察：工具卡片显示真实名称、执行过程可见、答案生成时自动折叠、完成后无残留卡片
5. 观察侧栏：文献引用和网络来源正常显示
6. 观察上下文面板：上限 1,000,000 tokens
