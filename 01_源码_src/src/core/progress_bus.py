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
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

# 全局兜底队列（非 SSE 上下文，如 CLI/测试）；SSE 请求使用 per-request 队列
_progress_queue: asyncio.Queue | None = None

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


# ── v10.5（F8）: SSE/JSONL 行分隔符消毒 ────────────────────────────────────
#   SSE 的 data 段是**单行 JSON**，任何"被解析方当作行分隔符"的字符都会把这条事件
#   劈成两半。Python 的 str.splitlines()（requests.iter_lines 用的就是它）把
#   \x0b \x0c \x1c \x1d \x1e \x85 \u2028 \u2029 全部算作行边界；而 **F7 之后
#   citations 载荷里带着联网网页正文**，真实网页文本里 U+2028/U+0085 并不罕见。
#   实测代价（webcontent_v10_8 的 C48-time-004）：
#     · 抓取解析器丢掉整条 citations 事件 → 该题被判成"回执 0 条、正文 20 个标记
#       全越界"（假缺陷）；
#     · 落盘的 answers.jsonl 因 U+2028 被 read_text().splitlines() 切开，
#       **整个 JSONL 变成不可解析**（连指标脚本都读不进来）。
#   浏览器端只按 '\n' 切分故不受影响，但"判分数据必须能被任何 JSONL 读者解析"
#   是硬要求。这里在**唯一出网点**统一消毒，覆盖 answer / citations / status 全部事件。
_LINE_BREAKERS = {0x0B: " ", 0x0C: " ", 0x1C: " ", 0x1D: " ", 0x1E: " ",
                  0x85: " ", 0x2028: " ", 0x2029: " ", 0x0000: ""}


def _sanitize_wire(obj):
    """递归把"行分隔符类"字符替换成空格（仅文本，不改结构、不改语义）。"""
    if isinstance(obj, str):
        if not any(ord(ch) in _LINE_BREAKERS for ch in obj):
            return obj
        return "".join(_LINE_BREAKERS.get(ord(ch), ch) for ch in obj)
    if isinstance(obj, dict):
        return {k: _sanitize_wire(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize_wire(v) for v in obj]
    return obj


def _encode_event(event_type: str, data: dict) -> dict:
    """Encode an SSE event dict with JSON-string data field (canonical format).

    v10.5（F8）: 出网前统一消毒行分隔符，保证「一条事件 = 一行」（见上）。
    """
    return {"event": event_type,
            "data": json.dumps(_sanitize_wire(data), ensure_ascii=False)}


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
                     output_tokens: int = 0, total: int = 0,
                     cache_hit: int = 0, cache_miss: int = 0) -> None:
    """context_usage 增量推送 (v8.3.3): total 为单次调用累计，delta 为相对上次的增量。

    cache_hit/cache_miss 为 DeepSeek prompt_cache 命中/未命中 token
    （阶段0 缓存观测，见 core/cache_metrics.py）。
    """
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
    cache_denom = cache_hit + cache_miss
    # v8.13: 语义分家——此前单次调用绝对值与 session 累计值平铺混在一起
    # （input_tokens 是单次、total 是累计、cache 又是单次），消费端无法区分。
    # 改为: call.* = 本次调用绝对值；total/delta = session+source 累计与增量。
    emit_encoded("context_usage", {
        "call": {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cache_hit": cache_hit,
            "cache_miss": cache_miss,
            "cache_ratio": round(cache_hit / cache_denom, 4) if cache_denom else 0,
        },
        "total": total,
        "delta": delta,
        "source": source,
    })


# ═══════════════════════════════════════════════════════════════════════
# SSE Raw-Frame Debug Logger (prints every outgoing SSE chunk)
# ═══════════════════════════════════════════════════════════════════════

# v9.2: 删除恒 True 的开关 _sse_debug_enabled——守卫分支永不生效，
# log_sse_frame 随时可调用（debug 级别输出本身已是可关的开关）

def log_sse_frame(event_dict: dict) -> None:
    """Print an outgoing SSE frame for debugging purposes."""
    event_type = event_dict.get("event", "?")
    data_str = event_dict.get("data", "")
    preview = data_str[:200] if len(data_str) > 200 else data_str
    logger.debug(f"[SSE→] event={event_type} | data={preview}")


# ═══════════════════════════════════════════════════════════════════════
# Tool Execution Heartbeat / Progress tracker
# ═══════════════════════════════════════════════════════════════════════

_tool_start_times_var: contextvars.ContextVar = contextvars.ContextVar(
    "tool_start_times", default=None)


def _tool_start_times() -> dict[str, tuple[float, str]]:
    timers = _tool_start_times_var.get()
    if timers is None:
        timers = {}
        _tool_start_times_var.set(timers)
    return timers


def mark_tool_start(tool_call_id: str, tool_name: str = "") -> None:
    _tool_start_times()[tool_call_id] = (time.perf_counter(), tool_name)


def mark_tool_end(tool_call_id: str) -> None:
    _tool_start_times().pop(tool_call_id, None)


# v9.2: 删除 get_tool_elapsed——grep 证实全仓库无调用方（get_running_tools
# 已提供聚合视图，单个工具耗时可直接从 get_running_tools 结果推导）


def get_running_tools() -> list[tuple[str, str, float]]:
    """Returns list of (call_id, tool_name, elapsed_seconds)."""
    now = time.perf_counter()
    return [(cid, name, now - ts) for cid, (ts, name) in _tool_start_times().items()]


def clear_tool_timers() -> None:
    _tool_start_times().clear()


# ═══════════════════════════════════════════════════════════════════════
# Structured event emitters (canonical type constants)
# ═══════════════════════════════════════════════════════════════════════


def emit_thinking(content: str) -> None:
    """Emit a safe progress label; never forward model-generated reasoning text."""
    emit_status("agent_progress", message="正在规划下一步工具操作…")


def emit_reasoning(content: str) -> None:
    """Deprecated: private model reasoning must never be sent to clients."""
    return


_TOOL_LABELS = {
    "deepseek_search": "Web Search",
    "deepseek_fetch": "Web Fetch",
    "deepseek_fetch_paper": "Web Fetch (paper full text)",
    "local_bm25_search": "本地关键词核对",
    "local_dense_search": "本地语义检索",
    "local_hybrid_search": "本地混合检索",
    "draw_svg": "draw",
    "citrus_rag_search": "本地语义检索",
    "pdf_read": "读取 PDF",
    "read_local_file": "读取文件",
    "write_local_file": "写入文件",
    "statistical_analysis": "统计分析",
    "experimental_design": "实验设计",
    "inspect_uploaded_table": "读取表格结构",
    "audit_uploaded_table": "审查表格数据",
    "audit_uploaded_data_bundle": "跨表数据审查",
    "verify_uploaded_references": "核验文献元数据",
    "fetch_and_audit_open_supplements": "获取并审查公开补充数据",
    "call_write_agent": "写作助手",
    "call_analyze_agent": "分析助手",
}


def _safe_tool_summary(tool_name: str, args: dict | None = None,
                       output: str = "") -> str:
    """Build a compact, non-secret execution label for progress/history UI."""
    args = args if isinstance(args, dict) else {}
    label = _TOOL_LABELS.get(tool_name, tool_name.replace("_", " ")[:48] or "工具")
    if tool_name in {"deepseek_search", "local_bm25_search", "local_dense_search", "local_hybrid_search", "citrus_rag_search"}:
        detail = str(args.get("query") or "")
        detail = " ".join(detail.split())[:180]
        families = args.get("corpus_types") or []
        if isinstance(families, str):
            families = [families]
        if families and tool_name.startswith("local_"):
            detail = f"[{', '.join(str(x) for x in families)}] {detail}"
        return f"{label}：{detail}" if detail else label
    if tool_name == "deepseek_fetch":
        detail = str(args.get("url") or "")[:200]
        return f"{label}：{detail}" if detail else label
    if tool_name == "deepseek_fetch_paper":
        detail = str(args.get("doi") or args.get("pmid") or args.get("title") or "")[:160]
        return f"{label}：{detail}" if detail else label
    if tool_name in {"read_local_file", "pdf_read"}:
        detail = str(args.get("path") or args.get("file_path") or "")
        detail = detail.replace("\\", "/").rsplit("/", 1)[-1][:100]
        return f"{label}：{detail}" if detail else label
    if output:
        text = " ".join(str(output).split())
        if text.startswith(("[WEB_TOOL_ERROR]", "[LOCAL_SEARCH_ERROR]", "[TOOL_ERROR]")):
            return f"{label}：{text[:180]}"
        match = re.search(r"Search returned (\d+) result", text, re.I)
        if match:
            return f"{label}：找到 {match.group(1)} 条结果"
        if text.startswith("Fetched page:"):
            return text.split("\n", 1)[0][:180]
        return f"{label}：{text[:150]}"
    return label


def emit_tool_call_start(tool_name: str, args: dict, tool_call_id: str = "") -> None:
    """Emit a sanitized, user-auditable TOOL_CALL_START event."""
    args = args if isinstance(args, dict) else {}
    safe_args = {}
    for key in ("query", "url", "count", "limit", "need_content", "task",
                "corpus_types", "l1_categories"):
        value = args.get(key)
        if isinstance(value, (str, int, float, bool)):
            safe_args[key] = str(value)[:240] if isinstance(value, str) else value
        elif isinstance(value, list) and key in {"corpus_types", "l1_categories"}:
            safe_args[key] = [str(item)[:80] for item in value[:8]]
    summary = _safe_tool_summary(tool_name, safe_args)
    emit_encoded("tool_call_start", {
        "tool_name": tool_name,
        "display_name": _TOOL_LABELS.get(tool_name, tool_name),
        "args": safe_args,
        "summary": summary,
        "tool_call_id": tool_call_id,
    })
    if tool_call_id:
        mark_tool_start(tool_call_id, tool_name)


def emit_tool_executing(message: str, tool_name: str = "", tool_call_id: str = "") -> None:
    """Emit a TOOL_EXECUTING event (progress / heartbeat during tool run)."""
    display_name = _TOOL_LABELS.get(tool_name, tool_name.replace("_", " ")[:48] or "工具")
    if tool_name and message.startswith(tool_name):
        message = display_name + message[len(tool_name):]
    emit_encoded("tool_executing", {
        "message": message,
        "tool_name": tool_name,
        "display_name": display_name,
        "tool_call_id": tool_call_id,
    })


def emit_tool_result(tool_name: str, output: str, tool_call_id: str = "",
                     is_error: bool = False, summary: str = "") -> None:
    """Emit a TOOL_RESULT event with output content and optional summary."""
    if tool_call_id:
        mark_tool_end(tool_call_id)
    is_error = bool(is_error or str(output).startswith((
        "[WEB_TOOL_ERROR]", "[WEB_DISABLED]", "[WEB_FETCH_EMPTY]",
        "[LOCAL_SEARCH_ERROR]", "[TOOL_ERROR]", "[ERR_REQUEST_DEADLINE]")))
    summary = summary or _safe_tool_summary(tool_name, output=output)
    emit_encoded("tool_result", {
        "tool_name": tool_name,
        "output": output[:6000],
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
