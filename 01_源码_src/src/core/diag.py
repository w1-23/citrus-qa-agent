"""Diag — 结构化诊断事件层（v8.13）。

目标：一次真实测试后，可程序化重建"请求全链路发生了什么"（延迟/决策/token 快照）：
  - JSONL 落盘: logs/diag/diag_YYYY-MM-DD.jsonl（每行一个事件，可 grep/jq/脚本聚合）
  - 同时写 agent.log（INFO，经 req= 过滤器串线，人工可读）
  - 自动注入 req/session/job 关联字段（复用 core/tracing.py 的 contextvars，无新机制）
  - diag_span 上下文管理器：进入打 <event>_start，正常退出打 <event>_done(+dur_ms)，
    异常打 <event>_error（err/msg/dur_ms）后原样抛出

原则：永不抛异常；事件量级小（每请求几十~几百条）；per-date 句柄缓存 + 线程锁；
日志目录跟随 CITRUS_LOG_DIR 覆盖（与 agent.log/business.log 同口径，测试独立 sink）。
"""
from __future__ import annotations

import contextlib
import json
import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_file_state: dict = {"date": "", "fh": None}
_MAX_STR = 500  # 单个字段截断上限（诊断事件不承载正文，防超长污染）


def _diag_dir() -> Path:
    override = os.environ.get("CITRUS_LOG_DIR", "").strip()
    if override:
        return Path(override) / "diag"
    from src.config import PROJECT_ROOT
    return PROJECT_ROOT / "logs" / "diag"


def _correlation() -> dict:
    """从 tracing contextvars 取请求关联字段（取不到时兜底，不报错）。"""
    try:
        from src.core.tracing import get_job_id, get_request_id, get_session_id
        return {"req": get_request_id() or "-", "session": get_session_id() or "",
                "job": get_job_id() or ""}
    except Exception:
        return {"req": "-", "session": "", "job": ""}


def _append_jsonl(event: str, payload: dict) -> None:
    """追加一行 JSON 事件。per-date 句柄缓存；任何失败都自愈并吞掉。"""
    day = time.strftime("%Y-%m-%d")
    try:
        with _lock:
            if _file_state["date"] != day or _file_state["fh"] is None:
                if _file_state["fh"] is not None:
                    try:
                        _file_state["fh"].close()
                    except Exception:
                        pass
                d = _diag_dir()
                d.mkdir(parents=True, exist_ok=True)
                _file_state["fh"] = open(d / f"diag_{day}.jsonl", "a", encoding="utf-8")
                _file_state["date"] = day
            line = json.dumps(
                {"ts": time.strftime("%Y-%m-%dT%H:%M:%S"), "event": event, **payload},
                ensure_ascii=False, default=str)
            _file_state["fh"].write(line + "\n")
            _file_state["fh"].flush()
    except Exception:
        try:
            if _file_state["fh"] is not None:
                _file_state["fh"].close()
        except Exception:
            pass
        _file_state["fh"] = None
        _file_state["date"] = ""


def diag(event: str, **fields) -> None:
    """写一条诊断事件（JSONL + agent.log 结构化行）。永不抛异常。"""
    try:
        payload = {**_correlation(), **fields}
        for k in list(payload):
            v = payload[k]
            if isinstance(v, str) and len(v) > _MAX_STR:
                payload[k] = v[:_MAX_STR] + "..."
        _append_jsonl(event, payload)
        kv = " ".join(f"{k}={str(payload[k])[:120]}".replace(" ", "_")
                      for k in sorted(payload) if k != "req")
        logger.info(f"[diag] {event} {kv}")
    except Exception:
        pass


@contextlib.contextmanager
def diag_span(event: str, **fields):
    """计时区间诊断。用法: with diag_span("plan", target_chars=300): ..."""
    t0 = time.perf_counter()
    diag(event + "_start", **fields)
    try:
        yield
    except Exception as e:
        diag(event + "_error", err=type(e).__name__, msg=str(e)[:300],
             dur_ms=round((time.perf_counter() - t0) * 1000, 1), **fields)
        raise
    else:
        diag(event + "_done", dur_ms=round((time.perf_counter() - t0) * 1000, 1),
             **fields)
