"""Business Logger — 业务功能日志 (v8.4.1).

独立于 agent.log 的业务事件日志，用于排查"业务功能"类问题
（用户问"为什么看不到历史/为什么回答这么短"这类链路定位）：

  文件: logs/business.log（RotatingFileHandler, 10MB × 5）
  格式: 2026-08-13 18:00:00 | req=e373bd08 | event=xxx | k=v | k=v

与 agent.log 的分工:
  - agent.log: 模块级调试/告警（技术细节）
  - business.log: 每个请求的业务事件线（一次请求可 grep req= 串出完整链路）

使用: from src.core.business_logger import blog
      blog("supervisor_done", turn=2, answer_chars=1632, tools=2, ms=161633)
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

_logger: logging.Logger | None = None


def _get_logger() -> logging.Logger:
    global _logger
    if _logger is not None:
        return _logger
    import os
    from pathlib import Path
    from src.config import PROJECT_ROOT

    # v8.4.5: 与 agent.log 一致——CITRUS_LOG_DIR 覆盖日志目录（测试独立 sink，
    # 防测试帧污染生产 business.log）
    override = os.environ.get("CITRUS_LOG_DIR", "").strip()
    log_dir = Path(override) if override else PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    biz = logging.getLogger("business")
    biz.setLevel(logging.INFO)
    biz.propagate = False
    handler = RotatingFileHandler(
        log_dir / "business.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    biz.addHandler(handler)
    _logger = biz
    return biz


def blog(event: str, **fields) -> None:
    """写一条业务事件。永不抛异常（日志失败不得影响主链路）。"""
    try:
        from src.core.tracing import get_request_id
        rid = get_request_id() or "-"
        parts = [f"req={rid}", f"event={event}"]
        for k, v in fields.items():
            s = str(v).replace("|", "/").replace("\r", " ").replace("\n", " ")
            if len(s) > 300:
                s = s[:300] + "..."
            parts.append(f"{k}={s}")
        _get_logger().info(" | ".join(parts))
    except Exception:
        pass
