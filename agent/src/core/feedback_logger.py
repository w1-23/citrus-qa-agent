"""Feedback Logger — 回答反馈日志（v8.7）。

独立文件 logs/feedback.log（RotatingFileHandler 10MB × 5），记录每次
用户 👍/👎 反馈事件（含"哪里好/哪里不好"评论），与落库（sessions.db
feedback 表）双写：库表供结构化统计，日志供人工排查与回溯。

格式与 business.log 一致：时间 | req= | event=feedback | k=v | ...
日志目录跟随 CITRUS_LOG_DIR 覆盖（测试独立 sink，防测试帧污染生产日志）。
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

    override = os.environ.get("CITRUS_LOG_DIR", "").strip()
    log_dir = Path(override) if override else PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    fb = logging.getLogger("feedback")
    fb.setLevel(logging.INFO)
    fb.propagate = False
    handler = RotatingFileHandler(
        log_dir / "feedback.log",
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    fb.addHandler(handler)
    _logger = fb
    return fb


def feedback_log(event: str, **fields) -> None:
    """写一条反馈事件。永不抛异常（日志失败不得影响主链路）。"""
    try:
        from src.core.tracing import get_request_id
        from src.core.pii_mask import mask_sensitive
        rid = get_request_id() or "-"
        parts = [f"req={rid}", f"event={event}"]
        for k, v in fields.items():
            s = mask_sensitive(str(v))
            s = s.replace("|", "/").replace("\r", " ").replace("\n", " ")
            if len(s) > 300:
                s = s[:300] + "..."
            parts.append(f"{k}={s}")
        _get_logger().info(" | ".join(parts))
    except Exception:
        pass
