"""Business Logger — 业务功能日志（v8.4.1 / v8.13 合并单实现）.

独立于 agent.log 的业务事件日志，用于排查"业务功能"类问题
（用户问"为什么看不到历史/为什么回答这么短"这类链路定位）：

  文件: logs/business.log（RotatingFileHandler, 10MB × 5）
  格式: 2026-08-13 18:00:00 | req=e373bd08 | event=xxx | k=v | k=v

与 agent.log 的分工:
  - agent.log: 模块级调试/告警（技术细节）
  - business.log: 每个请求的业务事件线（一次请求可 grep req= 串出完整链路）

v8.13: 原 feedback_logger.py 与本文 99% 复制（各自维护 _get_logger/格式化/脱敏），
合并为单实现——两个薄入口 blog()/feedback_log() 共用同一写入核心与 logger 工厂，
feedback.log 文件行为不变（历史工具/测试依赖）。

使用: from src.core.business_logger import blog
      blog("supervisor_done", turn=2, answer_chars=1632, tools=2, ms=161633)
      from src.core.business_logger import feedback_log   # 也可从 feedback_logger re-export
"""
from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler

_loggers: dict = {}


def _get_logger(name: str, file_name: str) -> logging.Logger:
    if name in _loggers:
        return _loggers[name]
    import os
    from pathlib import Path
    from src.config import PROJECT_ROOT

    # v8.4.5: 与 agent.log 一致——CITRUS_LOG_DIR 覆盖日志目录（测试独立 sink，
    # 防测试帧污染生产业务日志）
    override = os.environ.get("CITRUS_LOG_DIR", "").strip()
    log_dir = Path(override) if override else PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    lg = logging.getLogger(name)
    lg.setLevel(logging.INFO)
    lg.propagate = False
    handler = RotatingFileHandler(
        log_dir / file_name,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(
        "%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    lg.addHandler(handler)
    _loggers[name] = lg
    return lg


def _write(logger_name: str, file_name: str, event: str, fields: dict) -> None:
    """写入一条业务事件。永不抛异常（日志失败不得影响主链路）。"""
    try:
        from src.core.tracing import get_request_id
        from src.core.pii_mask import mask_sensitive
        rid = get_request_id() or "-"
        parts = [f"req={rid}", f"event={event}"]
        for k, v in fields.items():
            s = mask_sensitive(str(v))  # v8.4.6 B6: 日志脱敏
            s = s.replace("|", "/").replace("\r", " ").replace("\n", " ")
            if len(s) > 300:
                s = s[:300] + "..."
            parts.append(f"{k}={s}")
        _get_logger(logger_name, file_name).info(" | ".join(parts))
    except Exception:
        pass


def blog(event: str, **fields) -> None:
    """业务事件线入口（logs/business.log）。"""
    _write("business", "business.log", event, fields)


def feedback_log(event: str, **fields) -> None:
    """反馈事件入口（logs/feedback.log）——v8.13 起与 blog 共用实现。"""
    _write("feedback", "feedback.log", event, fields)
