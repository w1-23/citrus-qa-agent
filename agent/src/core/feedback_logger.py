"""Feedback Logger — 回答反馈日志（v8.7 / v8.13 实现合并到 business_logger）。

文件 logs/feedback.log 记录每次用户 👍/👎 反馈事件，与落库（sessions.db
feedback 表）双写：库表供结构化统计，日志供人工排查与回溯。

v8.13: 与 business_logger 99% 复制的实现已合并——本文件仅保留 re-export，
调用方（session/manager）与测试无需改动，feedback.log 行为不变。
"""
from __future__ import annotations

from src.core.business_logger import feedback_log  # noqa: F401
