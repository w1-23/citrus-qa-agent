"""Write Pipeline 断点续传状态 (v8.3.2).

pipeline_tasks 表持久化 Plan 与已完成章节，支持中断后从下一章继续。
复用 session 库 SQLite，零新依赖。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import time
import uuid
from datetime import datetime
from typing import Optional

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DB_PATH = PROJECT_ROOT / "state" / "sessions.db"


def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    # WAL + busy_timeout: 并发写不互锁、不丢更新
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_tasks (
            task_id TEXT PRIMARY KEY,
            session_id TEXT,
            output_path TEXT,
            plan_json TEXT,
            completed_sections TEXT,
            status TEXT,
            created_at TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    return conn


def start_task(session_id: str, output_path: str, plan: dict) -> str:
    """Plan 成功后写入任务记录。返回 task_id。"""
    task_id = uuid.uuid4().hex[:12]
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO pipeline_tasks (task_id, session_id, output_path, plan_json, "
                "completed_sections, status, created_at, updated_at) VALUES (?,?,?,?,?,?,?,?)",
                (task_id, session_id, output_path, json.dumps(plan, ensure_ascii=False),
                 "[]", "running", datetime.now().isoformat(), datetime.now().isoformat()),
            )
            conn.commit()
        logger.info(f"[WritePipeline] task {task_id} started ({output_path})")
        return task_id
    except Exception as e:
        logger.warning(f"[WritePipeline] start_task failed: {e}")
        return ""


def find_resumable_task(session_id: str, output_path: str) -> Optional[dict]:
    """查找可续传任务（同 session+路径且未完成）。"""
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM pipeline_tasks WHERE session_id=? AND output_path=? "
                "AND status='running' ORDER BY updated_at DESC LIMIT 1",
                (session_id, output_path),
            ).fetchone()
        if row is None:
            return None
        return {
            "task_id": row["task_id"],
            "plan": json.loads(row["plan_json"]),
            "completed": json.loads(row["completed_sections"] or "[]"),
        }
    except Exception as e:
        logger.warning(f"[WritePipeline] find_resumable failed: {e}")
        return None


def mark_section_done(task_id: str, section_index: int) -> None:
    """章节完成后追加到 completed_sections。

    BEGIN IMMEDIATE 事务包裹读-改-写，保证并发续传不丢 completed 标记
    （防止章节重复 append）。
    """
    if not task_id:
        return
    try:
        with _connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute("SELECT completed_sections FROM pipeline_tasks WHERE task_id=?",
                               (task_id,)).fetchone()
            done = json.loads(row[0] or "[]") if row else []
            if section_index not in done:
                done.append(section_index)
            conn.execute(
                "UPDATE pipeline_tasks SET completed_sections=?, updated_at=? WHERE task_id=?",
                (json.dumps(done), datetime.now().isoformat(), task_id),
            )
            conn.commit()
    except Exception as e:
        logger.warning(f"[WritePipeline] mark_section_done failed: {e}")


def finish_task(task_id: str, status: str = "done") -> None:
    """标记任务完成/失败。"""
    if not task_id:
        return
    try:
        with _connect() as conn:
            conn.execute("UPDATE pipeline_tasks SET status=?, updated_at=? WHERE task_id=?",
                         (status, datetime.now().isoformat(), task_id))
            conn.commit()
    except Exception as e:
        logger.warning(f"[WritePipeline] finish_task failed: {e}")


def cleanup_stale_tasks(days: int = 7) -> int:
    """清理 N 天前仍未完成的过期任务记录（仅 running 状态）。"""
    try:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with _connect() as conn:
            cur = conn.execute(
                "DELETE FROM pipeline_tasks WHERE status='running' AND updated_at < ?",
                (cutoff,))
            conn.commit()
            return cur.rowcount
    except Exception as e:
        logger.warning(f"[WritePipeline] cleanup failed: {e}")
        return 0
