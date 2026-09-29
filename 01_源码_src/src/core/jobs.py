"""Task jobs — 长任务最小闭环状态表（v8.3.7 M2）。

write 类长任务断连后保活执行，状态落 task_jobs 表，前端/接口可查询。
不做完整任务中心（无队列/优先级/暂停恢复/多节点）。
"""
from __future__ import annotations

import json
import logging
import sqlite3
import threading
import uuid
from datetime import datetime
from typing import Optional

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

_DB_PATH = PROJECT_ROOT / "state" / "sessions.db"
_write_lock = threading.Lock()

VALID_STATUS = {"running", "completed", "failed", "cancelled"}


def _connect():
    import sqlite3
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS task_jobs (
            job_id TEXT PRIMARY KEY,
            session_id TEXT,
            request_id TEXT,
            job_type TEXT,
            status TEXT,
            current_step TEXT,
            progress_summary TEXT,
            result_path TEXT,
            error TEXT,
            created_at TEXT,
            updated_at TEXT,
            finished_at TEXT
        )
    """)
    conn.commit()
    return conn


def create_job(session_id: str, request_id: str, job_type: str = "chat") -> str:
    """创建 job（每个 expert 请求一个；write 类工具执行时升级 job_type='write'）。"""
    job_id = uuid.uuid4().hex[:12]
    now = datetime.now().isoformat()
    try:
        with _write_lock:
            with _connect() as conn:
                conn.execute(
                    "INSERT INTO task_jobs (job_id, session_id, request_id, job_type, "
                    "status, current_step, progress_summary, result_path, error, "
                    "created_at, updated_at, finished_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (job_id, session_id, request_id, job_type, "running", "init",
                     "", "", "", now, now, ""),
                )
                conn.commit()
    except Exception as e:
        logger.warning(f"[Jobs] create_job failed: {e}")
    return job_id


def update_job(job_id: str, **fields) -> None:
    """更新 job 字段（status/job_type/current_step/progress_summary/result_path/error）。

    status -> completed/failed/cancelled 时自动写 finished_at。
    """
    if not job_id:
        return
    allowed = {"job_type", "status", "current_step", "progress_summary",
               "result_path", "error"}
    updates = {k: v for k, v in fields.items() if k in allowed and v is not None}
    if not updates:
        return
    if updates.get("status") in ("completed", "failed", "cancelled"):
        updates["finished_at"] = datetime.now().isoformat()
    set_clauses = ", ".join(f"{k}=?" for k in updates)
    values = list(updates.values())
    values.append(job_id)
    try:
        with _write_lock:
            with _connect() as conn:
                conn.execute(
                    f"UPDATE task_jobs SET {set_clauses}, updated_at=? WHERE job_id=?",
                    values[:-1] + [datetime.now().isoformat(), job_id],
                )
                conn.commit()
    except Exception as e:
        logger.warning(f"[Jobs] update_job failed: {e}")


def get_job(job_id: str) -> Optional[dict]:
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM task_jobs WHERE job_id=?",
                               (job_id,)).fetchone()
            if row is None:
                return None
            return dict(row)
    except Exception as e:
        logger.warning(f"[Jobs] get_job failed: {e}")
        return None


def list_for_session(session_id: str, limit: int = 10) -> list:
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM task_jobs WHERE session_id=? "
                "ORDER BY created_at DESC LIMIT ?",
                (session_id, limit)).fetchall()
            return [dict(r) for r in rows]
    except Exception as e:
        logger.warning(f"[Jobs] list_for_session failed: {e}")
        return []


def is_write_job(job_id: str) -> bool:
    """断连保活判定：仅 write 类任务继续后台执行，普通问答仍取消。"""
    job = get_job(job_id)
    return bool(job and job.get("job_type") == "write")
