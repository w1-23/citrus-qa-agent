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
    # v8.3.7 G2: materials_json 列迁移（材料持久化，断点续传复用）
    try:
        cur = conn.execute("PRAGMA table_info(pipeline_tasks)")
        existing = {row[1] for row in cur.fetchall()}
        if "materials_json" not in existing:
            conn.execute("ALTER TABLE pipeline_tasks ADD COLUMN materials_json TEXT")
            conn.commit()
    except Exception as e:
        logger.warning(f"[WritePipeline] materials column migration: {e}")
    return conn


def start_task(session_id: str, output_path: str, plan: dict,
               materials: Optional[list] = None) -> str:
    """Plan 成功后写入任务记录。返回 task_id。"""
    task_id = uuid.uuid4().hex[:12]
    try:
        with _connect() as conn:
            conn.execute(
                "INSERT INTO pipeline_tasks (task_id, session_id, output_path, plan_json, "
                "completed_sections, status, materials_json, created_at, updated_at) "
                "VALUES (?,?,?,?,?,?,?,?,?)",
                (task_id, session_id, output_path, json.dumps(plan, ensure_ascii=False),
                 "[]", "running",
                 json.dumps(materials or [], ensure_ascii=False)[:2_000_000],
                 datetime.now().isoformat(), datetime.now().isoformat()),
            )
            conn.commit()
        logger.info(f"[WritePipeline] task {task_id} started ({output_path}, "
                    f"{len(materials or [])} materials)")
        return task_id
    except Exception as e:
        logger.warning(f"[WritePipeline] start_task failed: {e}")
        return ""


def find_resumable_task(session_id: str, output_path: str) -> Optional[dict]:
    """查找可续传任务（同 session+路径且未完成）。

    v8.13: 放宽到 running/resumed/partial 三态——此前只查 running，导致
    claim 置 resumed 后崩溃（无 aborted 兜底）或 partial 缺章任务永远不可续。
    """
    try:
        with _connect() as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute(
                "SELECT * FROM pipeline_tasks WHERE session_id=? AND output_path=? "
                "AND status IN ('running','resumed','partial') "
                "ORDER BY updated_at DESC LIMIT 1",
                (session_id, output_path),
            ).fetchone()
        if row is None:
            return None
        materials = None
        try:
            mj = row["materials_json"] if "materials_json" in row.keys() else None
            materials = json.loads(mj) if mj else None
        except Exception:
            materials = None
        return {
            "task_id": row["task_id"],
            "plan": json.loads(row["plan_json"]),
            "completed": json.loads(row["completed_sections"] or "[]"),
            "materials": materials,
        }
    except Exception as e:
        logger.warning(f"[WritePipeline] find_resumable failed: {e}")
        return None


# v8.13: resumed 状态陈旧阈值——claim 置 resumed 后若进程崩溃/取消（无
# aborted 兜底到达），超过该时长视为孤儿任务可被重新领取（续传恢复而非卡死）
RESUME_STALE_SECONDS = 600


def _age_seconds(iso_ts: str) -> float:
    try:
        return (datetime.now() - datetime.fromisoformat(iso_ts)).total_seconds()
    except Exception:
        return 0.0


def claim_resumable_task(session_id: str, output_path: str) -> Optional[dict]:
    """v8.10r: 原子领取可续传任务——BEGIN IMMEDIATE 内查可续状态 → 置 'resumed'，
    防同 session+path 双请求并发续传同一任务双写草稿文件。

    v8.13: 状态放宽到 running/resumed/partial——
      - running: 新任务，直接领取；
      - partial: 上次缺章未写完，可续写补齐（此前只能整篇重来）；
      - resumed: 他人续传中；updated_at 超过 RESUME_STALE_SECONDS 视为孤儿
        （崩溃/取消未回收）可重新领取，否则跳过该条看下一条。
    返回与 find_resumable_task 同构（仅成功领取者拿到）。
    """
    try:
        with _connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM pipeline_tasks WHERE session_id=? AND output_path=? "
                "AND status IN ('running','resumed','partial') "
                "ORDER BY updated_at DESC LIMIT 3",
                (session_id, output_path),
            ).fetchall()
            row = None
            for r in rows:
                if r["status"] == "resumed" and _age_seconds(r["updated_at"]) < RESUME_STALE_SECONDS:
                    continue  # 他人正在续传且未陈旧 → 看下一条
                row = r
                break
            if row is None:
                conn.rollback()
                return None
            conn.execute(
                "UPDATE pipeline_tasks SET status='resumed', updated_at=? WHERE task_id=?",
                (datetime.now().isoformat(), row["task_id"]))
            conn.commit()
        materials = None
        try:
            mj = row["materials_json"] if "materials_json" in row.keys() else None
            materials = json.loads(mj) if mj else None
        except Exception:
            materials = None
        logger.info(f"[WritePipeline] task {row['task_id']} claimed (resumed)")
        return {
            "task_id": row["task_id"],
            "plan": json.loads(row["plan_json"]),
            "completed": json.loads(row["completed_sections"] or "[]"),
            "materials": materials,
        }
    except Exception as e:
        logger.warning(f"[WritePipeline] claim_resumable failed: {e}")
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
    """清理 N 天前仍未完成的过期任务记录。

    v8.13: 状态放宽到 running/resumed/partial 三态——此前只删 running，
    resumed 死态（崩溃/取消未回收）与 partial 残留永远积累。
    """
    try:
        from datetime import timedelta
        cutoff = (datetime.now() - timedelta(days=days)).isoformat()
        with _connect() as conn:
            cur = conn.execute(
                "DELETE FROM pipeline_tasks "
                "WHERE status IN ('running','resumed','partial') AND updated_at < ?",
                (cutoff,))
            conn.commit()
            return cur.rowcount
    except Exception as e:
        logger.warning(f"[WritePipeline] cleanup failed: {e}")
        return 0
