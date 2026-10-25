"""Account-scoped administrative request audit with a fixed one-year retention."""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "state" / "admin_audit.sqlite3"
RETENTION_SECONDS = 365 * 24 * 60 * 60
_lock = threading.RLock()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    if os.name != "nt":
        try:
            DB_PATH.parent.chmod(0o700)
            DB_PATH.chmod(0o600)
        except OSError:
            pass
    return conn


def init_store() -> None:
    with _lock, _connect() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS request_audit (
                request_id TEXT PRIMARY KEY,
                user_id INTEGER NOT NULL,
                username TEXT NOT NULL,
                session_id TEXT NOT NULL,
                query TEXT NOT NULL,
                answer TEXT NOT NULL DEFAULT '',
                attachment_metadata_json TEXT NOT NULL DEFAULT '[]',
                tool_trace_json TEXT NOT NULL DEFAULT '[]',
                retrieval_trace_json TEXT NOT NULL DEFAULT '{}',
                status TEXT NOT NULL DEFAULT 'running',
                elapsed_ms INTEGER,
                created_at REAL NOT NULL,
                finished_at REAL
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_request_audit_created ON request_audit(created_at DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_request_audit_user ON request_audit(user_id, created_at DESC)")
        conn.execute("DELETE FROM request_audit WHERE created_at < ?", (time.time() - RETENTION_SECONDS,))
        conn.commit()


def start_request(*, request_id: str, user_id: int, username: str,
                  session_id: str, query: str, attachments: list[dict[str, Any]]) -> None:
    metadata = []
    for item in (attachments or [])[:8]:
        data = str(item.get("data") or "")
        metadata.append({
            "name": str(item.get("name") or "attachment")[:180],
            "type": str(item.get("type") or "application/octet-stream")[:120],
            "approx_bytes": max(0, (len(data) * 3 // 4) if data else 0),
        })
    with _lock, _connect() as conn:
        conn.execute("""
            INSERT OR REPLACE INTO request_audit
            (request_id,user_id,username,session_id,query,attachment_metadata_json,created_at)
            VALUES(?,?,?,?,?,?,?)
        """, (request_id, int(user_id), str(username)[:64], str(session_id),
              str(query)[:20000], json.dumps(metadata, ensure_ascii=False), time.time()))
        conn.commit()


def finish_request(*, request_id: str, answer: str = "", tool_trace: list[Any] | None = None,
                   retrieval_trace: dict[str, Any] | None = None,
                   status: str = "completed", elapsed_ms: int | None = None) -> None:
    safe_tools = []
    for item in (tool_trace or [])[:100]:
        if isinstance(item, str):
            safe_tools.append({"name": item[:160]})
        elif isinstance(item, dict):
            safe_tools.append({str(k)[:80]: v for k, v in item.items()
                               if k in {"name", "tool", "status", "elapsed_ms", "summary"}})
    with _lock, _connect() as conn:
        conn.execute("""
            UPDATE request_audit SET answer=?, tool_trace_json=?, retrieval_trace_json=?,
                status=?, elapsed_ms=?, finished_at=? WHERE request_id=?
        """, (str(answer)[:100000], json.dumps(safe_tools, ensure_ascii=False),
              json.dumps(retrieval_trace or {}, ensure_ascii=False)[:12000],
              str(status)[:32], elapsed_ms, time.time(), request_id))
        conn.commit()


def mark_interrupted_requests() -> int:
    """Close audit rows left running when the single service process restarted."""
    with _lock, _connect() as conn:
        cursor = conn.execute(
            "UPDATE request_audit SET status='failed', answer=?, finished_at=? "
            "WHERE status='running'",
            ("服务进程在请求完成前停止；请重新提交。", time.time()),
        )
        conn.commit()
        return int(cursor.rowcount or 0)


def list_requests(*, limit: int = 50, offset: int = 0, user_id: int | None = None,
                  query: str = "") -> dict[str, Any]:
    limit = max(1, min(int(limit), 100))
    offset = max(0, int(offset))
    cutoff = time.time() - RETENTION_SECONDS
    clauses = ["created_at >= ?"]
    params: list[Any] = [cutoff]
    if user_id is not None:
        clauses.append("user_id = ?")
        params.append(int(user_id))
    if query.strip():
        clauses.append("(username LIKE ? OR session_id LIKE ? OR request_id LIKE ? OR query LIKE ?)")
        like = f"%{query.strip()[:100]}%"
        params.extend([like] * 4)
    where = " AND ".join(clauses)
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM request_audit WHERE created_at < ?", (cutoff,))
        total = int(conn.execute(f"SELECT COUNT(*) FROM request_audit WHERE {where}", params).fetchone()[0])
        rows = conn.execute(
            f"SELECT request_id,user_id,username,session_id,query,answer,attachment_metadata_json,"
            f"tool_trace_json,retrieval_trace_json,status,elapsed_ms,created_at,finished_at "
            f"FROM request_audit WHERE {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            [*params, limit, offset]).fetchall()
    items = []
    for row in rows:
        item = dict(row)
        for col, key in (("attachment_metadata_json", "attachments"),
                         ("tool_trace_json", "tools"),
                         ("retrieval_trace_json", "retrieval")):
            try:
                item[key] = json.loads(item.pop(col) or ("{}" if key == "retrieval" else "[]"))
            except (TypeError, ValueError):
                item[key] = {} if key == "retrieval" else []
        item["created_at"] = datetime.fromtimestamp(item["created_at"], timezone.utc).isoformat()
        if item.get("finished_at"):
            item["finished_at"] = datetime.fromtimestamp(item["finished_at"], timezone.utc).isoformat()
        items.append(item)
    return {"total": total, "items": items, "retention_days": 365}
