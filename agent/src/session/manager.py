"""Session Manager — SQLite-based session persistence.

v8.1.1: Return LangChain Messages, no auto-compression.
v8.3.7: 幂等写入（client_request_id/idempotency_key）+ 会话级写锁 + 统一历史写入入口。
"""
from __future__ import annotations

import hashlib
import json
import logging
import threading
import time
import uuid
import asyncio
from pathlib import Path
from datetime import datetime
from typing import List, Optional

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    SystemMessage,
)

from src.config import PROJECT_ROOT

logger = logging.getLogger(__name__)

STATE_DIR = PROJECT_ROOT / "state"
DB_PATH = STATE_DIR / "sessions.db"

# 会话级写锁分桶：同一 session 的历史写入串行化，防止 save/replace 并发交错
_session_locks: dict = {}
_session_locks_guard = threading.Lock()


def _get_session_lock(session_id: str) -> threading.Lock:
    with _session_locks_guard:
        lock = _session_locks.get(session_id)
        if lock is None:
            lock = threading.Lock()
            _session_locks[session_id] = lock
        return lock


def compute_idempotency_key(session_id: str, query: str,
                            client_request_id: str = "") -> str:
    """幂等键（v8.3.7）：优先客户端稳定 ID（重试复用）；无则服务端 30s 时间桶兜底。"""
    if client_request_id:
        return f"crid:{client_request_id[:64]}"
    bucket = int(time.time() // 30)
    raw = f"{session_id}|{query[:500]}|{bucket}"
    return "srv:" + hashlib.sha256(raw.encode("utf-8")).hexdigest()[:24]


class SessionManager:
    _instance: Optional["SessionManager"] = None

    def __new__(cls):
        if cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False
            cls._instance = instance
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        self.db_path = str(DB_PATH)
        self._init_db_sync()
        self._initialized = True
        logger.info(f"[SessionManager] ready | DB: {self.db_path}")

    def _init_db_sync(self):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    msg_type TEXT NOT NULL,
                    content TEXT,
                    tool_call_id TEXT,
                    tool_calls_json TEXT,
                    name TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (session_id) REFERENCES sessions(session_id)
                )
            """)
            conn.commit()
        self._migrate_schema()
        self._ensure_idempotency_index()

    def _migrate_schema(self):
        """Add columns if missing from older schemas."""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute("PRAGMA table_info(messages)")
                existing = {row[1] for row in cur.fetchall()}
                for col, col_type in [
                    ("msg_type", "TEXT NOT NULL DEFAULT ''"),
                    ("tool_call_id", "TEXT"),
                    ("tool_calls_json", "TEXT"),
                    ("name", "TEXT"),
                    ("client_request_id", "TEXT"),   # v8.3.7 M1
                    ("idempotency_key", "TEXT"),     # v8.3.7 M1
                ]:
                    if col not in existing:
                        try:
                            conn.execute(f"ALTER TABLE messages ADD COLUMN {col} {col_type}")
                            logger.info(f"[SessionManager] added column: {col}")
                        except Exception:
                            pass
                conn.commit()
        except Exception as e:
            logger.debug(f"[SessionManager] schema migration: {e}")

    def _ensure_idempotency_index(self):
        """幂等唯一索引（v8.3.7 M1）：数据库层兜底，防多进程/竞态重复写。"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_msg_idempotency "
                    "ON messages(session_id, idempotency_key) "
                    "WHERE idempotency_key IS NOT NULL"
                )
                conn.commit()
        except Exception as e:
            logger.warning(f"[SessionManager] idempotency index failed: {e}")

    async def get_or_create_session(self, session_id: Optional[str]) -> str:
        if not session_id or session_id.lower() == "new":
            new_id = str(uuid.uuid4())
            await asyncio.to_thread(self._create_session_sync, new_id)
            return new_id
        exists = await asyncio.to_thread(self._check_session_exists, session_id)
        if not exists:
            await asyncio.to_thread(self._create_session_sync, session_id)
        return session_id

    def _create_session_sync(self, session_id: str):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO sessions (session_id) VALUES (?)",
                (session_id,),
            )
            conn.commit()

    def _check_session_exists(self, session_id: str) -> bool:
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "SELECT 1 FROM sessions WHERE session_id = ?", (session_id,)
            )
            return cur.fetchone() is not None

    async def get_messages(self, session_id: str) -> List[BaseMessage]:
        return await asyncio.to_thread(self._get_messages_sync, session_id)

    def _get_messages_sync(self, session_id: str) -> List[BaseMessage]:
        import sqlite3
        messages: List[BaseMessage] = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT msg_type, content, tool_call_id, tool_calls_json, name "
                "FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            for row in cur.fetchall():
                msg_type = row["msg_type"] or ""
                content = row["content"] or ""
                tool_call_id = row["tool_call_id"]
                tool_calls_json = row["tool_calls_json"]
                name = row["name"]

                try:
                    if msg_type == "human":
                        messages.append(HumanMessage(content=content))
                    elif msg_type == "ai":
                        kw = {"content": content, "name": name or None}
                        if tool_calls_json:
                            try:
                                tool_calls = [
                                    {"id": tc["id"], "name": tc["name"],
                                     "args": tc.get("args", {})}
                                    for tc in json.loads(tool_calls_json)
                                ]
                                kw["tool_calls"] = tool_calls
                            except Exception:
                                pass
                        messages.append(AIMessage(**{k: v for k, v in kw.items() if v is not None}))
                    elif msg_type == "tool":
                        messages.append(ToolMessage(
                            content=content,
                            tool_call_id=tool_call_id or "unknown",
                            name=name or None,
                        ))
                    elif msg_type == "system":
                        messages.append(SystemMessage(content=content))
                except Exception as e:
                    logger.debug(f"[SessionManager] skip unparseable message: {e}")
        return messages

    async def save_messages(self, session_id: str, messages: List[BaseMessage],
                            idempotency_key: str = "") -> bool:
        """追加历史（v8.3.7 幂等）：同 session 同幂等键已存在则跳过，返回是否写入。"""
        try:
            written = await asyncio.to_thread(
                self._save_messages_sync, session_id, messages, idempotency_key)
            if written:
                logger.debug(f"[SessionManager] saved {len(messages)} msgs for {session_id[:8]}")
            else:
                logger.debug(f"[SessionManager] duplicate skipped (key={idempotency_key[:16]}) "
                             f"for {session_id[:8]}")
            return written
        except Exception as e:
            logger.error(f"[SessionManager] save failed: {e}")
            return False

    def _serialize_message(self, msg: BaseMessage) -> tuple:
        """Extract (msg_type, content, tool_call_id, tool_calls_json, name) from a message."""
        msg_type = ""
        content = ""
        tool_call_id = None
        tool_calls_json = None
        name = None

        if hasattr(msg, "type"):
            msg_type = msg.type
        elif hasattr(msg, "role"):
            msg_type = msg.role

        if hasattr(msg, "content") and msg.content:
            c = msg.content
            if isinstance(c, list):
                content = json.dumps(c, ensure_ascii=False)
            else:
                content = str(c)

        if hasattr(msg, "tool_call_id"):
            tool_call_id = msg.tool_call_id

        if hasattr(msg, "tool_calls") and msg.tool_calls:
            try:
                tool_calls_json = json.dumps([
                    {"id": tc.get("id", ""), "name": tc.get("name", ""),
                     "args": tc.get("args", {})}
                    for tc in msg.tool_calls
                ], ensure_ascii=False)
            except Exception:
                pass

        if hasattr(msg, "name"):
            name = msg.name

        return (msg_type, content, tool_call_id, tool_calls_json, name)

    def _save_messages_sync(self, session_id: str, messages: List[BaseMessage],
                            idempotency_key: str = "") -> bool:
        import sqlite3
        # v8.3.7: 会话级写锁串行化所有历史写入（save/replace 同一入口）
        lock = _get_session_lock(session_id)
        with lock:
            with sqlite3.connect(self.db_path) as conn:
                if idempotency_key:
                    dup = conn.execute(
                        "SELECT 1 FROM messages WHERE session_id=? AND idempotency_key=? LIMIT 1",
                        (session_id, idempotency_key)).fetchone()
                    if dup:
                        return False
                for idx, msg in enumerate(messages):
                    # v8.3.7: 幂等键为"轮次级"——仅批首行携带（唯一索引防整批重放）
                    row_key = idempotency_key if (idx == 0 and idempotency_key) else None
                    conn.execute(
                        "INSERT INTO messages "
                        "(session_id, msg_type, content, tool_call_id, tool_calls_json, "
                        "name, client_request_id, idempotency_key) "
                        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                        (session_id, *self._serialize_message(msg),
                         idempotency_key if (idx == 0 and idempotency_key.startswith("crid:")) else None,
                         row_key),
                    )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (datetime.now().isoformat(), session_id),
                )
                conn.commit()
        return True

    async def replace_history(self, session_id: str, messages: List[BaseMessage]) -> None:
        """原子替换会话历史（压缩/截断后持久化，v8.3.1）。事务内 DELETE+INSERT，失败不改动原数据。"""
        try:
            await asyncio.to_thread(self._replace_history_sync, session_id, messages)
            logger.info(f"[SessionManager] history replaced: {len(messages)} msgs for {session_id[:8]}")
        except Exception as e:
            logger.error(f"[SessionManager] replace_history failed: {e}")

    def _replace_history_sync(self, session_id: str, messages: List[BaseMessage]):
        import sqlite3
        lock = _get_session_lock(session_id)
        with lock:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
                for msg in messages:
                    conn.execute(
                        "INSERT INTO messages "
                        "(session_id, msg_type, content, tool_call_id, tool_calls_json, name) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (session_id, *self._serialize_message(msg)),
                    )
                conn.execute(
                    "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                    (datetime.now().isoformat(), session_id),
                )
                conn.commit()

    async def clear_session(self, session_id: str):
        await asyncio.to_thread(self._clear_session_sync, session_id)

    def _clear_session_sync(self, session_id: str):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
            conn.commit()


session_manager = SessionManager()
