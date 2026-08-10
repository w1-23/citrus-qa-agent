"""Session Manager — SQLite-based session persistence.

v8.1.1: Return LangChain Messages, no auto-compression.
"""
from __future__ import annotations

import json
import logging
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

    async def save_messages(self, session_id: str, messages: List[BaseMessage]) -> None:
        try:
            await asyncio.to_thread(self._save_messages_sync, session_id, messages)
            logger.debug(f"[SessionManager] saved {len(messages)} msgs for {session_id[:8]}")
        except Exception as e:
            logger.error(f"[SessionManager] save failed: {e}")

    def _save_messages_sync(self, session_id: str, messages: List[BaseMessage]):
        import sqlite3
        with sqlite3.connect(self.db_path) as conn:
            for msg in messages:
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

                conn.execute(
                    "INSERT INTO messages "
                    "(session_id, msg_type, content, tool_call_id, tool_calls_json, name) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (session_id, msg_type, content, tool_call_id, tool_calls_json, name),
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
