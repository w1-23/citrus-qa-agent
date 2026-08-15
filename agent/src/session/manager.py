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

# v8.4.2: 历史污染清理——旧版 supervisor 收尾 bug 会把伪造的"用户指令"写进历史
# （"You have reached the maximum number of turns..."），读时幂等过滤（非破坏性，
# 只滤 HumanMessage，保留其后的正式回答 AIMessage）。修复后新请求不再产生。
_SYNTH_FORCE_FINAL_MARK = "You have reached the maximum number of turns"


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


def _validate_trace(messages: list) -> list:
    """v8.3.8 协议安全（INV-01 持久化路径）：无配对 tool_calls 的孤立 ToolMessage 丢弃。

    历史消息重放给模型时，孤立 ToolMessage 会触发 OpenAI 400。
    """
    valid_ids = set()
    for m in messages:
        tcs = getattr(m, "tool_calls", None) or []
        for tc in tcs:
            tc_id = tc.get("id", "") if isinstance(tc, dict) else getattr(tc, "id", "")
            if tc_id:
                valid_ids.add(tc_id)
    out = []
    for m in messages:
        tc_id = getattr(m, "tool_call_id", "")
        if isinstance(m, ToolMessage) and tc_id and tc_id not in valid_ids:
            logger.warning(f"[SessionManager] 孤立 ToolMessage 丢弃 (tool_call_id={tc_id})")
            continue
        out.append(m)
    return out


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
            # v8.3.8: 证据账本（跨轮检索复用；session 隔离，生产级多用户预留）
            conn.execute("""
                CREATE TABLE IF NOT EXISTS session_evidence (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT,
                    turn_seq INTEGER,
                    query TEXT,
                    evidence_json TEXT,
                    report_text TEXT,
                    created_at TEXT
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_se_session "
                "ON session_evidence(session_id, created_at)")
            conn.commit()
        self._migrate_schema()
        self._ensure_idempotency_index()
        # v8.4.3: 存量伪造指令一次性 purge（幂等，读时过滤保留为 DEBUG 安全网）
        self._purge_synth_history()
        # v8.7: 存量 msg_type 归一化——v8.4.13 流式化阶段写入的 "AIMessageChunk"
        # 统一改为 "ai"（加载端兼容 + 数据一致性；幂等，无则跳过）
        try:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "UPDATE messages SET msg_type='ai' WHERE msg_type='AIMessageChunk'")
                if cur.rowcount:
                    logger.info(f"[SessionManager] 迁移 {cur.rowcount} 条 "
                                f"AIMessageChunk -> ai")
                conn.commit()
        except Exception as e:
            logger.debug(f"[SessionManager] msg_type migration skipped: {e}")

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
                # v8.4 压缩检查点（存储全量·发送裁剪：记录"已摘要至哪条消息 + 摘要文本"，
                # 原始轨迹永不删除——发送视图基于 checkpoint 增量构建，防摘要套摘要）
                cur = conn.execute("PRAGMA table_info(sessions)")
                session_cols = {row[1] for row in cur.fetchall()}
                for col, col_type in [
                    ("checkpoint_msg_id", "INTEGER DEFAULT 0"),
                    ("checkpoint_summary", "TEXT DEFAULT ''"),
                    ("checkpoint_updated_at", "TEXT DEFAULT ''"),
                ]:
                    if col not in session_cols:
                        try:
                            conn.execute(f"ALTER TABLE sessions ADD COLUMN {col} {col_type}")
                            logger.info(f"[SessionManager] added column: {col}")
                        except Exception:
                            pass
                conn.commit()
        except Exception as e:
            logger.debug(f"[SessionManager] schema migration: {e}")

    def _purge_synth_history(self):
        """v8.4.3: 一次性清理存量污染——旧版 supervisor 收尾 bug 写入的伪造
        "用户指令"（"You have reached the maximum number of turns..."）。
        读时过滤（_get_messages_sync）只保证新读路径干净，脏行仍留库且每请求
        触发过滤日志；此处启动期 DELETE 根治，迁移版本标记幂等（参照
        _ltm_migrated_v2 模式）。修复后新请求不再产生此类消息。
        """
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS memory_store (
                        session_id TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        PRIMARY KEY (session_id, key))"""
                )
                marked = conn.execute(
                    "SELECT COUNT(*) FROM memory_store WHERE key='_synth_purge_v843'"
                ).fetchone()[0]
                if marked:
                    return
                cur = conn.execute(
                    "DELETE FROM messages WHERE msg_type='human' "
                    "AND content LIKE ?",
                    (_SYNTH_FORCE_FINAL_MARK + "%",))
                deleted = cur.rowcount
                conn.execute(
                    "INSERT OR REPLACE INTO memory_store "
                    "(session_id, key, value, updated_at) VALUES (?, ?, ?, ?)",
                    ("__system__", "_synth_purge_v843", "1",
                     datetime.now().isoformat()))
                conn.commit()
                if deleted:
                    logger.info(
                        f"[SessionManager] 存量污染清理: 删除 {deleted} 条伪造收尾指令")
        except Exception as e:
            logger.warning(f"[SessionManager] purge synth history failed: {e}")

    # ── v8.4.3 结构化权限授权（permission_mode=ask 时前端卡片闭环）──

    def _ensure_permission_schema(self, conn) -> None:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS permission_grants (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id TEXT,
                tool_name TEXT,
                scope TEXT,
                created_at TEXT
            )"""
        )

    def grant_permission(self, session_id: str, tool_name: str, scope: str) -> bool:
        """记录授权（once/session/workspace）。once 由 consume 消费删除。"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_permission_schema(conn)
                conn.execute(
                    "INSERT INTO permission_grants "
                    "(session_id, tool_name, scope, created_at) VALUES (?, ?, ?, ?)",
                    (session_id or "", tool_name or "", scope or "session",
                     datetime.now().isoformat()))
                conn.commit()
            logger.info(f"[SessionManager] 权限授权: tool={tool_name} "
                        f"scope={scope} session={str(session_id)[:8]}")
            return True
        except Exception as e:
            logger.warning(f"[SessionManager] grant_permission failed: {e}")
            return False

    def consume_grant(self, session_id: str, tool_name: str, path: str) -> bool:
        """校验并消费授权。workspace 范围匹配 workspace/output 路径免 session；
        once 消费即删；session 范围要求 session 匹配。"""
        import sqlite3
        try:
            # 与 registry._is_workspace_output_path 同一口径（相对路径按 workspace/output 解析）
            in_workspace = False
            try:
                from src.tools.registry import _is_workspace_output_path
                in_workspace = _is_workspace_output_path(str(path or ""))
            except Exception:
                in_workspace = False
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_permission_schema(conn)
                rows = conn.execute(
                    "SELECT id, session_id, scope FROM permission_grants "
                    "WHERE tool_name=? ORDER BY id DESC",
                    (tool_name,)).fetchall()
                for rid, gsid, scope in rows:
                    if scope == "workspace":
                        if in_workspace:
                            return True
                        continue
                    if scope == "once":
                        conn.execute("DELETE FROM permission_grants WHERE id=?", (rid,))
                        conn.commit()
                        return True
                    if scope == "session" and gsid == session_id:
                        return True
            return False
        except Exception as e:
            logger.warning(f"[SessionManager] consume_grant failed: {e}")
            return False

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

    def _get_messages_sync(self, session_id: str, with_ids: bool = False):
        """读回会话历史。with_ids=True 时返回 (messages, row_ids)，
        row_ids 与 messages 一一对应（压缩 checkpoint 定位用）。"""
        import sqlite3
        messages: List[BaseMessage] = []
        row_ids: List[int] = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cur = conn.execute(
                "SELECT id, msg_type, content, tool_call_id, tool_calls_json, name "
                "FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            )
            ai_tool_pending = 0   # v8.7 INV-01 防御：待配对的工具结果数（跨行保持）
            for row in cur.fetchall():
                msg_type = row["msg_type"] or ""
                content = row["content"] or ""
                tool_call_id = row["tool_call_id"]
                tool_calls_json = row["tool_calls_json"]
                name = row["name"]

                try:
                    if msg_type == "human":
                        # v8.4.2/8.4.3: 过滤旧版收尾 bug 写入的伪造"用户指令"。
                        # 启动期已 purge（幂等），此处仅作 DEBUG 级安全网。
                        if content.startswith(_SYNTH_FORCE_FINAL_MARK):
                            logger.debug(
                                f"[SessionManager] 过滤历史伪造收尾指令 "
                                f"(session={session_id[:8]})")
                            continue
                        messages.append(HumanMessage(content=content))
                        ai_tool_pending = 0
                    elif msg_type in ("ai", "AIMessageChunk"):
                        # v8.7: "AIMessageChunk" 为 v8.4.13 流式化后的旧数据
                        # 类型残留（保存端已归一化为 "ai"）——按 ai 还原，避免
                        # 其 ToolMessage 配对消息因跳过而孤立（API 400 根因）。
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
                        # v8.7 INV-01 防御：记录待配对工具结果数（孤立 ToolMessage 防护）
                        ai_tool_pending = len(kw.get("tool_calls") or [])
                    elif msg_type == "tool":
                        # v8.7 INV-01 协议配对防御：tool 消息前必须存在带 tool_calls
                        # 的 AI 消息（预算跳过/熔断/压缩切割任何路径破坏配对时，
                        # 宁可丢弃也不让非法消息列表进入 API 触发 400）
                        if ai_tool_pending <= 0:
                            logger.warning(
                                f"[SessionManager] 丢弃孤立 ToolMessage "
                                f"(session={session_id[:8]}, id={row['id']}, "
                                f"tc_id={(tool_call_id or '')[:12]})")
                            continue
                        ai_tool_pending -= 1
                        messages.append(ToolMessage(
                            content=content,
                            tool_call_id=tool_call_id or "unknown",
                            name=name or None,
                        ))
                    elif msg_type == "system":
                        messages.append(SystemMessage(content=content))
                    else:
                        continue
                    row_ids.append(row["id"])
                except Exception as e:
                    logger.debug(f"[SessionManager] skip unparseable message: {e}")
        if with_ids:
            return messages, row_ids
        return messages

    async def get_messages_with_ids(self, session_id: str):
        """v8.4: 返回 (messages, row_ids)（压缩视图构建/checkpoint 定位用）。"""
        return await asyncio.to_thread(self._get_messages_sync, session_id, True)

    # ── v8.4 压缩检查点（存储全量·发送裁剪）──

    def get_checkpoint(self, session_id: str) -> dict:
        """返回 {"msg_id": int, "summary": str}；无则 msg_id=0。"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                row = conn.execute(
                    "SELECT checkpoint_msg_id, checkpoint_summary "
                    "FROM sessions WHERE session_id = ?",
                    (session_id,)).fetchone()
                if not row:
                    return {"msg_id": 0, "summary": ""}
                return {"msg_id": row["checkpoint_msg_id"] or 0,
                        "summary": row["checkpoint_summary"] or ""}
        except Exception as e:
            logger.debug(f"[SessionManager] get_checkpoint failed: {e}")
            return {"msg_id": 0, "summary": ""}

    def set_checkpoint(self, session_id: str, msg_id: int, summary: str) -> None:
        """持久化压缩检查点：原始消息永不删除，只记录摘要位置。"""
        import sqlite3
        try:
            lock = _get_session_lock(session_id)
            with lock:
                with sqlite3.connect(self.db_path) as conn:
                    conn.execute(
                        "UPDATE sessions SET checkpoint_msg_id=?, "
                        "checkpoint_summary=?, checkpoint_updated_at=? "
                        "WHERE session_id=?",
                        (int(msg_id or 0), summary or "",
                         datetime.now().isoformat(), session_id))
                    conn.commit()
        except Exception as e:
            logger.warning(f"[SessionManager] set_checkpoint failed: {e}")

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

        # v8.7: 类型归一化——v8.4.13 真流式化后 supervisor 消息为 AIMessageChunk，
        # 其 .type == "AIMessageChunk"，入库后加载端不认识会跳过 → 工具配对断裂
        # （孤立 ToolMessage → API 400 "tool must be a response to tool_calls"）。
        # 统一归一为 "ai"，与历史数据一致。
        if msg_type in ("AIMessageChunk", "AI", "AIMessage"):
            msg_type = "ai"

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
        """[DEPRECATED v8.4] 原子替换会话历史。

        存储全量·发送裁剪架构下不再调用：原始轨迹 append-only 永不改写，
        压缩改为发送视图构建 + checkpoint（见 set_checkpoint）。
        保留此方法仅为兼容旧调用方。
        """
        try:
            await asyncio.to_thread(self._replace_history_sync, session_id, messages)
            logger.info(f"[SessionManager] history replaced (DEPRECATED): {len(messages)} msgs for {session_id[:8]}")
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
            conn.execute("DELETE FROM session_evidence WHERE session_id = ?", (session_id,))
            conn.execute(
                "UPDATE sessions SET updated_at = ?, checkpoint_msg_id = 0, "
                "checkpoint_summary = '' WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
            conn.commit()

    # ── Evidence Ledger（v8.3.8）──

    async def save_evidence(self, session_id: str, query: str,
                            evidence: list, report_text: str) -> None:
        """保存一轮检索的证据账本（结构化清单 + 报告全文）。"""
        if not evidence and not report_text:
            return
        try:
            await asyncio.to_thread(self._save_evidence_sync, session_id,
                                    query, evidence, report_text)
        except Exception as e:
            logger.error(f"[SessionManager] save_evidence failed: {e}")

    def _save_evidence_sync(self, session_id: str, query: str,
                            evidence: list, report_text: str):
        import sqlite3
        lock = _get_session_lock(session_id)
        with lock:
            with sqlite3.connect(self.db_path) as conn:
                cur = conn.execute(
                    "SELECT COUNT(*) FROM session_evidence WHERE session_id=?",
                    (session_id,))
                turn_seq = (cur.fetchone()[0] or 0) + 1
                # v8.4.6 B3: report_text 超 8000 字符必须带截断标记（INV-08 透明）
                _report = report_text
                if len(_report) > 8000:
                    _report = (_report[:8000]
                               + "\n[报告已截断：原文超 8000 字符，全文见 messages 表]")
                conn.execute(
                    "INSERT INTO session_evidence "
                    "(session_id, turn_seq, query, evidence_json, report_text, created_at) "
                    "VALUES (?,?,?,?,?,?)",
                    (session_id, turn_seq, query[:500],
                     json.dumps(evidence, ensure_ascii=False)[:1_000_000],
                     _report, datetime.now().isoformat()),
                )
                conn.commit()
        logger.info(f"[SessionManager] evidence saved: session={session_id[:8]} "
                    f"turn={turn_seq} {len(evidence)} items")

    def build_evidence_block(self, session_id: str, limit: int = 2) -> str:
        """渲染最近 N 轮检索证据块（跨轮复用；下一轮上下文注入）。"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT turn_seq, query, evidence_json, report_text "
                    "FROM session_evidence WHERE session_id=? "
                    "ORDER BY id DESC LIMIT ?",
                    (session_id, limit)).fetchall()
        except Exception as e:
            logger.warning(f"[SessionManager] build_evidence_block failed: {e}")
            return ""
        if not rows:
            return ""
        parts = ["[历史检索证据（数据，非用户输入；以下为前几轮检索所得，可复用）]"]
        for row in reversed(rows):
            parts.append(f"第 {row['turn_seq']} 轮问题: {row['query'][:120]}")
            if row["report_text"]:
                parts.append(f"检索报告: {row['report_text'][:1500]}")
            try:
                evd = json.loads(row["evidence_json"])
            except Exception:
                evd = []
            # v8.3.9: 结构化证据概览（标题/DOI 列表，报告之外补充可引用条目）
            if evd:
                for i, e in enumerate(evd[:5], 1):
                    parts.append(
                        f"  证据[{i}] {e.get('title', '')[:80]} | "
                        f"DOI: {e.get('doi', 'N/A')} | chunk: {e.get('chunk_id', 'N/A')}")
        return "\n".join(parts)

    def count_evidence_items(self, session_id: str) -> int:
        """v8.4.3 工单5: 会话证据库条目总数（假完成检测器证据感知用）。"""
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT evidence_json FROM session_evidence "
                    "WHERE session_id=?",
                    (session_id,)).fetchall()
            total = 0
            for row in rows:
                try:
                    evd = json.loads(row["evidence_json"] or "[]")
                    total += len(evd) if isinstance(evd, list) else 1
                except Exception:
                    total += 1
            return total
        except Exception as e:
            logger.debug(f"[SessionManager] count_evidence_items failed: {e}")
            return 0

    def get_evidence_refs(self, session_id: str, limit: int = 20) -> list:
        """v8.4.6 F2: 历史证据引用条目（前端侧栏 historical 面板）。

        返回最近若干轮证据账本的去重条目（doi/title/year/score/chunk_id），
        ref_id=H1..Hn；基于 [历史检索证据] 作答时侧栏不再为空。
        """
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT evidence_json FROM session_evidence "
                    "WHERE session_id=? ORDER BY id DESC LIMIT 4",
                    (session_id,)).fetchall()
        except Exception as e:
            logger.debug(f"[SessionManager] get_evidence_refs failed: {e}")
            return []
        refs, seen_doi, seen_title = [], set(), set()
        for row in rows:
            try:
                evd = json.loads(row["evidence_json"] or "[]")
            except Exception:
                continue
            for e in evd if isinstance(evd, list) else []:
                if not isinstance(e, dict):
                    continue
                doi = str(e.get("doi") or "").strip()
                title = str(e.get("title") or "").strip()[:120]
                key = doi or title
                if not key or key in seen_doi or (not doi and key in seen_title):
                    continue
                if doi:
                    seen_doi.add(key)
                else:
                    seen_title.add(key)
                refs.append({
                    "ref_id": f"H{len(refs) + 1}",
                    "type": "historical",
                    "doi": doi or "N/A",
                    "title": title,
                    "year": str(e.get("year", "")),
                    "score": e.get("score", 0) or 0,
                    "chunk_id": e.get("chunk_id", ""),
                })
                if len(refs) >= limit:
                    return refs
        return refs

    def get_evidence_materials(self, session_id: str, limit: int = 30) -> list:
        """v8.4.6 F7: 会话证据账本转写作材料包（dict 列表，供 write_pipeline 消费）。

        本轮无新检索时，写任务仍可走 plan_execute 主路径（材料来自历史
        证据的 chunk 片段），避免降级 ReAct 导致证据链二次转写。
        """
        import sqlite3
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT evidence_json FROM session_evidence "
                    "WHERE session_id=? ORDER BY id DESC LIMIT 4",
                    (session_id,)).fetchall()
        except Exception as e:
            logger.debug(f"[SessionManager] get_evidence_materials failed: {e}")
            return []
        materials, seen = [], set()
        for row in rows:
            try:
                evd = json.loads(row["evidence_json"] or "[]")
            except Exception:
                continue
            for e in evd if isinstance(evd, list) else []:
                if not isinstance(e, dict):
                    continue
                doi = str(e.get("doi") or "").strip()
                title = str(e.get("title") or "").strip()
                key = doi or title
                if not key or key in seen:
                    continue
                seen.add(key)
                materials.append({
                    "doi": doi,
                    "title": title,
                    "year": str(e.get("year", "")),
                    "score": e.get("score", 0) or 0,
                    "chunk_id": e.get("chunk_id", ""),
                    "text": str(e.get("snippet") or ""),
                    "_source": "session_evidence",
                })
                if len(materials) >= limit:
                    return materials
        return materials

    # ── v8.6 用户反馈落库（书 §4.6 反馈循环 / O7 经验学习第一步）──
    # 纯记录、零副作用：不进入对话历史、不影响问答/检索/写作流程；
    # 数据作为未来"经验沉淀"（书 §8.2.1）的原料，等待离线分析。

    def record_feedback(self, session_id: str, message_id: str,
                        rating: int, comment: str = "") -> bool:
        """记录 👍(1)/👎(-1) 反馈。幂等（同 session+message+rating 去重）。

        v8.7: 落库（feedback 表）+ 双写 logs/feedback.log（人工排查/回溯）。
        """
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """CREATE TABLE IF NOT EXISTS feedback (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id TEXT NOT NULL,
                        message_id TEXT DEFAULT '',
                        rating INTEGER NOT NULL,
                        comment TEXT DEFAULT '',
                        created_at TEXT NOT NULL)""")
                conn.execute(
                    "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_dedup "
                    "ON feedback(session_id, message_id, rating)")
                conn.execute(
                    """INSERT OR IGNORE INTO feedback
                       (session_id, message_id, rating, comment, created_at)
                       VALUES (?, ?, ?, ?, ?)""",
                    (session_id, message_id or "", int(rating),
                     str(comment or "")[:500], datetime.now().isoformat()))
                conn.commit()
            # v8.7: 反馈日志双写（落库 + feedback.log）
            try:
                from src.core.feedback_logger import feedback_log
                feedback_log("feedback_recorded",
                             session=session_id[:12] or "-",
                             message_id=(message_id or "-")[:32],
                             rating=int(rating),
                             comment=str(comment or "")[:200])
            except Exception:
                pass
            return True
        except Exception as e:
            logger.warning(f"[SessionManager] record feedback failed: {e}")
            try:
                from src.core.feedback_logger import feedback_log
                feedback_log("feedback_failed", err=str(e)[:200],
                             rating=int(rating))
            except Exception:
                pass
            return False

    def get_feedback_stats(self) -> dict:
        """反馈统计（正向/负向计数），供运营查看与测试断言。"""
        try:
            import sqlite3
            with sqlite3.connect(self.db_path) as conn:
                pos = conn.execute(
                    "SELECT COUNT(*) FROM feedback WHERE rating=1").fetchone()[0]
                neg = conn.execute(
                    "SELECT COUNT(*) FROM feedback WHERE rating=-1").fetchone()[0]
            return {"positive": int(pos), "negative": int(neg), "total": int(pos) + int(neg)}
        except Exception as e:
            logger.debug(f"[SessionManager] feedback stats failed: {e}")
            return {"positive": 0, "negative": 0, "total": 0}


session_manager = SessionManager()
