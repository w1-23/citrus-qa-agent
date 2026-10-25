"""Small account/auth store for the public CitrusQA web service.

Passwords are PBKDF2-HMAC-SHA256 hashed, browser sessions are opaque random
tokens stored only as SHA-256 digests, and account creation is serialized in
SQLite so the first account becomes the bootstrap administrator exactly once.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import secrets
import sqlite3
import threading
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config import PROJECT_ROOT

DB_PATH = PROJECT_ROOT / "state" / "accounts.sqlite3"
COOKIE_NAME = "citrus_auth"
COOKIE_TTL_SECONDS = 7 * 24 * 60 * 60
PBKDF2_ROUNDS = 310_000
_lock = threading.RLock()
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9_.-]{2,31}$")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _stamp(value: datetime | None = None) -> str:
    return (value or _now()).isoformat()


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, mode=0o700, exist_ok=True)
    if os.name != "nt":
        try:
            DB_PATH.parent.chmod(0o700)
        except OSError:
            pass
    conn = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None)
    if os.name != "nt":
        try:
            DB_PATH.chmod(0o600)
        except OSError:
            pass
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=30000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init_store() -> None:
    with _lock, _connect() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL COLLATE NOCASE UNIQUE,
            salt BLOB NOT NULL,
            password_hash BLOB NOT NULL,
            role TEXT NOT NULL DEFAULT 'user',
            active INTEGER NOT NULL DEFAULT 1,
            daily_limit INTEGER NOT NULL DEFAULT 30,
            created_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS browser_sessions (
            token_hash TEXT PRIMARY KEY,
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_browser_sessions_user ON browser_sessions(user_id);
        CREATE TABLE IF NOT EXISTS auth_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            occurred_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_auth_events_lookup ON auth_events(action, source_hash, occurred_at);
        CREATE TABLE IF NOT EXISTS daily_usage (
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            day TEXT NOT NULL,
            request_count INTEGER NOT NULL DEFAULT 0,
            PRIMARY KEY(user_id, day)
        );
        CREATE TABLE IF NOT EXISTS workspace_files (
            user_id INTEGER NOT NULL REFERENCES users(user_id) ON DELETE CASCADE,
            session_id TEXT NOT NULL,
            relative_path TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(session_id, relative_path)
        );
        CREATE INDEX IF NOT EXISTS idx_workspace_files_owner ON workspace_files(user_id, session_id);
        """)
        conn.execute("DELETE FROM browser_sessions WHERE expires_at < ?", (_stamp(),))
        _ensure_session_owner_column()


def _ensure_session_owner_column() -> None:
    sessions_db = PROJECT_ROOT / "state" / "sessions.db"
    if not sessions_db.exists():
        return
    conn = sqlite3.connect(str(sessions_db), timeout=30)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "owner_user_id" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN owner_user_id INTEGER")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_sessions_owner ON sessions(owner_user_id, updated_at)")
        conn.commit()
    finally:
        conn.close()


def _password_hash(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ROUNDS)


def _public_user(row: sqlite3.Row) -> dict[str, Any]:
    return {
        "user_id": int(row["user_id"]),
        "username": str(row["username"]),
        "role": str(row["role"]),
        "active": bool(row["active"]),
        "daily_limit": int(row["daily_limit"]),
        "created_at": str(row["created_at"]),
    }


def validate_credentials(username: str, password: str) -> str | None:
    username = str(username or "").strip()
    if not _USERNAME_RE.fullmatch(username):
        return "用户名需为 3–32 位字母、数字、点、下划线或短横线。"
    if len(password or "") < 10 or len(password or "") > 256:
        return "密码长度需为 10–256 个字符。"
    return None


def register(username: str, password: str) -> tuple[dict[str, Any] | None, str | None, str | None]:
    username = str(username or "").strip()
    error = validate_credentials(username, password)
    if error:
        return None, None, error
    salt = secrets.token_bytes(16)
    digest = _password_hash(password, salt)
    with _lock, _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        try:
            admin_exists = conn.execute("SELECT 1 FROM users WHERE role='admin' LIMIT 1").fetchone()
            role = "user" if admin_exists else "admin"
            try:
                cur = conn.execute(
                    "INSERT INTO users(username,salt,password_hash,role,active,daily_limit,created_at) VALUES(?,?,?,?,1,30,?)",
                    (username, salt, digest, role, _stamp()))
            except sqlite3.IntegrityError:
                conn.rollback()
                return None, None, "该用户名已被使用。"
            user_id = int(cur.lastrowid)
            token = secrets.token_urlsafe(36)
            conn.execute("INSERT INTO browser_sessions VALUES(?,?,?,?)",
                         (hashlib.sha256(token.encode()).hexdigest(), user_id,
                          _stamp(), _stamp(_now() + timedelta(seconds=COOKIE_TTL_SECONDS))))
            row = conn.execute("SELECT * FROM users WHERE user_id=?", (user_id,)).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
    if role == "admin":
        assign_legacy_sessions_to_admin(user_id)
    return _public_user(row), token, None


def login(username: str, password: str) -> tuple[dict[str, Any] | None, str | None]:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE username=? COLLATE NOCASE", (str(username or "").strip(),)).fetchone()
    if not row or not row["active"]:
        return None, None
    candidate = _password_hash(str(password or ""), bytes(row["salt"]))
    if not hmac.compare_digest(candidate, bytes(row["password_hash"])):
        return None, None
    token = secrets.token_urlsafe(36)
    with _connect() as conn:
        conn.execute("INSERT INTO browser_sessions VALUES(?,?,?,?)",
                     (hashlib.sha256(token.encode()).hexdigest(), row["user_id"],
                      _stamp(), _stamp(_now() + timedelta(seconds=COOKIE_TTL_SECONDS))))
    return _public_user(row), token


def verify_password(user_id: int, password: str) -> bool:
    with _connect() as conn:
        row = conn.execute("SELECT salt,password_hash,active FROM users WHERE user_id=?",
                           (int(user_id),)).fetchone()
    if not row or not row["active"]:
        return False
    candidate = _password_hash(str(password or ""), bytes(row["salt"]))
    return hmac.compare_digest(candidate, bytes(row["password_hash"]))


def logout(token: str) -> None:
    if not token:
        return
    with _connect() as conn:
        conn.execute("DELETE FROM browser_sessions WHERE token_hash=?",
                     (hashlib.sha256(token.encode()).hexdigest(),))


def user_for_token(token: str) -> dict[str, Any] | None:
    if not token:
        return None
    digest = hashlib.sha256(token.encode()).hexdigest()
    with _connect() as conn:
        row = conn.execute(
            "SELECT u.* FROM users u JOIN browser_sessions s ON s.user_id=u.user_id "
            "WHERE s.token_hash=? AND s.expires_at>? AND u.active=1",
            (digest, _stamp())).fetchone()
    return _public_user(row) if row else None


def user_by_id(user_id: int) -> dict[str, Any] | None:
    with _connect() as conn:
        row = conn.execute("SELECT * FROM users WHERE user_id=?", (int(user_id),)).fetchone()
    return _public_user(row) if row else None


def require_admin(user: dict[str, Any] | None) -> bool:
    return bool(user and user.get("role") == "admin" and user.get("active"))


def check_rate_limit(action: str, source: str, limit: int, window_seconds: int) -> bool:
    source_hash = hashlib.sha256(str(source or "unknown").encode()).hexdigest()
    cutoff = time.time() - int(window_seconds)
    with _lock, _connect() as conn:
        conn.execute("DELETE FROM auth_events WHERE occurred_at < ?", (time.time() - 86400,))
        count = conn.execute("SELECT COUNT(*) FROM auth_events WHERE action=? AND source_hash=? AND occurred_at>=?",
                             (action, source_hash, cutoff)).fetchone()[0]
        if int(count) >= int(limit):
            return False
        conn.execute("INSERT INTO auth_events(action,source_hash,occurred_at) VALUES(?,?,?)",
                     (action, source_hash, time.time()))
    return True


def reserve_chat_request(user: dict[str, Any]) -> tuple[bool, int]:
    if user.get("role") == "admin":
        return True, 0
    day = _now().date().isoformat()
    with _lock, _connect() as conn:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute("SELECT daily_limit FROM users WHERE user_id=? AND active=1",
                           (user["user_id"],)).fetchone()
        if not row:
            conn.rollback()
            return False, 0
        limit = int(row["daily_limit"])
        conn.execute("INSERT OR IGNORE INTO daily_usage(user_id,day,request_count) VALUES(?,?,0)",
                     (user["user_id"], day))
        current = int(conn.execute("SELECT request_count FROM daily_usage WHERE user_id=? AND day=?",
                                   (user["user_id"], day)).fetchone()[0])
        if current >= limit:
            conn.rollback()
            return False, limit
        conn.execute("UPDATE daily_usage SET request_count=request_count+1 WHERE user_id=? AND day=?",
                     (user["user_id"], day))
        conn.commit()
    return True, limit


def list_users() -> list[dict[str, Any]]:
    day = _now().date().isoformat()
    with _connect() as conn:
        rows = conn.execute(
            "SELECT u.*,COALESCE(d.request_count,0) AS requests_today FROM users u "
            "LEFT JOIN daily_usage d ON d.user_id=u.user_id AND d.day=? ORDER BY u.user_id", (day,)).fetchall()
    return [{**_public_user(row), "requests_today": int(row["requests_today"])} for row in rows]


def set_user_status(user_id: int, active: bool) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE users SET active=? WHERE user_id=?", (int(active), int(user_id)))
        if not active:
            conn.execute("DELETE FROM browser_sessions WHERE user_id=?", (int(user_id),))
    return cur.rowcount > 0


def set_user_daily_limit(user_id: int, limit: int) -> bool:
    with _connect() as conn:
        cur = conn.execute("UPDATE users SET daily_limit=? WHERE user_id=?", (int(limit), int(user_id)))
    return cur.rowcount > 0


def register_workspace_file(user_id: int, session_id: str, relative_path: str) -> None:
    """Record an output file as belonging to one user's session."""
    relative_path = str(relative_path or "").replace("\\", "/").lstrip("/")
    if not session_id or not relative_path or ".." in Path(relative_path).parts:
        return
    with _connect() as conn:
        conn.execute("INSERT INTO workspace_files(user_id,session_id,relative_path,created_at) "
                     "VALUES(?,?,?,?) ON CONFLICT(session_id,relative_path) DO UPDATE SET "
                     "user_id=excluded.user_id,created_at=excluded.created_at",
                     (int(user_id), session_id, relative_path, _stamp()))


def reset_password(user_id: int) -> tuple[str | None, bool]:
    temporary = secrets.token_urlsafe(15)
    salt = secrets.token_bytes(16)
    digest = _password_hash(temporary, salt)
    with _connect() as conn:
        cur = conn.execute("UPDATE users SET salt=?,password_hash=? WHERE user_id=?",
                           (salt, digest, int(user_id)))
        if cur.rowcount:
            conn.execute("DELETE FROM browser_sessions WHERE user_id=?", (int(user_id),))
    return (temporary, cur.rowcount > 0)


def assign_legacy_sessions_to_admin(user_id: int) -> None:
    sessions_db = PROJECT_ROOT / "state" / "sessions.db"
    if not sessions_db.exists():
        return
    conn = sqlite3.connect(str(sessions_db), timeout=30)
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if "owner_user_id" not in cols:
            conn.execute("ALTER TABLE sessions ADD COLUMN owner_user_id INTEGER")
        conn.execute("UPDATE sessions SET owner_user_id=? WHERE owner_user_id IS NULL", (int(user_id),))
        conn.commit()
    finally:
        conn.close()


def delete_user_account(user_id: int) -> bool:
    """Disable account and purge its conversation records and session artifacts."""
    sessions_db = PROJECT_ROOT / "state" / "sessions.db"
    session_ids: list[str] = []
    if sessions_db.exists():
        conn = sqlite3.connect(str(sessions_db), timeout=30)
        try:
            cols = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "owner_user_id" not in cols:
                return False
            session_ids = [row[0] for row in conn.execute(
                "SELECT session_id FROM sessions WHERE owner_user_id=?", (int(user_id),))]
            conn.execute("BEGIN IMMEDIATE")
            for sid in session_ids:
                for table in ("messages", "session_evidence", "permission_grants", "pipeline_tasks", "task_jobs", "feedback"):
                    try:
                        conn.execute(f"DELETE FROM {table} WHERE session_id=?", (sid,))
                    except sqlite3.OperationalError:
                        pass
                conn.execute("DELETE FROM sessions WHERE session_id=?", (sid,))
            conn.commit()
        finally:
            conn.close()
    with _lock, _connect() as conn:
        row = conn.execute("SELECT role FROM users WHERE user_id=?", (int(user_id),)).fetchone()
        if not row or row["role"] == "admin":
            return False
        conn.execute("DELETE FROM users WHERE user_id=?", (int(user_id),))
    artifact_db = PROJECT_ROOT / "state" / "research_artifacts.sqlite3"
    if artifact_db.exists():
        with sqlite3.connect(str(artifact_db), timeout=30) as conn:
            for sid in session_ids:
                try:
                    conn.execute("DELETE FROM artifacts WHERE session_id=?", (sid,))
                except sqlite3.OperationalError:
                    pass
                try:
                    conn.execute("DELETE FROM research_notes WHERE session_id=?", (sid,))
                except sqlite3.OperationalError:
                    pass
    try:
        from src.guardrails.memory import memory_store
        for sid in session_ids:
            memory_store.clear_session(sid)
    except Exception:
        pass
    for sid in session_ids:
        for root in (PROJECT_ROOT / "state" / "artifacts" / sid,
                     PROJECT_ROOT / "workspace" / "output" / "users" / str(int(user_id))):
            try:
                if root.exists():
                    import shutil
                    shutil.rmtree(root)
            except Exception:
                pass
    return True
