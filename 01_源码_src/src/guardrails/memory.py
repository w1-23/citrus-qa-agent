"""
Memory System — 多类型记忆存储
参考 Claude Code "只保存无法推导的信息" 原则设计。

记忆类型：
  - entity_memory: {entity_name: {type, aliases, relations, last_mentioned}}
  - concept_memory: {concept_key: {summary, evidence, confidence, last_mentioned}}
  - preference_memory: {key: value} 用户显式偏好
"""
import json
import logging
import re
import threading
from datetime import datetime

# v9.2 重构清理：顶层 sqlite3（原仅 _connect_db 使用，已收敛至 core/db；
# 各方法内保留局部 import）与 typing（全文件无类型注解使用）删除

# v9.2 CON-8: 统一连接工厂/建表收敛至 src/core/db（WAL + busy_timeout=30s 口径
# 与 v8.13 SQL-1 逐位一致；原 manager/memory 双份 _connect_db 实现在此收敛）
from src.core.db import connect_db as _connect_db, ensure_memory_store

logger = logging.getLogger(__name__)


class MemoryStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    # v8.13 SQL-2: RLock 可重入——set_preference 持锁后内部再经
                    # _save_store 等路径写库不会自死锁
                    cls._instance._mem_lock = threading.RLock()
        return cls._instance

    def __init__(self):
        from src.config import PROJECT_ROOT
        # v8.4: DB 路径实例化属性（测试可定向到临时库；此前硬编码 PROJECT_ROOT）
        self.db_path = str(PROJECT_ROOT / "state" / "sessions.db")

    # ─── Preference Memory ───

    def set_preference(self, user_id: int, key: str, value: str) -> None:
        """Save an explicit preference within one authenticated account."""
        user_id = int(user_id or 0)
        key, value = str(key or "").strip(), str(value or "").strip()
        if user_id <= 0 or not key or not value:
            return
        with self._mem_lock, _connect_db(self.db_path) as conn:
            self._ensure_account_memory_schema(conn)
            conn.execute(
                """INSERT INTO account_preferences (user_id, pref_key, pref_value, updated_at)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(user_id, pref_key) DO UPDATE SET
                     pref_value=excluded.pref_value, updated_at=excluded.updated_at""",
                (user_id, key[:80], value[:300], datetime.now().isoformat()),
            )
            conn.commit()

    def get_preferences(self, user_id: int, max_chars: int = 300) -> str:
        """Return this account's explicit preferences for the current prompt."""
        user_id = int(user_id or 0)
        if user_id <= 0:
            return ""
        try:
            with _connect_db(self.db_path) as conn:
                self._ensure_account_memory_schema(conn)
                rows = conn.execute(
                    "SELECT pref_key, pref_value FROM account_preferences "
                    "WHERE user_id=? ORDER BY updated_at DESC LIMIT 10", (user_id,)
                ).fetchall()
            if not rows:
                return ""
            header = ("## 用户偏好（仅当前账号；与本轮要求冲突时以本轮要求为准）\n")
            parts = []
            total = len(header)
            for row in rows:
                line = f"- {str(row[0])[:60]}: {str(row[1])[:120]}"
                if total + len(line) + 1 > max_chars:
                    break
                parts.append(line)
                total += len(line) + 1
            return header + "\n".join(parts) if parts else ""
        except Exception as exc:
            logger.debug("[Memory] account preference recall failed: %s", exc)
            return ""

    def _ensure_account_memory_schema(self, conn) -> None:
        """Create account-scoped memory tables and safely attribute identifiable legacy rows.

        Legacy rows without a session owner stay orphaned and are never injected into prompts.
        Rows with a session are assigned only when sessions.owner_user_id identifies an account.
        """
        conn.execute("""CREATE TABLE IF NOT EXISTS account_preferences (
            user_id INTEGER NOT NULL, pref_key TEXT NOT NULL, pref_value TEXT NOT NULL,
            updated_at TEXT NOT NULL, PRIMARY KEY(user_id, pref_key))""")
        conn.execute("""CREATE TABLE IF NOT EXISTS account_resident_cards (
            user_id INTEGER NOT NULL, fact_key TEXT NOT NULL, fact_value TEXT NOT NULL,
            confidence REAL DEFAULT 0.5, updated_at TEXT, PRIMARY KEY(user_id, fact_key))""")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_account_resident_user ON account_resident_cards(user_id, confidence DESC)")
        conn.execute("""CREATE TABLE IF NOT EXISTS account_memory_migrations (
            migration_key TEXT PRIMARY KEY, completed_at TEXT NOT NULL)""")

        store_columns = {row[1] for row in conn.execute("PRAGMA table_info(memory_store)")}
        session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        if {"session_id", "key", "value", "updated_at"}.issubset(store_columns) and "owner_user_id" in session_columns:
            migrated = conn.execute(
                "SELECT 1 FROM account_memory_migrations WHERE migration_key='session_preferences_v1'"
            ).fetchone()
            if not migrated:
                # Migrate only session-owned legacy preferences; the old __global__ bucket
                # has no trustworthy owner and is intentionally never assigned. A marker
                # prevents a deliberate memory clear from being undone on the next read.
                for row in conn.execute(
                    """SELECT s.owner_user_id, m.value, m.updated_at FROM memory_store m
                       JOIN sessions s ON s.session_id=m.session_id
                       WHERE m.key='preference_memory' AND m.session_id<>''
                         AND s.owner_user_id IS NOT NULL"""
                ).fetchall():
                    try:
                        legacy = json.loads(row[1])
                        if isinstance(legacy, dict):
                            conn.executemany(
                                """INSERT INTO account_preferences(user_id,pref_key,pref_value,updated_at)
                                   VALUES(?,?,?,?)
                                   ON CONFLICT(user_id,pref_key) DO UPDATE SET
                                     pref_value=excluded.pref_value, updated_at=excluded.updated_at
                                   WHERE excluded.updated_at > account_preferences.updated_at""",
                                [(int(row[0]), str(k)[:80], str(v)[:300], str(row[2] or datetime.now().isoformat()))
                                 for k, v in legacy.items() if k and v]
                            )
                    except Exception:
                        continue
                conn.execute(
                    "INSERT INTO account_memory_migrations(migration_key,completed_at) VALUES('session_preferences_v1',?)",
                    (datetime.now().isoformat(),))

        for table in ("long_term_memory", "ltm_facts"):
            columns = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
            if not columns:
                continue
            if "owner_user_id" not in columns:
                conn.execute(f"ALTER TABLE {table} ADD COLUMN owner_user_id INTEGER")
            session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "owner_user_id" in session_columns and "owner_session" in columns:
                conn.execute(
                    f"UPDATE {table} SET owner_user_id=(SELECT owner_user_id FROM sessions "
                    f"WHERE sessions.session_id={table}.owner_session) "
                    f"WHERE owner_user_id IS NULL AND owner_session<>'' "
                    f"AND EXISTS (SELECT 1 FROM sessions WHERE sessions.session_id={table}.owner_session "
                    f"AND owner_user_id IS NOT NULL)"
                )
        if {row[1] for row in conn.execute("PRAGMA table_info(ltm_facts)")}:
            conn.execute("CREATE INDEX IF NOT EXISTS idx_ltm_account_owner ON ltm_facts(owner_user_id, updated_at)")

        # Attribute old resident cards only through a known session owner. Global legacy cards
        # have no trustworthy account and deliberately remain excluded.
        resident_columns = {row[1] for row in conn.execute("PRAGMA table_info(resident_cards)")}
        if resident_columns and "session_id" in resident_columns:
            session_columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
            if "owner_user_id" in session_columns:
                migrated = conn.execute(
                    "SELECT 1 FROM account_memory_migrations WHERE migration_key='resident_cards_v1'"
                ).fetchone()
                if not migrated:
                    conn.execute(
                        """INSERT OR IGNORE INTO account_resident_cards
                           (user_id, fact_key, fact_value, confidence, updated_at)
                           SELECT s.owner_user_id, r.fact_key, r.fact_value, r.confidence, r.updated_at
                           FROM resident_cards r JOIN sessions s ON s.session_id=r.session_id
                           WHERE r.session_id<>'' AND s.owner_user_id IS NOT NULL"""
                    )
                    conn.execute(
                        "INSERT INTO account_memory_migrations(migration_key,completed_at) VALUES('resident_cards_v1',?)",
                        (datetime.now().isoformat(),))

    # ─── Long-term Memory (Cross-session) ───

    def _ensure_ltm_schema(self, conn) -> None:
        """AG-5: 迁移 long_term_memory 表，补齐 owner_session / source_query 列。

        v8.4 (Mem0 v3 模式): 新增 ltm_facts 表（自增 id，ADD-only 写入——
        同 key 多版本并存，冲突留到检索时用时间+置信度排序解决；
        旧表写入式 UPDATE/DELETE 不可逆丢历史）。老表数据一次性迁移，
        之后只读老表、只写新表。
        """
        conn.execute(
            """CREATE TABLE IF NOT EXISTS long_term_memory (
                fact_key TEXT PRIMARY KEY,
                fact_value TEXT,
                confidence REAL DEFAULT 0.5,
                updated_at TEXT,
                owner_session TEXT DEFAULT '',
                source_query TEXT DEFAULT ''
            )"""
        )
        # CREATE 之后重新读取列（旧库缺列才 ALTER，新库已有）
        cols = {r[1] for r in conn.execute("PRAGMA table_info(long_term_memory)")}
        if "owner_session" not in cols:
            conn.execute("ALTER TABLE long_term_memory ADD COLUMN owner_session TEXT DEFAULT ''")
        if "source_query" not in cols:
            conn.execute("ALTER TABLE long_term_memory ADD COLUMN source_query TEXT DEFAULT ''")

        conn.execute(
            """CREATE TABLE IF NOT EXISTS ltm_facts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fact_key TEXT,
                fact_value TEXT,
                confidence REAL DEFAULT 0.5,
                updated_at TEXT,
                owner_session TEXT DEFAULT '',
                source_query TEXT DEFAULT ''
            )"""
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_ltm_fact_key ON ltm_facts(fact_key)")

        # 一次性迁移老表 → 新表（幂等：迁移标记行）
        # v9.2 CON-8: memory_store 建表收敛共享常量（core/db）
        ensure_memory_store(conn)
        self._ensure_account_memory_schema(conn)
        migrated = conn.execute(
            "SELECT COUNT(*) FROM memory_store WHERE key='_ltm_migrated_v2'"
        ).fetchone()[0]
        if not migrated:
            conn.execute(
                """INSERT INTO ltm_facts (fact_key, fact_value, confidence,
                                          updated_at, owner_session, source_query, owner_user_id)
                   SELECT fact_key, fact_value, confidence, updated_at,
                          owner_session, source_query, owner_user_id
                   FROM long_term_memory"""
            )
            conn.execute(
                "INSERT OR REPLACE INTO memory_store (session_id, key, value, updated_at) "
                "VALUES ('__system__', '_ltm_migrated_v2', '1', ?)",
                (datetime.now().isoformat(),))
            logger.info("[Memory] LTM 迁移至 ltm_facts (ADD-only) 完成")

    # ─── v8.4 常驻卡片层（双层记忆: 少量高频事实常驻上下文 ≤500 字符）───

    def _ensure_resident_schema(self, conn) -> None:
        conn.execute(
            """CREATE TABLE IF NOT EXISTS resident_cards (
                fact_key TEXT PRIMARY KEY,
                fact_value TEXT,
                confidence REAL DEFAULT 0.5,
                updated_at TEXT,
                session_id TEXT DEFAULT ''
            )"""
        )
        # v8.9 常驻卡片域化迁移（''=全局域；旧数据全部归全局）
        try:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(resident_cards)")}
            if "session_id" not in cols:
                conn.execute("ALTER TABLE resident_cards ADD COLUMN session_id TEXT DEFAULT ''")
        except Exception:
            pass

    RESIDENT_CARD_MIN_CONF = 0.8     # 高置信度事实才进常驻卡片
    RESIDENT_CARD_MAX = 8            # 常驻卡片上限（约 ≤500 字符预算）
    RESIDENT_CARD_MAX_CHARS = 60     # 单条卡片值上限

    def _upsert_resident_card(self, conn, fact_key: str, fact_value: str,
                              confidence: float, user_id: int = 0) -> None:
        self._ensure_resident_schema(conn)
        self._ensure_account_memory_schema(conn)
        if not fact_key or not fact_value or int(user_id or 0) <= 0:
            return
        if confidence < self.RESIDENT_CARD_MIN_CONF:
            return
        if len(fact_value) > self.RESIDENT_CARD_MAX_CHARS:
            return
        conn.execute(
            """INSERT INTO account_resident_cards (user_id, fact_key, fact_value, confidence, updated_at)
               VALUES (?, ?, ?, ?, ?)
               ON CONFLICT(user_id, fact_key) DO UPDATE SET
                 fact_value=excluded.fact_value,
                 confidence=excluded.confidence,
                 updated_at=excluded.updated_at""",
            (int(user_id), fact_key, fact_value, confidence, datetime.now().isoformat()),
        )
        # 超上限淘汰：最低置信度 + 最旧的先出（代码维护，非 LLM）
        cur = conn.execute(
            "SELECT COUNT(*) FROM account_resident_cards WHERE user_id=?", (int(user_id),))
        if (cur.fetchone()[0] or 0) > self.RESIDENT_CARD_MAX:
            conn.execute(
                """DELETE FROM account_resident_cards WHERE user_id=? AND fact_key IN (
                    SELECT fact_key FROM account_resident_cards WHERE user_id=?
                    ORDER BY confidence ASC, updated_at ASC
                    LIMIT 1)""",
                (int(user_id), int(user_id)))

    def get_resident_cards(self, user_id: int = 0, max_chars: int = 500) -> str:
        """Return high-confidence resident facts belonging to this account only."""
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            if not db_path.exists():
                return ""
            with _connect_db(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                if int(user_id or 0) <= 0:
                    return ""
                self._ensure_resident_schema(conn)
                self._ensure_account_memory_schema(conn)
                rows = conn.execute(
                    "SELECT fact_key, fact_value, confidence, updated_at "
                    "FROM account_resident_cards WHERE user_id=? ORDER BY confidence DESC, updated_at DESC",
                    (int(user_id),)).fetchall()
            if not rows:
                return ""
            parts = []
            total = 0
            for r in rows:
                line = f"- {r['fact_value']} (置信 {float(r['confidence']):.1f})"
                if total + len(line) > max_chars:
                    break
                parts.append(line)
                total += len(line)
            if not parts:
                return ""
            return "## 用户常驻记忆（高频稳定事实，可能过期，冲突时以本次检索证据为准）\n" \
                   + "\n".join(parts)
        except Exception as e:
            logger.debug(f"[Memory] 常驻卡片读取失败: {e}")
            return ""

    def save_long_term_fact(self, fact_key: str, fact_value: str, confidence: float = 0.5,
                            owner_session: str = "", source_query: str = "",
                            owner_user_id: int = 0) -> bool:
        """v8.4 (Mem0 v3): ADD-only 写入——同 key 多版本并存，绝不覆盖/删除历史。
        检索阶段用时间+置信度排序消歧。置信度 <0.5 的未核验事实拒绝写入。
        """
        if not fact_key or not fact_value or int(owner_user_id or 0) <= 0:
            return False
        if float(confidence or 0) < 0.5:
            logger.debug(f"[Memory] 低置信度事实拒绝写入: {fact_key[:30]} ({confidence})")
            return False
        with self._mem_lock:
            try:
                import sqlite3
                from pathlib import Path
                db_path = Path(self.db_path)
                db_path.parent.mkdir(parents=True, exist_ok=True)
                with _connect_db(str(db_path)) as conn:
                    self._ensure_ltm_schema(conn)
                    conn.execute(
                        """INSERT INTO ltm_facts
                           (fact_key, fact_value, confidence, updated_at,
                            owner_session, source_query, owner_user_id)
                           VALUES (?, ?, ?, ?, ?, ?, ?)""",
                        (fact_key, fact_value, confidence, datetime.now().isoformat(),
                         owner_session, source_query, int(owner_user_id)),
                    )
                    # High-confidence overview facts use the same account boundary as LTM.
                    self._upsert_resident_card(conn, fact_key, fact_value, confidence,
                                               user_id=owner_user_id)
                    conn.commit()
                return True
            except Exception as e:
                logger.debug(f"[Memory] 保存长期事实失败: {e}")
                return False

    def recall_long_term_memory(self, query: str, top_k: int = 5,
                                owner_user_id: int = 0, max_chars: int = 1500) -> str:
        """语义向量检索跨会话长期记忆 (fallback 到关键词)。
        AG-5: 置信度按时间衰减(0.95/天)，输出带来源与更新时间标注，超 max_chars 截断。
        v8.9 记忆域化（会话为主 + 高置信全局共享）：owner_session 传入当前会话——
        召回该会话域事实 + 全局高置信（≥0.9）事实；跨域事实排序权重 0.9（本会话优先）。
        """
        try:
            if int(owner_user_id or 0) <= 0:
                return ""
            return self._recall_semantic(query, top_k, int(owner_user_id), max_chars)
        except Exception as e:
            logger.debug(f"[Memory] 语义召回失败, 回退关键词: {e}")
            return self._recall_keyword_fallback(query, top_k, int(owner_user_id or 0), max_chars)

    # v8.9 记忆域化参数：跨域（全局共享）事实的排序权重（本会话域事实 1.0）
    CROSS_DOMAIN_WEIGHT = 0.9
    GLOBAL_SHARE_CONF = 0.9   # 全局共享置信度门槛

    def _fetch_ltm_rows(self, conn, limit: int = 500, owner_user_id: int = 0):
        """v8.4: 从 ltm_facts 读取；若新表为空回退老表（迁移前兼容）。
        v8.9 域过滤：本会话域 + 全局高置信（≥0.9）+ 旧数据（无域标记）三类。"""
        import sqlite3  # v9.2: 原依赖顶层 import（已收敛删除），本函数补局部导入
        conn.row_factory = sqlite3.Row
        self._ensure_account_memory_schema(conn)
        if owner_user_id:
            rows = conn.execute(
                "SELECT fact_key, fact_value, confidence, updated_at, owner_session, source_query, owner_user_id "
                "FROM ltm_facts "
                "WHERE owner_user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (int(owner_user_id), limit)).fetchall()
        else:
            return []
        if not rows:
            rows = conn.execute(
                "SELECT fact_key, fact_value, confidence, updated_at, owner_session, source_query, owner_user_id "
                "FROM long_term_memory WHERE owner_user_id=? ORDER BY updated_at DESC LIMIT ?",
                (int(owner_user_id), limit)).fetchall()
        return rows

    def _recall_semantic(self, query: str, top_k: int, owner_user_id: int, max_chars: int) -> str:
        import numpy as np
        import sqlite3
        from pathlib import Path

        db_path = Path(self.db_path)
        if not db_path.exists():
            return ""
        with _connect_db(str(db_path)) as conn:
            self._ensure_ltm_schema(conn)
            rows = self._fetch_ltm_rows(conn, owner_user_id=owner_user_id)
        if not rows:
            return ""

        from src.engine.embedder import Embedder
        embedder = Embedder()
        texts = [f"{r['fact_key']}: {r['fact_value']}" for r in rows]
        query_vec = np.array(embedder.embed_query(query), dtype=np.float32)
        doc_vecs = np.array(embedder.embed_docs(texts), dtype=np.float32)

        q_norm = query_vec / np.linalg.norm(query_vec)
        d_norm = doc_vecs / np.linalg.norm(doc_vecs, axis=1, keepdims=True)
        scores = (q_norm @ d_norm.T)

        now = datetime.now()
        eff_confs = np.zeros(len(rows), dtype=np.float32)
        for i, row in enumerate(rows):
            conf = float(row["confidence"])
            ts = row["updated_at"] or ""
            try:
                days = max((now - datetime.fromisoformat(ts)).days, 0)
            except Exception:
                days = 0
            eff = conf * (0.95 ** days)  # AG-5: 时间衰减
            # v8.9 记忆域化：跨域（全局共享）事实轻微降权，本会话域优先
            eff_confs[i] = eff

        # v8.4 (Mem0 v3): 混合信号排序 = 语义相似度 × 有效置信度
        # 同 key 的多版本（ADD-only 并存）同时参与排序，由时间衰减自然
        # 让最新版本胜出——冲突在检索阶段解决，不写时覆盖
        combined = scores * eff_confs
        top_indices = np.argsort(combined)[::-1][:top_k * 2]

        parts = []
        seen_values = set()
        for i in top_indices:
            if len(parts) >= top_k:
                break
            score = float(scores[i])
            if score < 0.30:
                continue
            row = rows[i]
            eff_conf = float(eff_confs[i])
            if eff_conf < 0.30:
                continue
            # 完全相同的 value 去重（多版本共存但同值只列一次）
            dedup_key = row["fact_value"]
            if dedup_key in seen_values:
                continue
            seen_values.add(dedup_key)
            ts = row["updated_at"] or ""
            src = (row["source_query"] or "")[:20]
            parts.append(
                f"- [{eff_conf:.1f}] {row['fact_value']} "
                f"(更新: {ts[:10]} 来源: {src or 'N/A'})"
            )
        if not parts:
            return ""
        text = ("## 跨会话记忆\n以下为历史会话中提取的相关事实（可能过期，"
                "若与本次检索证据冲突，以本次检索证据为准）：\n" + "\n".join(parts))
        return text[:max_chars]

    def _recall_keyword_fallback(self, query: str, top_k: int, owner_user_id: int, max_chars: int) -> str:
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            if not db_path.exists():
                return ""
            with _connect_db(str(db_path)) as conn:
                self._ensure_ltm_schema(conn)
                rows = self._fetch_ltm_rows(conn, limit=200, owner_user_id=owner_user_id)
            query_tokens = set(re.findall(r'[\w\u4e00-\u9fff]{2,}', query.lower()))
            scored = []
            now = datetime.now()
            for row in rows:
                key_tokens = set(re.findall(r'[\w\u4e00-\u9fff]{2,}', row["fact_key"].lower()))
                overlap = len(query_tokens & key_tokens)
                if overlap > 0:
                    conf = float(row["confidence"])
                    ts = row["updated_at"] or ""
                    try:
                        days = max((now - datetime.fromisoformat(ts)).days, 0)
                    except Exception:
                        days = 0
                    eff_conf = conf * (0.95 ** days)
                    # v8.9 记忆域化：跨域轻微降权
                    if eff_conf >= 0.30:
                        scored.append((overlap * eff_conf, row["fact_value"], eff_conf, ts,
                                       row["source_query"] or ""))
            scored.sort(key=lambda x: x[0], reverse=True)
            if not scored:
                return ""
            parts = []
            seen_values = set()
            for score, value, conf, ts, src in scored:
                if value in seen_values:
                    continue
                seen_values.add(value)
                parts.append(f"- [{conf:.1f}] {value} (更新: {ts[:10]} 来源: {src[:20] or 'N/A'})")
                if len(parts) >= top_k:
                    break
            text = ("## 跨会话记忆\n以下为历史会话中提取的相关事实（可能过期，"
                    "若与本次检索证据冲突，以本次检索证据为准）：\n" + "\n".join(parts))
            return text[:max_chars]
        except Exception:
            return ""

    def extract_key_facts(self, query: str, answer: str, session_id: str = "") -> list[dict]:
        """提取 3-5 条核心事实 + 用户偏好（异步调用，用最便宜模型）。

        v8.6 (书 §3.1 偏好追踪): 输出项可带 "type": "preference"——用户明确表达的
        长期偏好（语言/风格/格式约定等），由调用方路由到 preference_memory；
        其余默认事实走 ADD-only LTM。
        """
        if not answer or len(answer) < 50:
            return []
        try:
            from src.config import settings
            from openai import OpenAI
            client = OpenAI(api_key=settings.RESOLVED_MAIN_API_KEY,
                            base_url=settings.RESOLVED_MAIN_BASE_URL,
                            default_headers=settings.RESOLVED_PROVIDER_HEADERS)
            prompt = (
                "从以下问答对中提取最多5条不可推导的核心事实，每条用一句话描述。\n"
                "只保留那些如果不记录就会丢失的信息（如：具体数值、疾病-基因关联、实验条件）。\n"
                "不要保留可以重新检索的常识性信息。\n"
                "如果回答中包含用户明确表达的长期偏好（如写作语言、篇幅、格式、风格、"
                "是否要求局限与边界等），单独提取为偏好项："
                "{\"type\": \"preference\", \"key\": \"偏好名\", \"value\": \"偏好内容\", \"confidence\": 0.9}。\n"
                "输出JSON数组，格式：[{\"key\": \"柑橘黄龙病病原为Candidatus Liberibacter\", \"value\": \"CLas 为黄龙病病原\", \"confidence\": 0.9}]\n\n"
                f"问题: {query[:500]}\n回答: {answer[:2000]}"
            )
            resp = client.chat.completions.create(
                model=settings.RESOLVED_FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
                timeout=15,
            )
            if session_id:
                from src.core.cache_metrics import emit_usage_from_response
                emit_usage_from_response(session_id, "memory_extraction", resp)
            content = resp.choices[0].message.content.strip()
            content = content.replace("```json", "").replace("```", "").strip()
            import json as _json
            facts = _json.loads(content)
            if isinstance(facts, list):
                return facts[:5]
            return []
        except Exception as e:
            logger.debug(f"[Memory] 提取事实失败: {e}")
            return []

    # ─── Internal Storage ───

    def _load_store(self, session_id: str, store_name: str) -> dict:
        """从 SQLite 读取特定记忆存储（延迟到使用才连接）"""
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            if not db_path.exists():
                return {}
            with _connect_db(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                cur = conn.execute(
                    "SELECT value FROM memory_store WHERE session_id = ? AND key = ?",
                    (session_id, store_name),
                )
                row = cur.fetchone()
                if row:
                    return json.loads(row["value"])
                return {}
        except Exception as e:
            logger.debug(f"[Memory] 加载 {store_name} 失败: {e}")
            return {}

    def _save_store(self, session_id: str, store_name: str, data: dict) -> None:
        # 注: 调用方 set_preference 已持 _mem_lock（RLock 可重入），此处不重复加锁
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            with _connect_db(str(db_path)) as conn:
                # v9.2 CON-8: memory_store 建表收敛共享常量（core/db）
                ensure_memory_store(conn)
                conn.execute(
                    """INSERT OR REPLACE INTO memory_store (session_id, key, value, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, store_name, json.dumps(data, ensure_ascii=False),
                     datetime.now().isoformat()),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[Memory] 保存 {store_name} 失败: {e}")

    def list_user_memory(self, user_id: int) -> dict:
        """Return inspectable memory belonging only to one authenticated account."""
        user_id = int(user_id or 0)
        if user_id <= 0:
            return {"preferences": [], "facts": []}
        try:
            with self._mem_lock, _connect_db(self.db_path) as conn:
                self._ensure_ltm_schema(conn)
                preferences = conn.execute(
                    "SELECT pref_key, pref_value, updated_at FROM account_preferences "
                    "WHERE user_id=? ORDER BY updated_at DESC", (user_id,)).fetchall()
                facts = conn.execute(
                    "SELECT id, fact_key, fact_value, confidence, updated_at FROM ltm_facts "
                    "WHERE owner_user_id=? ORDER BY updated_at DESC LIMIT 200", (user_id,)).fetchall()
            return {
                "preferences": [{"key": r[0], "value": r[1], "updated_at": r[2]} for r in preferences],
                "facts": [{"id": int(r[0]), "key": r[1], "value": r[2],
                           "confidence": float(r[3] or 0), "updated_at": r[4]} for r in facts],
            }
        except Exception as exc:
            logger.warning("[Memory] account memory listing failed: %s", type(exc).__name__)
            return {"preferences": [], "facts": []}

    def clear_user_memory(self, user_id: int) -> bool:
        """Clear account-scoped preferences, facts, and resident cards."""
        user_id = int(user_id or 0)
        if user_id <= 0:
            return False
        try:
            with self._mem_lock, _connect_db(self.db_path) as conn:
                self._ensure_ltm_schema(conn)
                self._ensure_resident_schema(conn)
                conn.execute("DELETE FROM account_preferences WHERE user_id=?", (user_id,))
                conn.execute("DELETE FROM ltm_facts WHERE owner_user_id=?", (user_id,))
                conn.execute("DELETE FROM long_term_memory WHERE owner_user_id=?", (user_id,))
                conn.execute("DELETE FROM account_resident_cards WHERE user_id=?", (user_id,))
                # Remove identifiable legacy sources too. Otherwise an old row could
                # be re-imported (or be left behind after a user asks to clear memory).
                conn.execute(
                    """DELETE FROM memory_store WHERE key='preference_memory' AND session_id IN
                       (SELECT session_id FROM sessions WHERE owner_user_id=?)""", (user_id,))
                conn.execute(
                    """DELETE FROM resident_cards WHERE session_id IN
                       (SELECT session_id FROM sessions WHERE owner_user_id=?)""", (user_id,))
                conn.commit()
            return True
        except Exception as exc:
            logger.warning("[Memory] account memory clear failed: %s", type(exc).__name__)
            return False

    def clear_session(self, session_id: str) -> None:
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            with _connect_db(str(db_path)) as conn:
                # v9.2 CON-8: memory_store 建表收敛共享常量（core/db）
                ensure_memory_store(conn)
                conn.execute("DELETE FROM memory_store WHERE session_id = ?", (session_id,))
                conn.commit()
            logger.info(f"[Memory] 会话 {session_id[:8]}... 记忆已清空")
        except Exception as e:
            logger.error(f"[Memory] 清空记忆失败: {e}")


memory_store = MemoryStore()
