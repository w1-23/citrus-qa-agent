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
import sqlite3
import threading
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MemoryStore:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._mem_lock = threading.Lock()
        return cls._instance

    def __init__(self):
        from src.config import PROJECT_ROOT
        # v8.4: DB 路径实例化属性（测试可定向到临时库；此前硬编码 PROJECT_ROOT）
        self.db_path = str(PROJECT_ROOT / "state" / "sessions.db")

    # ─── Entity Memory ───

    def update_entities(self, session_id: str, entities: List[Dict]) -> None:
        """批量更新实体记忆"""
        with self._mem_lock:
            store = self._load_store(session_id, "entity_memory")
            for ent in entities:
                name = ent.get("name", "")
                if not name:
                    continue
                if name in store:
                    store[name]["last_mentioned"] = datetime.now().isoformat()
                    existing_rels = set(store[name].get("relations", []))
                    new_rels = set(ent.get("relations", []))
                    store[name]["relations"] = list(existing_rels | new_rels)
                    store[name]["mention_count"] = store[name].get("mention_count", 0) + 1
                else:
                    store[name] = {
                        "type": ent.get("type", "unknown"),
                        "aliases": ent.get("aliases", []),
                        "relations": ent.get("relations", []),
                        "first_mentioned": datetime.now().isoformat(),
                        "last_mentioned": datetime.now().isoformat(),
                        "mention_count": 1,
                    }
            self._save_store(session_id, "entity_memory", store)

    def get_entities(self, session_id: str) -> Dict:
        return self._load_store(session_id, "entity_memory")

    # ─── Concept Memory ───

    def update_concept(self, session_id: str, concept_key: str, summary: str, confidence: float = 0.5) -> None:
        with self._mem_lock:
            store = self._load_store(session_id, "concept_memory")
            if concept_key in store:
                store[concept_key]["summary"] = summary
                store[concept_key]["confidence"] = max(store[concept_key]["confidence"], confidence)
                store[concept_key]["last_mentioned"] = datetime.now().isoformat()
                store[concept_key]["mention_count"] += 1
            else:
                store[concept_key] = {
                    "summary": summary,
                    "confidence": confidence,
                    "first_mentioned": datetime.now().isoformat(),
                    "last_mentioned": datetime.now().isoformat(),
                    "mention_count": 1,
                }
            self._save_store(session_id, "concept_memory", store)

    # ─── Preference Memory ───

    def set_preference(self, session_id: str, key: str, value: str) -> None:
        with self._mem_lock:
            store = self._load_store(session_id, "preference_memory")
            store[key] = {"value": value, "updated_at": datetime.now().isoformat()}
            self._save_store(session_id, "preference_memory", store)

    def get_preferences(self, session_id: str, max_chars: int = 300) -> str:
        """用户显式偏好（书 §3.1 偏好追踪）→ 上下文块（≤max_chars，空则 ""）。

        v8.6 消费点接线：LTM 提取出 type=preference 的事实写入本表，
        这里在 build_human_message 中作为 <user_preferences> 块注入——
        偏好是用户明确表达的长期约定（如"综述一律中文、要含局限与边界"），
        注入后模型在写作/回答中自动遵循，无需用户每次重申。
        """
        store = self._load_store(session_id, "preference_memory")
        if not store:
            return ""
        parts: list[str] = []
        total = 0
        for key, item in list(store.items())[:10]:
            value = item.get("value", "") if isinstance(item, dict) else str(item)
            if not value:
                continue
            line = f"- {str(key)[:60]}: {str(value)[:120]}"
            if total + len(line) > max_chars:
                break
            parts.append(line)
            total += len(line)
        if not parts:
            return ""
        return ("## 用户偏好（历史交互中用户明确表达的偏好；如与用户最新要求冲突，"
                "以用户最新要求为准）\n" + "\n".join(parts))

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
        conn.execute(
            """CREATE TABLE IF NOT EXISTS memory_store (
                session_id TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                PRIMARY KEY (session_id, key))"""
        )
        migrated = conn.execute(
            "SELECT COUNT(*) FROM memory_store WHERE key='_ltm_migrated_v2'"
        ).fetchone()[0]
        if not migrated:
            conn.execute(
                """INSERT INTO ltm_facts (fact_key, fact_value, confidence,
                                          updated_at, owner_session, source_query)
                   SELECT fact_key, fact_value, confidence, updated_at,
                          owner_session, source_query
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
                updated_at TEXT
            )"""
        )

    RESIDENT_CARD_MIN_CONF = 0.8     # 高置信度事实才进常驻卡片
    RESIDENT_CARD_MAX = 8            # 常驻卡片上限（约 ≤500 字符预算）
    RESIDENT_CARD_MAX_CHARS = 60     # 单条卡片值上限

    def _upsert_resident_card(self, conn, fact_key: str, fact_value: str,
                              confidence: float) -> None:
        self._ensure_resident_schema(conn)
        if not fact_key or not fact_value:
            return
        if confidence < self.RESIDENT_CARD_MIN_CONF:
            return
        if len(fact_value) > self.RESIDENT_CARD_MAX_CHARS:
            return
        conn.execute(
            """INSERT INTO resident_cards (fact_key, fact_value, confidence, updated_at)
               VALUES (?, ?, ?, ?)
               ON CONFLICT(fact_key) DO UPDATE SET
                 fact_value=excluded.fact_value,
                 confidence=excluded.confidence,
                 updated_at=excluded.updated_at""",
            (fact_key, fact_value, confidence, datetime.now().isoformat()),
        )
        # 超上限淘汰：最低置信度 + 最旧的先出（代码维护，非 LLM）
        cur = conn.execute(
            "SELECT COUNT(*) FROM resident_cards")
        if (cur.fetchone()[0] or 0) > self.RESIDENT_CARD_MAX:
            conn.execute(
                """DELETE FROM resident_cards WHERE fact_key IN (
                    SELECT fact_key FROM resident_cards
                    ORDER BY confidence ASC, updated_at ASC
                    LIMIT 1)"""
            )

    def get_resident_cards(self, max_chars: int = 500) -> str:
        """常驻卡片（双层记忆的'概览'层）：按置信度取 top-N 拼文本，≤max_chars。"""
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            if not db_path.exists():
                return ""
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                self._ensure_resident_schema(conn)
                rows = conn.execute(
                    "SELECT fact_key, fact_value, confidence, updated_at "
                    "FROM resident_cards ORDER BY confidence DESC, updated_at DESC"
                ).fetchall()
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
                            owner_session: str = "", source_query: str = "") -> bool:
        """v8.4 (Mem0 v3): ADD-only 写入——同 key 多版本并存，绝不覆盖/删除历史。
        检索阶段用时间+置信度排序消歧。置信度 <0.5 的未核验事实拒绝写入。
        """
        if not fact_key or not fact_value:
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
                with sqlite3.connect(str(db_path)) as conn:
                    self._ensure_ltm_schema(conn)
                    conn.execute(
                        """INSERT INTO ltm_facts
                           (fact_key, fact_value, confidence, updated_at,
                            owner_session, source_query)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (fact_key, fact_value, confidence, datetime.now().isoformat(),
                         owner_session, source_query),
                    )
                    self._upsert_resident_card(conn, fact_key, fact_value, confidence)
                    conn.commit()
                return True
            except Exception as e:
                logger.debug(f"[Memory] 保存长期事实失败: {e}")
                return False

    def _fetch_ltm_rows(self, conn, limit: int = 500):
        """v8.4: 从 ltm_facts 读取；若新表为空回退老表（迁移前兼容）。"""
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT fact_key, fact_value, confidence, updated_at, owner_session, source_query "
            "FROM ltm_facts ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        if not rows:
            rows = conn.execute(
                "SELECT fact_key, fact_value, confidence, updated_at, owner_session, source_query "
                "FROM long_term_memory ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return rows

    def recall_long_term_memory(self, query: str, top_k: int = 5,
                                owner_session: str = "", max_chars: int = 1500) -> str:
        """语义向量检索跨会话长期记忆 (fallback 到关键词)。
        AG-5: 置信度按时间衰减(0.95/天)，输出带来源与更新时间标注，超 max_chars 截断。"""
        try:
            return self._recall_semantic(query, top_k, owner_session, max_chars)
        except Exception as e:
            logger.debug(f"[Memory] 语义召回失败, 回退关键词: {e}")
            return self._recall_keyword_fallback(query, top_k, owner_session, max_chars)

    def _recall_semantic(self, query: str, top_k: int, owner_session: str, max_chars: int) -> str:
        import numpy as np
        import sqlite3
        from pathlib import Path

        db_path = Path(self.db_path)
        if not db_path.exists():
            return ""
        with sqlite3.connect(str(db_path)) as conn:
            self._ensure_ltm_schema(conn)
            rows = self._fetch_ltm_rows(conn)
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
            eff_confs[i] = conf * (0.95 ** days)  # AG-5: 时间衰减

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

    def _recall_keyword_fallback(self, query: str, top_k: int, owner_session: str, max_chars: int) -> str:
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            if not db_path.exists():
                return ""
            with sqlite3.connect(str(db_path)) as conn:
                self._ensure_ltm_schema(conn)
                rows = self._fetch_ltm_rows(conn, limit=200)
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

    def extract_key_facts(self, query: str, answer: str) -> list[dict]:
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
            client = OpenAI(api_key=settings.RESOLVED_MAIN_API_KEY, base_url=settings.MAIN_BASE_URL)
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
                model=settings.FAST_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1000,
                timeout=15,
            )
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
            with sqlite3.connect(str(db_path)) as conn:
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
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_store ("
                    "session_id TEXT NOT NULL, "
                    "key TEXT NOT NULL, "
                    "value TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL, "
                    "PRIMARY KEY (session_id, key))"
                )
                conn.execute(
                    """INSERT OR REPLACE INTO memory_store (session_id, key, value, updated_at)
                       VALUES (?, ?, ?, ?)""",
                    (session_id, store_name, json.dumps(data, ensure_ascii=False),
                     datetime.now().isoformat()),
                )
                conn.commit()
        except Exception as e:
            logger.error(f"[Memory] 保存 {store_name} 失败: {e}")

    def clear_session(self, session_id: str) -> None:
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = Path(self.db_path)
            with sqlite3.connect(str(db_path)) as conn:
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS memory_store ("
                    "session_id TEXT NOT NULL, "
                    "key TEXT NOT NULL, "
                    "value TEXT NOT NULL, "
                    "updated_at TEXT NOT NULL, "
                    "PRIMARY KEY (session_id, key))"
                )
                conn.execute("DELETE FROM memory_store WHERE session_id = ?", (session_id,))
                conn.commit()
            logger.info(f"[Memory] 会话 {session_id[:8]}... 记忆已清空")
        except Exception as e:
            logger.error(f"[Memory] 清空记忆失败: {e}")


# ─── Entity Extraction Helpers ───

_ENTITY_PATTERNS = {
    "病害": re.compile(r'(黄龙病|溃疡病|枯萎病|砂皮病|炭疽病|疮痂病|脂点黄斑病|根腐病|青霉病|绿霉病)'),
    "品种": re.compile(r'(沃柑|丑橘|砂糖橘|蜜桔|脐橙|血橙|葡萄柚|柠檬|柚子|金桔|佛手|枸橼)'),
    "化合物": re.compile(r'\b([A-Za-z]+(?:酸|素|醇|醛|酮|酯|胺|苷|霉素))\b'),
    "基因/蛋白": re.compile(r'\b([A-Z][a-z]+(?:\d+)?(?:基因|蛋白|因子|酶))\b|\b(Cs[A-Z][a-z]+\d*)\b'),
    "数值指标": re.compile(r'(\d+(?:\.\d+)?\s*(?:μM|mM|m?g/mL|g/L|%|°C|pH|bp|kb|kDa))'),
}




memory_store = MemoryStore()
