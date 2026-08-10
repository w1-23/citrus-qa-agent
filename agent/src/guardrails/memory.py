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
        pass

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

    def get_recent_entities(self, session_id: str, top_k: int = 20) -> List[Dict]:
        """按 last_mentioned 排序的最近实体"""
        store = self._load_store(session_id, "entity_memory")
        sorted_ents = sorted(
            store.items(),
            key=lambda x: x[1].get("last_mentioned", ""),
            reverse=True,
        )
        return [{"name": k, **v} for k, v in sorted_ents[:top_k]]

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

    def get_concepts(self, session_id: str) -> Dict:
        return self._load_store(session_id, "concept_memory")

    # ─── Preference Memory ───

    def set_preference(self, session_id: str, key: str, value: str) -> None:
        with self._mem_lock:
            store = self._load_store(session_id, "preference_memory")
            store[key] = {"value": value, "updated_at": datetime.now().isoformat()}
            self._save_store(session_id, "preference_memory", store)

    def get_preference(self, session_id: str, key: str, default: Optional[str] = None) -> Optional[str]:
        store = self._load_store(session_id, "preference_memory")
        entry = store.get(key)
        return entry["value"] if entry else default

    def get_all_preferences(self, session_id: str) -> Dict:
        return self._load_store(session_id, "preference_memory")

    # ─── Long-term Memory (Cross-session) ───

    def save_long_term_fact(self, fact_key: str, fact_value: str, confidence: float = 0.5) -> None:
        """跨会话长期事实存储（不绑定 session_id）"""
        with self._mem_lock:
            try:
                import sqlite3
                from pathlib import Path
                from src.config import PROJECT_ROOT
                db_path = PROJECT_ROOT / "state" / "sessions.db"
                db_path.parent.mkdir(parents=True, exist_ok=True)
                with sqlite3.connect(str(db_path)) as conn:
                    conn.execute(
                        """CREATE TABLE IF NOT EXISTS long_term_memory (
                            fact_key TEXT PRIMARY KEY,
                            fact_value TEXT,
                            confidence REAL DEFAULT 0.5,
                            updated_at TEXT
                        )"""
                    )
                    conn.execute(
                        """INSERT OR REPLACE INTO long_term_memory (fact_key, fact_value, confidence, updated_at)
                           VALUES (?, ?, ?, ?)""",
                        (fact_key, fact_value, confidence, datetime.now().isoformat()),
                    )
                    conn.commit()
            except Exception as e:
                logger.debug(f"[Memory] 保存长期事实失败: {e}")

    def recall_long_term_memory(self, query: str, top_k: int = 5) -> str:
        """语义向量检索跨会话长期记忆 (fallback 到关键词)"""
        try:
            return self._recall_semantic(query, top_k)
        except Exception as e:
            logger.debug(f"[Memory] 语义召回失败, 回退关键词: {e}")
            return self._recall_keyword_fallback(query, top_k)

    def _recall_semantic(self, query: str, top_k: int) -> str:
        import numpy as np
        import sqlite3
        from pathlib import Path
        from src.config import PROJECT_ROOT

        db_path = PROJECT_ROOT / "state" / "sessions.db"
        if not db_path.exists():
            return ""
        with sqlite3.connect(str(db_path)) as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT fact_key, fact_value, confidence, updated_at FROM long_term_memory ORDER BY updated_at DESC LIMIT 500"
            ).fetchall()
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

        top_indices = np.argsort(scores)[::-1][:top_k]
        parts = []
        for i in top_indices:
            score = float(scores[i])
            if score < 0.30:
                continue
            row = rows[i]
            parts.append(
                f"- [{row['confidence']:.1f}] {row['fact_value']} "
                f"(更新: {row['updated_at'][:10]})"
            )
        if not parts:
            return ""
        return "## 跨会话记忆\n以下为历史会话中提取的相关事实：\n" + "\n".join(parts)

    def _recall_keyword_fallback(self, query: str, top_k: int) -> str:
        try:
            import sqlite3
            from pathlib import Path
            from src.config import PROJECT_ROOT
            db_path = PROJECT_ROOT / "state" / "sessions.db"
            if not db_path.exists():
                return ""
            with sqlite3.connect(str(db_path)) as conn:
                conn.row_factory = sqlite3.Row
                rows = conn.execute(
                    "SELECT fact_key, fact_value, confidence, updated_at FROM long_term_memory ORDER BY updated_at DESC LIMIT 200"
                ).fetchall()
            query_tokens = set(re.findall(r'[\w\u4e00-\u9fff]{2,}', query.lower()))
            scored = []
            for row in rows:
                key_tokens = set(re.findall(r'[\w\u4e00-\u9fff]{2,}', row["fact_key"].lower()))
                overlap = len(query_tokens & key_tokens)
                if overlap > 0:
                    scored.append((overlap * row["confidence"], row["fact_value"], row["confidence"], row["updated_at"]))
            scored.sort(key=lambda x: x[0], reverse=True)
            if not scored:
                return ""
            parts = []
            for score, value, conf, ts in scored[:top_k]:
                parts.append(f"- [{conf:.1f}] {value} (更新: {ts[:10]})")
            return "## 跨会话记忆\n以下为历史会话中提取的相关事实：\n" + "\n".join(parts)
        except Exception:
            return ""

    def extract_key_facts(self, query: str, answer: str) -> list[dict]:
        """提取 3-5 条核心事实（异步调用，用最便宜模型）"""
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
            db_path = PROJECT_ROOT / "state" / "sessions.db"
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
            db_path = PROJECT_ROOT / "state" / "sessions.db"
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
            db_path = PROJECT_ROOT / "state" / "sessions.db"
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


def extract_entities(text: str) -> List[Dict]:
    """从文本中提取科研实体"""
    entities = []
    seen: Set[str] = set()
    for etype, pattern in _ENTITY_PATTERNS.items():
        for match in pattern.finditer(text):
            name = match.group(0).strip()
            if name and name not in seen:
                seen.add(name)
                entities.append({"name": name, "type": etype})
    return entities


def extract_entity_relations(text: str, entities: List[Dict]) -> List[Tuple[str, str, str]]:
    """简单启发式关系提取：同一句中出现的实体互为关联"""
    relations = []
    sentences = re.split(r'[。；\n]', text)
    ent_names = {e["name"] for e in entities}
    for sent in sentences:
        present = [e for e in ent_names if e in sent]
        if len(present) >= 2:
            for i in range(len(present)):
                for j in range(i + 1, len(present)):
                    relations.append((present[i], "related_to", present[j]))
    return relations


memory_store = MemoryStore()
