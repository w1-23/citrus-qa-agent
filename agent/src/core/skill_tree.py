"""
SkillTree — writing skill semantic matching via fastembed
384 skills from scientific papers, organized by section/rhetorical_move
"""
import os
import hashlib
import json
import logging
import threading
import time
from pathlib import Path

# 默认离线模式：不连 HuggingFace，直接用本地缓存。设 HF_HUB_OFFLINE=0 可联网检查模型更新。
os.environ.setdefault("HF_HUB_OFFLINE", "1")
from typing import List, Dict, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)

_SKILLS_DIR = Path(__file__).resolve().parent.parent / "skills"
_INDEX_PATH = _SKILLS_DIR / "skills_index.json"
_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".hf_cache"
_CACHE_VECTORS = _CACHE_DIR / "skill_vectors.npy"
_CACHE_HASH = _CACHE_DIR / "skill_texts_hash.txt"


class SkillTree:
    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._loaded = False
        return cls._instance

    def __init__(self):
        if self._loaded:
            return
        with self._lock:
            if self._loaded:
                return
            self._skills: List[dict] = []
            self._texts: List[str] = []
            self._vectors: Optional[np.ndarray] = None
            self._load_index()
            self._build_vectors()
            self._loaded = True

    def _load_index(self):
        if not _INDEX_PATH.exists():
            logger.warning(f"[SkillTree] {_INDEX_PATH} not found")
            return
        with open(_INDEX_PATH, encoding="utf-8") as f:
            data = json.load(f)
        self._skills = data.get("skills", [])
        self._texts = []
        for s in self._skills:
            domains = ", ".join(s.get("domain", [])[:3])
            text = (
                f"{s.get('name', '')} {s.get('section', '')} "
                f"{s.get('rhetorical_move', '')} {domains} "
                f"{' '.join(s.get('keywords', [])[:8])} "
                f"{s.get('text_preview', '')[:200]}"
            )
            self._texts.append(text)
        logger.info(f"[SkillTree] loaded {len(self._skills)} skills")

    def _compute_texts_hash(self) -> str:
        return hashlib.md5("".join(self._texts).encode("utf-8")).hexdigest()

    def _load_cached_vectors(self) -> Optional[np.ndarray]:
        try:
            if _CACHE_VECTORS.exists() and _CACHE_HASH.exists():
                cached_hash = _CACHE_HASH.read_text(encoding="utf-8").strip()
                current_hash = self._compute_texts_hash()
                if cached_hash == current_hash and self._texts:
                    vectors = np.load(str(_CACHE_VECTORS))
                    logger.info(f"[SkillTree] loaded cached vectors: {vectors.shape}")
                    return vectors
                else:
                    logger.info("[SkillTree] cache hash mismatch, rebuilding")
        except Exception as e:
            logger.warning(f"[SkillTree] cache load failed: {e}")
        return None

    def _save_cached_vectors(self, vectors: np.ndarray):
        try:
            _CACHE_DIR.mkdir(parents=True, exist_ok=True)
            np.save(str(_CACHE_VECTORS), vectors)
            _CACHE_HASH.write_text(self._compute_texts_hash(), encoding="utf-8")
            logger.info(f"[SkillTree] cached vectors saved: {vectors.shape}")
        except Exception as e:
            logger.warning(f"[SkillTree] cache save failed: {e}")

    def _build_vectors(self):
        if not self._texts:
            return
        cached = self._load_cached_vectors()
        if cached is not None:
            self._vectors = cached
            return
        try:
            from fastembed import TextEmbedding
            t0 = time.perf_counter()
            model = TextEmbedding("intfloat/multilingual-e5-large")
            vecs = list(model.embed(self._texts))
            self._vectors = np.array(vecs, dtype=np.float32)
            dt = (time.perf_counter() - t0) * 1000
            logger.info(f"[SkillTree] built vectors: {self._vectors.shape} ({dt:.0f}ms)")
            self._save_cached_vectors(self._vectors)
        except Exception as e:
            logger.warning(f"[SkillTree] vector build failed: {e}")

    _embedder_lock = threading.Lock()
    _shared_embedder = None

    def _get_embedder(self):
        if self._shared_embedder is None:
            with self._embedder_lock:
                if self._shared_embedder is None:
                    from fastembed import TextEmbedding
                    self._shared_embedder = TextEmbedding("intfloat/multilingual-e5-large")
        return self._shared_embedder

    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """Semantic search for matching skills. Returns [(skill_id, score, metadata), ...]."""
        if self._vectors is None or not self._skills:
            return []

        try:
            model = self._get_embedder()
            q_vec = np.array(list(model.embed([query]))[0], dtype=np.float32)
        except Exception:
            return []

        q_norm = q_vec / np.linalg.norm(q_vec)
        v_norm = self._vectors / np.linalg.norm(self._vectors, axis=1, keepdims=True)
        scores = (q_norm @ v_norm.T)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for i in top_indices:
            if scores[i] < 0.4:
                continue
            results.append((
                self._skills[i]["id"],
                float(scores[i]),
                self._skills[i],
            ))
        return results

    def load_content(self, skill_id: str) -> str:
        """Load the full content of a skill .md file."""
        for s in self._skills:
            if s["id"] == skill_id:
                file_path = _SKILLS_DIR / s["file"]
                if file_path.exists():
                    return file_path.read_text(encoding="utf-8")
        return ""

    def get_search_prompt(self, query: str, top_k: int = 5) -> str:
        """Generate a prompt block showing matching skills for the planner."""
        results = self.search(query, top_k)
        if not results:
            return ""

        lines = [f"## 匹配的写作技能 ({len(results)} 个)"]
        for skill_id, score, meta in results:
            lines.append(
                f"- [{score:.2f}] {meta['name']} | {meta['section']} | "
                f"{', '.join(meta.get('domain', [])[:2])} | {meta['rhetorical_move']} "
                f"(id: {skill_id})"
            )
        return "\n".join(lines)

    def search_strategy_cards(self, query: str, card_type: str = None, top_k: int = 5) -> str:
        """Search strategy cards and return formatted prompt block."""
        from src.core.strategy_cards import get_all_cards, get_card_texts, build_card_map, load_card_prompt

        cards = get_all_cards()
        if card_type:
            cards = [c for c in cards if c["type"] == card_type]
        if not cards:
            return ""

        texts = [f"{c['name']} {c['description']} {c['prompt'][:200]}" for c in cards]
        card_ids = [c["id"] for c in cards]

        try:
            model = self._get_embedder()
            q_vec = np.array(list(model.embed([query]))[0], dtype=np.float32)
            c_vecs = np.array(list(model.embed(texts)), dtype=np.float32)
        except Exception:
            return ""

        q_norm = q_vec / np.linalg.norm(q_vec)
        c_norm = c_vecs / np.linalg.norm(c_vecs, axis=1, keepdims=True)
        scores = (q_norm @ c_norm.T)

        top_idx = np.argsort(scores)[::-1][:top_k]
        lines = []
        for i in top_idx:
            if scores[i] < 0.35:
                continue
            c = cards[i]
            lines.append(f"- [{scores[i]:.2f}] [{c['type']}] {c['name']}: {c['description']} (id:{c['id']})")
        if not lines:
            return ""
        return "## 匹配的策略卡片\n" + "\n".join(lines) + "\n"

    def load_strategy_prompts(self, card_ids: List[str]) -> str:
        """Load full prompt content for strategy cards by ID."""
        from src.core.strategy_cards import load_card_prompt
        parts = []
        for cid in card_ids:
            p = load_card_prompt(cid)
            if p:
                parts.append(p)
        return "\n".join(parts) if parts else ""

    def get_stats(self) -> str:
        """Category stats for planner display."""
        sections = {}
        for s in self._skills:
            sec = s.get("section", "unknown")
            move = s.get("rhetorical_move", "unknown")
            sections.setdefault(sec, {}).setdefault(move, 0)
            sections[sec][move] += 1
        lines = ["## Skill 系统 (384 个)"]
        for sec, moves in sorted(sections.items()):
            total = sum(moves.values())
            move_strs = [f"{m}:{c}" for m, c in sorted(moves.items())]
            lines.append(f"  {sec}({total}): {', '.join(move_strs)}")
        return "\n".join(lines)
