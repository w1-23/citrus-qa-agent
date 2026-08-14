import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

def _tokenize(text: str) -> List[str]:
    text = text.lower()
    return [t for t in re.findall(r"[a-z0-9_\-]+|[\u4e00-\u9fff]+", text) if len(t) > 1]

class BM25Plus:
    def __init__(self, k1: float = 1.5, b: float = 0.75, delta: float = 1.0):
        self.k1, self.b, self.delta = k1, b, delta
        self.corpus_size = 0
        self.avgdl = 0.0
        self.idf = {}
        self.doc_lens = []
        self.tokenized_corpus = []
        # v8.6 倒排索引（term -> [(doc_idx, tf), ...]）：
        # fit 时同步构建；top_k 优先走倒排（结果与全量枚举逐位一致，
        # 查询从 O(全部文档) 降到 O(命中项)）；持久化缓存只存倒排，省内存。
        self.inv: Dict[str, List[Tuple[int, float]]] = {}

    def fit(self, corpus: List[str]):
        self.corpus_size = len(corpus)
        if self.corpus_size == 0: return
        
        self.tokenized_corpus = [_tokenize(doc) for doc in corpus]
        self.doc_lens = [len(tokens) for tokens in self.tokenized_corpus]
        self.avgdl = sum(self.doc_lens) / self.corpus_size
        
        df = Counter()
        for tokens in self.tokenized_corpus:
            df.update(set(tokens))
            
        self.idf = {
            term: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._build_inverted_index()

    def _build_inverted_index(self):
        """从 tokenized_corpus 构建倒排索引（评分公式与全量枚举完全一致）。"""
        self.inv = {}
        for i, tokens in enumerate(self.tokenized_corpus):
            tf = Counter(tokens)
            for term, f in tf.items():
                self.inv.setdefault(term, []).append((i, f))

    def top_k(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens or self.corpus_size == 0: return []
        if self.inv:
            return self._top_k_inverted(query_tokens, k)
        return self._top_k_scan(query_tokens, k)

    def _top_k_scan(self, query_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        """全量枚举（旧路径；无倒排索引时兜底，如手工构造的实例）。"""
        scores = []
        for i, doc_tokens in enumerate(self.tokenized_corpus):
            doc_tf = Counter(doc_tokens)
            score = 0.0
            doc_len = self.doc_lens[i]
            for qt in query_tokens:
                if qt not in self.idf: continue
                tf = doc_tf.get(qt, 0)
                num = tf + self.delta * tf if tf > 0 else 0.0
                den = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                score += self.idf[qt] * num / den if den > 0 else 0.0
            scores.append(score)
        indexed = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in indexed if score > 0][:k]

    def _top_k_inverted(self, query_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        """倒排路径：仅统计包含查询词的文档，累加顺序与全量枚举一致
        （外层按查询词、内层按文档），评分结果逐位等价。"""
        scores: Dict[int, float] = {}
        for qt in query_tokens:
            idf = self.idf.get(qt)
            if idf is None:
                continue
            for doc_idx, tf in self.inv.get(qt, ()):
                doc_len = self.doc_lens[doc_idx]
                num = tf + self.delta * tf
                den = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avgdl)
                if den > 0:
                    scores[doc_idx] = scores.get(doc_idx, 0.0) + idf * num / den
        indexed = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(idx, score) for idx, score in indexed if score > 0][:k]

def rrf_fuse(*hit_lists: List[Tuple[int, float]], k: int = 60, weights: List[float] | None = None) -> List[Tuple[int, float]]:
    """Weighted Reciprocal Rank Fusion.

    Accepts any number of ranked hit lists and optional per-list weights.
    Without weights → equal-weight RRF (backward compatible).
    With weights → each list's rank score multiplied by its weight.
    """
    scores: defaultdict[int, float] = defaultdict(float)
    if weights is None:
        weights = [1.0] * len(hit_lists)
    for hit_list, w in zip(hit_lists, weights):
        for rank, (idx, _) in enumerate(hit_list):
            scores[idx] += w / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


# ── v8.6 索引持久化（书 §3.2 离线建索引：缓存避免每次启动重建 ~10s）──

BM25_CACHE_FORMAT = 1


def bm25_to_cache_dict(bm: BM25Plus) -> dict:
    """序列化 BM25 索引为紧凑 dict（不含 tokenized_corpus，省内存）。"""
    return {
        "format": BM25_CACHE_FORMAT,
        "k1": bm.k1, "b": bm.b, "delta": bm.delta,
        "corpus_size": bm.corpus_size,
        "avgdl": bm.avgdl,
        "idf": bm.idf,
        "doc_lens": bm.doc_lens,
        "inv": bm.inv,
    }


def bm25_from_cache_dict(d: dict) -> BM25Plus:
    """从缓存 dict 还原 BM25Plus（tokenized_corpus 为空，top_k 走倒排路径）。"""
    bm = BM25Plus(k1=d["k1"], b=d["b"], delta=d["delta"])
    bm.corpus_size = d["corpus_size"]
    bm.avgdl = d["avgdl"]
    bm.idf = d["idf"]
    bm.doc_lens = d["doc_lens"]
    bm.inv = d["inv"]
    return bm
