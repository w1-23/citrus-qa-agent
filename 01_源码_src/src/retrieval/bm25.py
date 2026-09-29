import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

import numpy as np

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
        # v8.11: doc_lens 用 numpy 数组（评分路径向量化读取）
        self.doc_lens = np.array([len(tokens) for tokens in self.tokenized_corpus],
                                 dtype=np.uint32)
        self.avgdl = float(self.doc_lens.sum()) / self.corpus_size

        df = Counter()
        for tokens in self.tokenized_corpus:
            df.update(set(tokens))

        self.idf = {
            term: math.log(1 + (self.corpus_size - freq + 0.5) / (freq + 0.5))
            for term, freq in df.items()
        }
        self._build_inverted_index()
        # v8.11: 倒排建成后释放原文分词（省内存；正常查询走倒排路径，
        # _top_k_scan 在 tokenized_corpus 为 None 时从倒排重建兜底）
        self.tokenized_corpus = None

    def _build_inverted_index(self):
        """从 tokenized_corpus 构建倒排索引（评分公式与全量枚举完全一致）。

        v8.11: 由 List[Tuple] 改为 numpy 平行数组 (doc_ids uint32[], tfs uint32[])——
        去掉每个 posting 一个 tuple 的对象开销，内存降 4-6 倍（36 万 chunk 级
        语料倒排从 ~1GB+ 降至 ~300MB 内）。
        """
        postings: Dict[str, List[Tuple[int, int]]] = {}
        for i, tokens in enumerate(self.tokenized_corpus):
            tf = Counter(tokens)
            for term, f in tf.items():
                postings.setdefault(term, []).append((i, f))
        self.inv = {
            term: (np.array([p[0] for p in pairs], dtype=np.uint32),
                   np.array([p[1] for p in pairs], dtype=np.uint32))
            for term, pairs in postings.items()
        }

    def top_k(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens or self.corpus_size == 0: return []
        if self.inv:
            return self._top_k_inverted(query_tokens, k)
        return self._top_k_scan(query_tokens, k)

    def _top_k_scan(self, query_tokens: List[str], k: int) -> List[Tuple[int, float]]:
        """全量枚举（旧路径；无倒排索引时兜底，如手工构造的实例）。

        v8.11: fit 后 tokenized_corpus 已释放（None）→ 从倒排重建每文档 term
        计数再枚举，评分结果与倒排路径逐位一致（正常查询不会走到此路径，
        仅测试/手工构造场景）。
        """
        corpus = self.tokenized_corpus
        if corpus is None:
            corpus = [Counter() for _ in range(self.corpus_size)]
            for term, (ids, tfs) in self.inv.items():
                for doc_idx, tf in zip(ids.tolist(), tfs.tolist()):
                    corpus[doc_idx][term] = int(tf)
        scores = []
        for i, doc_tokens in enumerate(corpus):
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
        """倒排路径（v8.11 numpy 平行数组向量化）：仅统计包含查询词的文档，
        累加顺序与全量枚举一致（外层按查询词、内层按文档），结果逐位等价。"""
        scores: Dict[int, float] = {}
        for qt in query_tokens:
            idf = self.idf.get(qt)
            if idf is None:
                continue
            entry = self.inv.get(qt)
            if entry is None:
                continue
            ids, tfs = entry
            dl = self.doc_lens[ids]                     # uint32 -> float64 精确（<2^53）
            num = tfs + self.delta * tfs                # float64
            # den = tf + k1*(1-b+b*dl/avgdl)；k1>0 且 (1-b+b*dl/avgdl)>0 → den>0 恒成立
            den = tfs + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            contrib = idf * num / den
            for idx, s in zip(ids.tolist(), contrib.tolist()):
                scores[idx] = scores.get(idx, 0.0) + s
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
# v8.11: 格式升级为 numpy 平行数组倒排（旧 v1 缓存自动失效重建，指纹含版本号）

BM25_CACHE_FORMAT = 2


def bm25_to_cache_dict(bm: BM25Plus) -> dict:
    """序列化 BM25 索引为紧凑 dict（不含 tokenized_corpus，省内存）。

    v8.11: inv 为 {term: (doc_ids uint32[], tfs uint32[])}，numpy 数组 pickle
    往返无损；doc_lens 同样为 numpy 数组。
    """
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
