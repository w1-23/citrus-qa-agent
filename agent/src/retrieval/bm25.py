import math
import re
from collections import Counter, defaultdict
from typing import List, Tuple

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

    def top_k(self, query: str, k: int = 20) -> List[Tuple[int, float]]:
        query_tokens = _tokenize(query)
        if not query_tokens or self.corpus_size == 0: return []
        
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