import json, time, pickle
from pathlib import Path
from src.config import settings
from src.retrieval.bm25 import BM25Plus
from collections import Counter

chunks = []
for b in sorted(settings.DATA_DIR.iterdir()):
    p = b / "chunks" / "chunks.jsonl"
    if not p.exists():
        continue
    for line in open(p, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        chunks.append(json.loads(line))
texts = [f"{c.get('section_name','')} {c.get('text','')}" for c in chunks]
t0 = time.time()
bm = BM25Plus()
bm.fit(texts)
fit_sec = time.time() - t0
print(f"fit={fit_sec:.1f}s avgdl={bm.avgdl:.0f} terms={len(bm.idf)}")

t0 = time.time()
inv = {}
for i, toks in enumerate(bm.tokenized_corpus):
    tf = Counter(toks)
    for term, f in tf.items():
        inv.setdefault(term, []).append((i, f))
print(f"inverted build={time.time()-t0:.1f}s terms={len(inv)} pairs={sum(len(v) for v in inv.values())}")

t0 = time.time()
data = pickle.dumps(
    {"inv": inv, "idf": bm.idf, "avgdl": bm.avgdl,
     "corpus_size": bm.corpus_size, "doc_lens": bm.doc_lens},
    protocol=4)
print(f"pickle={time.time()-t0:.1f}s size={len(data)/1e6:.0f}MB")
