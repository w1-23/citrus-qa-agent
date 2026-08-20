# -*- coding: utf-8 -*-
"""probe_scores_dist — 双语 rerank 分数分布（F3 视觉证据）。

对 5 对同义中英查询：合并 vector top40 + BM25 top40（去重）→ 直接 rerank
（不经 RRF/阈值，以保留全量分数）→ 输出 top-k 内全部 rerank_score。
比较两语言的分数分布：中文文献型表述整体偏低 ⇒ 阈值拦截（箱线图 fig10）。

输出: logs/scores_dist.json
用法: python tests/probe_scores_dist.py [--cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

_file_dir = Path(__file__).resolve().parent if Path(__file__).resolve().name != "<string>" else None
PROJ = (Path.cwd() if (Path.cwd() / "src").is_dir()
        else (_file_dir.parent if _file_dir else Path.cwd()))
sys.path.insert(0, str(PROJ))

PAIRS = [
    ("Nagami kumquat 的品种特征是什么？", "What are the characteristics of the Nagami kumquat cultivar?"),
    ("Poncirus trifoliata 作为柑橘砧木的作用", "The role of Poncirus trifoliata as a citrus rootstock"),
    ("血橙中花青素积累的分子机制有哪些报道？",
     "What molecular mechanisms are reported for anthocyanin accumulation in blood oranges?"),
    ("kumquat（金柑）属的物种学名及其与柑橘属的关系",
     "The scientific name of the kumquat genus and its relationship to Citrus"),
    ("Valencia orange 果汁加工用途的品种特性",
     "Valencia orange characteristics for juice processing"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--topk", type=int, default=40)
    args = ap.parse_args()
    if args.cpu:
        import src.engine.hardware as hw
        hw.get_ort_providers = lambda: ["CPUExecutionProvider"]  # noqa
    from src.config import settings
    from src.retrieval.multi_retriever import MultiBatchRetriever

    r = MultiBatchRetriever()
    batch_names = list(r.lance_tables.keys()) if r.backend == "lancedb" else list(r.batches.keys())
    out = []
    for i, (q_zh, q_en) in enumerate(PAIRS):
        for q, lang in ((q_zh, "zh"), (q_en, "en")):
            t0 = time.time()
            vecs = r.embedder.embed_docs([q])
            hits: dict = {}
            for name in batch_names:
                for idx, s in r._vector_search(name, vecs[0], settings.TOP_K_VECTOR):
                    if idx not in hits:
                        hits[idx] = s
            for idx, s in r.bm25.top_k(q, k=settings.TOP_K_BM25):
                if idx not in hits:
                    hits[idx] = s
            cands = [r._chunk_full(idx) for idx in list(hits.keys())[: args.topk]]
            reranked = r.reranker.rerank(q, cands, top_k=args.topk)
            scores = [c.get("rerank_score", 0.0) for c in reranked]
            out.append({"pair": i, "lang": lang, "query": q,
                        "n": len(scores),
                        "scores": [round(s, 4) for s in scores],
                        "top1": scores[0] if scores else None,
                        "ms": round((time.time() - t0) * 1000)})
            print(f"  pair{i} {lang} n={len(scores)} top1={scores[0] if scores else '-'} "
                  f"p50={sorted(scores)[len(scores)//2] if scores else '-'}", flush=True)
    (PROJ / "logs" / "scores_dist.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")
    print("SCORES_DIST_DONE -> logs/scores_dist.json")


if __name__ == "__main__":
    main()