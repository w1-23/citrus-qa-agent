# -*- coding: utf-8 -*-
"""overlap_analysis — 稠密(向量)与 BM25 的 top-40 命中集合交叠率。

回答 04 文档 §2 补实验第 2 项（稀疏-稠密互补度分析）：
对黄金集 13 题，取 vector top40 ∪ bm25 top40 的 global_idx 集合，
输出两路交集大小 / Jaccard，量化"互补性"而非只靠单路消融间接推断。

输出: logs/overlap.json + 控制台表格
用法: python tests/overlap_analysis.py [--cpu]
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
GOLD = PROJ / "tests" / "golden_qa.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cpu:
        import src.engine.hardware as hw
        hw.get_ort_providers = lambda: ["CPUExecutionProvider"]  # noqa
    from src.config import settings
    from src.retrieval.multi_retriever import MultiBatchRetriever

    r = MultiBatchRetriever()
    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    batch_names = list(r.lance_tables.keys()) if r.backend == "lancedb" else list(r.batches.keys())

    print("| 题 | |V∩B| | |V∪B| | Jaccard | 稠密独有 | BM25独有 |")
    print("|---|---|---|---|---|---|---|")
    out = []
    for g in rows:
        q = g["question"]
        t0 = time.time()
        vecs = r.embedder.embed_docs([q])
        vset: set = set()
        for name in batch_names:
            for idx, _s in r._vector_search(name, vecs[0], settings.TOP_K_VECTOR):
                vset.add(idx)
        bset: set = set()
        for idx, _s in r.bm25.top_k(q, k=settings.TOP_K_BM25):
            bset.add(idx)
        inter = vset & bset
        union = vset | bset
        jac = len(inter) / len(union) if union else 0.0
        v_only = len(vset - bset)
        b_only = len(bset - vset)
        print(f"| {g['id']} | {len(inter)} | {len(union)} | {jac:.3f} | {v_only} | {b_only} |")
        out.append({"id": g["id"], "inter": len(inter), "union": len(union),
                    "jaccard": jac, "dense_only": v_only, "bm25_only": b_only,
                    "ms": round((time.time() - t0) * 1000)})
    med = sorted(d["jaccard"] for d in out)[len(out) // 2] if out else 0
    mean = sum(d["jaccard"] for d in out) / len(out) if out else 0
    print(f"OVERLAP_SUMMARY mean_jac={mean:.3f} median_jac={med:.3f} n={len(out)}")
    (PROJ / "logs" / "overlap.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                encoding="utf-8")


if __name__ == "__main__":
    main()