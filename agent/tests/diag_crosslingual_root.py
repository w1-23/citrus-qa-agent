# -*- coding: utf-8 -*-
"""根因分解诊断 v2：论文级匹配（与 eval_crosslingual._match 同口径）。

沿产品路径逐环节记录 gold 论文去向 + 每环节命中论文 id/标题：
  dense top40（embed_docs 产品口径；zh 附加 embed_query 前缀对照）
  → BM25 top40 → RRF 池 20 → rerank top10 → 动态阈值 → passed
输出 JSON + 控制台摘要。

用法: python tests/diag_crosslingual_root.py [--cpu] [--questions l1-004,l1-005,l1-008]
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

logging.basicConfig(level=logging.WARNING, stream=sys.stdout)
CWD = Path.cwd()
sys.path.insert(0, str(CWD))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    ap.add_argument("--questions", default="l1-004,l1-005,l1-008")
    ap.add_argument("--out", default="logs/diag_crosslingual_root.json")
    args = ap.parse_args()
    if args.cpu:
        import src.engine.hardware as hw
        hw.get_ort_providers = lambda: ["CPUExecutionProvider"]  # noqa

    from src.config import settings
    from src.retrieval.multi_retriever import MultiBatchRetriever
    from src.retrieval.bm25 import rrf_fuse

    rows = [json.loads(l) for l in
            (CWD / "tests" / "golden_qa.jsonl").read_text(encoding="utf-8-sig").splitlines()
            if l.strip()]
    qs = {r["id"]: r for r in rows if r.get("question_en")}
    want = set(args.questions.split(","))
    r = MultiBatchRetriever()
    batch_names = list(r.lance_tables.keys()) if r.backend == "lancedb" \
        else list(r.batches.keys())
    by_gidx = {c["_global_idx"]: c for c in r.global_chunks}

    def paper_match(pid: str, gid: str) -> bool:
        pid = (pid or "").strip()
        if not pid:
            return False
        if ":" in gid:
            return False
        return pid == gid or pid.startswith(gid.rstrip("*"))

    def chunk_match(c: dict, gid: str) -> bool:
        pid = str(c.get("paper_id") or "").strip()
        if not pid:
            return False
        if ":" in gid:
            return f"{pid}:{c.get('chunk_index')}" == gid
        return paper_match(pid, gid)

    def gidx_hits(gold, ids):
        """gold 中哪些被 ids 里的 chunk 命中（原评测口径）。返回命中 gold 下标集合。"""
        hit = set()
        for gi, gid in enumerate(gold):
            for idx in ids:
                if chunk_match(by_gidx[idx], gid):
                    hit.add(gi)
                    break
        return hit

    def dedupe(hits):
        hits = sorted(hits, key=lambda x: x[1], reverse=True)
        seen, out = set(), []
        for idx, s in hits:
            if idx in seen:
                continue
            seen.add(idx)
            out.append((idx, s))
        return out

    def run_dense(vec, topk):
        hits = []
        for name in batch_names:
            hits.extend(r._vector_search(name, vec, topk))
        hits.sort(key=lambda x: x[1], reverse=True)
        return hits

    def brief(c):
        t = c.get("title") or ""
        if len(t) > 58:
            t = t[:55] + "..."
        return f"{c.get('paper_id')}|{c.get('_global_idx')}|{t}"

    out = []
    for qid in want:
        g = qs[qid]
        gold = g.get("required_evidence") or []
        for lang, q in (("zh", g["question"]), ("en", g["question_en"])):
            t0 = time.time()
            rec = {"id": qid, "lang": lang, "query": q,
                   "gold": [(gi, gid) for gi, gid in enumerate(gold)]}
            v_docs = r.embedder.embed_docs([q])[0]
            dense_docs = dedupe(run_dense(v_docs, settings.TOP_K_VECTOR))[:settings.TOP_K_VECTOR]
            dense_pref = None
            if lang == "zh":
                v_pref = r.embedder.embed_query(q)
                dense_pref = dedupe(run_dense(v_pref, settings.TOP_K_VECTOR))[:settings.TOP_K_VECTOR]
            bm25 = dedupe(r.bm25.top_k(q, k=settings.TOP_K_BM25))[:settings.TOP_K_BM25]
            fused = rrf_fuse(dedupe(dense_docs), dedupe(bm25),
                             k=settings.RRF_K, weights=[1.0, 1.0])
            pool20 = fused[: settings.TOP_K_FINAL * 2]
            pool_ids = [i for i, _ in pool20]
            cands = [r._chunk_full(i) for i in pool_ids]
            reranked = r.reranker.rerank(q, cands, top_k=settings.TOP_K_FINAL)
            top_score = reranked[0].get("rerank_score", 0) if reranked else 0
            thresh = max(settings.RERANK_THRESHOLD,
                         top_score * settings.DYNAMIC_THRESHOLD_RATIO)
            passed_ids = [c["_global_idx"] for c in reranked
                          if c.get("rerank_score", 0) >= thresh]
            d40_ids = [i for i, _ in dense_docs]
            b40_ids = [i for i, _ in bm25]
            t10_ids = [c["_global_idx"] for c in reranked]
            rec["dense40_gold"] = len(gidx_hits(gold, d40_ids))
            rec["dense_pref40_gold"] = (len(gidx_hits(gold, [i for i, _ in dense_pref]))
                                        if dense_pref else None)
            rec["bm25_gold"] = len(gidx_hits(gold, b40_ids))
            rec["pool20_gold"] = len(gidx_hits(gold, pool_ids))
            rec["top10_gold"] = len(gidx_hits(gold, t10_ids))
            rec["passed_gold"] = len(gidx_hits(gold, passed_ids))
            rec["dense40"] = [brief(by_gidx[i]) for i in d40_ids][:6]
            rec["bp_pref_delta"] = None
            if dense_pref is not None:
                pref_ids = [i for i, _ in dense_pref]
                rec["dense_pref40"] = [brief(by_gidx[i]) for i in pref_ids][:6]
                rec["bp_dense40"] = [brief(by_gidx[i]) for i in d40_ids[:6]]
            rec["bm25_40"] = [brief(by_gidx[i]) for i in b40_ids][:6]
            rec["top10_reranked"] = [
                {"idx": c["_global_idx"], "paper": c.get("paper_id"),
                 "title": (c.get("title") or "")[:80], "score": c.get("rerank_score")}
                for c in reranked]
            rec["top_rerank_score"] = top_score
            rec["threshold_used"] = round(thresh, 4)
            rec["n_pool"] = len(pool_ids)
            rec["n_passed"] = len(passed_ids)
            rec["ms"] = round((time.time() - t0) * 1000)
            out.append(rec)
            print(f"[{qid}/{lang}] gold(dense/bm25/pool/top10/pass)="
                  f"{rec['dense40_gold']}/{rec['bm25_gold']}/{rec['pool20_gold']}/"
                  f"{rec['top10_gold']}/{rec['passed_gold']} "
                  f"top={top_score:.3f} thresh={rec['threshold_used']} "
                  f"n_passed={rec['n_passed']}", flush=True)
            for c in reranked:
                print(f"    #{c.get('rerank_score')} {brief(c)}", flush=True)
    (CWD / args.out).write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                encoding="utf-8")
    print("DIAG_DONE ->", args.out)


if __name__ == "__main__":
    main()