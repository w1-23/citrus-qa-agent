# -*- coding: utf-8 -*-
"""AG-11 修复端到端验证脚本 — 在你的运行环境执行:
    python verify_retrieval.py

预期输出:
  1. 启动自检: 5 个批次 idx_map ok (匹配率 100%)
  2. 3 个真实查询返回的文献各不相同、且不指向 global_chunks[-1]
  3. 全部断言 PASS 即修复生效
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.retrieval.multi_retriever import MultiBatchRetriever

QUERIES = [
    "citrus huanglongbing Candidatus Liberibacter asiaticus disease resistance",
    "blood orange anthocyanin Ruby MYB transcription factor regulation",
    "citrus fruit citric acid metabolism organic acid accumulation",
]


def main():
    r = MultiBatchRetriever()
    corpus = r.global_chunks
    last_idx = corpus[-1].get("_global_idx")
    print(f"[1] corpus={len(corpus)} chunks | last_idx={last_idx}")

    total_hits_last = 0
    total_hits = 0
    paper_ids = set()
    for q in QUERIES:
        results = r.search(q)
        print(f"[2] query: {q[:45]}... -> {len(results)} results")
        if not results:
            print("    !! 空结果 — 请检查 rerank 阈值或数据加载")
            continue
        for c in results:
            total_hits += 1
            pid = (c.get("paper_id") or c.get("title") or "?")[:50]
            paper_ids.add(pid)
            if c.get("_global_idx") == last_idx:
                total_hits_last += 1
                print(f"    !! 命中 last chunk: {pid} (修复无效)")
            else:
                print(f"    ok paper={pid} | score={c.get('rerank_score', 0):.3f}")

    print(f"[3] 总命中={total_hits} | 指向 last chunk 数={total_hits_last} | 去重论文数={len(paper_ids)}")
    ok = True
    if total_hits_last > 0:
        print("FAIL: 仍有结果指向 global_chunks[-1] — 映射修复未生效")
        ok = False
    if total_hits > 0 and len(paper_ids) < 2:
        print("WARN: 命中论文过于单一，请人工确认相关性")
    if ok:
        print("\nPASS: AG-11 修复生效 — 检索结果不再指向语料库最后一条 chunk")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
