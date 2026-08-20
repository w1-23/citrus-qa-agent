# -*- coding: utf-8 -*-
"""probe_outofdomain — 域外查询的空归因测试（C3/06 边界证据）。

6 条与柑橘无关的通用问题 → 完整产品路径检索（RRF+重排+动态阈值）。
预期：passed=0 且 last_empty_reason ∈ {threshold_blocked, no_match}，
即系统对域外输入"诚实空答"而非抛出无关片段（空归因设计正确性）。

输出: logs/outofdomain.json + 控制台表格
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

QUERIES = [
    "量子计算的基本原理是什么？",
    "肺炎支原体的治疗方案",
    "巴黎三日游攻略",
    "深度学习中的注意力机制怎么实现？",
    "高血压患者的饮食建议",
    "法国大革命的时间线",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cpu:
        import src.engine.hardware as hw
        hw.get_ort_providers = lambda: ["CPUExecutionProvider"]  # noqa
    from src.retrieval.multi_retriever import MultiBatchRetriever

    r = MultiBatchRetriever()
    print("| 查询 | passed | top1分数 | 空归因 |")
    print("|---|---|---|---|")
    out = []
    for q in QUERIES:
        t0 = time.time()
        hits = r.search(q)
        reason = getattr(r, "last_empty_reason", None)
        top = hits[0].get("rerank_score") if hits else None
        print(f"| {q} | {len(hits)} | {top if top is not None else '-'} | {reason} |")
        out.append({"query": q, "passed": len(hits), "top1": top,
                    "reason": reason, "ms": round((time.time() - t0) * 1000)})
    blocked = sum(1 for d in out if d["passed"] == 0)
    print(f"OUTOFDOMAIN_SUMMARY empty={blocked}/{len(out)}")
    (PROJ / "logs" / "outofdomain.json").write_text(json.dumps(out, ensure_ascii=False, indent=2),
                                                    encoding="utf-8")


if __name__ == "__main__":
    main()