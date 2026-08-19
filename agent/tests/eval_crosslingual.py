# -*- coding: utf-8 -*-
"""eval_crosslingual — 跨语言召回矩阵（F3 的量化版）。

对每道含 question_en 的题：
  zh 查询 vs 英回填空格gold → Recall@10 / passed
  en 查询 vs 同 gold          → Recall@10 / passed
输出 markdown 表行 + logs/crosslingual.json。

用法: python tests/eval_crosslingual.py [--cpu]
"""
import argparse
import json
import sys
import time
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent if Path(__file__).resolve().name != "<string>" else None
PROJ = (Path.cwd() if (Path.cwd() / "src").is_dir()
        else (_FILE_DIR.parent if _FILE_DIR else Path.cwd()))
sys.path.insert(0, str(PROJ))
GOLD = PROJ / "tests" / "golden_qa.jsonl"


def _match(c: dict, gold_ids: list) -> bool:
    pid = str(c.get("paper_id") or "").strip()
    if not pid:
        return False
    for g in gold_ids:
        if not g:
            continue
        if ":" in g:
            if f"{pid}:{c.get('chunk_index')}" == g:
                return True
        elif pid == g or pid.startswith(g.rstrip("*")):
            return True
    return False


def _recall(hits, gold):
    if not gold:
        return None, None
    matched = set()
    first = None
    for i, c in enumerate(hits[:10]):
        for g in gold:
            if g in matched:
                continue
            if _match(c, [g]):
                matched.add(g)
                if first is None:
                    first = i + 1
                break
    return len(matched) / len(gold), (1.0 / first if first else 0.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    if args.cpu:
        import src.engine.hardware as hw
        hw.get_ort_providers = lambda: ["CPUExecutionProvider"]  # noqa
    from src.retrieval.multi_retriever import MultiBatchRetriever

    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = MultiBatchRetriever()
    print("| 题 | 语言 | passed | Recall@10 | MRR@10 |")
    print("|---|---|---|---|---|")
    out = []
    for g in rows:
        if not g.get("question_en"):
            continue
        gold = g.get("required_evidence") or []
        for q, lang in ((g["question"], "zh"), (g["question_en"], "en")):
            t0 = time.time()
            hits = r.search(q)
            rec, mrr = _recall(hits, gold)
            rec_s = "-" if rec is None else f"{rec:.3f}"
            mrr_s = "-" if mrr is None else f"{mrr:.3f}"
            print(f"| {g['id']} | {lang} | {len(hits)} | {rec_s} | {mrr_s} |")
            out.append({"id": g["id"], "lang": lang, "passed": len(hits),
                        "recall@10": rec, "mrr@10": mrr, "ms": round((time.time() - t0) * 1000)})
    (PROJ / "logs" / "crosslingual.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print("CROSSLINGUAL_DONE -> logs/crosslingual.json")


if __name__ == "__main__":
    main()