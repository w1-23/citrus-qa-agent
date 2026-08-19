# -*- coding: utf-8 -*-
"""enrich_golden — 文献题的 EN-surrogate 金标回填（半自动，凌晨批步骤 0）。

原理：中文文献型查询在 cross-encoder 阈值下被拦（F3），但其英文同义查询能
完整召回；语言无关地讲，同一批英文文献就是该问题的相关证据。因此取
question_en 检索 top5 的 paper_id 作为 required_evidence 回填，并标记
"EN-surrogate, 待人工复核"。回填后中文查询的 Recall@10 将如实反映被阈值
拦截的损失（这正是 F3 的量化证据）。

用法: python tests/enrich_golden.py [--topk 5]
"""
import argparse
import json
import sys
from pathlib import Path

_FILE_DIR = Path(__file__).resolve().parent if Path(__file__).resolve().name != "<string>" else None
PROJ = (Path.cwd() if (Path.cwd() / "src").is_dir()
        else (_FILE_DIR.parent if _FILE_DIR else Path.cwd()))
sys.path.insert(0, str(PROJ))
GOLD = PROJ / "tests" / "golden_qa.jsonl"


def main():
    import src.engine.hardware as hw
    hw.get_ort_providers = lambda: ["CPUExecutionProvider"]  # noqa

    from src.retrieval.multi_retriever import MultiBatchRetriever

    ap = argparse.ArgumentParser()
    ap.add_argument("--topk", type=int, default=5)
    args = ap.parse_args()

    rows = [json.loads(l) for l in GOLD.read_text(encoding="utf-8").splitlines() if l.strip()]
    r = MultiBatchRetriever()
    changed = 0
    for g in rows:
        qen = g.get("question_en")
        if not qen or g.get("required_evidence"):
            continue
        hits = r.search(qen)[: args.topk]
        ids = []
        for c in hits:
            pid = str(c.get("paper_id") or "").strip()
            if pid and pid not in ids:
                ids.append(pid)
        g["required_evidence"] = ids
        g["notes"] = (g.get("notes") or "") + f" | EN-surrogate top{args.topk}，待人工复核"
        changed += 1
        print(f"  {g['id']}: gold={len(ids)} <- {[i[:40] for i in ids]}")
    if changed:
        GOLD.write_text("\n".join(json.dumps(g, ensure_ascii=False) for g in rows) + "\n",
                        encoding="utf-8")
    print(f"ENRICH_END changed={changed}")


if __name__ == "__main__":
    main()