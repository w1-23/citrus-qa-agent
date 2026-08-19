# -*- coding: utf-8 -*-
"""probe_bilingual — 中英双语 rerank 分数分布对照（F3 主线的独立实验维度）。

背景（paper/13 F3）：中文文献型查询（如 'Poncirus trifoliata 作为柑橘砧木的作用'）
在全配置下被动态阈值全拦，而英文近义查询（'Poncirus trifoliata rootstock'）能过——
推测 cross-encoder（bge-reranker-v2-m3）对英文查询的分数分布更高。
本探针：同义中英问句 × 各批次检索 → 打印 passed 数、阈值、前3分数。

用法（agent/ 目录）: python tests/probe_bilingual.py
输出: markdown 表格行（可直接粘进 paper/13 §F3）
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import src.engine.hardware as hw  # noqa: E402
hw.get_ort_providers = lambda: ["CPUExecutionProvider"]  # noqa

PAIRS = [
    ("Nagami kumquat 的品种特征是什么？", "What are the cultivar characteristics of Nagami kumquat?"),
    ("Poncirus trifoliata 作为柑橘砧木的作用", "The role of Poncirus trifoliata as a citrus rootstock"),
    ("血橙中花青素积累的分子机制有哪些报道？", "What molecular mechanisms are reported for anthocyanin accumulation in blood oranges?"),
    ("kumquat（金柑）属的物种学名及其与柑橘属的关系", "The scientific name of the kumquat genus and its relationship to Citrus"),
    ("Valencia orange 果汁加工用途的品种特性", "Valencia orange characteristics for juice processing"),
]

def main():
    from src.retrieval.multi_retriever import MultiBatchRetriever
    from src.config import settings
    r = MultiBatchRetriever()
    print("| 查询 | 语言 | passed | 阈值 | top1 | top2 | top3 |")
    print("|---|---|---|---|---|---|---|")
    for zh, en in PAIRS:
        for q, lang in ((zh, "中"), (en, "英")):
            hits = r.search(q)
            n = len(hits)
            th = round(max(settings.RERANK_THRESHOLD,
                           (hits[0].get("rerank_score") or 0) * settings.DYNAMIC_THRESHOLD_RATIO), 4) if hits else "-"
            tops = " | ".join(f"{h.get('rerank_score', 0):.3f}" for h in hits[:3]) or "-"
            print(f"| {zh[:22]}.. | {lang} | {n} | {th} | {tops} |")
    print("\n注: passed<10 即被阈值部分拦截；passed=0 为全拦。")
    print("阈值 = max(0.25, top1×", settings.DYNAMIC_THRESHOLD_RATIO, ")")

if __name__ == "__main__":
    main()