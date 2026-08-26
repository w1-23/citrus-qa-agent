# 投稿附属物草稿：Highlights / CRediT / Data Availability / 声明

> 目标：CEA（Computers and Electronics in Agriculture, Elsevier）。
> 状态：草稿 v1（2026-08-25）。Highlights 5 条英文（Elsevier 要求 3-5 条，
> 每条 ≤85 字符含空格）；CRediT 按实际贡献起草；Data Availability 声明
> fig_data 只读仓库可复现。待用户裁决 query_mode 后微调摘要/结论措辞。

---

## Highlights（英文，5 条，≤85 字符/条）

1. A real-world citrus QA system with determinism-first agent architecture.
2. Multi-granularity query generation ablated across eight modes (MRR 0.65 vs 0.26).
3. Rerank-query choice, not recall, dominates multi-query retrieval degradation.
4. Chinese queries show significant retrieval bias (paired p<0.001); window mitigates.
5. Code-level early stopping cuts cost 21% with zero gold-coverage loss.

（字符数核查：[1]=81 [2]=83 [3]=76 [4]=88 → 第4条超限，改短：
4. Chinese-query retrieval bias is significant (paired p<0.001) and window-restorable.
   = 85 字符，恰在限内；若余量不足可再压为
   "Chinese queries show significant retrieval bias (p<0.001); window mitigates."
   = 82 字符 ← 推荐）

**推荐最终 5 条（已核验 2026-08-26：全部 ≤85 字符含空格）**：
- A real-world citrus QA system with determinism-first agent architecture. (72)
- Eight-mode ablation of multi-granularity query generation (MRR 0.65 vs 0.26). (77)
- Rerank-query choice, not recall, dominates multi-query retrieval degradation. (77)
- Chinese queries show significant retrieval bias (p<0.001); window mitigates. (76)
- Code-level early stopping cuts cost 21% with zero gold-coverage loss. (69)

---

## CRediT（作者贡献声明，按角色填空名）

- **Conceptualization**:
- **Methodology**:
- **Software**:
- **Validation**:
- **Formal analysis**:
- **Investigation**:
- **Resources**:
- **Data Curation**:
- **Writing – Original Draft**:
- **Writing – Review & Editing**:
- **Visualization**:
- **Supervision**:
- **Project administration**:
- **Funding acquisition**:

（作者名由投稿人回填；若单作者可合并为 2-3 项）

---

## Data Availability（数据可用性声明）

Data and code used in this study are reproducible from the following locations:

- Evaluation dataset (120 bilingual questions with gold-locator triplets):
  `experiment/data/eval120.jsonl` and `experiment/data/s5_bilingual_pairs.jsonl`
  (60 language-equivalent pairs).
- Raw experimental results (read-only): `experiment/results/fig_data/*.csv`
  (12 CSVs covering Section 7.4–7.12 main figures) plus per-experiment JSON
  snapshots (`exp1_ablation.json`, `exp1b_rerank_attribution.json`,
  `exp1c_pool_orig.json`, `exp2_query_mode.json`, `exp4a_scan.json`,
  `exp4b_e2e.json`, `exp7_threshold_grid.json`, `end2end_judge.json`,
  `b3_temporal_judge.json`, `production_raw_trio.json`,
  `e2e_three_groups/summary.json`).
- Figure scripts: `experiment/scripts/plotting/*.py` (public style utilities
  in `plot_utils.py`); all plots are exported as vector PDF + 300-dpi PNG and
  read only from `fig_data/` CSV inputs (no hard-coded numbers).
- Reproduction scripts for every experiment: `experiment/scripts/*.py`
  (a1–a6, b1–b3, exp1/1b/1c/2/4a/4b/7, S3–S5 builders, production_raw_trio,
  e2e_three_groups).

Additional data (the 252,681-chunk citrus corpus and the deployed `agent/`
codebase) are available from the corresponding author upon reasonable request,
subject to institutional data-transfer agreements.

---

## Declaration of Competing Interest（利益冲突声明）

The authors declare that they have no known competing financial interests or
personal relationships that could have appeared to influence the work reported
in this paper.

---

## Acknowledgements（致谢，联系人回填）

---

## 附：受体裁微调清单（2026-08-26 已执行）

1. ✅ 裁决①（query_mode→raw）已回写：config.yaml:32 raw + multi_retriever.py:733
   防御 + config.py:92 兜底同步 + 231 tests 全绿 + git 837ddb2。
2. ✅ §8.3 补"检索配置修正"陈述（MRR 0.256→0.557 生产实测，+117%；0.653
   为对齐英文上界）；摘要/结论同口径；§7.12（实验九）端到端显著验证已落笔。
3. ✅ APPENDIX A query_mode 行终值 = raw（config.yaml:32 / config.py:92）。