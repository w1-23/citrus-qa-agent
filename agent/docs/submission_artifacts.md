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
  (10 CSVs covering Section 7.4–7.10 main figures) plus per-experiment JSON
  snapshots (`exp1_ablation.json`, `exp1b_rerank_attribution.json`,
  `exp2_query_mode.json`, `exp4a_scan.json`, `exp4b_e2e.json`,
  `exp7_threshold_grid.json`, `end2end_judge.json`,
  `b3_temporal_judge.json`).
- Figure scripts: `experiment/scripts/plotting/*.py` (public style utilities
  in `plot_utils.py`); all plots are exported as vector PDF + 300-dpi PNG and
  read only from `fig_data/` CSV inputs (no hard-coded numbers).
- Reproduction scripts for every experiment: `experiment/scripts/*.py`
  (a1–a6, b1–b3, exp1/1b/2/4a/4b/7, S3–S5 builders).

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

## 附：§8.3 / 摘要 / 结论 待裁决微调清单（裁决后执行）

1. §8.3 若裁决①（query_mode→raw）：补充"生产默认修正"小节（MRR 0.256→0.653
   实测 + config.yaml:29 回写记录）。
2. 若裁决②（rerank 用 original_query）：改为"rerank query 选择修正"
   （0.256→0.584，保留多路召回面）；须在摘要/结论同步改"0.653 vs 0.256"
   的对照数字。
3. 若裁决③（维持 full）：摘要/结论改回中性表述（不写"显著优于"），
   论文如实声明"生产默认配置的非最优性已披露"。