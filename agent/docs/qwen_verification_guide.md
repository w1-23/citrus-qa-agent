# Qwen-Max 独立核验说明书（v1 · 2026-08-26）

> 目的：用 Qwen-Max 作为**独立裁判**，对 e2e 实验 A 组（raw 修正组）6 题答案
> 重新打四维分，与 DeepSeek 原判分算 Cohen's Kappa（一致性 ≥ 0.8）。
> **核验输入里不含原分**（防止锚定），Qwen 完全凭「问题+证据+答案」独立判断。

---

## 一、CSV 里那 8 列是什么（e2e_kappa_human_template.csv）

| 列 | 含义 | 谁来填 |
|---|---|---|
| `id` | 题号（cor-007/079/053/040/035/036） | 已填 |
| `question_zh` | 中文问题 | 已填 |
| `llm_correctness` | 正确性分（1-5）——**DeepSeek 原判** | 已填（参考，不代表真理） |
| `llm_completeness` | 完整性分（1-5） | 已填 |
| `llm_citation_fidelity` | 引用忠实度分（1-5） | 已填 |
| `llm_evidence_sufficiency` | 证据充分性分（1-5） | 已填 |
| `human_correctness` | **你要评的分** | 空 |
| `human_completeness` | **你要评的分** | 空 |
| `human_citation_fidelity` | **你要评的分** | 空 |
| `human_evidence_sufficiency` | **你要评的分** | 空 |
| `answer_excerpt` | 模型答案摘录 | 已填 |

> 你（或 Qwen）只填 4 个 `human_*` 列；其余不动。用 Qwen 时，它输的
> 四维分就是这 4 列的值。

---

## 二、四个维度的判据（评分必须照此口径，否则 Kappa 反映的是"口径差"而非"质量差"）

| 维度 | 要评什么 | 1 分（差） | 5 分（好） |
|---|---|---|---|
| **correctness 正确性** | 答案与证据是否一致、有无编造 | 与证据矛盾/明显幻觉/编造数字 | 完全基于证据、无幻觉、数字事实全对 |
| **completeness 完整性** | 是否覆盖问题所有要点 | 只答了 1 个点，漏掉核心 | 问题要求的所有点全部覆盖 |
| **citation_fidelity 引用忠实度** | 每个 [n] 引用是否真对应某条证据、用得恰当 | 引用编号乱标/引用的证据不支持该句 | 每个 [n] 都对得上且支持该句 |
| **evidence_sufficiency 证据充分性** | 给的证据是否足以支撑答案 | 答案断言远超证据能证明的 | 证据恰好支撑答案的全部断言 |

**中间档**：3 分 = 部分正确/部分覆盖/部分引用对/证据部分支撑（如 cor-053
只有 1 个证据、且证据在关键处被截断——DeepSeek 原判 3 分合理）。

**整数评分**：只能 1/2/3/4/5，不允许多数点。

---

## 三、评分规则（给 Qwen 的提示词，直接复制粘贴）

把下面整段贴在每条【问题+证据+答案】前面：

```
你是端到端质量裁判。对【问题】+【证据原文】+【模型答案】打四个维度的
分数（1-5 整数，只允许 1/2/3/4/5）：
1) correctness（正确性）：答案与证据一致、无幻觉（5=完全一致，无编造）；
2) completeness（完整性）：是否覆盖问题全部要点（5=完整覆盖）；
3) citation_fidelity（引用忠实度）：每个 [n] 引用是否真实对应某条证据
   且用得恰当（5=全部忠实，无错标）；注意证据是【证据 1】【证据 2】…
   编号，答案里的 [1] 应对应【证据 1】；
4) evidence_sufficiency（证据充分性）：证据是否足以支撑答案全部断言
   （5=充分支撑；若证据被截断或缺失关键信息导致无法支撑，应给低分）。
先理性判断，再输出严格 JSON（不要多余文字）：
{"correctness":n,"completeness":n,"citation_fidelity":n,
 "evidence_sufficiency":n,"rationale":"一句中文理由"}
```

---

## 四、材料位置（每份都齐"问题+证据+答案"）

| 题号 | 独立输入文件（推荐用这个） |
|---|---|
| cor-007 | `experiment\results\qwen_judge_input\cor-007.md` |
| cor-079 | `experiment\results\qwen_judge_input\cor-079.md` |
| cor-053 | `experiment\results\qwen_judge_input\cor-053.md` |
| cor-040 | `experiment\results\qwen_judge_input\cor-040.md` |
| cor-035 | `experiment\results\qwen_judge_input\cor-035.md` |
| cor-036 | `experiment\results\qwen_judge_input\cor-036.md` |

或合并版：`experiment\results\qwen_judge_input\qwen_judge_input_6q.json`

**做法**：①打开上面的 .md → ②把第三节的评分规则段粘贴到前面 →
③整段发给 Qwen-Max（temperature 设 0 或最低）→ ④把返回的唯一 JSON 里的
四个分数回填到 `e2e_kappa_human_template.csv` 的 `human_*` 4 列。

> 为什么用 .md 而不是 CSV：CSV 表把证据挤在一格里，Qwen 不易对应
> [1][2] 引用编号；.md 里证据明确编号为【证据 1】【证据 2】，与答案的
> [1][2] 引用一一对应——这正是判 citation_fidelity 的关键。

---

## 五、回填后交给我做的事

1. 读取你的 `human_*` 4 列（Qwen 的 6 题 × 4 维 = 24 个分）；
2. 与 `llm_*`（DeepSeek 原判）逐题逐维配对，算 **Cohen's Kappa（加权线性）**；
3. 报告：Kappa 值、逐题一致性矩阵、分歧点（哪几题哪个维度差>1）；
4. 若 Kappa ≥ 0.8 → 论文增补"cross-checked with an independent LLM judge
   (Qwen-Max), κ=0.82"；若 < 0.8 → 分歧题人工仲裁后重算。

---

## 六、可选扩展：6 条数值题也可以同步核

`agent\docs\numeric_questions_human_check.md` 里每题有 gold_note + 语料原文片段，
让 Qwen 凭**原文证据**核 gold 是否属实（不是凭记忆背答案），每题输出"一致/不一致+证据句"。

材料足够，直接可用。完毕。