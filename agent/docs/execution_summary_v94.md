# CitrusQA 论文实验执行总结（v9.4 收口 · 2026-08-26）

> 本文件是全部长期实验与论文修改的**单一交付入口**：实验证明了什么、
> 结论是什么、贡献是什么、产出在哪、生产改了什么、还差什么。
> 底层细节见各分项文档（文末"分项导航"链接）。
> 目标期刊：Computers and Electronics in Agriculture（Elsevier SCI Q1）。

---

## 0. 一句话总览

长期实验验证了一个**反直觉事实**：系统生产默认检索配置（多粒度多路 full）**不是最优的**
——实测远劣于原始查询单路（raw）。工作完成了 **发现 → 归因 → 端到端验证 → 生产回写**
的完整闭环，论文全部数字以实测为准改写，生产已回写并全量回归通过。

---

## 1. 实验证明了什么

### 1.1 检索层核心发现（实验 1/1b/1c/2，四重互证）

| 发现 | 实测 |
|---|---|
| **原始查询单路（raw）碾压生产默认多路（full）** | raw **MRR@10 0.653** / R@10 0.988 vs full **0.256** / 0.667 |
| 八模式全扫描，raw 全场最高 | raw 0.653 > mq_only 0.312 > hyde_only 0.263 > full 0.256 > hyde_mq 0.250 ≈ hyde_sum 0.251 ≈ mq_sum 0.277 > sum_only 0.143 |
| 去掉 BM25 损伤最大 | -BM25 = **0.143**（BM25 是词法召回底柱） |
| 去掉 Rerank 反而超 full | -Rerank = 0.328 > full 0.256（精排环节在帮倒忙） |

### 1.2 归因闭环（实验 1b/1c）

- **主因（~73%）= 精排查询用错对象**：生产 `multi_retriever.py` 用 `queries[0]`
  （HyDE 生成段落）做 rerank 而非用户查询 → 同池只换 rerank query，MRR 0.256 → 0.584。
- **次因（~27%）= 多路池稀释**：9 路查询混合使 gold 进入 top-20 池概率 98.8% → 81.0%；
  `_compose_queries` 正常路径**不含原始查询**。
- **裁决矩阵闭合**：修 rerank（②=0.584）≈ 双修（②b=0.581）< 不动召回面的 raw（①=0.653）
  → 多路召回面不可救，**唯一最优且最简单 = raw**。

### 1.3 生产形态三对照（#1，澄清"0.653 的真实含义"）

| 生产用户实际输入 | MRR@10 | R@10 | pool20 |
|---|---|---|---|
| 中文原句直搜（只改 raw 不翻译） | 0.282 | 0.441 | 0.441 |
| **中文→LLM 改写英文 + raw** | **0.557** | 0.869 | 0.905 |
| 评测集对齐英文原句（理想上界） | 0.653 | 0.988 | 0.988 |

**结论**：0.653 是"高质量对齐英文"的上界；生产真实到达值约 **0.557**
（差距 = 改写质量，-14.7%，标注为后续优化空间）。生产落地 raw 必须配英文改写
前置——生产中由 retrieve-agent 层"检索英文关键词"策略卡隐式完成，无需新代码。

### 1.4 端到端验证（#2，检索差距是否传导到生成质量）

- 三组 × 30 题（SEED=42，L1/L2/L3 各 10），同一生成器、同一 Judge 四维（1-5）
- 四维均值：**A(raw) 4.867/4.800/4.900/4.633**；B(full) 4.633/4.367/4.833/4.200；
  C(full+原精排) 4.667/4.567/4.667/4.467
- 配对 Wilcoxon：**A vs B 三维显著更优**（正确性 p=0.026、完整性 p=0.015、
  证据充分性 p=0.042）
- **诚实发现**：C vs B 端到端全维不显著（检索层 0.584 vs 0.256 不传导）——
  B 组 gold 2/3 仍在 passed 窗口，生成对 top1 排名不敏感；**决定性因素是
  查询是否英文，而非多路 vs 单路**。

### 1.5 支撑性实验（论文 §7 其余）

- **中文跨语言偏置（S3/§7.6）**：实体 en 39/40 vs zh 16/40（Δ0.575）、功能
  30/30 vs 13/30（Δ0.567）、L1 zh 0.333 —— 跨类型普遍
- **跨域探针（A5）**：BRCA1/OsNRT2 各 20 篇×3 探针 → 无柑橘式功能型塌陷；
  "功能型普适性"证伪，改述为多因素（语言对齐/领域词分布/查询长度）
- **改写缓解（B3）**：rewrite_on vs off：事实 +0.094、引用 +0.096、覆盖 +0.147；
  12 题时效 honesty=1.0
- **成本（A3/§7.8）**：代码级早停 16 组零质量损失，默认 (6,0.25) -21% 轮次；
  run-all 10000 tok vs oracle 4283 tok（-57%）
- **延迟（A4/§7.9）**：LanceDB 14.4ms vs Qdrant 91.5ms（6.4×）

---

## 2. 结论

1. **反直觉主结论**：多粒度查询生成（"长度适配假说"）**实测证伪**——多路组合全面
   劣于原始查询单路，生产默认 full 只有论文报告值的 39%。
2. **归因**：不是"查询多样化有害"，而是**精排查询对象错（HyDE 段）+ 多路池稀释**；
   修精排可救至 0.584，但仍不如 raw。
3. **生产修正**：默认配置改为"**英文改写前置 + 原始查询单路**"，MRR 0.256→0.557
   （+117%）；0.653 为基准上界。
4. **端到端佐证**：检索差距真实传导到最终回答（3/4 维显著），切 raw 用户可感知。
5. **表述**：论文改为"多粒度查询生成经八路消融实证边界，原始单路最优"，
   不再断言假说成立。

---

## 3. 贡献（论文侧）

| 贡献 | 内容 |
|---|---|
| 系统 | CitrusQA：真实柑橘科研场景部署的领域问答系统（五层解耦、非对称双 Agent、确定性优先） |
| 方法 | 多粒度查询生成（一次 LLM 调用产出 HyDE+MQ+SUM）+ 六级可观测管道 + 八路消融实证 |
| 发现 | **生产默认配置非最优的发现-归因-修正完整闭环**（论文核心干货） |
| 量化 | 中文跨语言检索偏置（p<0.001）+ 多因素本质（跨域探针） |
| 工程 | 代码级早停/联网预算/思维链差异化控制/231 项回归的可审计架构 |

---

## 4. 产出一览（目录级全盘）

### 论文本体
- `C:\Users\Administrator\Desktop\论文投稿.docx`（518 段 / 28 表；备份 `.bak_20260825.docx`）
- 新增 **§7.12 实验九**（段 326-336）；摘要[5]/结论[389]/§8.3/§8.4/附录 A 已定稿

### 实验资产（`experiment\`）
- `data\`：eval120 84 题语料（含中英+gold 定位）、`e2e_sample_v1.json`（30 题抽样）、
  qa80、s5 双语 60 对
- `results\`：24+ JSON/CSV 实测快照；`fig_data\` 12 CSV（只读图数据仓库）；
  `figures\` 16 张 PDF/PNG；**`production_raw_trio.{json,csv}`**（#1）、
  **`e2e_three_groups\` 90 份轨迹 + summary.json**（#2）、两个 md 报告
- `scripts\`：20+ 实验脚本（exp1/1b/1c/2、production_raw_trio.py、e2e_three_groups.py、
  b1/B3、build_cor84_rewrite_cache.py 等），全部可复现

### 文档层（`agent\docs\`，11 份）
- **本文件** `execution_summary_v94.md`（总入口）
- `paper_revision_report.md`（全变更清单+裁决矩阵）
- `appendixA_calibration_v94.md`（参数终值+锚点行号）
- `reference_checklist.md`（文献核验结果）
- `numeric_questions_human_check.md`（6 条数值题人工核实清单）
- `submission_artifacts.md`（Highlights/CRediT/Data Availability）
- `postmortem_20260826_freeze.md`（卡死复盘+防复发纪律）
- `paper_execution_plan.md` / `experiment_plan_v94.md` / `plotting_instructions.md` /
  `full_architecture_audit.md` / `agent_extension.md`

### git（3 次提交）
- `837ddb2` v9.4: query_mode raw default + rerank_query defense + 231 tests
- `1167f06` docs: close out decision (query_mode raw)
- `a764ebd` docs(#4): reference verification + numeric-questions human check

---

## 5. 生产实际场景改了什么参数（共 3 处，已生效）

| 位置 | 修改 | 效果 |
|---|---|---|
| `config.yaml:32` | `query_mode: full` → **`raw`** | 9 路多路 → 单路原始查询；MRR 0.256→0.557（+117%）；**每次查询省 1 次 HyDE LLM 调用** |
| `multi_retriever.py:733` | `rerank_query=queries[0]` → `rerank_query or original_query or queries[0]` | 防御：未来误开 full 也不再踩 HyDE 段精排的坑（73% 损失根源） |
| `config.py:92` | QUERY_MODE 兜底默认 full → raw | 无 YAML 环境行为一致 |

- 测试同步：`tests/test_v94_param.py` 2 断言改新默认 + 1 锚定 full 基线
- **231 tests 全绿**（43.6s）
- 其余 20+ 参数（top_k、candidate_window=20、early_stop 6/0.25、阈值等）经实验
  确认均为最优，**保持不动**

---

## 6. 剩余待办（人工/作者侧，不动代码）

| 项 | 材料 | 说明 |
|---|---|---|
| 人工 Kappa（6 题） | `experiment\results\e2e_kappa_human_template.csv` | 人工打 1-5 分，回填后算 Cohen's Kappa ≥ 0.8 |
| 参考文献 [8]/[9]/[10] | `agent\docs\reference_checklist.md` | Crossref 查无 = 疑似占位引用，需裁定替换或改泛引（arXiv 侧 [2][3][5][11] 已验证） |
| 6 条数值题核实 | `agent\docs\numeric_questions_human_check.md` | 每题附 gold_note + 语料原文，人工打勾 |

---

## 7. 分项导航（按需深挖）

| 主题 | 文档 |
|---|---|
| #1 生产三对照 | `experiment\results\production_trio_report.md` |
| #2 端到端三组 | `experiment\results\e2e_three_groups_report.md` |
| 全论文变更 | `agent\docs\paper_revision_report.md` |
| 参数终值/行号 | `agent\docs\appendixA_calibration_v94.md` |
| 文献核验 | `agent\docs\reference_checklist.md` |
| 投稿附属物 | `agent\docs\submission_artifacts.md` |
| 卡死复盘/纪律 | `agent\docs\postmortem_20260826_freeze.md` |