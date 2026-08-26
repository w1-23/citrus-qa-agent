# CitrusQA 论文 — 新叙事写作思路（Master Narrative v2）

> **用途**：本文件是"论文实证研究重构"的完整写作蓝图 + 已核验的实验事实底座。
> 换模型后，请**只读本文件 + 底稿 `C:\Users\Administrator\Desktop\论文投稿.docx`**，
> 将删改后的论文写入 **`C:\Users\Administrator\Desktop\论文投稿CEA.docx`**（新文件）。
> **严禁直接覆盖 `论文投稿.docx`**（保留可回退的旧思路）。
>
> 本文件所有实验数字均来自 `experiment/results/` 真实 JSON，非估算。
> 每个数字后标 `[源文件]`。三处与原设想有出入的"诚实纠正"用 ⚠️ 标出，写作时必须采用纠正后的口径。

---

## 0. 战略定位（一句话）

把论文从"一篇优秀的农业 RAG 工程报告"升格为"**一篇揭示 RAG 底层机制的顶刊实证研究**"——

> **"我们将确定性架构作为科学仪器，揭示了决定 RAG 生成质量的底层非对称机制：**
> **端到端质量由 Evidence Availability（证据入池）主导，而非 Ranking Sophistication（排序精致度）。"**

---

## 1. 核心控制变量铁律：Prompt 必须绝对固定 ✅ 已核验通过

**结论：Generator Control Protocol 已被代码验证成立。** [源 `experiment/scripts/e2e_three_groups.py`]

- 生成侧 system prompt 是**模块级常量 `GEN_SYS`**（第 60–62 行），三组 A/B/C 共用同一常量，经 `call_llm(GEN_SYS, user, ...)`（第 145 行）调用，**字节级一致**。
- `call_llm` 固定 `model="deepseek-chat"`, `temperature=0`（确定性）。
- 唯一被操纵的变量是 `user` 消息里注入的 `passed_evidence`（第 144 行 `证据：{joined}`），即**检索侧差异**。
- 裁判侧 `JUDGE_SYS` 同为常量（第 64–70 行），四维打分对三组一致。

⚠️ **纠正 1（措辞精确化）**：Exp9 的 `GEN_SYS` 是**简化生成提示**（"只依据证据回答，引用用 [n]，不足则说证据不足"），**不含**生产 Supervisor 的"跨源仲裁规则/输出模板"。因此论文的 Generator Control Protocol 应这样写（诚实且更强）：

> *"To isolate retrieval-stage effects on generation, we enforce a strict Generator Control Protocol. A single module-level constant system prompt `GEN_SYS` (byte-identical, with `temperature=0`) is used across all comparative groups (A/B/C of Exp. 9, and the rewrite mitigation of Exp. 8). The sole manipulated independent variable is the `passed_evidence` list injected into the context window; the generation and judging prompts never vary. Any observed variance in quality is therefore strictly attributable to evidence availability and ranking, not prompt-induced bias."*

---

## 2. 新叙事结构（Master Narrative）

### 2.1 摘要重写要点
- 方法论升格：**Determinism-First as an Epistemic Foundation**（确定性优先 = 认识论基础：早停/预算/路由写成代码硬约束 → 系统可测量 → 才能做严格消融）。
- 三大发现：
  1. **Availability > Ranking**：多粒度查询假说被证伪；排序精致度对生成质量不敏感（引入 RRAI 量化）。
  2. **Language & Lexicon dictate Availability**：跨语言偏置的 2×2 实证 + BM25 对农业长尾实体的兜底必要性。
  3. **Determinism guarantees Fidelity**：信息增益早停 + 代码级引用过滤 → -57% 成本 + 100% 引用诚实 + κ=0.812。

### 2.2 引言故事线
- 痛点：农业 QA 的**极端 Evidence Availability 挑战** = 中文用户查英文文献的跨语言鸿沟 + 拉丁学名/品种登记号/农药缩写的长尾语义混淆。
- 三大核心发现（对应上三）。把"三大发现"明确排在引言贡献列表。

---

## 3. 新指标：RRAI（排序-召回不对称指数）—— 已用真实数字算好

### 3.1 定义
RRAI 量化"生成质量对召回率的敏感度 ÷ 对排序精度(MRR)的敏感度"：

```
ReS = ΔQ/ΔRecall@10        （召回敏感度，用 A 组 vs B 组）
RS  = ΔQ/ΔMRR@10           （排序敏感度，用 C 组 vs B 组）
RRAI = ReS / RS
```

### 3.2 真实输入 [源 `e2e_three_groups/summary.json` + `exp1b_rerank_attribution.json`]

| 组 | 配置 | MRR@10 | Recall@10 | R@10 | Quality(4维均) |
|---|---|---|---|---|---|
| A | raw 单路英文改写 | 0.6535 | 0.9881 | 0.9881 | 4.800 |
| B | full 多路(HyDE精排) | 0.2556 | 0.6667 | 0.6667 | 4.508 |
| C | full + 原查询精排 | 0.5837 | 0.8095 | 0.8095 | 4.592 |

（Quality = correctness/completeness/citation/evidence 四维均值：
A=(4.867+4.8+4.9+4.633)/4=4.800；B=4.508；C=4.592）

### 3.3 计算
- **RS**（C 对 B，同为 9 路池、pool@20 均=0.8095，仅精排不同）：
  ΔMRR = 0.5837−0.2556 = **0.3281**；ΔQ = 4.592−4.508 = **0.0837**
  → RS = 0.0837/0.3281 = **0.255**（每单位 MRR 只买 0.255 质量分）
- **ReS**（A 对 B，召回差）：
  ΔRecall = 0.9881−0.6667 = **0.3214**；ΔQ = 4.800−4.508 = **0.2917**
  → ReS = 0.2917/0.3214 = **0.908**（每单位召回买 0.908 质量分）
- **RRAI = ReS/RS = 0.908/0.255 ≈ 3.56**

⚠️ **纠正 2（重要）**：RRAI 的结论是 **≈3.6，而非"接近 1"**。生成器对**召回率的敏感度是对 MRR 的约 3.6 倍**。这比"接近 1"更强、更站得住：检索层里"把 gold chunk 捞进池子"（召回）的价值远大于"把它排得更靠前"（MRR）。

### 3.4 更强的表述（推荐写进论文）
B/C 两组**池完全相同**（pool@20 = 0.8095），仅精排不同 → MRR 差 2.28×（0.256 vs 0.584），但端到端质量无显著差（4.51 vs 4.59，Wilcoxon p≥0.158）。这**直接、无需任何指数**就证明"排序精致度对生成质量钝感"；而 A 组靠召回 +0.32 换来质量 +0.29（三维显著 p<0.05）。二者合起来即 RRAI≈3.6 的机制来源。

### 3.5 论文落点
§7.12（Exp 9 Discussion）新增一小节 **"Quantifying the Ranking–Generation Decoupling via RRAI"**，写 3.2–3.4 三步。

---

## 4. 未引用结果的落点 + 真实数据

| 结果 | 真实数据 [源] | 落点 | 诚实口径 |
|---|---|---|---|
| **① RIGS 标定** | `rigs_calib_grid.json`：grid=192，公式 `stop ⟺ \|E\|≥m ∧ ΔI<ρ·ΔI₁* ∧ medCE<δ·medCE₁ ∨ 空批 ∨ 预算B`；推荐 L1(δ0.4,m4,B2,q=0.3573)、L2(δ0.7,m4,B12,q=0.9681)、L3(δ0.4,m4,B6,q=0.9625) | §3.6/§4 代码级早停 + 附录 | ⚠️**纠正 3**：RIGS 用 `δ∈{0.4..0.7}、m∈{4..10}、B∈{2..16}` 三参，**不产生 (6,0.25)**。论文里的 (6,0.25)=(N_min,α) 来自**另一套简单早停**（exp4b 16 组扫描，零 gold 损失、-21% 轮次）。二者是两套机制： (6,0.25)=简单信息增益早停；RIGS=带 CE 门+信息增益的标定早停。**不要写 "(6,0.25) 由 RIGS 推导"**。正确写法见下 |
| **② 英文改写 2×2** | `rewrite_2x2.json`：recover_frac = {zh×hyde=off **0.750**, en×hyde=off **0.900**, zh×hyde=on **0.867**, en×hyde=on **0.967**}；n={80,80,30,30} | §5.3 缓解 + 热力图 `figures/rewrite_2x2.pdf` | ⚠️ 真实 2×2 是 **rewrite(zh→en) × hyde(伪文档)**，不是"改写×语言"——语言和改写是同一根轴（rewrite=on 即英文）。结论："翻译是 availability 的主导杠杆（+15pp），HyDE 是次要杠杆（+11.7pp），二者叠加到 0.967" |
| **③ CPU/DML** | `e6_cpu.json`/`e6_dml.json`：embed p50 cpu 2958ms → dml **163ms**（≈18×）；rerank p50 cpu 2200ms → dml **62.2ms**（≈35×） | §8 部署 + 附录 | DirectML 加速让无 GPU 农技站可用（一句带过） |
| **④ Token 画像** | `token_profile.json`：mean n_passed=7.63，tok_all=5259.6/题，k5 压缩省 **41.7%**，字段占比 text=81.5%/title=9.6%/section=8.9% | §7 成本+帕累托 | 早停主要砍"边际递减的合成 token"，保留证据抽取 token |

### 4.1 RIGS 段落的**诚实写法**（纠正 3 后的推荐文本）
> *"Early stopping is implemented at code level. A simple information-gain rule stops retrieval when at least N_min=6 unique evidence chunks are accumulated and the fraction of newly-added evidence in the latest round falls below α=0.25; a 16-configuration sweep confirmed zero gold-coverage loss and 21% round savings. We further developed Retrieval Information Gain Stopping (RIGS), a calibrated formulation `stop ⟺ |E|≥m ∧ ΔI<ρ·ΔI₁* ∧ medCE<δ·medCE₁ ∨ empty-batch ∨ budget B`, whose 192-configuration grid (δ∈{0.4..0.7}, m∈{4..10}, B∈{2..16}, with a cross-entropy gate) yields per-difficulty operating points: L1 (δ=0.4,m=4,B=2, quality 0.357 at 1389 tokens), L2 (δ=0.7,m=4,B=12, quality 0.968 at 4611 tokens), L3 (δ=0.4,m=4,B=6, quality 0.963 at 1688 tokens), demonstrating that stop thresholds are calibration-derived rather than heuristic."*

---

## 5. 证据饱和点 ESP（需补跑的小实验，可选）

- 定义：使 LLM 生成质量达最优 95% 所需的最小 passed_evidence 数。
- 方法：固定 A 组检索结果，人为截断 Top-1/2/3/5/10 注入生成器，画质量-入池数曲线。
- 预期：Top-3~5 处饱和。说明"只要 gold chunk 混进池子（哪怕 rank 低），质量就不崩"。
- ⚠️ 此实验**尚未跑**（`results/` 无对应产出）。论文若要写 ESP，必须先补跑；否则只写 RRAI（已有数据）。

---

## 6. Naive RAG Baseline（下界，诚实拼装）

⚠️ **纠正 4（诚实拼装）**：不存在一个"中文+dense-only+无rerank"的已测 baseline。现有**已测**的最近似点是：

| 配置 | MRR@10 | 来源 |
|---|---|---|
| **naive 单语（中文直搜，无改写，完整链路）** | **0.2816** | `production_raw_trio.json` zh_direct（这已是"中文用户对英文语料原始查询"的最近 LangChain/LlamaIndex 默认） |
| 密集-only 下界（EN，去掉 BM25） | **0.1431** | `exp1_ablation.json` -BM25 |

**推荐口径**（诚实且强）：
- 写"naive monolingual baseline = 中文直搜 MRR 0.282"（已测），raw+rewrite 0.557 是它的 **≈1.98×**，对齐英文上界 0.653 是 **≈2.32×**。
- 可选补一句"进一步去掉 BM25（dense-only）后 MRR 跌至 0.143"作为下界，但要说清这是英文 84 题的测定。

**不要直接写"Naive ≈ 0.14"当作已测**（那是把 EN 的 -BM25 消融叠加到 zh 基线上的推断，审查会被抓）。

---

## 7. 三条审稿防御（最终整合 + 真实数据挂钩）

### 防御 1：农业特异性（最核心）
- 论据：`-BM25` 使 MRR 0.2556→**0.1431**（-0.1125，全组件最大损伤），Recall@10 0.667→0.298。[`exp1_ablation.json`]
- 例证实体：拉丁学名（*Citrus reticulata*）、品种登记号（S-11/CRC 1459）、地方俗名（沙糖橘/沃柑）、农药缩写（2,4-D）。
- 话术要点：Dense 语义向量在这些近词/编号上语义碰撞 → 必须 BM25 字面精确匹配兜底；"hybrid retrieval 不是通用最佳实践，而是农业词汇忠实度的硬约束"。
- 时空冲突：黄龙病防控补贴（强时效→联网）vs 砧木育种机制（长期→本地文献）→ Supervisor 仲裁规则解此冲突。

### 防御 2：84 题"小与精"
- 领域专家逐句精标 gold chunk（非众包噪声）。
- 结构：中英等价 + 四类型(entity/functional/statistical/review) + 跨域探针(BRCA1/OsNRT2)。
- 配对 Wilcoxon 设计（同 30 题×三组），p<0.05 已有统计功效；结论处承诺开源评测集。

### 防御 3：Naive Baseline（必须补的下界）
- 用 §6 的 **0.282（已测）**，写"raw+rewrite ≈ 2× naive"；六级管道每级必要性 = 从 0.282 提到 0.557 的逐级贡献（exp1 ablation 的 delta 表支撑）。

---

## 8. 目标期刊矩阵（原样保留）

| 期刊 | IF(~) | 匹配 | 侧重 |
|---|---|---|---|
| **CEA** | 8.3 Q1 | ★★★★★ | **主投**。农业长尾实体(BM25)、跨语言偏置(中文农技人员查英文)、边缘部署(DML)；柑橘为 Testbed |
| **IP&M** | 8.6 Q1 | ★★★★ | 备选/平行。跨语言偏置、多粒度证伪、RRAI；弱化农业、强化检索-生成解耦 |
| **ESWA** | 8.5 Q1 | ★★★★ | 稳妥。多智能体、确定性规则、早停预算、231 回归测试 |
| **KBS** | 8.8 Q1 | ★★★ | 备选。知识库管道、跨源仲裁规则、引用忠实/幻觉抑制 |

---

## 9. 已核验的实验事实速查表（写作直接引用）

| 指标 | 值 | 源 |
|---|---|---|
| 跨语言 en/zh 实体 | 39/40 vs 16/40 (Δ0.575) | bias_quant |
| 跨语言 en/zh 功能 | 30/30 vs 13/30 (Δ0.567) | bias_quant |
| 跨语言 Wilcoxon | p=2.78e-10 | s3 |
| raw MRR/R@10 | 0.6535 / 0.9881 | production_raw_trio en_original |
| zh 直搜 MRR/R@10 | 0.2816 / 0.4405 | production_raw_trio zh_direct |
| rewrite 生产 MRR/R@10 | 0.5572 / 0.8690 | production_raw_trio rewrite_en |
| full(旧默认) MRR/R@10 | 0.2556 / 0.6667 | exp2/full |
| -BM25 MRR | 0.1431 | exp1_ablation |
| -Rerank MRR | 0.3280 | exp1_ablation |
| SingleQuery MRR | 0.6535 | exp1_ablation |
| 归因 rerank 修正 MRR | 0.5837 (pool@20 不变 0.8095) | exp1b |
| 归因双修正 MRR | 0.5808 | exp1c |
| E2E A/B/C 质量 | 4.800 / 4.508 / 4.592 | e2e summary |
| E2E Wilcoxon | correctness .026 / completeness .015 / evidence .042 / citation .294 | e2e 报告 |
| C vs B | 全维 p≥0.158 | e2e 报告 |
| RRAI | ≈3.56 (ReS 0.908 / RS 0.255) | 计算见 §3 |
| 2×2 recover | 0.750/0.900/0.867/0.967 | rewrite_2x2 |
| RIGS 推荐 | L1(δ.4,m4,B2) L2(δ.7,m4,B12) L3(δ.4,m4,B6) | rigs_calib_grid |
| 早停(6,0.25) | 16 组扫描、零损失、-21% 轮次 | exp4b |
| 成本 | 10000→4283 tok (-57%) | exp5 |
| 延迟 LanceDB/Qdrant | 14.4 / 91.5 ms (6.4×) | exp6 |
| DML embed/rerank p50 | 163 / 62.2 ms | e6_dml |
| κ 交叉验证 | 0.812, 22/24 | qwen_judge_scores |
| 回归测试 | 231 全绿 | 论文 §7.1 |

---

## 10. 执行清单（换模型后操作）

1. 读本文件 + `论文投稿.docx`。
2. 按 §2 新叙事重写摘要/引言/结论；按 §3.4 写 RRAI 小节；按 §4.1 写 RIGS 诚实版本；按 §6 写 Naive baseline；按 §7 织入三条防御。
3. 图：新增 `figures/rewrite_2x2.pdf`（2×2 热力图）；其余沿用已优化 12 图（fig1–12）。
4. **输出到 `论文投稿CEA.docx`（新文件），不覆盖 `论文投稿.docx`。**
5. 生成器控制协议声明放 §7.1（用 §1 的诚实版）。

---

## 附：本轮已完成的实验侧产物

- ✅ Generator Control Protocol 代码核验（GEN_SYS/temperature=0/JUDGE_SYS 常量）
- ✅ RRAI 计算（≈3.56）
- ✅ `figures/rewrite_2x2.pdf/.png`（2×2 热力图）
- ✅ 全部真实数据提取 + 4 处诚实纠正（纠正 1–4）
- ✅ 本"写作思路留存"文档
- ⏳ 尚未：ESP 补跑实验（论文若要写则需先补）；`论文投稿CEA.docx` 正文撰写（留给换模型）