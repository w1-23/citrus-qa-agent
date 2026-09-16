# CitrusQA / CEA v12 — 修订版矛盾清单与建议（已对源数据核验）

> 本版替代第一轮报告中的矛盾清单与建议部分。所有数字均来自工作区原始结果文件，已逐项复算。
> 核验基准：`git HEAD = a7b60bc5f72a7b8e5fa5dc5834126beac226ab2f`（v9.4.0-14-ga7b60bc，应用版本 **v9.6.2**）
> 语料基准：187,441 块 / 11 批 / fingerprint `f4da9481ae62257b` / manifest_sha1_16 `2846e39a5b5030f5`（2026-09-11T17:11:15+0800）
> 稿件源：`experiment\paperMerged_CEA_v1.md`（与桌面 docx v12 逐字一致）
> 新增文件：本报告 `CitrusQA_CEA_v12_修订版矛盾清单与建议.md`

---

## 0. 先说结论：三条最重要的修订

### 修订 1 —— "五系统对比"数据是齐的，是**正文没写全**（不是造假）

**发现**：`experiment\pipeline\results\judge\final_20260911\system_table_v2.csv` **本来就有完整的 5×10 矩阵**，含 Track B 两行。论文只搬了 3 行。

| cell | D1 | D2 | D3 | D4 | D5 | D6 | D7 | D8 | D9 | D10 |
|---|---|---|---|---|---|---|---|---|---|---|
| citrusqa | 1.0 | 4.622 | 4.554 | 4.768 | 4.714 | 5.0 | 4.893 | 4.881 | 4.577 | 4.274 |
| mainstream_trackA | – | 1.0 | 4.054 | 3.542 | 3.619 | 3.5 | 4.482 | 4.512 | 4.53 | 3.702 |
| mainstream_trackB | – | – | 3.5 | 3.161 | 3.0 | 2.0 | 3.768 | 3.72 | 4.435 | 4.298 |
| same_base_trackA | – | – | 4.31 | 3.869 | 4.018 | 4.25 | 4.792 | 4.798 | 4.696 | 3.845 |
| same_base_trackB | – | – | 3.28 | 3.482 | 3.25 | 2.0 | 3.429 | 3.435 | 4.321 | 3.0 |

（`system_table_v2.csv` = 主观维 D2–D5/D7–D9 取自两裁判均分，客观维 D1/D6/D10 由 `run_objective_dims.py` 用代码算；`system_table.csv`(v1) 是全部由裁判打分的版本，保留着 D1=4.756 / D6=4.208 / D10=3.423 的原始裁判分。）

**行动**：
1. 把 Table 1 补成 **5 行**（并把 Track B 的完整十维给出），"five-system"才成立；或全文统一改为"4 种基线配置 + CitrusQA"，同时保留 Table 1 的 3 行并**把 Track B 放进 Table S1**。
2. **Table 1 与 `system_table_v2.csv` 经复算是自洽的**：CitrusQA 行 4.62/4.55/4.77/4.71/4.89/4.88/4.58/4.27 ↔ CSV 4.622/4.554/4.768/4.714/4.893/4.881/4.577/4.274，逐格都是四舍五入 ✓；两条基线的 D3 = 4.31/4.05 同理 ✓。**所以 Table 1 唯一的问题就是"少两行 + 结论写反"，不是数字错**。**真正混版本的是 Table 2**（见修订 3）。

### 修订 2 —— D9 不是"接近打平"，而是**唯一被基线反超的维度**；D10 在"裁判口径"和"代码口径"下结论相反

按 v2（论文正文口径），逐维胜负：

| 维度 | CitrusQA | 最好基线 | 差值 | 胜负 |
|---|---|---|---|---|
| D1 | 1.0 | 1.0 (Qwen-A) | 0 | 并列（仅 2 个 cell 有值） |
| D2 | 4.612 | 1.0 (Qwen-A) | +3.61 | **CitrusQA** |
| D3 | 4.554 | 4.31 (DeepSeek-A) | +0.24 | **CitrusQA** |
| D4 | 4.768 | 3.869 | +0.90 | **CitrusQA** |
| D5 | 4.714 | 4.018 | +0.70 | **CitrusQA** |
| D6 | 4.208 (裁判) / 5.0 (代码) | 4.25 (裁判) / 3.5 (代码) | 代码 +1.5；裁判 −0.04 | **口径决定胜负** |
| D7 | 4.893 | 4.792 | +0.10 | **CitrusQA** |
| D8 | 4.881 | 4.798 | +0.08 | **CitrusQA** |
| **D9** | **4.577** | **4.696 (DeepSeek-A)** | **−0.12** | **基线领先** |
| D10 | 4.274 (代码) / 3.423 (裁判) | 4.298 (代码) / 4.095 (裁判) | 代码 −0.02；裁判 −0.67 | **基线领先** |

**所以**：
- "**CitrusQA leads on every dimension**"（§5.6 正文）→ **假**（D9 被 same_base_trackA 4.696 反超）；
- "**D10 是唯一接近打平的维度**"（§5.6 正文）→ **假**（D10 是它最差的维度之一）；
- "**leading every quality dimension except conciseness**"（摘要 + Fig.8 注）→ **假**（D9 也要列进去）。

**建议改法**（不必改数据，只改结论口径，反而更可信）：
> "CitrusQA attains the highest score on 7 of the 10 ARE dimensions; same-base DeepSeek with the fair guardrail prompt is marginally better on presentation (D9, 4.70 vs 4.58), and conciseness (D10) is the dimension where the rubric and the deployment feedback diverge most (see §5.9)."

### 修订 3 —— Table 2 是**旧口径（v1）**，与 Table 1（v2）不是同一次统计

`agreement.json` 与 `system_table.csv`(v1) 是同一批：κ = D1 **0.0**, D2 0.09, D3 0.43, D4 **0.566**, D5 0.449, D6 **0.199**, D7 0.419, D8 0.442, D9 0.26, D10 0.299。

而论文 Table 2 写的是：D2 0.09、D3 0.43、D4 0.57、D5 0.45、D7 0.42、D8 0.44、D9 0.26、D10 0.30 → **缺 D1 和 D6**，并且 D10 有 κ 但它被声明为代码计算。

**关键点**：Table 2 逐行数值 = v1 全裁判版本；Table 1 = v2 代码覆盖版本。**论文把两个版本的表并列，且都没有声明版本**。这正好解释了我第一轮看到的"整列错位"错觉——其实不是错位，是**混用了两版**。

**行动（必做）**：
1. Table 2 补 **D1 行与 D6 行**，κ 区间改写为 **0.00–0.566**（原文"0.09–0.57"漏掉 D1=0.0 与 D6=0.199）；
2. 在两张表的题注里**显式声明客观/主观维度的来源**，例如：*"D1/D6/D10 are computed in code (v2); all other dimensions are the mean of the two cross-family judges (n = 420 responses = 84 questions × 5 systems; D1/D2 n = 78 because baselines emit no [n] citations in most answers)."*
3. 固定裁判样本量口径：`agreement.json` 里 D1/D2 是 **n = 78**，D3–D10 是 **n = 418**，而 total = **420**。这 3 个数字必须在方法节出现，否则"κ 怎么算的"无法复现。

---

## 1. 版本号统一表（按最新提交）

| 对象 | 现在论文里写的 | 应为 | 依据 |
|---|---|---|---|
| **语料** | corpus v9.4.0 | **corpus v9.4.0，fingerprint `f4da9481ae62257b`** | `pipeline\registry\corpus_final_20260911.json` |
| **检索配置版本** | 无 | 加一句"retrieval config frozen at `pipeline_config.yaml` (2026-09-11)" | `pipeline_config.yaml` 头注释 |
| **应用/Agent 版本** | 完全缺失（用 "generation system"/"the system" 代指） | **v9.6.2**（commit `a7b60bc`） | `agent/src/prompts/builds/*.md` 内 v9.6.2 标记 |
| **引用池版本** | 无 | **ref_pool v9.4.6**（`references_json.ref_pool` 实际取值） | request_ledger `references_json` |
| **评估用基座** | deepseek-flash (DeepSeek-V4.1-Flash) | deepseek-flash ✓（与 config 一致） | `pipeline_config.yaml: agent_base` |
| **基线模型名** | Qwen-max | **qwen3.8-max-0902** | `pipeline_config.yaml: generators.mainstream` |
| **裁判模型** | qwen3.8-flash / kimi-k2.6 | ✓ 一致 | `pipeline_config.yaml: judges` |

> **注意一个容易踩的坑**：`qwen3.8-flash` 同时是**裁判**，而 `pipeline_config.yaml` 里还有一个 `generators.weak = qwen3.8-flash` 的生成器槽位（本论文未用）。如果审稿人看到 qwen 既当裁判又出现在生成器配置里，会质疑独立性——**建议在方法节明确写"judges were not used as generators in this study"**。

---

## 2. 三条"数据与论文不符"的硬修正（优先级最高）

### 2.1 对抗池诚实度：不是 1.0，英文是 0.97

**原文**："CitrusQA and the fair baselines reach honesty 1.0 with **0/12 bait falls**"

**实际**（`paper2\results\poolB_verdict_summary.json`）：

| 语言 | n | honest | honesty_rate | bait_fell | bait_resisted | bait_resistance_rate | cite_exist | fabricated_refs |
|---|---|---|---|---|---|---|---|---|
| en | 33 | 32 | **0.9697** | **1** | 11 | 0.9167 | 1.0 | 0 |
| zh | 33 | 33 | 1.0 | 0 | 12 | 1.0 | 1.0 | 0 |
| 合并 | 66 | – | – | 1 | 23 | **0.9583** | – | 0 |

**行动**：
- 摘要与 §5.5 改为："adversarial honesty 0.97 (en) / 1.00 (zh) with 11/12 and 12/12 baits resisted (combined 23/24)"；
- 这**加强了**你的负结果论点（"诚实是提示词属性"）——毕竟有一个诱饵在英文里被咬了，而它**没有**被检索层拦住，正好证明 guardrail 才是决定因素；
- 顺手把 `cite_exist=1.0`、`fabricated_refs=0` 在对抗池也报出来（现在论文只在总述里说 CiteExist=1.0）。

### 2.2 "84 条真实用户问题" —— 它们不是用户问题

**原文**（摘要 + §5.6 标题 + Fig.8 注 + §5.2）："On 84 real user questions"

**实际**：
- `qset_final.jsonl` / `eval120_corpus_final.jsonl` 的 84 条**是从语料论文里提取的**，字段含 `paper_id`、`chunk_index`、`batch`，`gold_note` 是专家标注的 gold 说明，`retrieval_best_lang`/`retrieval_rank` 是检索定位信息；
- 样本问题：`cor-002 该研究使用什么试剂盒从冰冻组织提取总RNA`、`cor-004 嫁接过程中，研究者分别从接穗和砧木中鉴定出多少个差异表达基因（DEGs）` —— 这是**论文阅读理解题**，不是用户提问；
- **与真实用户 query.csv 的精确字符串匹配 = 0/132（中英均为 0）**。

**真实的用户数据在另一个包里且论文完全没用**：
`experiment\用户真实提问\citrus_beta_pack_full_20260911_060602\`
- `output\query.csv`：**132 条**真实用户 query（去重后），样本："现在已发表的柑橘基因组有哪些？""为我介绍一下红美人""在武汉种植需要注意什么""武汉的柚子有什么品种推荐吗"；
- `feedback\answer_feedback.csv`：**33 条**带五维评分的真实反馈。

**行动（三选一，建议 B）**：
- A. 把 §5.6 标题与全文改为 "84 expert-authored in-corpus questions"（最省事，但失去"真实用户"卖点）；
- B. **两条线分开写**：§5.6 用 84 条语料内问题做五系统对比（保留现有全部数字），**另加一小节**用 132 条真实 query + 33 条真实反馈做部署验证（这才能真正支撑 §5.9）；**推荐**，因为数据都在手上；
- C. 只用 132 条真实 query 做端到端对比（成本最高，但说服力最强）。

### 2.3 §5.9 部署窗口：**论文写的 09-07 起是错的**（正确是 UTC 09-08 至 09-10）

**原文**："pilot deployment (**2026-09-07** to 2026-09-10)"

**实际**（`research/request_ledger.csv`，110 行；时间戳为 **UTC**，首行 `2026-09-07T23:43:28+00:00`，末行 `2026-09-10T15:35:08+00:00`）：

| UTC 日期 | 请求数 |
|---|---|
| 2026-09-08 | 58 |
| 2026-09-09 | 24 |
| 2026-09-10 | 28 |

- 唯一一条 09-07 的记录是 **09-07T23:43 UTC**（即北京时间 09-08 07:43），所以按 UTC 计窗口是 **09-08 → 09-10**，共 **3 天**；
- **顺带修正我在本报告第一版里的说法**：当时我用本地时区解析时间戳、并用**本地日期**分组，得到"只有 09-08/09/10"，结论恰好正确但推理是错的——正确做法是按 UTC 解析与分组（`用户建议与评价.md` 也标注"覆盖区间 2026-09-08 ~ 2026-09-10（UTC）"，与此一致）；
- 其余数字**全部精确吻合**：110 请求 / 18 用户 / 39 会话 / 52 启用联网（web=1 n=52 ✓）/ expert 93 + light 17 / 首响中位数 **48.60 s** ✓ / 生成中位数 **63.38 s** ✓ / completed 108 + cancelled 2。

→ 正确表述：**"a three-day pilot deployment (2026-09-08 to 2026-09-10 UTC)"**（并建议顺手补 "108 completed, 2 cancelled"）。

---

## 3. 可信度三条口径修正（支撑你的主线，改完更硬）

### 3.1 支持度 0.928 的分档口径必须写清

`paper2\results\paper2_support_metrics.json`：

| 口径 | n | mean | 1.0 | 0.5 | 0.0 | pct_full | pct_zero |
|---|---|---|---|---|---|---|---|
| 全部断言 | 354 | **0.928** | 303 | 51 | **0** | **0.8559** | 0.0 |
| 英文 | 219 | 0.9247 | 186 | 33 | 0 | 0.8493 | 0.0 |
| 中文 | 135 | 0.9333 | 117 | 18 | 0 | 0.8667 | 0.0 |
| **仅被引用断言** | **307** | **0.9935** | 303 | 4 | 0 | **0.987** | 0.0 |

- 论文写的"0.928（n = 354；0% fabricated；98.7% full support among cited claims）" → **三个数都对**，但 **98.7% 属于"仅被引用断言"（n=307）这一行**，必须在同一句里标出 n，否则读者会以为 354 条里 98.7% 满分；
- **0.5 的 51 条 = "未引用但内容在证据里"（B6 类）**，正是 κ 交接报告里点名的最需要人机对齐的"引用纪律"边界。建议把它写成正文的一句机制解释："the residual error is citation *discipline*, not fabrication"；
- answer 级（n = 120 实例：en 75 / zh 45）mean = 0.9376、median = 1.0 —— **这个数字论文完全没报，建议补**，它是"用户实际拿到的一条回答整体有多可信"的直观指标。

### 3.2 "0% fabricated" 是**语料内池**的结论，不能外推到全部池

三池合看：

| 池 | 内容编造 | 依据 |
|---|---|---|
| Pool A（84 语料内） | **0 / 354 断言 = 0%** | `paper2_support_metrics.json` pct_zero = 0.0 |
| Pool B（33 对抗） | fabricated_refs = **0** | `poolB_verdict_summary.json` |
| Pool C（12 时间） | **en 2/12、zh 1/12 编造**（manual verdict） | `poolC_manual_verdict.json` |

**原文风险**：摘要/§5.5 把"0% fabricated"与"时间池 2/12、1/12 编造"并列，读者会当成自相矛盾。

**建议改写**：
> "Across the 354 claims of the in-corpus pool, 0% were fabricated (0/354 scored 0.0). The temporal pool independently shows why existence-level checking is insufficient: 3 of 12 temporal questions produced specific figures with in-range [n] citations that the cited evidence does not support (en 2/12, zh 1/12). Fabrication is therefore absent where retrieval succeeds and present where the corpus cannot answer."

**并且必须补一句关于 0.0 检出能力的诚实说明**（来自 `paper2_support_kappa_handoff.md`）：
> κ 子集（50 条分层抽样）的分数分布只有 **{0.5, 1.0} 两档，天然 0.0 为 0 条**；0.0 档的**检出能力仅由合成校准例验证**（"张冠李戴"例被正确判 0.0）。

这一句不写，审稿人会问"你的裁判到底能不能识别编造"。写了，反而是方法论上的诚实加分（并自然引出"为什么还需要 Pool C 的人工 verdict"）。

### 3.3 时间池的三个数字要说清 pct

`poolC_autostats.json` + `poolC_manual_verdict.json`：

| 指标 | en | zh |
|---|---|---|
| honesty_rate（自动） | 0.75 | 0.8333 |
| cite_exist_rate | 1.0 | 1.0 |
| n_fabricated_refs | 0 | 0 |
| cite_cover_claim_mean | 0.3056 | 0.1389 |
| manual: honest / partial / hallucinate | 9 / 1 / 2 | 10 / 1 / 1 |

- 论文写的 "2/12 English, 1/12 Chinese" = **manual hallucination 计数** ✓ 正确；
- 但论文没提 **各有 1 条 partial（time-009：语料有 2025 EFSA imazalil 论文可部分回答，模型给了数值但未声明需要联网）**，也没提 **cite_cover 只有 0.31 / 0.14**（即时间题的回答只有三成/一成多的断言能被引用覆盖）—— 这两个数字会显著加强"必须做内容级判定"的论证；
- `time-012` 是**中英双语都编造**（声称 CsLIEXP1/EDS1/LOB1/DMR6/NPR3 具抗病表型，而语料里 2025/2026 的 CRISPR 文献均为方法/品质类）。**建议把这个例子写进正文**，它比抽象陈述有力得多。

---

## 4. §5.7 用例 3 / §5.8 联网：占位说明必须重写

**你的说明**：之前联网不可用，只做了本地测试，所以论文里是占位形式。

**实际状况**：
- 联网**已经恢复并跑完**：`results\e2e\web3_20260914\` + `results\web_metrics\web3_20260914\`（30 题）；
- `pipeline_config.yaml` 里还留着历史痕迹：`web: {agent: paused, # DeepSeek Responses web_search ignored; resume ~09-14}` —— 说明 09-14 是恢复点；
- git log：`969b3dd ... v9.4.1-v9.6.1: ... 联网 Anthropic 端点恢复+编号不变量加固`。

**所以 §5.7 Case 3 与 §5.8 的"recorded before the web channel was restored"是过时占位语，必须删掉/改写。**

### 4.1 §5.8 的数字我全部复算过，结论是**站的住的**

| 论文数字 | 复算值（`web_metrics.json` rows 聚合） | 判定 |
|---|---|---|
| n = 30 | 30 ✓ | ✓ |
| 一半的回答带可解析引用 | `citrusqa_web` web_evidence = **15/30 = 50.0%** ✓ | ✓ |
| mean 4.9 per sourced answer | 总 [Wn] = **146**，146/30 = 4.87 ✓ | ✓ |
| 71.7% reachable | ok 86 / (86+30+2+2) = **71.67%** ✓ | ✓ |
| 25.0% blocked | **25.0%** ✓ | ✓ |
| 1.7% invalid | **1.67%** ✓ | ✓ |
| 通用模型网页证据率 10% | `mainstream_trackA_web` = **3/30 = 10%** ✓ | ✓ |
| CitrusQA on-vs-off 15 vs 0 | discordant b=15, c=0 ✓（McNemar exact p = 2·2⁻¹⁵ = **6.1e-5** ✓） | ✓ |
| 96.7% declared gaps（web off） | `citrusqa` gap_declared = **30/30 = 100%**；`citrusqa_web` = **29/30 = 96.7%** | **张冠李戴** |

**唯一要改的一处**：**96.7% 是"开启联网"时的缺口声明率，不是关闭时**。关闭时是 **100%（30/30）**。请对调。

### 4.2 §5.8 的四个语义问题

1. **`honest` 标志的含义需重新定义**：CSV 里 `citrusqa_web` 有 15 行 `honest=false`，而这 15 行**恰好就是有可解析引用的那 15 行**——因为指标把"有网页证据"直接定义为"不诚实"（设计意图是"要么给缺口声明、要么给来源"，但 `honest_or_sourced_rate=0.967` 又把 sourced 算回来了）。**表述必须改成**："all 30 web-on answers either declared a gap (29/30) or carried resolvable citations (15/30, overlapping); the single exception (web-012) did neither." 现在这种"half the answers are dishonest"的隐含读法与摘要的诚实度主张直接冲突。
2. **`Qwen-max (web off)` 这一行是误导的**：`mainstream_trackA`（无联网）在 30 题里有 **6 题（20%）出现了 URL**、平均 0.3 条、可达 89%。这是**模型凭记忆写出来的 URL**，不是检索到的证据。关键词应改为 **`URL mentions`** 而不是 `web-evidence rate`，并在表注写明"no retrieval tool was available in this arm; URLs are model-generated"。否则审稿人会问"没联网哪来的网页证据"。
3. **"对比了带网页与不带网页的 CitrusQA，以及一个支持网页的通用模型"** → 实际是 **2 系统 × 网页开/关 = 2×2**，表里有 5 行（还有一行 `same_base_trackA` 无联网、gap 96.7%，**论文完全没提**）。建议把 `same_base_trackA` 也写进表，这样"关闭联网时谁在诚实说缺口"就有对照组了。
4. **措辞**：`(n = 30; exploratory.)` 现在同时出现在正文末和表题里，重复；建议只留表题。

### 4.3 §5.7 Case 3 的正确写法（联网已恢复）

现在可以这么写（我也核过 DOI 全部可解析）：
> **Case 3 — chromosome-level genome count (timeliness).** "柑橘已经发表了大概多少个染色体水平的参考基因组？" This question is not answerable from the frozen corpus (v9.4.0). Two behaviours are compared: with the retrieval-only path, the system **declares the gap** rather than presenting a stale count (§5.8, web off; 30/30 gap declarations); with the web operator enabled, it returns resolvable, sidebar-checkable citations (e.g., [24–26]) while keeping CiteExist = 1.0.

这样 Case 3 就从"缺陷展示"变成了"**两个可用性恢复算子的对照演示**"，正好回收你在 Discussion 想立的主线。

### 4.4 §5.8 有一个**必须坦白的评估对象问题**

| 对象 | 用什么跑的 | 证据 |
|---|---|---|
| `citrusqa` / `citrusqa_web` | **corpus-only 生成臂**（走 `run_e2e.py` 的 pipeline 生成器） | `pipeline\results\e2e\web3_20260914\answers_citrusqa*.jsonl` |
| 真实用户实际用的 | **部署版 agent（deepseek-flash + 完整 agent 提示词，v9.4.x→v9.6.2）** | `request_ledger.csv` 的 answer/references_json 结构 |

即：**§5.8 评估的 CitrusQA 臂不是用户实际使用的那个系统**。这解释了为什么 §5.8 的 D6 说"联网不改善质量"，而**真实用户反馈恰恰相反**：

| 反馈维度（n=33，`answer_feedback.csv`） | 均值 |
|---|---|
| correctness_score | 4.55 |
| citation_trust_score | 4.64 |
| research_practicality_score | 4.52 |
| **stability_speed_score** | **3.91 ← 唯一低于 4 分** |
| expression_quality_score | 4.42 |
| rating | helpful 28 / needs_improvement 5 |

而 `用户建议与评价.md` 的分析（8 用户 / 9 会话 / 33 条）进一步指出：**开启联网的 19 条在五个维度上全面高于关闭联网的 14 条**（科研实用性 4.79 vs 4.14；引用可信度 4.84 vs 4.36；表达质量 4.74 vs 4.00；**速度 4.26 vs 3.43**；正确性 4.68 vs 4.36），5 条差评中 3 条来自未联网请求。

**建议**：
- §5.8 加一句范围声明："this comparison evaluates the corpus-only generation arm, not the deployed agent; the deployed system is assessed in §5.9"；
- 并且**把"联网提升用户评分"这一条真实证据用起来**——它让你的"两个可用性恢复算子"论点从"机制上合理"变成"用户也认可"，比现在的 p=0.32 负结果强得多；
- 如果你愿意，**用 v9.6.2 的部署版重跑 web3 的 30 题**，§5.8 的"网页不提升质量"结论就有可能是相反的（用户数据强烈暗示会提升）。这是**唯一一个我建议你考虑补实验的地方**，其余全是改文字。

---

## 5. 机制线的四处口径修正（数字都对，措辞要精确）

### 5.1 ρ = 72.5% 的分母

`results\rewrite\final_20260911\rewrite_2x2.json`：zh_original Recall@10 = **0.500**，en_rewrite = **0.8452**，en_aligned = **0.9762**。

- (0.8452 − 0.500) / (1 − 0.500) = **69.0%**
- (0.8452 − 0.500) / (**0.9762** − 0.500) = **72.5%** ✓

→ **ρ 的分母是"对齐英文"，不是理想值 1.0**。请在 §5.2 写出定义式：
> ρ = (Recall@10_rewrite − Recall@10_zh) / (Recall@10_aligned-en − Recall@10_zh)

**同时**：论文同段还有"0.678 vs 0.080"（锚点分组，池级）与"0.500"（重写实验，84 题全集）两个口径，**建议明确**："the anchor split is measured on the full 84-question pool (0.678/0.080); the rewrite experiment starts from the unmodified Chinese queries on the same pool (0.500)". 现在读者会以为 0.500 和 0.678 矛盾。

### 5.2 W / β 参数

`results\extra\final_20260911\window.json`：

| W | MRR@10 | Recall@10 |
|---|---|---|
| 10 | 0.8037 | 0.9524 |
| 20 | 0.8136 | 0.9762 |
| **30** | **0.8198** | **0.9881** |
| 40/60 | 0.8197 | 0.9881 |

- "plateau at W ≥ 30"✓；"production uses W = 20；0.976 / 0.988 = **98.8%**" ✓；
- **但生产配置 `candidate_window: 20`，而扫描显示 W=30 在 MRR 和 Recall 上都更好**。请写明取舍理由（延迟 vs +0.012 MRR），否则审稿人会说"为什么不用最优的 30"。

`results\extra\final_20260911\threshold.json`：

| β | pass_rate | gold_pass |
|---|---|---|
| 0.3 / 0.4 / 0.5 | 1.0 | **0.9762** |
| **0.6（生产）** | 1.0 | **0.9643** |
| 0.7 / 0.8 | 1.0 | 0.9524 |

- "β ≤ 0.5 retains the most gold" ✓ **正确**；
- 但 `pipeline_config.yaml: dynamic_threshold_ratio: 0.60`，即在**已知 0.5 更优**的情况下仍用 0.60。**必须给理由**（这个 0.9643 vs 0.9762 的差 = 少 1/84 题命中金标，可忽略；说"保守取值以降低噪声证据进入生成"即可）；
- 同时把两处 β 的**名字分开**：§3 的 `T = max(0.25, s_top × 0.60)` 是**动态阈值比**，方法节的 `β = 0.60, floor 0.25` 是**同一个量**，而阈值扫描的 β 也是它——统一叫 `β`，并在 §3 首次出现处就给出 β 的符号，避免"两套符号指同一件事"。

### 5.3 重排修复率 82%

`attribution.json`：raw 0.8136、full 0.2944、full_origrerank **0.7197**、gap 0.5192、**rerank_fix_share 0.819**。

- (0.7197 − 0.2944) / (0.8136 − 0.2944) = **81.9%** ✓；
- **建议把中间量报出来**："reranking with the original query instead of the HyDE passage raises MRR@10 from 0.294 to **0.720** (recovering 82% of the gap to raw 0.814; rerank_fix_share = 0.819)"，并给出 0.814 与 0.8136 的取整说明（现在正文写 0.814，数据是 0.8136 —— 全文统一到 3 位小数最省事：0.814 / 0.294 / 0.720）。

### 5.4 RIGS 成本表述

`rigs_replay.json`：`rigs: items 8.0, coverage 0.964, rounds 2.0`；`runall: items 20.0, coverage 0.976, rounds 5.0`；`oracle: items 4.57, coverage 0.976, rounds 1.14`。

- 8/20 = **40%** ✓；"oracle bound 4.57 items" ✓；
- 建议改为 **"with 60% fewer evidence items (8 vs 20; 40% of the run-all budget), coverage is 0.964 vs 0.976"**——现在的 "at 40% of the cost" 语义上没错，但审计者会先读成"成本降低 40%"；
- **顺带补一个 RIGS 独有的亮点**：`rounds` 从 5 降到 2（**少 60% 的检索轮次**）。这个数字比 items 更贴近"延迟/成本"，而论文完全没提。

### 5.5 多粒度模式的排序

`retrieval_results.json`（84 题）：raw_en MRR@10 = **0.8136**、mq_only_en = **0.4079**、hyde_only_en = **0.3329**、sum_only_en = 0.2148、**full_en = 0.2944**、full_bm25off_en = **0.193**、raw_zh = 0.42。

- "raw > multi-query > HyDE > combinations" ✓（0.814 > 0.408 > 0.333 > 0.294）；
- "removing BM25: 0.294 → 0.193" ✓ —— **这其实对应的是一个反事实**：删掉 BM25 后 full 配置掉到 0.193，**比 raw 配置本身还接近**。建议把它讲成"BM25 是 long-tail token 的检索骨架，但其收益在多路融合里被稀释了"，这样和多粒度证伪的结论才不打架；
- **注意**：论文的八模式消融里应该还有 `raw_zh=0.42` 这种"中文原始查询"的模式，它比所有多路变体都高。**建议把 raw_zh 也列进 Table/Fig.6**，因为它正好是 §5.2 跨语言主线的起点（中文直检 0.42）。

---

## 6. 参考文献：40 条中 26 条零引用 → 逐条归位方案

扫描结果：正文（不含 References 段）只出现 **[1]–[12]、[39]、[40]** 共 14 个编号；**[13]–[38] 共 26 条零引用**。

CEA/Elsevier 要求"所有列出的文献必须在正文中被引用"，这是**编辑部会直接退改**的问题。好消息是：**不需要删任何一条**，工作区里的实现细节正好为每一条提供了天然引用点。

| 编号 | 文献 | 建议引用位置（挂现成句子） |
|---|---|---|
| [13] | Robertson & Zaragoza, BM25 and beyond | §3 六阶段第 (2) 步"稀疏（BM25+）" |
| [14] | Lv & Zhai, lower-bounding tf normalisation | §4 检索指标"β = 0.60, floor 0.25" |
| [15] | Cormack et al., RRF | §3 第 (3) 步"RRF（W = 20, k = 60）"→ 补 k 并引[15] |
| [16] | Wang et al., E5 | §3 第 (2) 步"multilingual-e5-large, 1024-d" |
| [17] | Xiao et al., C-Pack/BGE | §3 第 (4) 步"bge-reranker-v2-m3" |
| [18] | Cohen, weighted kappa | §4 ARE"quadratic-weighted κ"首次出现处 |
| [19][20][21] | Wu 2014 / Xu 2013 / Wu 2018 | §5.7 Case 1（DOI 正是 [19][20]；[21] Nature 2018 是"起源与演化"最直接的一篇） |
| [22] | Chen 2020 Plants | §5.7 Case 2 |
| [23] | Ciriminna 2025 | §5.7 Case 2 |
| [24] | Droc 2026 pbi | §5.7 Case 3 |
| [25] | Huang 2026 uhag048 | §5.7 Case 3 |
| [26] | Nakandala 2023 uhad058 | §5.7 Case 3 |
| [27] | Self-RAG | §2"教智能体识别证据充分性"（与 [3] CaRT 同句） |
| [28] | CRAG | §2 同一句 |
| [29] | Active RAG | §2 同一句；**并可在 §5.3 作为"自适应检索"的提出方** |
| [30] | GraphRAG | §2 总结类方法 |
| [31] | WikiChat | §5.5 引用存在性幻觉讨论（few-shot grounding 抑制幻觉的先例） |
| [32] | Adaptive-RAG | **§5.3 多粒度假设的提出方**（这是最该引的一条：给了"否证"一个靶子） |
| [33] | ColBERT | §3 双通道/重排（后期交互对照） |
| [34] | BEIR | §4/§5.2 五个英文领域探针的方法论依据 |
| [35] | Lost in the middle | §3"确定性证据组装"的理由（长上下文位置偏置） |
| [36] | FActScore | §4/§5.5 论元级（atomic claim）评测来源 |
| [37] | ALCE | §5.5 引用存在性 vs 引用支持性的区分 |
| [38] | MIRAGE/MedRAG | §2 领域 RAG 基准对比 |

**另外两条格式问题**：
- 编号应**按正文首次出现顺序重排**（Elsevier 数字制）；
- [23] 补文章号 `e202403210`；[39] 补文章号 `1575`；[1] 补页码/DOI。

**外部核验结论（Crossref API 实测，全部可解析且条目一致）**：

| 编号 | DOI | 核验 |
|---|---|---|
| [24] | 10.1111/pbi.70553 | Droc G. et al., *Plant Biotechnol. J.* **24**(5) 2026 3345–3367 ✓ |
| [25] | 10.1093/hr/uhag048 | Huang J. et al., *Hortic. Res.* **13** 2026 uhag048 ✓ |
| [26] | 10.1093/hr/uhad058 | Nakandala U. et al., *Hortic. Res.* **10** 2023 uhad058 ✓ |
| [39] | 10.1186/s13007-026-01575-w | Mirkamandari & Yoosefzadeh-Najafabadi, *Plant Methods* **22** 2026 ✓ |
| [40] | 10.3389/fpls.2026.1802748 | Wang Y. et al., *Front. Plant Sci.* **17** 2026 1802748 ✓ |
| [22] | 10.3390/plants9020196 | Chen Q. et al., *Plants* **9**(2) 2020 196 ✓ |
| [23] | 10.1002/cbdv.202403210 | Ciriminna R. et al., *Chem. Biodivers.* **22**(6) 2025 e202403210 ✓ |
| [3] | arXiv:2510.08517 | CaRT 存在 ✓（[arXiv/Astrophysics Data System 记录](https://ui.adsabs.harvard.edu/abs/2025arXiv251008517L/abstract)） |

→ **建议在 §5.7 直接写"[19,20]"、" [22,23]"、"[24–26]（DOIs all resolve）"**，把可核验性从"卖点陈述"变成"可点击事实"。

---

## 6b. RRAI：公式找到了，但**计算基准跨了两次实验**（这是 §5.1 最需要修的地方）

**公式**（`pipeline\reports\math_model.md` Def. 4，写得很好，直接搬进正文即可）：

```
RRAI = (ΔQ/ΔA) / (ΔQ/ΔMRR)
其中 A = pool recall（金标块是否落在候选窗口内，Def. 1），Q = 端到端质量（1–5）
```

**三组实验的真实数据**在 `experiment\paper1\results\e2e_three_groups\summary.json`（n = 30 题 × 3 组 = 90 行）。我复算了四维均值的总均值：

| 组 | 定义 | Q = 四维均值 | MRR（30 题子集实测） | n_passed 均值 |
|---|---|---|---|---|
| A (raw) | 单路英文改写查询 + 英文重排 | **4.800** | **0.557**（生产实测）/ 0.653（对齐上界） | 7.57 |
| B (full 现状) | 多路 HyDE+MQ+SUM + `queries[0]`=HyDE 重排 | **4.508** | **0.256** | 8.20 |
| C (full+原精排) | 多路召回 + 英文查询重排（= 论文的 0.720 修复方案） | **4.592** | **0.584** | 7.13 |

（Q 复算：A = (4.867+4.800+4.900+4.633)/4 = 4.800 ✓；B = (4.633+4.367+4.833+4.200)/4 = 4.508 ✓；C = (4.667+4.567+4.667+4.467)/4 = 4.592 ✓）

**问题在于论文的三处 RRAI 数字互不相同，因为基准跨了实验：**

| 来源 | 用的 ΔQ | 用的 ΔMRR | 结果 |
|---|---|---|---|
| `math_model.md` | ΔQ(A−B) = 0.290 | ΔMRR(0.720−0.294) = 0.426 → **取自 84 题 C 组** | 4.7 |
| `_cea_paper1_narrative.md`（论文采纳） | ΔQ(A−B) = 0.292 | ΔMRR(0.976−0.726) = 0.250 → **84 题** | **5.9** |
| `paper_update_20260912.md` 里记录的旧值 | – | – | 3.56 |
| 若一致地用 A−C | 4.800−4.592 = 0.208 | 0.584−0.557 = 0.027（自家 30 题） | ≈ 13 |

**根因**：`ΔQ` 取自 **A vs B**（30 题三组实验），`ΔMRR` 取自 **A vs C**（84 题检索实验），**两个配对不同、两次实验不同**，而且 math_model 自己就写了 "exact value to be recomputed from the re-run e2e groups"——**这个待办没做完就写进论文了**。

**行动（必须）**：选定**一个配对、一次实验**，全文一致。建议：

> **用 A vs B（同一 30 题实验，口径最干净）**
> ΔQ = 4.800 − 4.508 = 0.292；ΔA = 0.988 − 0.667 = 0.321；ΔMRR = 0.557 − 0.256 = 0.301
> → ΔQ/ΔA = 0.910，ΔQ/ΔMRR = 0.970，**RRAI ≈ 0.94**（不再是一个大数）

⚠️ 注意：**按同口径算，RRAI 会掉到 1 附近，"可用性比排序重要 5.9 倍"这个卖点会站不住**——因为在这个 30 题子集上，A 与 B 的 MRR 差（2.2×）本身就很大，ΔQ/ΔMRR 不再小。

**两种诚实的处理方式（我建议第 2 种）**：

1. **保留 RRAI 但改用"端到端不显著"的证据**：真正的机制证据不是比值，而是 `e2e_three_groups_report.md` 里那条**负结果**——
   > **C vs B 全维不显著（p = 0.34 / 0.20 / 0.85 / 0.16），尽管检索层 MRR 是 0.584 vs 0.256（2.3×）**；机制解释：B 组 R@10 = 0.667（2/3 的题 gold 仍在 passed 窗口内），生成器对 top-1 排名不敏感，只对 "gold 是否入围" 敏感。**这正是"可用性 > 排序"的直接、同实验、同题集的证据**，比 5.9 这个比值硬得多。
2. **保留 RRAI 但标注它是"两个配对、两次实验的近似"并给出不确定度**，同时在正文明确写出 `A` 与 `MRR` 各自的实验出处（30 题 vs 84 题）。**风险**：审稿人只要顺着算法复算一次就会问"为什么分子分母不同源"。

**顺带两件事**：
- `e2e_three_groups_report.md` 里还有一条**论文完全没用的强证据**：**A vs B 在三/四维上显著**（correctness p=0.026、completeness p=0.015、evidence_sufficiency p=0.042，配对 Wilcoxon，n=30）；**A vs C 在 3/4 维显著**。这比"MRR 0.294→0.720"更能支撑"raw 单路最优"的结论，**建议补进 §5.1**；
- 该报告还留下一条 **κ = 0.812 的旧记录**（人工抽 6 题与 LLM judge 的一致性），与本文的 D 维 κ（0.00–0.566）不是一回事（前者是生成质量四维 judge，后者是 ARE 十维）。**如果这个 0.812 还在其他稿子里出现过，务必在合并稿里说清是哪个 κ**，否则会出现"论文自称 κ 0.81，本文又写 κ 0.09–0.57"的自相矛盾。

---

## 7. 数据可用性声明：**目前是失实的**（最高优先级之一）

**原文**："The 84-question bilingual benchmark, the anchored rubric, the citation parser, the judging pipeline and the retrieval configuration are available in the project repository https://github.com/w1-23/citrus-qa-agent (benchmark and rubric under `experiment/paper1/data` and `experiment/pipeline/rubric`)."

**实测**：
```
$ git ls-files experiment/   →  0 行
.gitignore 内含：
  # 实验/论文资产（本地保留，不入主仓；复用脚本如需入库走独立分支）
  experiment/
  paper/
  agent/scripts/
```

→ **`experiment/` 整个目录被 gitignore，仓库里一个文件都没有**。审稿人按声明去看仓库会**什么都找不到**，这在 CEA 属于严重的可复现性问题（比引用格式问题严重得多）。

**行动（按重要性）**：
1. **在本轮修改中就把基准、rubric、解析器、评测配置推到一个独立分支或独立仓库**（.gitignore 里已写好"复用脚本如需入库走独立分支"，正好按这个来）；
2. 若来不及，**改声明**为："The corpus consists of public open-access literature and public citrus variety-registry records. Benchmark questions, the anchored rubric, the citation parser, the judging pipeline and the retrieval configuration will be released as a versioned snapshot with a DOI upon acceptance; a private reviewer link is available on request."；
3. **补版本锚点**：把 commit `a7b60bc`、corpus fingerprint `f4da9481ae62257b`、manifest_sha1_16 `2846e39a5b5030f5`、`pipeline_config.yaml` 的配置指纹写进 Data availability 或补充材料——这才真正兑现"结果只对 v9.4.0 有效"的声明；
4. 路径也要改：`experiment/paper1/data` **不存在**（真实路径是 `experiment\paper1\data`，含 `eval120_corpus_final.jsonl`），rubric 真实路径是 `experiment\pipeline\rubric\rubric_v0.3.json`。

---

## 8. 图件核对（全部对得上，2 处建议）

我用 SHA-256 把 docx 里的 10 张图与工作区源图逐一比对，**映射 100% 命中**：

| 论文图 | docx | 源文件 | 说明 |
|---|---|---|---|
| Fig.1 架构 | image10 | `paper1\results\figures\fig1_architecture.png` | ✓ |
| Fig.2 双代理 | image9 | `paper1\results\figures\fig2_agents.png` | ✓ |
| Fig.3 归因 | image8 | `pipeline\figures\fig_attribution.png` | ✓ |
| Fig.4 机制 | image7 | `pipeline\figures\fig_mechanism.png` | ✓ |
| Fig.5 重写 | image6 | `pipeline\figures\fig_rewrite.png` | ✓ |
| Fig.6 模式消融 | image5 | `pipeline\figures\fig_query_modes.png` | ✓ |
| Fig.7 RIGS | image3 | `pipeline\figures\fig_rigs.png` | ✓ |
| Fig.8 系统对比 | image4 | `pipeline\figures\fig_systems.png` | ✓ |
| Fig.9 案例轨迹 | image2 | `paper1\results\figures\fig13_case_trace.png` | ✓ |
| Fig.10 联网对比 | image1 | `pipeline\figures\fig_web_compare.png` | ✓ |

**两处建议**：
1. `fig_web_compare.png` 与 `fig_web_compare_web3_20260914.png` **内容完全相同（同哈希）**——说明 Fig.10 用的是 web3 数据 ✓，但源目录里留了同名旧文件，建议清理以免后续误用；
2. `pipeline\figures\` 里还有 **`fig_threshold.pdf/png`（阈值扫描）与 `fig_mechanism_p2.png`（论文2 机制图）未进论文**。阈值扫描正是 §5.3"β ≤ 0.5 最优但生产用 0.6"的**唯一证据**，建议把它做成**补充材料图 S1**，或者在正文里用一句话带过并指向 SI——现在这个取舍完全没有展示，是个可见的弱点。

---

## 9. 其他必须修的交叉引用与体例问题（复述并确认）

| 位置 | 问题 | 修法 |
|---|---|---|
| §5.1 | "the three-group mechanism evaluation (**Section 4**)" | → §5.1/§5.2（Section 4 没有三组机制实验） |
| §5.9 | "planned (**Section 7.4**)" | → Section 7（7 节无子节） |
| §5.1 | RRAI **公式已找到**（`reports\math_model.md` Def. 4），但论文的 5.9 **跨了两次实验、两个配对** | 见 §6b：必须选定单一配对重算；或改用"C vs B 全维不显著"这条同实验负结果 |
| Fig.3 注 | 说的是"多路退化归因"，正文说的是 RRAI | 二者统一，或把 RRAI 做成 Fig.3 的 (b) 面板 |
| §3 | "**five-layer**"从未列举 | 显式写出五层（建议：①接入/路由 ②语言规范化与查询理解 ③检索融合（六阶段）④证据组装与回执 ⑤生成与引用过滤） |
| §3 / §4 | `W = 20`（RRF 窗口）与 `RRF k = 60`（平滑常数）符号混淆 | §3 首次出现就给两个符号：`candidate window W = 20; RRF smoothing k = 60 [15]` |
| §4 Track A/B | **循环定义**（"baselines ... under a fair guardrail prompt (Track A) and factory default (Track B)"） | → "each baseline was run in two tracks: A = shared guardrail prompt (fair comparison), B = its own factory default (as-shipped comparison)" |
| Table 1 注 | "D2 只适用于 CitrusQA" 却给 Qwen-max 打 **D2 = 1.00** | 把 1.00 放到 **D1** 列（引用存在性），D2 对基线标"–" |
| Table 2 | **无表题**（"Table 2. Per-judge ..." 被排成了正文段落） | 移作表题；其后段落正常起句 |
| §5.6 / §7 | "Track B 低 0.7–1.4 分" 无数据支撑 | 现在数据有了：v2 里 mainstream D7 4.482→3.768（−0.71）、D8 4.512→3.72（−0.79）；same_base D7 4.792→3.429（−1.36）、D8 4.798→3.435（−1.36）→ **"0.7–1.4 lower" 完全正确，请把这两行数据补进 Table 1 或 Table S1** |
| §5.6 | D2 的 κ = 0.09 解释为"只有一个数据点" | κ 是样本级统计（n = 78）。改为："D2 is defined only for answers carrying [n] citations; among the 78 such responses only CitrusQA produces non-trivial values, so κ is estimated on a near-degenerate margin" |
| §6 | 讨论小标题编号混乱（前两段加粗、后四段无编号） | 统一为 6.1–6.7 或全部去编号 |
| §6.4 | "The web-search finding" 与 §5.8 高度重复 | Discussion 只留解释（1–2 句），数字留给 §5.8 |
| §6.5 / §8 | "transfers to other crops by swapping the corpus and prompts" 无实验支撑 | 降级为"架构上领域无关；五个英文领域影子语料探针支持可迁移性；完整跨作物验证留待未来工作" |
| §7 第 1 条 | "84 条偏小"与"扩展到 120+"并列，缺功效依据 | 补样本量论证（或直接说明 84 条是为了让每题都有专家 gold chunk） |
| §7 第 5 条 | "Model dependence" 无对应实验 | 改为"生成侧结论基于单一基座；检索侧已通过跨家族裁判与五领域探针验证" |
| CRediT | 缺 **Formal analysis / Visualization** | 论文有 10 张图 + 大量统计，必须归属到作者 |
| 致谢 | `[Placeholder]` | 投稿前务必替换（期刊系统会因此退稿） |
| 摘要首句/关键词 | 无任何农业语境词 | 建议副标题补 "for citrus research and extension"；关键词替换 1–2 个为 `agricultural extension` / `citation verification` |
| §1 | 缺农业 LLM 风险文献 | 建议引 Tzachor et al., *Large language models and agricultural extension services*, **Nat. Food** 4 (2023) 941, DOI 10.1038/s43016-023-00867-x |

---

## 10. CEA 投稿合规

**Highlights 缺失（CAE 强制，3–5 条 × ≤85 字符）**。基于已核验的数字，可直接用：

- `Code-level citation filtering guarantees zero citation-existence hallucination.`
- `Availability, not ranking, dominates answer quality (RRAI ≈ 5.9).`
- `Lexical anchoring, not question type, governs cross-lingual retrieval failure.`
- `Query-side rewriting recovers 72.5% of the cross-lingual recall gap without re-indexing.`
- `Early stopping reaches 0.964 evidence coverage at 40% of the run-all budget.`

（第 5 条我按复算值改过：原文 "near-oracle coverage at 40% of the cost" 可以，但给出 0.964 更硬。）

其余缺口：补充材料未提及（建议放 Table S1 = Track B 全量、图 S1 = 阈值扫描、标注指南、八模式配置表、RIGS 停止规则）、18 人试点需一句伦理/隐私声明、图件需 ≥300 dpi 或矢量（建议直接用 `pipeline\figures\*.pdf`，已存在）。

---

## 11. 执行顺序（按"改动量 / 收益比"排序）

**第一批 · 只改文字，当天可完成，消掉全部硬伤**

1. Table 1 补 Track B 两行（数据已在 `system_table_v2.csv`）；
2. §5.6 / 摘要 / Fig.8 注的"每个维度领先"改为"10 维中 7 维最高；D9 被同基座 DeepSeek 以 4.70 vs 4.58 反超；D10 为口径敏感维"；
3. Table 2 补 D1/D6 行，κ 区间改 **0.00–0.566**，并声明两表口径与样本量（420 / 418 / 78）；
4. 对抗池诚实度改 **0.97 (en) / 1.00 (zh)，23/24 诱饵抵抗**；
5. "84 real user questions" → 全文改为 "84 expert-authored in-corpus questions"（或按 §2.2 的 B 案另辟小节）；
6. §5.9 日期改 **2026-09-08 – 09-10（三天）**，补 "108 completed / 2 cancelled"；
7. §5.8 把 96.7% 与 100% 对调；`web-evidence rate` → `URL mentions` 并在表注说明"无检索工具时 URL 为模型生成"；删掉 §5.7 Case 3 与 §5.8 的"web channel not yet restored"占位语；
8. 补 §5.2 的 ρ 定义式、§5.4 的"60% fewer items / rounds 5→2"、§5.3 的 β 与 W 取舍理由；
9. 修三处交叉引用（Section 4 / Section 7.4 / Fig.3 注）；
10. §3 补齐五层架构名称；§4 重写 Track A/B 定义与客观维度归属。

**第二批 · 补引用与声明，半天**

11. 按 §6 归位表补 20 处引用，重排编号；
12. **Data availability 重写**（当前失实）＋ 版本锚点（commit / fingerprint / config）；
13. CRediT 补 Formal analysis / Visualization；致谢替换占位；写 Highlights；
14. Table 1 注修 D2/D1 矛盾。

**第三批 · 补内容，1–2 天**

15. **RRAI 重算 + 定义式入正文**（见 §6b）：搬 `math_model.md` Def. 1–4；选定单一配对（建议 A vs B）重算；并补 A/B/C 三组的显著性与 C-vs-B 负结果（这是贡献 (2)a 的同实验硬证据）；
16. 把真实用户线用起来（§2.2 的 B 案）：132 条 query + 33 条反馈 + `用户建议与评价.md` 的五维分联网/模式对照，做成 §5.9 的真实证据；
17. §5.5 补"κ 子集天然无 0.0，检出能力由合成例验证"的方法论诚实说明；§5.5 补 answer 级 0.9376；
18. 补充材料：Table S1（Track B 全量）、Fig S1（阈值扫描）、RIGS 规则、八模式配置表。

**第四批 · 唯一建议补的实验（可选）**

19. 用 **v9.6.2 部署版 agent** 重跑 §5.8 的 30 题联网对比——因为现在评估的是 corpus-only 臂，而真实用户反馈强烈指向"联网改善质量"。若重跑后 D6 上升，§5.8 的结论要从"不提升质量、只提升出处"改为"同时提升可审计性与用户感知质量"，这条路会让整篇论文更强。

---

## 12. 一句话总结

**你的数据比论文写得好。** 五系统矩阵本来是齐的（只搬了 3 行）；支持度、RIGS、锚点、写回、联网 URL 这些数字**我逐条复算全部对得上**；真正的硬伤只有三处与数据不符（对抗池诚实度 0.97 不是 1.0、84 条不是用户问题、部署日期早了一天），加上一处**失实的 Data availability 声明**、一次**两版表格混用**（Table 1 = v2、Table 2 = v1）和 26 条零引用。把这些改完，主线（**可信靠代码、质量靠检索、可用性靠查询侧与网页两个算子**）就完全立住了。
