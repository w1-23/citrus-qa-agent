# v9.4 论文实验计划（更新代码后 · 投稿导向）

> 目标期刊：Computers and Electronics in Agriculture (Elsevier, SCI Q1)
> 版本：v9.4（candidate_window / early_stop / query_mode 参数化已合入，231 回归通过）
> 与本计划配套：`agent/docs/paper_execution_plan.md`（总体路线）、`experiment/`（隔离实验区，不入 git）
> 铁律：① 改 `agent/` 仅限"参数化/日志/纯函数"并经回归；② 实验脚本只读复用 agent 公开接口
> （`MultiBatchRetriever/rrf_fuse/Embedder/Reranker`），变体经 `fixed_pipeline` 参数注入，零侵入；
> ③ 任何含中文文件修改一律 write/edit（UTF-8）；④ API key 绝不入结果/日志/git。

---

## 0. 可扫描参数总表（v9.4 就绪状态）

| 参数 | config.yaml 键 | settings 字段 | 默认 | 扫描范围 | 对应实验 |
|---|---|---|---|---|---|
| 候选窗口 | `retrieval.candidate_window` | `CANDIDATE_WINDOW` | 20 | 20/30/40/60 | 4a |
| 精排 top_k | `retrieval.top_k_final` | `TOP_K_FINAL` | 10 | 10/15/20/30 | 4a |
| 早停门槛 | `retrieval.early_stop_min_evidence` | `EARLY_STOP_MIN_EVIDENCE` | 6 | 4/6/8/10 | 4b |
| 早停新增占比 | `retrieval.early_stop_new_ratio` | `EARLY_STOP_NEW_RATIO` | 0.25 | 0.15/0.20/0.25/0.30 | 4b |
| 动态阈值比 | `retrieval.dynamic_threshold_ratio` | `DYNAMIC_THRESHOLD_RATIO` | 0.60 | 0.3/0.5/0.6/0.7/0.8 | 7 |
| 阈值底线 | `retrieval.rerank_threshold` | `RERANK_THRESHOLD` | 0.25 | 0.20/0.25/0.30 | 7 |
| 查询组成 | `retrieval.query_mode` | `QUERY_MODE` | full | raw/hyde_only/mq_only/sum_only/hyde_mq/hyde_sum/full(+mq_sum) | 2 |
| 向量 top_k | `retrieval.top_k_vector` | `TOP_K_VECTOR` | 40 | 40/60/80 | 1 |
| BM25 top_k | `retrieval.top_k_bm25` | `TOP_K_BM25` | 40 | 40/60/80 | 1 |
| RRF k | `retrieval.rrf_k` | `RRF_K` | 60 | 20/60/100 | 1 |

注入方式（两套并行，结果互相印证）：
- **检索级**：`experiment/scripts/retrieval_common.fixed_pipeline(r, q, topk_final, pool_mult, bm25_w, dense_w, rrf_k, ratio)` — 与 `search.py` 检索层逐级同构（embed→vector→BM25→RRF→pool→rerank→动态阈值），参数注入零改 `agent/`。
- **工具级**：`setattr(settings, "QUERY_MODE", ...)` 后调用 `citrus_rag_search`（走 v9.4 全链路，含 HyDE+缓存），验证端到端一致性与路数真实分布。

---

## 1. 评测集构建（目标：120 题 + 30 对跨域探针）

### 1.1 底盘与目标分布

现状底盘：`experiment/data/qa80_v2.jsonl`，85 条（`load_qa80()` 已支持），全部含中英双语（`has_en=85/85`）：
L1 ×30 / L2 ×30 / L3 ×20 / net ×5；负样本 4；evidence 部分回填。

**目标 120 题（论文 §7.2 三类比例：70/20/10）**：

| 类别 | 数量 | 来源 | 期望行为 |
|---|---|---|---|
| 语料库内（有 gold chunk） | 84（70%） | qa80 精选 60 + 新造 24 | 检索到 gold、引用正确 |
| 语料库外（无对应文献） | 24（20%） | 现有负样本 4 + 新造 20（柑橘相关但库内无答案） | 诚实声明缺口、不编造引用 |
| 时效性（需联网） | 12（10%） | qa80 net 5 + 新造 7（2025-2026 政策/事件） | 正确路由联网、`[Wn]` 引用可访问 |

**类型分布（论文 §7.1 需要实体/功能分层）**：语料库内 84 题内部分层：
实体型 40（品种/基因/病害专名）· 功能型 30（机制/关系/作用）· 统计型 8（数值题）· 综述型 6（综合概述）。
**中英等价对 60 对**：从语料库内 84 题中选 60 对（实体 30 + 功能 30），人工翻译保证语义一致（禁用机翻）。
**跨域探针 30 对**（不入 120，独立 §5 实验3 用）：BRCA1×15 + OsNRT2×15，中英等价。

### 1.2 构建流程

1. **语料库内题（84）**：
   - 候选提取：`agent/data/` 下 `business.log` + 用户反馈高频查询（若日志不可用 → 从 `agent/` 代码与语料标题人工构造）；
   - gold chunk 定位：`fixed_pipeline` 检索 + 人工确认 1~3 个 `paper_id:chunk_index`（复用 qa80 evidence 回填法）；
   - 数值题标记 `needs_verify`（现有 6 条 q-050/051/052/053/065/066）。
2. **域外题（24）**：构造柑橘相关但无答案的问题（参照现有 4 个负样本模式：语义贴近但库内无对应文献）。
3. **时效题（12）**：构造需联网的最新政策/事件（HLB 防治政策、新品种审定、产业数据等）。
4. **双语与类型标签**：人工翻译 + 实体/功能标注；schema 校验脚本（字段齐全、gold 存在性、分布断言）。
5. 产物：`experiment/data/eval120.jsonl`（含 `build_info/version=v9.4` 头），gold 定位可复现。

---

## 2. 消融实验矩阵（对应论文 §7.4–7.11）

> 实现注：论文实验1 的 `-HyDE` 变体（仅 MQ+SUM，无 HyDE）需 `query_mode` 新增 **`mq_sum`** 模式
> （当前 7 模式缺此组合）——已列入前置改动（§6.1，3 行 + 单测）。

### 实验1：六级管道消融（§7.4）— 检索级，无需 key
| 变体 | 实现（fixed_pipeline / settings） |
|---|---|
| Full | 默认参数 |
| -HyDE | `query_mode=mq_sum`（前置新增） |
| -BM25 | `bm25_w=0.0` |
| -Dense | 仅 BM25（跳过 vector 通道） |
| -Rerank | 直接用 RRF 分数（跳过 reranker） |
| FixedTop5 | `top_k_final=5` + 阈值放宽 |
| FixedTop10 | `top_k_final=10` + 阈值放宽 |
| SingleQuery | `query_mode=raw`（仅原始查询） |
指标：Recall@5/10、MRR、通过率、各阶段耗时（`fixed_pipeline` 已输出 ms 分解）。

### 实验2：多粒度查询消融（§7.5，长度适配假说验证）— 工具级（走真 HyDE）
| 变体 | query_mode | 路数 |
|---|---|---|
| RawQuery | raw | 1 |
| HyDE-Only | hyde_only | 1 |
| MQ-Only | mq_only | 3 |
| SUM-Only | sum_only | 3~5 |
| HyDE+MQ | hyde_mq | 4 |
| HyDE+SUM | hyde_sum | 4~6 |
| Full（本文） | full | 7~9 |
| HyDE+MQ+SUM×5 | full（sum[:5] 上限） | 9 |
指标：Recall@5/10、MRR、通过率、平均精排分数；**成本对比表**（一次结构化调用 ~1.5s/500 token vs 三次独立 ~4.5s/1500 token）。

### 实验3：中文跨语言偏置（§7.6）— 需要改写（key）
对比组：zh-raw / en-raw / zh→en（改写）/ zh→en+HyDE。
- 复用 `experiment/results/rewrite_cache.jsonl`（80 题 zh→en 改写缓存，脱机）做 zh→en 组；
- `bias_quant`（A2）结论为**基础参照**（功能型 zh 显著塌陷 p=0.037）；
- 指标：Recall@10（实体/功能分层）、通过率、分数分布；
- **诚实修正**：A5 跨域探针显示"zh 未现柑橘主库式全线塌陷 → 偏置是多因素"，论文表述改为
  "柑橘主库内显著、跨域有边界"，不做"必塌"过度宣称（阶段 5 落笔）。

### 实验4：候选窗口×早停联动（§7.7）— 检索级扫描
- 4a：`candidate_window` 20/30/40/60 × `top_k_final` 10/15/20/30；
- 4b（固定 4a 最优）：`early_stop_min_evidence` 4/6/8/10 × `early_stop_new_ratio` 0.15/0.20/0.25/0.30
  — 早停为 `agent_runner` 层逻辑，检索级用 RIGS 回放近似（`a3_rigs_replay.py` 已具备全规则回放基建），
  最优组再经 agent 级冒烟验证（key）；
- 指标：Recall@5/10、首轮通过数（均值/中位）、精排耗时；预期"窗口↑→首轮通过数↑→早停易触发→轮次↓"。

### 实验5：成本-质量帕累托（§7.8）— agent 级，需 key
变体：NoStop（跑满3轮）/ CodeStop（早停≥6且<25%）/ BudgetOnly（仅预算无早停）/ Full（早停+预算+去重）/ Oracle（事后最优）。
- 复用 `a3_rigs_replay` 的 tok/质量口径（run-all q=0.867 tok=10000 vs oracle q=0.867 tok=4283，-57% 成本）作为半代理；
- 产出：帕累托前沿图（X=token，Y=质量分）。

### 实验6：并发延迟（§7.9）— 需联网，需 key
变体：Parallel（asyncio.gather 现状）/ Sequential / LocalOnly / WebOnly。
- 指标：端到端延迟、本地耗时、联网耗时；预期 `Parallel ≈ max(Local, Web) << Local+Web`。

### 实验7：动态阈值消融（§7.10）— 检索级
变体：Dynamic（现状 `max(0.25, top×0.6)`）/ Fixed3 / Fixed5 / Fixed10 / Ratio0.3 / Ratio0.8 / Floor0.20 / Floor0.30。
指标：通过条数分布、相关率。

### 实验8：端到端问答质量与引用可信度（§7.11）— 需 key + 人工
- 30 语料内 + 10 域外 + 5 时效 × {联网开/关}；
- 复用 `b3_judge_e2e`（Judge 三维）框架重跑；20 题/45 题人工 κ 回填；
- 引用核查：`[n]` DOI 真实性、`[Wn]` URL 可达性、缺口声明率。

---

## 3. 参数调优：扫描 → 分布验证 → 回写

1. 按"§0 参数总表"逐参数扫描（先检索级，全量 80-120 题或分层子集 30 题，CPU 可控）；
2. 分布验证：通过率/Recall/分数分布随参数变化曲线（复用 `a6_figs_v2` 出图管线）；
3. **回写规则**：最优参数 → 改 `agent/config.yaml` 对应键（注释 `# v9.4 实验校准: ...`）→
   全量回归（231+）→ 更新汇报给用户确认 → 论文附录A 参数终值同步；
4. 联动提醒（论文已注明）：窗口↑→早停易触发→轮次↓成本↓但精排耗时↑；早停门槛↑→轮次↑质量↑成本↑；
   动态阈值β↑→通过数↓噪声↓但可能漏边；推荐顺序 4a→4b→7→2→1。

---

## 4. 脚本清单（依托 experiment/ 既有资产，增量新建）

复用（纯检索，验后直接可用）：`retrieval_common` / `a1_evidence_check` / `a2_bias_quant` /
`a3_*`（RIGS 家族）/ `a5_crossdomain_probe` / `a6_figs_v2` / `p01_token_profile` / `diag_pipeline` / `smoke_test`。
需对齐重跑（绑定旧流程/key）：`b1_rewrite_2x2`（改写缓存可复用）/ `b3_judge_e2e`。
新建：
1. `build_eval120.py` — 120 题构建 + schema 校验 + 分布断言；
2. `run_granularity.py` — 实验2（query_mode 全模式 × 工具级/检索级双口径）；
3. `run_param_sweep.py` — 实验4a/4b/7 网格扫描（复用 `a6_figs_v2` 的网格骨架）；
4. `run_pipeline_ablation.py` — 实验1；
5. `eval_metrics.py` — Recall@5/10、MRR、通过率、延迟、token 聚合（`end2end_judge`/`rigs_replay` 指标口径对齐）。

---

## 5. 执行顺序（依赖+优先级，总估 3–4 周）

| 序号 | 任务 | 依赖 | 预计 | 产出 |
|---|---|---|---|---|
| P0a | 前置改动：`query_mode` 增 `mq_sum`（search.py `_VALID_QUERY_MODES`/`_compose_queries` + 单测） | §6.1 | 0.5 天 | 实验1 -HyDE 可跑 |
| P0b | `build_eval120.py` + 120 题（84 语料内含 60 中英对 + 24 域外 + 12 时效） | qa80 | 1–1.5 周（人工标注为主） | eval120.jsonl |
| P1 | 实验1/2/7（检索级/工具级，无 key 或缓存） | eval120 | 4–5 天 | CSV+图+结论段 |
| P2 | 实验4a/4b 扫描 + 回写 config + 回归 | P1 | 3–4 天 | 参数终值 |
| P3 | 实验3（改写+偏置，key）与实验5（成本质量，key） | P2 | 3–5 天 | 偏置表+帕累托 |
| P4 | 实验6（并发延迟，联网）与实验8（端到端+引用核查，人工） | P3 | 3–5 天 | 延迟对照+Judge 数据 |
| P5 | 论文 §7 替换实测 + 附录A 校准 + 图入册 | P0-P4 | 2–3 天 | 论文修订稿 |

---

## 6. 前置待办与风险

### 6.1 前置代码改动（唯一识别的缺口）
- **`mq_sum` 模式**：`agent/src/tools/search.py` `_VALID_QUERY_MODES` 增 `"mq_sum"`；
  `_compose_queries` 分支 `queries = list(_mq) + list(_sum)`（无需 HyDE 段）；
  `tests/test_v94_param.py` 增用例（`mq_sum` 无 hyde 时 = MQ+SUM，None-hyde 语义不变）。
  其余参数全部已参数化，无其它前置。

### 6.2 风险
1. 联网链路（实验 5/6/8）依赖 DeepSeek key 与网络可达（沙箱无外网时降级为报告口径）；
2. 120 题标注需人工（gold chunk 确认 + 中英翻译），是工期下界；
3. key 侧改写（实验3）需授权使用 `DEEPSEEK_API_KEY`（bootstrap 已有占位，不落盘）；
4. 论文 §7"预期结论"若与实测不符：以实测为准并诚实改写（尤其实验3 跨域边界、实验2 边际收益递减）。

### 6.3 验收
- 每实验：JSON/CSV 指标 + 图 + 结论段（可直接贴论文 §7）；
- 参数回写后：`tests/` 全量回归 ≥231 通过；
- 论文附录A 参数终值/锚点行号与 v9.4 代码逐项对应。