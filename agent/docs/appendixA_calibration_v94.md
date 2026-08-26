# 附录 A 锚点行号校准（v9.4 实测，2026-08-25）

> 用途：P5 论文修改直接引用。行号以当前工作区最新代码为准。
> 更新日志：2026-08-25 —— 全量校准（config/代码/实验资产三侧）。

## 1. config.yaml（检索参数终值，A 类三实验确认默认最优）

| 参数 | config.yaml 行号 | 值 | 实验确认 |
|---|---|---|---|
| top_k_final | :8 | 10 | 4a：K 零影响，10 最优成本 |
| candidate_window | :9 | 20 | 4a：W=20 MRR 最高（0.6535） |
| rrf_k | :10 | 60 | 历史校准（doc13） |
| rerank_threshold | :15 | 0.25 | 7：floor 0.20-0.30 零影响 |
| dynamic_threshold_ratio | :16 | 0.60 | 7：β=0.6 pass 平台右边界 |
| early_stop_min_evidence | :19 | 6 | 4b：(6, 0.25) 平台中段 |
| early_stop_new_ratio | :20 | 0.25 | 4b：-21% 轮次零损失 |
| query_mode | :32 | **raw** | v9.4b 裁决①：生产默认改 raw——exp1/1b/1c/2 四重互证 full 0.256 劣于 raw 0.557（生产实测）/0.653（对齐上界）；端到端实验九另证（详见 §7.12） |

## 2. src/config.py（字段定义）

| 字段 | 行号 | 默认 |
|---|---|---|
| TOP_K_FINAL | :75 | 10 |
| CANDIDATE_WINDOW | :78 | 20 |
| RERANK_THRESHOLD | :80 | 0.25 |
| DYNAMIC_THRESHOLD_RATIO | :82 | 0.60 |
| EARLY_STOP_MIN_EVIDENCE | :84 | 6 |
| EARLY_STOP_NEW_RATIO | :85 | 0.25 |
| QUERY_MODE | :92 | raw |

## 3. src/core/agent_runner.py（早停逻辑）

- :586-601 早停主逻辑（`_uniq_now` 唯一证据计数 → `_new_ratio` 新增占比
  → `prev_unique >= 6 and _new_ratio < 0.25` → break）
- :594 注释：由 retrieval.early_stop_* 控制
- :394 `_prev_unique` 初始化

## 4. src/retrieval/multi_retriever.py（检索链路）

- :580-669 `_fuse_rerank_select`（RRF→候选→rerank→动态阈值→日志）
- :598 `rrf_fuse`（k=settings.RRF_K）
- :601-602 候选窗口 `fused[:settings.CANDIDATE_WINDOW]`（v9.4 参数化）
- :605-606 rerank（rerank_query → **B 类发现主因锚点；v9.4b 已防御**）
- :616-618 动态阈值 `max(RERANK_THRESHOLD, top*DYNAMIC_THRESHOLD_RATIO)`
- :671-734 `search_multi`（多路并发；:733 `rerank_query=rerank_query or
  original_query or queries[0]`——v9.4b 防御：防未来误开 full 时重蹈 HyDE 段
  精排损失；显式传参优先，其次用户查询原文，最后 queries[0]）
- :736-738 `search` = search_multi([query])（raw 与基准同构）

## 5. src/tools/search.py（查询组装）

- :495-510 `_HYDE_PROMPT`（HyDE 生成系统提示）
- :520-554 `parse_hyde_structured`（解析）
- :556-624 `_generate_hyde_structured`（生成器，用 settings.RESOLVED_FAST_*）
- :637-647 `_cached_hyde_parsed`
- :652-654 `_VALID_QUERY_MODES`（8 模式，含 mq_sum）
- :656-689 `_compose_queries`（模式→查询列表纯函数；:676-677 mq_sum 分支）
- :717-423 工具主路由（hyde_parsed → search_multi(queries) 否则 search(query)）

## 6. 实验资产（论文图表数据源）

| 资产 | 路径 | 用途 |
|---|---|---|
| 120 题全量 | experiment/data/eval120.jsonl（终版，含 s5_pair） | §7.2 评测集 |
| S5 双语等价 60 对 | experiment/data/s5_bilingual_pairs.jsonl | §7.6 语言对比 |
| HyDE 缓存 | experiment/results/hyde_cache_84.jsonl | §7.4/7.5 复现 |
| 4a 扫描 | results/exp4a_scan.json + fig_data CSV | §7.7/8.3 |
| 4b 端到端 | results/exp4b_e2e.json | §7.7/8.3 |
| 7 阈值网格 | results/exp7_threshold_grid.json | §7.10 |
| 1/2/1b/1c 消融 | results/exp1_ablation.json, exp2_query_mode.json, exp1b_rerank_attribution.json, exp1c_pool_orig.json | §7.4/7.5 |
| B3 时效 | results/b3_temporal_judge.json | 实验 8 |
| #1 生产三对照 | results/production_raw_trio.json/.csv + rewrite_cache_cor84.jsonl | §7.12/8.3 生产实测 0.557 |
| #2 端到端三组 | results/e2e_three_groups/（90 轨迹）+ summary.json | §7.12 |
| 图 1-8 | results/figures/*.{pdf,png} | 全文 |
| 图数据仓库 | results/fig_data/*.csv（含 exp1bc_attribution.csv，只读） | 可复现 |

## 7. 裁决收口（2026-08-26，全部完成）

- config.yaml :32 query_mode = **raw**（裁决①，已回写并提交）
- multi_retriever.py :733 防御（rerank_query=rerank_query or original_query or queries[0]）
- §7.4/7.5 叙事按①落笔；§7.12 新增（端到端三组 + 配对 Wilcoxon）
- §8.3 "默认最优"→"生产默认修正为 raw 的实证回写"（MRR 0.256→0.557，+117%）