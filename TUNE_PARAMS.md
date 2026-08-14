# Citrus QA Agent — 可优化参数清单

> 所有参数按模块分类，标注当前值、位置、影响范围、优化建议。
> ✅ = 已实现可配置 | ⚠️ = 硬编码待配置化 | ❌ = 未实现

---

## 1. 检索（Retrieval）

| 参数 | 当前值 | 位置 | 影响 | 优化建议 |
|------|--------|------|------|---------|
| `top_k_vector` | 40 | `config.yaml:5` | Qdrant 向量检索每批取回数 | 提高可增加召回覆盖，但增内存和 Reranker 耗时 |
| `top_k_bm25` | 40 | `config.yaml:6` | BM25 词法检索取回数 | 同上 |
| `top_k_final` | 10 | `config.yaml:7` | Rerank 后最终保留数 | 直接影响 LLM 看到多少文献 |
| `rrf_k` | 60 | `config.yaml:8` | RRF 融合常数 | 越大向量和 BM25 权重越接近 |
| `rrf_weights.orig_dense` ✅ | 1.0 | `config.yaml:9` | original_query 向量检索权重 | HyDE 幻觉时降到 0.6~0.8 |
| `rrf_weights.hyde_dense` ✅ | 1.0 | `config.yaml:10` | HyDE 假想答案向量检索权重 | 效果不好时降到 0.7 |
| `rrf_weights.bm25` ✅ | 1.0 | `config.yaml:11` | BM25 词法检索权重 | 名词多时提到 1.2 |
| `rerank_threshold` | 0.25 | `config.yaml:13` | Sigmoid 概率底线 | HyDE 后可能需重新校准 |
| `dynamic_threshold_ratio` | 0.60 | `config.yaml:14` | 动态阈值 = top_score × ratio | HyDE 后分数分布可能右移 |
| `rag_hyde_enabled` ✅ | true | `config.yaml:16` | HyDE 开关 | 关闭时完全回退原检索 |
| BM25 `k1` | 1.5 | `bm25.py:11` | 词频饱和度 | 长文档霸榜时降到 1.2 |
| BM25 `b` | 0.75 | `bm25.py:11` | 文档长度归一化 | 长文档霸榜时提到 0.85 |
| BM25 `delta` | 1.0 | `bm25.py:11` | BM25+ 平滑因子 | 提高 = 低频词更多曝光 |
| HyDE `temperature` | 0.2 | `search.py` | 假想答案创造性 | 0.2 是甜点值（兼顾事实性和多样性） |
| HyDE `max_tokens` | 300 | `search.py` | 假想答案长度 | 影响 embedding 质量 |
| HyDE `timeout` | 3s | `search.py` | HyDE 超时 | 超时自动回退原检索 |
| HyDE LRU Cache ✅ | 500 条 | `search.py` | MD5 key 去重缓存 | 避免同一 query 重复生成，可调 maxsize |
| `academic_sources` | ["crossref"] | `config.yaml:21` | 学术 API 源 | P3 加 semantic_scholar(自带 Abstract) |

---

## 2. HyDE 三路权重（已实现，`config.yaml` 可调）

| 路 | 默认权重 | 调参预判 |
|----|:---:|------|
| orig_dense | 1.0 | 锚定向量，保持主力 |
| hyde_dense | 1.0 | HyDE 幻觉 → 降到 0.6~0.8 |
| bm25 | 1.0 | 柑橘专有名词多 → 提到 1.2 |

**SOP**：上线后先保持等权 1.0 跑一周，收集日志 → 拉 50 条真实 Query 人工标注 → 根据"好文献是否被挤掉"决定调整方向。

---

## 3. 模型 & 推理

| 参数 | 当前值 | 位置 | 优化建议 |
|------|--------|------|---------|
| `main_model` | deepseek-v4-flash | `.env` | v8.3.1: 已全切 Flash 正式版（V4 Pro 移除）；可降级 deepseek-chat 降成本 |
| `fast_model` | deepseek-v4-flash | `.env` | HyDE/hints 用, 可换更便宜模型 |
| `embedding_model` | multilingual-e5-large | `config.yaml:27` | 可换 BGE-M3(更好多语言) |
| `reranker_model` | bge-reranker-v2-m3 | `config.yaml:28` | 可换更小更快的 minicpm |
| `temperature_main` | 0.2 | `config.yaml:56` | 可提至 0.3 让论述更流畅 |
| `temperature_fast` | 0.0 | `config.yaml:58` | HyDE 建议 0.2（0.0 太死） |
| `max_tokens` | 4096 | `config.yaml:61` | 深度综述可提至 8192 |
| ONNX provider | DML>CUDA>CPU | `config.yaml:35-38` | 换设备可加速 |

---

## 4. Graph & Agent 轮次

| 参数 | 当前值 | 位置 | 优化建议 |
|------|--------|------|---------|
| `SUPERVISOR_MAX_TURNS` | 8 | `config.yaml supervisor.max_turns` | v8.3.3: 已接线配置（此前硬编码 4）；综述类长任务建议 ≥6 |
| `LIGHT_MAX_TURNS` | 2 | `config.yaml light.max_turns` | v8.3.3: 已接线配置 |
| `retrieve-agent max_turns` | 3 | `config.yaml subagents.retrieve-agent.max_turns` | v8.3.3: 已接线配置；v8.3.1: 1→3 支持多轮迭代换关键词；结果足够时 LLM 提前收尾，实际通常 1-2 轮 |
| `write-agent max_turns` | 6 | `config.yaml subagents.write-agent.max_turns` | 通常 1-2 轮够了（综述走 Plan-Execute 流水线，不经 ReAct 多轮） |
| `analyze-agent max_turns` | 2 | `config.yaml subagents.analyze-agent.max_turns` | 分析 Agent |
| `recursion_limit` | 25 | `config.yaml:109` | v8.3.3: 已接线 astream（此前未传参） |

---

## 5. 上下文预算

| 参数 | 当前值 | 位置 | 说明 |
|------|--------|------|------|
| `context_max_tokens` | 1,000,000 | `config.yaml` | v8.4.4 发送视图预算=模型窗口（DeepSeek V4 Flash 1M） |
| `context_soft_threshold` | 0.75 | `config.yaml` | 触发批量压缩（一次压到 ~50%，用户轮边界） |
| `context_hard_threshold` | 0.93 | `config.yaml` | 硬截断兜底 |
| `target_ratio` | 0.50 | `config.yaml` | 批量压缩目标（缓存只破坏一次） |
| `protect_recent_turns` | 3 | `config.yaml` | 保护名单：最近 N 轮 Q/A 不压缩 |
| `keep_recent_turns` | 2 | `config.yaml` | 截断时保留 N 轮 |
| `compact_max_tokens` | 800 | `config.yaml` | 压缩输出上限（已真正传入 LLM） |

---

## 6. 并发 & 分组

| 参数 | 当前值 | 位置 |
|------|--------|------|
| Qdrant `max_workers` | batches × queries | `multi_retriever.py` |
| HyDE 双路并发 ✅ | 2 × batches | `multi_retriever.py:283-293` |
| `concurrency_key` 分组 | rag/web/file/analysis | `config.yaml` |

---

## 7. 文件 I/O

| 参数 | 当前值 | 位置 | 说明 |
|------|--------|------|------|
| `PDF_DEFAULT_MAX_CHARS` ✅ | 30000 | `readfile.py:20` | 已从 10000 提升，约覆盖 15 页论文的 60% |
| `FILE_READ_MAX_SIZE_MB` | 50 | `config.yaml:151` | 单文件大小上限 |
| `FILE_WRITE_MAX_SIZE_MB` | 10 | `config.yaml:152` | 写入大小上限 |

---

## 8. SSE / 心跳 ⚠️ 硬编码待配置化

| 参数 | 当前值 | 位置 | 可配置化 |
|------|--------|------|:---:|
| `SSE_HEARTBEAT_INTERVAL` | 5s | `main.py:185` | ✅ |
| `TOOL_HEARTBEAT_INTERVAL` | 2s | `main.py:193` | ✅ |
| `BRIDGE_QUEUE_TIMEOUT` | 0.3s | `main.py:172` | ✅ |
| `TOOL_RESULT_TRUNCATE` | 100000 | `expert_graph.py:477` | ✅ |
| `SSE_DRAIN_MAX_LOOPS` | 10 | `main.py:270` | ⚠️ |

---

## 9. 缓存（已实现 / 待实现）

| 参数 | 当前值 | 位置 | 状态 |
|------|--------|------|:---:|
| HyDE LRU Cache | 500 条, MD5 key | `search.py` | ✅ 已实现 |
| Embedding Cache | 未设 | — | ❌ 待实现 |
| Reranker Cache | 未设 | — | ❌ 待实现 |

---

## 优化优先级

| 优先级 | 参数 | 原因 |
|:---:|------|------|
| **P0** | `rrf_weights` 重新校准 + `rerank_threshold` 观察 | HyDE 后必须调，但第一周先保持默认收集日志 |
| **P1** | HyDE `temperature` 0.0→0.2 + `top_k_final` | 直接影响质量 |
| **P2** | BM25 `k1/b` + `PDF_DEFAULT_MAX_CHARS` | 体验微调 |
| **P3** | Embedding/Reranker 缓存 + 模型升级 | 长期降本增效 |

---

## HyDE 上线调参 SOP

| 天 | 行动 |
|----|------|
| Day 1-2 | 保持默认（等权 1.0, 阈值 0.25），收集日志 |
| Day 3 | 拉 50 条 Query 的 Reranker 输入输出日志，人工标注 |
| Day 4 | 检查 PDF 读取是否被截断，决定是否继续调高 `PDF_DEFAULT_MAX_CHARS` |
| Day 5 | 观察 HyDE 缓存命中率，决定是否加大 cache size |
| Week 2 | 根据人工标注结果调整 `rrf_weights` 降 hyde 权重或提 bm25 权重 |
