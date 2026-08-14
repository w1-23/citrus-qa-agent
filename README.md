# Citrus QA Agent v8.3.3

柑橘科研问答 RAG + Multi-Agent 系统：Light/Expert 双图、HyDE+RRF 混合检索、SSE 流式输出、上下文预算与长期记忆。

> v8.3.3（2026-08-12）修复综述写作首跑必崩（UnboundLocalError）、补齐 Plan-Execute 完整性（写后校验/引用校验/落盘回退）并完成安全与健壮性加固（沙箱 fail-closed、路径白名单、XSS 转义、per-request 事件队列、request_id 追踪）。历史审计与修复记录见 `AUDIT_2026-08-10.md`。

---

## 快速开始（运行流程）

```bash
# 1. 依赖安装（已补齐 langgraph/optimum/transformers/openai）
cd agent
pip install -r requirements.txt

# 2. 配置 API Key（.env.example 复制为 .env 后填写 DEEPSEEK_API_KEY）
copy .env.example .env        # Windows
# cp .env.example .env        # Linux/macOS

# 3. 模型缓存（首次运行自动下载）
#    大陆网络用清华镜像预下载 embedder：
set HF_ENDPOINT=https://hf-mirror.com
python -c "from fastembed import TextEmbedding; TextEmbedding('intfloat/multilingual-e5-large')"
#    reranker 首次启动自动导出到 .hf_cache/onnx_reranker（也可整目录拷贝）

# 4. 启动服务
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000

# 5. 访问
#    前端页面: http://localhost:8000/
#    健康检查: http://localhost:8000/health
```

**启动自检**（可选）：`python tests/start_check.py` — 后台起服务、轮询 /health、自动关闭。

---

## 最新执行流程

```
用户请求 POST /api/v2/chat
    │
    ├─ ① FastGuard: 问候语直接回复（不调 LLM/RAG）
    ├─ ② 会话: get_or_create_session（"new"/空 → 新 UUID；修复 AG-4）
    ├─ ③ 模式: 完全由客户端 light_mode 决定（用户手动切换，
    │    无服务端路由自动升级——AG-12 已按用户决策彻底移除）
    │
    ├─ ④ 图执行（expert_graph / light_graph）
    │    ├─ load: 历史加载 → ContextBudget.check（修复 AG-3）
    │    │         1M 窗口 / soft 0.60 / hard 0.93 → SUMMARIZE/TRUNCATE
    │    │         压缩结果 replace_history 持久化（修复 AG-6）
    │    │         压缩用 MAIN 模型（质量优先）
    │    ├─ supervisor: ReAct 循环（expert ≤8 轮 / light ≤2 轮，config 可调）
    │    │    ├─ LLM 调用 3 次重试（修复 AG-7）
    │    │    ├─ 工具执行 asyncio.wait_for 60s 超时（修复 AG-7）
    │    │    ├─ 工具失败 → [ERR_*]/[AgentError] 回传 LLM 可感知（修复 AG-14）
    │    │    └─ 职责矩阵（v8.3.1）：保存现成内容 → supervisor 直写 write_local_file；
    │    │       撰写/创作新内容 → call_write_agent；无任何第三方兜底写入
    │    └─ save: 会话落库 + LTM 事实提取（to_thread 异步，修复 AG-5）
    │
    └─ ⑤ SSE 流式返回（thinking/tool_*/text/citations/done）
```

**性能预算（v8.3.1，防空转/防截断）**：
- `academic_search` citrus 过滤全丢时 **min_keep=3** 保留并标注「非柑橘」，打断"0 结果→换词→再 0 结果"空转循环
- write-agent 单轮输出上限 **12000 tokens**（约 1-2 章节）+ prompt 强制分块续写（禁止单轮生成全文），避免 32768 硬截断浪费
- `write_local_file` 返回**内容预览**（前 200 字符），帮助 LLM 判断已写内容、防分块重写
- 检索由 LLM 自主决定次数（无硬阈值），靠 min_keep + 原因回传自然收敛

**统一回传协议（v8.3.1，状态+原因+建议）**：
- 所有工具空结果/失败时附**归因 + 建议策略**，让 LLM 对症下药而非盲目重试：
  - RAG 空结果 → `THRESHOLD_BLOCKED`（候选被阈值拦，建议换特异词）/ `NO_MATCH`（本地库无收录，建议换源或模型知识）
  - academic 空结果 → `[ERR_NETWORK]`（网络失败，建议依赖本地 RAG）/ 无匹配（建议换关键词）
  - 通用错误 → `[ERR_类别]` 每条含操作建议（`_classify_error`，如 FILE_NOT_FOUND→检查路径、TIMEOUT→改用本地源）
- Agent 输出要求强化：write/analyze 必须说明结果归因与信息缺口（哪些有文献支持、哪些方向未检索到）

**Write Pipeline（v8.3.2，Plan-Execute 长文写作）**：
- 写任务**四路路由**（supervisor 层 `classify_write_task`）：`plan_execute`（综述/报告）/ `react`（渐进式写作）/ `direct_write`（保存现成内容）/ `modify`（定向修改已有文档章节）
- **Plan 阶段**：结构化大纲（title/摘要/关键词/章节/每章 refs 文献分配），动态字数阈值校验（严格/自主双模式）+ 单章容量上限（≤3000 字防截断）+ 重试 + ReAct 回退（带大纲落盘）
- **Execute 阶段**：每章**独立 LLM 调用**（输出预算 100% 聚焦一章，无截断）+ 章间 `running_context`（前章 `<summary>` 标签，代码提取）+ 材料按 refs 语义分配 + **运行时容量兜底**（实际输出超限 → 精简重试 + truncated 标记）
- 单章失败 → 缺章占位 + 部分成功返回（优雅降级）；**断点续传**（pipeline_tasks 表，WAL+原子事务，中断后从下一章继续）
- **写后校验（v8.3.3）**：read-back 确认章节落盘（防静默写失败）+ 引用完整性校验（正文 [n] vs 参考文献列表）
- SSE 进度事件：`plan_ready`（大纲）/ `section_start` / `section_done`（前端实时章节进度）

### 检索管线（v8.3.1 关键修复）

```
citrus_rag_search(query)
    ├─ HyDE: FAST 生成英文假想答案（3s 超时 → 平滑回退基础检索）
    ├─ Dense: orig + hyde 双向量 → Qdrant ×5 批次（线程池并发）
    ├─ BM25: 全局索引
    ├─ RRF 加权融合 → Reranker（bge-reranker-v2-m3 ONNX）
    └─ 阈值过滤 → top_k_final
```

**⚠️ 重要（修复 AG-11）**：Qdrant 命中点通过 **payload 的 `(paper_id, chunk_index)`** 映射回语料（point id 为随机整数、chunk_index 为论文内编号）。旧实现按 `(batch, chunk_index)` 映射导致检索系统性错乱，已修复并加启动自检（`idx_map ok` 日志，匹配率 <95% 告警）。

---

## 关键配置（config.yaml 片段）

```yaml
retrieval:
  top_k_final: 10
  rerank_threshold: 0.25      # 未校准（AG-13 按用户要求跳过，改前先看 logs/retrieval 分数分布）
  rag_hyde_enabled: true

context_budget:               # v8.3.1：配置单向化，图内不再硬编码
  enabled: true
  max_tokens: 1000000         # 对齐 DeepSeek V4 Flash 1M 窗口
  soft_threshold: 0.60        # 早期压缩：~60% 窗口触发摘要（长会话控延迟/成本）
  hard_threshold: 0.93        # 硬截断兜底

agent:
  timeout_sec: 30             # 子Agent LLM 超时
  tool_exec_timeout_sec: 60   # 单工具执行超时（修复 AG-7）
  tool_result_caps:           # 子Agent→supervisor 分档截断（字符）
    write-agent: 1000
    retrieve-agent: 40000
    analyze-agent: 10000
    default: 100000


## 测试与回归（脚本归档于 `agent/tests/`）

| 脚本 | 覆盖 | 运行（agent/ 下） |
|---|---|---|
| `tests/test_batch1.py` | AG-1/2/3/6 修复回归（20 项） | `python tests/test_batch1.py` |
| `tests/test_batch2.py` | AG-4/5/7/8/9/10/11/12/14 修复回归（36 项） | `python tests/test_batch2.py` |
| `tests/test_batch3.py` | AG-15/16 + 预算回归（14 项） | `python tests/test_batch3.py` |
| `tests/test_direct_write.py` | direct-write 双层判定（12 项） | `python tests/test_direct_write.py` |
| `tests/verify_retrieval.py` | AG-11 端到端检索验收（真实数据） | `python tests/verify_retrieval.py` |
| `tests/start_check.py` | 服务启动 + /health + idx_map 自检 | `python tests/start_check.py` |

全部无外部 API 依赖（仅 verify_retrieval 需本地模型缓存），改代码后三套全绿即可回归。

---

## 目录与文档索引

```
agent/
├── config.yaml / .env.example / index.html / pack_deploy.py
├── src/                    # 全部源码（graph/core/tools/retrieval/engine/guardrails/session/api）
├── data/                   # 检索语料（Qdrant 5 批次 119K chunks，不入库）
├── tests/                  # 回归/验收脚本（test_batch1/2/3、test_direct_write、verify_retrieval、start_check）
├── MODEL_ROUTING.md        # 模型路由矩阵（节点→模型→温度→超时）
└── verify_retrieval.py     # （已归档至 tests/）
AUDIT_2026-08-10.md         # 生产审计 + 修复状态（P0→P3 全记录）
DEPLOY.md                   # 新服务器部署指南
readme2.md                  # v8.3.0 历史架构手册（部分信息已被本文件更新取代）
TUNE_PARAMS.md              # 可调参数清单
AGENT_CHANGES.md            # v8.3 代码变更对照
```

---

## 常见问题

| 现象 | 处理 |
|---|---|
| `Could not load model ... e5-large` | 清华镜像预下载（见快速开始步骤 3） |
| 启动日志无 `idx_map ok` | 检查 `data/` 完整性；出现 mismatch 告警见 AUDIT AG-11 |
| 工具卡死无响应 | 已内置 60s 超时返回 `[ERR_TIMEOUT]`（`agent.tool_exec_timeout_sec` 可调） |
| 保存文件内容重复 | AG-2 已修复（write=覆盖 + file_saved 防双写）；确认代码为最新 commit |
| 长会话越来越慢 | 正常现象；soft_threshold 0.60 触发摘要压缩后回落（context_status.compressed 可观察） |
| 端口占用 | 换 `--port` 或先停旧进程（Qdrant 单例，禁止双实例同时运行） |
| 想回滚代码 | `git log` → `git checkout <commit> -- 文件路径` |
