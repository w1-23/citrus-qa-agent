# Citrus QA Agent & Pipeline1 全景架构逆向工程与代码审计

> **审计基准**：当前工作区代码（feature/v8.17-draft-native-ucr 分支，HEAD `2f3c07a`，v9.2 批次后）
> **方法**：全部论断来自对 `.py/.yaml/.ps1/.md` 源文件的逐行读取（read/grep），并以 `[相对路径:行号]` 锚定
> **铁律**：忽略历史文档（README/AGENT_CHANGES/TUNE_PARAMS/未来建议待办 等），只报告当前代码库真实状态；不提炼卖点
>
> **v2026-08 复核修订**（对照实测巡检 `lancedb 0.37.1`）：
> 1. LanceDB 索引状态修正为**实测确认**：8 张表全部已建 `IvfHnswFlat`（COSINE）索引（由 `reindex_lance.py:187-189` / `ingest.py:162-167` 创建，检索自动使用，见 §3.2）。此前"无显式索引/flat"为审计漏读所致，已更正。
> 2. 语料规模更新为**实测**：LanceDB 可检索行 **206,664**（8 表合计，chunks.jsonl 252,681 行列差异见 §3.2 实测表）。
> 3. `_web_streak_step`/`web_unavailable` 修正为"保留但未接生产链路"（有测试锚点 `tests/test_v8153_fixes.py`），非裸死代码（附录 C #1）。
> 4. 管道补丁 `ingest.py:169-179`：LanceDB 追加路径补建索引（幂等，与重建/新建路径同参数），v9.2 审计修正。
>
> **v9.3.0 论文对齐修订**（用户指令"论文对齐代码修改"，全量回归 225 passed 后同步）：
> 1. **联网引用不再截断**：`src/tools/deepseek_web.py` 删除 `_WEB_MAX_ITEMS=8` 上限（原 39/134 行），web_results 全量入证据侧栏；URL 去重与 `#ws_call_id` 剥离保留。`queries[:8]`（查询清单展示上限）保留不变。
> 2. **academic_search / fetch_fulltext 全链删除**：`search.py` 四源 API 常量+函数（原 225-233、874-1149）与 `fulltext.py`（整文件）删除；`tools/__init__.py`、`registry.py`（readonly 白名单+YAML 注册门控）、`agent_runner.py`（白名单+学术预算分支）、`config.py`（ACADEMIC_* 字段）、`api/main.py`（academic_enabled）、`strategy_cards.py`（策略卡文案）、`config.yaml`（academic_search 段+tool_tags 段）同步清理；受影响测试（test_fulltext.py 删除；test_batch3/test_optimization/test_supervisor_final/test_v815_features/test_v817 同步改写）。
> 3. **pipeline1 来源标签摄取阶段自动标注**：`run_pipeline.py` 新增 `--source-type`（默认从 `--pdf-dir` 父文件夹名自动提取）；Qdrant payload 与 `qdrant_ingestor._build_payload` 写入 `_src`；批次 metadata.json `summary.source_type` 与 `_src` 一致。
> 4. **前端侧栏手风琴动态来源**：`index.html` `SOURCE_META` 4 内置组硬编码不变；未知来源（pipeline1 新 source_type）经 `srcKey()` 动态成组排在固定 4 组之后，不再并入 RAG 组（P0#4 确认+增强）。
> 5. **P2 死代码/兜底清理**：`_web_streak_step` 与 `build_evidence_report.web_unavailable` 参数整链删除（含测试锚点）；`tool_tags` 配置段删除；`_corpus_fingerprint` 异常兜底由返回 `"?"` 改为抛 `ValueError`+日志（防缓存污染）。
> 6. **P1 管道整合**：新增 `pipeline1/src/pipeline/lancedb_exporter.py`（`LanceDBExporter.export_batch`，输出 `<output>/<batch>/chunks.jsonl + metadata.json + _idx_map.json`、`<output>/lancedb/<batch>.lance` 表名=批次名 cosine+IVF_HNSW_FLAT）；`run_pipeline.py` 新增 `--backend lancedb|qdrant`（默认 lancedb）、`--output-dir`、`--batch-name`；输出与 `multi_retriever._detect_batch_source`/`_load_lance_batch` 读取约定兼容（E2E 实测通过）。不修改 agent/ 任何文件。
> 7. **版本号统一 9.3.0**：`config.py:146` VERSION 单源 + `index.html`/`run.ps1`/`pack_release.ps1`/`README.md`/`api v2` 同步。
> 8. **语料批次重命名**（用户指令）：非 UCR 批次目录与 LanceDB 表同步重命名 `xrz→paper1`、`dxy-1→paper2`、`720600→paper3`、`7.20→paper4`、`1-1200→paper5`、`1-50→paper6`、`51-101→paper7`；`categories-cn`（UCR）不动。agent 端实测加载 8 表 / 252,681 chunks / `categories-cn→ucr`、`paper1-7→rag`，前端侧栏"本地文献库"与"UCR品种库"同级展示（数字命名供前端正则隐去）。BM25/RAG 缓存指纹随批次名变化自动失效重建（预期）。
>
> 仓库根：`E:\codex_WORKSPACES\Citrus_QA_Agent`（git 仓库，`agent/` 为核心子目录）
> 摄取管道：`E:\codex_WORKSPACES\pipeline1`（独立目录，非 git 仓库）

---

## 第一部分：宏观拓扑与目录结构

### 1.1 全局目录树（2-3 层 + 职责）

```
E:\codex_WORKSPACES\Citrus_QA_Agent\
├── agent\                          # 核心代码区（问答服务、多智能体、检索、上下文）
│   ├── src\
│   │   ├── api\main.py             # FastAPI 入口 / SSE 网关 / 全部 HTTP 端点（1056 行）
│   │   ├── core\                   # 核心业务层：agent_loop / agent_runner / context_budget /
│   │   │                           #   context_manager / search_both / llm_pool / tracing /
│   │   │                           #   progress_bus / jobs / write_pipeline / evidence /
│   │   │                           #   cache_metrics / diag / business_logger / db / background
│   │   ├── engine\                 # 本地推理引擎：embedder(fastembed e5) / reranker(onnx) /
│   │   │                           #   gpu_lock(全局互斥) / hardware(onnx provider 选择)
│   │   ├── graph\                  # LangGraph 状态图：expert_graph(专家) / light_graph(轻量) /
│   │   │                           #   graph.py(模式分发) / state.py(TypedDict)
│   │   ├── guardrails\             # 上下文压缩 history_compactor / 记忆内存 memory(LTM/偏好/常驻卡片)
│   │   ├── prompts\                # 提示词工程：source/(21 源文件) / builds/(固定拼接产物) /
│   │   │                           #   snapshots/(版本快照) / loader.py(拼接器) / snapshot.py(快照工具)
│   │   ├── retrieval\              # 检索后端：multi_retriever(双管道融合) / bm25(自实现 BM25+RRF) /
│   │   │                           #   __init__(RAG 预热)
│   │   ├── session\manager.py      # 会话/消息/证据账本 SQLite 持久层（979 行）
│   │   ├── tools\                  # 7 个工具 + 注册中心 registry / supervisor_tools(schema)
│   │   ├── skills\                 # 写作技能库（按学术章节组织：intro/methods/results/...）
│   │   └── config.py               # pydantic-settings 统一配置（env > yaml > 默认值）
│   ├── config.yaml                 # 全量运行配置（304 行，检索/模型/预算/工具注册等）
│   ├── tests\                      # 225 个 Pytest 用例（v9.3.0 全量回归绿）
│   ├── data\                       # 语料库（lancedb/qdrant_data/chunks.jsonl，gitignore 内）
│   ├── docs\                       # 文档（agent_extension.md）
│   ├── state\sessions.db           # SQLite 会话/消息/记忆持久化
│   └── ingest.py / reindex_lance.py / prepare_models.py  # 语料导入/重索引/模型准备脚本
├── run.ps1                         # 一键启动脚本（语料下载→Python→依赖→模型→uvicorn）
└── e2e_check.py / pack_release.ps1 # 端到端自检 / 打包发布

E:\codex_WORKSPACES\pipeline1\
├── run_pipeline.py                 # 摄取管道入口（PENDING→PARSED→TAGGED→CHUNKED→EMBEDDED）
├── src\pipeline\
│   ├── config.py                   # 管道配置（分块/嵌入/Qdrant/三级标签体系）
│   ├── parser_cleaner.py           # LlamaParse PDF→MD + 4 阶段清洗
│   ├── chunker.py                  # LangChain Markdown 头 + 递归字符双级分块
│   ├── vectorizer.py               # FastEmbed 嵌入（GPU 崩溃智能降级 CPU）
│   ├── qdrant_ingestor.py          # Qdrant 幂等 upsert 入库
│   ├── metadata_extractor.py       # 本地元数据提取（fitz，0 API 调用）
│   ├── tagger.py                   # DeepSeek 三级标签 + Validator/Reflection 重试
│   └── state_manager.py            # SQLite 纸面状态机（file_hash 变更检测）
├── data\                           # chunks / parsed_markdown / qdrant_data / metadata.json / pipeline_state.db
└── scripts\repair_missing_doi.py   # 缺失 DOI 修复脚本
```

### 1.2 进程与入口

| 项 | 结论 | 锚点 |
|---|---|---|
| Agent 服务入口 | `uvicorn src.api.main:app --host 127.0.0.1 --port 8000`（run.ps1 启动；浏览器打开 `http://localhost:8000`） | `run.ps1:285` |
| FastAPI 应用 | `app = FastAPI(lifespan=lifespan)`（SSE 用 sse_starlette `EventSourceResponse`） | `src/api/main.py:78,26` |
| 启动预热 | lifespan 依次：`eager_load_rag()`（MultiBatchRetriever+SkillTree+Embedder+Reranker）、`ensure_fixed_prompts()`（固定提示词）、supervisor 工具 schema 快照日志 | `src/api/main.py:77-108` |
| 管道入口 | `run_pipeline.py --pdf-dir <dir>`（argparse；`--force/--skip-parse/--skip-chunk/--skip-ingest/--dry-run`） | `pipeline1/run_pipeline.py:45-62` |
| 多进程/多线程模型 | **单进程** FastAPI（asyncio）；工具执行走 `run_in_executor` 线程池；检索用 `ThreadPoolExecutor`；GPU 推理受**全局 threading.Lock** 串行化（Windows DML 单设备限制） | `src/core/background.py`, `src/engine/gpu_lock.py:12`, `src/retrieval/multi_retriever.py:709` |
| 后台任务 | `spawn()`（强引用 + 异常日志）、`drain()`（shutdown 等待） | `src/core/background.py:29-56` |
| 断连保活 | write 类 job 断连 → `adopt(graph_task)` 转后台继续 | `src/api/main.py:463-469` |

### 1.3 外部依赖边界

| 外部服务 | 客户端初始化点 | 说明 |
|---|---|---|
| DeepSeek Chat API | `src/core/llm_pool.py:139-198`（`ChatOpenAI` 进程级缓存，键=model/key/base_url/temp/timeout/max_tokens/thinking_off） | 主模型 `deepseek-v4-flash`；快模型同池 |
| DeepSeek Responses API（联网） | `src/tools/deepseek_web.py:190-206`（`requests.post` → `{base}/v1/responses`，`tools=[{"type":"web_search"}]`） | 每请求预算 1 次 |
| 本地向量库 LanceDB | `src/retrieval/multi_retriever.py:470-477`（`lancedb.connect(data/lancedb)` + `open_table(batch)`） | backend=auto→lancedb（yaml 默认 `lancedb`） |
| 本地向量库 Qdrant（旧） | `src/retrieval/multi_retriever.py:280`（`QdrantClient(path=..., timeout=QDRANT_TIMEOUT)`） | 与 LanceDB 并存（`self.batches`）；全锁冲突降级 BM25 |
| 本地 SQLite | `src/core/db.py`（`connect_db`：WAL + busy_timeout=30s）；消费方 `session/manager.py`、`guardrails/memory.py`、`core/jobs.py`、`core/write_pipeline_state.py` | state/sessions.db |
| 本地 ONNX 模型 | `src/engine/embedder.py:83-90`（fastembed e5-large）、`src/engine/reranker.py:66-77`（optimum ORT，`.hf_cache/onnx_reranker`） | 首启从 HF 镜像下载（`hf-mirror.com`）并缓存 |
| Qdrant（pipeline1 写入端） | `pipeline1/src/pipeline/qdrant_ingestor.py:20-22`（local 模式 `QdrantClient(path=data/qdrant_data)`） | 见第五部分 |
| LlamaParse（云端 PDF 解析） | `pipeline1/src/pipeline/parser_cleaner.py:35-42`（`LlamaParse(api_key=LLAMAPHASE_API_KEY)`） | 管道专用，agent 端不用 |

---

## 第二部分：多智能体编排与并发控制

### 2.1 Agent 角色与工具白名单

**Supervisor（expert 模式）**：主业决策者。绑定 6 个工具（schema 单一来源 `src/tools/supervisor_tools.py:12-180`）：

| 工具 | 用途 | 锚点 |
|---|---|---|
| `call_search_both(local_goal, web_goal)` | **统一检索入口**（v9.1）：本地+联网并行，空 goal 自动互填 | `supervisor_tools.py:16-45`, `search_both.py:91-153` |
| `call_write_agent(goal, context, output_path)` | 长文写作（write-agent 子代理） | `supervisor_tools.py:49-78` |
| `call_analyze_agent(goal, data_context)` | 统计/实验设计（analyze-agent） | `supervisor_tools.py:82-101` |
| `read_local_file(path, max_chars)` | 读本地文件 | `supervisor_tools.py:105-127` |
| `pdf_read(file_path)` | 学术 PDF 结构提取 | `supervisor_tools.py:131-148` |
| `write_local_file(path, content, mode)` | 直写已存在内容（workspace/output） | `supervisor_tools.py:152-179` |

**子代理（agent_runner 硬编码映射，`src/core/agent_runner.py:773-790`）**：

| Agent | 工具白名单 | 轮次上限 |
|---|---|---|
| `retrieve-agent` | `citrus_rag_search`（v9.2 起仅此一项，academic 全链删除） | 3（config `subagents.retrieve-agent.max_turns`） |
| `write-agent` | `write_local_file` | 6 |
| `analyze-agent` | `statistical_analysis` + `experimental_design` | 2 |

**Web-agent（v9.1，无 LLM）**：不消费 system prompt；直接调 `deepseek_web_search.func` 一次（`search_both.py:28-38`）。

**Light 模式**：LLM 自主路由，工具 = `citrus_rag_search` + `read_local_file`（`light_graph.py:34`），不分配联网工具（`light_graph.py:199`）。

**全局工具注册（9 个）**：`src/tools/__init__.py:11-16`；YAML 注册（category/concurrency_key，`registry.py:473-502`，academic 门控）。
**只读白名单**：`_READONLY_TOOLS`（`registry.py:128-132`）。

### 2.2 并发调度引擎 call_search_both

`call_search_both`（`src/core/search_both.py:91-153`）是 v9.1 架构核心：

1. **空 Goal 互填兜底**：`fallback = local_goal or web_goal or fallback_query`；然后 `local_goal = local_goal or fallback; web_goal = web_goal or fallback`（`search_both.py:104-111`）——任一为空自动补另一个，保证双向都不空转。
2. **并行执行**：`asyncio.gather(_run_local(), run_web_agent(web_goal))`（`search_both.py:121-122`）。`_run_local` = `run_agent("retrieve-agent", {"goal": local_goal, "query": local_goal[:200]}, timeout_sec=120)`（`search_both.py:113-119`）；`run_web_agent` = `asyncio.to_thread(deepseek_web_search.func, _goal)`（无 LLM，`search_both.py:38`）。
3. **总等待 = max(本地, 联网)**（消除木桶效应，注释于 `search_both.py:1-16`）。
4. **结果合并**：`merged_arts` 聚合 main_results/web_results/web_summaries 三个 artifact 通道（`search_both.py:124-133`）；回执文本两段拼接（`search_both.py:135-142`）；状态 = 仅当本地与联网**都** error 才 error（`search_both.py:144-152`）。
5. **Web-agent 判定六态**：`[DISABLED]`→disabled / `[WEB_BUDGET_EXHAUSTED]`→budget / `[ERR_*]`→error / 空结果→empty / 有引用无正文→ok / 正常→ok（`search_both.py:50-69`）。

**supervisor 工具执行**：`_execute_supervisor_tools`（`expert_graph.py:约610-945`）——call_search_both 分派（`expert_graph.py:237-248`）、call_retrieve_agent 废弃防御（`expert_graph.py:225-233`）、write 类 job 升级（`expert_graph.py:900-907`）、**连续工具失败≥3 熔断强制收尾**（`expert_graph.py:913-943`，`[ERR_HITL_REJECT]` 不计数，`expert_graph.py:925-926`；剩余 tool_calls 补 `[circuit_breaker]` 占位 ToolMessage 保 INV-01 配对）。

### 2.3 状态机与路由

**Supervisor 拆解 local_goal/web_goal**：由 LLM 经 `call_search_both` 工具参数产出（prompt 中 supervisor 被要求"完整意图、不拆解"，拆解归 retrieve-agent 的 2-4 个角度）。见 `supervisor_tools.py:16-45` 工具描述（"local_goal = Complete local retrieval intent... ALWAYS provide BOTH goals"）。

**LangGraph 状态（AgentState，`src/graph/state.py:8-40`）**：`messages`(add_messages 归约)、`query`、`session_id`、`mode`、`idempotency_key`、`turn_trace`（tool_calls/ToolMessage 配对）、`history_evidence_block`、`main_results`/`web_results`/`tools_called`、`answer`、`history_summary`、`long_term_memory`、`resident_cards`、`search_suggestions`、`format_hint`、`retrieval_context`、`references_data`、`_trace`。

**Graph 拓扑**：
- expert：`expert_load → supervisor(ReAct 循环 8 轮) → expert_save → END`（`expert_graph.py` 底部 build_expert_graph）
- light：`load_context → light_retrieve(代码级预检索) → light_supervisor(2 轮) → save_context → END`（`light_graph.py:375-389`）
- 模式分发：`build_graph(mode)`（`graph.py:16-20`），main.py 中 `graph.astream(initial_state, stream_mode="updates")`（`main.py:324-325`）。

**ContextVar 贯穿（`src/core/tracing.py`）**——请求生命周期写入点：
- `request_id`（`tracing.py:10-11`）→ `chat_v2` 开头 `new_request_id()`（`main.py:164-165`）
- `session_id` → `set_session_id(sid)`（`main.py:167-168`）
- `job_id` → `set_job_id(job_id)`（`main.py:226-227`）
- `web_search_enabled`（前端开关）→ `set_web_search_enabled`（`main.py:241`）
- **`web_budget_left` 默认 1** → `reset_web_budget(1)`（每请求重置，`main.py:244`）；消费点 `deepseek_web_search` 的 `consume_web_budget()`（`deepseek_web.py:161-166`，超限返回 `[WEB_BUDGET_EXHAUSTED]`）
- `original_query`（用户原文）→ `set_original_query(query)`（`main.py:247`）
- 工具线程经 `contextvars.copy_context()` + `ctx.run(...)` 保持（`registry.py:233-243`）

---

## 第三部分：检索管道微观实现

### 3.1 查询准备（Query Generation）

**结构化 HyDE（v8.16.3）**：一次 fast LLM 调用产出三条 labeled 行：`HyDE:`（200-500 词英文假想段落）+ `Multi-Query:`（3 路）+ `Summary:`（3-5 要点）。位于 `src/tools/search.py`：

| 函数 | 行号 | 逻辑 |
|---|---|---|
| `_HYDE_PROMPT` | `search.py:490-505` | 提示词模板（三行输出约束） |
| `_generate_hyde_structured` | `search.py:551-620` | 生成：`OpenAI(FAST)` 调用，`max_tokens=HYDE_MAX_TOKENS(2048)`，`temperature=0.2`；空 content 重试一次；extra_body 关思维链（`HYDE_THINKING_OFF`，fail-soft 去参重试）；失败/非英文→None 回退 |
| `parse_hyde_structured` | `search.py:515-549` | 解析：剥围栏、HyDE 段跨行折叠、HyDE<30 字符或 CJK>5% → None |
| `_cached_hyde` | `search.py:621-630` | HyDE LRU 缓存（500 条，md5(query) 键） |
| `_cached_hyde_parsed` | `search.py:632-650` | 旧式纯段落兼容（退化单路） |

**容错链**：生成空 → 重试 1 次 → 仍空/不可解析 → `rag.search(query)` 基础检索（无 HyDE，`search.py:420`）。

**Multi-Query 并发**：结构化 HyDE 后 queries = `[hyde] + multi_query[:3] + summary[:5]`（7-9 路），`rag.search_multi(queries, original_query=query)`（`search.py:418`）。

### 3.2 双通道召回

**稠密路（向量）**：

| 项 | 值 | 锚点 |
|---|---|---|
| Embedding 模型 | `intfloat/multilingual-e5-large`（fastembed，本地 ONNX） | `config.yaml:57`, `embedder.py:58` |
| 维度 | 1024（运行时探测 `dim`） | `embedder.py:86-89,161-168` |
| Query 前缀 | `"query: "`（E5 训练分布）；文档侧管道用 `"passage: "` | `embedder.py:134`, `pipeline1/vectorizer.py:109` |
| LanceDB 索引 | **IVF_HNSW_FLAT（COSINE）**——建表时即创建（`reindex_lance.py:187-189` 重建 / `ingest.py:162-167` 新建），参数 `num_partitions=64, m=16, ef_construction=200, metric=cosine`；`table.search()` 自动使用已有索引（LanceDB 0.37 查询时不支持指定 metric，由索引决定；无索引表才退化为 flat，注释 `multi_retriever.py:535-536` 自证），相似度 = `1 - _distance` | `multi_retriever.py:532-548`, `reindex_lance.py:185-191`, `ingest.py:162-167,169-179` |
| 批次数 | 全部批次并发（lanes/batches 遍历） | `multi_retriever.py:705-713` |
| top_k 默认 | `TOP_K_VECTOR=40` | `config.yaml:6` |
| 过滤条件 | **无**——混合库不分区，靠 idx_map 定位 + 后置 rerank 过滤 | `multi_retriever.py:541-546,325-330` |

**语料实测（2026-08 环境巡检，`lancedb` 0.37.1）**——8 张表全部已建 `IvfHnswFlat` 索引（metric=COSINE，`m=16/ef=200`，未索引行 = 0；`index_type` 实测为 `IvfHnswFlat`，即 `IVF_HNSW_FLAT` 参数在 LanceDB 0.37 的实际落盘类型）：

| 表（批次） | LanceDB 行数 | 索引 | chunks.jsonl 行数 |
|---|---|---|---|
| 1-1200 | 61,509 | ✅ IvfHnswFlat | 61,509 |
| 1-50 | 5,858 | ✅ | 15,940 |
| 51-101 | 1,848 | ✅ | 1,848 |
| 7.20 | 11,783 | ✅ | 43,416 |
| 720600 | 37,375 | ✅ | 41,602 |
| categories-cn | 8,752 | ✅ | 8,752 |
| dxy-1 | 16,232 | ✅ | 16,307 |
| xrz | 63,307 | ✅ | 63,307 |
| **合计** | **206,664** | 8/8 已索引 | 252,681 |

> **行数与 jsonl 差异**：`reindex_lance.py` 建表时跳过空文本 chunk（`reindex_lance.py:169` "empty-text chunks skipped"）+ 按 `(paper_id, chunk_index)` 键合并，故 1-50/7.20/720600/dxy-1 等批次行数小于 jsonl 行数。**实际可检索向量数 = 206,664**（LanceDB 表行数，8 表总和），与历史文档中的"119,113 / 73,096"均不同——历史数字为旧快照，以本次实测为准。

**稀疏路（BM25）**：**自实现 `BM25Plus`**（非第三方库，`src/retrieval/bm25.py:12-121`）：
- 参数：`k1=1.5, b=0.75, delta=1.0`（BM25+ 变体）
- 分词：`[a-z0-9_\-]+|[\u4e00-\u9fff]+`（`bm25.py:10`，v8.11 numpy 倒排索引，`bm25.py:48-64`）
- 持久化：`.hf_cache/bm25/bm25_v{fmt}_{指纹}.pkl`，指纹 = 全语料 md5 + 参数（`multi_retriever.py:31-41`，语料变化自动失效重建；保留最近 3 份）
- 查询：`_top_k_inverted`（numpy 平行数组向量化，释放 GIL，多线程安全）
- top_k 默认 `TOP_K_BM25=40`

**并行机制**：`search_multi`（`multi_retriever.py:669-732`）——BM25 先提交独立线程（max_workers=1 包装 `_bm25_search_parallel`，查询间最多 4 线程，`multi_retriever.py:44-60`）与 embed 重叠；向量检索 `ThreadPoolExecutor(max_workers=len(batches)*len(queries))` 全批次并发（`multi_retriever.py:709`）；`search_hyde` 双 dense + BM25 三流并行（`multi_retriever.py:758-787`）。**注：`max_workers` 未设上限**（批次×查询数）。

### 3.3 融合与精排

**RRF 融合**：`rrf_fuse(*hit_lists, k=60, weights=None)`（`bm25.py:123-136`）——加权 RRF，`score += w / (k + rank + 1)`；k 默认 60（`config.yaml:9` `rrf_k: 60`）；权重 `RRF_WEIGHT_ORIG_DENSE/HYDE_DENSE/BM25` 均 1.0（`config.yaml:10-13`）。

**收敛出口 `_fuse_rerank_select`**（`multi_retriever.py:580-667`，v8.13-b4b 双管道收敛点）：
1. 每流按分数降序 + global_idx 去重（保留最高分，`multi_retriever.py:589-596`）
2. `rrf_fuse`（`multi_retriever.py:598`）
3. 候选 = fused 前 `TOP_K_FINAL*2 = 20` 条（懒加载全文，`multi_retriever.py:599-600`）
4. Rerank：`Reranker.rerank(rerank_query, candidates, top_k=10)`（`multi_retriever.py:602-604`）

**Reranker**（`src/engine/reranker.py`）：
- 模型：`BAAI/bge-reranker-v2-m3`（optimum ONNX，`.hf_cache/onnx_reranker` 缓存）
- 输入：(query, chunk_text) pairs，`max_length=512`，**整批一次推理**（`reranker.py:94-104`）
- 得分：sigmoid `1/(1+exp(-logit))`（`reranker.py:20-21`）
- GPU：DML/CUDA 时全局 `GPULockGuard` 串行（`reranker.py:98-102`），CPU 线程本地会话
- 降级：UnicodeDecodeError/运行时异常 → 强制 CPU 重建（`reranker.py:112-135`）
- **批量"调度"**：无显式分 batch（一次跑全部候选）

### 3.4 动态阈值

**公式**（`multi_retriever.py:614-617`，代码原文）：

```python
top_score = reranked[0].get("rerank_score", 0)
dynamic_thresh = top_score * settings.DYNAMIC_THRESHOLD_RATIO   # ratio = 0.60
final_threshold = max(settings.RERANK_THRESHOLD, dynamic_thresh) # floor = 0.25
passed = [c for c in reranked if c.get("rerank_score", 0) >= final_threshold]
```

即 `T = max(0.25, top_score * 0.60)`。**强/弱证据不显式划分**——只有通过/拦截二值；拦截全部时 `last_empty_reason = "threshold_blocked"`（`multi_retriever.py:618-623`）。通过后回传 `last_stats`（candidates/passed/filtered，`multi_retriever.py:664-666`）供 `rag_stats_note` 生成早停提示（`search.py:169-194`）。

### 3.5 联网搜索 deepseek_web_search

（详见 `src/tools/deepseek_web.py`，309 行）

1. **短路链**（`deepseek_web.py:152-166`）：前端开关关 → `[DISABLED]`；`consume_web_budget()` 失败（请求级预算=1 已用尽）→ `[WEB_BUDGET_EXHAUSTED]`——**均零网络请求**。
2. **调用**：`requests.post(f"{base}/v1/responses", tools=[{"type":"web_search"}], stream=False)`（`deepseek_web.py:190-206`），HTTP 超时 `_web_http_timeout()`（config `web_search.timeout_sec=90`，钳制 30~300s，`deepseek_web.py:43-51`）；关思维链 `reasoning: {effort: none}`（`config.yaml:50-52`，400/422 去参重试，`deepseek_web.py:207-213`）。
3. **解析**：`_parse_response_output`（`deepseek_web.py:93-135`）——`message` 块 `output_text` 文本 + Markdown 链接/裸 URL 正则提取（`_MD_LINK_RE`/`_BARE_URL_RE`，`deepseek_web.py:54-55`）+ `web_search_call` 的 queries（过滤 `ws_call_id=` 伪查询，`deepseek_web.py:119-121`）+ 深扫兜底 `_extract_urls_deep`（任意深层 dict/list 找 url/title，`deepseek_web.py:73-90`）；URL 去重 + 剥离 `#ws_call_id` 内部锚点（`deepseek_web.py:60-62,125-133`）；**引用不设上限**（v9.3.0 删除 `_WEB_MAX_ITEMS=8`，全量进入 web_results）。
4. **返回**：content = 摘要(≤2000) + 检索词 + `[Wn]` 引用清单（`deepseek_web.py:260-265`）；artifact `web_results`（每项 `ref_id=W{idx}, type=web, source=web, url/title/abstract/snippet`，`deepseek_web.py:233-244`）+ `web_summary`（≤4000，`deepseek_web.py:278`）。
5. **联网失败防护（v9.1 起）**：v8.15.3 的 `_web_fail_streak` 熔断状态机已移出 retrieve-agent；v9.3.0 将遗留纯函数 `_web_streak_step` 与 `build_evidence_report.web_unavailable` 参数整链删除（含测试锚点）。实际联网失败防护 = **请求级预算 1 次短路**（`deepseek_web.py` 内 `consume_web_budget()`，无跨请求熔断状态机）。

### 3.6 查询级缓存

`src/tools/search.py:278-344`：`_RAG_CACHE`（OrderedDict LRU，`RAG_CACHE_SIZE=300` 条、`RAG_CACHE_TTL_HOURS=24`）。
- **cache_key** = `md5(规范化query | hyde=0/1 | 语料指纹)`（`search.py:296-309`）
- **语料指纹** = `f"{len(global_chunks)}:{','.join(sorted(batch_source))}"`（总块数+批次名集合，`search.py:284-294`）——语料增删（块数/批次变化）自动失效；重启自动失效。v9.3.0：异常不再降级 `"?"`（防缓存污染），抛 `ValueError` 并记录日志
- 命中返回浅拷贝隔离（`search.py:330`）；TTL 检查（`search.py:325`）
- agent 端 LanceDB/Qdrant 均为**每查询重新 search**（无连接复用问题），**无需重启即见新数据**（文件级/本地客户端；qurant local 模式单实例锁在 `_clean_stale_locks` 处理，`multi_retriever.py:202-230`——**同一进程内多实例会锁冲突**，但 pipeline1 与 agent 是**不同进程不同库目录**，互不影响）

---

## 第四部分：上下文工程与记忆网络

### 4.1 提示词矩阵（v9.0 固定拼接）

**source/ 21 个源文件**（`agent/src/prompts/source/`，唯一维护入口）：

| 编号 | 文件 | 内容 |
|---|---|---|
| 01 | global_role_domain | 全局角色与领域 |
| 02 | global_data_fidelity_citation | 数据保真与引用纪律 |
| 03 | supervisor_routing_fusion | Supervisor 路由与融合 |
| 04 | retrieve_agent_search | 检索子代理搜索 |
| 05 | academic_writing_common | 学术写作通用 |
| 06 | review_planner | 综述规划（Plan） |
| 07 | review_chapter_writer | 章节撰写（Section） |
| 08 | data_analysis_experiment | 数据分析与实验设计 |
| 09-15 | format_fact/mechanism/compare/review/experiment/task/default | **7 个输出格式模板**（全部固定进 supervisor prompt） |
| 16 | tool_usage_file_rules | 工具使用/文件规则 |
| 17 | data_source_boundaries | 数据源边界 |
| 18 | evidence_arbitration_citation | 证据仲裁与引用 |
| 19 | lite_mode | 轻量模式规则 |
| 20 | terminology_domain | 术语领域 |
| 21 | web_agent_search | Web 检索（v9.1） |

**固定拼接机制**（`src/prompts/loader.py`）：
- `ROLE_SOURCE_FILES`（`loader.py:30-87`）：supervisor=14 文件、retrieve=5、write=6、lite=4、analyze=4、web=4
- `build_fixed_prompts()`（`loader.py:149-172`）：启动时拼接 + 落盘 `builds/*.md`（幂等）；`ensure_fixed_prompts()` 进程级缓存（`loader.py:175-180`）
- `assemble_system_prompt(mode=...)` → 固定字符串（参数仅签名兼容，`loader.py:192-206`）
- `build_dynamic_blocks()` 恒返回空串（`loader.py:209-217`）——**无动态格式块**
- 动态内容（skills/task_type）经 `build_agent_extra_block` 追加到首条 HumanMessage 尾部（不破坏前缀，`loader.py:239-266`）
- 快照工具 `src/prompts/snapshot.py` 渲染确定性 .txt 到 `snapshots/`（审查用）

### 4.2 上下文压缩（存储全量·发送裁剪）

**架构**：SQLite 原始轨迹 append-only；压缩是"发送视图构建"，只在**用户轮边界**（新请求 load 时）执行（`context_manager.py:295-360`, `context_budget.py:1-14`）。

**触发阈值**（`ContextBudgetConfig`，`context_budget.py:32-41`；config 接线 `config.yaml:196-208`）：

| 参数 | 默认值 | 语义 |
|---|---|---|
| `max_tokens` | 1,000,000（=V4 Flash 窗口） | 发送视图预算 |
| `soft_threshold` | 0.75（750K） | 视图≥75% → LLM 批量压缩 |
| `hard_threshold` | 0.93（930K） | ≥93% → 规则式保护截断 |
| `target_ratio` | 0.50 | 压缩目标：压到~50% |
| `protect_recent_turns` | 3 | 保护名单：最近 3 轮 Q/A 不压 |
| `keep_recent_turns` | 2 | TRUNCATE 硬截断保留轮数 |
| `compact_max_tokens` | 800 | 摘要输出上限 |

**token 估算**：`_estimate_chars_tokens`（`context_budget.py:52-64`，CJK 1.2 tok/字 + 其他 4 字符/tok）。

**压缩流程**（`ContextBudget.check → _compress`，`context_budget.py:256-347`）：去噪（`_trim_noise`，占位 `[ERR_*]`/circuit_breaker/budget_skip 优先删 + 同步修剪配对 tool_calls 保 INV-01，`context_budget.py:203-254`）→ 保护最近 N 轮 → 压缩 → 压缩后仍超硬阈值 → `_hard_trim`。

**保护名单**（压缩时强制保留，代码级）：
- LLM 压缩提示词要求保留：**DOI、evidence_id、artifact_id、chunk_id、文件路径**、关键实体/数值/研究意图/结论状态（`history_compactor.py:71-76`）；`[n]` 引用编号可不保留（原引用列表不持久化）
- 规则式截断 `_keep_identifiers`（`context_budget.py:135-152`）：正则匹配 `doi|evidence_id|artifact_id|chunk_id|source_id|pmid|url` 行保留 + 显式 `[TRUNCATED]` 透明标记
- `_hard_trim`（`context_budget.py:378-423`）：大 ToolMessage(>3000 字符) 用 `_keep_identifiers` 截断；仍超限 → 只留首条 + 最近 keep_recent_turns 轮 + 中间轮标识符行注入 `[硬截断保留标识符]`

**摘要规则双实现（当前并存）**：
- `_rules_summary`（`context_budget.py:368-376`）：熔断时规则式（最近 6 条 + 2000 字符）
- `_fallback_summary`（`history_compactor.py:46-51`）：无 LLM 时兜底（最近 6 条 + 2000 字符）
- 两者逻辑同构（无 LLM 时的最近 N 条 + 截断）——**批次 11 候选收敛点**

**压缩熔断（P9 半开）**：连续失败≥3 → 规则式；5 分钟冷却窗（`_COMPACTION_COOLDOWN_SEC=300`，`context_budget.py:80`）后放行一次 LLM 重试（半开）；条目清理（>64 惰性剔除 1h 过期，硬上限 1024，`context_budget.py:85-100`）。

### 4.3 记忆架构

**LTM 提取（save 节点后台）**：
- 门槛 `len(answer)>500` + `default_ltm_gate`（回答含 "###"/结论/摘要/引言等结构化关键词，`agent_loop.py:222-227`，expert）；light 仅长度门槛
- `_extract_and_save_ltm`（`agent_loop.py:331-352`）→ `extract_key_facts`（`memory.py:451-490`，fast 模型 JSON 数组 ≤5 条）→ `type=preference` 走全局偏好域；否则 `save_long_term_fact`
- **置信度门槛**：`save_long_term_fact` 拒绝 `confidence < 0.5`（`memory.py:259-261`）

**LTM 召回（语义）**：`recall_long_term_memory(query, top_k=5, session_id)`（`memory.py:287-298`）：
- 混合排序 = 余弦相似度 × 有效置信度（`memory.py:366-369`）；时间衰减 `0.95^天`（`memory.py:359`）；score≥0.30 且 eff_conf≥0.30（`memory.py:378-383`）
- **域隔离**：`owner_session` 过滤——本会话域 + 全局高置信(≥0.9)共享 + 旧数据（`_fetch_ltm_rows`，`memory.py:304-324`）；跨域事实排序降权 0.9（`CROSS_DOMAIN_WEIGHT`，`memory.py:301-302,361-363`）
- 关键词兜底 `_recall_keyword_fallback`（`memory.py:401-449`）

**常驻卡片（resident_cards）**：高置信(≥0.8)事实常驻，上限 8 条、单条 60 字符（`memory.py:172-174`）；底部淘汰最低置信+最旧（`memory.py:195-204`）；会话域(''=全局)（`memory.py:164-170,222-228`）。

**偏好（preference_memory）**：`GLOBAL_PREF_DOMAIN="__global__"`（用户级偏好跨会话，`memory.py:54`）；`get_preferences` 全局为底 + 会话域覆盖（`memory.py:65-89`）→ `<user_preferences>` 块注入。

**证据账本（session_evidence，会话级）**：
- `save_evidence`（`session_manager.save_evidence`，`manager.py:728-764`）：turn_seq 递增、report_text 8000 字符截断带标记、evidence_json ≤1M
- 统一读取器 `_load_evidence_rows`（`manager.py:766-796`，v9.2 CON-7 收敛）
- 跨轮复用：`build_evidence_block(session_id, limit=2)`（`manager.py:798-821`）→ `[历史检索证据...]` HumanMessage 注入（`context_manager.py:547`, `assemble_supervisor_messages` `context_manager.py:85-100`）；`get_evidence_refs`（最近 10 轮 → H1..Hn 前端侧栏 historical 组，`manager.py:835-875`）；`get_evidence_materials`（最近 4 轮 → 写材料包，`manager.py:877-910`，expert_graph.py:336-341 消费）

**checkpoint**：`get_checkpoint/set_checkpoint`（`manager.py:440-458`）——sessions 表 checkpoint_msg_id/summary 持久化（`manager.py:193-197`），防摘要套摘要。

---

## 第五部分：数据摄取与动态知识库（Pipeline1 + Agent 侧）

### 5.1 解析与分块

**格式与库**：PDF（LlamaParse 云端 → Markdown，`parser_cleaner.py:35-42`）；agent 端 ingest.py 支持 PDF/txt/md（fitz/直接读，`agent/ingest.py:40-51`）。

**4 阶段清洗**（`parser_cleaner.py:91-143`）：①参考文献截断（标题向前扫 + 编号反向扫 ≥10 行，`parser_cleaner.py:263-327`）→ ②噪声行/元数据段删除（`parser_cleaner.py:390-431`）+ 标题编号清理 → ③高频行去重（≥3 次页眉页脚，`parser_cleaner.py:456-474`）→ ④YAML frontmatter（paper_id/doi/parsed_at/cleaning_stats，`parser_cleaner.py:481-493`）。

**分块策略（`chunker.py:31-90`）**：**结构化双级**——
1. `MarkdownHeaderTextSplitter`（`#`/`##`/`###` 按标题切，`strip_headers=False`，`chunker.py:37-43`）
2. `RecursiveCharacterTextSplitter`（`CHUNK_SIZE=1600` / `CHUNK_OVERLAP=160`，分隔符 `["\n\n","\n",". "," "]`，`chunker.py:45-51` + `config.py:26-27`）

chunk 字段：`paper_id / doi / l1_category / l2_subcategories / l3_entities / section_name / chunk_index / text`（`chunker.py:78-87`）。

### 5.2 Qdrant 动态写入（pipeline1）

- **入口**：`run_pipeline.py`；每论文状态机 PENDING→PARSED→TAGGED→CHUNKED→EMBEDDED（`state_manager.py:22-34`）
- **向量**：FastEmbed e5-large，"passage: " 前缀，1024 维，batch 128（`vectorizer.py:103-113` + `config.py:30-32`）
- **Collection**：`citrus_literature_v2`，COSINE/1024（`config.py:35`, `qdrant_ingestor.py:31-33`）
- **写入 API**：`client.upsert`（**幂等 upsert 而非 add**，`qdrant_ingestor.py:84-87`；主循环 batch=16，`run_pipeline.py:142,274-278`）
- **点 ID**：`md5(f"{doi}::chunk_{chunk_index}")[:16]` 转 int ——**确定性**（同文档重跑覆盖同点，`qdrant_ingestor.py:43-47` / `run_pipeline.py:65-69`）
- **Payload metadata**：`paper_id, doi, l1_category, l2_subcategories, l3_entities, section_name, chunk_index`（`qdrant_ingestor.py:50-60`；主循环同构，`run_pipeline.py:262-270`）
- **注意**：payload **不含** `_src/batch_id` 这类 agent 端使用的字段；agent 端批次来源由 `data/<batch>/metadata.json` 的 `summary.source_type` 判定（`multi_retriever.py:341-356`）

### 5.3 热更新机制

**关键结论：两条数据链路是异构的。**
- **agent 端生产语料**（`agent/ingest.py` / GitHub Releases 分发）：写 **LanceDB**（新建表即建 IVF_HNSW_FLAT 索引 `ingest.py:162-167`；追加路径 v9.2 补丁保证无索引旧表也补建 `ingest.py:169-179`；`table.add(rows)` 追加语义）+ 每批次 `metadata.json`。LanceDB 为嵌入式文件库，agent 进程内单例 `lancedb.connect(data/lancedb)` + `open_table(batch)`（每查询 search），**写入新表后重开表即可见，无需重启**（`multi_retriever.py:470-477`）。
- **pipeline1 生产语料**：写 **Qdrant local**（`data/qdrant_data`）。agent 端检索器在 `backend=auto` 时优先 LanceDB（`config.yaml:5` 默认 `lancedb`）；pipeline1 的 Qdrant 数据**只有在 agent 端无 lance 表时**才被加载（`multi_retriever.py:134-138`）。且 Qdrant local 有**单实例锁**——pipeline1 与 agent 若同时打开同一目录会锁冲突（但两者数据目录不同：pipeline1 在 `pipeline1/data/qdrant_data`，agent 在 `agent/data/<batch>/qdrant_data`，实际不冲突）。
- **重启需求**：**不需要**。LanceDB 无锁读取（文件级句柄），BM25 索引按**语料指纹**（md5 全文+参数）缓存——语料变化自动失效重建（`multi_retriever.py:35-41,313-325`）；RAG 缓存 key 含语料指纹（总块数+批次名）自动失效（`search.py:284-309`）。

---

## 第六部分：成本约束与可观测性

### 6.1 早停与预算

**检索早停（代码裁决）**——两处：

1. **子代理边际收益收敛**（`agent_runner.py:588-607`）：

```python
_uniq_now = len(_dedup_evidence_items(collected_artifacts["main_results"]))
_new_ratio = (_uniq_now - _prev_unique) / max(_uniq_now, 1)
if _prev_unique >= 6 and _new_ratio < 0.25:   # 累计≥6 且新增<25% → 提前结束
    break
```

即"新增<25% 且 ≥6 条"的实现：**代码强制**（不是模型判断），在 retrieve-agent 每轮工具后裁决。

2. **检索统计回传提示**（`rag_stats_note`，`search.py:169-194`）：passed≤2 或过滤占比≥50% → 提示模型"停止该角度"（软提示，模型配合）。

**工具预算拦截器（不破坏消息序列）**：
- 检索去重：`[DEDUP]` 占位 ToolMessage（`agent_runner.py:485-487`）
- 检索预算：`[SEARCH_BUDGET]` 占位 ToolMessage（`agent_runner.py:491-504`）——上限：每轮 rag≤2 次（`_MAX_RAG_PER_TURN=2`，`agent_runner.py:473`）、请求级 rag≤6（`_MAX_RAG_PER_REQUEST=6`，`agent_runner.py:474`）。v9.3.0：academic 预算分支已随 academic_search 删除
- light 补充检索同机制（`light_graph.py:266-302`）
- **关键点**：占位符是**合法 ToolMessage**（带 tool_call_id=原始 tc id），保留 INV-01 配对（tool_calls ↔ ToolMessage 一一对应），序列不破坏；回执中 `[SEARCH_BUDGET]` 次数显式告知 supervisor（`agent_runner.py:668-682`）
- supervisor 熔断：`[circuit_breaker]` 占位（`expert_graph.py:933-936`）

### 6.2 思维链差异化（llm_pool extra_body 透传）

**机制**（`src/core/llm_pool.py`）：
- `get_llm(..., thinking_off)`：True 时经 **`extra_body`** 构造参数透传 `MODEL_REASONING_OFF_BODY`（默认 `{"thinking": {"type": "disabled"}}`，`config.py:126-127`）——`_new_client` 显式传 `kw["extra_body"]`（`llm_pool.py:142-157`，v8.17.18 曾顶层透传致 TypeError 的修复）
- fail-soft：`_ThinkingOffWrapper`（`llm_pool.py:86-136`）——网关拒绝（400/404/422 或文本含 thinking/reasoning）自动去参重试一次
- `is_thinking_rejected`（`llm_pool.py:73-83`）

**各调用点接线**：

| 调用点 | thinking_off | 锚点 |
|---|---|---|
| supervisor（expert/light） | **false**（不传，思维链开启做融合判断） | `agent_runner.py:367-371` 注释 |
| retrieve-agent | `MODEL_REASONING_MODE == "off"` 时 **true**（config `model.reasoning_mode: "off"`） | `agent_runner.py:370-371` |
| HyDE 生成 | `HYDE_THINKING_OFF=true` → extra_body（chat 端点字段 thinking:disabled） | `search.py:574-607` |
| 联网 Responses | `reasoning: {effort: none}`（`WEB_REASONING_OFF_BODY`，实测 chat 端点字段不适用 Responses 端点） | `config.yaml:50-52`, `deepseek_web.py:185-197` |
| hints（search_suggestions） | **true**（高频低价值，`context_manager.py:374,399`） | `context_manager.py:150-175` |
| 压缩 LLM | **true**（v4-flash 思维链吃 800 tok 预算，`context_manager.py:183-191`） | `context_manager.py:177-192` |

另有 `reasoning_content` 透传 monkeypatch（`llm_pool.py:23-66`）→ 前端「深度思考」折叠块。

### 6.3 日志钩子与导出通道

**检索明细 `retrieval_stages`**（`multi_retriever.py:634-646`，经 `diag` 写入）：

```python
diag("retrieval_stages", mode=..., queries=n_queries,
     embed_ms=..., vector_ms=..., bm25_ms=..., rerank_ms=...,
     total_ms=..., **{label: len(s) for ...}, candidates=..., passed=...,
     top_score=..., threshold=...)
```

同期三通道：
1. **diag JSONL**：`logs/diag/diag_YYYY-MM-DD.jsonl`（每行事件，自动带 req/session/job 关联，`diag.py:78-91`）——**可程序化导出**（grep/jq/脚本聚合），是实验分析主通道
2. **business.log**：`blog("retrieval_done", mode, queries, docs, filtered, ms)`（`multi_retriever.py:652-657`→`business_logger.py:76-78`）
3. **retrieval 过滤日志**：`logs/retrieval/retrieval_YYYY-MM-DD.log`（`RetrievalLogger.log_retrieval`，query/阈值/passed/filtered 明细+片段，`logger.py:90-161`）

**业务日志整体**：
- `blog(event, **fields)` → `logs/business.log`（Rotating 10MB×5，req= 关联，`business_logger.py:58-78`）
- `diag(event, **fields)` → JSONL + agent.log 结构化行 + `diag_span` 区间（`diag.py:78-107`）
- `agent.log`：按日轮转（`logger.py:59-62`）+ `RequestIdFilter`（req= 注入）+ `MaskingFormatter`（PII 脱敏，`logger.py:37-46`）
- 全量事件清单：`thinking/reasoning/tool_call_start/tool_executing/tool_result/text/status/context_usage/agent_summary/permission_request/plan/plan_ready/context_degraded` 等（`progress_bus.py:159-213` + main.py done/citations/error/heartbeat）
- PII 脱敏规则：邮箱/手机/身份证/sk-密钥/密钥赋值（`pii_mask.py:12-24`）

---

## 第七部分：全局参数与配置字典（当前真实默认值）

优先级链（`config.py:3,47-52`）：**运行时覆盖(state/model_config.json) > 环境变量 > YAML > pydantic 默认值**。`settings = Settings()` 模块级单例（`config.py:338`）。

| 参数类别 | 参数/常量 | 当前默认值 | 位置 |
|---|---|---|---|
| 模型 | 基座模型名 | `deepseek-v4-flash`（main/fast 同） | `config.yaml:54-56` |
| 模型 | 上下文窗口（发送视图预算） | 1,000,000（V4 Flash 1M） | `config.yaml:198`, `config.py:140` |
| 模型 | 生成 max_tokens | 4096（supervisor 实际 32768 / light 16384 / write 12000） | `config.yaml:113`, `expert_graph.py:1118`, `light_graph.py:45`, `agent_runner.py:380` |
| 模型 | 温度 | main 0.2 / fast 0.0 | `config.yaml:107-110` |
| 检索 | Embedding 模型/维度 | `intfloat/multilingual-e5-large` / 1024 | `config.yaml:57`, `embedder.py:86-89` |
| 检索 | Reranker 模型 | `BAAI/bge-reranker-v2-m3`（512 tok pair） | `config.yaml:58`, `reranker.py:94-96` |
| 检索 | top_k_vector / bm25 / final | 40 / 40 / 10 | `config.yaml:6-8` |
| 检索 | **RRF k 值** | **60**（`rrf_k: 60`） | `config.yaml:9`, `bm25.py:123` |
| 检索 | RRF 权重 | orig_dense/hyde_dense/bm25 均 1.0 | `config.yaml:10-13` |
| 检索 | **动态阈值 Ratio** | **0.60**（`T=max(0.25, top×0.6)`） | `config.yaml:15`, `multi_retriever.py:614-617` |
| 检索 | rerank 底线阈值 | 0.25 | `config.yaml:14` |
| 检索 | HyDE 开关/max_tokens/关思维链 | true / 2048 / true | `config.yaml:17,21-22` |
| 检索 | **缓存 LRU 大小/TTL** | **300 条 / 24h** | `config.yaml:26-27`, `search.py:307,322` |
| 检索 | 检索结果语料指纹键 | 规范化query+HyDE+块数+批次名 | `search.py:296-309` |
| 上下文 | **压缩 Soft/Hard 阈值** | **0.75 / 0.93**（target 0.50） | `config.yaml:200-202`, `context_budget.py:35-37` |
| 上下文 | 保护轮数/硬截断保留 | 3 / 2 | `config.yaml:204-206` |
| 上下文 | 压缩摘要 max_tokens | 800 | `config.yaml:208` |
| 上下文 | 压缩熔断 | ≥3 失败 → 规则式；300s 冷却半开 | `context_budget.py:80,103-132` |
| 联网 | **Web Search 超时** | **90s**（HTTP，钳制 30-300） | `config.yaml:45`, `deepseek_web.py:43-51` |
| 联网 | 每请求联网预算 | 1 次 | `tracing.py:30-31,47-58`, `main.py:244` |
| 联网 | 引用条目上限 | 不设限（v9.3.0 删除 `_WEB_MAX_ITEMS=8`，全量入 web_results） | `deepseek_web.py:133-135` |
| 联网 | Responses 端点 | `{base}/v1/responses` | `config.yaml:40`, `deepseek_web.py:65-70` |
| 早停 | 边际收益阈值 | 累计唯一≥6 且新增<25% | `agent_runner.py:588-607` |
| 预算 | 检索每轮/请求上限 | rag≤2/轮、请求级 rag≤6（v9.3.0 无 academic） | `agent_runner.py:472-474` |
| 预算 | 工具执行超时 | 60s（deepseek_web_search 覆盖 120s） | `config.yaml:137-148`, `registry.py:246-259` |
| 预算 | supervisor 轮次/每轮工具 | 8 / 2 | `config.yaml:231-233` |
| 预算 | 子代理轮次 | retrieve 3 / write 6 / analyze 2 | `config.yaml:236-242` |
| 预算 | light 轮次 | 2 | `config.yaml:245-246` |
| 写作 | 材料最小数/plan 重试/refs 覆盖率 | 8 / 1 / 0.4 | `config.yaml:152-155` |
| 写作 | 单章 max_tokens/超时/并行度 | 4000 / 120s / 3 | `config.yaml:156-159` |
| 写作 | 续传 | 启用（pipeline_tasks 表，600s 陈旧可重领） | `write_pipeline_state.py:113,123-172` |
| 文件 | 读/写大小上限 | 50MB / 10MB | `config.yaml:162-164` |
| 工具 | 大结果 offload 阈值 | 15000 字符 | `registry.py:19` |
| 数据库 | SQLite WAL / busy_timeout | 30s | `db.py` |
| API | 重试 | LLM 3 次 / 2-3s；HyDE 2 次 | `agent_loop.py:164-187`, `search.py:596` |

---

## 附录 A：最复杂/代码量最大的 3 个模块

| 排名 | 模块 | 路径 | 代码量（近似，实测行号上限） | 复杂度特征 |
|---|---|---|---|---|
| 1 | **API 网关 + SSE** | `src/api/main.py` | 1056 行 | SSE 双队列桥、事件驱动 flush、断连保活、20+ 端点、cancel/权限/jobs/sessions/citations/context-detail |
| 2 | **ExpertGraph（supervisor 编排）** | `src/graph/expert_graph.py` | 1253 行 | ReAct 循环、工具路由、预算守卫、熔断、引用装配、skill 注入、direct-write 分类、收尾统一 |
| 3 | **SessionManager（持久层）** | `src/session/manager.py` | 979 行 | 消息/证据账本/checkpoint/权限/反馈/会话管理的全量 SQLite 逻辑 |

次高：`src/core/write_pipeline.py`（1476 行，Plan-Execute 长文流水线）、`src/tools/search.py`（演讲管道的多源工具+HyDE）、`src/core/agent_runner.py`（835 行，子代理 ReAct）。

## 附录 B：硬编码领域词汇（评估领域解耦程度）

| 词汇 | 出现形态 | 位置示例 | 语义 |
|---|---|---|---|
| `citrus` | 工具名 `citrus_rag_search`、`_CITRUS_KEYWORDS`、"CitrusQA/1.0" UA、`citrus_literature_v2`(pipeline1) | `config.yaml:251,258,183`、`search.py:217-250`、`pipeline1/config.py:35` | 柑橘领域核心词，渗入工具命名/UA/检索过滤 |
| `UCR` | 批次来源类型 `ucr`、`CROSS_DOMAIN`、前端徽标 `[UCR]` | `multi_retriever.py:341-356`、`evidence.py:130-138`、`search.py:243-255` | 品种库来源标识（v8.15 起），与 rag/web/historical 并列 |
| `GSRA` | config 注释 "Graph-Structured Retrieval Agent" | `config.yaml:1` | 产品代号（仅注释） |
| `HLB/CLas/黄龙病/溃疡病` | 语义词典 | `search.py:217-250`、`pipeline1/config.py:65-73`、`agent_loop.py` 提示词 | 柑橘疾病术语，进入检索关键词库 |
| `variety/cultivar/品种/品系/种质` | 品种意图检测关键词 | `search.py:232-250` | 领域路由启发式 |
| `pipeline1` / `citrus_literature_v2` | Qdrant collection 名 | `pipeline1/config.py:35` | 管道名/集合名领域化 |
| `["RAG","UCR","Web","历史"]` | 证据来源徽标 | `evidence.py:130-138` | 领域展示层 |

**领域解耦程度**：领域词主要出现在 (1) 工具命名 `citrus_rag_search`、(2) `_CITRUS_KEYWORDS`/品种意图/黄龙病等检索启发式词典、(3) metadata/source_type 判定（UCR）、(4) prompt 内容。核心算法层（BM25/RRF/rerank/压缩/记忆）领域无关。**向通用领域迁移需替换点**：`search.py:217-250`、`multi_retriever.py:341-356`、`evidence.py:130-138`、`config.yaml` 关键词相关段。

## 附录 C：遗留/未使用/被注释的旧机制

| # | 现象 | 证据 | 说明 |
|---|---|---|---|
| 1 | **~~`_web_streak_step` / `web_unavailable`~~（v9.3.0 已删除）** | 见上方 v9.3.0 修订块 5 | v8.15.3 联网熔断状态机，v9.1 联网移出 retrieve-agent 后生产路径不再触发；v9.3.0 按论文对齐指令整链删除（函数 + `web_unavailable` 参数 + 专测锚点） |
| 2 | **~~`academic_search` / `fetch_fulltext` 工具体保留但默认禁用~~（v9.3.0 已整链删除）** | 见上方 v9.3.0 修订块 2 | 学术 4 源 API（Semantic Scholar/CrossRef/PubMed/OpenAlex）与 fulltext.py 已按论文对齐指令全链删除，注册表/白名单/配置/API 字段同步清理 |
| 3 | **`web_search.probe_enabled` / websearch_probe / thinking_probe** | `config.yaml:41`、`src/tools/websearch_probe.py`、`src/tools/thinking_probe.py` | 探测脚本（`--run` 手动执行），非运行路径；`probe_enabled` 在代码中无消费点 |
| 4 | **旧 supervisor 工具 `call_retrieve_agent` 废弃防御** | `expert_graph.py:225-233` | 只返回 `[ERR_DEPRECATED]` 引导；schema 已不在 `supervisor_tools.py` |
| 5 | **草稿机制全链删除（v8.17.15）** | `main.py:431-432`、`expert_graph.py:439-440`、`light_graph.py:104-106`、`search_both.py:177-179`（均为删除注释） | 代码已无 draft_worker/draft_answer；仅注释残留说明 |
| 6 | **~~`config.py` VERSION="8.14.1" 与分支不一致~~（v9.3.0 已统一）** | `config.py:146` VERSION="9.3.0"（单源） | v9.3.0 论文对齐：版本号统一 9.3.0（config/index.html/run.ps1/pack_release.ps1/README/api v2 同步） |
| 7 | **~~`tool_tags` 配置段~~（v9.3.0 已删除）** | 无（v9.3.0 移除） | 原 yaml 段 `config.py`/`registry.py` 未消费（grep 无引用）；随 P2#8 删除 |
| 8 | **`feature_flags: {}` / sandbox 默认配置** | `config.yaml:122,183-192` | 框架保留，无业务开关接入 |
| 9 | **~~`_corpus_fingerprint` 兜底 `?`~~（v9.3.0 已改抛 ValueError）** | `search.py:284-297` | v9.3.0：语料指纹异常不再降级 "?"（防缓存污染），改为抛 `ValueError` 并记录日志 |
| 10 | **`light_graph` 无历史引用注入（与 expert 不一致）** | `light_graph.py:342-343`（注释"原 v8.4.6 F2 行为已移除"） | 设计决策而非死代码；但意味着 light 侧栏无 historical 组 |
| 11 | **`websearch_probe.websearch` 未注册** | `src/tools/websearch_probe.py` 未在 `tools/__init__.py` 注册 | 探测脚本非工具 |
| 12 | **"RIGS / Graph-Structured Retrieval Agent" 仅注释** | `config.yaml:1` 注释；`src/` 全树 grep 零命中 | 概念名残留于配置文件头注释，代码无实现痕跡——论文中若提及只能作为未来工作 |

---

*审计完成。所有行号基于本工作区当前快照（HEAD `2f3c07a`），如需针对特定模块深度展开（如 write_pipeline 1476 行、agent_runner 835 行、expert_graph 工具轮 610-880 行等），可另行分卷。*