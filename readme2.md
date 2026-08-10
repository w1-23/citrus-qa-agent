# Citrus QA Agent v8.3.0 — 架构、数据流、配置手册

> 2026-07-29 | v8.3.0 | Light/Expert 统一为 LLM 自主路由 Supervisor | HyDE 混合检索 | opencode 风格前端

---

## 一、目录结构

```
agent/
├── config.yaml              # 所有可配置参数（检索权重、HyDE开关、阈值）
├── index.html               # 前端 SPA（opencode 风格：单行状态 + 答案流式）
├── src/
│   ├── config.py            # pydantic Settings（70+字段，优先级 env > YAML > default）
│   ├── logger.py            # 日志配置
│   ├── api/main.py          # FastAPI + SSE 事件流 + 心跳
│   ├── core/
│   │   ├── progress_bus.py     # 事件总线 + 结构化发射器 + 工具心跳追踪
│   │   ├── registries.py       # Agent/Skill 注册表（精简后）
│   │   ├── agent_runner.py     # 子Agent ReAct 执行器（1轮）
│   │   ├── context_manager.py  # 上下文加载 + HumanMessage 组装
│   │   ├── context_budget.py   # Token 预算（上限 1M）
│   │   ├── skill_tree.py       # 384 写作技能语义匹配
│   │   ├── strategy_cards.py   # 策略卡片
│   │   └── pipeline_utils.py   # 检索上下文构建
│   ├── graph/
│   │   ├── graph.py            # 根调度器 build_graph(mode)
│   │   ├── state.py            # AgentState TypedDict
│   │   ├── light_graph.py      # Light Supervisor（2 轮, 1 工具）
│   │   └── expert_graph.py     # Expert Supervisor（4 轮, 5 工具）
│   ├── tools/
│   │   ├── __init__.py         # 工具注册
│   │   ├── search.py           # RAG + HyDE + academic_search + pdf_read
│   │   ├── readfile.py         # read_local_file（PDF/CSV/XLSX/TXT/MD）
│   │   ├── file_ops.py         # write_local_file
│   │   ├── analyze.py          # statistical_analysis + experimental_design
│   │   └── registry.py         # PartitionedToolNode（并发分组）
│   ├── retrieval/
│   │   ├── multi_retriever.py  # Qdrant多批次 + search_hyde双通道
│   │   ├── bm25.py             # BM25+ 中文分词 + 加权RRF
│   │   └── init.py             # 启动预热
│   ├── engine/                 # Embedder + Reranker ONNX
│   ├── guardrails/             # Memory + HistoryCompactor
│   └── session/manager.py      # SQLite 会话持久化
├── data/                       # Qdrant 5批次 119K chunks
├── TUNE_PARAMS.md              # 可优化参数清单（50+ 参数）
└── AGENT_CHANGES.md            # v8.3 全量代码变更对照
```

---

## 二、架构总览

```
                   ┌─────────────────────────────────┐
                   │          POST /api/v2/chat       │
                   │     SSE: event + data JSON       │
                   └───────────────┬─────────────────┘
                                   │
                         build_graph(mode)
                                   │
                   ┌───────────────┴───────────────┐
                   ▼                               ▼
          ┌──────────────────┐          ┌────────────────────────┐
          │   LIGHT GRAPH    │          │    EXPERT GRAPH        │
          │  (2 节点)         │          │   (2 节点)             │
          │                  │          │                        │
          │ load_context     │          │ expert_load            │
          │     ↓            │          │     ↓                  │
          │ light_supervisor │          │ supervisor (ReAct ×4)  │
          │  LLM bind_tools  │          │   ├─ call_retrieve_agent│
          │  自主决定调用      │          │   ├─ call_write_agent  │
          │  1 工具, 2 轮    │          │   ├─ call_analyze_agent│
          │     ↓            │          │   ├─ read_local_file   │
          │ save_context     │          │   └─ pdf_read          │
          └──────────────────┘          │     ↓                  │
                                        │ expert_save            │
                                        └────────────────────────┘
```

---

## 三、检索管线（HyDE Hybrid）

```
用户 query: "ruby基因的调控网络"
       │
       ├─ HyDE: fast LLM (~2s) → 英文假想答案
       │   "The Ruby gene encodes a MYB transcription factor..."
       │
       ├─ Dense: embed(original_query) + embed(hyde_answer)
       │        → Qdrant × 5 batches (ThreadPoolExecutor)
       │
       ├─ BM25: original_query → 全局索引
       │
       ├─ RRF: 三路加权融合 (config.yaml 可调)
       │   weights = {orig_dense: 1.0, hyde_dense: 1.0, bm25: 1.0}
       │
       ├─ Reranker: cross-encoder(original_query, chunks)
       │
       └─ Dynamic threshold → top_k results
```

**HyDE 回退**：生成失败 / <30 字 / 开关关闭 → 回退原检索 `rag.search(query)`

---

## 四、上下文管理

```
ContextManager.load(session_id, query, mode)
    ├─ session_manager.get_messages(sid) → SQLite
    ├─ ContextBudget.check(msgs) → 1M token上限
    │    <90%: NORMAL | 90-95%: SUMMARIZE | ≥95%: TRUNCATE
    ├─ memory_store.recall_long_term_memory(query) → 向量匹配
    └─ _generate_hints(query)
         ├─ fast_model → search_suggestions (英文搜索角度)
         └─ fast_model → format_hint (fact/compare/method/...)

LoadedContext → build_human_message() → XML 块 → LLM
```

---

## 五、SSE 事件协议

| 事件类型 | 含义 | 数据结构 |
|---------|------|---------|
| `thinking` | LLM 推理内容 | `{content: str}` |
| `tool_call_start` | 工具调用开始 | `{tool_name, args, tool_call_id}` |
| `tool_executing` | 工具执行进度（心跳2s） | `{message, tool_name, tool_call_id}` |
| `tool_result` | 工具执行结果 | `{tool_name, output, summary, is_error}` |
| `text` | 流式答案片段 | `{content: str}` |
| `citations` | 文献引用 | `{cited: [...], uncited: []}` |
| `context_status` | 上下文概览 | `{estimated_tokens, max_tokens,...}` |
| `done` | 请求完成 | `{session_id, answer, gen_time_ms}` |
| `heartbeat` | SSE 保活 | `{}` |

---

## 六、前端 v8.3（opencode 风格）

```
┌─ Header ─────────────────────────────────┐
│ [C] Citrus QA Agent v8.3   ⊙ 清空 新会话  │
├────────────┬──────────────────────────────┤
│ 文献引用    │  ⚙ citrus_rag_search 执行中..│  ← 单行瞬时状态
│ [1] xxx    │                              │
│ [2] yyy    │  Ruby基因是MYB转录因子家族...  │  ← markdown 答案流式
│            │                              │
│            │  ⏱ 45.9s                     │
└────────────┴──────────────────────────────┘
```

**特性**：
- 执行过程只显示 1 行瞬时状态指示器（💭/⚙/◆/✍）
- 工具完成后状态消失，不留痕迹
- 只有最终答案持久显示
- 侧边栏保留文献引用（含 DOI、标题、摘要）

---

## 七、工具总览（7 个）

| 工具 | 类型 | 调用者 | 联网 |
|------|------|--------|:---:|
| `citrus_rag_search` | 本地RAG | Light supervisor + retrieve-agent | ❌ |
| `academic_search` | CrossRef | retrieve-agent | ✅ |
| `read_local_file` | 文件读取 | supervisor 直接调用 | ❌ |
| `pdf_read` | 学术PDF提取 | supervisor 直接调用 | ❌ |
| `write_local_file` | 文件写入 | write-agent | ❌ |
| `statistical_analysis` | 统计分析 | analyze-agent | ❌ |
| `experimental_design` | 实验设计 | analyze-agent | ❌ |

---

## 八、Agent 轮次

| Agent | 最大轮次 | 绑定工具 |
|-------|:---:|------|
| **Light supervisor** | 2 | citrus_rag_search ×1 |
| **Expert supervisor** | 4 | call_retrieve/write/analyze + read_local_file + pdf_read |
| retrieve-agent | 1 | citrus_rag_search + academic_search |
| write-agent | 6 | write_local_file |
| analyze-agent | 2 | statistical_analysis + experimental_design |

---

## 九、目录层面变更（v8.1.1 → v8.3.0）

| 操作 | 文件 |
|------|------|
| 删除 | `retrieval/query_expansion.py`（QueryExpander 从未使用） |
| 删除 | `tools/search.py` web_search 函数 + WebSearchProvider 类 |
| 新增 | `core/progress_bus.py` 结构化事件总线 |
| 新增 | `TUNE_PARAMS.md` 可优化参数清单 |
| 新增 | `AGENT_CHANGES.md` 代码变更对照 |
| 删除 | `index.html` 网络来源侧栏 + msg-part/proc-card CSS/JS |
| 重写 | `index.html` sendMsg 函数（900行→150行） |
| 重写 | `light_graph.py` 4节点→2节点（删除硬编码检索+ReAct fallback） |

---

## 十、关键配置可调参数

```yaml
retrieval:
  rrf_weights: {orig_dense: 1.0, hyde_dense: 1.0, bm25: 1.0}
  rag_hyde_enabled: true   # HyDE 开关
  rerank_threshold: 0.25   # HyDE 后需重新校准
context_budget:
  max_tokens: 1000000       # DeepSeek v4 Pro 1M 上限
tools:
  max_tool_calls: 4
# 详见 TUNE_PARAMS.md（50+ 可调参数）
```
