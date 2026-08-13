# 模型路由矩阵（MODEL_ROUTING）

节点 → 模型 / 温度 / 超时 / 输出上限 一览（v8.4，2026-08-13 修订）。
原则：**推理与生成用 MAIN（deepseek-v4-flash, temp 0.2），辅助分类/压缩用 FAST（temp 0）**。

## 主链路

| 节点 | 位置 | 模型 | 温度 | max_tokens | 超时 | 重试 |
|---|---|---|---|---|---|---|
| Expert supervisor（ReAct 循环） | `graph/expert_graph.py` supervisor_node | MAIN | 0.2 | 32768 | 120s | 3 次/轮 |
| Light supervisor | `graph/light_graph.py` `_build_light_llm` | MAIN | 0.2 | 16384 | 60s | 3 次/轮 |
| 子 Agent（retrieve/write/analyze） | `core/agent_runner.py` run_agent | MAIN | 0.2 | write:12000 / 其他:4096 | 120s(入参) | 3 次/轮 |
| 强制收尾（max turns/熔断后） | expert/light/agent_runner | MAIN | 0.2 | 同主链 | 同主链 | —（v8.4 保持带工具客户端，载荷结构一致） |

## 辅助节点（FAST）

| 节点 | 位置 | 模型 | 温度 | max_tokens | 超时 | 说明 |
|---|---|---|---|---|---|---|
| 搜索建议 gen_suggestions | `core/context_manager.py` `_get_fast_llm` | FAST | 0 | — | 12s | 输出 JSON 数组 |
| 输出格式 format_hint | 同上 | FAST | 0 | — | 12s | 单字分类 |
| 历史压缩 compact_messages | `core/context_manager.py` `_get_compact_llm` | **FAST** | 0 | 800(实传) | 30s | v8.4 修订 |
| HyDE 假想答案 | `tools/search.py` | FAST | 0.2 | 300 | **3s** | 超时即降级基础检索 |
| LTM 事实提取 extract_key_facts | `guardrails/memory.py` | FAST | 0 | 1000 | 15s | v8.4 转后台 spawn，不阻塞响应 |

## 例外与注意

- **历史压缩 v8.4 修订**（原 AG-3 决策为 MAIN）：改回 **FAST**。理由：压缩现在
  是"发送视图构建"的高频操作（512K 视图预算，软阈值 0.75 批量触发），摘要质量由
  保留优先级提示（实体/数值/决策/标识符）保证，无需 main 模型；且压缩提示已
  上下文感知（带 query + prior_summary 增量整合）。
- **路由兜底 `_resolve_mode`**（AG-12）：纯规则（关键词/长度），无 LLM 调用。
- `settings.FAST_MODEL` 默认与 MAIN 相同（deepseek-v4-flash），可在 `.env` 用 `FAST_MODEL` 覆盖为更便宜的模型。
- 修改任一节点模型/温度后，回归：`test_batch1.py` + `test_batch2.py` + 一次真实问答。
