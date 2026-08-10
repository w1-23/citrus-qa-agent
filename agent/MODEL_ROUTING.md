# 模型路由矩阵（MODEL_ROUTING）

节点 → 模型 / 温度 / 超时 / 输出上限 一览（v8.3.1，2026-08-11 固化）。
原则：**推理与生成用 MAIN（deepseek-v4-flash, temp 0.2），辅助分类/压缩辅助用 FAST（temp 0）**；
历史压缩例外——触发频率低，质量优先，已切换 MAIN。

## 主链路

| 节点 | 位置 | 模型 | 温度 | max_tokens | 超时 | 重试 |
|---|---|---|---|---|---|---|
| Expert supervisor（ReAct 循环） | `graph/expert_graph.py:411` | MAIN | 0.2 | 32768 | 120s | 3 次/轮 |
| Light supervisor | `graph/light_graph.py:33` | MAIN | 0.2 | 16384 | 60s | 3 次/轮 |
| 子 Agent（retrieve/write/analyze） | `core/agent_runner.py:98` | MAIN | 0.2 | write:32768 / 其他:4096 | 120s(入参) | 3 次/轮 |
| 强制收尾（max turns 后） | `expert_graph.py:568` / `light_graph.py:240` | MAIN | 0.2 | 同主链 | 同主链 | — |

## 辅助节点（FAST）

| 节点 | 位置 | 模型 | 温度 | max_tokens | 超时 | 说明 |
|---|---|---|---|---|---|---|
| 搜索建议 gen_suggestions | `core/context_manager.py` `_get_fast_llm` | FAST | 0 | — | 12s | 输出 JSON 数组 |
| 输出格式 format_hint | 同上 | FAST | 0 | — | 12s | 单字分类 |
| HyDE 假想答案 | `tools/search.py:321` | FAST | 0.2 | 300 | **3s** | 超时即降级基础检索 |
| LTM 事实提取 extract_key_facts | `guardrails/memory.py:238` | FAST | 0 | 1000 | 15s | 已改 `asyncio.to_thread`，不阻塞事件循环 |

## 例外与注意

- **历史压缩 compact_messages**（AG-3 决策）：`core/context_manager.py` `_get_compact_llm` → **MAIN**，temp 0，timeout 30s。理由：触发频率低（60% 窗口才触发），摘要质量直接影响后续推理，成本可忽略。
- **路由兜底 `_resolve_mode`**（AG-12）：纯规则（关键词/长度），无 LLM 调用。
- `settings.FAST_MODEL` 默认与 MAIN 相同（deepseek-chat），可在 `.env` 用 `FAST_MODEL` 覆盖为更便宜的模型。
- 修改任一节点模型/温度后，回归：`test_batch1.py` + `test_batch2.py` + 一次真实问答。
