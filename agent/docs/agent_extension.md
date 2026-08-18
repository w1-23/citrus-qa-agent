# AgentLoop 基座：扩展与新增 Agent 指南（v8.13-b5）

本文说明「加工具 / 加子 Agent / 加 supervisor」在 AgentLoop 收敛后的姿势，
以及 `src/core/agent_loop.py` 里可复用的共享原语。

## 1. 共享原语（`src/core/agent_loop.py`）

三层 ReAct（专家 supervisor / 轻量 supervisor / 子 Agent）此前各自复制了：
工具调用 id 提取、DOI 去重、LLM 重试、强制收尾、空答兜底与 token 上报。现已收敛为：

| 原语 | 用途 | 三层差异点（参数表达） |
|---|---|---|
| `tc_id(tc)` | 工具调用 id 提取（dict/对象两种形态兼容，缺 id 兜底 uuid） | 无差异（各层 `extract_tc_id` 别名导入，避免与局部变量 `tc_id` 撞名） |
| `count_unique_docs(rows)` | 按 DOI 去重计数（无 DOI 按条计，DOI 大小写不敏感） | 无差异 |
| `dedup_by_doi(rows)` | 按 DOI 去重文献列表（保留首次顺序；无 DOI 原样保留，**不做 lower**） | 无差异（expert/light 收尾去重共用） |
| `last_message_content(msgs, mode)` | 收尾空答兜底 | `aimessage`（专家）/ `any`（轻量）/ `nonsystem`（子 Agent） |
| `invoke_llm_with_retry(invoke, *, retries, sleep_s, label, on_exhausted)` | LLM 调用重试 | 3s 失败上抛（专家/轻量）vs 2s 失败返回 None（子 Agent） |
| `force_final_answer(msgs, *, stream_call, label, fallback_mode)` | 强制收尾 | label 含 reason（专家）/ 空即不记日志（轻量） |
| `emit_llm_usage(sid, source, resp)` | 真实 token 增量上报（含 prompt_cache 命中） | 无差异（三层 LLM 调用后各上报一次，source 各异） |
| `FINAL_ANSWER_PROMPT` | 统一收尾 prompt（专家/轻量字节一致） | 无差异 |

**设计纪律**：原语是「纯函数 / 注入式骨架」（`invoke`、`stream_call` 由调用方注入），
不写死任何一层的策略——新 Agent 想怎么执行、怎么收尾都能表达。

## 2. 加新工具 —— 姿势不变

1. `@tool` 装饰器定义工具，注册进 `src/tools` 的 `_TOOL_REGISTRY_BY_NAME`。
2. 把工具名加入对应 Agent 的工具名单（见各 graph 的 `*_TOOL_NAMES`）。

AgentLoop 只负责「执行 LLM 发出的 tool_calls」，不关心工具具体是什么，
故加工具对基座零改动。

## 3. 加新子 Agent —— 姿势不变，自动复用原语

1. `config.yaml` 的 `subagents.<name>.max_turns` 加一条。
2. `agent_runner._resolve_tool_names` 补充 `<name>` 的工具映射。
3. `prompts` 目录补该子 Agent 的 system prompt。

循环 / LLM 重试 / 检索角度去重 / 收尾兜底全部自动复用 `agent_loop`
（经 `run_agent` → `invoke_llm_with_retry` / `last_message_content`），
**无需再手写轮次循环**。

## 4. 加新 supervisor —— 两个可选姿势

最终**未引入统一 `run_agent_loop`**：实测三层 per-turn 内容（预算/熔断/写序/收敛/SSE）
差异远大于「轮次循环」本身可共享的 ~8 行，强抽一层骨架会得到巨大钩子面、更难维护。
因此收敛策略是「**共享原语 + 各层自己的循环**」：

- **expert（最常用，已瘦身）**：`supervisor_node` 拆成四个命名子过程——
  `_guard_supervisor_budget`（状态栏+预算守卫）、`_execute_supervisor_tools`（工具执行）、
  `_assemble_supervisor_answer`（收尾/引用装配），循环主体只有四步；行为由
  `test_supervisor_final.py`（SF-1/2/3…）兜底。
- **light（独立循环）**：保持自己的 `for turn` 循环，仅复用第 1 节原语；行为由
  `test_light_final.py`（LF-1/LF-2）兜底。

加一个新图形模式（supervisor）二选一：
1. 结构像 expert：把循环写成「预算守卫 → LLM → 工具执行 → 收尾装配」四步命名函数；
2. 结构像 light：直接写独立循环，per-turn 只复用第 1 节原语。

## 5. 保证不阻碍迭代的边界

- 工具注册表、graph 组装、`config.yaml` 的 `subagents` 声明、prompt 加载 **全部原样**；
- 原语不改变任何行为的**契约**（每个原语有单测，行为对齐旧实现）；
- 每步重构均以「全量测试(150) + SF/LF 集成冒烟验证行为一致」为前提提交。