# AgentLoop 基座：扩展与新增 Agent 指南（v8.13-b5）

本文说明「加工具 / 加子 Agent / 加 supervisor」在 AgentLoop 收敛后的姿势，
以及 `src/core/agent_loop.py` 里可复用的共享原语。

## 1. 共享原语（`src/core/agent_loop.py`）

三层 ReAct（专家 supervisor / 轻量 supervisor / 子 Agent）此前各自复制了一套：
工具调用 id 提取、DOI 去重、LLM 重试、强制收尾与空答兜底。现已收敛为：

| 原语 | 用途 | 三层差异点（参数表达） |
|---|---|---|
| `tc_id(tc)` | 工具调用 id 提取（dict/对象两种形态兼容，缺 id 兜底 uuid） | 无差异（各层 `extract_tc_id` 别名导入，避免与局部变量 `tc_id` 撞名） |
| `count_unique_docs(rows)` | 按 DOI 去重计数（无 DOI 按条计） | 无差异 |
| `last_message_content(msgs, mode)` | 收尾空答兜底 | `aimessage`（专家）/ `any`（轻量）/ `nonsystem`（子 Agent） |
| `invoke_llm_with_retry(invoke, *, retries, sleep_s, label, on_exhausted)` | LLM 调用重试 | 3s 失败上抛（专家/轻量）vs 2s 失败返回 None（子 Agent） |
| `force_final_answer(msgs, *, stream_call, label, fallback_mode)` | 强制收尾 | label 含 reason（专家）/ 空即不记日志（轻量） |
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

## 4. 加新 supervisor（step 2 抽出循环骨架后，模板预览）

当前（step 1）已把原语抽离；step 2 会把「轮次循环骨架」本身也抽成
`run_agent_loop`。届时加一个新图形模式（类比 expert/light）从「复制几百行循环」
退化为「填配置 + 两个钩子」：

```python
# 差异点只是配置 + 钩子，循环/重试/预算边界/截断/收尾兜底全在基座里
async def my_supervisor_node(state: AgentState) -> dict:
    messages = [...]
    return await run_agent_loop(
        AgentLoopConfig(max_turns=MY_MAX_TURNS, llm=...),   # 轮次上限 + 客户端
        execute_tools=my_execute_tools,                      # 差异：执行/预算策略
        finalize=my_force_final,                             # 差异：收尾策略
        messages=messages,
    )
```

其中 `execute_tools` / `finalize` 只需复用本文第 1 节的
`tc_id` / `count_unique_docs` / `force_final_answer` 等原语，各层只写差异点。

## 5. 保证不阻碍迭代的边界

- 工具注册表、graph 组装、`config.yaml` 的 `subagents` 声明、prompt 加载 **全部原样**；
- 原语不改变任何行为的**契约**（测试抽成对每个原语的单测，行为对齐旧实现）；
- 每步重构均以「全量测试 + 集成冒烟（SF 系列跑真实 graph）验证行为一致」为前提提交。