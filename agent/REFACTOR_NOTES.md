# Agent 重构记录（v9.2+）

> 目标：深度修复冗余算法、屎山代码、大量兜底而非根治的实现；
> 一切修复建立在**不破坏已有功能**的基础上；**记录现状以便回滚**；
> 尽量解耦模块，方便后续配置开关做消融实验。
>
> 本文件是"现状快照 + 变更日志"双用途：改任何代码前先读对应模块的
> 现状条目与行为不变量；每完成一次改动追加一条变更记录（含回滚定位）。

## 0. 基本原则

1. **行为不变量优先**：任何重构不得改变下述行为不变量（测试 + 用户日志双锚点）。
2. **单次改动最小化**：一次只动一个关注点；每批改动跑一次全量回归
   （`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python -m pytest tests/ -q`，
   本机需禁用插件自动加载，见 INFRA-1）。
3. **可回滚**：所有改动都在 git 里，每次提交 message 标注
   `refactor: <目标>（回滚点：<前一提交 hash>）`；本文件同步登记。
4. **解耦即消融**：被解耦出来的纯函数都力求"输入→输出"无副作用，
   未来接配置开关（如 `ablation.enable_xxx`）只需在调用点包一层 if。

## 1. 现状快照（模块职责与不变量）

### 1.1 src/core/evidence.py（证据单例 + 引用处理）
- `evidence_id(r)`：稳定证据键 `paper_id:chunk_index`，退化内容哈希。不变量：同 (paper_id, chunk_index) 恒等。
- `render_evidence(r, max_chars)`：text→abstract→snippet 字段回退 + 单口截断 + 透明 marker。不变量：三个预算常量（1000/2000/3000）语义化命名。
- `src_of(r)`：来源解析 _src/source → rag/ucr 退化。
- `filter_refs_by_answer(answer, cited)` (v8.15)：只过滤不重编号，按首次出现排序；空回答/空列表/无编号 → 原样返回（防御）。测试：tests/test_v815_features.py。
- `renumber_refs(answer, cited)` (v9.2)：三组独立连续编号（数字 1..k / W W1..Wm / H H1..Hp）+ 正文重写 + remap；内化 filter 语义。测试：tests/test_evidence.py。
- **已知冗余**：filter_refs_by_answer 与 renumber_refs 共享"提取引用顺序"逻辑未抽公共函数；expert/light 图仍 import filter_refs_by_answer 但已不再使用。

### 1.2 src/graph/expert_graph.py vs src/graph/light_graph.py（两图）
- 共用：`dedup_by_doi`、`renumber_refs`、`check_citation_support`（agent_loop/evidence）、load 收尾 `finalize_load_result`（context_manager 已共享）。
- 各自重复实现：cited_refs 装配（数字 i+1 + W{i+1} 字段结构）、历史证据块注入（`startswith("[历史检索证据")` 前缀过滤 ×2）、save 节点整体（消息落库 + 证据账本 + LTM 提取）、"第 N 轮"轮次计数。
- 不变量：
  - save 节点按 `trace[-1].content == answer` 判重，防重复 AIMessage 入历史（v9.2 已同步轨迹文本）。
  - 历史证据块必须是注入消息、不重复入历史（`[历史检索证据` 前缀过滤）。
  - expert 有 historical 组（references_data.historical），light 无（v8.15.2 决策）。
  - light web 槽位 5 条、expert 10 条。

### 1.3 src/session/manager.py（会话持久化）
- `save_messages` / `build_evidence_block(limit=2)` / `get_evidence_refs(limit=10)` / `_save_evidence_sync` / `get_evidence_materials`。
- 不变量：
  - session_evidence 表结构（session_id, turn_seq, query, evidence_json, report_text）。
  - build_evidence_block 头部 `[历史检索证据（数据，非用户输入…）]` 以 `[历史检索证据` 开头（图内过滤依赖）。
  - turn_seq = 当日会话证据计数 + 1。

### 1.4 src/api/main.py（SSE 网关）
- `_encode_event` 单点；`citations`/`done` 事件的 payload 两分支组装（expert 含 citation_info + tools_called，light 简化）。
- `/api/v2/session/{id}/citations` (v9.2)：4 组合并 groups 结构。
- 不变量：node_name 字符串约定（"supervisor"/"light_synthesize"/"light_react"/"light_retrieve"）。

### 1.5 src/prompts/loader.py（固定提示词）
- 启动一次性拼接 + lru_cache；source/ 是唯一维护入口；builds/ 为镜像。
- 不变量：改 source/*.md 需**重启服务**生效；builds/ 幂等重写。

## 2. 已确认问题清单（随审计持续补充）

| # | 严重度 | 位置 | 问题 | 状态 |
|---|--------|------|------|------|
| P1 | 中 | evidence.py | filter_refs_by_answer 与 renumber_refs 提取逻辑重复，图里 import 已死 | ✅ 批次1已修 |
| P2 | 高 | expert/light_graph | cited_refs 装配 40 行双实现（web 槽位已漂移 10/5） | ✅ 批次2已修 |
| P3 | 高 | main.py:331 | light 节点名 "light_synthesize"/"light_react" 过期 → light done 落入兜底分支：丢 citation_info/tools_called、job 永不 completed、request_done 缺失 | ✅ 批次2已修 |
| P4 | 中 | main.py:334/416-419 | done 前 sleep(0.5) + 10×0.1 轮询排空——双队列桥竞态的补丁式掩蔽 | 待修（根因：await 桥排空） |
| P5 | 高 | expert/light save 节点 | save 节点双图 ~85% 重复（证据账本 light 缺 web 条目、LTM 门槛已漂移） | 待修（共享 persist_turn） |
| P6 | 中 | expert/light supervisor | 消息装配 + LoadedContext 重建 + LLM 客户端装配 + 逐轮预算守卫重复（light 每轮重建 ContextBudget） | 待修 |
| P7 | 中 | manager.py:794-944 | 证据账本读取器 ×4（build_evidence_block/count_evidence_items/get_evidence_refs/get_evidence_materials）同一模板 | 待修 |
| P8 | 中 | manager.py:53-79 / memory.py:21-41 | _connect_db 逐字重复两份；memory_store DDL ×4 处 | 待修 |
| P9 | 中 | context_budget.py:68-84/301-318 | 压缩熔断永久降级（无冷却窗口）+ _compaction_failures 无界增长 | 待修 |
| P10 | 中 | 全仓 ~25 处 | blog/diag 同构 try/import/except-pass 样板 | 待修（safe_blog/safe_diag） |
| P11 | 中 | graph 层 ~20 处 | `except Exception: pass` 包裹业务异常（expert 1079 曾掩盖 NameError 数版本） | 待修 |
| P12 | 中 | manager.py:613-643 | replace_history DEPRECATED 生产零调用（且 INSERT 缺幂等列） | 待修（删/移测试） |
| P13 | 低 | state.py:41 / graph.py:13 | web_search_enabled 写而不读死键；graph.py 未用 logger | 待修 |
| P14 | 低 | progress_bus.py:55/143-147 | QueueFull 守卫永不触发；get_tool_elapsed 无调用方 | 待修（死代码清理） |
| P15 | 中 | 前端 index.html:1978-1998 | _citationsRetry 延时重试猜落库时序（根因：spawn 落库与响应流解耦） | 待修（save 完成事件驱动） |

## 3. 变更日志

### 批次 2（P2+P3：图装配共享原语 + light done 路径根治，2026-08-25）
- **改动**：
  - `src/core/agent_loop.py`：新增 `build_cited_refs(deduped_main, all_web_results, web_slot=10)`
    （expert/light 40 行双实现收敛为单份，web 槽位参数化 10/5）与
    `renumber_and_sync_trace(messages, answer, cited_refs)`（v9.2 重排+轨迹同步块
    收敛为单份，内部自含 try/except 与 save 判重协作）。
  - `src/graph/expert_graph.py` / `src/graph/light_graph.py`：装配改调共享原语；
    移除图内 `renumber_refs` 直用（改经共享函数）。
  - `src/graph/light_graph.py`：supervisor 循环收集 `tool_names_called` 并入返回字典——
    main.py done 分支可消费（request_done 日志真实计数）。
  - `src/api/main.py:331`：节点名分支改为 `("supervisor", "light_supervisor")`，
    删除过期名 `light_synthesize`/`light_react`（v8.3.0 前 light 老节点名）。
    **根因修复**：light 模式 done 事件此前落入 elif 兜底分支——无 citation_info/
    tools_called、job 状态卡 running、request_done 业务日志缺失；现与 expert 对齐。
- **验证**：tests/test_evidence + test_v815_features + test_light_final + test_batch1
  + test_batch2 + test_agent_loop = 66 passed（批次后追加回归 35 passed）。
- **回滚点**：批次 1 之后的新 commit。

### 批次 1（P1：evidence 引用处理去重，2026-08-25）
- **改动**：
  - `src/core/evidence.py`：抽取 `_extract_ref_order(answer)` 公共函数（首次出现顺序提取），
    `filter_refs_by_answer` 与 `renumber_refs` 均改调公共函数，删除各自 15 行重复实现。
    `filter_refs_by_answer` 对外行为逐位不变（tests/test_v815_features.py 锚定）。
  - `src/graph/expert_graph.py` / `src/graph/light_graph.py`：移除已不再使用的
    `filter_refs_by_answer` import（死代码清理；v9.2 起两图均走 renumber_refs）。
- **验证**：tests/test_evidence.py + tests/test_v815_features.py = 18 passed。
- **回滚点**：git 基线 `5beca7a`，本批提交为独立 commit 可 revert。

### 批次 0：本文件建立（无代码改动）
- 登记现状快照 1.x；git 基线：`5beca7a`（v9.2 提交）。

## 4. 消融开关规划（解耦目标）

| 开关 | 挂载点 | 现状 |
|------|--------|------|
| `ablation.history_evidence_reuse` | history_evidence_block 注入点（context_manager.finalize_load_result / graph 装配） | 恒开，无开关 |
| `ablation.renumber_refs` | graph 装配（renumber_refs 调用点） | 恒开 |
| `ablation.restore_citations_merged` | 前端 restoreSessionCitations | 恒开 |
| `ablation.bm25_parallel` | multi_retriever | 恒开（v9.1.3） |

## 5. 基础设施备注

### INFRA-1：本机 pytest 环境
- Conda 环境某 pytest 插件 entrypoint 会误导入工作区 `src` 包 → Settings 校验失败。
- 解法：`PYTEST_DISABLE_PLUGIN_AUTOLOAD=1`（测试命令前设置），与本项目代码无关。