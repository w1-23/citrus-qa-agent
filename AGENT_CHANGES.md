# Citrus QA Agent 变更记录

> 本文件为项目变更日志（UTF-8）。历史版本（v8.1.1–v8.3.2）的逐文件对照内容因编码事故（GBK 误转 UTF-8）已不可恢复，此处以架构级变更摘要替代；代码级细节以 git 历史为准。

---

## v8.3.3（2026-08-12）— 写作流水线修复 + 安全加固

### P0 修复
- **综述写作首跑必崩**：`write_pipeline.run_write_pipeline` 的 `plan` 变量仅在断点续传命中时初始化，非续传路径触发 `UnboundLocalError`（实测日志 `[Error: cannot access local variable 'plan']`）→ 前置初始化 + 显式分支。
- **运行时单章容量兜底**：Plan 校验期的 3000 字上限在 Execute 期无对应检测，API 截断无法感知 → 实际输出超限时触发一次"精简重写"，仍超限则写盘并标记 `truncated_sections`。

### Plan-Execute 完整性（对齐 Orchestrator-Workers 惯例）
- **断点续传原子化**：`pipeline_tasks` 读-改-写包 `BEGIN IMMEDIATE` 事务 + WAL + busy_timeout；修复 `cleanup_stale_tasks` 无日期过滤（原实现会清掉全部任务）。
- **写后 read-back 校验**：每章写盘后回读确认章节落盘（`write_local_file` 失败返回错误字符串不抛异常，原实现静默当成功）；标题容错匹配。
- **引用完整性校验**：正文 `[n]` 编号 vs 参考文献区编号比对，缺失/未用文献在结果中提示。
- **react_fallback 落盘回退**：Plan 失败时带大纲一次性生成全文并写盘（原只回传 500 字提示，supervisor 收到后无实质产出）。
- react 分支改为结构化交接（原为误导性文本）。

### 安全与健壮性
- **沙箱 fail-closed**：删除对不存在的 `engine.sandbox` 的 ImportError 吞错，内置模式检查（危险工具类拒绝，注册表只读/分析类放行），异常时拒绝执行。
- **绝对路径读白名单**：`read_local_file` / `statistical_analysis` 仅允许项目根目录内（可配 `file_io.read_extra_roots` 扩展）。
- **前端 XSS**：marked 渲染前转义（`renderMarkdownSafe`）+ 引用卡片/状态栏字段转义。
- **SSE 错误脱敏**：前端只收"内部错误"提示，完整异常进日志。
- **CORS**：移除 `*` + credentials 非法组合。

### 工程化
- **context_usage 增量推送**：`emit_usage_delta`（按 session+source），前端改为累加 delta（原累加单次调用累计值，多轮重复计数）。
- **per-request 事件队列**：contextvars 绑定，SSE 并发会话互不串扰（原全局队列 + 每请求 reset 清他人事件）。
- **request_id 追踪**：contextvar 贯穿请求链路，日志注入 `req=` 字段。
- **配置接线**：supervisor/light/subagents 轮次上限、recursion_limit 全部读取 config.yaml（原硬编码且与配置脱节）；启动时 `validate_config()` 自检。
- **死代码清理**：删除 `CircuitBreaker`（零引用）、`ToolCallAccumulator`（unused）、死配置字段（`MAX_TOOL_CALLS`/`LLM_TIMEOUT`/`INTENT_*`/`AGENT_MAX_TURNS`/`PIPELINE_DEFAULT_TARGET_CHARS` 等）。
- **部署包**：zip 纳入根 `requirements.txt` 与 `verify_retrieval.py`（原缺失导致 DEPLOY.md 步骤不可执行）。
- **版本号统一**：/health、app、HTML、CLI → 8.3.3。

### 测试
- `test_write_pipeline.py` 33 → 65 项：入口级（四路+续传+回退）、state 原子操作、read-back、引用校验、容量兜底。
- 全量回归 199 项全绿（batch1 20 / batch2 37 / batch3 14 / direct_write 12 / file_saved 19 / optimization 32 / write_pipeline 65）。

---

## v8.3.2（2026-08-11）— Write Pipeline Plan-Execute

- 写任务四路路由（`classify_write_task`）：direct_write / react / plan_execute / modify。
- 结构化 Plan（write-plan.md，严格 JSON）+ 动态校验（指定字数严格 / 自主模式宽松）+ 单章容量上限（≤3000 字防截断）+ 重试 + ReAct 回退。
- 逐章独立 LLM 生成（write-section.md + `<summary>` running_context），材料按 refs 语义分配，缺章降级占位。
- 断点续传（pipeline_tasks 表）、SSE 进度事件（plan_ready / section_start / section_done）、modify 定向修改（章节定位/替换/追加）。

## v8.3.1（2026-08-11）— 生产就绪度修复（16 项）

- 职责矩阵重构：删 forced save / file_saved 信号（零补偿写入）；supervisor 直写 `write_local_file`；retrieve-agent 3 轮 + 轮次语义；light 加 `read_local_file`。
- 综述被覆盖根因：`agent_runner` file_saved 判定 dict/getattr bug；删 encyclopedia_search 死工具。
- 统一回传协议：RAG 空结果归因（THRESHOLD_BLOCKED / NO_MATCH + 建议）、academic 网络失败 `[ERR_NETWORK]`、`_classify_error` 全补操作建议。
- 性能：min_keep=3 打断空转、write-agent 12000 分块防截断、write 返回内容预览防重写。
- 上下文 token 面板实时更新（context_usage 真实用量推送）；检索性能优化与防截断。
- 内容预览缺陷修复（append 显示本轮新增块）；`simulate_flow.py` 模拟验证。

## v8.3.0（2026-07-26）— 结构化 SSE + 上下文管理

- 结构化事件（thinking / tool_call_start / tool_executing / tool_result / text）。
- ContextManager / ContextBudget（1M 窗口，soft 0.60 / hard 0.93 压缩截断）。
- 长期记忆（LTM）事实提取 + 置信度衰减；Fast Guard 问候语短路。

## v8.1.1（基线）

- 双图（Light/Expert）+ HyDE+RRF 混合检索 + 学术搜索（CrossRef/PubMed）+ 实验设计/统计分析工具。
