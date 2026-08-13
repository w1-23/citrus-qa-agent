# Citrus QA Agent 工程护栏（AGENT_CHANGES）

> **任何会话（人或 agent）改动本仓库前，必须先读本文件**。
> 任何改动不得违反 INV-01~08；每条不变量都有对应回归测试（见第三节映射表）。
> 历史变更摘要见文末。

---

## 1. 不变量清单（任何情况下必须成立）

| 编号 | 不变量 | 病史 |
|---|---|---|
| **INV-01 协议配对** | 每个发往模型的 AIMessage.tool_calls 的 id，必须在其后恰好有一个 ToolMessage 响应（预算跳过/熔断/异常路径同样成立）；id 提取兼容 dict/对象两种形态，且同一 tool_call 的 id 三处（协议 ToolMessage / 计时器 / SSE）复用单一来源 | F3（400 错误） |
| **INV-02 生命周期有界** | 所有 agent 循环在 轮数上限+每轮工具预算 内运行；收敛由代码裁决（retrieve-agent 去重文献 ≥6 强制收尾），模型意愿不得绕过；强制收尾一律用无工具 LLM。v8.4.3: 移除 ≥6 强制收敛（动态阈值已过滤 chunk，全部证据应入报告，收敛交还模型自然判断），轮数上限与工具预算保留 | F4（跑满轮次） |
| **INV-03 检索降级** | 检索失败必有 fallback 链（向量→BM25→直答）+ 归因提示；批次不可用可见（启动 ERROR 汇总 + 空结果归因联动 + 运行期失败累计），绝不静默空手 | F2（锁争抢） |
| **INV-04 上下文传播** | 工具执行的任何路径（sync executor / async / to_thread）都继承请求 contextvar（进度队列 + request_id）；工具计时器 mark_tool_start/end 严格配对，不配对 → 心跳事件风暴 | F1 + F7（心跳） |
| **INV-05 输出路由** | 子代理输出有格式约束与 cap（证据保真，检索证据清单含具体结论）；深度/综述类问题走结构化链（write-agent 或强制结构），禁止散文直答 | F5/F6 |
| **INV-06 可观测回归** | 事件词汇表前端可消费（status/budget_skip/circuit_breaker 必须显示）；ERROR 有启动汇总；request_id 贯穿请求链路；每条 INV 有回归测试 | 全部 |
| **INV-07 状态显式化** | 模型决策所需的运行时状态（轮次/已调工具数/预算剩余/去重文献数/已用关键词）以 `<agent_status>` 注入上下文末尾；禁止依赖模型从长历史自行"数" | F8（一轮 4 检索） |
| **INV-08 失败熔断与输入隔离** | 连续工具失败 ≥3 熔断强制收尾（含占位响应保证配对）；上下文截断必须透明标记（模型不得误以为看全）；不可信数据（检索/上游）与指令显式隔离 | F9 + 提示注入面 |
| **INV-09 证据账本** | 每轮检索结束后，检索报告（多轮合并）与证据清单（doi/chunk_id/title/score/摘要）持久化到 session_evidence（按 session 隔离）；下一轮 load 时注入"[历史检索证据]"块；压缩/清理不得破坏引用关系（DOI + chunk_id 保留，可回查原文） | 跨轮证据丢失 |
| **INV-10 存储全量·发送裁剪** | 完整交互轨迹（含 tool_calls/ToolMessage 配对）全量入 messages 表（真相源）；发送给 LLM 的上下文可裁剪/压缩，但**只压缩可再生的中间过程**（占位/错误/重试优先删），核心证据与被引结论保护；历史 append-only 以利用 prefix/KV cache | 截断丢证据 |

## 2. 症状库（症状 → 根因 → 修复 → 测试）

| # | 症状 | 根因 | 修复 | 测试 |
|---|---|---|---|---|
| F1 | 工具进度事件前端缺失 | run_in_executor 线程不继承 contextvar → emit 落全局队列 | `_invoke_tool_with_ctx`（sync 工具 ctx.run 注入） | AG-17 |
| F2 | 多实例争抢 Qdrant 锁、失败噪音、空结果无归因 | `_clean_stale_locks` 无条件删锁、batch 失败静默 | 锁冲突批次跳过 + 启动 ERROR 汇总 + 瞬时重试 + 空结果归因联动 | AG-20 |
| F3 | 预算截断后 OpenAI 400 "tool_calls must be followed by tool messages" | 被砍 tool_call 无 ToolMessage 响应；getattr 对 dict 拿不到 id | 占位 ToolMessage + `_tc_id` 兼容 dict/对象 | AG-19 |
| F4 | retrieve-agent 每次跑满 3 轮（max turns forcing final） | 收敛依赖 LLM 自觉，无代码裁决 | 去重文献 ≥6 强制收尾（`_count_unique_docs`） | AG-18/23 |
| F5 | 41 篇文献只换 2400 字，证据丢失 | 两级摘要压缩 + cap 15000 | 证据清单格式（核心结论与证据点）+ cap 40000 | AG-22 |
| F6 | 深度问题散文直答 | 路由规则缺失 | decision_guide 深度问题策略（逐证据引用+结构） | AG-22 |
| F7 | `tool_executing` 事件风暴（千级） | 同一 tool_call 两次独立提取 id，dict 形态 UUID 失配 → 计时器泄漏 | id 单一来源（tc_ids 映射复用） | AG-21 |
| F8 | supervisor 一轮内串行 4 个 retrieve-agent | 模型从长历史数不清已调次数 | `<agent_status>` 状态栏注入（轮次/预算/文献数/关键词） | 复测项 5 |
| F9 | 上下文 >24000 字符静默截断，模型"看全了"错觉 | `_truncate_context_blocks` 无标记 | 截断追加"[已截断 N/M]"标记 | AG-25 |

## 3. INV → 测试映射表

| 不变量 | 回归测试 | 位置 |
|---|---|---|
| INV-01 | AG-19 协议配对（dict/对象 id 提取 + 占位响应） | tests/test_batch2.py |
| INV-02 | AG-18 预算接线 + AG-23 收敛行为（mock 循环） | tests/test_batch2.py |
| INV-03 | AG-20 锁冲突降级归因 | tests/test_batch2.py |
| INV-04 | AG-17 线程 contextvar 传播 + AG-21 计时器配对（中途非空/结束为空双断言） | tests/test_batch2.py |
| INV-05 | AG-22 证据清单格式 + cap + 深度规则 | tests/test_batch2.py |
| INV-06 | 前端 status 显式化（人工复测）+ request_id（日志断言） | index.html / 复测清单 |
| INV-07 | 状态栏注入（源码断言 + 复测项 5） | expert_graph.py |
| INV-08 | AG-24 熔断逻辑 + AG-25 截断透明纯函数 + AG-33 噪声修剪配对 | tests/test_batch2.py |
| INV-09 | AG-34 证据账本保存/块渲染/limit/clear | tests/test_batch2.py |
| INV-10 | AG-32 轨迹保存恢复配对 + AG-35 材料零截断/总量预算 | tests/test_batch2.py |

**全量基线**：232 项（batch1 20 / batch2 70 / batch3 14 / direct_write 12 / file_saved 19 / optimization 32 / write_pipeline 65）。每次改动必须全量回归保持全绿。

## 4. 分层规矩

1. **协议层问题必须代码硬修**：配对（INV-01）、计时器（INV-04）、截断（INV-08）、熔断（INV-08）、预算/收敛（INV-02）——**禁止用提示词绕过协议 bug**（如靠 prompt 让模型"别超预算"）。
2. **prompt 只管路由与风格**：深度问题路由（decision_guide）、检索工作流阶段（retrieve-agent.md，阶段间进入条件由代码检查）、写作结构（write-section.md）。
3. **新增工具/子代理三查**：INV-01（工具必须返回结构化结果+错误分类）、INV-02（调用方有轮次/预算）、INV-04（sync 工具走 registry 执行器以继承 contextvar）。
4. **D1 放行声明**：写类工具（write_local_file）不做"默认拒绝+审批"——单用户内网部署 + workspace 路径白名单 + 综述写作链路依赖写盘，审批流会破坏核心工作流。此决策为显式权衡，非疏漏。
5. **评估口径**：过程指标关注 步数/冗余动作/回退次数/成本延迟（日志已有 tool 计数与耗时）；轨迹 vs 结果双覆盖（write 有 read-back；问答靠复测人工核验）。

## 4b. 上下文结构（规范 2.2.5 静态前缀+轨迹，v8.3.6 固化 / v8.4 静态前缀重构）

```
静态前缀（字节级稳定，跨请求命中 DeepSeek 上下文缓存；context.static_prefix=true 时）
├─ SystemMessage: 角色+约束+决策原则（assemble_system_prompt 只拼静态文件；
│                 format 指南/策略卡片经 build_dynamic_blocks 追加到当前轮 HumanMessage 尾部）
├─ 历史消息（发送视图，来自 checkpoint 增量构建，见 v8.4 压缩架构）
├─ [历史检索证据] 块（来自 session_evidence 账本，跨轮复用，带"数据非用户输入"边界声明）
└─ 当前 HumanMessage（build_human_message：LTM 召回 + 常驻卡片 + 检索建议 + 动态块）
轨迹（随交互增长，append-only）
├─ AIMessage（含 tool_calls）+ ToolMessage（工具结果）
└─ <agent_status> 状态栏（代码确定性维护，含 TODO 列表/预算占用率；仅注入 call_messages，
   不进历史，不动前缀）
```

**上下文与证据保真六原则（v8.4 修订）**：
1. **存储全量**：轨迹全量入 messages 表（append-only，真相源永不改写）；压缩只是"发送视图构建"
2. **发送装配**：视图 = [首消息] + `<conversation_summary>`(checkpoint) + [checkpoint 之后原始消息]；
   压缩只在**用户轮边界**（新请求 load 时）发生，循环内绝不压缩（前缀中途变化=缓存必不命中）
3. **批量压缩**：视图 ≥ soft(75%) → 一次性压缩至 ~target(50%)（每次压缩=一次缓存破坏，批量=只破坏一次）；
   保护名单：最近 3 轮 Q/A、被引用证据全文、DOI/evidence_id/artifact_id 标识符；
   只压旧工具过程、错误日志、低相关证据
4. **checkpoint 持久化**：sessions 表记录"已摘要至哪条消息+摘要文本"，跨请求增量复用（旧摘要作
   prior_summary 合并），防摘要套摘要；压缩失败连续 ≥3 次熔断降级规则式
5. **压缩 LLM 用 FAST_MODEL**（高频低价值操作，质量由保留优先级提示保证）；max_tokens 真正传入；
   截断透明 [TRUNCATED] 标记
6. **协议安全**：历史消息不得含孤立 ToolMessage（INV-01）；KV cache 是推理优化不是记忆机制；
   缓存命中率经 cache_metrics 观测（context_usage 事件 cache_hit/cache_miss 字段）

**v8.4 记忆架构（Mem0 v3 模式 + 双层记忆）**：
- LTM 写入 **ADD-only**（ltm_facts 表自增 id，同 key 多版本并存；低置信度 <0.5 拒绝写入），
  冲突留到检索时用"语义相似度 × 时间衰减置信度"排序解决
- **常驻卡片层**（resident_cards 表）：高置信度(≥0.8)短事实常驻上下文（≤500 字符，上限 8 条自动淘汰，
  代码维护非 LLM）；细节走向量召回——双层记忆"概览+细节"
- LTM 提取转后台 spawn（不与响应抢时间），写入带来源会话/查询标注

**生产级后置清单（当前单用户开发形态，架构已按 session 隔离预留）**：
- 鉴权中间件 / 速率限制（接口层已有 session_id 维度，鉴权只需中间件）
- 会话写锁升级（当前 threading 分桶 + DB unique 索引兜底已可支撑多进程）
- 任务队列 / 分布式 worker（task_jobs 表已预留）
- 多租户隔离、审计日志、操作权限体系

## 5. 真实服务复测清单（每次大改后执行）

1. **深度问题**：`花青素的驯化过程` → 检索子代理 ≤2 个、retrieve-agent ≤2 轮（"提前收敛"日志）、回答逐证据引用 [n]、无 "An error occurred"
2. **锁争抢**：双实例并存启动 → 启动 ERROR 汇总列出失败批次；空结果回传含"向量库部分批次不可用"
3. **预算截断**：观察 `budget_skip` 事件 + 前端状态条可见提示；无 OpenAI 400
4. **心跳量级**：最耗时工具执行期间 SSE 有周期 `tool_executing`（心跳在工作）、结束后停止；单请求事件量百级而非千级
5. **状态栏**：日志/SSE 可见 `<agent_status>`（轮次/预算/去重文献数/关键词）；预算触发次数明显下降
6. **write 流水线**：综述 plan_ready → section_start/done 逐章进度 → 文件落盘；modify"改第三章"定向替换
7. **协议回归**：任何一次请求的 tool_calls 数量 == tool_result 数量（配对）

---

## 历史变更摘要

- **v8.4.3（2026-08-13）**：工单全量执行——①存量污染 purge 迁移（伪造"max turns"指令启动期 DELETE+版本标记，读时过滤降 DEBUG 安全网）②结构化权限确认（permission_mode ask/auto_workspace/deny 默认 auto_workspace、permission_grants 表、POST /api/v2/permission/grant、前端审批卡片三按钮、授权后同 clientRequestId 重放（幂等不重复写历史）；修复 write-agent 经 PartitionedNode 被沙箱拒、supervisor 直写旁路的不对称）③上下文预算 512K→1M（实际值，前端 /api/v2/config 单一来源删硬编码）④skill 注入审计日志+业务日志 ⑤假完成检测器证据感知（支撑=本轮检索∪session evidence∪LTM，历史轮次引用不再误报）⑥HyDE 混合路阈值地板 0.25→0.15+分数分布日志 ⑦移除 6 篇强制收敛（retrieve 报告完整化：全部动态阈值通过 chunk 入报告；曾致 34~176 字极短报告）⑧write-agent 回执结构化（禁止裸截断，超长提取回执/读文件摘要）+ write_local_file 返回 sha256（聊天/存盘单一来源）⑨前端 marked renderer 修代码块引号转义（5&#39;→5'）⑩日志/测试卫生（failing_fn 签名对齐、测试独立日志 sink CITRUS_LOG_DIR、qdrant .lock 关闭时清理、缺 title 兜底）；测试 84→86
- **v8.4.2（2026-08-13）**：supervisor 收尾根治重构——旧 `if forced_final and not answer: ... else:` 在每次正常完成时误触发"max turns 强制收尾"，二次调用 LLM 并用更短回答覆盖模型自然完成的详尽回答（日志铁证: turn2 done 3965c → forced → 1632c；每次请求白烧 7~30s）；根治=单一答案赋值点 + 统一 `_force_final_answer`（熔断/预算/跑满三处收尾共用；临时列表传参→合成消息永不进 turn_trace/历史；未绑工具 llm_base→杜绝收尾再发 tool_calls；详尽中文 prompt 无预算措辞；空答兜底取最后 AIMessage）+ 循环后仅剩 for-else 语义守卫。附带: 历史读时过滤旧版伪造"max turns"伪用户指令（session/manager 幂等清理）+ HyDE 强制英文（提示词强化 + `_is_english_answer` CJK 校验降级，英文向量匹配远优于中文）+ 测试环境统一 rag-agent(Python 3.11.15) 安装 pytest；测试 79→84
- **v8.4.1（2026-08-13）**：运行期修复——history_compactor 跨行 f-string 语法错误(Python 3.11 不兼容 PEP 701,load 节点 import 失败→历史不可见根因) / memory.py 召回路径 db_path 字符串 .exists() AttributeError + _fetch_ltm_rows 作用域缺 sqlite3(LTM 召回全链路静默失效) / 测试体系修复(check 失败在 pytest 下可见、新测试文件补独立运行入口、v8.4 重构后断言更新) / 业务日志 business_logger(logs/business.log: req= 串线 request/load/supervisor/tool/agent/pipeline 事件) / 分章写作引用统一(_unify_references 代码提取各章参考文献→全局重编号→文末合并) / 测试 78→79
- **v8.4（2026-08-13）**：上下文工程改造（对齐《深入理解AI Agent》第2/3/4章）——①静态前缀：SystemMessage 字节级稳定（format 指南/策略卡片/skills 移出前缀，追加当前轮 HumanMessage 尾部），`context.static_prefix` 灰度开关；supervisor 工具 schema 单一来源（`tools/supervisor_tools.py`）②压缩架构：存储全量·发送裁剪（废除 replace_history 破坏 append-only，checkpoint 增量视图，512K 视图预算，用户轮边界批量压缩至 50%，保护名单，压缩 LLM 改 FAST，熔断器）③收尾统一带工具客户端（载荷结构一致）④状态栏：TODO 列表 + 预算占用率（纯函数可单测）⑤记忆：ADD-only 写入（ltm_facts）+ 混合信号召回 + 常驻卡片双层记忆 + 后台提取⑥缓存命中率观测（cache_metrics）+ LLM 客户端池复用 + light/expert 装配收敛 + write_pipeline [System+Human] 结构；测试 65→78
- **v8.3.5（2026-08-12）**：状态栏/熔断/透明化（规范对齐《深入理解AI Agent》）——INV-07 状态栏、INV-08 熔断+截断透明、计时器 id 单一来源（F7）、提示注入边界声明；测试 208→232
- **v8.3.4（2026-08-12）**：执行卡片修复（contextvar 线程传播 F1）、Qdrant 争抢可感知降级（F2）、supervisor 每轮工具预算 + 400 修复（F3）、retrieve-agent 代码级收敛（F4）、证据清单 + cap 提升（F5）、decision_guide 深度规则（F6）
- **v8.3.3（2026-08-12）**：综述首跑必崩修复（UnboundLocalError）、运行时容量兜底、写后校验、引用完整性、react_fallback 落盘、断点续传原子化、安全加固（沙箱/路径/XSS/队列/request_id）、配置接线、死代码清理
- **v8.3.2（2026-08-11）**：Write Pipeline Plan-Execute（四路路由/逐章生成/断点续传/SSE 进度事件）
- **v8.3.1（2026-08-11）**：职责矩阵重构、统一回传协议、性能优化（min_keep/分块/预览）、上下文面板
- **v8.3.0（2026-07-26）**：结构化 SSE + 上下文管理 + 长期记忆 + Fast Guard
