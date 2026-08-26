# 论文投稿执行计划（CEA · Computers and Electronics in Agriculture, SCI Q1）

> 状态：**P1-P4 全部完成；P5 进行中（§7 实测替换/附录A/附录C/摘要/结论已落笔
> docx；§8.3 与摘要/结论最终措辞待用户裁决 query_mode 后同步）。**
> 依据：`C:\Users\Administrator\Desktop\论文投稿.docx`（已备份
> .bak_20260825.docx）+ 代码实测核对。
> 纪律：中文文件一律 write/edit（UTF-8）；PowerShell 不重写含中文文件；以最新代码为准；每项修改附文件+行号。

---

## 0. 现状盘点 — 文档声明 vs 代码实测（2026-09 会话复核）

**核心结论：论文正文（§1-10 + 附录A/B/C）与 v9.3.0 代码高度吻合；尾部「Coder 指令执行状态」表是乐观推断而非现状。**
同时发现 `experiment/`（单数）目录是一套**已执行完毕的论文实验资产**（2026-08-20 会话，A1–A6/B1–B3 全部完成），
论文文档声称"全部实验待执行 / experiments/ 框架已建"与事实不符——实际实验已完成大半，且产出论文图表源头。

### 0.1 差异表（文档声称 ✅ 已 完成 vs 实测）

| 项 | 文档声称 | 代码实测 | 处置 |
|---|---|---|---|
| P0#1 候选窗口参数化 | config.yaml 新增 candidate_window / rerank_top_k | ❌ 无键；`multi_retriever.py:600` 隐式 `TOP_K_FINAL*2` | ✅ 已参数化（v9.4，config.yaml `candidate_window: 20`） |
| P0#2 早停参数化 | config.yaml 新增 early_stop_* | ❌ 无键；`agent_runner.py:593` 硬编码 6/0.25 | ✅ 已参数化（v9.4） |
| P0#3 动态阈值参数化 | config.yaml 已有 | ✅ 真实（`config.py:77,79`；config.yaml `rerank_threshold`/`dynamic_threshold_ratio`） | 无需改 |
| P0#4 实验日志增强 | retrieval_stages 已含全部字段 | 待核（retrieval_stages 含 mode/queries/ms/候选/通过等） | 阶段 1 复核 |
| P0#5 query_mode | config.yaml 新增 query_mode | ❌ 无键；仅 `RAG_HYDE_ENABLED` 布尔 | ✅ 已参数化（v9.4，7 模式） |
| P0#6 联网引用不限量 | ✅ 已完成 | ✅ 真实（`_WEB_MAX_ITEMS=8` 已删，deepseek_web.py） | 无需改 |
| P0#7 academic 删除 | ✅ 已完成 | ✅ 真实（fulltext.py 删除、工具 9→7、grep 零残留） | 无需改 |
| P1#8 experiments/ 框架 | ✅ 8 个 run_*.py 已建 | ❌ `experiments/`（复数）不存在 | 🔄 对齐 `experiment/`（单数）既有资产，不新建空壳 |
| P1#9 eval_dataset.json 模板 | ✅ 已建 | ❌ 不存在 | 🔄 已有 `experiment/data/qa80_v2.jsonl`（85 行，80 题） |
| P1#10 死代码清理 | ✅ 已完成 | ✅ 真实（_web_streak_step/工具口已删） | 无需改 |
| 附录A 锚点 config.yaml:63-66 | 基座/嵌入/精排行号 | ❌ 已过时（v9.3.0 后 296→278 行；现实 `config.yaml:54-58`） | 阶段 5 全量校准 |
| 附录A agent_runner.py:623-632 | 早停锚点 | ❌ 实际 593（v9.4 后另有新行） | 阶段 5 校准 |
| §7 实验全部"待执行" | 8 实验待执行 | ❌ 大部分已有数据（见 §0.2 资产图） | 实验复现/对齐 + 补缺 |

### 0.2 `experiment/` 既有资产（2026-08-20 已执行，未入库、零 git 操作）

| 产物 | 对应论文环节 | 关键结论（EXECUTION_SUMMARY.md） |
|---|---|---|
| `qa80_evidence.json` (A1) | §7.2 评测集 | 80 题×zh/en 检索核验+evidence 回填；负样本 4/4 "库内无"确认 |
| `bias_quant.json/png/csv` (A2) | §5/实验3 | 30 对双语等价题；功能型 zh 显著塌陷（p=0.037，Mann-Whitney）——**发现点** |
| `rigs_replay.json/png/csv` (A3) | 实验4/5 | RIGS 同轨迹回放：run-all q=0.867 tok=10000 vs oracle q=0.867 tok=4283（-57% 成本） |
| `e5_e6_compare.json` (A4) | §8.1 | LanceDB 14.4ms vs Qdrant 91.5ms（6.4×）；DML 加速 embed 5.1×/rerank 2.9× |
| `crossdomain_probe.json` (+snapshot) (A5) | §5.1/实验3 跨域 | BRCA1/OsNRT2（EuropePMC OA，各 20 条）存活表：跨域 on-topic 双语对齐良好，zh 未现柑橘主库式塌陷 → **偏置是多因素**（诚实防御素材） |
| `figs_v2/` 10 图 (A6) | §7 各实验图表 | fig01 pass、fig02 阈值网格、fig03 BM25 权重、fig04 topk 预算、fig05 RRF k、fig06 跨语言、fig07 延迟、fig08 交叠、fig09 域外拒答、fig10 分数分布 |
| `rewrite_2x2.json/png` (B1/B2) | 实验3 缓解 | 改写 2×2：rewrite off/hyde off recover 0.750 → rewrite on + hyde on 0.967；**主线 A 核心** |
| `end2end_judge.json/csv` (B3) | 实验8 | 80 题×{改写 on/off}：改写 on 三维全面提升（+0.094/+0.096/+0.147）；20 题待人工 κ |
| `token_profile.json` | 成本 | token 画像 |

**注意**：以上结果基于「修复后流水线（英文改写 on 为正式基线）」与 v9.3.0 检索层一致（8 表 252,681 chunks）。
**存疑点**：① 结果基于实验期检索层，v9.4 参数化后需复跑/对齐确认等价；② 80 题金标准仍为"检索回填+6 题数值待人工核验"；
③ B3 的 κ 为人工字段未填；④ fig04/05 为 doc13 归档复现（CPU 全量网格不可行，provenance 已注明）。

---

## 1. 阶段 0：基线固化 ✅（已完成 2026-09-xx）

- git 提交 v9.3.0 基线 ✅ — `5331ba3`（23 文件，+648/−1305，feature/v8.17-draft-native-ucr）
- 删除 `agent/data/lancedb_backup_v9.2/` ✅（1,920.2 MB，E2E+225 回归均过）
- 临时文件清理 ✅（`_extract_submission.py`/`_submission_extract.txt`）
- **遗留**：`agent/_convert_blog_diag.py` 未跟踪保留（勿动）

## 2. 阶段 1（P0）：参数化校准 + 框架对齐 — ⏳ 编码完成

### 2.1 四项参数化（v9.4，已提交代码，含单测 `tests/test_v94_param.py` 6 项）

| 子任务 | 文件+行号 | 操作 | 验收 |
|---|---|---|---|
| ① 候选窗口 | `agent/config.yaml`（retrieval 段 `candidate_window: 20`）；`agent/src/config.py`（`CANDIDATE_WINDOW` 字段）；`agent/src/retrieval/multi_retriever.py:600` | `fused[:settings.TOP_K_FINAL*2]` → `fused[:settings.CANDIDATE_WINDOW]` | ✅ 单测：默认 20、无 *2 残留、yaml 同步 |
| ② 精排 top_k 统一 | 代码保留 `top_k_final`（config.yaml:8）；**改论文附录A** `rerank_top_k`→`top_k_final` | 零行为风险 | 阶段 5 落笔 |
| ③ 早停参数 | `agent/config.yaml`（`early_stop_min_evidence: 6`/`early_stop_new_ratio: 0.25`）；`config.py`（两字段）；`agent_runner.py:593` | 硬编码 → settings 读取 | ✅ 单测 |
| ④ query_mode | `agent/config.yaml`（`query_mode: full`）；`config.py`（`QUERY_MODE`）；`search.py` `_compose_queries` 纯函数 + 缓存键含 mode | 7 模式路由（full/raw/hyde_only/mq_only/sum_only/hyde_mq/hyde_sum），未知模式降级 full | ✅ 单测：full=9 路、raw=None 单路、bogus 降级 |

行为语义铁律：默认配置下与 v9.3.0 完全一致（candidate_window=20=10*2、early_stop 6/0.25、query_mode=full、缓存键含 mode 仅一次性冷启动）。

### 2.2 框架对齐（不新建空壳）

- **复用** `experiment/`（单数，已 gitignore）：`data/qa80_v2.jsonl`（80 题）、`data/species_catalog.txt`、`scripts/*.py`（a1-a6/b1-b3 全部可复现）、`results/*`（论文图表源头）。
- **待补**（相对论文 §7.2）：① 语料库外 ≥20 题（qa80 仅 4 负样本+6 数值）；② 时效性联网题 ≥12；③ 中英等价 60 对（qa80 已有双语文案可派生出等价对，A2 用 30 对）；④ cross-domain 探针 30 对（A5 已有 BRCA1/OsNRT2 各 20 条 shadow——混合探针方案部分已由 A5 覆盖）。
- **结论调整**：论文实验计划中"跨域验证普适性"的表述需按 A5 实测（偏置多因素、非必塌）诚实改写（阶段 5）。

## 3. 阶段 2（P0）：评测集整合

1. 将 `qa80_v2.jsonl`（85 行→80 题有效）作为语料库内底盘；补 20% 语料库外 + 10% 时效性题。
2. 从 `business.log`/用户反馈提取候选查询（`agent/data/` 下日志），人工精选 + 中英翻译（人工，非机翻）。
3. gold chunk 标注：定位 `chunks.jsonl`（8 批次 252,681 chunks）；可用 `experiment/results/qa80_evidence.json` 的 evidence_id 回填辅助。
4. schema 校验脚本（`experiment/data/` 侧新建 `validate_dataset.py`，或并入 eval_metrics）。
5. 核验 6 条数值题（q-050/051/052/053/065/066）人工核实 + 20 题 κ 人工回填。

**验收**：120 题（60 对中英等价）+ 30 跨域探针，gold 可复现定位，双人抽检一致率 ≥90%。

## 4. 阶段 3（P1）：实验脚本复跑/对齐

`experiment/scripts/` 已含完整脚本链（retrieval_common.py 为公共封装）。v9.4 参数化后：
1. 先跑 `retrieval_common`-based smoke（确认 agent 侧 settings.CANDIDATE_WINDOW 等不破坏实验脚本只读 import）。
2. 实验 1/2/4/7（管道消融/多粒度/窗口早停/阈值）多数需**新跑**：query_mode/candidate_window 扫描脚本（新增 run_granularity.py、run_param_sweep.py，可复用 retrieval_common 模式）。
3. 实验 3/5/6/8 以既有 results 为准复跑关键 subset 对齐版本；差异记录 provenance。
4. 产出落 `experiment/results/`（不 commit）。

**验收**：每实验 CSV/JSON 指标 + 结论草稿段（可直接贴 §7）。

## 5. 阶段 4（P1）：实验执行

按论文推荐顺序：实验4a（定窗口）→ 4b（定早停）→ 7（定阈值）→ 2（定查询组成）→ 1（定管道）→ 3（偏置+缓解，含混合探针）→ 5（成本质量）→ 6（延迟）→ 8（端到端+引用可信度）。

## 6. 阶段 5（P1）：论文修订

- §7 全节"预期结论"→ 实测数据（实验3 尤其要按 A5 诚实改写：偏置多因素）。
- 附录A 参数终值 + 锚点行号全量校准（config.yaml 54/57/58 等 v9.4 行号、agent_runner 早停新行号、multi_retriever 596-604）。
- 附录B（parse_hyde_structured 保持）；附录C 状态列改写。
- 参考文献真实性核查（[3] CaRT arXiv:2510.08517、[11] RAG-Fusion 等 web 检索确认）。
- 图：直接复用 `experiment/results/figs_v2/` 10 图 + fig_e5e6/fig_crossdomain/fig_judge3d 等。

## 7. 阶段 6（P2）：英文成稿与投稿物

- 全文英译（CEA 英文刊）+ 术语表；Highlights/Cover Letter/作者贡献/利益冲突/数据可用性。
- Elsevier numeric 参考文献格式；图表 caption；Editorial Manager 提交清单。

---

## 8. 风险与依赖

1. **A5 结论与论文原表述冲突**（偏置多因素 vs 论文"通用刻面"）——必须诚实改写，否则审稿防御失败。
2. **联网实验成本**：实验 5/6/8 需 DeepSeek key 预算。
3. **人工标注**：评测集 120+30、κ 回填、80 题核验——瓶颈。
4. **版本对齐**：v9.4 参数化后既有 results 需复跑子集验证等价，避免论文数据与代码版本漂移。
5. 工期粗估 5–7 周（标注与实验8 为下界约束）。

## 9. 待用户确认

- **query_mode 终值裁决（当前唯一阻塞 P5 收口）**：生产默认 full 实测
  MRR=0.256 劣于 raw=0.653（实验 1/2 双脚本互证 + 1b 归因）。方案①默认改
  raw（0.653，最优且省 LLM）；方案②multi_retriever rerank 改用
  original_query（0.584）；方案②b="原始查询入池+原始查询 rerank"双修正
  （exp1c 测量中）；方案③维持 full+论文如实声明。未裁决前不回写
  config.yaml、不改生产代码。
- stage1-5 其余决策点均已执行或由本计划覆盖（评测集整合✅、experiment/
  资产保持 ignore、跨域探针=模型知识+gold_note 标注）。