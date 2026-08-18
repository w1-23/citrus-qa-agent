# 🍊 Citrus QA Agent

**本地运行的科研问答 Agent**——基于 DeepSeek V4 Flash 的检索增强问答与文献综述写作助手。

启动后浏览器打开本地地址即可使用（WebUI），无需配置任何服务端环境变量：
**首次打开页面时在界面中填写你的 DeepSeek API Key 即可**，Key 仅保存在本机，不会上传、不会写入项目配置文件。

## ✨ 功能特性

| 能力 | 说明 |
|---|---|
| 🔍 **本地 RAG 检索** | 语料库向量检索（LanceDB）+ HyDE 假设生成 + 交叉编码重排，科研问题保证至少一次检索 |
| 📚 **学术搜索** | CrossRef / Semantic Scholar / PubMed / OpenAlex 多源检索（默认 crossref+pubmed，可全开），跨源 DOI/标题去重，结构化证据回执 |
| 💬 **流式回答 + 深度思考** | 回答逐 token 上屏；DeepSeek 深度思考（reasoning）实时展示并**持久化**——退出会话再回来仍可展开查看思考过程 |
| ⚙️ **执行过程透明** | 工具调用实时记录（⚙ 每步可见、日志可展开）；执行完成保留「工具执行 N 项」折叠摘要；输入框执行动态光环提示任务进行中 |
| 📝 **文献综述写作** | 自动规划大纲（回答气泡内「执行计划」折叠块）→ 逐章撰写（每章 300-3000 字）→ 引用统一校验 → 草稿-发布原子落盘 |
| ⚡ **双模式** | 专家模式（完整 ReAct + 多子 Agent 检索/写作/分析）/ 轻量模式（预检索直答，更快） |
| 🛑 **随时停止/改问** | 输入有误可点击「⏹ 停止」中断当前任务，修改后重新发送 |
| 👍 **回答反馈** | 回答**完整生成后**才显示 👍/👎 一键反馈（附"哪里好/哪里不好"评论），记录在本机 |
| 🔐 **权限分级（图标化）** | 🔓 全权（工作区免询问）/ 🛡️ 询问（危险操作弹卡片审批）/ 🔒 只读——SVG 线条图标，输入框左下角一键切换，即时生效 |
| 🧠 **上下文管理** | 1M token 预算、软/硬阈值压缩、KV 缓存静态前缀、历史会话持久化；「上下文细分」面板逐段查看 system/历史/记忆原文 |
| 📊 **证据可追溯** | 回答引用编号 ↔ 侧栏文献卡片一一对应（刷新/切换会话自动恢复历史引用），区分「文献确证」与「模型推演」 |
| 📁 **会话管理** | 侧栏会话列表（自动创建/切换/重命名/删除/首问自动命名），手风琴并列收缩 + 展开状态记忆 |

## 🚀 快速开始

### 方式一：一键运行（推荐，零配置）

从 [Releases](https://github.com/w1-23/citrus-qa-agent/releases) 下载主包，解压后**双击运行 `run.ps1`**（或右键 → 使用 PowerShell 运行）：

```
下载主包 → 解压 → 运行 run.ps1 → 首次自动下载语料 + 模型 → 浏览器自动打开 http://localhost:8000 → 页面填 API Key
```

发布包：

| 包 | 大小 | 说明 |
|---|---|---|
| `citrus-qa-agent-v8.13.0.zip` | ~2.3 MB | **必下**：代码 + 一键脚本 |
| `corpus-v8.13.0-1.zip` | ~940 MB | 语料分卷 1/3：xrz、dxy-1、1-50、51-101 |
| `corpus-v8.13.0-2.zip` | ~800 MB | 语料分卷 2/3：1-1200、7.20 |
| `corpus-v8.13.0-3.zip` | ~510 MB | 语料分卷 3/3：720600 |

> 语料 = **公开文献 7 批次**（老 5 批 + 新增 xrz / 1-1200，LanceDB 向量库 + chunks.jsonl），因单附件超 GitHub 2GiB 上限拆为 3 个分卷。**无需手动下载**：`run.ps1` 首次运行自动按序号下载**全部分卷**并解压合并到 `agent/data/`，之后秒级跳过；也可手动下载后全部解压到项目根目录（自动合并）直接运行。

**模型自动安装**：向量编码（multilingual-e5-large）与重排（bge-reranker-v2-m3）模型不打包（重排模型单文件超 GitHub 2GB 上限）——首次运行 `run.ps1` 自动经 **HuggingFace 国内镜像（hf-mirror.com）** 下载，约 5-15 分钟，一次完成后秒级启动；也可手动运行 `python prepare_models.py`（`--skip-reranker` 可跳过重排模型）。

`run.ps1` 全自动处理：

| 步骤 | 行为 |
|---|---|
| 1. 语料 | 检测 `agent/data/lancedb`，缺失则自动从 GitHub Releases 下载（仅首次，~1.05GB） |
| 2. Python | 未安装则自动 `winget install Python 3.11` |
| 3. 虚拟环境 | 自动创建 `agent/.venv`（仅首次） |
| 4. 依赖 | 自动 `pip install -r requirements.txt`（仅首次，5-10 分钟） |
| 5. 模型 | 自动下载向量编码模型 + 导出重排模型到本地缓存（仅首次，5-15 分钟；之后秒级启动） |
| 6. 启动 | 启动服务并自动打开浏览器 |

> 首次等待较长是因为下载语料和安装依赖/模型（语料 ~1.05GB，依赖+模型约 2GB）；一次完成后下次启动秒级。
> 也可选择 `-full` 完整包（含模型缓存 + 语料，约 3.3GB，需自行打包），下载后跳过步骤 1/5 直接运行。

### 部署给其他人

对部署者的要求：**Windows 电脑 + 能上网**，无需安装任何东西（Python 都会自动装）。

1. 下载 [citrus-qa-agent-v8.13.0.zip](https://github.com/w1-23/citrus-qa-agent/releases)（~2.3MB）→ 解压到任意目录
2. **双击 `run.ps1`**（或右键 → 使用 PowerShell 运行）
3. 首次运行全自动完成（约 20-40 分钟，取决于网络）：
   - 语料自动下载（~1.05GB，从 GitHub Releases）
   - Python 检测/安装 → 虚拟环境 → 依赖安装
   - 模型经 hf-mirror 国内镜像自动下载
4. 浏览器自动打开 `http://localhost:8000` → 页面填写 DeepSeek API Key → 开始使用

**GPU 自动适配（v8.13-b5b）**：`run.ps1` 检测到独立显卡会自动安装 **DirectML 版 onnxruntime**——嵌入/重排模型放进**显存**（不占系统内存、速度更快）；无独显则自动用 CPU。使用者无需任何手动选择。

**模型与缓存可跨机器复用**：`agent/.hf_cache/` 里的模型权重文件和 BM25 语料缓存**与硬件无关**，可整目录拷贝到新机器复用（省去每台重新下载）；如需重新下载也默认走 hf-mirror 国内镜像。语料（`agent/data/`）同样可整体拷贝分发。

> 注：本仓库 .ps1 脚本按 Windows PowerShell 5.1 兼容编写（UTF-8 带 BOM），任何编辑器编辑后请保持该编码，否则双击运行会乱码报错。

**常见问题（FAQ）**

- **语料下载慢/失败？** 先设置环境变量 `GH_MIRROR`（如 `https://ghproxy.net/`）再运行 run.ps1；或手动下载 `corpus-v8.13.0.zip` 解压到项目根目录（zip 内含 `agent/data/...`）后直接运行。
- **模型下载慢？** 已默认走国内镜像 hf-mirror.com；如需官方源，注释 run.ps1 中 `$env:HF_ENDPOINT` 一行。
- **首次等了很久正常吗？** 正常——语料 1.05GB + 依赖 + 模型约 2GB，总计首次约 20-40 分钟；一次完成后下次启动秒级。
- **怎么停止？** 直接关闭 PowerShell 窗口，或运行 `stop.ps1`。
- **怎么更新版本？** 下载新主包解压**覆盖旧目录**（`agent/data/` 与 `agent/state/` 自动保留，会话与语料不丢），重新运行 run.ps1 即可。
- **换电脑/多人使用？** 每台机器独立部署；语料数据共用时直接拷贝 `agent/data/` 目录（LanceDB 无单实例锁限制，可多实例并行）。
- **不同硬件的电脑能共用一份模型文件吗？** 能——`.hf_cache/` 模型文件与 BM25 缓存与硬件无关，可整体拷贝共用；onnxruntime 的 CPU/GPU 版本由 `run.ps1` 按显卡自动选择安装（有独显→DirectML 进显存，无独显→CPU）。

### 停止服务

```powershell
# 方式一（推荐）：项目根目录运行
.\stop.ps1

# 方式二：手动结束
#   PowerShell: Stop-Process -Id (Get-NetTCPConnection -LocalPort 8000).OwningProcess -Force
#   或任务管理器结束对应 python 进程
```

> ✅ v8.9 起向量库为 **LanceDB 嵌入式存储**（`agent/data/lancedb`）：无单实例锁限制、支持热更新，可放心重启/多实例。

### 方式二：手动安装（开发/自定义）

### 环境要求

- Python 3.11+
- Windows / macOS / Linux 均可

### 安装

```bash
git clone https://github.com/w1-23/citrus-qa-agent.git
cd citrus-qa-agent
python -m venv .venv
# Windows: .venv\Scripts\activate    macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
cd agent
python prepare_models.py        # 预下载模型（向量编码 + 重排，首次）
```

### 启动

现有 conda 环境（项目运行时环境 `rag-agent`，Python 3.11）：

```bash
conda activate rag-agent
cd agent
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
```

或使用项目自带虚拟环境：

```bash
cd agent
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```

打开浏览器访问 **http://localhost:8000**。

### 首次使用（30 秒）

1. 页面自动弹出 **API Key 配置卡片**
2. 填入你的 DeepSeek API Key（`sk-` 开头，[platform.deepseek.com](https://platform.deepseek.com) → API Keys 创建）
3. 点击「保存并开始使用」→ 立即进入聊天界面

> 模型固定为 **deepseek-v4-flash**（已针对该模型调优全部提示词与参数）。Key 保存于本机 `agent/state/api_key`，跨重启保留；更换 Key 只需删除该文件后刷新页面。

## 🖥️ 界面速览

- **左侧**：会话区（列表/切换/新建 ＋/重命名/删除，手风琴展开即懒加载）+ 工作区（写作成果可点开）+ 文献引用（历史引用自动恢复），手风琴并列收缩、状态记忆
- **中间**：对话区（用户气泡右侧 / 回答气泡左侧；气泡内自上而下：工具执行摘要 → 工具日志 → 深度思考 → 回答正文；回答开头 3-5 条要点）
- **右上角**：上下文占用环（点击看「上下文细分」——system/历史消息/记忆块逐段查看原文）、主题切换
- **输入框**：左下角权限图标（🔓/🛡️/🔒 点击切换，即时生效）；右下角发送 ⇄ 停止 + **执行动态光环**（任务进行中旋转提示）；权限审批卡片在输入框上方弹出

## ⚙️ 配置（可选）

所有可调参数在 `agent/config.yaml`，常用项：

```yaml
permission:
  mode: auto_workspace      # ask=危险操作弹卡片审批 | deny=一律拒绝
  wait_sec: 90              # ask 模式审批超时

context_budget:
  max_tokens: 1000000       # 上下文预算（=模型窗口）
  soft_threshold: 0.75      # 软阈值：批量压缩
  hard_threshold: 0.93      # 硬阈值：强制收尾

pipeline:
  parallel_sections: 3      # 写作并行章数（1=串行）
```

## 📁 项目结构

```
citrus-qa-agent/
├── agent/
│   ├── index.html              # 单文件 WebUI（聊天/引用/上下文/审批卡片）
│   ├── config.yaml             # 全部可调配置
│   ├── src/
│   │   ├── api/main.py         # FastAPI 网关（SSE 流式 + REST）
│   │   ├── core/               # agent_runner / context_budget / stream_llm / jobs / ...
│   │   ├── graph/              # expert_graph / light_graph（LangGraph 状态机）
│   │   ├── session/            # 会话持久化（SQLite）/ 权限授权
│   │   ├── tools/              # 检索 / 读文件 / 统计 / 写文件（沙箱化）
│   │   ├── retrieval/          # 向量检索 + 重排（LanceDB/BM25/混合）
│   │   ├── prompts/            # 全部提示词（静态前缀 + 动态块）
│   │   └── guardrails/         # 记忆 / 提示注入消毒 / 日志脱敏
│   ├── tests/                  # 150 个回归测试
│   └── workspace/output/       # 写作成果输出目录（运行时生成）
├── requirements.txt
└── README.md
```

## 📚 语料库：自带示例 + 导入自己的文献

### 内置示例语料

`-full-data` 完整包内置**示例语料**（公开下载的科研文献，已编码为向量）——开箱即可测试检索、引用、综述写作全链路。示例语料位于 `agent/data/`，可随时删除或替换。

### 向量后端（v8.13-b5b 起默认 lancedb）

`config.yaml` 的 `retrieval.backend` 已显式设为 `lancedb`——LanceDB 嵌入式向量库（百万级、热更新、无锁）统一承载新旧数据包。可手动改为 `qdrant`（旧后端兼容）或 `auto`（检测到 `data/lancedb/` 有表则用 lancedb，否则回退 Qdrant local）。

> v8.9 已内置 LanceDB 语料（示例语料包即 LanceDB 格式），无需再迁移；历史 Qdrant 数据迁移脚本已随 v8.10 清理（见 git 历史 `agent/migrate_qdrant_to_lancedb.py`）。

### 导入自己的文献

使用内置导入工具（在 `agent/` 目录）：

```bash
# 把 PDF/txt/md 放进 agent/data/import/，然后：
python ingest.py                        # 自动分块 → 向量化 → 写入批次
python ingest.py --dir 我的文献目录 --batch mycorpus
python ingest.py --backend lancedb     # 显式指定后端（默认 auto）
```

导入后**重启服务即生效**（LanceDB 后端同一连接 add 后即查即得）。检索器自动扫描 `agent/data/` 下所有批次目录：

```
agent/data/
├── 批次名/                # 任意命名，自动被发现
│   ├── chunks/chunks.jsonl   # 旧数据包：分块文本在 chunks/ 子目录
│   ├── chunks.jsonl          # 新数据包（pipeline1 系）：分块文本在批次根目录
│   ├── qdrant_data/          # Qdrant 向量库（打包产物，建 LanceDB 表时复用）
│   └── metadata.json
└── lancedb/              # LanceDB 向量库（检索数据源，表名=批次名）
```

### 新数据包（pipeline 系预分块文献包）入库

**标准流程（三步）**：

1. **放包**：把新数据包放进 `agent/data/批次名/`（根目录 `chunks.jsonl` + `qdrant_data/` + `metadata.json`；任意命名，自动被发现）
2. **建表**：`reindex_lance.py --batch 批次名`（在 `agent/` 目录、用项目 Python 环境执行）
3. **重启服务**：restart 后检索器自动装载新 lance 表 + 重建/取用新 BM25 指纹缓存，新批次即可检索

```bash
cd agent
rag-agent\python.exe reindex_lance.py --batch 我的新批次          # 单个包（推荐）
rag-agent\python.exe reindex_lance.py --batch 1-1200 --no-qdrant   # qdrant 不可用时全量重嵌入
rag-agent\python.exe reindex_lance.py --all                        # 一次性重建全部批次
```

**新数据如何匹配**（`reindex_lance.py` 的换算逻辑）：

- 以 `(paper_id, chunk_index)` 为全局键（与检索器 `global_chunks` 一致）对齐「包内 qdrant 向量」与「chunks.jsonl 文本」
- **键命中** → 直接复用包内既有向量（免重新嵌入，分钟级）；**键缺失**（qdrant 损缺/新块）→ 用本地嵌入模型**小批量补嵌入**（64/批 + 逐批 gc，防止内存爆）
- qdrant 里**重复键只留一份**、**chunks.jsonl 里已不存在的孤立点自动丢弃** → 生成的 LanceDB 表与 `chunks.jsonl` **严格 1:1**（表行数 = 块数，每块必有向量）
- 修正/替换某批次内容后：重新对同批次执行 `reindex_lance.py --batch 该批` 即可（自动删旧表重建）

**注意**：① 新包入库/重索引期间**先停止服务**（模型双份会吃满内存，曾在 16GB 机器上死机）；② 建表后必须**重启服务**才生效（`--reload` 只监听 .py 代码，不会因 data/ 目录变化自动重载）；③ 完成后可在 UI 直接提问验证，或用 `e2e_check.py` 做多查询核对（health 轮询 + SSE 答案/引用解析）。

- `data/` 属于你的私有知识库：**不进 Git 仓库、不进主包**（`pack_release.ps1` 与 `.gitignore` 均已排除；公开示例语料经 `corpus` 附件分发——`run.ps1` 自动下载，或 `-full-data` 打包附带）
- 检索阈值可在 `agent/config.yaml` 的 `retrieval:` 段调整（相似度下限、动态阈值比例等）

## 🔒 安全设计

- **API Key**：只存本机 `state/api_key`（权限 600），API 响应永不回传 Key，仓库不含任何密钥
- **沙箱**：写文件仅限 `workspace/output`（可切换审批/拒绝模式）；读取仅限项目内路径
- **提示注入防护**：检索证据块 `<evidence>` 标签隔离 + "数据边界"声明；日志邮箱/手机/身份证/密钥自动脱敏
- **隐私**：全本地运行，除 DeepSeek API 调用外无任何外部请求
- **凭据红线（v8.13-b5b）**：任何 GitHub/API token 永不写入 `.git/config`、仓库文件、脚本或日志；remote 地址保持纯 `https://...` 形式；推送/拉取凭据只走系统凭据管理器（Git Credential Manager）或用户本人在提示框输入——token 不落盘、不复制、不转发
- **启动性能**：BM25 倒排索引按语料指纹持久化缓存（`.hf_cache/bm25/`），语料不变时启动跳过重建

## 🧪 测试

```bash
cd agent
python -m pytest tests -q
```

> 提示：若本机安装了 anndata 等自带 pytest 插件的包导致插件加载失败，可加 `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` 运行。

## 🛠️ 开发辅助

```bash
cd agent
python -m src.prompts.snapshot   # 渲染提示词快照到 src/prompts/snapshots/（提示词变更后可 diff 审查）
```

## 📄 许可

MIT License
