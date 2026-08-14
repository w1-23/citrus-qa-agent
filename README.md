# 🍊 Citrus QA Agent

**本地运行的科研问答 Agent**——基于 DeepSeek V4 Flash 的检索增强问答与文献综述写作助手。

启动后浏览器打开本地地址即可使用（WebUI），无需配置任何服务端环境变量：
**首次打开页面时在界面中填写你的 DeepSeek API Key 即可**，Key 仅保存在本机，不会上传、不会写入项目配置文件。

## ✨ 功能特性

| 能力 | 说明 |
|---|---|
| 🔍 **本地 RAG 检索** | 语料库向量检索 + HyDE 假设生成 + 交叉编码重排，科研问题保证至少一次检索 |
| 📚 **学术搜索** | CrossRef / Semantic Scholar / PubMed / OpenAlex 多源检索，结构化证据回执（全文进上下文） |
| 💬 **流式回答 + 思维链** | 回答逐 token 上屏；DeepSeek 深度思考（reasoning）实时展示在「🧠 深度思考」折叠块 |
| 📝 **文献综述写作** | 自动规划大纲（「📋 执行计划」展示）→ 并行逐章撰写 → 引用统一校验 → 草稿-发布原子落盘 |
| ⚡ **双模式** | 专家模式（完整 ReAct + 多子 Agent 检索/写作/分析）/ 轻量模式（预检索直答，更快） |
| 🛑 **随时停止/改问** | 输入有误可点击「⏹ 停止」中断当前任务，修改后重新发送 |
| 🔐 **权限分级** | auto_workspace（工作区写文件免询问）/ ask（危险操作弹卡片审批）/ deny，WebUI 一键切换 |
| 🧠 **上下文管理** | 1M token 预算、软/硬阈值压缩、KV 缓存静态前缀、历史会话持久化（刷新不丢失） |
| 📊 **证据可追溯** | 回答引用编号 ↔ 侧栏文献卡片一一对应，区分「文献确证」与「模型推演」 |

## 🚀 快速开始

### 方式一：一键运行（推荐，零配置）

从 [Releases](https://github.com/w1-23/citrus-qa-agent/releases) 下载发布包
`citrus-qa-agent-v8.5.0.zip`，解压后**双击运行 `run.ps1`**（或右键 → 使用 PowerShell 运行）：

```
解压 → 运行 run.ps1 → 自动完成 → 浏览器自动打开 http://localhost:8000 → 页面填 API Key
```

`run.ps1` 全自动处理：

| 步骤 | 行为 |
|---|---|
| 1. Python | 未安装则自动 `winget install Python 3.11` |
| 2. 虚拟环境 | 自动创建 `agent/.venv`（仅首次） |
| 3. 依赖 | 自动 `pip install -r requirements.txt`（仅首次，5-10 分钟） |
| 4. 模型 | 自动下载向量编码模型 + 导出重排模型到本地缓存（仅首次，5-15 分钟；之后秒级启动） |
| 5. 启动 | 启动服务并自动打开浏览器 |

> 首次等待较长是因为安装依赖和下载模型（向量编码/重排约 2GB）；一次完成后下次启动秒级。
> 也可选择 `-full` 完整包（含模型缓存，约 2.5GB），下载后跳过第 4 步直接运行。

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

- **左侧**：模式切换（专家/轻量）+ 文献引用面板
- **中间**：对话区（流式回答、思维链折叠块、执行计划、执行日志）
- **右上角**：权限模式徽标（点击切换）、上下文占用环（点击看明细）、主题/清空/新会话
- **输入框**：发送 ⇄ 停止同一位置切换

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
│   │   ├── retrieval/          # 向量检索 + 重排（Qdrant/BM25/混合）
│   │   ├── prompts/            # 全部提示词（静态前缀 + 动态块）
│   │   └── guardrails/         # 记忆 / 提示注入消毒 / 日志脱敏
│   ├── tests/                  # 103 个回归测试
│   └── workspace/output/       # 写作成果输出目录（运行时生成）
├── requirements.txt
└── README.md
```

## 📚 添加自己的语料库

检索器启动时自动扫描 `agent/data/` 下的所有批次目录（无需改配置、无需重新打包）。每个批次目录的约定结构：

```
agent/data/
└── 我的批次名/            # 任意命名，自动被发现
    ├── chunks/
    │   └── chunks.jsonl   # 分块文本（每行一条：{text, doi?, title?, ...}）
    ├── qdrant_data/       # Qdrant 本地向量库（用你习惯的方式写入向量）
    └── metadata.json      # 批次元数据（可选）
```

- `data/` 属于你的私有知识库：**不进 Git 仓库、不进发布包**（`pack_release.ps1` 与 `.gitignore` 均已排除）
- 新增/更新批次后**重启服务即生效**；检索与写作、引用、证据回执全链路自动适配
- 检索阈值可在 `agent/config.yaml` 的 `retrieval:` 段调整（相似度下限、动态阈值比例等）

## 🔒 安全设计

- **API Key**：只存本机 `state/api_key`（权限 600），API 响应永不回传 Key，仓库不含任何密钥
- **沙箱**：写文件仅限 `workspace/output`（可切换审批/拒绝模式）；读取仅限项目内路径
- **提示注入防护**：检索证据块 `<evidence>` 标签隔离 + "数据边界"声明；日志邮箱/手机/身份证/密钥自动脱敏
- **隐私**：全本地运行，除 DeepSeek API 调用外无任何外部请求

## 🧪 测试

```bash
cd agent
python -m pytest tests -q
```

## 📄 许可

MIT License
