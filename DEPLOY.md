# Citrus QA Agent 部署指南（新服务器）

## 一、代码传输（任选其一）

**方式 A：zip 包（推荐，最简单）**
1. 在 `agent/` 目录执行 `python pack_deploy.py`，产出 `citrus_agent_deploy.zip`（仅代码+配置+prompts，1MB）
2. 拷贝 zip 到服务器，解压到目标目录，目录结构保持 `agent/...`

**方式 B：git 远程**
```bash
# 本机（首次）：推送到 GitHub/Gitee 私有仓库（.gitignore 已排除 .env/data/state，推送安全）
git remote add origin <私有仓库地址>
git push -u origin master
# 服务器：clone + 后续 pull
git clone <私有仓库地址>
git pull   # 以后更新代码用这个
```

## 二、服务器环境准备

```bash
# 1. Python 3.10+（建议 3.11）
python --version

# 2. 安装依赖（requirements.txt 已补齐 langgraph/optimum/transformers/openai）
cd agent
pip install -r requirements.txt

# 3. 如无 GPU，可强制 CPU（可选，性能会降）
#    config.yaml 中 embedder.force_cpu: true
```

## 三、运行时数据（不随代码包走，需单独拷贝）

| 目录 | 内容 | 处理 |
|---|---|---|
| `agent/data/` | 检索语料库（Qdrant + chunks，数 GB） | **必须**从原机拷贝 |
| `agent/.env` | API Key | 复制 `.env.example` 为 `.env` 并填写 `DEEPSEEK_API_KEY` |
| `agent/state/` | 会话历史 DB | 可空（首次启动自动创建）；如需保留历史则拷贝 |
| `agent/workspace/output/` | 生成文档 | 可空（自动创建） |

## 四、模型缓存（reranker + embedder）

**reranker（bge-reranker-v2-m3 ONNX）**：首次启动自动从 HuggingFace 导出到 `agent/.hf_cache/onnx_reranker`。
- 大陆网络建议把原机 `agent/.hf_cache/` 目录整体拷贝过去（约几百 MB），跳过导出。

**embedder（e5-large）**：fastembed 首次运行自动下载。大陆网络用清华镜像：
```bash
# Linux/macOS
export HF_ENDPOINT=https://hf-mirror.com
# Windows PowerShell
set HF_ENDPOINT=https://hf-mirror.com

# 预下载（一次性，之后缓存到本地，离线可用）
python -c "from fastembed import TextEmbedding; TextEmbedding('intfloat/multilingual-e5-large')"
```

## 五、启动

```bash
cd agent
python -m uvicorn src.api.main:app --host 0.0.0.0 --port 8000
```
- 前端页面：`http://<服务器IP>:8000/`
- 健康检查：`http://<服务器IP>:8000/health`

## 六、部署后验证（10 分钟）

```bash
# 1. 检索链路（AG-11 修复验收，脚本归档于 agent/tests/）
cd agent
python tests/verify_retrieval.py
# 期望: 3 个查询各返回 10 条、0 条指向 last chunk、末尾 PASS

# 2. 启动日志检查（应出现 5 行 idx_map ok）
#    [Retriever] idx_map ok: batch=1-50 ...
#    [Retriever] idx_map ok: batch=dxy-1 ...

# 3. 功能冒烟
#    问"柑橘黄龙病的病原是什么" → 回答带相关文献引用
#    问"写一份黄龙病综述并保存到 test.md" → 文件只含一份正文（AG-2 双写修复验证）
```

## 七、常见问题

| 现象 | 处理 |
|---|---|
| 启动报 `Could not load model ... e5-large` | 按第四节走清华镜像预下载 |
| 检索无结果 | 检查 `data/` 是否拷贝完整、`verify_retrieval.py` 输出 |
| `optimum` 导入失败 | `pip install optimum transformers` |
| 端口被占用 | 换端口或用 `--port` 参数 |
| 数据库锁（Qdrant .lock） | 服务退出后再启动；多实例禁止同时开（单例设计）。v8.3.4：若仍发生占用，冲突批次自动跳过向量加载、以 BM25 兜底并在启动日志 ERROR 列出失败批次；空结果回传会附"向量库部分批次不可用"提示 |
