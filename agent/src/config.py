"""
Unified configuration for Citrus RAG Agent using pydantic-settings + YAML.
Priority: environment variable > YAML > pydantic-settings default
"""
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from openai import OpenAI

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_yaml_config: Dict[str, Any] = {}

def _load_yaml() -> Dict[str, Any]:
    yaml_path = PROJECT_ROOT / "config.yaml"
    if not yaml_path.exists():
        return {}
    try:
        import yaml
        with open(yaml_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception as e:
        logging.warning(f"[Config] YAML 加载失败: {e}")
        return {}

_yaml_config = _load_yaml()

def _yaml_val(*keys: str, default=None):
    """Traverse _yaml_config with dot-path keys."""
    val = _yaml_config
    for k in keys:
        if isinstance(val, dict):
            val = val.get(k)
        else:
            return default
    return val if val is not None else default


class Settings(BaseSettings):
    # v8.4.5: case_sensitive=False——兼容 Windows 上历史遗留的小写环境变量
    # （deepseek_api_key 等）。case_sensitive=True 在 pydantic-settings>=2.12
    # 下会把大小写不匹配的 env 键判为 extra 并拒绝启动（跨版本兼容性修复）。
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False
    )

    # 1. API Keys
    LLAMA_CLOUD_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""

    # 2. Model Endpoints & Fallback Keys
    MAIN_API_KEY: str = Field(default="")
    MAIN_BASE_URL: str = "https://api.deepseek.com"
    MAIN_MODEL: str = Field(default_factory=lambda: _yaml_val("model", "main", default="deepseek-chat"))

    FAST_API_KEY: str = ""
    FAST_BASE_URL: str = ""
    FAST_MODEL: str = Field(default_factory=lambda: _yaml_val("model", "fast", default="deepseek-chat"))

    EMBEDDING_API_KEY: str = ""
    EMBEDDING_BASE_URL: str = ""
    EMBEDDING_MODEL: str = Field(default_factory=lambda: _yaml_val("model", "embedding", default="intfloat/multilingual-e5-large"))
    RERANKER_MODEL: str = Field(default_factory=lambda: _yaml_val("model", "reranker", default="BAAI/bge-reranker-v2-m3"))

    # 3. Retrieval Parameters (核心检索阈值)
    TOP_K_VECTOR: int = Field(default_factory=lambda: _yaml_val("retrieval", "top_k_vector", default=40))
    TOP_K_BM25: int = Field(default_factory=lambda: _yaml_val("retrieval", "top_k_bm25", default=40))
    TOP_K_FINAL: int = Field(default_factory=lambda: _yaml_val("retrieval", "top_k_final", default=10))
    RRF_K: int = Field(default_factory=lambda: _yaml_val("retrieval", "rrf_k", default=60))
    RERANK_THRESHOLD: float = Field(default_factory=lambda: _yaml_val("retrieval", "rerank_threshold", default=0.25))
    MAX_REACT_STEPS: int = Field(default_factory=lambda: _yaml_val("react", "max_steps", default=5))
    DYNAMIC_THRESHOLD_RATIO: float = Field(default_factory=lambda: _yaml_val("retrieval", "dynamic_threshold_ratio", default=0.60))
    RAG_HYDE_ENABLED: bool = Field(default_factory=lambda: _yaml_val("retrieval", "rag_hyde_enabled", default=True))
    # v8.4.3 指令A: RETRIEVE_CONVERGE_MIN_DOCS 已移除（动态阈值已过滤 chunk，
    # 全部证据应进入报告，代码级收敛不再需要）
    HYDE_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("retrieval", "hyde_max_tokens", default=512))
    RRF_WEIGHT_ORIG_DENSE: float = Field(default_factory=lambda: _yaml_val("retrieval", "rrf_weights", "orig_dense", default=1.0))
    RRF_WEIGHT_HYDE_DENSE: float = Field(default_factory=lambda: _yaml_val("retrieval", "rrf_weights", "hyde_dense", default=1.0))
    RRF_WEIGHT_BM25: float = Field(default_factory=lambda: _yaml_val("retrieval", "rrf_weights", "bm25", default=1.0))

    # Academic search settings
    ACADEMIC_SOURCES: list = Field(default_factory=lambda: _yaml_val("academic_search", "enabled_sources", default=["crossref", "pubmed"]))
    ACADEMIC_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("academic_search", "timeout_per_source", default=8))
    # v8.15: 联网学术检索总开关（默认关闭——质量差+延迟高；工具与代码保留不删，重开只改 config）
    ACADEMIC_ENABLED: bool = Field(default_factory=lambda: _yaml_val("academic_search", "enabled", default=False))

    # v8.15: 查询级结果缓存（citrus_rag_search 全链路；键=规范化query+HyDE开关+语料指纹）
    RAG_CACHE_ENABLED: bool = Field(default_factory=lambda: _yaml_val("retrieval", "rag_cache_enabled", default=True))
    RAG_CACHE_SIZE: int = Field(default_factory=lambda: _yaml_val("retrieval", "rag_cache_size", default=300))
    RAG_CACHE_TTL_HOURS: int = Field(default_factory=lambda: _yaml_val("retrieval", "rag_cache_ttl_hours", default=24))

    # v8.15: 联网搜索（DeepSeek 原生 Responses API web_search tool）主开关
    WEB_SEARCH_ENABLED: bool = Field(default_factory=lambda: _yaml_val("web_search", "enabled", default=False))
    WEB_SEARCH_PROVIDER: str = Field(default_factory=lambda: _yaml_val("web_search", "provider", default="deepseek_responses"))
    WEB_SEARCH_RESPONSES_PATH: str = Field(default_factory=lambda: _yaml_val("web_search", "responses_path", default="/v1/responses"))
    # v8.15.3b: 联网 HTTP 读取超时（秒）——DeepSeek 原生联网实测 33-50s，
    # 旧 30s 把本会成功的慢响应自掐成"伪失败"（7×30s 白等根因）
    WEB_SEARCH_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("web_search", "timeout_sec", default=90))

    # v8.16.1: 草稿先行——load 后后台 fast 调用一次性产出
    # DRAFT_ZH（中文展示）/ DRAFT_EN（HyDE 复用）/ MULTI_QUERY / SUMMARY，
    # 前端 3-5s 见「预检索草稿·验证中」；草稿衍生查询并入多路检索。
    DRAFT_ENABLED: bool = Field(default_factory=lambda: _yaml_val("draft", "enabled", default=True))
    DRAFT_LABEL: str = Field(default_factory=lambda: _yaml_val("draft", "label", default="预检索草稿·验证中"))
    DRAFT_MAX_CHARS: int = Field(default_factory=lambda: _yaml_val("draft", "max_chars", default=800))
    # 草稿 HTTP 超时（秒）——非联网调用 2-5s，留足余量防慢响应被自掐
    DRAFT_TIMEOUT_SEC: int = Field(default_factory=lambda: _yaml_val("draft", "timeout_sec", default=15))
    # v8.16.3: 草稿调用输出上限——真空输出根因（v4-flash 思维链吃光 max_tokens，
    # v8.15.3c HyDE 同款事故）：1024→2048 给思维链留足余量（结构化区块实际 400-800 tokens）
    DRAFT_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("draft", "max_tokens", default=2048))
    DRAFT_MAX_ANGLES: int = Field(default_factory=lambda: _yaml_val("draft", "max_angles", default=3))
    # 已确认：SUMMARY 上限 3 条（不做 5-8），与草稿提示词「恰好 3 条」一致
    DRAFT_SUMMARY_POINTS: int = Field(default_factory=lambda: _yaml_val("draft", "summary_points", default=3))
    # 草稿衍生查询（DRAFT_EN + MULTI_QUERY + SUMMARY）是否并入 search_multi 多路检索
    DRAFT_EXTRA_RETRIEVAL: bool = Field(default_factory=lambda: _yaml_val("draft", "extra_retrieval", default=True))

    # 4. Chat Parameters (centralized)
    TEMPERATURE_MAIN: float = Field(default_factory=lambda: _yaml_val("chat", "temperature_main", default=0.2))
    TEMPERATURE_FAST: float = Field(default_factory=lambda: _yaml_val("chat", "temperature_fast", default=0.0))
    MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("chat", "max_tokens", default=4096))
    # v8.15.3: 检索决策子代理思维链控制（"default"=不发送任何参数；"off"= 给
    # retrieve-agent 下发 thinking:disabled 关闭推理——需先在真机确认 API 兼容后再
    # 开启，默认不发送以避免未实测的 400/参数忽略风险）
    MODEL_REASONING_MODE: str = Field(default_factory=lambda: _yaml_val("model", "reasoning_mode", default="default"))
    # v8.4 清理: RECENT_CONTENT_MAX_CHARS / COMPACT_MAX_TOKENS / FALLBACK_CONTENT_MAX_CHARS
    # 死配置已删（对应旧 graph.py check_history/compact_history 节点，代码已无引用；
    # 压缩统一走 context_budget 段的 ContextBudgetConfig）

    # Context Budget (v8.4: 存储全量·发送裁剪——v8.4.3 视图预算=模型窗口 1M)
    CONTEXT_BUDGET_ENABLED: bool = Field(default_factory=lambda: _yaml_val("context_budget", "enabled", default=True))
    CONTEXT_BUDGET_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("context_budget", "max_tokens", default=1000000))
    CONTEXT_BUDGET_SOFT_THRESHOLD: float = Field(default_factory=lambda: _yaml_val("context_budget", "soft_threshold", default=0.75))
    CONTEXT_BUDGET_HARD_THRESHOLD: float = Field(default_factory=lambda: _yaml_val("context_budget", "hard_threshold", default=0.93))
    CONTEXT_BUDGET_TARGET_RATIO: float = Field(default_factory=lambda: _yaml_val("context_budget", "target_ratio", default=0.50))
    CONTEXT_BUDGET_PROTECT_RECENT_TURNS: int = Field(default_factory=lambda: _yaml_val("context_budget", "protect_recent_turns", default=3))

    # ── Permission (v8.4.3 结构化权限确认) ──
    PERMISSION_MODE: str = Field(default_factory=lambda: _yaml_val("permission", "mode", default="auto_workspace"))
    # v8.4.5: ask 模式授权等待秒数（前端卡片未回应时超时按拒绝处理）
    PERMISSION_WAIT_SEC: int = Field(default_factory=lambda: _yaml_val("permission", "wait_sec", default=90))

    # ── Version (v8.4.5: 版本单源——UI/健康检查/文档以此为准) ──
    VERSION: str = "8.14.1"

    # ── Context Engineering (阶段1: 静态前缀灰度开关) ──
    # true = SystemMessage 字节级稳定（format 指南/策略卡片/skills 移出前缀，
    #        追加到当前轮 HumanMessage 尾部），利于 DeepSeek 上下文缓存命中；
    # false = 旧行为（动态内容进 SystemMessage，前缀每次请求重算）
    CONTEXT_STATIC_PREFIX: bool = Field(default_factory=lambda: _yaml_val("context", "static_prefix", default=False))

    # 5. Tools Parameters
    # v8.3.3: MAX_TOOL_CALLS 死配置已删除（无引用）

    # 6. Graph Parameters
    RECURSION_LIMIT: int = Field(default_factory=lambda: _yaml_val("graph", "recursion_limit", default=25))

    # 7. API Timeouts & Limits
    QDRANT_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("api", "qdrant_timeout", default=60))

    # v8.9 向量检索后端（auto | qdrant | lancedb）：lancedb 嵌入式百万级+热更新+无锁；
    # auto = 检测 data/lancedb 有表则用 lancedb，否则回退 qdrant（新旧数据包开箱即用）
    RETRIEVAL_BACKEND: str = Field(default_factory=lambda: _yaml_val("retrieval", "backend", default="auto"))

    # 8. System Paths & Logging
    DATA_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    LOG_DIR: str = Field(default_factory=lambda: _yaml_val("logging", "dir", default="logs"))
    LOG_LEVEL: str = Field(default_factory=lambda: _yaml_val("logging", "level", default="INFO"))
    EMBEDDER_FORCE_CPU: bool = Field(default_factory=lambda: _yaml_val("embedder", "force_cpu", default=False))

    # ── Agent / Sub-agent ──
    AGENTS_DIR: str = Field(default_factory=lambda: _yaml_val("agent", "agents_dir", default="agents"))
    WORKSPACE_DIR: str = Field(default_factory=lambda: _yaml_val("agent", "workspace_dir", default="workspace"))
    # v8.3.3: 轮次上限接线 config（此前 supervisor/light/子代理全部硬编码且与 config 脱节）
    SUPERVISOR_MAX_TURNS: int = Field(default_factory=lambda: _yaml_val("supervisor", "max_turns", default=8))
    # v8.3.4: 每轮工具调用预算（防一轮内串行执行多个子代理）
    SUPERVISOR_MAX_TOOLS_PER_TURN: int = Field(default_factory=lambda: _yaml_val("supervisor", "max_tools_per_turn", default=2))
    LIGHT_MAX_TURNS: int = Field(default_factory=lambda: _yaml_val("light", "max_turns", default=2))
    SUBAGENT_MAX_TURNS: dict = Field(default_factory=lambda: _yaml_val("subagents", default={}))
    TOOL_EXEC_TIMEOUT_SEC: int = Field(default_factory=lambda: _yaml_val("agent", "tool_exec_timeout_sec", default=60))
    # v8.15.3b: 单工具超时覆盖（{tool_name: 秒}，覆盖 TOOL_EXEC_TIMEOUT_SEC 默认值）
    TOOL_TIMEOUTS: dict = Field(default_factory=lambda: _yaml_val("agent", "tool_timeouts", default={}))

    # ── Write Pipeline (v8.3.2) ──
    PIPELINE_MATERIAL_MIN_COUNT: int = Field(default_factory=lambda: _yaml_val("pipeline", "material_min_count", default=8))
    PIPELINE_MAX_PLAN_RETRIES: int = Field(default_factory=lambda: _yaml_val("pipeline", "max_plan_retries", default=1))
    PIPELINE_REFS_COVERAGE_RATIO: float = Field(default_factory=lambda: _yaml_val("pipeline", "refs_coverage_ratio", default=0.4))
    PIPELINE_SECTION_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("pipeline", "section_max_tokens", default=4000))
    PIPELINE_SECTION_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("pipeline", "section_timeout", default=120))
    PIPELINE_RESUME_ENABLED: bool = Field(default_factory=lambda: _yaml_val("pipeline", "resume_enabled", default=True))
    # v8.4.6: 单章生成并发度（生成并发、按序写盘；1=串行，测试环境用 1 保证确定性）
    PIPELINE_PARALLEL_SECTIONS: int = Field(default_factory=lambda: _yaml_val("pipeline", "parallel_sections", default=3))
    TOOL_RESULT_CAPS: dict = Field(default_factory=lambda: _yaml_val("agent", "tool_result_caps", default={}))

    # ── File I/O ──
    FILE_READ_MAX_SIZE_MB: int = Field(default_factory=lambda: _yaml_val("file_io", "read_max_size_mb", default=50))
    FILE_WRITE_MAX_SIZE_MB: int = Field(default_factory=lambda: _yaml_val("file_io", "write_max_size_mb", default=10))
    # v8.3.3: 绝对路径读取仅限项目根 + 此额外根目录列表（默认空 = 严格工作区模式）
    FILE_READ_EXTRA_ROOTS: list = Field(default_factory=lambda: _yaml_val("file_io", "read_extra_roots", default=[]))

    # ── Statistics ──
    STATS_ALPHA: float = Field(default_factory=lambda: _yaml_val("statistics", "alpha", default=0.05))
    STATS_MAX_SAMPLE_SIZE: int = Field(default_factory=lambda: _yaml_val("statistics", "max_sample_size", default=100000))

    # ── Databases (可插拔，从 databases: 段读取) ──
    @property
    def DATABASES(self) -> dict:
        return _yaml_val("databases", default={})

    # v8.4.4: BIO_DB_CACHE_TTL 死配置已删（无消费点）
    NCBI_ENTREZ_EMAIL: str = Field(default_factory=lambda: _yaml_val("bio_db", "entrez_email", default="citrus-agent@localhost"))
    NCBI_ENTREZ_TOOL: str = Field(default_factory=lambda: _yaml_val("bio_db", "entrez_tool", default="CitrusQAAgent"))
    KEGG_API_URL: str = Field(default_factory=lambda: _yaml_val("bio_db", "kegg_api_url", default="https://rest.kegg.jp"))
    UNIPROT_API_URL: str = Field(default_factory=lambda: _yaml_val("bio_db", "uniprot_api_url", default="https://rest.uniprot.org"))
    STRING_API_URL: str = Field(default_factory=lambda: _yaml_val("bio_db", "string_api_url", default="https://string-db.org/api"))

    # ── Baidu Unlimited-OCR ──
    BAIDU_OCR_API_KEY: str = Field(default_factory=lambda: _yaml_val("baidu_ocr", "api_key", default=""))
    BAIDU_OCR_SECRET_KEY: str = Field(default_factory=lambda: _yaml_val("baidu_ocr", "secret_key", default=""))
    BAIDU_OCR_URL: str = Field(default_factory=lambda: _yaml_val("baidu_ocr", "url", default="https://aip.baidubce.com/rest/2.0/brain/online/v2/unlimited-ocr-parser"))

    # ── LaTeX ──
    LATEX_OUTPUT_DIR: str = Field(default_factory=lambda: _yaml_val("latex", "output_dir", default="workspace/output"))

    # v8.4.4: HITL_* 死配置已删（stub 端点无消费点）；权限由 permission 体系承担

    # ── Sandbox ──
    SANDBOX_ENABLED: bool = Field(default_factory=lambda: _yaml_val("sandbox", "enabled", default=True))
    SANDBOX_ALLOWED_PATTERNS: list = Field(default_factory=lambda: _yaml_val("sandbox", "allowed_patterns", default=["read_*", "search_*", "list_*"]))
    SANDBOX_DANGEROUS_PATTERNS: list = Field(default_factory=lambda: _yaml_val("sandbox", "dangerous_patterns", default=["delete_*", "exec_*", "write_*"]))

    # v8.5.0 开源版：运行时 API Key（WebUI 引导填写）——优先于 .env，
    # 持久化到 state/api_key（gitignore 内，不入仓库）；.env 保持不动
    _runtime_api_key: str = ""

    def _load_runtime_api_key(self) -> None:
        """启动时从 state/api_key 读取（WebUI 填写的 key 跨重启保留）。"""
        try:
            p = PROJECT_ROOT / "state" / "api_key"
            if p.exists():
                v = p.read_text(encoding="utf-8").strip()
                if v:
                    self._runtime_api_key = v
        except Exception:
            pass

    def save_runtime_api_key(self, api_key: str) -> bool:
        """持久化 WebUI 填写的 key（state/api_key，权限收紧）。"""
        try:
            p = PROJECT_ROOT / "state" / "api_key"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(api_key.strip(), encoding="utf-8")
            try:
                import os
                os.chmod(p, 0o600)
            except Exception:
                pass
            self._runtime_api_key = api_key.strip()
            return True
        except Exception:
            return False

    # ── v8.14.1 运行时模型覆盖（前端设置面板切换底座模型）──
    # state/model_config.json（gitignore 内）存 {main_model,fast_model,main_base_url,fast_base_url}；
    # 空字段 = 回退 yaml/env 默认。优先级: 运行时覆盖 > 环境变量 > YAML > 默认值。
    _runtime_model_cfg: Dict[str, str] = {}

    def _load_runtime_model_config(self) -> None:
        try:
            p = PROJECT_ROOT / "state" / "model_config.json"
            if p.exists():
                data = json.loads(p.read_text(encoding="utf-8"))
                self._runtime_model_cfg = {
                    k: str(v).strip() for k, v in data.items()
                    if k in ("main_model", "fast_model", "main_base_url", "fast_base_url")
                    and str(v).strip()
                }
        except Exception:
            self._runtime_model_cfg = {}

    def save_runtime_model_config(self, *, main: str = "", fast: str = "",
                                  main_base_url: str = "", fast_base_url: str = "") -> bool:
        cfg = {}
        for k, v in (("main_model", main), ("fast_model", fast),
                     ("main_base_url", main_base_url), ("fast_base_url", fast_base_url)):
            if (v or "").strip():
                cfg[k] = v.strip()
        try:
            p = PROJECT_ROOT / "state" / "model_config.json"
            if not cfg:
                p.unlink(missing_ok=True)
            else:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")
            self._runtime_model_cfg = cfg
            return True
        except Exception:
            return False

    @property
    def has_model_overrides(self) -> bool:
        return bool(self._runtime_model_cfg)

    @property
    def RESOLVED_MAIN_MODEL(self) -> str:
        return self._runtime_model_cfg.get("main_model") or self.MAIN_MODEL

    @property
    def RESOLVED_FAST_MODEL(self) -> str:
        return self._runtime_model_cfg.get("fast_model") or self.FAST_MODEL

    @property
    def RESOLVED_MAIN_BASE_URL(self) -> str:
        return self._runtime_model_cfg.get("main_base_url") or self.MAIN_BASE_URL

    @property
    def RESOLVED_MAIN_API_KEY(self) -> str:
        return self._runtime_api_key or self.DEEPSEEK_API_KEY or self.MAIN_API_KEY

    @property
    def RESOLVED_FAST_API_KEY(self) -> str:
        return self.FAST_API_KEY or self.RESOLVED_MAIN_API_KEY

    @property
    def RESOLVED_FAST_BASE_URL(self) -> str:
        return (self._runtime_model_cfg.get("fast_base_url")
                or self.FAST_BASE_URL
                or self.RESOLVED_MAIN_BASE_URL)

settings = Settings()

# v8.5.0 模型镜像兜底：任何启动方式（run.ps1 / 手动 uvicorn）下，
# HuggingFace 模型（reranker/embedding）首次下载统一走国内镜像，
# 已有 HF_ENDPOINT 环境变量则尊重用户自定义
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
settings._load_runtime_api_key()
settings._load_runtime_model_config()

# ── 启动配置校验（v8.3.3 fail-fast）──
def validate_config() -> list:
    """关键配置自检，返回问题列表（缺失 API key 等）。"""
    issues = []
    if not settings.RESOLVED_MAIN_API_KEY:
        issues.append("缺少主模型 API Key（.env 的 DEEPSEEK_API_KEY 或 MAIN_API_KEY）")
    if not settings.RESOLVED_FAST_API_KEY:
        issues.append("缺少快速模型 API Key（将回退主 Key）")
    if settings.CONTEXT_BUDGET_MAX_TOKENS <= 0:
        issues.append("CONTEXT_BUDGET_MAX_TOKENS 必须为正数")
    if settings.PIPELINE_SECTION_MAX_TOKENS <= 0:
        issues.append("PIPELINE_SECTION_MAX_TOKENS 必须为正数")
    if not isinstance(settings.SUBAGENT_MAX_TURNS, dict):
        issues.append("subagents 配置必须是 dict（{name: {max_turns: n}}）")
    return issues


_config_issues = validate_config()
if _config_issues:
    _w = logging.getLogger("src.config")
    for _issue in _config_issues:
        _w.warning(f"[Config] 配置问题: {_issue}")

# --- 主模型解析（v8.14.1）---
# 全链路主模型唯一入口 get_deepseek_model()：运行时覆盖（前端设置面板，
# state/model_config.json）优先，未覆盖回退 yaml/env(model.main)。
# 遗留的 switch_model/_available_models 运行时切换已删除（无调用点；
# 切换统一走前端设置面板 POST /api/v2/config/model）。

def get_deepseek_model() -> str:
    return settings.RESOLVED_MAIN_MODEL