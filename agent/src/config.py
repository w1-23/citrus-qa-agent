"""
Unified configuration for Citrus RAG Agent using pydantic-settings + YAML.
Priority: environment variable > YAML > pydantic-settings default
"""
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


class FeatureFlags:
    """特征标志系统 — 运行时动态开关实验性功能"""
    _flags: Dict[str, bool] = {}

    @classmethod
    def load(cls, yaml_cfg: Dict[str, Any]):
        raw = yaml_cfg.get("feature_flags", {}) if isinstance(yaml_cfg, dict) else {}
        cls._flags = {k: bool(v) for k, v in raw.items()}

    @classmethod
    def is_enabled(cls, flag: str, default: bool = False) -> bool:
        return cls._flags.get(flag, default)

    @classmethod
    def set_flag(cls, flag: str, value: bool):
        cls._flags[flag] = value

    @classmethod
    def all_flags(cls) -> Dict[str, bool]:
        return dict(cls._flags)

# 加载特征标志（必须在 FeatureFlags 定义之后）
FeatureFlags.load(_yaml_config)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True
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

    # 5. Web Search Provider
    WEB_SEARCH_API_KEY: str = ""
    WEB_SEARCH_PROVIDER: str = "tavily"
    SERPER_API_KEY: str = ""

    # 3. Retrieval Parameters (核心检索阈值)
    TOP_K_VECTOR: int = Field(default_factory=lambda: _yaml_val("retrieval", "top_k_vector", default=40))
    TOP_K_BM25: int = Field(default_factory=lambda: _yaml_val("retrieval", "top_k_bm25", default=40))
    TOP_K_FINAL: int = Field(default_factory=lambda: _yaml_val("retrieval", "top_k_final", default=10))
    RRF_K: int = Field(default_factory=lambda: _yaml_val("retrieval", "rrf_k", default=60))
    RERANK_THRESHOLD: float = Field(default_factory=lambda: _yaml_val("retrieval", "rerank_threshold", default=0.25))
    MAX_REACT_STEPS: int = Field(default_factory=lambda: _yaml_val("react", "max_steps", default=5))
    DYNAMIC_THRESHOLD_RATIO: float = Field(default_factory=lambda: _yaml_val("retrieval", "dynamic_threshold_ratio", default=0.60))
    RAG_HYDE_ENABLED: bool = Field(default_factory=lambda: _yaml_val("retrieval", "rag_hyde_enabled", default=True))
    HYDE_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("retrieval", "hyde_max_tokens", default=512))
    RRF_WEIGHT_ORIG_DENSE: float = Field(default_factory=lambda: _yaml_val("retrieval", "rrf_weights", "orig_dense", default=1.0))
    RRF_WEIGHT_HYDE_DENSE: float = Field(default_factory=lambda: _yaml_val("retrieval", "rrf_weights", "hyde_dense", default=1.0))
    RRF_WEIGHT_BM25: float = Field(default_factory=lambda: _yaml_val("retrieval", "rrf_weights", "bm25", default=1.0))

    # Academic search settings
    ACADEMIC_SOURCES: list = Field(default_factory=lambda: _yaml_val("academic_search", "enabled_sources", default=["crossref", "pubmed"]))
    ACADEMIC_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("academic_search", "timeout_per_source", default=8))

    # 4. Chat Parameters (centralized)
    TEMPERATURE_MAIN: float = Field(default_factory=lambda: _yaml_val("chat", "temperature_main", default=0.2))
    TEMPERATURE_FAST: float = Field(default_factory=lambda: _yaml_val("chat", "temperature_fast", default=0.0))
    MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("chat", "max_tokens", default=4096))
    RECENT_CONTENT_MAX_CHARS: int = Field(default_factory=lambda: _yaml_val("chat", "recent_content_max_chars", default=800))
    COMPACT_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("context_budget", "compact_max_tokens", default=800))
    FALLBACK_CONTENT_MAX_CHARS: int = Field(default_factory=lambda: _yaml_val("chat", "fallback_content_max_chars", default=300))

    # Context Budget (v8.3.1: 配置单向化，图内不再硬编码)
    CONTEXT_BUDGET_ENABLED: bool = Field(default_factory=lambda: _yaml_val("context_budget", "enabled", default=True))
    CONTEXT_BUDGET_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("context_budget", "max_tokens", default=1000000))
    CONTEXT_BUDGET_SOFT_THRESHOLD: float = Field(default_factory=lambda: _yaml_val("context_budget", "soft_threshold", default=0.60))
    CONTEXT_BUDGET_HARD_THRESHOLD: float = Field(default_factory=lambda: _yaml_val("context_budget", "hard_threshold", default=0.93))

    # 5. Tools Parameters
    MAX_TOOL_CALLS: int = Field(default_factory=lambda: _yaml_val("tools", "max_tool_calls", default=4))

    # 6. Graph Parameters
    RECURSION_LIMIT: int = Field(default_factory=lambda: _yaml_val("graph", "recursion_limit", default=25))

    # 7. API Timeouts & Limits
    QDRANT_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("api", "qdrant_timeout", default=60))
    LLM_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("api", "llm_timeout", default=15))
    INTENT_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("api", "intent_timeout", default=10))
    INTENT_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("api", "intent_max_tokens", default=10))

    # 8. System Paths & Logging
    DATA_DIR: Path = Field(default_factory=lambda: PROJECT_ROOT / "data")
    LOG_DIR: str = Field(default_factory=lambda: _yaml_val("logging", "dir", default="logs"))
    LOG_LEVEL: str = Field(default_factory=lambda: _yaml_val("logging", "level", default="INFO"))
    EMBEDDER_FORCE_CPU: bool = Field(default_factory=lambda: _yaml_val("embedder", "force_cpu", default=False))

    # ── Agent / Sub-agent ──
    AGENTS_DIR: str = Field(default_factory=lambda: _yaml_val("agent", "agents_dir", default="agents"))
    WORKSPACE_DIR: str = Field(default_factory=lambda: _yaml_val("agent", "workspace_dir", default="workspace"))
    AGENT_MAX_TURNS: int = Field(default_factory=lambda: _yaml_val("agent", "max_turns", default=3))
    AGENT_TIMEOUT_SEC: int = Field(default_factory=lambda: _yaml_val("agent", "timeout_sec", default=30))
    TOOL_EXEC_TIMEOUT_SEC: int = Field(default_factory=lambda: _yaml_val("agent", "tool_exec_timeout_sec", default=60))

    # ── Write Pipeline (v8.3.2) ──
    PIPELINE_DEFAULT_TARGET_CHARS: int = Field(default_factory=lambda: _yaml_val("pipeline", "default_target_chars", default=6000))
    PIPELINE_MATERIAL_MIN_COUNT: int = Field(default_factory=lambda: _yaml_val("pipeline", "material_min_count", default=8))
    PIPELINE_MAX_PLAN_RETRIES: int = Field(default_factory=lambda: _yaml_val("pipeline", "max_plan_retries", default=1))
    PIPELINE_REFS_COVERAGE_RATIO: float = Field(default_factory=lambda: _yaml_val("pipeline", "refs_coverage_ratio", default=0.4))
    PIPELINE_SECTION_MAX_TOKENS: int = Field(default_factory=lambda: _yaml_val("pipeline", "section_max_tokens", default=4000))
    PIPELINE_SECTION_TIMEOUT: int = Field(default_factory=lambda: _yaml_val("pipeline", "section_timeout", default=120))
    PIPELINE_RESUME_ENABLED: bool = Field(default_factory=lambda: _yaml_val("pipeline", "resume_enabled", default=True))
    TOOL_RESULT_CAPS: dict = Field(default_factory=lambda: _yaml_val("agent", "tool_result_caps", default={}))

    # ── File I/O ──
    FILE_READ_MAX_SIZE_MB: int = Field(default_factory=lambda: _yaml_val("file_io", "read_max_size_mb", default=50))
    FILE_WRITE_MAX_SIZE_MB: int = Field(default_factory=lambda: _yaml_val("file_io", "write_max_size_mb", default=10))

    # ── Statistics ──
    STATS_ALPHA: float = Field(default_factory=lambda: _yaml_val("statistics", "alpha", default=0.05))
    STATS_MAX_SAMPLE_SIZE: int = Field(default_factory=lambda: _yaml_val("statistics", "max_sample_size", default=100000))

    # ── Databases (可插拔，从 databases: 段读取) ──
    @property
    def DATABASES(self) -> dict:
        return _yaml_val("databases", default={})

    BIO_DB_CACHE_TTL: int = Field(default_factory=lambda: _yaml_val("bio_db", "cache_ttl", default=3600))
    NCBI_ENTREZ_EMAIL: str = Field(default_factory=lambda: _yaml_val("bio_db", "entrez_email", default="citrus-agent@localhost"))
    NCBI_ENTREZ_TOOL: str = Field(default_factory=lambda: _yaml_val("bio_db", "entrez_tool", default="CitrusQAAgent"))
    KEGG_API_URL: str = Field(default_factory=lambda: _yaml_val("bio_db", "kegg_api_url", default="https://rest.kegg.jp"))
    UNIPROT_API_URL: str = Field(default_factory=lambda: _yaml_val("bio_db", "uniprot_api_url", default="https://rest.uniprot.org"))
    STRING_API_URL: str = Field(default_factory=lambda: _yaml_val("bio_db", "string_api_url", default="https://string-db.org/api"))

    # ── Baidu Unlimited-OCR ──
    BAIDU_OCR_API_KEY: str = Field(default_factory=lambda: _yaml_val("baidu_ocr", "api_key", default=""))
    BAIDU_OCR_SECRET_KEY: str = Field(default_factory=lambda: _yaml_val("baidu_ocr", "secret_key", default=""))
    BAIDU_OCR_URL: str = Field(default_factory=lambda: _yaml_val("baidu_ocr", "url", default="https://aip.baidubce.com/rest/2.0/brain/online/v2/unlimited-ocr-parser"))

    # ── Compute Scheduler ──
    PREFER_DEVICE: str = Field(default_factory=lambda: _yaml_val("compute", "prefer_device", default="auto"))

    # ── LaTeX ──
    LATEX_OUTPUT_DIR: str = Field(default_factory=lambda: _yaml_val("latex", "output_dir", default="workspace/output"))

    # ── HITL ──
    HITL_ENABLED: bool = Field(default_factory=lambda: _yaml_val("hitl", "enabled", default=True))
    HITL_AUTO_APPROVE: bool = Field(default_factory=lambda: _yaml_val("hitl", "auto_approve", default=False))
    HITL_SKIP_SANDBOX: bool = Field(default_factory=lambda: _yaml_val("hitl", "skip_sandbox", default=False))
    HITL_TIMEOUT_SEC: int = Field(default_factory=lambda: _yaml_val("hitl", "timeout_sec", default=120))

    # ── Sandbox ──
    SANDBOX_ENABLED: bool = Field(default_factory=lambda: _yaml_val("sandbox", "enabled", default=True))
    SANDBOX_ALLOWED_PATTERNS: list = Field(default_factory=lambda: _yaml_val("sandbox", "allowed_patterns", default=["read_*", "search_*", "list_*"]))
    SANDBOX_DANGEROUS_PATTERNS: list = Field(default_factory=lambda: _yaml_val("sandbox", "dangerous_patterns", default=["delete_*", "exec_*", "write_*"]))

    @property
    def RESOLVED_MAIN_API_KEY(self) -> str:
        return self.DEEPSEEK_API_KEY or self.MAIN_API_KEY

    @property
    def RESOLVED_FAST_API_KEY(self) -> str:
        return self.FAST_API_KEY or self.RESOLVED_MAIN_API_KEY

    @property
    def RESOLVED_FAST_BASE_URL(self) -> str:
        return self.FAST_BASE_URL or self.MAIN_BASE_URL

settings = Settings()

# --- Model Switching Logic ---
# v8.3.1: 全部切换 DeepSeek V4 Flash 正式版，V4 Pro 已移除
_available_models: Dict[str, str] = {
    "deepseek-v4-flash": "DeepSeek V4 Flash",
    "deepseek-chat": "DeepSeek V4 Flash (legacy)",
    "deepseek-reasoner": "DeepSeek V4 Flash thinking (legacy)",
}
_current_model = settings.MAIN_MODEL

def get_deepseek_client() -> OpenAI:
    return OpenAI(api_key=settings.RESOLVED_MAIN_API_KEY, base_url=settings.MAIN_BASE_URL)

def get_deepseek_model() -> str:
    return _current_model

def get_available_models() -> dict:
    return dict(_available_models)

def switch_model(model_id: str) -> bool:
    global _current_model
    if model_id in _available_models:
        _current_model = model_id
        return True
    return False