import os
import hashlib
import logging
import re
import threading
import traceback
import onnxruntime as ort
from typing import List
from fastembed import TextEmbedding
from src.config import settings

# 默认离线模式：不连 HuggingFace，直接用本地缓存。设 HF_HUB_OFFLINE=0 可联网检查模型更新。
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# v8.4.4: query embedding LRU（进程内 512 条，与 HyDE 缓存同款模式）
_QUERY_CACHE_MAX = 512

logger = logging.getLogger(__name__)


def _clean_text(text: str) -> str:
    if not text:
        return ""
    try:
        text = text.encode("utf-8", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        pass
    text = re.sub(r"[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]", "", text)
    text = text.replace("\ufffd", "")
    return text.strip()


def _validate_utf8(text: str) -> str:
    try:
        text.encode("utf-8")
        return text
    except UnicodeEncodeError:
        return text.encode("utf-8", errors="replace").decode("utf-8")


class Embedder:
    _instance = None
    _singleton_lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._singleton_lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if getattr(self, "_loaded", False):
            return
        with self._singleton_lock:
            if getattr(self, "_loaded", False):
                return
            self.model_name = settings.EMBEDDING_MODEL
            self._dim = None
            self._force_cpu = settings.EMBEDDER_FORCE_CPU
            self._providers = None
            self._needs_lock = False
            self._shared_model = None
            self._shared_lock = threading.Lock()
            self._query_cache: dict = {}          # v8.4.4
            self._cache_lock = threading.Lock()   # v8.4.4
            self._load_config()
            self._loaded = True

    def _load_config(self):
        if self._force_cpu:
            self._providers = ["CPUExecutionProvider"]
            logger.info(f"[Embedder] 配置 | Provider: CPU(forced) | Per-thread: yes | 模型: {self.model_name}")
            return
        from src.engine.hardware import get_ort_providers
        raw = get_ort_providers()
        self._providers = raw
        if "DmlExecutionProvider" in raw:
            self._needs_lock = True
        label = raw[0] if raw else "CPU"
        logger.info(f"[Embedder] 配置 | Provider: {label} | Per-thread: {not self._needs_lock} | 模型: {self.model_name}")

    def _create_session(self):
        with self._singleton_lock:
            model = TextEmbedding(model_name=self.model_name, providers=self._providers)
            if self._dim is None:
                test_emb = list(model.embed(["validate"]))
                self._dim = len(test_emb[0])
                logger.info(f"[Embedder] 新会话 | Dim: {self._dim} | Provider: {self._providers}")
        return model

    def _get_model(self):
        # v8.10k: 统一共享单例——CPU 模式不再 per-thread 复制。
        # 旧实现每个线程首次调用都重新加载 e5-large（~10-20s），而 LTM 语义召回
        # 走 asyncio.to_thread（线程池线程不固定）→"每次首问特别慢"的根因。
        # onnxruntime CPU 会话并发 Run 是线程安全的，锁仅兜底。
        with self._shared_lock:
            if self._shared_model is None:
                self._shared_model = self._create_session()
        return self._shared_model

    def _safe_embed(self, texts: List[str]) -> List[List[float]]:
        for attempt in range(2):
            try:
                model = self._get_model()
                if self._needs_lock:
                    from src.engine.gpu_lock import GPULockGuard
                    with GPULockGuard():
                        with self._shared_lock:
                            return [emb.tolist() for emb in model.embed(texts)]
                return [emb.tolist() for emb in model.embed(texts)]
            except UnicodeDecodeError as e:
                logger.error(f"[Embedder] UnicodeDecodeError (尝试 {attempt + 1}/2): {e}")
                if attempt == 0 and not self._force_cpu:
                    with self._singleton_lock:
                        self._force_cpu = True
                        self._needs_lock = False
                        self._shared_model = None
                        self._providers = ["CPUExecutionProvider"]
                    continue
                raise
            except Exception as e:
                logger.error(f"[Embedder] 运行时异常 (尝试 {attempt + 1}/2): {type(e).__name__}: {e}")
                if attempt == 0 and not self._force_cpu:
                    with self._singleton_lock:
                        self._force_cpu = True
                        self._needs_lock = False
                        self._shared_model = None
                        self._providers = ["CPUExecutionProvider"]
                    continue
                raise

    def embed_query(self, text: str) -> List[float]:
        prefixed = f"query: {_validate_utf8(text)}"
        # v8.4.4: query embedding LRU 缓存（进程内，512 条）——检索场景同一
        # 关键词跨请求/跨轮复用，避免重复推理（TUNE_PARAMS §9 补全）
        key = hashlib.md5(prefixed.encode("utf-8")).hexdigest()
        with self._cache_lock:
            cached = self._query_cache.get(key)
            if cached is not None:
                return cached
        result = self.embed_docs([prefixed])
        vec = result[0] if result else []
        if vec:
            with self._cache_lock:
                if key not in self._query_cache:
                    if len(self._query_cache) >= _QUERY_CACHE_MAX:
                        self._query_cache.pop(next(iter(self._query_cache)))
                    self._query_cache[key] = vec
        return vec

    def embed_docs(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        cleaned = [_clean_text(_validate_utf8(t)) for t in texts]
        cleaned = [t for t in cleaned if t]
        if not cleaned:
            return [[0.0] * self.dim for _ in texts]
        return self._safe_embed(cleaned)

    @property
    def dim(self) -> int:
        if self._dim is None:
            model = self._get_model()
            if self._dim is None:
                result = list(model.embed(["dummy"]))
                self._dim = len(result[0])
        return self._dim
