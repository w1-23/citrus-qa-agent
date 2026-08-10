import math
import logging
import threading
import traceback
import warnings
from pathlib import Path

warnings.filterwarnings("ignore", message=".*incorrect regex pattern.*")
warnings.filterwarnings("ignore", message=".*Some nodes were not assigned.*")
warnings.filterwarnings("ignore", message=".*mean pooling instead of CLS.*")
from typing import List, Dict
from optimum.onnxruntime import ORTModelForSequenceClassification
from transformers import AutoTokenizer
from src.config import settings
from src.engine.hardware import get_ort_providers

logger = logging.getLogger(__name__)


def _reranker_score(logit: float) -> float:
    return round(1.0 / (1.0 + math.exp(-logit)), 4)


class Reranker:
    _instance = None
    _singleton_lock = threading.Lock()
    _thread_local = threading.local()

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
            self._cache_dir = Path(".hf_cache/onnx_reranker")
            self._force_cpu = False
            self._providers = None
            self._needs_lock = False
            self._shared_model = None
            self._shared_tokenizer = None
            self._shared_lock = threading.Lock()
            self._load_config()
            self._loaded = True

    @staticmethod
    def _is_mistral_model() -> bool:
        return "mistral" in settings.RERANKER_MODEL.lower()

    def _load_config(self):
        raw_providers = get_ort_providers()
        self._providers = raw_providers
        if any("DmlExecutionProvider" in p or "CUDAExecutionProvider" in p for p in raw_providers):
            self._needs_lock = True
        logger.info(f"[Reranker] 配置 | Provider: {raw_providers} | Per-thread: {not self._needs_lock}")

    def _create_session(self):
        if not self._cache_dir.exists():
            logger.info("Exporting Reranker to ONNX (one-time)...")
            model = ORTModelForSequenceClassification.from_pretrained(settings.RERANKER_MODEL, export=True)
            model.save_pretrained(self._cache_dir)
            tok = AutoTokenizer.from_pretrained(settings.RERANKER_MODEL, fix_mistral_regex=self._is_mistral_model())
            tok.save_pretrained(self._cache_dir)
        provider = self._providers[0] if self._providers else "CPUExecutionProvider"
        logger.info(f"Loading Reranker from ONNX cache | Provider: {provider} | Cache: {self._cache_dir}")
        tokenizer = AutoTokenizer.from_pretrained(self._cache_dir, fix_mistral_regex=self._is_mistral_model())
        model = ORTModelForSequenceClassification.from_pretrained(self._cache_dir, provider=provider)
        return model, tokenizer

    def _get_model(self):
        if self._needs_lock:
            with self._shared_lock:
                if self._shared_model is None:
                    self._shared_model, self._shared_tokenizer = self._create_session()
            return self._shared_model, self._shared_tokenizer
        if not hasattr(self._thread_local, "_model") or self._thread_local._model is None:
            self._thread_local._model, self._thread_local._tokenizer = self._create_session()
        return self._thread_local._model, self._thread_local._tokenizer

    def _safe_rerank(self, query: str, chunks: List[Dict], top_k: int) -> List[Dict]:
        for attempt in range(2):
            try:
                model, tokenizer = self._get_model()
                texts = [c["text"] for c in chunks]
                inputs = tokenizer(
                    [(query, t) for t in texts],
                    padding=True, truncation=True, max_length=512, return_tensors="pt"
                )
                if self._needs_lock:
                    from src.engine.gpu_lock import GPULockGuard
                    with GPULockGuard():
                        with self._shared_lock:
                            outputs = model(**inputs)
                else:
                    outputs = model(**inputs)
                logits = outputs.logits.squeeze(-1).tolist()
                if isinstance(logits, float):
                    logits = [logits]
                for c, s in zip(chunks, [_reranker_score(x) for x in logits]):
                    c["rerank_score"] = s
                chunks.sort(key=lambda x: x["rerank_score"], reverse=True)
                return chunks[:top_k]
            except UnicodeDecodeError as e:
                logger.error(f"[Reranker] UnicodeDecodeError (尝试 {attempt + 1}/2): {e}")
                if attempt == 0 and not self._force_cpu:
                    with self._singleton_lock:
                        self._force_cpu = True
                        self._needs_lock = False
                        self._shared_model = None
                        self._shared_tokenizer = None
                        self._providers = ["CPUExecutionProvider"]
                        self._thread_local = threading.local()
                    continue
                raise
            except Exception as e:
                logger.error(f"[Reranker] 运行时异常 (尝试 {attempt + 1}/2): {type(e).__name__}: {e}")
                if attempt == 0 and not self._force_cpu:
                    with self._singleton_lock:
                        self._force_cpu = True
                        self._needs_lock = False
                        self._shared_model = None
                        self._shared_tokenizer = None
                        self._providers = ["CPUExecutionProvider"]
                        self._thread_local = threading.local()
                    continue
                raise

    def rerank(self, query: str, chunks: List[Dict], top_k: int = 10) -> List[Dict]:
        if not chunks:
            return []
        return self._safe_rerank(query, chunks, top_k)
