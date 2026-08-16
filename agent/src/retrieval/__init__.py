"""Retrieval initialization — unified entry for preloading RAG models."""
import logging
import time
from src.retrieval.multi_retriever import MultiBatchRetriever

logger = logging.getLogger(__name__)


def eager_load_rag():
    """Preload embedder, retriever, reranker, and SkillTree at app startup.
    MultiBatchRetriever is a singleton — instantiation triggers _load_data() once.
    SkillTree is also preloaded so its vectors are cached/loaded from disk.
    """
    t0 = time.perf_counter()
    try:
        MultiBatchRetriever()  # singleton __init__ → _load_data() → Qdrant + BM25 + Models
        logger.info(f"[RAG] MultiBatchRetriever loaded ({time.perf_counter()-t0:.1f}s)")
    except Exception as e:
        logger.error(f"[RAG] MultiBatchRetriever preload failed: {e}")
        raise

    try:
        from src.core.skill_tree import SkillTree
        st = SkillTree()
        logger.info(f"[RAG] SkillTree loaded ({time.perf_counter()-t0:.1f}s)")
    except Exception as e:
        logger.warning(f"[RAG] SkillTree preload skipped: {e}")

    # v8.10k: 预热 memory 语义召回用的 Embedder（e5-large 共享单例）——
    # 启动即加载，首个请求不再现场加载模型（"每次首问 30s+ 慢"的根因之一）
    try:
        from src.engine.embedder import Embedder
        e = Embedder()
        e.embed_query("warmup")
        logger.info(f"[RAG] Embedder preloaded ({time.perf_counter()-t0:.1f}s)")
    except Exception as e:
        logger.warning(f"[RAG] Embedder preload skipped: {e}")

    total = time.perf_counter() - t0
    logger.info(f"[RAG] Preload complete ({total:.1f}s)")
