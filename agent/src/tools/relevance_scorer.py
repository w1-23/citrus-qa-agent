"""
RelevanceScorer — Web 检索语义相关性过滤
复用系统已有的 BGE Reranker (ONNX cross-encoder)，对 web 结果做精排过滤。
Cross-encoder 比 Embedding + cosine similarity 区分度更高。
"""
import logging
from typing import List

logger = logging.getLogger(__name__)


async def score_web_results(
    query: str,
    results: List[dict],
    threshold: float = 0.3,
) -> List[dict]:
    """Score and filter web results by semantic relevance to the query.

    Uses the system's existing BGE Reranker (cross-encoder) which jointly
    encodes query+passage and outputs relevance scores in [0, 1].

    Args:
        query: The original user query
        results: List of web result dicts with 'title' and 'content' keys
        threshold: Minimum reranker score to keep (default 0.3).
                   BGE reranker scores: 0.0 (irrelevant) ~ 1.0 (perfect match).
                   0.3 is a conservative threshold that catches obvious noise.

    Returns:
        Filtered list of results with 'relevance_score' added to each dict
    """
    import asyncio

    if not results or not query:
        return []

    chunks = [
        {
            "text": f"{r.get('title', '')}. {r.get('content', '')[:800]}",
            "_index": i,
        }
        for i, r in enumerate(results)
    ]

    try:
        def _rerank_all():
            from src.engine.reranker import Reranker
            reranker = Reranker()
            return reranker.rerank(query, chunks, top_k=len(chunks))

        reranked = await asyncio.to_thread(_rerank_all)
    except Exception as e:
        logger.warning(f"[RelevanceScorer] rerank failed: {e}, passing all through")
        return results

    index_map = {c["_index"]: c.get("rerank_score", 0) for c in reranked}

    filtered = []
    for i, r in enumerate(results):
        score = index_map.get(i, 0)
        if score >= threshold:
            r["relevance_score"] = round(float(score), 4)
            filtered.append(r)
        else:
            logger.debug(
                f"[RelevanceScorer] dropped (score={score:.3f} < {threshold}): "
                f"{r.get('title', '?')[:60]}"
            )

    logger.info(
        f"[RelevanceScorer] {len(filtered)}/{len(results)} passed "
        f"(threshold={threshold})"
    )
    return filtered
