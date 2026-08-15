import hashlib
import json
import logging
import os
import pickle
import re
import time
import threading
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from qdrant_client import QdrantClient
from qdrant_client.http import models
from src.config import settings
from src.engine.embedder import Embedder
from src.engine.reranker import Reranker
from src.retrieval.bm25 import (
    BM25Plus, rrf_fuse,
    bm25_to_cache_dict, bm25_from_cache_dict, BM25_CACHE_FORMAT,
)

logger = logging.getLogger(__name__)

# v8.6 (书 §3.2 离线建索引): BM25 倒排索引持久化——语料不变则启动直接加载，
# 跳过 fit（实测 119k chunk 语料 fit+倒排 ~15s → 加载 ~2s）；查询走倒排加速。
_BM25_CACHE_DIR = Path(__file__).resolve().parent.parent.parent / ".hf_cache" / "bm25"
_BM25_CACHE_KEEP = 3   # 只保留最近 3 份缓存（不同语料指纹各自成文件）


def _bm25_cache_path(fp: str) -> Path:
    return _BM25_CACHE_DIR / f"bm25_v{BM25_CACHE_FORMAT}_{fp}.pkl"


def _bm25_fingerprint(texts: List[str], k1: float, b: float, delta: float) -> str:
    """语料指纹 = 全部检索文本的 md5 + BM25 参数（内容变化即失效）。"""
    h = hashlib.md5()
    h.update(f"v{BM25_CACHE_FORMAT}|{k1}|{b}|{delta}".encode("utf-8"))
    for t in texts:
        h.update(t.encode("utf-8", "ignore"))
    return h.hexdigest()


def _load_bm25_cache(fp: str) -> Optional[BM25Plus]:
    p = _bm25_cache_path(fp)
    if not p.exists():
        return None
    try:
        with open(p, "rb") as f:
            data = pickle.load(f)
        if not isinstance(data, dict) or data.get("format") != BM25_CACHE_FORMAT:
            logger.warning(f"[Retriever] BM25 cache format mismatch, rebuilding: {p.name}")
            return None
        return bm25_from_cache_dict(data)
    except Exception as e:
        logger.warning(f"[Retriever] BM25 cache load failed, rebuilding: {e}")
        return None


def _save_bm25_cache(bm: BM25Plus, fp: str) -> None:
    try:
        _BM25_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = _bm25_cache_path(fp).with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(bm25_to_cache_dict(bm), f, protocol=4)
        os.replace(tmp, _bm25_cache_path(fp))
        # 清理旧指纹缓存（保留最新 _BM25_CACHE_KEEP 份）
        try:
            files = sorted(_BM25_CACHE_DIR.glob("bm25_v*.pkl"),
                           key=lambda p: p.stat().st_mtime, reverse=True)
            for old in files[_BM25_CACHE_KEEP:]:
                old.unlink(missing_ok=True)
        except Exception:
            pass
        logger.info(f"[Retriever] BM25 cache saved: {_bm25_cache_path(fp).name} "
                    f"({_bm25_cache_path(fp).stat().st_size // (1024*1024)}MB)")
    except Exception as e:
        logger.warning(f"[Retriever] BM25 cache save failed: {e}")

class MultiBatchRetriever:
    # ========== 单例核心 ==========
    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            logger.debug("[MultiBatchRetriever] 单例已就绪，跳过重复初始化")
            return

        with self._lock:
            if self._initialized:
                return

            logger.info("[MultiBatchRetriever] 首次初始化 - 加载 Qdrant 客户端与 BM25 索引")
            self.data_dir = settings.DATA_DIR
            self.embedder = Embedder()
            self.reranker = Reranker()
            self.batches: Dict[str, Tuple[QdrantClient, str]] = {}
            self.global_chunks: List[Dict] = []
            self.bm25 = BM25Plus()
            self._idx_map = {}
            self.last_empty_reason: str = ""   # v8.3.1: 空结果归因（threshold_blocked / no_match），供工具回传 LLM
            self.failed_batches: Dict[str, str] = {}  # v8.3.4: 向量加载失败批次（lock 冲突/异常），供降级提示
            self.runtime_failed_batches: set = set()  # v8.3.5: 运行期查询失败的批次（动态降级提示）

            try:
                self._load_data()
                self._initialized = True
                logger.info(f"[MultiBatchRetriever] 初始化完成 | 批次: {len(self.batches)} | 块数: {len(self.global_chunks)}")
                if self.failed_batches:
                    logger.error(
                        f"[MultiBatchRetriever] {len(self.failed_batches)} 批次向量库不可用（BM25-only 降级）: "
                        f"{list(self.failed_batches.keys())} | 原因: {list(self.failed_batches.values())[:2]}"
                    )
            except Exception as e:
                logger.error(f"[MultiBatchRetriever] 初始化失败: {e}")
                raise

    def close(self):
        """Release Qdrant file locks so next startup can load all batches.

        v8.4.3 工单10: 关闭后删除本实例持有的 .lock——否则每次启动都触发
        "cleaned stale lock" 噪音（进程退出不释放锁文件）。
        """
        for batch_name, (client, _) in list(self.batches.items()):
            try:
                client.close()
            except Exception:
                pass
        try:
            if self.data_dir.exists():
                for batch_dir in sorted(self.data_dir.iterdir()):
                    lock = batch_dir / "qdrant_data" / ".lock"
                    if lock.exists():
                        try:
                            lock.unlink()
                        except Exception:
                            pass
        except Exception:
            pass
        self.batches.clear()
        self.global_chunks.clear()
        self._initialized = False
        logger.info("[MultiBatchRetriever] closed all Qdrant clients")

    # ========== 原有业务逻辑 ==========
    def _clean_stale_locks(self) -> set:
        """清理陈旧 .lock 文件；返回被另一实例占用的批次集合（v8.3.4）。

        Qdrant local 模式每个存储目录一个 .lock。崩溃残留锁可删；
        若 unlink 失败（PermissionError = 另一实例正持有）→ 标记冲突批次，
        调用方跳过该批次的 Qdrant 加载（不再产生无意义的失败噪音与竞态）。
        """
        conflict = set()
        if not self.data_dir.exists():
            return conflict
        for batch_dir in sorted(self.data_dir.iterdir()):
            if not batch_dir.is_dir():
                continue
            lock = batch_dir / "qdrant_data" / ".lock"
            if not lock.exists():
                continue
            try:
                lock.unlink()
                logger.info(f"[MultiBatchRetriever] cleaned stale lock: {lock}")
            except PermissionError:
                conflict.add(batch_dir.name)
                logger.error(
                    f"[MultiBatchRetriever] Qdrant 数据目录被另一实例占用: {lock} — "
                    f"local 模式单实例限制，该批次向量检索不可用（BM25 兜底），"
                    f"请先停止其他服务进程再启动以获得完整检索"
                )
            except Exception as e:
                logger.debug(f"[MultiBatchRetriever] lock cleanup: {lock}: {e}")
        return conflict

    def _load_data(self):
        lock_conflict = self._clean_stale_locks()
        self.failed_batches = {name: "另一实例占用 (lock)" for name in lock_conflict}
        if not self.data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")
        logger.info(f"Scanning multi-source batches in {self.data_dir}...")
        for batch_dir in sorted(self.data_dir.iterdir()):
            if not batch_dir.is_dir(): continue
            qdrant_path = batch_dir / "qdrant_data"
            chunks_path = batch_dir / "chunks" / "chunks.jsonl"
            if qdrant_path.exists() and chunks_path.exists():
                batch_name = batch_dir.name
                if batch_name in lock_conflict:
                    # v8.3.4: 另一实例占用 → 跳过 Qdrant（避免重复失败日志），BM25 仍可构建
                    with open(chunks_path, encoding="utf-8") as f:
                        for local_idx, line in enumerate(f):
                            if not line.strip(): continue
                            chunk = json.loads(line)
                            chunk["_batch"] = batch_name
                            chunk["_global_idx"] = len(self.global_chunks)
                            self.global_chunks.append(chunk)
                    continue
                try:
                    client = QdrantClient(path=str(qdrant_path), timeout=settings.QDRANT_TIMEOUT, prefer_grpc=False)
                    colls = client.get_collections().collections
                    coll_name = colls[0].name if colls else "citrus_literature"
                    self.batches[batch_name] = (client, coll_name)
                except Exception as e:
                    # v8.3.4: 瞬时窗口重试一次（删锁后 Qdrant 内部清理延迟/竞态）
                    msg = str(e)
                    if "already accessed" in msg or "lock" in msg.lower():
                        time.sleep(2)
                        try:
                            client = QdrantClient(path=str(qdrant_path), timeout=settings.QDRANT_TIMEOUT, prefer_grpc=False)
                            colls = client.get_collections().collections
                            coll_name = colls[0].name if colls else "citrus_literature"
                            self.batches[batch_name] = (client, coll_name)
                        except Exception as e2:
                            self.failed_batches[batch_name] = str(e2)[:120]
                            logger.error(f"Failed to load Qdrant for {batch_name}: {e2}")
                            continue
                    else:
                        self.failed_batches[batch_name] = msg[:120]
                        logger.warning(f"Failed to load Qdrant for {batch_name}: {e}")
                        continue
                with open(chunks_path, encoding="utf-8") as f:
                    for local_idx, line in enumerate(f):
                        if not line.strip(): continue
                        chunk = json.loads(line)
                        chunk["_batch"] = batch_name
                        chunk["_global_idx"] = len(self.global_chunks)
                        self.global_chunks.append(chunk)
        logger.info(f"Loaded {len(self.batches)} batches, {len(self.global_chunks)} total chunks.")
        if not self.batches:
            logger.error(
                "[MultiBatchRetriever] 所有批次 Qdrant 加载失败 — 向量检索不可用，仅 BM25 降级运行。"
                "请检查: ① data/ 目录完整性 ② 是否有其他服务实例占用（local 模式单实例限制）"
            )
        if self.global_chunks:
            self._enrich_all_metadata()
            logger.info("Building Global BM25 Index...")
            texts = [f"{c.get('section_name','')} {c.get('text','')}" for c in self.global_chunks]
            # v8.6: 缓存优先（语料指纹不变 → 直接加载倒排索引，省 ~13s 启动重建）
            fp = _bm25_fingerprint(texts, self.bm25.k1, self.bm25.b, self.bm25.delta)
            cached = _load_bm25_cache(fp)
            if cached is not None:
                self.bm25 = cached
                logger.info(f"[Retriever] BM25 index loaded from cache "
                            f"({_bm25_cache_path(fp).name})")
            else:
                t0 = time.time()
                self.bm25.fit(texts)
                logger.info(f"[Retriever] BM25 fit done: {time.time()-t0:.1f}s "
                            f"({len(self.global_chunks)} chunks, {len(self.bm25.idf)} terms)")
                _save_bm25_cache(self.bm25, fp)
            # AG-11 修正: chunk_index 为论文内编号, 全局唯一键 = (paper_id, chunk_index)
            self._idx_map = {
                (c.get("paper_id", ""), c.get("chunk_index")): c["_global_idx"]
                for c in self.global_chunks
            }
            self._verify_idx_map()

    def _verify_idx_map(self):
        """AG-11: 映射完整性自检 — 抽样 qdrant 点，按 payload (paper_id, chunk_index) 匹配率告警。"""
        import random
        for batch_name, (client, coll_name) in self.batches.items():
            try:
                pts, _ = client.scroll(collection_name=coll_name, limit=200, with_payload=True)
                if not pts:
                    continue
                matched = sum(
                    1 for p in pts
                    if ((p.payload or {}).get("paper_id", ""), (p.payload or {}).get("chunk_index"))
                    in self._idx_map
                )
                rate = matched / len(pts)
                if rate < 0.95:
                    logger.warning(
                        f"[Retriever] idx_map match rate LOW: batch={batch_name} "
                        f"{matched}/{len(pts)} ({rate:.1%}) — 检索将跳过未映射 chunk"
                    )
                else:
                    logger.info(f"[Retriever] idx_map ok: batch={batch_name} {matched}/{len(pts)} ({rate:.1%})")
            except Exception as e:
                logger.warning(f"[Retriever] idx_map self-check failed for {batch_name}: {e}")

    def _enrich_all_metadata(self):
        """Extract title/authors/year from chunk text since external pipeline didn't store them."""
        from collections import defaultdict
        papers = defaultdict(list)
        for c in self.global_chunks:
            papers[c.get("paper_id", "")].append(c)

        enriched = 0
        _YEAR_RE = re.compile(r'(?:19|20)\d{2}')
        _GENERIC_SECTIONS = {
            "introduction", "abstract", "discussion", "results", "conclusion",
            "conclusions", "references", "acknowledgements", "acknowledgment",
            "keywords", "citation", "keywords:", "__preamble__",
            "materials and methods", "methods", "methodology",
            "experimental", "results and discussion", "background",
            "related works", "related work",
            "supplementary", "appendix", "author contributions",
            "conflict of interest", "data availability", "funding",
            "materials", "material and methods", "statistical analysis",
            "experimental procedures", "experimental section",
            "declarations", "abbreviations", "nomenclature",
            "one-sentence summary", "short running title", "running title",
        }

        for pid, chunks in papers.items():
            if not pid:
                continue
            preamble = next((c for c in chunks if c.get("section_name") == "__preamble__"), None)

            # --- title ---
            title = None
            _NUM_PREFIX = re.compile(r'^[\d]+(\.[\d]+)*\s+')
            for c in chunks:
                sn = (c.get("section_name") or "").strip()
                sn_clean = _NUM_PREFIX.sub('', sn).strip()
                if sn_clean and sn_clean.lower() not in _GENERIC_SECTIONS and len(sn_clean) > 12:
                    title = sn_clean
                    break
            if not title:
                for c in chunks[:20]:
                    sn = (c.get("section_name") or "").strip()
                    sn_clean = _NUM_PREFIX.sub('', sn).strip()
                    if sn_clean and sn_clean.lower() not in _GENERIC_SECTIONS:
                        title = sn_clean
                        break
            if not title:
                title = chunks[0].get("doi", pid) if chunks else pid

            # --- year ---
            year = None
            for c in chunks[:3]:
                text = c.get("text", "")
                m = _YEAR_RE.search(text[:500])
                if m:
                    y = m.group(0)
                    if 1900 <= int(y) <= 2030:
                        year = y
                        break
            if not year:
                m = _YEAR_RE.search(pid)
                if m:
                    year = m.group(0)

            for c in chunks:
                if title and not c.get("title"):
                    c["title"] = title
                if year and not c.get("year"):
                    c["year"] = year
            enriched += 1

        # v8.4.3 工单10: 空 paper_id 论文被跳过 → title 缺失兜底（首行正文/section/DOI）
        for c in self.global_chunks:
            if c.get("title"):
                continue
            first_line = next(
                (l.strip() for l in str(c.get("text", "")).splitlines()
                 if l.strip() and len(l.strip()) > 8), None)
            c["title"] = (first_line
                          or str(c.get("section_name") or "").strip()
                          or c.get("doi") or "Untitled")[:120]

        with_title = sum(1 for c in self.global_chunks if c.get("title"))
        with_year = sum(1 for c in self.global_chunks if c.get("year"))
        logger.info(
            f"[MultiBatchRetriever] enriched metadata for {enriched}/{len(papers)} papers "
            f"(title:{with_title}/{len(self.global_chunks)} year:{with_year}/{len(self.global_chunks)})"
        )

    def _search_qdrant(self, batch_name: str, client: QdrantClient, coll_name: str,
                       query_vec: List[float], limit: int) -> List[Tuple[int, float]]:
        try:
            res = client.query_points(collection_name=coll_name, query=query_vec, limit=limit)
            out = []
            for p in res.points:
                # AG-11 修正: point id 为随机整数, 必须用 payload 的 (paper_id, chunk_index) 定位
                pl = p.payload or {}
                g = self._idx_map.get((pl.get("paper_id", ""), pl.get("chunk_index")))
                if g is None:
                    logger.warning(f"[Retriever] unmapped point {p.id} in {batch_name} — skipped")
                    continue
                out.append((self.global_chunks[g]["_global_idx"], p.score))
            return out
        except Exception as e:
            # v8.3.5: 运行时失败累计 → 空结果归因联动（用户/LLM 可感知降级）
            self.runtime_failed_batches.add(batch_name)
            logger.error(f"Qdrant search failed for {batch_name}: {e}")
            return []

    def search_multi(self, queries: List[str]) -> List[Dict]:
        if not queries: return []
        _t0 = time.time()
        logger.info(f"[Retriever] 启动多路并发检索 | 总 Query 数: {len(queries)}")

        t_embed = time.time()
        query_vecs = self.embedder.embed_docs(queries)
        dt_embed = (time.time() - t_embed) * 1000

        all_vector_hits = []
        if not self.batches:
            logger.warning("[Retriever] 无可用 Qdrant 批次，降级为 BM25 纯文本检索")
        t_qdrant = time.time()
        workers = max(1, len(self.batches) * len(queries))
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futures = [ex.submit(self._search_qdrant, name, client, coll, vec, settings.TOP_K_VECTOR)
                       for vec in query_vecs for name, (client, coll) in self.batches.items()]
            for f in as_completed(futures): all_vector_hits.extend(f.result())
        dt_qdrant = (time.time() - t_qdrant) * 1000

        t_bm25 = time.time()
        all_bm25_hits = [hit for q in queries for hit in self.bm25.top_k(q, k=settings.TOP_K_BM25)]
        dt_bm25 = (time.time() - t_bm25) * 1000

        all_vector_hits.sort(key=lambda x: x[1], reverse=True)
        all_bm25_hits.sort(key=lambda x: x[1], reverse=True)
        v_seen, b_seen = set(), set()
        unique_v = [(idx, s) for idx, s in all_vector_hits if not (idx in v_seen or v_seen.add(idx))]
        unique_b = [(idx, s) for idx, s in all_bm25_hits if not (idx in b_seen or b_seen.add(idx))]
        fused = rrf_fuse(unique_v, unique_b, k=settings.RRF_K)
        candidates = [self.global_chunks[idx] for idx, _ in fused[:settings.TOP_K_FINAL * 2]]
        primary_query = queries[0]
        t_rerank = time.time()
        reranked = self.reranker.rerank(primary_query, candidates, top_k=settings.TOP_K_FINAL)
        dt_rerank = (time.time() - t_rerank) * 1000
        if not reranked:
            self.last_empty_reason = "no_match"
            return []
        top_score = reranked[0].get("rerank_score", 0)
        dynamic_thresh = top_score * settings.DYNAMIC_THRESHOLD_RATIO
        final_threshold = max(settings.RERANK_THRESHOLD, dynamic_thresh)
        passed = [c for c in reranked if c.get("rerank_score", 0) >= final_threshold]
        if not passed:
            logger.warning(f"动态阈值 {final_threshold:.4f} 拦截所有结果，触发模型知识兜底")
            self.last_empty_reason = "threshold_blocked"
            return []
        total_ms = (time.time() - _t0) * 1000
        logger.info(f"[Retriever] rerank={dt_rerank:.0f}ms total={total_ms:.0f}ms | top_score={top_score:.4f} threshold={final_threshold:.4f} | passed={len(passed)}/{len(reranked)}")
        # v8.7: 统一检索过滤日志（合并 retrieval/ 与 debug_filter/——原 debug_filter
        # 的 filtered 参数恒为空，被拦截明细从未记录；现在真实计算并记录）
        from src.logger import log_retrieval
        filtered = [c for c in reranked if c.get("rerank_score", 0) < final_threshold]
        log_retrieval(primary_query, reranked, final_threshold, passed, filtered,
                      elapsed=time.time() - _t0, extra={"mode": "search_multi"})
        # v8.7: 执行过程日志（business.log）补检索事件——此前检索完成无业务事件线
        try:
            from src.core.business_logger import blog
            blog("retrieval_done", mode="search_multi", queries=len(queries),
                 docs=len(passed), filtered=len(filtered), ms=int(total_ms))
        except Exception:
            pass
        return passed

    def search(self, query: str) -> List[Dict]:
        return self.search_multi([query])

    def search_hyde(self, original_query: str, hyde_answer: str) -> List[Dict]:
        """HyDE hybrid search: dual dense (original + hyde) + BM25 + RRF + Reranker(original_query)."""
        _t0 = time.time()
        logger.info(f"[Retriever] HyDE hybrid search | query_len={len(original_query)} hyde_len={len(hyde_answer)}")

        def _emit(msg):
            try:
                from src.core.progress_bus import emit_progress
                emit_progress("tool_progress", {"message": msg, "tool_call_id": ""})
            except Exception: pass

        _emit("向量化查询...")
        orig_vec = self.embedder.embed_query(original_query)
        hyde_vec = self.embedder.embed_query(hyde_answer)

        _emit(f"并发检索 Qdrant ({len(self.batches)} 批次)...")
        orig_v_hits: list[tuple[int, float]] = []
        hyde_v_hits: list[tuple[int, float]] = []
        if self.batches:
            workers = max(1, len(self.batches) * 2)
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures_orig: list = []
                futures_hyde: list = []
                for name, (client, coll) in self.batches.items():
                    futures_orig.append(ex.submit(self._search_qdrant, name, client, coll, orig_vec, settings.TOP_K_VECTOR))
                    futures_hyde.append(ex.submit(self._search_qdrant, name, client, coll, hyde_vec, settings.TOP_K_VECTOR))
                for f in as_completed(futures_orig):
                    orig_v_hits.extend(f.result())
                for f in as_completed(futures_hyde):
                    hyde_v_hits.extend(f.result())

        bm25_hits = self.bm25.top_k(original_query, k=settings.TOP_K_BM25)
        _emit(f"BM25 词法检索完成, 共 {len(orig_v_hits)+len(hyde_v_hits)} 个向量命中 + {len(bm25_hits)} 个词法命中")

        orig_v_hits.sort(key=lambda x: x[1], reverse=True)
        hyde_v_hits.sort(key=lambda x: x[1], reverse=True)
        bm25_hits.sort(key=lambda x: x[1], reverse=True)

        # Dedup each list independently
        def _dedup_hits(hits):
            seen = set()
            return [(idx, s) for idx, s in hits if not (idx in seen or seen.add(idx))]

        u_orig = _dedup_hits(orig_v_hits)
        u_hyde = _dedup_hits(hyde_v_hits)
        u_bm25 = _dedup_hits(bm25_hits)

        # Weighted RRF: weights from config
        w_orig = getattr(settings, 'RRF_WEIGHT_ORIG_DENSE', 1.0)
        w_hyde = getattr(settings, 'RRF_WEIGHT_HYDE_DENSE', 1.0)
        w_bm25 = getattr(settings, 'RRF_WEIGHT_BM25', 1.0)

        fused = rrf_fuse(u_orig, u_hyde, u_bm25, k=settings.RRF_K,
                         weights=[w_orig, w_hyde, w_bm25])
        candidates = [self.global_chunks[idx] for idx, _ in fused[:settings.TOP_K_FINAL * 2]]
        _emit(f"Cross-Encoder 重排序 ({len(candidates)} 篇候选文献)...")

        reranked = self.reranker.rerank(original_query, candidates, top_k=settings.TOP_K_FINAL)
        if not reranked:
            logger.info("[Retriever] HyDE: no reranked results, falling back to original search")
            return self.search(original_query)

        # v8.4.3: 分数分布日志（校准用）；阈值与 RAG 基础路保持一致
        # （0.25 地板 + 动态阈值 max(0.25, top*ratio)）
        scores = sorted(c.get("rerank_score", 0) for c in reranked)
        p50 = scores[len(scores) // 2] if scores else 0
        p90 = scores[int(len(scores) * 0.9)] if scores else 0
        logger.info(
            f"[Retriever] HyDE rerank scores: top={scores[-1] if scores else 0:.4f} "
            f"p50={p50:.4f} p90={p90:.4f} n={len(scores)}")

        top_score = reranked[0].get("rerank_score", 0)
        dynamic_thresh = top_score * settings.DYNAMIC_THRESHOLD_RATIO
        final_threshold = max(settings.RERANK_THRESHOLD, dynamic_thresh)
        passed = [c for c in reranked if c.get("rerank_score", 0) >= final_threshold]
        if not passed:
            logger.warning(f"[Retriever] HyDE: threshold {final_threshold:.4f} blocked all, falling back")
            return self.search(original_query)

        total_ms = (time.time() - _t0) * 1000
        logger.info(f"[Retriever] HyDE done: {len(passed)}/{len(reranked)} passed | {total_ms:.0f}ms")
        # v8.7: HyDE 路径补记统一检索过滤日志（此前该路径只记通过结果、无过滤明细）
        from src.logger import log_retrieval
        filtered = [c for c in reranked if c.get("rerank_score", 0) < final_threshold]
        log_retrieval(original_query, reranked, final_threshold, passed, filtered,
                      elapsed=time.time() - _t0, extra={"mode": "hyde"})
        # v8.7: 执行过程日志补检索事件（HyDE 路径）
        try:
            from src.core.business_logger import blog
            blog("retrieval_done", mode="hyde", queries=1,
                 docs=len(passed), filtered=len(filtered), ms=int(total_ms))
        except Exception:
            pass
        return passed
