import hashlib
import json
import logging
import os
import pickle
import re
import time
import threading
from collections import OrderedDict
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


def _bm25_search_parallel(bm25: BM25Plus, queries: List[str], k: int = 20) -> List[Tuple[int, float]]:
    """v9.1.3: 多查询 BM25 并行 top_k（结果与串行逐位一致）。

    _top_k_inverted 是只读 numpy 向量化路径（无共享写入；doc_lens/inv 均为
    不可变数组）→ 多线程安全；numpy 向量段释放 GIL，多查询并行消除 Python
    累加段串行（9 路实测 5.2s → 预期 ≈2-3x）。顺序与旧串行循环完全一致：
    按 queries 提交顺序取结果拼接（每路 top_k 独立、无共享状态）。
    """
    if len(queries) <= 1:
        return [hit for q in queries for hit in bm25.top_k(q, k=k)]
    workers = min(4, len(queries))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(bm25.top_k, q, k) for q in queries]
        out: List[Tuple[int, float]] = []
        for f in futures:  # 按提交顺序取回 → 与串行逐位一致
            out.extend(f.result())
    return out


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
            self.lance_tables: Dict[str, object] = {}   # v8.9 LanceDB 后端: batch -> table
            # v8.15: 批次 → 证据来源（metadata.json summary.source_type 驱动，可扩展：
            # "UCR citrus variety"→ucr，其余→rag；将来新批次类型只增映射规则）
            self.batch_source: Dict[str, str] = {}
            self.lance_db = None
            self.backend = (settings.RETRIEVAL_BACKEND or "auto").strip().lower()
            # v8.9 auto：优先 LanceDB（data/lancedb 有表），否则回退 Qdrant——
            # 新数据包（lancedb）与旧数据包（qdrant_data）开箱即用
            if self.backend == "auto":
                lance_root = self.data_dir / "lancedb"
                has_lance = lance_root.exists() and any(lance_root.glob("*.lance"))
                self.backend = "lancedb" if has_lance else "qdrant"
                logger.info(f"[Retriever] backend auto -> {self.backend}")
            self.global_chunks: List[Dict] = []
            # v8.11 全文懒加载: 稳态只保留 chunk 元数据（无 text），
            # 文本按 (chunks.jsonl 路径, 行起始字节偏移) 按需读取 + LRU 缓存——
            # 内存从 O(全部chunk全文) 降到 O(元数据)，3 倍语料后稳态基本不涨
            self._text_offsets: List[Tuple[Path, int]] = []   # global_idx -> (路径, 偏移)
            self._text_cache: "OrderedDict[int, str]" = OrderedDict()
            self._text_lock = threading.Lock()
            self._TEXT_CACHE_MAX = 2048   # ~2048 条 × 1KB ≈ 2MB 常驻上限
            self.bm25 = BM25Plus()
            self._idx_map = {}
            self.last_empty_reason: str = ""   # v8.3.1: 空结果归因（threshold_blocked / no_match），供工具回传 LLM
            self.last_stats: Dict[str, int] = {}  # v8.15.3: 最近一次 _fuse_rerank_select 的候选/通过/过滤统计（工具回传 LLM 早停依据）
            self.failed_batches: Dict[str, str] = {}  # v8.3.4: 向量加载失败批次（lock 冲突/异常），供降级提示
            self.runtime_failed_batches: set = set()  # v8.3.5: 运行期查询失败的批次（动态降级提示）

            try:
                self._load_data()
                self._initialized = True
                logger.info(f"[MultiBatchRetriever] 初始化完成 | backend={self.backend} | "
                            f"批次: {len(self.batches) + len(self.lance_tables)} | "
                            f"块数: {len(self.global_chunks)}")
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
        v8.9: LanceDB 后端无需关闭（文件级句柄，无锁）。
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
        self.lance_tables.clear()
        self.lance_db = None
        self.global_chunks.clear()
        self._text_offsets.clear()
        self._text_cache.clear()
        self._initialized = False
        logger.info("[MultiBatchRetriever] closed all vector backends")

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
            chunks_path = batch_dir / "chunks" / "chunks.jsonl"
            if not chunks_path.exists():
                # v8.13-b5b: 兼容新数据包（pipeline1 系）——chunks.jsonl 直接位于批次根目录
                chunks_path = batch_dir / "chunks.jsonl"
            if not chunks_path.exists():
                continue
            batch_name = batch_dir.name
            # v8.15: 批次来源判定（metadata.json → source_type；读失败按本地文献库兜底）
            src = self._detect_batch_source(batch_dir)
            self.batch_source[batch_name] = src
            # 公共：chunks.jsonl → global_chunks（两种后端共用；Qdrant 锁冲突时仅此处可用）
            # v8.11: 二进制逐行扫描并记录每行起始字节偏移（文本模式 tell 返回
            # opaque cookie 不可跨 open seek，必须二进制）——稳态剥离 text，
            # 查询时按偏移懒加载单条文本（见 _load_chunk_text）
            with open(chunks_path, "rb") as f:
                while True:
                    offset = f.tell()
                    line = f.readline()
                    if not line:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    chunk = json.loads(line.decode("utf-8"))
                    chunk["_batch"] = batch_name
                    chunk["_src"] = src
                    chunk["_global_idx"] = len(self.global_chunks)
                    self._text_offsets.append((chunks_path, offset))
                    self.global_chunks.append(chunk)
            if self.backend == "lancedb":
                self._load_lance_batch(batch_name)
                continue
            # ── qdrant 后端 ──
            if batch_name in lock_conflict:
                # v8.3.4: 另一实例占用 → 跳过 Qdrant（避免重复失败日志），BM25 仍可构建
                continue
            qdrant_path = batch_dir / "qdrant_data"
            if not qdrant_path.exists():
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
        logger.info(f"Loaded {len(self.batches) + len(self.lance_tables)} batches, "
                    f"{len(self.global_chunks)} total chunks.")
        if not self.batches and not self.lance_tables:
            logger.error(
                "[MultiBatchRetriever] 所有批次向量后端加载失败 — 向量检索不可用，仅 BM25 降级运行。"
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
            # v8.11 懒加载收尾：元数据/BM25/idx_map 均已就绪，剥离全文只留元数据——
            # 启动期峰值与旧版相同（enrich/fit 需要全文），稳态内存大幅下降
            self.global_chunks = [
                {k: v for k, v in c.items() if k != "text"} for c in self.global_chunks
            ]
            logger.info(f"[Retriever] 全文懒加载就绪: {len(self._text_offsets)} 条偏移 "
                        f"| 稳态仅保留元数据（text 按需读取）")

    @staticmethod
    def _detect_batch_source(batch_dir: Path) -> str:
        """v8.15: 批次证据来源判定（metadata.json → summary.source_type）。

        v9.4 修订（来源按文件夹名透传）: metadata.json 存在且
        summary.source_type 非空 → 原样返回（如 "paper1" / "Citrus varieties1"，
        前端 srcKey 对非内置来源去尾部数字归组为 "paper" / "Citrus varieties"）；
        无元数据 → 返回批次文件夹名（同口径）；读取失败 → 空串兜底（调用方回退 "rag"）。
        旧格式批次（source_type="UCR citrus variety"）原样透传，兼容由
        evidence.src_of 的 UCR 兜底判定与 is_variety_source 保持品种语义。
        """
        try:
            meta_path = batch_dir / "metadata.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                st = str((meta.get("summary") or {}).get("source_type") or "").strip()
                if st:
                    return st
        except Exception as e:
            logger.debug(f"[Retriever] batch source detect failed for {batch_dir.name}: {e}")
        return batch_dir.name

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

    def _load_lance_batch(self, batch_name: str) -> None:
        """v8.9 LanceDB 后端：打开 data/lancedb 中该批次的表（表名=批次名）。

        v9.4: 兼容名含空格的用户批次——LanceDB 表名只允许字母数字/_/-/.，
        空格批次按空格→下划线回退打开（表名 = sanitize(batch_name)）。
        """
        try:
            if self.lance_db is None:
                import lancedb
                self.lance_db = lancedb.connect(str(self.data_dir / "lancedb"))
            candidates = [batch_name]
            sanitized = re.sub(r"\s+", "_", batch_name).strip("_")
            if sanitized and sanitized != batch_name:
                candidates.append(sanitized)
            table = None
            for cand in candidates:
                try:
                    table = self.lance_db.open_table(cand)
                    break
                except Exception:
                    table = None
            if table is None:
                raise FileNotFoundError(f"table not found: {batch_name}")
            self.lance_tables[batch_name] = table
            logger.info(f"[Retriever] LanceDB table loaded: {batch_name} "
                        f"({table.count_rows()} rows)")
        except Exception as e:
            self.failed_batches[batch_name] = str(e)[:120]
            logger.warning(f"Failed to load LanceDB table for {batch_name}: {e}")

    # ========== v8.11 全文懒加载 ==========
    def _read_chunk_text(self, global_idx: int) -> str:
        """按偏移从 chunks.jsonl 读取单条 chunk 的 text（无锁磁盘读，单行 seek）。"""
        path, offset = self._text_offsets[global_idx]
        with open(path, "rb") as f:
            f.seek(offset)
            line = f.readline()
        return json.loads(line.decode("utf-8")).get("text", "")

    def _load_chunk_text(self, global_idx: int) -> str:
        """LRU 文本缓存（线程安全；double-check：锁内查缓存，锁外读盘）。

        查询只用到每轮 ~20 条候选文本（top_k_final*2），缓存 2048 条足够
        覆盖多轮对话/同批文献复用；第二次起命中内存，额外延迟 ≈ 0。
        """
        with self._text_lock:
            if global_idx in self._text_cache:
                self._text_cache.move_to_end(global_idx)
                return self._text_cache[global_idx]
        text = self._read_chunk_text(global_idx)
        with self._text_lock:
            self._text_cache[global_idx] = text
            self._text_cache.move_to_end(global_idx)
            while len(self._text_cache) > self._TEXT_CACHE_MAX:
                self._text_cache.popitem(last=False)
        return text

    def _chunk_full(self, global_idx: int) -> Dict:
        """懒加载组装完整 chunk（元数据 + 文本），供 rerank/下游使用——
        返回字段集与懒加载前 global_chunks 元素完全一致。"""
        c = dict(self.global_chunks[global_idx])
        c["text"] = self._load_chunk_text(global_idx)
        return c

    def _vector_search(self, batch_name: str, query_vec: List[float],
                       limit: int) -> List[Tuple[int, float]]:
        """按后端分派向量检索（v8.9）。返回 [(global_idx, score)]，score 越大越相关。"""
        if self.backend == "lancedb":
            table = self.lance_tables.get(batch_name)
            if table is None:
                return []
            return self._search_lance(batch_name, table, query_vec, limit)
        item = self.batches.get(batch_name)
        if item is None:
            return []
        client, coll = item
        return self._search_qdrant(batch_name, client, coll, query_vec, limit)

    def _search_lance(self, batch_name: str, table, query_vec: List[float],
                      limit: int) -> List[Tuple[int, float]]:
        try:
            # v8.9: 与 Qdrant 同口径 cosine 度量——由建索引时 metric="cosine" 决定
            # （LanceDB 0.37 search() 不支持查询时指定 metric；无索引 flat 默认 L2）
            res = table.search(query_vec).limit(limit).to_list()
            out = []
            for r in res:
                # AG-11 同款定位：LanceDB 行内 (paper_id, chunk_index) → global_idx
                g = self._idx_map.get((str(r.get("paper_id", "")),
                                       int(r.get("chunk_index", -1))))
                if g is None:
                    continue
                # cosine 度量下 _distance = 1 - cos_sim → 相似度 = 1 - _distance
                score = 1.0 - float(r.get("_distance", 0.0))
                out.append((self.global_chunks[g]["_global_idx"], score))
            return out
        except Exception as e:
            # v8.3.5: 运行时失败累计 → 空结果归因联动（用户/LLM 可感知降级）
            self.runtime_failed_batches.add(batch_name)
            logger.error(f"LanceDB search failed for {batch_name}: {e}")
            return []

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

    # ========== v8.13-b4b 检索后段单一出口（D-4 双管道收敛点）==========
    # search_multi / search_hyde 只生成命中流（embedding/vector/BM25），RRF 权重、
    # 去重、动态阈值、候选预算与三级日志（diag/retrieval/business）统一在此——
    # 此前两管道各写一遍已漂移过一次（v8.10r 注释自证：config rrf_weights.*
    # 一度对 search_multi 无感）。
    def _fuse_rerank_select(self, *, streams: List[List[Tuple[int, float]]],
                            weights: List[float], stream_labels: List[str],
                            rerank_query: str, mode: str, t0: float,
                            stage_ms: Dict[str, float], n_queries: int = 1,
                            fallback_query: Optional[str] = None,
                            original_query: Optional[str] = None) -> List[Dict]:
        """去重 → RRF → 候选全文 → rerank → 动态阈值 → 空归因 → 日志，返回 passed 证据。
        """
        # 每流独立「分数降序 → 按 global_idx 去重（保留最高分）」
        deduped: List[List[Tuple[int, float]]] = []
        for hits in streams:
            hits = sorted(hits, key=lambda x: x[1], reverse=True)
            seen: set = set()
            deduped.append([
                (idx, s) for idx, s in hits
                if not (idx in seen or seen.add(idx))
            ])

        fused = rrf_fuse(*deduped, k=settings.RRF_K, weights=weights)
        # v9.4 (P0#1): 候选窗口参数化——原 top_k_final*2 隐式硬编码，现由
        # config.yaml retrieval.candidate_window 控制（默认 20 = 10*2，行为不变）
        candidates = [self._chunk_full(idx)
                      for idx, _ in fused[:settings.CANDIDATE_WINDOW]]

        t_rerank = time.time()
        reranked = self.reranker.rerank(rerank_query, candidates,
                                        top_k=settings.TOP_K_FINAL)
        dt_rerank = (time.time() - t_rerank) * 1000

        if not reranked:
            logger.info(f"[Retriever] {mode}: no reranked results")
            self.last_empty_reason = "no_match"
            if fallback_query is not None:
                return self.search(fallback_query)
            return []

        top_score = reranked[0].get("rerank_score", 0)
        dynamic_thresh = top_score * settings.DYNAMIC_THRESHOLD_RATIO
        final_threshold = max(settings.RERANK_THRESHOLD, dynamic_thresh)
        passed = [c for c in reranked if c.get("rerank_score", 0) >= final_threshold]
        if not passed:
            logger.warning(f"[Retriever] {mode}: 动态阈值 {final_threshold:.4f} 拦截所有结果")
            self.last_empty_reason = "threshold_blocked"
            if fallback_query is not None:
                return self.search(fallback_query)
            return []

        total_ms = (time.time() - t0) * 1000
        scores = sorted(c.get("rerank_score", 0) for c in reranked)
        p50 = scores[len(scores) // 2] if scores else 0
        p90 = scores[int(len(scores) * 0.9)] if scores else 0
        logger.info(
            f"[Retriever] {mode}: rerank={dt_rerank:.0f}ms total={total_ms:.0f}ms "
            f"| top={top_score:.4f} p50={p50:.4f} p90={p90:.4f} n={len(scores)} "
            f"| threshold={final_threshold:.4f} passed={len(passed)}/{len(reranked)}"
        )
        try:
            from src.core.diag import diag
            diag("retrieval_stages", mode=mode, queries=n_queries,
                 embed_ms=round(stage_ms.get("embed", 0.0), 1),
                 vector_ms=round(stage_ms.get("vector", 0.0), 1),
                 bm25_ms=round(stage_ms.get("bm25", 0.0), 1),
                 rerank_ms=round(dt_rerank, 1),
                 total_ms=round(total_ms, 1),
                 **{label: len(s) for label, s in zip(stream_labels, deduped)},
                 candidates=len(candidates), passed=len(passed),
                 top_score=round(top_score, 4), threshold=round(final_threshold, 4))
        except Exception:
            pass
        # v8.7: 统一检索过滤日志（合并 retrieval/ 与 debug_filter/——真实计算被拦截明细）
        from src.logger import log_retrieval
        filtered = [c for c in reranked if c.get("rerank_score", 0) < final_threshold]
        log_retrieval(rerank_query, reranked, final_threshold, passed, filtered,
                      elapsed=time.time() - t0, extra={"mode": mode})
        try:
            from src.core.business_logger import blog
            blog("retrieval_done", mode=mode, queries=n_queries,
                 docs=len(passed), filtered=len(filtered), ms=int(total_ms))
        except Exception:
            pass
        # v8.15.3: 候选/通过统计（决策器早停依据：filtered 占比高 → 角度相关性低）。
        # 附带本次 rerank_query 作防串号（同轮并发检索共享单例，取回时核对 query）。
        # v9.2: 记录 original_query（用户原文）而非 rerank_query——默认 HyDE 主路径
        # queries[0]=生成段落 ≠ 用户原文，工具侧 expect_query=用户原文比对恒不等，
        # 早停统计被防串号校验误杀（HyDE 关闭时反而生效）。防串号语义保留：
        # 取回时仍核对用户查询归属。
        self.last_stats = {"candidates": len(candidates), "passed": len(passed),
                           "filtered": len(filtered),
                           "query": original_query or rerank_query}
        return passed

    def search_multi(self, queries: List[str],
                     original_query: str = "",
                     rerank_query: str | None = None) -> List[Dict]:
        """多路并发检索。

        v8.17 修订：不做检索层来源路由/加权（数据库混合存储无物理分区，两阶段
        路由在架构上不可行）——UCR 优先改由 evidence_report 组装时聚拢置前
        （元数据标记层）+ retrieve-agent Prompt 引导（模型生成检索词倾向 UCR），
        保持各批次融合公平性。

        v9.2: 新增 original_query 入参（用户原文）——统计归属按用户查询记录，
        修复默认 HyDE 主路径上检索早停统计被防串号校验误杀的问题。

        v9.4b (#2 端到端 C 组)：新增 rerank_query 入参（可选）——None 时沿用
        queries[0]（生产行为不变）；显式传入时覆盖精排查询（论文实验 1b 揭示
        rerank_query 用 HyDE 段为主性能损失源，C 组=多路召回面 + 原始查询精排）。
        """
        if not queries:
            return []
        _t0 = time.time()
        logger.info(f"[Retriever] 启动多路并发检索 | 总 Query 数: {len(queries)}")

        # v8.15 BM25 并行化：BM25 不依赖向量 embedding——先提交独立线程执行，
        # 与 embed（ONNX 释放 GIL）/向量检索重叠，融合前取回（省 ≈ min(bm25, embed) 延迟）
        # v9.1.3（用户日志: 9 路查询 bm25_ms=5229s 为本地检索大头）: 查询间再并行——
        # BM25 为只读 numpy 向量化路径（多线程安全），多查询并行可再获 ≈2-3x 收益，
        # 结果与串行逐位一致（每路 top_k 独立、无共享状态）。
        t_bm25 = time.time()
        with ThreadPoolExecutor(max_workers=1) as _bm25_ex:
            _bm25_fut = _bm25_ex.submit(
                lambda: _bm25_search_parallel(self.bm25, queries, k=settings.TOP_K_BM25))

            t_embed = time.time()
            # v8.14-bugfix(2026-08-20): 查询编码必须走 embed_query（带 "query: " 前缀，E5 训练分布
            # 一致）；此前误用 embed_docs → 查询向量缺前缀，dense 检索整体降分（21 号 l1-004 2/3→3/3）
            query_vecs = [self.embedder.embed_query(q) for q in queries]
            dt_embed = (time.time() - t_embed) * 1000

            if not self.batches and not self.lance_tables:
                logger.warning("[Retriever] 无可用向量后端，降级为 BM25 纯文本检索")
            t_vector = time.time()
            batch_names = list(self.lance_tables.keys()) if self.backend == "lancedb" \
                else list(self.batches.keys())
            all_vector_hits: List[Tuple[int, float]] = []
            workers = max(1, len(batch_names) * len(queries))
            with ThreadPoolExecutor(max_workers=workers) as ex:
                futures = [ex.submit(self._vector_search, name, vec, settings.TOP_K_VECTOR)
                           for vec in query_vecs for name in batch_names]
                for f in as_completed(futures):
                    all_vector_hits.extend(f.result())
            dt_vector = (time.time() - t_vector) * 1000

            all_bm25_hits = _bm25_fut.result()
        dt_bm25 = (time.time() - t_bm25) * 1000

        return self._fuse_rerank_select(
            streams=[all_vector_hits, all_bm25_hits],
            weights=[
                getattr(settings, 'RRF_WEIGHT_ORIG_DENSE', 1.0),
                getattr(settings, 'RRF_WEIGHT_BM25', 1.0),
            ],
            stream_labels=["unique_v", "unique_b"],
            rerank_query=rerank_query or original_query or queries[0],
            mode="search_multi",
            t0=_t0,
            stage_ms={"embed": dt_embed, "vector": dt_vector, "bm25": dt_bm25},
            n_queries=len(queries),
            original_query=original_query,
        )

    def search(self, query: str) -> List[Dict]:
        # v9.2: 单查询路径同样登记用户原文（保持非 HyDE 路径统计校验不回归）
        return self.search_multi([query], original_query=query)

    def search_hyde(self, original_query: str, hyde_answer: str) -> List[Dict]:
        """HyDE hybrid search: dual dense (original + hyde) + BM25 + RRF + Reranker(original_query).

        v8.13-b4b: 仅保留命中流生成（原/假想双 dense + BM25），RRF 融合/去重/阈值/
        日志统一走 _fuse_rerank_select；阈值拦截或无 rerank 结果时回退基础检索。
        """
        _t0 = time.time()
        logger.info(f"[Retriever] HyDE hybrid search | query_len={len(original_query)} hyde_len={len(hyde_answer)}")

        def _emit(msg):
            try:
                from src.core.progress_bus import emit_progress
                emit_progress("tool_progress", {"message": msg, "tool_call_id": ""})
            except Exception:
                pass

        _emit("向量化查询...")
        t_embed = time.time()
        # v8.15 BM25 并行化：BM25 与 HyDE 双路 embedding 重叠执行（ONNX 释放 GIL）
        t_bm25 = time.time()
        with ThreadPoolExecutor(max_workers=1) as _bm25_ex:
            _bm25_fut = _bm25_ex.submit(
                lambda: self.bm25.top_k(original_query, k=settings.TOP_K_BM25))
            orig_vec = self.embedder.embed_query(original_query)
            hyde_vec = self.embedder.embed_query(hyde_answer)
            dt_embed = (time.time() - t_embed) * 1000

            batch_names = list(self.lance_tables.keys()) if self.backend == "lancedb" \
                else list(self.batches.keys())
            _emit(f"并发检索向量后端 ({len(batch_names)} 批次)...")
            t_vector = time.time()
            orig_v_hits: List[Tuple[int, float]] = []
            hyde_v_hits: List[Tuple[int, float]] = []
            if batch_names:
                workers = max(1, len(batch_names) * 2)
                with ThreadPoolExecutor(max_workers=workers) as ex:
                    futures_orig: list = []
                    futures_hyde: list = []
                    for name in batch_names:
                        futures_orig.append(ex.submit(
                            self._vector_search, name, orig_vec, settings.TOP_K_VECTOR))
                        futures_hyde.append(ex.submit(
                            self._vector_search, name, hyde_vec, settings.TOP_K_VECTOR))
                    for f in as_completed(futures_orig):
                        orig_v_hits.extend(f.result())
                    for f in as_completed(futures_hyde):
                        hyde_v_hits.extend(f.result())
            dt_vector = (time.time() - t_vector) * 1000

            bm25_hits = _bm25_fut.result()
        dt_bm25 = (time.time() - t_bm25) * 1000
        _emit(f"BM25 词法检索完成, 共 {len(orig_v_hits)+len(hyde_v_hits)} 个向量命中 + {len(bm25_hits)} 个词法命中")

        return self._fuse_rerank_select(
            streams=[orig_v_hits, hyde_v_hits, bm25_hits],
            weights=[
                getattr(settings, 'RRF_WEIGHT_ORIG_DENSE', 1.0),
                getattr(settings, 'RRF_WEIGHT_HYDE_DENSE', 1.0),
                getattr(settings, 'RRF_WEIGHT_BM25', 1.0),
            ],
            stream_labels=["unique_orig", "unique_hyde", "unique_bm25"],
            rerank_query=original_query,
            mode="hyde",
            t0=_t0,
            stage_ms={"embed": dt_embed, "vector": dt_vector, "bm25": dt_bm25},
            n_queries=1,
            fallback_query=original_query,
        )
