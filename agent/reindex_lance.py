# -*- coding: utf-8 -*-
"""v8.13-b5b qdrant 数据包 → LanceDB 表重建工具（复用既有向量 + 缺口补嵌入）。

把 data/<batch>/ 下的文献包（旧包 chunks/chunks.jsonl 或新包根目录 chunks.jsonl）
重建为 data/lancedb/<batch>.lance 表，供 lancedb 向量后端直接检索：
  - 优先复用批次自带 qdrant_data 里的既有向量（免重复嵌入，分钟级）；
  - qdrant 缺失/损坏时仅补嵌入缺口块；--no-qdrant 强制整体重嵌入；
  - 表结构与 ingest.py / multi_retriever._load_lance_batch 完全一致
    (vector: float32[1024], paper_id: string, chunk_index: int64, IVF_HNSW cosine)。

用法（在 agent/ 目录，用 rag-agent 环境）:
    rag-agent\\python.exe reindex_lance.py --batch xrz
    rag-agent\\python.exe reindex_lance.py --batch 1-1200 --no-qdrant   # qdrant 损坏时
    rag-agent\\python.exe reindex_lance.py --all                        # 全部批次
"""
import argparse
import json
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def find_chunks(batch_dir: Path):
    """兼容新旧包布局：chunks/chunks.jsonl（旧）或根目录 chunks.jsonl（新）。"""
    for rel in ("chunks", "chunks.jsonl"):
        p = batch_dir / rel
        if p.exists() and p.is_file():
            return p
    # 旧包 chunks 是目录，下面才是 chunks.jsonl
    p = batch_dir / "chunks" / "chunks.jsonl"
    if p.exists():
        return p
    return None


def derive_paper_id(c: dict) -> str:
    """稳定 paper_id（幂等）：优先现有字段；缺失时按 variety/registry/source 派生。

    v8.14.1: categories-cn（UCR 品种库）等新包 chunk 无 paper_id——直接用
    ("", chunk_index) 会让跨品种 chunk_index 互相覆盖（索引塌缩 + 向量错配）。
    派生后 (paper_id, chunk_index) 恢复全局唯一，检索器 idx_map 同口径。
    """
    pid = str(c.get("paper_id") or "").strip()
    if pid:
        return pid
    base = str(c.get("variety_id") or c.get("registry_id") or c.get("source_file") or "")
    slug = re.sub(r"[^A-Za-z0-9]+", "_", base).lower()
    slug = re.sub(r"_+", "_", slug).strip("_")[:64]
    if slug:
        return f"doc_{slug}"
    return f"chunk_{abs(hash(str(c.get('text', ''))))}"


def load_chunk_index(chunks_path: Path) -> dict:
    """(paper_id, chunk_index) -> 拼接文本（与检索器 section+text 口径一致）。"""
    index = {}
    with open(chunks_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            key = (derive_paper_id(c), int(c.get("chunk_index", -1)))
            index[key] = (c.get("section_name", "") + " " + c.get("text", ""))[:2000]
    return index


def read_qdrant_vectors(qdrant_path: Path) -> dict:
    """(paper_id, chunk_index) -> np.float32 vector；库损坏时向上抛异常。

    立即转 float32（Python list 会放大 ~7 倍内存：57k x 1024 x 28B ≈ 1.6GB）。
    """
    import numpy as np
    from qdrant_client import QdrantClient
    client = QdrantClient(path=str(qdrant_path))
    try:
        coll = client.get_collections().collections[0].name
        vecs = {}
        offset = None
        while True:
            pts, offset = client.scroll(coll, limit=500, offset=offset, with_vectors=True)
            for p in pts:
                pl = p.payload or {}
                key = (derive_paper_id(pl), int(pl.get("chunk_index", -1)))
                vecs[key] = np.asarray(p.vector, dtype=np.float32)
            if not offset:
                break
    finally:
        client.close()
    return vecs


def main():
    ap = argparse.ArgumentParser(description="qdrant 数据包 → LanceDB 表重建")
    ap.add_argument("--batch", action="append", default=[], help="批次名（可多次）")
    ap.add_argument("--all", action="store_true", help="重建 data/ 下所有含 chunks.jsonl 的批次")
    ap.add_argument("--no-qdrant", action="store_true",
                    help="忽略 qdrant_data，全部重嵌入（本地模型）")
    args = ap.parse_args()

    import numpy as np
    import lancedb
    from src.engine.embedder import Embedder

    data_dir = ROOT / "data"
    lance_root = data_dir / "lancedb"
    lance_root.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(lance_root))

    if args.all:
        batches = [d.name for d in sorted(data_dir.iterdir())
                   if d.is_dir() and find_chunks(d)]
    else:
        batches = args.batch
    if not batches:
        ap.error("请指定 --batch 或 --all")

    emb = Embedder()
    for name in batches:
        batch_dir = data_dir / name
        chunks_path = find_chunks(batch_dir)
        if chunks_path is None or not batch_dir.is_dir():
            print(f"[SKIP] {name}: batch dir / chunks.jsonl missing")
            continue
        t0 = time.time()
        index = load_chunk_index(chunks_path)
        print(f"[BATCH] {name}: {len(index)} chunks @ {chunks_path.relative_to(data_dir)}")

        existing = {}
        if not args.no_qdrant:
            qp = batch_dir / "qdrant_data"
            if qp.exists():
                try:
                    existing = read_qdrant_vectors(qp)
                    print(f"  reused {len(existing)} qdrant vectors")
                except Exception as e:
                    print(f"  [WARN] qdrant_data unavailable ({str(e)[:90]}) -> full re-embed")

        missing = [(k, t) for k, t in index.items() if k not in existing and t]
        print(f"  need embed {len(missing)} / total {len(index)} chunks")
        rows = []
        for k, _ in index.items():
            if k in existing:
                rows.append({"vector": np.asarray(existing[k], dtype=np.float32),
                             "paper_id": k[0], "chunk_index": k[1]})
        if missing:
            t1 = time.time()
            # 防御：小块补嵌入（64/批）+ 每批 gc——fastembed 大输入会复刻会话，
            # 小输入可把峰值压在「模型+一小批激活」内（实测一次塞 5.8k 会到 ~10GB）
            import gc
            CHUNK = 64
            emb_vecs = []
            for i in range(0, len(missing), CHUNK):
                part = missing[i:i + CHUNK]
                emb_vecs.extend(emb.embed_docs([t for _, t in part]))
                gc.collect()
                if (i // CHUNK) % 20 == 0:
                    print(f"  ... embedded {min(i + CHUNK, len(missing))}/{len(missing)}")
            for (k, _), v in zip(missing, emb_vecs):
                rows.append({"vector": np.asarray(v, dtype=np.float32),
                             "paper_id": k[0], "chunk_index": k[1]})
            del emb_vecs
            print(f"  embed done {time.time()-t1:.0f}s")
        if len(rows) != len(index):
            print(f"  [WARN] rows {len(rows)} != chunks {len(index)} (empty-text chunks skipped)")
        # 建索引前释放大对象（模型/文本索引/向量表）——避免与服务进程并发时撞内存
        del existing, index
        try:
            del vecs
        except NameError:
            pass
        try:
            del emb
        except NameError:
            pass

        try:
            db.drop_table(name)
        except Exception:
            pass
        table = db.create_table(name, data=rows)
        try:
            table.create_index(metric="cosine", index_type="IVF_HNSW_FLAT",
                               num_partitions=64, m=16, ef_construction=200,
                               replace=True)
        except Exception as e:
            print(f"  [WARN] index create failed (flat fallback): {e}")
        print(f"  [OK] LanceDB table [{name}] rows={table.count_rows()} | {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()