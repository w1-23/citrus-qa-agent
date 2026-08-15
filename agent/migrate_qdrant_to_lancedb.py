"""Qdrant local → LanceDB 全量迁移（v8.9）。

从 data/<batch>/qdrant_data scroll 全部向量（含 payload 的 paper_id/chunk_index），
写入 data/lancedb/<batch>.lance 表。检索定位仍走 chunks.jsonl + _idx_map
（与 AG-11 同款），因此表只存 vector + paper_id + chunk_index 即可。

用法（rag-agent 环境）:
    python migrate_qdrant_to_lancedb.py [--batch 1-50]   # 默认全量
幂等：表存在则 overwrite 重建。
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from qdrant_client import QdrantClient
import lancedb

DATA_DIR = Path(__file__).resolve().parent / "data"
LANCE_ROOT = DATA_DIR / "lancedb"


def migrate_batch(batch_dir: Path) -> tuple[str, int, float]:
    name = batch_dir.name
    qdrant_path = batch_dir / "qdrant_data"
    t0 = time.time()
    qd = QdrantClient(path=str(qdrant_path), timeout=60)
    colls = qd.get_collections().collections
    if not colls:
        qd.close()
        return name, 0, 0.0
    coll = colls[0].name

    pts = []
    offset = None
    while True:
        batch, offset = qd.scroll(collection_name=coll, limit=1000,
                                  with_vectors=True, with_payload=True, offset=offset)
        pts.extend(batch)
        if offset is None or not batch:
            break
    qd.close()
    if not pts:
        return name, 0, time.time() - t0

    rows = []
    for i, p in enumerate(pts):
        pl = p.payload or {}
        rows.append({
            "vector": np.asarray(p.vector, dtype=np.float32),
            "paper_id": str(pl.get("paper_id", "")),
            "chunk_index": int(pl.get("chunk_index", i)),
        })

    db = lancedb.connect(str(LANCE_ROOT))
    table = db.create_table(name, data=rows, mode="overwrite")
    # v8.9: 与 Qdrant 同口径 cosine 度量——LanceDB 度量在建索引时确定，
    # 无索引 flat 扫描默认 L2（与旧后端候选集合不一致）；同时 HNSW 索引
    # 支撑百万级查询。
    try:
        table.create_index(
            metric="cosine",
            index_type="IVF_HNSW_FLAT",
            num_partitions=64,
            m=16,
            ef_construction=200,
            replace=True,
        )
        print(f"[MIGRATE] {name}: cosine IVF_HNSW_FLAT index created", flush=True)
    except Exception as e:
        print(f"[MIGRATE] {name}: index failed ({e}) — flat scan fallback", flush=True)
    return name, table.count_rows(), time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="", help="只迁移指定批次名；默认全量")
    args = ap.parse_args()

    LANCE_ROOT.mkdir(parents=True, exist_ok=True)
    total_rows = 0
    t_start = time.time()
    for batch_dir in sorted(DATA_DIR.iterdir()):
        if not batch_dir.is_dir():
            continue
        if args.batch and batch_dir.name != args.batch:
            continue
        if not (batch_dir / "qdrant_data").exists():
            continue
        name, n, dt = migrate_batch(batch_dir)
        total_rows += n
        print(f"[MIGRATE] {name}: {n} rows in {dt:.1f}s", flush=True)
    print(f"[DONE] total={total_rows} rows in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
