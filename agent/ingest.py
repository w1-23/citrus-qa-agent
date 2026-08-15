# -*- coding: utf-8 -*-
"""v8.5.0 语料导入工具——把自己的文献变成可检索的知识库。

用法（在 agent/ 目录）:
    python ingest.py                          # 导入 data/import/ 下的 PDF/txt/md
    python ingest.py --dir 我的文献目录       # 指定输入目录
    python ingest.py --batch mycorpus         # 指定批次名（默认用输入目录名）
    python ingest.py --chunk-size 800         # 分块字符数（默认 800）

输出（与检索器约定完全一致，重启服务即生效）:
    data/<batch>/chunks/chunks.jsonl   # 分块文本（BM25/证据全文用）
    data/<batch>/qdrant_data/          # Qdrant 本地向量库（向量检索用）

说明:
    - 向量模型与检索器同一实例（intfloat/multilingual-e5-large，1024 维）
    - 文本提取: PDF 用 PyMuPDF，txt/md 直接读
    - 分块: 按段落聚合到 chunk_size 字符窗口，保持段落完整
    - data/ 不入 Git 仓库与发布包——你的知识库只属于你
"""
import argparse
import json
import os
import re
import sys
import time
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

CACHE = ROOT / ".hf_cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(CACHE / "fastembed"))

TEXT_EXTS = {".txt", ".md", ".markdown"}
PDF_EXTS = {".pdf"}


def extract_text(path: Path) -> str:
    """PDF → fitz 全文本；txt/md → 直接读。"""
    if path.suffix.lower() in TEXT_EXTS:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() in PDF_EXTS:
        import fitz
        doc = fitz.open(str(path))
        try:
            return "\n".join(page.get_text() for page in doc)
        finally:
            doc.close()
    return ""


def chunk_text(text: str, chunk_size: int) -> list:
    """按段落聚合分块：段落完整、窗口 ≈ chunk_size 字符。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks, cur = [], ""
    for p in paras:
        if len(cur) + len(p) + 2 > chunk_size and cur:
            chunks.append(cur)
            cur = p
        else:
            cur = f"{cur}\n\n{p}" if cur else p
    if cur:
        chunks.append(cur)
    return chunks


def main():
    ap = argparse.ArgumentParser(description="Citrus QA Agent 语料导入")
    ap.add_argument("--dir", default=str(ROOT / "data" / "import"),
                    help="输入目录（含 PDF/txt/md）")
    ap.add_argument("--batch", default="",
                    help="批次名（默认取输入目录名）")
    ap.add_argument("--chunk-size", type=int, default=800)
    ap.add_argument("--backend", default="auto", choices=["auto", "qdrant", "lancedb"],
                    help="向量后端（auto=检测 data/lancedb，默认）")
    args = ap.parse_args()

    # v8.9 后端选择：auto → 已有 lancedb 数据则用 lancedb，否则 qdrant
    backend = args.backend
    if backend == "auto":
        lance_root = ROOT / "data" / "lancedb"
        backend = "lancedb" if (lance_root.exists() and any(lance_root.glob("*.lance"))) \
            else "qdrant"
    print(f"🍊 语料导入 → 批次 [{batch}]（{len(files)} 个文件）| 向量后端: {backend}")

    src_dir = Path(args.dir)
    if not src_dir.exists():
        print(f"✗ 输入目录不存在: {src_dir}")
        print("  请把 PDF/txt/md 放进该目录后重试")
        sys.exit(1)
    files = sorted(p for p in src_dir.iterdir()
                   if p.suffix.lower() in TEXT_EXTS | PDF_EXTS)
    if not files:
        print(f"✗ {src_dir} 下没有 PDF/txt/md 文件")
        sys.exit(1)

    batch = args.batch or src_dir.name
    batch_dir = ROOT / "data" / batch
    chunks_path = batch_dir / "chunks" / "chunks.jsonl"
    qdrant_path = batch_dir / "qdrant_data"

    print(f"🍊 语料导入 → 批次 [{batch}]（{len(files)} 个文件）")
    print("  1/3 提取文本并分块 ...")
    all_chunks = []
    for f in files:
        text = extract_text(f)
        if not text.strip():
            print(f"    ⚠ 空文本跳过: {f.name}")
            continue
        paper_id = re.sub(r"[^\w\u4e00-\u9fff-]+", "_", f.stem) or f"doc_{uuid.uuid4().hex[:6]}"
        parts = chunk_text(text, args.chunk_size)
        for i, part in enumerate(parts):
            all_chunks.append({
                "paper_id": paper_id,
                "chunk_index": i,
                "text": part,
                "section_name": "",
                "title": f.name,
                "year": "",
                "doi": "",
            })
        print(f"    ✓ {f.name} → {len(parts)} 块")
    if not all_chunks:
        print("✗ 没有提取到任何文本")
        sys.exit(1)

    print(f"  2/3 向量化 {len(all_chunks)} 块（首次会自动下载向量模型，约 2-5 分钟）...")
    t0 = time.time()
    from src.engine.embedder import Embedder
    emb = Embedder()
    vecs = emb.embed_docs([c["text"] for c in all_chunks])
    print(f"    向量化完成（{time.time() - t0:.0f}s, 维度 {emb.dim}）")

    print("  3/3 写入向量库 + chunks.jsonl ...")
    batch_dir.mkdir(parents=True, exist_ok=True)
    (batch_dir / "chunks").mkdir(parents=True, exist_ok=True)

    if backend == "lancedb":
        import numpy as np
        import lancedb
        lance_root = ROOT / "data" / "lancedb"
        lance_root.mkdir(parents=True, exist_ok=True)
        db = lancedb.connect(str(lance_root))
        rows = [
            {
                "vector": np.asarray(vecs[i], dtype=np.float32),
                "paper_id": c["paper_id"],
                "chunk_index": c["chunk_index"],
            }
            for i, c in enumerate(all_chunks)
        ]
        # 追加式写入（v8.9 热更新：同一批次重复导入会追加，检索即查即得）
        table = None
        try:
            table = db.open_table(batch)
        except Exception:
            pass
        if table is None:
            table = db.create_table(batch, data=rows)
            try:
                table.create_index(metric="cosine", index_type="IVF_HNSW_FLAT",
                                   num_partitions=64, m=16, ef_construction=200,
                                   replace=True)
            except Exception as e:
                print(f"    ⚠ 索引创建失败（flat 扫描兜底）: {e}")
        else:
            table.add(rows)
        print(f"    ✓ LanceDB 表 [{batch}] 现有 {table.count_rows()} 行")
    else:
        from qdrant_client import QdrantClient
        from qdrant_client.models import Distance, VectorParams, PointStruct
        qdrant_path = batch_dir / "qdrant_data"
        client = QdrantClient(path=str(qdrant_path))
        coll = "citrus_literature"
        if coll not in {c.name for c in client.get_collections().collections}:
            client.create_collection(
                collection_name=coll,
                vectors_config=VectorParams(size=emb.dim, distance=Distance.COSINE),
            )
        points = [
            PointStruct(
                id=i + 1,
                vector=vecs[i],
                payload={
                    "paper_id": c["paper_id"],
                    "chunk_index": c["chunk_index"],
                    "title": c["title"],
                },
            )
            for i, c in enumerate(all_chunks)
        ]
        client.upsert(collection_name=coll, points=points)
        client.close()
    with open(chunks_path, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    (batch_dir / "metadata.json").write_text(
        json.dumps({"batch": batch, "files": len(files), "chunks": len(all_chunks),
                    "backend": backend},
                   ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\n✅ 完成: 批次 [{batch}] 共 {len(all_chunks)} 块（后端: {backend}）")
    print(f"   chunks:  {chunks_path}")
    print(f"   重启服务（或新会话）后检索自动加载该批次。")


if __name__ == "__main__":
    main()
