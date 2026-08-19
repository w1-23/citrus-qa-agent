# -*- coding: utf-8 -*-
"""v8.14.1 categories-cn 品种库入库准备：幂等归一化 + 批次元数据。

把 data/categories-cn/chunks/chunks.jsonl 中缺失的 paper_id 补全
（按 variety_id/registry_id 派生，与 reindex_lance.derive_paper_id 同源——
派生规则必须一致，qdrant 旧向量才能按 (paper_id, chunk_index) 复用），
并生成 data/categories-cn/metadata.json 批次摘要。

用法（agent/ 目录，rag-agent 环境）:
    E:\\anaconda\\envs\\rag-agent\\python.exe prepare_categories.py

幂等：已含 paper_id 的行不动；重复执行安全。
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from reindex_lance import derive_paper_id  # noqa: E402

BATCH = Path("data") / "categories-cn"


def main():
    chunks = BATCH / "chunks" / "chunks.jsonl"
    if not chunks.exists():
        sys.exit(f"[ERR] {chunks} 不存在")
    lines = chunks.read_text(encoding="utf-8").splitlines()
    out, changed = [], 0
    for ln in lines:
        ln = ln.strip()
        if not ln:
            continue
        c = json.loads(ln)
        if not str(c.get("paper_id") or "").strip():
            c["paper_id"] = derive_paper_id(c)
            changed += 1
        out.append(json.dumps(c, ensure_ascii=False))
    tmp = chunks.with_name(chunks.name + ".tmp")
    tmp.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
    tmp.replace(chunks)
    print(f"[OK] {len(out)} chunks 归一化完成；本次新增 paper_id {changed} 条")

    meta = {
        "pipeline": {"name": "categories-cn", "version": "1.0.0"},
        "summary": {
            "total": len(out),
            "source_type": "UCR citrus variety",
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        },
    }
    (BATCH / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("[OK] metadata.json 已生成 (total=%d)" % len(out))


if __name__ == "__main__":
    main()