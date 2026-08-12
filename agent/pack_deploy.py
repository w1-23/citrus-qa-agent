# -*- coding: utf-8 -*-
"""Citrus QA Agent 部署打包脚本 — 在 agent/ 目录运行:
    python pack_deploy.py
产出: 部署包.zip（仅代码+配置+验证脚本，不含 .env/state/logs/workspace/data）
另产出: 模型缓存包可选（--with-cache 附加 .hf_cache 与 fastembed 缓存说明）
"""
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "citrus_agent_deploy.zip"

EXCLUDE_DIRS = {
    "state", "logs", "workspace", "data", "__pycache__",
    ".git", ".hf_cache", ".pytest_cache",
}
EXCLUDE_FILES = {
    ".env", ".env.example", "test_batch1.py", "test_batch2.py",
    "pack_deploy.py", "AGENT_CHANGES.md",
    "TUNE_PARAMS.md", "readme2.md",
}
EXCLUDE_SUFFIX = {".pyc", ".tmp"}


def should_skip(rel: Path) -> bool:
    parts = set(rel.parts)
    if parts & EXCLUDE_DIRS:
        return True
    if rel.name in EXCLUDE_FILES or rel.name.startswith("._"):
        return True
    if rel.suffix in EXCLUDE_SUFFIX:
        return True
    return False


def main():
    if OUT.exists():
        OUT.unlink()
    n = 0
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as zf:
        # v8.3.3: 依赖清单 requirements.txt 在仓库根目录，必须随包发布
        req = ROOT.parent / "requirements.txt"
        if req.exists():
            zf.write(req, "agent/requirements.txt")
            n += 1
        for root, dirs, files in os.walk(ROOT):
            root_path = Path(root)
            # 剪枝：不进入被排除的目录
            dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
            for fn in files:
                fpath = root_path / fn
                rel = fpath.relative_to(ROOT)
                if should_skip(rel):
                    continue
                zf.write(fpath, f"agent/{rel.as_posix()}")
                n += 1
    size_mb = OUT.stat().st_size / (1024 * 1024)
    print(f"OK: {OUT.name} ({n} files, {size_mb:.1f} MB)")
    print("部署时还需单独拷贝: data/ (语料库), 并配置 agent/.env")
    print("验收: 解压后 cd agent && pip install -r requirements.txt && python tests/verify_retrieval.py")


if __name__ == "__main__":
    sys.exit(main())
