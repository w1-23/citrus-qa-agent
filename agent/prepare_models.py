# -*- coding: utf-8 -*-
"""v8.5.0 模型预下载/预导出——run.ps1 首次启动自动调用。

解决"向量编码器下载慢"的痛点：首次运行把模型下载/导出到项目内缓存
（agent/.hf_cache，已在 .gitignore，不入仓库），此后启动秒级就绪：
  1. 向量编码模型  intfloat/multilingual-e5-large（fastembed）
  2. 重排模型      BAAI/bge-reranker-v2-m3（ONNX 导出，~1-2GB，可选跳过）
  3. Skill 策略卡片向量索引（依赖 1，用于输出格式/策略卡片匹配）

用法:  python prepare_models.py [--skip-reranker]
"""
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent          # agent/
CACHE = ROOT / ".hf_cache"
CACHE.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("FASTEMBED_CACHE_PATH", str(CACHE / "fastembed"))
os.environ.setdefault("HF_HOME", str(CACHE / "hf"))
os.environ.setdefault("TRANSFORMERS_CACHE", str(CACHE / "hf"))

SKIP_RERANKER = "--skip-reranker" in sys.argv


def step(n, msg):
    print(f"\n[{n}] {msg}", flush=True)


def prepare_embedding() -> None:
    step("1/3", "下载向量编码模型 intfloat/multilingual-e5-large（首次约 2-5 分钟）...")
    t0 = time.time()
    from fastembed import TextEmbedding
    TextEmbedding(model_name="intfloat/multilingual-e5-large")
    print(f"    ✓ 向量编码模型就绪（{time.time() - t0:.0f}s）", flush=True)


def prepare_reranker() -> None:
    cache_dir = CACHE / "onnx_reranker"
    if (cache_dir / "model.onnx").exists():
        print("    ✓ 重排模型已存在，跳过", flush=True)
        return
    step("2/3", "导出重排模型 BAAI/bge-reranker-v2-m3 为 ONNX（首次约 5-15 分钟，"
                "模型较大约 2GB；如网络慢可 Ctrl+C 后用 --skip-reranker 跳过，"
                "检索将使用基础阈值降级）...")
    t0 = time.time()
    from transformers import AutoTokenizer
    from optimum.onnxruntime import ORTModelForSequenceClassification
    model = ORTModelForSequenceClassification.from_pretrained(
        "BAAI/bge-reranker-v2-m3", export=True)
    model.save_pretrained(cache_dir)
    tok = AutoTokenizer.from_pretrained("BAAI/bge-reranker-v2-m3")
    tok.save_pretrained(cache_dir)
    print(f"    ✓ 重排模型导出完成（{time.time() - t0:.0f}s）", flush=True)


def prepare_skill_index() -> None:
    step("3/3", "构建 Skill 策略卡片向量索引（输出格式/策略匹配用）...")
    sys.path.insert(0, str(ROOT))
    t0 = time.time()
    try:
        from src.core.skill_tree import SkillTree
        SkillTree().search_strategy_cards(query="综述", card_type="output", top_k=1)
        print(f"    ✓ Skill 索引就绪（{time.time() - t0:.0f}s）", flush=True)
    except Exception as e:
        print(f"    ⚠ Skill 索引构建跳过（{e}）——不影响核心功能", flush=True)


if __name__ == "__main__":
    print("Citrus QA Agent 模型准备（v8.5.0）", flush=True)
    prepare_embedding()
    if not SKIP_RERANKER:
        prepare_reranker()
    prepare_skill_index()
    print("\n全部就绪 ✓ 现在可以启动服务了。", flush=True)
