# -*- coding: utf-8 -*-
"""eval_qa — 黄金问答集评估 harness（论文 P0：检索指标 / 消融 / LLM-as-Judge 钩子）。

对应 paper/07-第6章-评估.md §3-§6 与 paper/12-创新点与需补实验.md P0/P1。

用法（在 agent/ 目录，venv python）:
  python tests/eval_qa.py --cpu                      # 检索指标（离线，无需模型 key；
                                                     # HyDE 无 key 自动降级——与无 key 部署等价）
  python tests/eval_qa.py --subset L1 --cpu
  python tests/eval_qa.py --ablation hyde_off --cpu  # 消融：HyDE 关闭 vs 默认（离线态等效）
  python tests/eval_qa.py --ablation threshold_soft  # 阈值: ratio 0.60→0.80（更严）
  python tests/eval_qa.py --use-hyde                 # 有模型 key 时走产品等价路径(_cached_hyde)
  python tests/eval_qa.py --mode judge --api-key ... # LLM-as-Judge 三维（事实/引用一致/缺口声明）

指标口径:
  Recall@10 = |gold ∩ top10| / |gold|（按 paper_id 前缀匹配，兼纳 paper_id:chunk_index）
  MRR@10    = 首个 gold 命中排名的倒数均值
  gold 为空的行不参与检索指标（标注 TBD，由人工补 required_evidence）
"""

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent if Path(__file__).resolve().name != "<string>" else None
PROJECT = (Path.cwd() if (Path.cwd() / "src").is_dir()
           else (_SCRIPT_DIR.parent if _SCRIPT_DIR else Path.cwd()))
sys.path.insert(0, str(PROJECT))

_GOLD_FILE = PROJECT / "tests" / "golden_qa.jsonl"
_REPORT = PROJECT / "logs" / "eval_qa_report.json"


def _load_golden(path: Path):
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))
    return rows


# ───────────────────────── 检索指标（离线可跑）─────────────────────────
def _paper_id_of(chunk: dict) -> str:
    return str(chunk.get("paper_id") or "").strip()


def _matches_gold(chunk: dict, gold_ids: list) -> bool:
    pid = _paper_id_of(chunk)
    if not pid:
        return False
    for g in gold_ids:
        if not g:
            continue
        if ":" in g:
            if f"{pid}:{chunk.get('chunk_index')}" == g:
                return True
        elif pid == g or pid.startswith(g.rstrip("*")):
            return True
    return False


def _retrieval_metrics(question: str, gold_ids: list, retriever, top_k: int = 10):
    """跑一次产品路径检索（HyDE 是否启用见开关），返回命中行 + 指标。

    指标按 gold id（论文级）去重计数：同一 paper_id 的多个 chunk 命中只算一次。
    """
    t0 = time.time()
    hits = retriever.search(question)[:top_k]
    ms = (time.time() - t0) * 1000
    if not gold_ids:
        return hits, {"recall@10": None, "mrr@10": None, "match_rank": None, "ms": round(ms, 1)}
    first_rank = None
    matched = set()
    for i, c in enumerate(hits):
        for g in gold_ids:
            if not g or g in matched:
                continue
            if _matches_gold(c, [g]):
                matched.add(g)
                if first_rank is None:
                    first_rank = i + 1
                break
    recall = len(matched) / len(gold_ids)
    mrr = 1.0 / first_rank if first_rank else 0.0
    return hits, {"recall@10": round(recall, 4), "mrr@10": round(mrr, 4),
                  "match_rank": first_rank, "ms": round(ms, 1)}


# ───────────────────────── 消融支持（打补丁 settings）─────────────────────────
_ABLATIONS = {
    # key: (说明, [ (attr, value), ... ])
    "default": ("基线（config.yaml 原始值）", []),
    "hyde_off": ("HyDE 关闭（RAG_HYDE_ENABLED=False；离线态与默认本就等价，有 key 时才有差异）",
                 [("RAG_HYDE_ENABLED", False)]),
    "threshold_soft": ("更严阈值（ratio 0.60→0.80）", [("DYNAMIC_THRESHOLD_RATIO", 0.80)]),
    "threshold_loose": ("更松阈值（ratio 0.60→0.40）", [("DYNAMIC_THRESHOLD_RATIO", 0.40)]),
    "rrf_bm25_up": ("BM25 权重 1.0→1.2", [("RRF_WEIGHT_BM25", 1.2)]),
    "rrf_bm25_down": ("BM25 权重 1.0→0.5", [("RRF_WEIGHT_BM25", 0.5)]),
}


def _apply_ablation(name: str):
    from src.config import settings
    saved = {}
    for attr, val in _ABLATIONS[name][1]:
        saved[attr] = getattr(settings, attr)
        setattr(settings, attr, val)
    return saved


def _restore(saved: dict):
    from src.config import settings
    for attr, val in saved.items():
        setattr(settings, attr, val)


# ───────────────────────── LLM-as-Judge 钩子（需模型 key）─────────────────────────
_JUDGE_RUBRIC = {
    "factual": "回答中的每条关键断言是否被给出的证据支持（不支持/部分/支持）",
    "citation": "回答引用的 [id] 是否与证据文本一致（错配率）",
    "gap": "证据不足处是否如实声明缺口（未声明/部分/声明）",
}


def judge_hook(question: str, answer: str, evidence: list, api_key: str = "") -> dict:
    """三维 rubric 打分（LLM-as-Judge 分离模型）。未实现时返回 skip。

    设计: Judge=独立模型、temperature 0、给理由；人工抽检 10% 校准（07 §4）。
    实现要点: 用 src.core.llm_pool 的 fast 模型；逐维 0-1 分数 + 一句话理由。
    """
    if not api_key:
        return {"status": "skipped", "reason": "no api key"}
    # TODO(P0): 接 llm_pool 或独立 judge 通道；先落格式保证 golden 字段完整
    return {"status": "todo", "rubric": _JUDGE_RUBRIC}


# ───────────────────────── 主流程 ─────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="强制 CPU（沙箱 DML 崩溃/无 GPU 环境）")
    ap.add_argument("--subset", default="ALL", help="L1/L2/L3/ALL")
    ap.add_argument("--ablation", default="default", choices=list(_ABLATIONS), help="消融配置")
    ap.add_argument("--use-hyde", action="store_true", help="有模型 key 时走产品等价 HyDE 路径")
    ap.add_argument("--mode", default="retrieval", choices=["retrieval", "judge"])
    ap.add_argument("--api-key", default="")
    args = ap.parse_args()

    if args.cpu:
        import src.engine.hardware as hw
        hw.get_ort_providers = lambda: ["CPUExecutionProvider"]  # noqa

    from src.retrieval.multi_retriever import MultiBatchRetriever

    gold = _load_golden(_GOLD_FILE)
    if args.subset != "ALL":
        gold = [g for g in gold if g.get("level") == args.subset]
    if not gold:
        print(f"[eval] 无匹配题目: subset={args.subset}")
        return

    print(f"[eval] 加载 {len(gold)} 题 | 消融={args.ablation} ({_ABLATIONS[args.ablation][0]}) | mode={args.mode}")
    saved = _apply_ablation(args.ablation)
    try:
        t0 = time.time()
        r = MultiBatchRetriever()
        print(f"[eval] retriever 加载 {time.time()-t0:.1f}s | 批次={len(r.lance_tables)}")

        rows = []
        for g in gold:
            hits, m = _retrieval_metrics(g["question"], g.get("required_evidence") or [], r)
            top = _paper_id_of(hits[0]) if hits else "-"
            rows.append({
                "id": g["id"], "level": g.get("level"), "question": g["question"],
                "gold": len(g.get("required_evidence") or []),
                **m, "top_hit": top[:52],
            })
            print("  {id} [{level}] {q}  gold={gold} recall={recall@10} mrr={mrr@10} ms={ms} top={top_hit}".format(
                **{**g, **m, "id": g["id"], "q": g["question"][:44],
                   "gold": len(g.get("required_evidence") or []),
                   "top_hit": top[:52]}))

        with_gold = [x for x in rows if x["recall@10"] is not None]
        n = len(with_gold)
        if n:
            avg_r = sum(x["recall@10"] for x in with_gold) / n
            avg_m = sum(x["mrr@10"] for x in with_gold) / n
            avg_ms = sum(x["ms"] for x in with_gold) / n
            print(f"\n[eval] 聚合（{n} 题含 gold）: Recall@10={avg_r:.4f}  MRR@10={avg_m:.4f}  p50耗时={avg_ms:.0f}ms")
        else:
            print("\n[eval] 无含 gold 的题目（全是 TBD）——请人工补 required_evidence")

        _REPORT.parent.mkdir(exist_ok=True)
        out = {"ablation": args.ablation, "subset": args.subset, "rows": rows}
        _REPORT.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[eval] 报告已写: {_REPORT}")

        if args.mode == "judge":
            print("[eval] judge 模式为钩子框架（judge_hook 需接 llm_pool，见 TODO）")
    finally:
        _restore(saved)


if __name__ == "__main__":
    main()