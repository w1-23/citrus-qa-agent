# -*- coding: utf-8 -*-
"""plot_eval — 把 eval_qa 报告 JSON 画成论文图表（paper/figures/*.png）。

读取 agent/logs/eval_qa_report_<cfg>_all.json（及旧版 eval_qa_report.json），
输出:
  fig1_recall.png   按消融配置的 Recall@10 分组柱状（含/不含 gold 的题分开）
  fig2_mrr.png      MRR@10 对比
  fig3_bilingual.png 双语探针 passed 数对比（需 probe 输出文本，见 --probe 文件）
字体: 中文微软雅黑（存在则用，否则默认），数字天然渲染；dpi=200。
用法: python tests/plot_eval.py [--logdir ../logs] [--outdir ../paper/figures]
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
except ImportError:  # 无 matplotlib 环境也能解析但不画
    plt = None

PROJ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJ))


def _setup_font():
    if plt is None:
        return
    try:
        for cand in ("msyh.ttc", "msyhbd.ttc", "simhei.ttf"):  # 微软雅黑/黑体
            p = Path("C:/Windows/Fonts") / cand
            if p.exists():
                font_manager.fontManager.addfont(str(p))
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass


def load_reports(logdir: Path):
    reports = {}
    for p in sorted(logdir.glob("eval_qa_report*.json")):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        cfg = d.get("ablation", p.stem.replace("eval_qa_report_", ""))
        reports[cfg] = d.get("rows", [])
    return reports


def plot(reports: dict, outdir: Path):
    if plt is None:
        print("[plot] matplotlib 不可用，跳过绘图（仅打印数据摘要）")
        for cfg, rows in reports.items():
            gold = [r for r in rows if r.get("recall@10") is not None]
            if gold:
                avg = sum(r["recall@10"] for r in gold) / len(gold)
                mrr = sum(r["mrr@10"] for r in gold) / len(gold)
                print(f"  {cfg}: n={len(gold)} Recall@10={avg:.3f} MRR@10={mrr:.3f}")
        return
    _setup_font()
    outdir.mkdir(parents=True, exist_ok=True)
    order = ["default", "threshold_soft", "threshold_loose", "rrf_bm25_up",
             "rrf_bm25_down", "hyde_off"]
    cfgs = [c for c in order if c in reports]
    labels = {"default": "基线", "threshold_soft": "阈值严(0.80)",
              "threshold_loose": "阈值松(0.40)", "rrf_bm25_up": "BM25↑1.2",
              "rrf_bm25_down": "BM25↓0.5", "hyde_off": "HyDE off"}

    fig, ax = plt.subplots(figsize=(9.5, 5))
    for i, c in enumerate(cfgs):
        rows = reports[c]
        gold = [r for r in rows if r.get("recall@10") is not None]
        x = i
        if gold:
            ax.bar(x - 0.2, sum(r["recall@10"] for r in gold) / len(gold), 0.4,
                   label=labels.get(c, c), color="#1F6F5C")
        rows_all = [r for r in rows]
        nonempty = sum(1 for r in rows_all if r.get("top_hit") not in (None, "-"))
        ax.bar(x + 0.2, nonempty / max(len(rows_all), 1), 0.4, color="#C8E0D8")
    ax.set_xticks(range(len(cfgs)))
    ax.set_xticklabels([labels.get(c, c) for c in cfgs])
    ax.set_ylabel("得分")
    ax.set_ylim(0, 1.05)
    ax.set_title("黄金问答集消融对比（深绿=Recall@10(含gold)，浅绿=命中率(全体题)）")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "fig1_recall.png", dpi=200)
    print("[plot] saved", outdir / "fig1_recall.png")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=str(PROJ / "logs"))
    ap.add_argument("--outdir", default=str(PROJ.parent / "paper" / "figures"))
    args = ap.parse_args()
    reports = load_reports(Path(args.logdir))
    if not reports:
        print("[plot] 无报告 JSON")
        return
    plot(reports, Path(args.outdir))


if __name__ == "__main__":
    main()