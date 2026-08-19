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
        # 优先用文件名标签（含 _r0.3/_w1.3 等网格后缀），兼容无标签旧文件
        stem = p.stem.replace("eval_qa_report_", "").replace("_all", "")
        cfg = stem if stem and stem != "report" else d.get("ablation", "default")
        reports[cfg] = d.get("rows", [])
    return reports


def _aggregate(rows):
    """返回 (recall_avg, mrr_avg, n_gold_rows, nonempty_ratio)。"""
    gold = [r for r in rows if r.get("recall@10") is not None]
    rec = sum(r["recall@10"] for r in gold) / len(gold) if gold else None
    mrr = sum(r["mrr@10"] for r in gold) / len(gold) if gold else None
    nonempty = sum(1 for r in rows if r.get("top_hit") not in (None, "-")) / max(len(rows), 1)
    return rec, mrr, len(gold), nonempty


def plot_curves(reports: dict, outdir: Path):
    """fig2 阈值比率曲线；fig3 BM25 权重曲线；fig4 重排预算；fig5 RRF k。
    读 *_r* / *_w* / *_tk* / *_kk* 网格报告。"""
    if plt is None:
        return
    _setup_font()
    re_mod = __import__("re")
    series = [
        (("default_r([\d.]+)", "fig2_threshold_ratio.png", "动态阈值比率 ratio",
          "阈值比率 vs Recall@10 / MRR"), 0.6),
        (("default_w([\d.]+)", "fig3_bm25_weight.png", "BM25 RRF 权重",
          "BM25 权重 vs Recall@10 / MRR"), 1.0),
        (("default_tk(\d+)", "fig4_topk_final.png", "重排预算 TOP_K_FINAL",
          "重排预算 vs Recall@10 / MRR"), 10.0),
        (("default_kk(\d+)", "fig5_rrf_k.png", "RRF k", "RRF k vs Recall@10 / MRR"), 60.0),
    ]
    for (pat, fname, xlab, title), anchor in series:
        pts = []
        for cfg, rows in reports.items():
            m = re_mod.match(pat, cfg)
            if m:
                pts.append((float(m.group(1)), _aggregate(rows)))
        if "default" in reports and anchor not in [p[0] for p in pts]:
            pts.append((anchor, _aggregate(reports["default"])))
        if not pts:
            continue
        pts.sort()
        xs = [p[0] for p in pts]
        rec = [p[1][0] for p in pts]
        mrr = [p[1][1] for p in pts]
        fig, ax = plt.subplots(figsize=(8.5, 4.8))
        ax.plot(xs, rec, "-o", color="#1F6F5C", label="Recall@10")
        ax.plot(xs, mrr, "-s", color="#C62828", label="MRR@10")
        ax.set_xlabel(xlab); ax.set_ylabel("得分")
        ax.set_ylim(0, 1.05); ax.grid(linestyle=":", alpha=0.4)
        ax.set_title(title); ax.legend(loc="lower left", fontsize=9)
        fig.tight_layout(); fig.savefig(outdir / fname, dpi=200)
        print("[plot] saved", outdir / fname)


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
    plot_curves(reports, Path(args.outdir))


if __name__ == "__main__":
    main()