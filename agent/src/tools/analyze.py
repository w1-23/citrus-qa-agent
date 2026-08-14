"""Analysis tools — statistics and experiment design"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
from langchain_core.tools import tool

from src.config import settings, PROJECT_ROOT

logger = logging.getLogger(__name__)

_WORKSPACE_ROOT = PROJECT_ROOT / settings.WORKSPACE_DIR


# ═══════════════════════════════════════════════════════════
# STATISTICAL ANALYSIS
# ═══════════════════════════════════════════════════════════

@tool
def statistical_analysis(file_path: str, method: str, data_json: str, alpha: Optional[float] = None) -> str:
    """执行统计分析，返回统计量和解释。
    底层自动读取 CSV 文件进行计算，全量数据不经过 LLM 上下文。

    Args:
        file_path: CSV/Excel 文件路径（相对于 workspace/ 或绝对路径）
        method: 分析方法。支持: descriptive, ttest, ttest_ind, ttest_paired, anova_oneway,
                chi_square, correlation_pearson, correlation_spearman, linear_regression
        data_json: 列规范 JSON。如 {"value_column": "height", "group_column": "treatment"}
        alpha: 显著性水平（默认使用配置值）

    Returns:
        统计结果和解释文本
    """
    alpha_val = alpha if alpha is not None else settings.STATS_ALPHA
    logger.info(f"[statistical_analysis] file={file_path} method={method} alpha={alpha_val}")

    try:
        import pandas as pd
    except ImportError:
        return "需要安装 pandas: pip install pandas"

    p = Path(file_path)
    abs_path = p if p.is_absolute() else _WORKSPACE_ROOT / p
    # v8.3.3/v8.4.14 安全: 与 read_local_file 对称，绝对路径仅限项目根目录内（或配置的额外根）
    # v8.4.14: 修复 commonpath 同盘前缀漏洞——E:\anywhere 与 E:\agent 同盘前缀
    # 会被放行；改用 is_relative_to 严格判定
    try:
        abs_path = abs_path.resolve()
        if p.is_absolute():
            proj = PROJECT_ROOT.resolve()
            in_project = abs_path.is_relative_to(proj)
            in_extra = any(
                abs_path.is_relative_to(Path(r).resolve())
                for r in getattr(settings, "FILE_READ_EXTRA_ROOTS", None) or [])
            if not in_project and not in_extra:
                return f"[ERR_PERMISSION] 拒绝读取项目目录外的文件: {file_path}"
    except ValueError:
        return f"[ERR_PERMISSION] 拒绝读取项目目录外的文件: {file_path}"
    if not abs_path.exists():
        return f"[ERR_FILE_NOT_FOUND] 文件不存在: {abs_path}"

    # AG-10: 文件大小上限（接线 FILE_READ_MAX_SIZE_MB）
    size_mb = abs_path.stat().st_size / (1024 * 1024)
    if size_mb > settings.FILE_READ_MAX_SIZE_MB:
        return (f"[ERR_FILE_TOO_LARGE] 文件过大: {size_mb:.1f}MB "
                f"> {settings.FILE_READ_MAX_SIZE_MB}MB 限制")

    # AG-10: 样本量上限（接线 STATS_MAX_SAMPLE_SIZE）
    nrows_cap = settings.STATS_MAX_SAMPLE_SIZE
    try:
        df = pd.read_csv(abs_path, nrows=(nrows_cap + 1) if nrows_cap else None)
    except Exception as e:
        return f"文件读取失败: {e}"

    note = ""
    if nrows_cap and len(df) > nrows_cap:
        df = df.head(nrows_cap)
        note = f"[数据截断] 文件超过样本量上限({nrows_cap}行)，仅使用前 {nrows_cap} 行。\n"

    try:
        col_spec = json.loads(data_json)
    except json.JSONDecodeError as e:
        return f"data_json JSON 解析错误: {e}"

    try:
        result = _run_analysis(df, method, col_spec, alpha_val)
    except Exception as e:
        logger.error(f"[statistical_analysis] 计算失败: {e}")
        return f"统计分析失败: {e}"

    return note + result


def _run_analysis(df, method: str, col_spec: dict, alpha_val: float) -> str:
    """核心统计计算（无 IO），由 statistical_analysis 调用 (AG-10 重构)。"""
    if method == "descriptive":
        col = col_spec.get("value_column", "")
        if col not in df.columns:
            return f"列 '{col}' 不存在，可用列: {list(df.columns)}"
        vals = df[col].dropna().tolist()
        if not vals:
            return "错误：数据列为空"
        arr = np.array(vals)
        n = len(vals)
        mean = float(np.mean(arr))
        median = float(np.median(arr))
        std = float(np.std(arr, ddof=1))
        var = float(np.var(arr, ddof=1))
        _min = float(np.min(arr))
        _max = float(np.max(arr))
        q25 = float(np.percentile(arr, 25))
        q75 = float(np.percentile(arr, 75))
        return (f"描述性统计 (n={n}):\n  均值={mean:.4f}  中位数={median:.4f}\n"
                f"  标准差={std:.4f}  方差={var:.4f}\n  最小值={_min:.4f}  最大值={_max:.4f}\n  Q1={q25:.4f}  Q3={q75:.4f}")

    elif method in ("ttest", "ttest_ind"):
        from scipy import stats as sp_stats
        val_col = col_spec.get("value_column", "")
        grp_col = col_spec.get("group_column", "")
        for c in (val_col, grp_col):
            if c not in df.columns:
                return f"列 '{c}' 不存在，可用列: {list(df.columns)}"
        groups = df.groupby(grp_col)[val_col].apply(list)
        if len(groups) < 2:
            return "错误：分组列需要至少 2 个不同值"
        g1 = groups.iloc[0]
        g2 = groups.iloc[1]
        t_stat, p_val = sp_stats.ttest_ind(g1, g2)
        sig = "显著" if p_val < alpha_val else "不显著"
        return (f"独立样本 t检验:\n  t={t_stat:.4f}  p={p_val:.6f}\n"
                f"  结果: {sig} (alpha={alpha_val})\n"
                f"  组1 ({groups.index[0]}): n={len(g1)} 均值={np.mean(g1):.2f}\n"
                f"  组2 ({groups.index[1]}): n={len(g2)} 均值={np.mean(g2):.2f}")

    elif method == "ttest_paired":
        from scipy import stats as sp_stats
        before_col = col_spec.get("before_column", "")
        after_col = col_spec.get("after_column", "")
        for c in (before_col, after_col):
            if c not in df.columns:
                return f"列 '{c}' 不存在，可用列: {list(df.columns)}"
        before = df[before_col].dropna().tolist()
        after = df[after_col].dropna().tolist()
        if not before or not after:
            return "错误：before 或 after 列为空"
        min_len = min(len(before), len(after))
        before, after = before[:min_len], after[:min_len]
        t_stat, p_val = sp_stats.ttest_rel(before, after)
        sig = "显著" if p_val < alpha_val else "不显著"
        diff_mean = sum(a - b for a, b in zip(after, before)) / len(before)
        return (f"配对 t检验:\n  t={t_stat:.4f}  p={p_val:.6f}\n"
                f"  结果: {sig} (alpha={alpha_val})\n  平均差异={diff_mean:.4f}\n  n={min_len}")

    elif method == "anova_oneway":
        from scipy import stats as sp_stats
        val_col = col_spec.get("value_column", "")
        grp_col = col_spec.get("group_column", "")
        for c in (val_col, grp_col):
            if c not in df.columns:
                return f"列 '{c}' 不存在，可用列: {list(df.columns)}"
        groups = df.groupby(grp_col)[val_col].apply(list)
        if len(groups) < 2:
            return "错误：分组列需要至少 2 个不同值"
        f_stat, p_val = sp_stats.f_oneway(*groups.tolist())
        sig = "显著" if p_val < alpha_val else "不显著"
        return f"单因素方差分析 (ANOVA):\n  F={f_stat:.4f}  p={p_val:.6f}\n  结果: {sig} (alpha={alpha_val})\n  组数: {len(groups)}"

    elif method == "chi_square":
        from scipy import stats as sp_stats
        row_col = col_spec.get("row_column", "")
        col_col = col_spec.get("col_column", "")
        for c in (row_col, col_col):
            if c not in df.columns:
                return f"列 '{c}' 不存在，可用列: {list(df.columns)}"
        crosstab = pd.crosstab(df[row_col], df[col_col])
        observed = crosstab.values
        chi2, p_val, dof, expected = sp_stats.chi2_contingency(observed)
        sig = "显著" if p_val < alpha_val else "不显著"
        return f"卡方检验:\n  chi2={chi2:.4f}  p={p_val:.6f}  dof={dof}\n  结果: {sig} (alpha={alpha_val})"

    elif method in ("correlation_pearson", "correlation_spearman"):
        from scipy import stats as sp_stats
        x_col = col_spec.get("x_column", "")
        y_col = col_spec.get("y_column", "")
        for c in (x_col, y_col):
            if c not in df.columns:
                return f"列 '{c}' 不存在，可用列: {list(df.columns)}"
        x = df[x_col].dropna().tolist()
        y = df[y_col].dropna().tolist()
        min_len = min(len(x), len(y))
        x, y = x[:min_len], y[:min_len]
        if len(x) < 3:
            return "错误：有效数据点少于 3"
        method_name = "Pearson" if method == "correlation_pearson" else "Spearman"
        fn = sp_stats.pearsonr if method == "correlation_pearson" else sp_stats.spearmanr
        r, p_val = fn(x, y)
        sig = "显著" if p_val < alpha_val else "不显著"
        return f"{method_name} 相关分析:\n  r={r:.4f}  p={p_val:.6f}\n  结果: {sig} (alpha={alpha_val})\n  n={len(x)}"

    elif method == "linear_regression":
        from scipy import stats as sp_stats
        x_col = col_spec.get("x_column", "")
        y_col = col_spec.get("y_column", "")
        for c in (x_col, y_col):
            if c not in df.columns:
                return f"列 '{c}' 不存在，可用列: {list(df.columns)}"
        x = df[x_col].dropna().tolist()
        y = df[y_col].dropna().tolist()
        min_len = min(len(x), len(y))
        x_arr = np.array(x[:min_len])
        y_arr = np.array(y[:min_len])
        if len(x_arr) < 3:
            return "错误：有效数据点少于 3"
        slope, intercept, r_val, p_val, std_err = sp_stats.linregress(x_arr, y_arr)
        r2 = r_val ** 2
        sig = "显著" if p_val < alpha_val else "不显著"
        return (f"线性回归:\n  y = {slope:.4f}x + {intercept:.4f}\n"
                f"  R²={r2:.4f}  p={p_val:.6f}\n  结果: {sig} (alpha={alpha_val})\n  n={len(x_arr)}")

    else:
        return f"不支持的方法: {method}（支持: descriptive, ttest, ttest_ind, ttest_paired, anova_oneway, chi_square, correlation_pearson, correlation_spearman, linear_regression）"


# ═══════════════════════════════════════════════════════════
# EXPERIMENT DESIGN
# ═══════════════════════════════════════════════════════════

def _power_ttest(d: float, alpha: float, power: float) -> int:
    from scipy import stats as sp_stats
    z_alpha = sp_stats.norm.ppf(1 - alpha / 2)
    z_beta = sp_stats.norm.ppf(power)
    return int(2 * ((z_alpha + z_beta) / d) ** 2) + 1


@tool
def experimental_design(design_type: str, num_groups: int = 2, effect_size: float = 0.5,
                        alpha: Optional[float] = None, power: float = 0.8) -> str:
    """实验设计辅助工具，计算样本量并给出设计建议。

    Args:
        design_type: 实验设计类型: completely_randomized, randomized_block, factorial, repeated_measures
        num_groups: 组数（默认 2）
        effect_size: 预期效应量 Cohen's d（默认 0.5 = 中等效应）
        alpha: 显著性水平（默认使用配置值）
        power: 统计功效（默认 0.8）

    Returns:
        样本量估算结果和设计建议
    """
    alpha_val = alpha if alpha is not None else settings.STATS_ALPHA
    logger.info(f"[experimental_design] type={design_type} groups={num_groups} d={effect_size}")

    try:
        if design_type == "completely_randomized":
            n = _power_ttest(effect_size, alpha_val, power)
            return (f"完全随机设计 样本量估算:\n  组数: {num_groups}\n  效应量 Cohen's d: {effect_size}\n"
                    f"  alpha: {alpha_val}  power: {power}\n  → 每组所需样本量: n={max(n, 3)}（共 {max(n, 3) * num_groups} 个样本）")
        elif design_type == "randomized_block":
            n = max(_power_ttest(effect_size, alpha_val, power), 3)
            return (f"随机区组设计 样本量估算:\n  → 每区组 {num_groups} 个样本，共 {n} 个区组\n  → 总样本量: {n * num_groups}")
        elif design_type == "factorial":
            n = max(_power_ttest(effect_size / 2, alpha_val, power), 3)
            total = n * (num_groups or 4)
            return f"析因设计 样本量估算:\n  → 每处理组合: n={n}\n  → 总样本量: {total}"
        elif design_type == "repeated_measures":
            n = max(int(_power_ttest(effect_size, alpha_val, power) * 0.7), 3)
            return f"重复测量设计 样本量估算:\n  → 每组: n={n}\n  → 总样本量: {n * num_groups}"
        else:
            return f"不支持的设计类型: {design_type}"
    except Exception as e:
        logger.error(f"[experimental_design] 失败: {e}")
        return f"样本量计算失败: {e}"# (removed: deep_citation_check — too expensive, 0 LLM calls for verification now)
