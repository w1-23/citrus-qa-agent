# 📊 最终论文图表绘制规范与执行指令

> 目标期刊：Computers and Electronics in Agriculture (Elsevier, SCI Q1)
> 状态：v1 定稿（2026-08-25，A 类数据已就绪，B 类数据待补）

---

## 1. 通用硬性规格（Baseline Requirements）

所有提交至 CEA 的图片必须满足以下底层规范：

- **字体（Font）**：全图（含轴标签、刻度、图例、显著性字母）必须使用 **Times New Roman**。
- **字号（Font Size）**：轴标题 10-11 pt，刻度标签 8-9 pt，图例 8-9 pt，显著性标记 10 pt。
- **格式（Format）**：导出为 **矢量PDF**（优先，用于出版）和 **300 DPI PNG**（用于预览）。严禁使用 JPG。
- **配色（Color）**：避免红绿对比（色盲友好）。建议使用 `viridis`、`cividis` 或 `Set2` 调色板。灰色（#666666）用于基线。
- **统计与误差棒**：
  - 重复实验（如不同种子/不同题组）必须展示 **均值 ± 标准差（SD）** 或 **标准误（SEM）**。
  - 柱状图顶部必须标注 **显著性检验结果**（如 `* p < 0.05`, `** p < 0.01`, `n.s.`），基于配对 T 检验或 Wilcoxon 检验。
- **原始数据存储（铁律）**：绘图脚本 **不得** 直接硬编码数字。必须从 `.csv` 或 `.json` 读取原始数据。所有原始数据备份至 `experiment/results/fig_data/` 目录。

---

## 2. 各实验具体绘图方案（及对应脚本指令）

### 图组 A：语言偏置与检索可达性（S3 + 实验4a 联合证据）
- **图型**：**分组箱线图（Grouped Boxplot）** + **散点抖动（Swarm）**。
- **数据来源**：`s3_reachability.json`（84题的zh/en rank分布）+ `exp4a_scan.json`（窗口扫描下的命中情况）。
- **X轴**：查询语言（English / Chinese）。
- **Y轴**：Gold Chunk 的检索排名（Rank）。
- **关键细节**：
  - 箱线图展示中位数和四分位距。
  - 叠加抖动散点（每个点代表一道题），展示数据分布。
  - 在顶部标注 `p-value`（Wilcoxon signed-rank test），证明中文排名显著低于英文。
- **针对4a的补充图**：**折线图**，展示不同窗口（W=20/30/40/60）下，中文盲点（Miss Rate）的变化，证明窗口增大对中文有恢复作用。

### 图组 B：A类调优 – 参数网格扫描（实验4a/7）
- **图型**：**热力图（Heatmap）** 和 **折线图（Line Plot）**。
- **4a 候选窗口**：
  - **热力图**：X轴为 `candidate_window`，Y轴为 `top_k_final`，填充颜色为 MRR 值。**效果**：一眼看出 K 对性能无影响，W=20 最优。
- **7 动态阈值**：
  - **双轴折线图**：X轴为 β 值（0.3~0.8）。
  - 左Y轴：Gold通过率（Pass Rate）。
  - 右Y轴：平均证据密度（平均通过文献数）。
  - **效果**：显示 β=0.6 是 100% 召回率的"悬崖点"，之后密度下降。

### 图组 C：A类调优 – 早停成本（实验4b）
- **图型**：**分组柱状图（Grouped Bar）**。
- **数据**：不同 `(N_min, α)` 组合下的平均轮次。
- **细节**：
  - 默认组用高亮颜色（橙色），其他组用灰色。
  - Y轴为"轮次节省百分比（%）"。
  - 在柱子上方标注该组对应的 `Gold Coverage`（应全部为 0.30，证明零损失）。
- **结论视觉**：清晰展示默认参数处于平台区。

### 图组 D：B类消融 – 多粒度查询（实验2）
- **图型**：**横向条形图（Horizontal Bar）** + **成本标注**。
- **数据**：8种 `query_mode` 的 MRR 和 Recall@10。
- **细节**：
  - X轴为 Recall@10。
  - Y轴为查询模式（按性能从高到低排序：Full > HyDE+MQ > HyDE+SUM > ... > Raw）。
  - 在每个条形右侧，用文本标注该模式的"路数（Paths）"和"LLM调用次数"。
- **显著性**：在 Full 和 HyDE-Only 之间画一条括弧线，标注 `**`，证明多粒度组合显著优于单一长度。

### 图组 E：B类消融 – 六级管道消融（实验1）
- **图型**：**蝴蝶图（Diverging Bar）** 或 **瀑布图**。
- **数据**：去掉某一组件（-HyDE, -BM25...）后，相对于 Full 管道的性能下降幅度（Δ MRR）。
- **细节**：
  - 下降幅度 > 0 的用红色（代表缺陷），影响最大的放顶部。
  - 虚线基线 = 0（Full 性能）。
- **效果**：直观证明"每一级管道都有贡献，且 Rerank 和 Dense 贡献最大"。

### 图组 F：系统级对比 – 并发与成本（实验5/6）
- **图型**：**帕累托散点图（Pareto Scatter）** 和 **时间分解堆叠图（Stacked Bar）**。
- **5 成本-质量**：
  - X轴：Token 消耗（对数刻度）。
  - Y轴：质量分（LLM Judge）。
  - 将 NoStop, CodeStop, BudgetOnly, Oracle 四点标出，连线画出帕累托前沿。
- **6 延迟对比**：
  - 堆叠柱状图：展示 Parallel vs Sequential 的总耗时，其中柱体内部用不同颜色填充"本地检索耗时"和"联网等待耗时"。

---

## 3. 原始数据管理规范（保护你的劳动成果）

为了防止"图不好看要重画"却没有原始数据的情况，请严格执行以下目录结构：

```text
experiment/
├── results/
│   ├── fig_data/                     # 【原始数据仓库，禁止手动修改】
│   │   ├── exp4a_mrr_matrix.csv
│   │   ├── exp4b_rounds_matrix.csv
│   │   ├── exp7_threshold_data.csv
│   │   ├── s3_language_rank.csv
│   │   └── (后续B类实验数据)
│   └── figures/                      # 【绘图脚本输出目录】
│       ├── fig1_language_bias.pdf
│       ├── fig2_window_heatmap.pdf
│       └── ...
└── scripts/
    └── plotting/                     # 【绘图专用脚本】
        ├── plot_fig1_bias.py
        ├── plot_fig2_heatmap.py
        └── plot_utils.py             # 公用样式（Times New Roman， 颜色等）
```

## 4. 绘图代码执行指令（Python + Matplotlib/Seaborn）

在 `experiment/scripts/plotting/` 下新建脚本时，必须包含以下代码头部（统一风格）：

```python
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import stats

# 1. 设置全局样式：Times New Roman， 论文标准字号
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['xtick.labelsize'] = 8
plt.rcParams['ytick.labelsize'] = 8
plt.rcParams['legend.fontsize'] = 8
plt.rcParams['figure.dpi'] = 300

# 2. 数据读取（必须从csv读取，不得硬编码数值）
df = pd.read_csv('../../results/fig_data/exp4a_mrr_matrix.csv')

# 3. 绘图逻辑 + 显著性计算
# 例如：p_value = stats.wilcoxon(df[df['lang']=='en']['rank'], df[df['lang']=='zh']['rank']).pvalue

# 4. 保存矢量图
plt.savefig('../../results/figures/fig1_bias.pdf', bbox_inches='tight')
plt.savefig('../../results/figures/fig1_bias.png', bbox_inches='tight', dpi=300)
```

---

## 5. 当前状态与下一步行动

- **已完成**：A类（4a/4b/7）的原始数据已存在 `exp4a_scan.json`, `exp4b_e2e.json`, `exp7_threshold_grid.json`。这些是绘图脚本的直接输入。
- **待完成**：B类（实验1/2）的数据。
- **流程**：跑完 B 类实验数据后 → 编写 6-8 张主图 Python 脚本 → 导出符合 CEA 规范的 PDF 矢量图 + 300 DPI PNG。
- **内存纪律**：绘图脚本与实验脚本避免同时加载 retriever；Matplotlib 关闭交互后端（Agg）。

## 6. 图表清单（与论文章节映射）

| 图号 | 内容 | 章节 | 数据文件 |
|---|---|---|---|
| fig1 | 语言偏置箱线图（en/zh rank 分布 + Wilcoxon） | §7.6 | s3_language_rank.csv |
| fig2 | 4a 候选窗口热力图（W×K→MRR） | §7.7/§8.3 | exp4a_mrr_matrix.csv |
| fig3 | 7 动态阈值双轴折线（β→pass/density） | §7.10/§8.3 | exp7_threshold_data.csv |
| fig4 | 4b 早停成本分组柱状图 | §7.7/§8.3 | exp4b_rounds_matrix.csv |
| fig5 | 中盲点恢复折线（W=20/30/40/60→zh miss rate） | §7.6 | exp4a_item_diff + s3 |
| fig6 | 实验2 多粒度查询横向条形图 | §7.4 | exp2_query_mode.csv |
| fig7 | 实验1 管道消融蝴蝶图 | §7.5 | exp1_ablation.csv |
| fig8 | 实验5/6 帕累托 + 堆叠延迟 | §7.8/§7.9 | exp5_pareto.csv / exp6_latency.csv |
| fig9 | 生产形态三对照（zh/rewrite/en × MRR/R@10/pool20；prod full=0.256 虚线） | §7.12/§8.3 | fig9_production_trio.csv |
| fig10 | 端到端三组四维均值（A/B/C × 1-5；AvsB 显著 * 标注） | §7.12 | fig10_e2e_quality.csv |

> **图9/10 数据铁律**：脚本**只读** `results/production_raw_trio.csv` 与
> `results/e2e_three_groups/summary.json`；fig_data 快照（`fig9_production_trio.csv`、
> `fig10_e2e_quality.csv`）为发布复现副本，生成后**不改**。绘图脚本：
> `plot_fig9_production_trio.py` / `plot_fig10_e2e_quality.py`（2026-08-26 新增）。