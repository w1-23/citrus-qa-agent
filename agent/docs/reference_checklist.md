# 参考文献真实性核查清单（待联网确认）

> 状态：2026-08-26。本会话 web 工具不可用（DSH web_search 不可用、
> PowerShell 出网超时），无法在线核验 DOI/arXiv 编号。以下为**待核查清单**，
> 按风险分级。投稿前必须完成在线核验（Editorial Manager 常用 Crossref/arXiv
> 校验），或由投稿人凭学识确认。

## 高风险（编号必须核验，错一个审稿即败）

| 编号 | 引用 | 风险点 | 建议核验方式 |
|---|---|---|---|
| [3] | CaRT arXiv:2510.08517, 2025 | arXiv 编号是否真实存在并对应 CaRT 论文？ | arXiv abs 页 + 标题精确匹配 |
| [11] | "Precise zero-shot dense retrieval without relevance labels" ACL 2023 | 真实作者 Gao 等（HyDE 原文应为 arXiv:2212.10496, NeurIPS 2022 更权威）；此处编号缺失 | 补 arXiv:2212.10496 |
| [12] | "RAG-Fusion" arXiv preprint | RAG-Fusion 无正式 arXiv 编号（社区文章）；"arXiv preprint" 提法不规范 | 改引为可核实的 RAG-Fusion 来源或删 arXiv 字样 |

## 中风险（内容准确建议，作者/年份为我记忆判断，须在线复核）

| 编号 | 引用 | 建议动作 |
|---|---|---|
| [4] | Anthropic "Introducing contextual retrieval" Blog 2024 | 博客确有此文；URL 需在 Editorial Manager 填 orcid/链接时核验 |
| [8] | Singh et al., plant disease detection CEA 2022 | CEA 同名论文多，须匹配 DOI（可能不止一篇） |
| [9] | Zhang et al., crop yield prediction CEA 2023 | 同上，多候选 |
| [10] | Alam et al., precision irrigation IoT+AI CEA 2023 | 同上，多候选 |
| [2] | Gao et al., "RAG survey" arXiv:2312.10997 | 该编号即 RAG survey，可核（arXiv 存在），作者名字母拼写复核 |

## 低风险（知名文献，基本可确认；仍建议顺带核验）

| 编号 | 引用 | 备注 |
|---|---|---|
| [1] | Lewis et al., RAG NeurIPS 2020 | 经典必引，卷页可补全 |
| [5] | AutoGen arXiv:2308.08155, 2023 | 编号真实 |
| [6] | Guo et al., multi-agents survey IJCAI 2024 | 卷页待补 |
| [7] | Nie, Cross-lingual IR | 待补出版社/年份（Synthesis Lectures） |

## 提交前 Checklist

1. 对 [3]/[11]/[12] 用 arXiv API 精确核验编号与标题。
2. [8]/[9]/[10] 用 Crossref DOI 精确匹配到唯一文献，补 DOI/卷页。
3. 全文引用格式统一为 Elsevier numeric（作者, 年份, 期刊缩写, DOI）。
4. 本清单与论文参考文献段（docx [380-382]）联动：核验后逐条回填。

## 附带：附录 B 状态

- 附录 B（多粒度查询结构化输出模板）按执行计划"保持"——docx [386-402]
  内容与 `search.py:495-510 _HYDE_PROMPT` 及 `parse_hyde_structured`
  （search.py:520-554）一致，无需修改。已复核。