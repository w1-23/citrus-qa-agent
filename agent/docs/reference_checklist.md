# 参考文献真实性核查清单（2026-08-26 联网核验完成，3 项待作者裁）

> 状态：2026-08-26。本会话 **已恢复出网**（arxiv.org:443 TcpTestSucceeded=True），
> 用 arXiv API（export.arxiv.org）+ Crossref API 完成在线核验。
> 关键结论：**[3]/[11]/[12] 编号风险已解除（[3] 完全匹配）；[8]/[9]/[10]
> Crossref 无法匹配到 CEA 精确文献 = 疑似占位/泛式引用，投稿前必须替换或改泛引。**

## ✅ 已核验通过（arXiv API 精确验证）

| 编号 | 引用 | 核验结果 |
|---|---|---|
| [2] | "Retrieval-augmented generation for large language models: A survey," arXiv:2312.10997, 2023 | ✅ 编号真实 = Yunfan Gao, et al.；**首作者拼写应为 Y. Gao（非 S. Gao）——需修正** |
| [3] | "CaRT: Teaching LLM agents to know when they know enough," arXiv:2510.08517, 2025 | ✅ 编号真实 = Grace Liu, Yuxiao Qu, Jeff Schneider, Aarti Singh；**作者 G. Liu 正确** |
| [5] | "AutoGen..." arXiv:2308.08155, 2023 | ✅ 编号真实 = Qingyun Wu, et al.（Q. Wu 正确） |
| [11] | "Precise zero-shot dense retrieval without relevance labels," ACL 2023 | ✅ = arXiv:2212.10496（Luyu Gao 等）；**实为 Findings of ACL, 2023（pp.1762-1777）—— "in ACL" 建议改 "Findings of ACL" 并补编号** |

## ⚠️ 高风险：Crossref 无法匹配（疑似占位/泛式引用，须作者裁）

| 编号 | 论文现引用 | Crossref 结果 | 建议 |
|---|---|---|---|
| [8] | A. K. Singh, et al., "Deep learning for plant disease detection," Comput. Electron. Agric., 2022 | CEA 2022 无 Singh 此题匹配；最接近 = Singh 2021 椰子树（10.1016/j.compag.2021.105986，DE 但题不同）或 Ferentinos 2018（10.1016/j.compag.2018.01.009，知名） | 替换为真实文献或改泛引 |
| [9] | L. Zhang, et al., "Machine learning for crop yield prediction," Comput. Electron. Agric., 2023 | CEA 2023 无 Zhang 此题匹配 | 同上（需作者提供原意文献） |
| [10] | M. A. Alam, et al., "Precision irrigation using IoT and AI," Comput. Electron. Agric., 2023 | CEA 2023 无 Alam 此题匹配 | 同上 |

## ⚠️ 中风险：规范性问题

| 编号 | 问题 | 建议 |
|---|---|---|
| [12] | K. Raudaschl, "RAG-Fusion," arXiv preprint, 2023 —— **RAG-Fusion 无形式 arXiv 论文**（社区方法，GitHub 仓库） | 改引真实来源（GitHub: Raudaschl/RAG-Fusion）或删除 |
| [4] | Anthropic "Introducing contextual retrieval" Blog 2024 —— 博客确实存在 | URL 在 Editorial Manager 填写时核验 |
| [7] | J. Nie, "Cross-lingual information retrieval," Synthesis Lectures on HLT, 2012 | 文献真实（Synthesis Lectures 2010/2012 版）；年份/出版社回填 |
| [1]/[6] | Lewis RAG NeurIPS 2020 / Guo multi-agents IJCAI 2024 | 已知权威文献，卷页可补全 |

## 投稿前 Checklist（更新）

1. **[8]/[9]/[10] 三选一**：作者提供原意真实文献 / 替换为 Crossref 命中的
   CEA 真实文献 / 改泛引（如 "recent deep-learning plant-disease studies ..."）。**待作者裁。**
2. [2] 首作者 S. Gao → **Y. Gao**；[11] 补 arXiv:2212.10496 且 "in ACL" → "Findings of ACL"。
3. [12] RAG-Fusion 改引 GitHub 仓库或删 arXiv 字样。
4. 全文引用格式统一 Elsevier numeric（作者, 年份, 期刊缩写, DOI）。
5. 回填本清单后与论文参考文献段（docx [392]）联动逐一修正。

## 附带：附录 B 状态

- 附录 B（多粒度查询结构化输出模板）按执行计划"保持"——docx [398-414]
  内容与 `search.py:495-510 _HYDE_PROMPT` 及 `parse_hyde_structured`
  （search.py:520-554）一致，无需修改。已复核。