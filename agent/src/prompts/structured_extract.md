# 检索素材提取（v8.17.6：容错标签行格式——不依赖包裹标记，联网模式专用）

下面是 DeepSeek 原生联网对用户问题的直接回答（已作为草稿展示给用户）。
你需要**从这段回答中提炼检索素材**，用于本地柑橘文献库（RAG + UCR 品种库）召回
与此主题相关的证据，最终回答将融合原生回答与本地证据生成。

【推荐格式（容错：独立标签行，不要求包裹标记）】
用两行**独立标签行**输出检索素材，解析器按行识别，标签行可以夹在任何文字中：

MULTI_QUERY: 英文角度1|英文角度2|英文角度3
SUMMARY: 英文要点1|英文要点2|英文要点3

- 标签行以 `MULTI_QUERY:` 或 `SUMMARY:` 开头（行首，允许前导空格），冒号后是该字段内容；
- 每条内容用竖线 `|` 分隔且两侧不加空格；
- 标签行前后可有其他说明文字（解析器只取标签行，不影响提取）；
- 若某字段无法提炼，保留空标签行 `MULTI_QUERY:`、`SUMMARY:`（不要省略标签名）；
- **必须英文**（英文向量匹配远优于中文）；
- Multi-Query：回答覆盖主题的 3 个互不重复角度（每个 15-30 字符），可直接作检索查询；
- Summary：3 条高密度要点（关键术语/数据/名称，每条 10-20 字符）；
- 检索目标是**学术文献/性状/机制/政策等柑橘主题**，可按回答中的实体扩展；
- 不要输出 JSON、不要用 Markdown 代码块包裹标签行、不要输出只有标签没有内容。

【兼容格式（可选）】
也可以用 `===STRUCTURED_START===` / `===STRUCTURED_END===` 包裹上述两行——
解析器两种格式都接受，但推荐上面的独立标签行格式（更不容易出错）。

【示例】
MULTI_QUERY: citrus HLB integrated control 2025|ACP monitoring trap counts|HLB trunk injection antimicrobial Cq value
SUMMARY: ACP trap decline 9%|Cq 37.73 near threshold|dsRNA biopesticide field validation