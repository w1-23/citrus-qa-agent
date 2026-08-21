# -*- coding: utf-8 -*-
"""证据单例 + render_evidence（v8.13-b4c，审计 §五.4 收敛点）。

chunk 全文此前在 ToolMessage / artifact / 检索回执 / 证据账本 / 历史五处各自截断
（1000 / 2000 / 3000 硬编码散落、截断 marker 也不统一），字段选择
（text → abstract → snippet）在 4 处重复实现。这里收敛为：

  - evidence_id()      稳定证据键（paper_id:chunk_index，缺省退化为内容哈希）
  - render_evidence()  单一全套字段选择 + 单一截断口径 + 统一透明标记
  - 具名预算常量       工具上下文 / 账本片段 / 回执与材料全文，按用途各一个语义名
"""
import hashlib

# 单条渲染预算（语义名，替换散落各文件的 1000/2000/3000 魔数）
EVIDENCE_TOOL_MAX_CHARS = 1000     # 检索工具上下文逐条（retrieve-agent 无 read 工具，预算敏感）
EVIDENCE_SNIPPET_MAX_CHARS = 2000  # 证据账本 snippet（跨轮复用/侧栏回查）
EVIDENCE_RENDER_MAX_CHARS = 3000   # 检索回执 / 写作材料包的"完整片段"安全阀

# 证据全文的字段回退顺序（text 含机制/数字细节，优先；摘要/片段次之）
_TEXT_KEYS = ("text", "abstract", "snippet")

# v8.15 证据来源体系（可扩展：将来加 web/patent 等只增一行）
# key = 证据 source 标识；tag = 卡片徽标/回执前缀；label = 中文展示名（手风琴组名）
SOURCE_TAG = {"rag": "RAG", "ucr": "UCR", "web": "Web", "historical": "历史"}
SOURCE_LABEL = {
    "rag": "本地文献库",
    "ucr": "UCR品种库",
    "web": "联网搜索",
    "historical": "历史证据",
}
# 来源分组展示顺序（前端侧栏手风琴固定顺序）
SOURCE_ORDER = ("rag", "ucr", "web", "historical")


def src_of(r) -> str:
    """证据来源解析：优先 chunk 上的显式字段，退化为 'rag'。"""
    if not isinstance(r, dict):
        return "rag"
    v = r.get("_src") or r.get("source") or ""
    if v:
        return str(v).strip() or "rag"
    # 兼容 UCR 品种库 chunk（无 _src 时按 source_type 判定）
    st = str(r.get("source_type") or "")
    return "ucr" if "UCR" in st else "rag"


def evidence_id(r) -> str:
    """稳定证据键：优先 ``paper_id:chunk_index``，缺省退化为标题+全文内容哈希。

    (paper_id, chunk_index) 是语料内的全局唯一键——同一论文被多批次重复索引时
    会得到不同 global_idx 却同 evidence_id，故按 evidence_id 去重才能真正收敛
    「同一 chunk 多份拷贝」。
    """
    if not isinstance(r, dict):
        return "h:empty"
    pid = str(r.get("paper_id") or "").strip()
    ci = r.get("chunk_index")
    if pid and ci is not None:
        return f"{pid}:{ci}"
    h = hashlib.md5()
    h.update(str(r.get("title") or "").encode("utf-8", "ignore"))
    for k in _TEXT_KEYS:
        if r.get(k):
            h.update(str(r[k]).encode("utf-8", "ignore"))
            break
    return "h:" + h.hexdigest()[:16]


def render_evidence(r, *, max_chars: int = EVIDENCE_RENDER_MAX_CHARS) -> str:
    """单一证据渲染：text 优先，摘要/片段次之；单一口径截断 + 透明标记。

    返回空串表示该证据无正文。``max_chars`` 使用具名预算常量，
    调用方不再各自硬编码截断阈值与 marker。
    """
    text = ""
    for k in _TEXT_KEYS:
        t = str((r.get(k) if isinstance(r, dict) else None) or "").strip()
        if t:
            text = t
            break
    if not text:
        return ""
    if len(text) > max_chars:
        return f"{text[:max_chars]} …[超长片段截断: 原文 {len(text)} 字符]"
    return text