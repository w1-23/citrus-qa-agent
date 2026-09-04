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
import re

# 单条渲染预算（语义名，替换散落各文件的 1000/2000/3000 魔数）
EVIDENCE_TOOL_MAX_CHARS = 1000     # 检索工具上下文逐条（retrieve-agent 无 read 工具，预算敏感）
EVIDENCE_SNIPPET_MAX_CHARS = 2000  # 证据账本 snippet（跨轮复用/侧栏回查）
EVIDENCE_RENDER_MAX_CHARS = 3000   # 检索回执 / 写作材料包的"完整片段"安全阀

# 回答内引用编号提取（[n] 与 [Wn]/[Hn]，v8.15 引用过滤用）
_REF_NUM_RE = re.compile(r"\[(\d{1,3})\]")
_REF_WH_RE = re.compile(r"\[([WH])(\d{1,3})\]", re.IGNORECASE)


def _extract_ref_order(answer: str) -> list[str]:
    """回答内引用编号的「首次出现顺序」列表（v9.2 抽取，filter/renumber 共用）。

    - 数字 [n] 组先扫描、[Wn]/[Hn] 组随后（与 v8.15 filter 语义逐位一致）；
    - 同一编号只记首次出现位置（后续重复引用不改变顺序）。
    """
    order: list[str] = []
    seen: set = set()
    for m in _REF_NUM_RE.finditer(answer):
        rid = m.group(1)
        if rid not in seen:
            seen.add(rid)
            order.append(rid)
    for m in _REF_WH_RE.finditer(answer):
        rid = f"{m.group(1).upper()}{m.group(2)}"
        if rid not in seen:
            seen.add(rid)
            order.append(rid)
    return order


def filter_refs_by_answer(answer: str, cited_refs: list) -> list:
    """v8.15: 只保留回答文本中真实引用的证据条目，并按首次出现顺序重排。

    - 提取回答中出现的引用编号（[n] + [Wn]/[Hn]），按首次出现记序；
    - cited_refs 仅保留 ref_id ∈ 引用集合的条目，顺序 = 回答中首次出现顺序；
    - 回答中未出现的条目一律舍弃（侧栏 RAG/UCR/Web 组只显示真实引用）；
    - 回答为空 → 原样返回（防御，不误伤）。
    """
    if not answer or not cited_refs:
        return cited_refs
    order = _extract_ref_order(answer)
    if not order:
        return cited_refs  # 无任何引用编号 → 保持原样（防御路径）
    by_id = {str(it.get("ref_id")): it for it in cited_refs}
    out: list = []
    for rid in order:
        it = by_id.get(rid)
        if it is not None and it not in out:
            out.append(it)
    return out


# v9.2: 回答引用编号统一重排——[n]（数字）/ [Wn] / [Hn] 三组各自连续编号
# （单个 regex 一次扫描重写，避免先重排数字后误伤 [Wn]/[Hn]）
_REF_ALL_RE = re.compile(r"\[([WH]?)(\d{1,3})\]", re.IGNORECASE)


def renumber_refs(answer: str, cited_refs: list) -> tuple:
    """v9.2: 回答引用编号统一重排（数字 [n] 连续 1..k、[Wn] 连续 W1..Wm、[Hn] 连续 H1..Hp）。

    收敛「本地 [n] 跳号（如 [1][2][4][9]）」与前端 v8.17.3 `_buildCompactRefMap`
    的 W 专用压缩——重排统一在后端完成（一处），返回 remap 供前端按映射
    重写正文 + 重置 ref_id，避免前后端双重重排冲突。

    规则（与 filter_refs_by_answer 同源：按回答首次出现顺序）：
    - 数字 / W / H 三组**各自**按首次出现顺序连续编号（编号语义与侧栏
      同组连续展示一致；正文组内呈现顺序由模型输出决定，不强制重排）；
    - cited_refs 只保留被引用的条目，顺序 = 首次出现顺序，ref_id 写为新编号；
    - remap = {旧 ref_id: 新 ref_id}，随 citations 事件附带；不在映射内的
      残留编号从正文中清除（v9.4.1：编号一致性保证——正文出现的每个引用
      标记都必须在侧栏可解析，宁可移除标记也不保留死编号）；
    - 回答为空 / cited 为空 / 无任何引用编号 → 原样返回（防御，不误伤）。

    返回 (new_answer, new_cited, remap, dropped)。
    """
    if not answer or not cited_refs:
        return answer, cited_refs, {}, []
    order = _extract_ref_order(answer)
    if not order:
        return answer, cited_refs, {}, []

    by_id = {str(it.get("ref_id")): it for it in cited_refs}
    counters: dict[str, int] = {"": 0, "W": 0, "H": 0}
    remap: dict[str, str] = {}
    new_cited: list = []
    out_ids: set = set()
    for rid in order:
        prefix = rid[0] if rid and rid[0] in ("W", "H") else ""
        num = rid[1:] if prefix else rid
        it = by_id.get(rid)
        if it is None:
            continue
        counters[prefix] += 1
        new_id = f"{prefix}{counters[prefix]}" if prefix else str(counters[prefix])
        remap[rid] = new_id
        if rid in out_ids:
            continue  # 同一条目被多次引用：编号已归位，不重复入列
        out_ids.add(rid)
        new_item = dict(it)
        new_item["ref_id"] = new_id
        new_cited.append(new_item)

    dropped: list[str] = []

    def _repl(m):
        prefix = m.group(1).upper()
        key = prefix + m.group(2) if prefix else m.group(2)
        new = remap.get(key)
        if new is not None:
            return "[" + new + "]"
        # v9.4.1: 无法解析的编号从正文清除（防死编号，见模块级说明）
        dropped.append(key)
        return ""

    new_answer = _REF_ALL_RE.sub(_repl, answer)
    return new_answer, new_cited, remap, sorted(set(dropped))

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


def normalize_source_key(src) -> str:
    """v9.4: 来源分组键规范化——非内置来源去尾部数字（paper1→paper、
    Citrus varieties1→Citrus varieties），前端 index.html srcKey/srcMeta
    与恢复接口（/api/v2/.../citations）用同一规则，同名文件夹自动归组。"""
    s = str(src or "").strip()
    if not s:
        return "rag"
    norm = re.sub(r"\d+$", "", s).strip()
    return norm or s


def is_variety_source(src) -> str:
    """v9.4: 品种库（UCR 语义）来源判定——原始值含 'ucr'，或规范化分组键
    为 'citrus varieties'（空格/下划线写法均认；建库批次 metadata.source_type
    = "Citrus varietiesN"，未带 metadata 时回退文件夹名 "citrus_varietiesN"）。
    用于回执 [UCR] 徽标与 ucr_first 聚拢（v8.17 src_of()=="ucr" 的放行扩展）。"""
    raw = str(src or "").strip().lower()
    if not raw or raw == "rag" or raw == "historical":
        return False
    if "ucr" in raw:
        return True
    key = normalize_source_key(raw).replace("_", " ").replace("-", " ").strip()
    return key == "citrus varieties"


def source_tag(src) -> str:
    """来源徽标（回执/工具上下文 [前缀]）：内置 4 组原样；
    未知来源按品种语义给 UCR，其余 RAG（避免 [paper1] 噪音）。"""
    tag = SOURCE_TAG.get(str(src or ""))
    if tag:
        return tag
    return "UCR" if is_variety_source(src) else "RAG"


def source_label(src) -> str:
    """来源中文展示名：内置 4 组原样；品种库给 'Citrus varieties'；其余本地文献库。"""
    lab = SOURCE_LABEL.get(str(src or ""))
    if lab:
        return lab
    return "Citrus varieties" if is_variety_source(src) else "本地文献库"


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


# ════════════════════════════════════════════════════════════════
# v9.4.2 单一证据编号池 —— 回执编号 [n] 与侧栏引用编号 [n] 同去重、同序。
# 此前"检索回执"（agent_runner._dedup_evidence_items，DOI→标题、正文优先）
# 与"侧栏引用列表"（agent_loop.dedup_by_doi，仅 DOI、首现序）是两套口径，
# 模型照着回执编号写 [n]，侧栏却按另一套编号解析 → 编号"能解析但指错文献"。
# canonical_evidence_items 是回执与侧栏共用唯一入口（同池同序同 ucr_first）。
# ════════════════════════════════════════════════════════════════

def clean_doi(raw) -> str:
    """DOI 字段清洗（v9.4.5，实跑发现语料入库残留脏字符）。

    实测侧栏出现过 `10.1186/s12870-025-07372-2"` / `10.1371/journal.ppat.1010071.g001"`
    （尾部英文引号，入库 CSV/JSON 解析尾巴）→ DOI 超链接打不开；且脏 DOI 与干净
    DOI 字符串不相等会让同篇论文的 chunk 去重键分裂、同一篇以两条出现。

    规则：str 化 → strip → 循环剥除首尾成组/单边的引号与常见包裹符 → 小写留给
    调用方（展示位保留原大小写，键归一另做）。空值返回 ""。
    """
    s = str(raw or "").strip()
    if not s or s.upper() in ("N/A", "NONE", "NULL"):
        return ""
    # 反复剥除首尾的引号/空白/包裹符号（仅剥边缘，不动 DOI 内部字符）
    while s and (s[0] in "\"'`“”‘’([{<）】>" or s[-1] in "\"'`“”‘’)]}>,;"):
        s = s.strip("\"'`“”‘’([{<>）】) ,;")
    return s


def dedup_evidence_items(items: list) -> list:
    """按 DOI（无 DOI 按标题）去重，保持首次出现位置，去重碰撞时保留正文更丰富的条目。

    v8.14 起用于检索/全文证据合并（原 agent_runner._dedup_evidence_items，
    v9.4.2 上移为本模块公开函数）：同 DOI 条目碰撞时保留正文（text > abstract/snippet、
    更长优先），否则先到的摘要会把更完整证据挤掉。
    v9.4.5: 去重键用 clean_doi 归一——脏引号/大小写不再让同篇论文分裂成两条。
    """
    def _key(r):
        doi = clean_doi(r.get("doi")).lower()
        if doi:
            return ("d", doi)
        return ("t", str(r.get("title") or "").strip().lower()[:80])

    def _priority(r):
        text = str(r.get("text") or "").strip()
        fallback = str(r.get("abstract") or r.get("snippet") or "").strip()
        return (bool(text), bool(fallback), len(text or fallback))

    out, seen = [], {}
    for r in items:
        if not isinstance(r, dict):
            continue
        k = _key(r)
        if not k[1]:
            continue
        if k in seen:
            i, prev = seen[k]
            if _priority(r) > _priority(prev):
                out[i] = r
                seen[k] = (i, r)
            continue
        seen[k] = (len(out), r)
        out.append(r)
    return out


def canonical_evidence_items(items: list, ucr_first: bool = False) -> list:
    """单一证据编号池（v9.4.2）：检索回执 [n] 与侧栏引用 [n] 必须同池同序。

    - 去重口径统一：dedup_evidence_items（DOI→标题、正文优先、首次出现序）；
    - ucr_first=True 时品种族（UCR / Citrus varietiesN）条目聚拢置前——仅展示
      意图（v8.17 回执行为上移），且回执与侧栏**必须传入相同的 ucr_first 判定**，
      否则编号顺序分裂；
    - 调用方限定两处：agent_runner.build_evidence_report（回执）与 expert/light
      的引用装配（侧栏），共用同一判定来源以保同序。
    """
    main = dedup_evidence_items(list(items or []))
    if ucr_first:
        main = ([r for r in main if is_variety_source(src_of(r))]
                + [r for r in main if not is_variety_source(src_of(r))])
    return main