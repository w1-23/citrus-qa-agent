# -*- coding: utf-8 -*-
"""本地与联网分离输出（v10.6 架构决策）.

背景——**为什么不再混合**
------------------------
五组对照实验（Pool C 48 题/臂，逐题配对，MDD = t·SD/√n）：

* 联网证据混入本地证据池会**显著提高幻觉率**：B（联网未过滤）vs A（不联网）
  **+20.8pp**，MDD 0.078，显著；
* 过滤管线**无法可靠挽回**：最好的混合臂 C′（模型可见级过滤）仍比 A 高
  **+15.6pp**（MDD 0.086，显著）；
* 阈值标定显示**过滤无效**：余弦 AUC 0.581、精排 AUC 0.544，各候选阈值误杀
  被引证据的比例**恒在 ~30%**；
* 内容门在筛**抓取可达性**，不是在判来源质量：被剔除的 547 条里 **545 条（99.6%）
  是"抓不到正文"**。
  ⚠️ **原稿此处曾写「内容门优先砍掉最权威来源：被剔除的 545 条中 T1+T2 占 49%，
  而存活集合里只占 33%」——该结论已被复核推翻，勿再引用。** 实测（v10.17 复核）：
  存活集（n=111）T1+T2 占 **52.3%**，被剔集（n=545）占 **49.2%** —— **存活集的
  高权威来源占比反而略高**，数据不支持"优先砍最权威来源"。逐级剔除率 T1 83.4% /
  T2 67.4% / T3 83.2%，**T1 与 T3 几乎相同**。出处：审计报告 30 §1.1；
  tier 分布原始数据 `reports/_data/webpipe_thresholds.json`
  （`tier_dist_kept` / `tier_dist_unverifiable`）。

决策——**分离而非仲裁**
----------------------
本地答案带完整引用 ``[n]``（证据池不变）；联网信息**净化后单独输出**（列 URL、
标来源分级），追加在本地答案之后，**由用户自行判断**可信度。

⚠️ **口径纪律（写进论文/专利/PPT 时不得越界）**
------------------------------------------------
这是**工程决策**，不是实验结论。实验**否定的是"混合"**；"分离更好"是**逻辑推论**，
**未经实验验证**。因此：

* ❌ 不得写"分离方案经实验验证"
* ❌ 不得写"分离降低幻觉率"
* ❌ 不得写"分离是实验结论"
* ✅ 只能写"**混合已被实验否定，分离是逻辑推论**"

开关
----
``CITRUS_SEPARATE_WEB_OUTPUT``

* ``1``（默认）：分离输出。联网条目不进引用池（无 ``[Wn]``），答案里的 ``[Wn]``
  被清除，系统在答案之后追加 **`# 联网获取信息`** 小节（v10.15 起为 Markdown H1）。
  **输出约束（v10.7，计划 §4.2）**：无 emoji；v10.15 起**允许并使用** Markdown 标题
  （``# 联网获取信息`` 为一级标题，吃前端 ``.answer h1`` 样式）。
* ``0``：**回滚**到旧的混合行为（六层管线 / F9 / F11 / F11b / F12 由各自开关决定）。
  代码全部保留，便于观察期回滚——按决策，仲裁代码"先功能关闭，确认无影响再删"。

**v10.7 多智能体架构下本模块的位置**：联网内容由 Web Agent 自主检索
（`core/web_agent_loop.py`，最多 2 轮、自评相关性/充分性）后交到这里净化并单独输出；
本模块是"输出分离"的**代码约束**那一环（计划 §2.4：输出分离由代码决定）。
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field

from src.core.web_text_cleaner import (
    clean_web_text,
    find_urls,
    host_of,
    strip_web_markers,
)

__all__ = [
    "SEPARATION", "TIER_LABELS", "DEFAULT_HEADING", "WARNING_LINE",
    "build_web_section", "append_web_section", "prepare_display_items",
    "strip_web_markers", "tier_label", "model_facing_web_notice",
]

#: 来源分级标签——**只用于标注**，绝不用于过滤/加权（决策 §2.1）
TIER_LABELS: dict[str, str] = {
    "T1": "T1 同行评议 / 官方 / 学术数据库",
    "T2": "T2 权威与行业媒体 / 机构新闻稿",
    "T3": "T3 自媒体 / 论坛 / 来源不明",
    # TU = unattributed：联网综述这类"整段返回、未逐条署名"的内容。
    # 刻意不塞进 T3——它不是"低可信"，而是**无从分级**，混为一谈会误导用户。
    "TU": "TU 来源未署名（联网综述，未逐条署名）",
}

# v10.13（改动 1）：标题去掉括号说明 —— 结尾已有 `（联网信息，请自行核验）`，
#   两处重复属于用户所指的"冗长前缀"（用户所见问题 2）。
DEFAULT_HEADING = "# 联网获取信息"

#: v10.16：一级标题探测（`# xxx` 独占一行）。
#: 用于**装配去重** —— Supervisor 已按提示词写了 `# 联网获取信息` 时，
#: `build_web_section` 不再重复拼接，避免标题出现两次。
_H1_RE = re.compile(r"^[ \t]*#[ \t]+\S", re.M)

#: 结尾提醒（**纯文本、无 emoji** —— 计划 §4.2 输出约束：无 emoji、无 `##`）
WARNING_LINE = "联网信息未经本地证据验证，请核对原始来源。"

_PREAMBLE = (
    "以下内容来自公开网络检索，未与本地文献证据混合，也未参与上方结论的生成。\n"
    "来源分级仅供参考：T1 同行评议/官方 > T2 权威与行业媒体 > T3 自媒体/来源不明。\n"
    "请自行判断其可信度与时效性。"
)

_TRUE = {"1", "true", "yes", "on"}


def _env(name: str, default: str) -> str:
    try:
        return os.environ.get(name, default)
    except Exception:                                          # noqa: BLE001
        return default


@dataclass
class SeparationConfig:
    """分离输出配置（env 驱动；``refresh()`` 供测试与运行期改开关）。"""

    enabled: bool = True
    heading: str = DEFAULT_HEADING
    #: **条目数上限；0 = 不限（默认）**。
    #: v10.9（决策 6「联网原文不截断」）：原默认 14 会**先被引用条目占满**，把
    #: `text_kind=="summary"` 的**联网综述整条挤掉**——实测 E2E：小节里 14 条引用
    #: 全有 URL，而 3354 字的综述一条都没出现。综述才是联网侧最实质的内容，
    #: 丢了它 = 用户只看到标题与链接。故默认不限；仍可用
    #: `CITRUS_SEPARATE_WEB_MAX_ITEMS=N` 显式设限。
    max_items: int = 0
    #: **每条联网正文的长度上限；0 = 不截断（默认）**。
    #: v10.9（用户决策 6「联网原文不截断」）：原默认 600 会把联网综述（web_summary）
    #: 与逐条摘要切到 600 字符——而分离输出的意义正是"把联网信息完整交给用户自行判断"，
    #: 静默截断与之相悖（历史上"中间内容被截断"的观感即来自这里）。
    #: 仍保留 `CITRUS_SEPARATE_WEB_MAX_CHARS=N` 作为**显式**上限（回滚/演示用）。
    max_chars_per_item: int = 0
    include_preamble: bool = True
    #: 是否把联网条目仍登记进回执（供侧栏展示 URL/分级）——**只展示，不可引用**
    register_for_display: bool = True
    #: **是否对模型隐藏联网正文**（默认开）。分离架构的关键一环：模型拿不到联网正文，
    #: "本地答案不受联网影响"才是**构造性**事实而非提示词约定；否则模型可能把联网事实
    #: 写进本地答案、再被净化器剥掉 [Wn]，留下无出处的断言（五臂实验里 C 臂
    #: 47.7% 无引用数值正是这个机制）。`=0` 可回退到"注入正文但要求不引用"的旧行为。
    hide_from_model: bool = True
    _extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls) -> "SeparationConfig":
        def _int(name: str, default: int) -> int:
            try:
                return int(_env(name, str(default)) or default)
            except Exception:                                  # noqa: BLE001
                return default

        return cls(
            enabled=_env("CITRUS_SEPARATE_WEB_OUTPUT", "1").strip().lower() in _TRUE,
            heading=_env("CITRUS_SEPARATE_WEB_HEADING", DEFAULT_HEADING),
            max_items=max(0, _int("CITRUS_SEPARATE_WEB_MAX_ITEMS", 0)),
            max_chars_per_item=max(0, _int("CITRUS_SEPARATE_WEB_MAX_CHARS", 0)),
            include_preamble=_env("CITRUS_SEPARATE_WEB_PREAMBLE", "1").strip().lower() in _TRUE,
            register_for_display=_env("CITRUS_SEPARATE_WEB_REGISTER", "1").strip().lower() in _TRUE,
            hide_from_model=_env("CITRUS_SEPARATE_HIDE_WEB_FROM_MODEL", "1").strip().lower() in _TRUE,
        )

    def refresh(self) -> "SeparationConfig":
        """就地刷新（测试改 env 后调用；返回 self 便于链式使用）。"""
        new = SeparationConfig.from_env()
        for k, v in vars(new).items():
            setattr(self, k, v)
        return self


#: 进程级单例（与 WEB_PIPELINE 同风格）
SEPARATION = SeparationConfig.from_env()


def tier_label(tier: str | None) -> str:
    """分级 → 展示标签；未知分级给中性标签（不猜、不隐藏）。"""
    t = str(tier or "").strip().upper()
    return TIER_LABELS.get(t, f"{t or '未分级'} 来源未分级")


def model_facing_web_notice(n_items: int, n_summary_chars: int = 0) -> str:
    """**给模型看**的联网占位说明：只报数量，不给正文/URL/编号。

    这是"分离"从提示词约定变成**构造性事实**的关键：模型拿不到联网正文，
    就不可能把联网事实写进本地答案（也就不会出现"写了事实、[Wn] 又被净化器剥掉"
    → 无出处断言）。用户仍然会收到完整联网小节。

    单一口径来源：`agent_runner.build_evidence_report` 与
    `expert_graph._execute_supervisor_tools`（工具回执）两处共用本函数，
    避免"回执说了分离、工具回执又把正文递过去"的自相矛盾。
    """
    extra = f"、原生综述正文 {n_summary_chars} 字" if n_summary_chars else ""
    return (
        f"本轮联网取回 {n_items} 条结果{extra}。按“本地与联网分离输出”架构，"
        "这些内容**不进入你的证据、不参与你的回答、不占用引用编号**；"
        "系统会在你的答案之后单独输出给用户（逐条 URL + 来源分级标签，由用户自行判断）。"
        "你只需依据本地 [n] 证据作答，不必在正文里解释联网情况。"
    )


def _item_url(it: dict) -> str:
    for key in ("url", "link", "source_url"):
        v = str(it.get(key) or "").strip()
        if v.startswith("http"):
            return v
    # 正文里可能带 URL（联网综述常见）
    for u in find_urls(str(it.get("text") or it.get("text_preview") or "")):
        return u
    return ""


def _item_text(it: dict) -> str:
    """取条目正文。**`abstract` 放在最后**作兜底。

    v10.9：`deepseek_web.py:187` 同时写 `abstract` 与 `snippet`（同一来源），故正常路径
    命中 `snippet` 即可；但 `abstract` 是**上游原始字段名**（搜索 API 的引用摘要），
    归档/其它通路可能只带它 ⇒ 兜底读一下，避免展示层正文意外为空。
    顺序有讲究：放最后，不会覆盖更精确的字段。
    """
    for key in ("text", "text_full", "text_preview", "snippet", "summary",
                "content", "abstract"):
        v = it.get(key)
        if v and str(v).strip():
            return str(v)
    return ""


def prepare_display_items(web_items: list[dict] | None,
                          cfg: SeparationConfig | None = None) -> list[dict]:
    """把内部联网条目转成**可展示**的净化条目（去 [Wn]、提 URL、标分级）。

    返回 ``[{title, url, host, tier, tier_label, text, chars, ref_id}]``；只做展示准备，
    不参与引用池。

    **v10.13（改动 3）**：新增 `ref_id`（`W1..Wn`）—— 按**输入顺序**编号，
    与 `agent_runner.build_evidence_report` 里给 Supervisor 看的 `[W1]…[Wn]`
    **同源同序**，使 Supervisor 在联网综述里写的 `[W3]` 与侧栏的 `W3` 指向同一条。
    编号在**排序之前**固定，故展示时按 tier 排序**不会打乱编号**。
    """
    c = cfg or SEPARATION
    out: list[dict] = []
    seen: set[str] = set()
    for it in (web_items or []):
        if not isinstance(it, dict):
            continue
        url = _item_url(it)
        host = host_of(url)
        # 去重键：优先 URL，其次标题（无 URL 的综述条目）
        title = strip_web_markers(str(it.get("title") or "")).strip()
        key = (url or title).lower().rstrip("/")
        if not key or key in seen:
            continue
        seen.add(key)
        tier = str(it.get("source_tier") or "").strip().upper()
        if not tier:
            try:
                from src.core.source_tiers import classify_url
                tier = classify_url(url).tier if url else "T3"
            except Exception:                                  # noqa: BLE001
                tier = ""
        text = clean_web_text(_item_text(it), c.max_chars_per_item)
        if not (title or url or text):
            continue
        # v10.9：**综述条目**（`display_only` / `text_kind=="summary"`）标记出来。
        #   它是联网侧最实质的内容，**不受条目数上限约束**——否则上限先被引用条目
        #   占满，综述被整条挤掉（实测 E2E：14 条引用全在、3354 字综述一条不见）。
        is_sum = bool(it.get("display_only")
                      or str(it.get("text_kind") or "") == "summary")
        if c.max_items and len(out) >= c.max_items and not is_sum:
            # 已达上限且本条不是综述 ⇒ 跳过（用 continue 而非 break：后面的综述仍可进来）
            continue
        # ── v10.14（阶段 2 实测）：条目 `title` **本身就是 URL** 时改用 host ─────
        #   实测（17:34 运行）：某条目 title = `https://public-pages-files-2025.
        #   frontiersin.org/journals/pla…`，被原样当标题渲染 ⇒ `done.answer` 出现
        #   明文链接（计划 §8.3 要求明文 URL 0 次）。
        #   这是**展示层**决定：链接已由 🔗 折叠给出、侧栏也保留完整 URL，
        #   **不改动任何联网正文内容**（`text` 原样）。
        _disp_title = title
        if re.match(r"^\s*(?:https?://|www\.)", _disp_title, re.IGNORECASE):
            _disp_title = host or "联网条目"
        elif "http" in _disp_title.lower():
            # 标题**内嵌** URL 片段（实测：`Search Results (67) - Biology 2025,
            # 14(4), 335; https://doi`）。链接已由右侧 🔗 给出 ⇒ 展示层去掉这段，
            # 其余标题文字原样保留（不改动任何联网正文内容 `text`）。
            _disp_title = re.sub(r"https?://\S*", "", _disp_title)
            _disp_title = re.sub(r"\s{2,}", " ", _disp_title).strip(" -;，,、")
        out.append({
            "title": _disp_title or (host or "联网条目"),
            # ── v10.13-B（用户决策：恢复 `[Wn]`）──────────────────────────────
            #   `ref_id` 来自 `expert_graph` 打的 `_w_id`（与 Supervisor 看到的
            #   `[W1]…[Wn]` 同源同序）。**仅供前端内部对齐，不在侧栏显示编号**
            #   （用户口径：侧栏显示「标题 + 🔗」）。
            "ref_id": str(it.get("_w_id") or ""),
            # v10.13（用户口径）：**取消 `[Wn]`** ⇒ 不再产出 `ref_id`。
            #   侧栏只按「标题 🔗」展示；正文里不出现任何联网编号。
            "url": url,
            "host": host,
            "tier": tier,
            "tier_label": tier_label(tier),
            "text": text,
            "chars": len(text),
            "is_summary": is_sum,
            # ── v10.13（Task E）：侧栏按 round 分组所需的轮次标记 ──────────────
            #   条目可能带 `round_no`（由 web_agent_loop 逐轮打标，见 search_both 的
            #   `_tag_round`）；缺省视为第 1 轮（单轮检索的绝大多数情况）。
            "round_no": int(it.get("round_no") or it.get("_round") or 1),
        })

    # ── v10.13（Task E）：**先按 round 升序，再按 tier 优先级排序** ────────────
    #   用户要求："一个 round 一个 web 侧栏，每一个 url 分开，T1T2T3 优先排列"。
    #   tier 优先级：T1 > T2 > T3 > TU（TU = 未署名综述，放最后）。
    #   同 tier 内保持原顺序（稳定排序），便于与工具回执对照。
    _TIER_ORDER = {"T1": 0, "T2": 1, "T3": 2, "TU": 3}
    out.sort(key=lambda x: (x.get("round_no") or 1,
                            _TIER_ORDER.get(str(x.get("tier") or "").upper(), 4)))
    # ── v10.13（Task E）：综述条目**排在最后** ────────────────────────────────
    #   用户要"每一个 url 分开，T1T2T3 优先排列" ⇒ 未署名的 TU 综述（无独立 URL）
    #   不应插在 T1 中间。上面按 tier 排序已把 TU 归到最后（_TIER_ORDER[TU]=3），
    #   但**综述条目**可能因 `round_no` 较小而排到前面；这里再稳定下沉一次。
    out.sort(key=lambda x: (0 if not x.get("is_summary") else 1,
                            x.get("round_no") or 1,
                            _TIER_ORDER.get(str(x.get("tier") or "").upper(), 4)))
    return out


def build_web_section(web_items: list[dict] | None,
                      cfg: SeparationConfig | None = None,
                      web_prose: str = "") -> str:
    """构建"联网信息"小节。无条目时返回空串（不输出空壳标题）。

    **输出约束**：无 emoji。v10.15 起标题为 **Markdown H1** `# 联网获取信息`
    （吃前端 `.answer h1` 样式），与第一段的 `# 本地文献库` 对称；
    条目清单**不进正文**（走侧栏），结尾只留一行 `（联网信息，请自行核验）`。
    """
    c = cfg or SEPARATION
    items = prepare_display_items(web_items, c)
    # ── v10.14（阶段 2 · 计划 §一 1.1「联网小节的生产者 → Supervisor 提炼输出」）──
    #   原来小节的**正文**来自 Web-Agent 的原始返回（DeepSeek 原生综述，含碎碎念）；
    #   现在改为**Supervisor 提炼的第二段**（`web_prose`）。
    #   条目清单（标题 + 分级 + 🔗 链接）仍由代码组装（那是确定性数据，模型不该抄）。
    #   传空串/None 时退回原行为（回滚档兼容）。
    _prose = str(web_prose or "").strip()
    if _prose:
        # 用 Supervisor 的提炼正文替换"综述条目"的 text；并把综述条目**排到最后**
        # （用户要求"每一个 url 分开、T1T2T3 优先排列" ⇒ 提炼正文不该插在 T1 中间）。
        items = [it for it in items if not it.get("is_summary")]
        items.append({
            "title": "联网检索综述（Supervisor 提炼）",
            "url": "",
            "host": "",
            "tier": "TU",
            "tier_label": tier_label("TU"),
            "text": _prose,
            "chars": len(_prose),
            "is_summary": True,
            "round_no": 0,
        })
    if not items:
        return ""
    # ── v10.13（改动 1 配套 · 任务书检查项 1/2/3/4/5）──────────────────────
    #   用户所见问题：正文里"标题 + 🔗 列表"消耗大量 token、前缀冗长、可靠性提示整段啰嗦。
    #   **条目清单移到侧栏**（改动 3.2/3.3 已把 4 组 + 标题 🔗 做进侧栏）
    #   ⇒ 正文**不再**输出条目列表、不再输出 `_PREAMBLE`、不再输出 `WARNING_LINE`，
    #      结尾只留一句话 `（联网信息，请自行核验）`。
    #   实测口径：正文内 🔗 数量 = 0、无"以下内容来自公开网络检索"、
    #             无"来源可靠性提示"、无"来源分级仅供参考"、结尾句恰好 1 次。
    _NOTE = "（联网信息，请自行核验）"
    # ── v10.16 实测修复：**一级标题重复** ────────────────────────────────────
    #   现象（用户真机截图）：正文里「联网获取信息」出现**两次** ——
    #     `# 联网获取信息`（Supervisor 按 B.1 提示词写的）
    #     `# 联网获取信息`（本函数又拼了一次 `c.heading`）
    #   根因：v10.15 的 B.1（要求模型写一级标题）与 B.3（代码拼 DEFAULT_HEADING）
    #   **撞车**。两处都对，但只能留一处。
    #   修法：**装配去重**（与结尾句 `_NOTE` 同款处理，不是内容清洗）——
    #   若正文已自带一级标题，就不再拼 `c.heading`；模型漏写时仍由代码兜底。
    _prose_heads = "\n".join(
        str(it.get("text") or "") for it in items if it.get("is_summary"))
    _has_h1 = bool(_H1_RE.search(_prose_heads))
    parts: list[str] = [] if _has_h1 else [c.heading, ""]
    # 只取**综述正文**（Supervisor 提炼的第二段）；条目清单不进正文
    _bodies = [str(it.get("text") or "").strip()
               for it in items if it.get("is_summary") and str(it.get("text") or "").strip()]
    for _b in _bodies:
        parts.append(_b)
        parts.append("")
    if not _bodies:
        # 边界：Supervisor 第二段为空（如联网无内容）⇒ 正文只给一句指引，不列条目
        parts.append("本轮联网检索未返回可提炼的内容。")
        parts.append("")
    # ── v10.13 实测修复：结尾句**去重** ────────────────────────────────────
    #   实测（首次 E2E）出现 **2 次**：Supervisor 按提示词写了一次，
    #   本函数又追加一次。⇒ 若正文**已以该句结尾**，就不再追加。
    #   （这是**装配去重**，不是内容清洗：正文逐字保留。）
    _joined = "\n".join(parts).rstrip()
    if not _joined.endswith(_NOTE):
        parts.append(_NOTE)
    return "\n".join(parts).rstrip() + "\n"


def append_web_section(answer: str, web_items: list[dict] | None,
                       cfg: SeparationConfig | None = None,
                       web_prose: str = "") -> str:
    """把联网信息小节**追加**在本地答案之后。

    分离语义的两条硬约束：
      1. 本地答案**逐字不动**（只做出口消毒：去 [Wn]、消毒假换行）；
      2. 联网信息只追加、不改写正文——用户看到的是"答案 + 独立参考"。
    无联网条目时返回消毒后的原答案（不追加任何东西）。
    """
    c = cfg or SEPARATION
    # 开关关闭 = **完全不动**（回滚语义必须是"逐字返回原答案"，包括保留 [Wn]）：
    # 早先这里仍会走 clean_answer_text，于是 `=0` 时虽然不追加小节、却仍把 [Wn]
    # 剥掉了——回滚不彻底。由实测（verify_separation_on_archived §4）发现并修正。
    if not c.enabled:
        return answer or ""
    from src.core.web_text_cleaner import clean_answer_text
    base = clean_answer_text(answer or "").rstrip()
    section = build_web_section(web_items, c, web_prose=web_prose)
    if not section:
        return base
    # ── v10.15（B.3）：标题改为 **Markdown H1** `# 联网获取信息` ──────────────
    #   用户要求"联网获取信息"是**一级标题**（加粗加大），与第一段的
    #   `# 本地文献库` 对称；前端 `.answer h1` 会渲染成 `<h1>`。
    #   空行分隔：Markdown 的 ATX 标题（`#`）需要**前置空行**才算标题，
    #   否则可能被当作上一段正文的一部分。此处 `\n\n` 已满足。
    return f"{base}\n\n{section}" if base else section
