# -*- coding: utf-8 -*-
"""联网证据六层过滤管线（v10.5，计划 §4.3 / 执行顺序第 2 步）。

架构（顺序即优先级；逐层可开关）::

    L1 source_tiers      确定性域名分级（只打标；T3 不作硬门，见报告 20）
    L2 内容有效性门      非空 / 最小长度 / 语言 / 目标实体 —— **优先级最高**
    L3 元数据预过滤      URL 去重、时效窗、黑名单、可选 tier 白名单
    L4 embedding 相关性  查询向量 vs 条目正文余弦（本地 fastembed，无 API 成本）
    L5 adjudicate        复用生产 fast 模型裁决（有 API 成本，上限沿用生产口径）
    L6 全文精排          cross-encoder 对**抓到的正文**精排 + 分数阈值

设计约束（来自前几轮实测，违反任何一条都会得出错误结论）：

1. **「抓取失败」≠「内容无效」**。T1 官方域名普遍 403（实测 EFSA/USDA-APHIS/EUR-Lex/
   NationalPlantBoard），L2 必须把"无正文"记成独立状态 ``unverifiable``，
   默认**不**当无效剔除（开关 ``CITRUS_WEBPIPE_DROP_UNVERIFIABLE=1`` 才剔）。
2. **T3 不得当硬门**：报告 20 实测 T3 中 95.5% 是 ``default_unknown``（不在 193 域名冷启动表内），
   当硬门等于按"没听说过"删证据。故 L1 只打标，剔除需显式开 ``CITRUS_WEBPIPE_DROP_T3=1``。
3. **语言只打标不默认剔除**：中文提问问欧盟 MRL 时，英文页面才是**一等来源**。
   只有显式设 ``CITRUS_WEBPIPE_LANG=zh_only|en_only`` 才按语言剔除。
4. **失败一律 fail-open**：任何一层异常都不阻断回答，异常写进报告（不静默）。
5. **默认全关**：``CITRUS_WEBPIPE=0`` 时不改变任何行为（与主实验/专利基线一致）。

逐请求记录生效状态：``run_web_pipeline`` 的 report 含每层 n_in/n_out/dropped/ms 与
逐条目状态（tier / state / chars / lang / cos / rerank / dropped_by），
由调用方挂进 ``audit_report``，随 ``done.audit`` 下发并落盘 —— 消融时可直接统计"哪层动了多少"。
"""
from __future__ import annotations

import logging
import math
import os
import re
import time
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# ── 层名（顺序即优先级）────────────────────────────────────────────────
#   `fetch` 不属计划里的"六层过滤"，而是**内容交付前置层**：
#   没有它，L2 内容有效性门面对的永远是空正文（原生联网只回标题+URL），
#   闸门会把 100% 的联网证据判成"无内容"。故放在最前，且可单独关闭。
LAYER_ORDER = ("fetch", "tiers", "content", "meta", "embed", "adjudicate", "rerank")

_TRUE = {"1", "true", "yes", "on", "y", "t"}

# 柑橘领域兜底关键词（仅在实体表不可用时用于"是否提到目标实体"）
_CITRUS_HINTS = ("citrus", "柑橘", "柑桔", "橙", "柚", "橘", "柑", "柠檬", "lemon",
                 "orange", "mandarin", "pomelo", "grapefruit", "lime", "kumquat",
                 "huanglongbing", "黄龙病", "hlb", "aphis", "eppo", "diaphorina")
_CJK = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")
_LATIN = re.compile(r"[A-Za-z]")
_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9_\-]{3,}")
_STOP = {"that", "this", "with", "from", "have", "what", "which", "when", "where",
         "does", "were", "will", "been", "than", "then", "they", "their", "there",
         "about", "into", "over", "under", "between", "during", "after", "before",
         "according", "based", "using", "used", "also", "such", "more", "most"}

# ── 语言判据（L2 内容有效性门第四判据）────────────────────────────────
#   仅判"**非工作文字系统**"：正文以 西里尔/阿拉伯/泰/天城文/谚文 为主即剔除。
#   刻意**不**剔日文：`japanfruit.jp` 是已白名单化的 T2 来源（报告 20），
#   按"非中文就删"会把它连带删掉——这正是"语言门最容易误杀"的地方。
_OTHER_SCRIPTS = {
    "ko": re.compile(r"[\uac00-\ud7af]"),
    "ru": re.compile(r"[\u0400-\u04ff]"),
    "ar": re.compile(r"[\u0600-\u06ff\u0750-\u077f]"),
    "th": re.compile(r"[\u0e00-\u0e7f]"),
    "hi": re.compile(r"[\u0900-\u097f]"),
}
_CJK_IDEO = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")

# ── 元数据预过滤（L3）：标题党 / 广告词（用户指定判据）────────────────
#   ⚠️ 领域假阳性（实测踩坑）：初版把「推广」列入广告词，而**品种推广/推广应用**
#   是农学标准术语（如"新品种推广面积"），一命中就删掉了真实证据。
#   故此处只保留**纯商业话术**，任何可能与农学语义混淆的词一律不收。
_AD_PATTERNS = ("点击查看", "点击进入", "点击了解", "限时抢购", "限时优惠", "免费领取",
                "扫码关注", "扫码咨询", "加微信", "加qq", "立即咨询", "立即购买",
                "代理加盟", "招商加盟", "秒杀", "抢购", "多少钱一台", "价格表",
                "buy now", "shop now", "add to cart", "coupon code", "promo code",
                "sponsored post", "advertisement", "best price guarantee",
                "click here to", "sign up now", "subscribe now", "limited offer")


def script_class(text: str) -> str:
    """文字系统粗判：``cjk``（中文/日文汉字）/ ``latin`` / 其它语言码 / ``none``。

    与 `lang_guess` 的区别：`lang_guess` 按比例判 zh/en（用于统计），
    本函数判**该不该因语言被剔除**（用于闸门），两者刻意分开。
    """
    t = text or ""
    n = len(t)
    if not n:
        return "none"
    for code, pat in _OTHER_SCRIPTS.items():
        if len(pat.findall(t)) / n >= 0.10:
            return code
    cjk = len(_CJK_IDEO.findall(t))
    lat = len(_LATIN.findall(t))
    if cjk and cjk >= lat:
        return "cjk"
    if lat:
        return "latin"
    return "none"


def ad_like(text: str) -> Optional[str]:
    """命中标题党/广告词则返回命中的词（供审计留痕）。"""
    low = (text or "").lower()
    for p in _AD_PATTERNS:
        if p in low:
            return p
    return None


class WebPipelineFlags:
    """环境变量开关（默认全关）。仿 ``enhance/flags`` 的语义，但**独立文件**：
    不动已冻结的 ``agent/src/enhance/*``。"""

    def __init__(self) -> None:
        self.reload()

    def reload(self) -> None:
        env = os.environ.get
        self.enabled = env("CITRUS_WEBPIPE", "0").strip().lower() in _TRUE
        raw = env("CITRUS_WEBPIPE_LAYERS", "").strip()
        if raw:
            want = [x.strip().lower() for x in raw.split(",") if x.strip()]
            self.layers = tuple(x for x in LAYER_ORDER if x in want)
            self.unknown_layers = tuple(x for x in want if x not in LAYER_ORDER)
        else:
            self.layers = LAYER_ORDER
            self.unknown_layers = ()
        # L2 内容有效性门
        self.min_chars = int(env("CITRUS_WEBPIPE_MIN_CHARS", "180"))
        self.drop_unverifiable = env("CITRUS_WEBPIPE_DROP_UNVERIFIABLE", "0").strip().lower() in _TRUE
        self.lang_mode = env("CITRUS_WEBPIPE_LANG", "").strip().lower()  # ""|zh_only|en_only
        self.require_entity = env("CITRUS_WEBPIPE_REQUIRE_ENTITY", "0").strip().lower() in _TRUE
        # L3 元数据
        self.drop_t3 = env("CITRUS_WEBPIPE_DROP_T3", "0").strip().lower() in _TRUE
        self.max_per_host = int(env("CITRUS_WEBPIPE_MAX_PER_HOST", "0"))  # 0=不限
        # L0 内容交付（抓原文；失败 fail-open 记 unverifiable）
        self.fetch_max = int(env("CITRUS_WEBPIPE_FETCH_MAX", "6"))
        self.fetch_workers = int(env("CITRUS_WEBPIPE_FETCH_WORKERS", "6"))
        # ★ 硬墙钟预算（秒）：实测未设预算时 L0 单题可耗 130–200 s（15s 超时 × 重试 + 跳转跟随），
        #   把整臂墙钟时间拖到数小时。超预算即停止等待，未拿到的条目记 unverifiable（不判"内容无效"）。
        self.fetch_budget_s = float(env("CITRUS_WEBPIPE_FETCH_BUDGET_S", "20"))
        # L4 embedding 相关性
        #   ⚠️ 默认 0.75 = **留白不生效**：首轮标定只有 16 条真正文（e5 余弦区间 0.805–0.878），
        #   样本量不足以定阈值；按"宁可不删也不误删"的原则先设为低于实测最小值，
        #   待三臂重跑（≈48×9 条带正文条目）后再按数据分位重定（见 reports/_data/webpipe_calibration.json）。
        self.embed_min = float(env("CITRUS_WEBPIPE_EMBED_MIN", "0.75"))
        # L6 全文精排
        self.rerank_top_k = int(env("CITRUS_WEBPIPE_TOP_K", "0"))  # 0=不截断
        self.rerank_min = float(env("CITRUS_WEBPIPE_RERANK_MIN", "0.0"))
        # F11：把抓到的正文片段注入**模型可见**的工具回执（0=关闭，生产默认关闭）
        self.excerpt_chars = int(env("CITRUS_WEBPIPE_EXCERPT_CHARS", "0"))
        # F12（E2）：如何处理"联网综述"——模型能读、但回执里从来没有的那段正文。
        #   keep     : 现状（综述进上下文、不进回执 → 从中取的数值按构造不可核验）
        #   cite     : 把综述登记为**可引用的联网条目**（有编号、侧栏可见、判官可核验）
        #   suppress : 保留在上下文里但**明确禁止引用**（数值只能来自带正文片段的编号）
        #   separate : **v10.6 分离输出架构的默认档**——综述既不进引用池（无编号、
        #              不可引用），也不被丢弃，而是作为"来源未署名"的联网条目
        #              **随分离小节单独呈现给用户**（见 web_output_separator）。
        #              这正是"F12 改为 separate 策略"的落点：同一段文本从
        #              "模型能读但谁也核验不了"变成"用户能看到并自行判断"。
        self.summary_policy = (env("CITRUS_WEBPIPE_SUMMARY_POLICY", "separate")
                               .strip().lower() or "separate")

    def any_on(self) -> bool:
        return bool(self.enabled and self.layers)

    def as_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}


WEB_PIPELINE = WebPipelineFlags()


# ── 工具函数（确定性，无副作用）────────────────────────────────────────
def item_text(it: dict) -> str:
    """条目正文：抓到的原文优先，其次片段/摘要。

    回退链与 `build_cited_refs` 同源，并额外认 `text_full` / `text_preview`
    （F7 归档后回执里的字段名），使**离线复算与在线过滤见到同一份正文**。
    """
    return str(it.get("text") or it.get("text_full") or it.get("snippet")
               or it.get("content") or it.get("abstract")
               or it.get("text_preview") or "").strip()


def text_kind(it: dict) -> str:
    if it.get("text_kind"):
        return str(it["text_kind"])
    return "fulltext" if str(it.get("text") or "").strip() else "snippet"


def lang_guess(text: str) -> str:
    if not text:
        return "none"
    cjk = len(_CJK.findall(text))
    lat = len(_LATIN.findall(text))
    tot = max(1, cjk + lat)
    if cjk / tot >= 0.15:
        return "zh"
    if lat / tot >= 0.50:
        return "en"
    return "other"


def entity_terms(query: str) -> list[str]:
    """目标实体词：优先用柑橘实体归一（只对齐，不改写）；不可用则退化为关键词+长词。

    **绝不**用实体归一做查询扩展（机制实验 exp3 已证会挤出 gold），此处只取词面。
    """
    terms: list[str] = []
    try:
        from src.enhance.entity_norm import normalize as _norm
        for obj in (_norm(query) or [])[:12]:
            t = str(obj if isinstance(obj, str) else (obj or {}).get("id") or "").strip()
            if t:
                terms.append(t)
    except Exception:
        pass
    q = (query or "").lower()
    for k in _CITRUS_HINTS:
        if k in q:
            terms.append(k)
    for m in _TOKEN.finditer(query or ""):
        w = m.group(0).lower()
        if w not in _STOP:
            terms.append(w)
    seen, out = set(), []
    for t in terms:
        tl = t.lower()
        if len(tl) >= 3 and tl not in seen:
            seen.add(tl)
            out.append(tl)
    return out[:24]


def mentions_entity(text: str, terms: list[str]) -> bool:
    if not terms:
        return True
    low = (text or "").lower()
    return any(t in low for t in terms)


def cosine(a, b) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _host(url: str) -> str:
    try:
        from urllib.parse import urlparse
        h = urlparse(url if "//" in url else "//" + url, scheme="http").hostname or ""
    except Exception:
        h = ""
    return (h or url or "").strip().lower()


def fetch_texts_for_items(items: list[dict], *, max_items: int = 0,
                          budget_s: float = 0.0, query: str = "") -> dict:
    """**共用的有界抓取**：给缺正文的条目抓网页正文，写回 `text`/`text_kind`/`text_chars`。

    为什么要共用：F7（生成后补齐回执正文）原先直接调 `enhance.web_fulltext.build_web_blocks`，
    它是**顺序阻塞**的循环（每条 15 s 超时，14 条最坏 ≈7 min），而调用点在
    `_assemble_supervisor_answer` 里、**同步执行于事件循环中** → 实测把整个网关卡住
    （臂 B 单题 ~9 min，3 个并发请求相互串行）。
    这里与 L0 用**同一份实现**（线程池 + 墙钟预算 + fail-open），既修吞吐也保证
    「评估与生产共用一套代码」。

    返回统计：n_need / n_ok / n_fail / n_timeout / ms。
    """
    from src.enhance.web_fulltext import chunk_text, fetch_page
    todo = [(i, it) for i, it in enumerate(items) if not item_text(it)]
    if max_items > 0:
        todo = todo[:max_items]
    stat = {"n_need": len(todo), "n_ok": 0, "n_fail": 0, "n_timeout": 0, "ms": 0.0}
    if not todo:
        return stat
    from concurrent.futures import as_completed
    from src.core.executors import WEB_FETCH_EXECUTOR

    def _one(pair):
        _i, it = pair
        try:
            return it, fetch_page(str(it.get("url") or it.get("link") or ""))
        except Exception:
            return it, None

    t0 = time.perf_counter()
    fetch_workers = max(1, min(int(WEB_PIPELINE.fetch_workers or 1), 12))
    futs = {WEB_FETCH_EXECUTOR.submit(_one, p): p for p in todo[:fetch_workers]}
    pending_items = iter(todo[fetch_workers:])
    done = set()
    deadline = time.monotonic() + max(0.1, budget_s or 20.0)
    try:
        while futs and time.monotonic() < deadline:
            timeout = max(0.1, deadline - time.monotonic())
            for fut in as_completed(tuple(futs), timeout=timeout):
                it, body = fut.result()
                done.add(fut)
                futs.pop(fut, None)
                if body:
                    chunks = chunk_text(body) or [body]
                    it["text"] = max(chunks, key=len)
                    # Keep all relevant chunks for later evidence selection.
                    it["_chunks"] = list(chunks)[:8]
                    it["text_kind"] = "fulltext"
                    it["text_chars"] = len(body)
                    stat["n_ok"] += 1
                else:
                    stat["n_fail"] += 1
                try:
                    next_item = next(pending_items)
                except StopIteration:
                    continue
                futs[WEB_FETCH_EXECUTOR.submit(_one, next_item)] = next_item
                break
    except TimeoutError:
        pass
    stat["n_timeout"] = len(todo) - stat["n_ok"] - stat["n_fail"]
    stat["ms"] = round((time.perf_counter() - t0) * 1000, 1)
    return stat


def best_excerpt(it: dict, query: str, max_chars: int) -> str:
    """从抓到的块里挑**与问题最相关**的一块，截取前 `max_chars` 字作为注入片段。

    为什么按问题挑而不是取最长块：实测模型"给出无法核验的具体数值"的根因是
    它只看到标题+URL，只能凭联网摘要推测；把**真正含答案的那一块**放进上下文，
    才有可能让数值有据可依（F11，臂 D 的干预）。
    词面重合是零成本的代理指标（与机制实验一致：可用性比排序更重要）。
    """
    chunks = list(it.get("_chunks") or [])
    if not chunks:
        t = str(it.get("text") or "")
        return t[:max_chars] if t else ""
    toks = [t for t in entity_terms(query or "") if len(t) >= 3][:16]
    if not toks:
        return max(chunks, key=len)[:max_chars]

    def score(c: str) -> tuple:
        low = (c or "").lower()
        hit = sum(low.count(t) for t in toks)
        return (hit, len(c))
    return max(chunks, key=score)[:max_chars]


# ── 各层实现 ──────────────────────────────────────────────────────────
def _layer_fetch(items: list[dict], ctx: dict):
    """L0：内容交付前置层 —— 抓原文（并行、有界、fail-open）。

    为什么必须在过滤**之前**：L2 内容有效性门要判"非空/长度/语言/实体"，
    而 DeepSeek 原生联网只回标题+URL（实测 480/480 条 snippet 为空）。
    不先交付正文，闸门只会把 100% 的联网证据判成"无内容"——
    这正是前几轮反复强调"内容有效性门必须在内容交付之后才可信"的原因。
    """
    f = WEB_PIPELINE
    todo = [(i, it) for i, it in enumerate(items) if not item_text(it)]
    if f.fetch_max > 0:
        todo = todo[:f.fetch_max]
    stat = fetch_texts_for_items(items, max_items=f.fetch_max,
                                 budget_s=f.fetch_budget_s,
                                 query=str(ctx.get("query") or ""))
    stat["max"] = f.fetch_max
    stat["budget_s"] = f.fetch_budget_s
    for _i, it in todo:
        st = ctx["state"].get(id(it))
        if st is not None:
            st["fetch"] = "ok" if str(it.get("text") or "") else "fail_or_timeout"
    return items, stat


def _layer_tiers(items: list[dict], ctx: dict):
    """L1：确定性域名分级，只打标（TierResult 见 source_tiers）。"""
    from src.core.source_tiers import classify_url
    for it in items:
        res = classify_url(str(it.get("url", "")), str(it.get("doi", "")))
        it["source_tier"] = res.tier
        it["tier_rule"] = res.rule
        ctx["state"][id(it)].update(tier=res.tier, tier_rule=res.rule)
    dist: dict = {}
    for it in items:
        t = str(it.get("source_tier") or "?")
        dist[t] = dist.get(t, 0) + 1
    rules: dict = {}
    for it in items:
        r = str(it.get("tier_rule") or "?")
        rules[r] = rules.get(r, 0) + 1
    note = {"tier_dist": dist, "rule_dist": rules,
            "note": "只打标（默认不剔）：T3 中 95.5% 为 default_unknown，当硬门=按没听说过删证据"}
    if not WEB_PIPELINE.drop_t3:
        return items, note
    kept = [it for it in items if it.get("source_tier") != "T3"]
    dropped = [it for it in items if it.get("source_tier") == "T3"]
    for it in dropped:
        ctx["state"][id(it)]["dropped_by"] = "tiers"
        ctx["state"][id(it)]["reason"] = "T3（显式开启 drop_t3）"
    note["note"] = "T3 显式剔除（非常规路径，仅消融用）"
    note["n_t3_dropped"] = len(dropped)
    return kept, note


def _layer_content(items: list[dict], ctx: dict):
    """L2：内容有效性门（**优先级最高**）。四判据：非空、最小长度、语言、目标实体。

    - 无正文 → ``unverifiable``（抓取失败/原生联网只回标题；**不等于内容无效**）
    - 过短 → ``too_short``；语言不符（仅显式设定时）→ ``lang_mismatch``；
      未提目标实体（仅显式开启时）→ ``no_entity``
    """
    f = WEB_PIPELINE
    terms = ctx["terms"]
    kept, dropped = [], []
    for it in items:
        txt = item_text(it)
        st = ctx["state"][id(it)]
        st["chars"] = len(txt)
        st["text_kind"] = text_kind(it)
        st["lang"] = lang_guess(txt)
        st["script"] = script_class(txt)
        st["mentions_entity"] = mentions_entity(txt, terms)
        if not txt:
            st["state"] = "unverifiable"
            (dropped if f.drop_unverifiable else kept).append(it)
            if f.drop_unverifiable:
                st["dropped_by"], st["reason"] = "content", "无正文（抓取失败/仅标题）"
            continue
        if len(txt) < f.min_chars:
            st["state"] = "too_short"
            st["dropped_by"] = "content"
            st["reason"] = f"正文 {len(txt)} 字 < 阈值 {f.min_chars}"
            dropped.append(it)
            continue
        if f.lang_mode == "worklang" and st["script"] in _OTHER_SCRIPTS:
            st["state"] = "lang_mismatch"
            st["dropped_by"] = "content"
            st["reason"] = f"非工作文字系统（{st['script']}）"
            dropped.append(it)
            continue
        if f.lang_mode == "zh_only" and st["lang"] != "zh":
            st["state"] = "lang_mismatch"
            st["dropped_by"], st["reason"] = "content", f"语言 {st['lang']} ≠ zh"
            dropped.append(it)
            continue
        if f.lang_mode == "en_only" and st["lang"] != "en":
            st["state"] = "lang_mismatch"
            st["dropped_by"], st["reason"] = "content", f"语言 {st['lang']} ≠ en"
            dropped.append(it)
            continue
        if f.require_entity and not st["mentions_entity"]:
            st["state"] = "no_entity"
            st["dropped_by"], st["reason"] = "content", "未提到目标实体"
            dropped.append(it)
            continue
        st["state"] = "valid"
        kept.append(it)
    return kept, {"min_chars": f.min_chars,
                  "drop_unverifiable": f.drop_unverifiable,
                  "lang_mode": f.lang_mode or "(仅打标)",
                  "require_entity": f.require_entity,
                  "n_unverifiable": sum(1 for it in items
                                        if ctx["state"][id(it)].get("state") == "unverifiable")}


def _layer_meta(items: list[dict], ctx: dict):
    """L3：元数据预过滤 —— URL 去重 + 单域名配额 + 标题党/广告词（用户指定判据）。"""
    f = WEB_PIPELINE
    kept, dropped, seen_url, per_host = [], [], set(), {}
    n_ad = 0
    for it in items:
        st = ctx["state"][id(it)]
        url = str(it.get("url") or "").split("#")[0].strip().lower()
        if url and url in seen_url:
            st["dropped_by"], st["reason"] = "meta", "URL 重复"
            dropped.append(it)
            continue
        hit = ad_like(f"{it.get('title') or ''} {item_text(it)[:400]}")
        if hit:
            n_ad += 1
            st["dropped_by"], st["reason"] = "meta", f"标题党/广告词命中「{hit}」"
            dropped.append(it)
            continue
        h = _host(str(it.get("url") or ""))
        if f.max_per_host > 0 and per_host.get(h, 0) >= f.max_per_host:
            st["dropped_by"], st["reason"] = "meta", f"单域名超过 {f.max_per_host} 条"
            dropped.append(it)
            continue
        seen_url.add(url)
        per_host[h] = per_host.get(h, 0) + 1
        kept.append(it)
    return kept, {"max_per_host": f.max_per_host, "n_hosts": len(per_host),
                  "n_ad_dropped": n_ad}


def _layer_embed(items: list[dict], ctx: dict):
    """L4：本地 embedding 相关性（无 API 成本）。查询=用户意图原话，条目=其正文。

    ⚠️ 实测踩坑（本层第一版）：``embed_docs`` 若返回的向量数**少于**输入条数，
    ``zip`` 会静默截断 → 多出来的条目既不进 kept 也不进 dropped（**无声丢证据**，
    与产品里那类"静默回退"同类）。此处显式校验长度，不一致就整层 fail-open。
    """
    f = WEB_PIPELINE
    from src.engine.embedder import Embedder
    # ⚠️ 实测踩坑（第二次）：`Embedder.embed_docs` 会**丢弃空文本**
    #   （embedder.py:155-159：`cleaned = [t for t in cleaned if t]`，只有全空时才补零向量），
    #   于是"混有空条目"的输入会返回**变短的**向量表 → 按序号 zip 会**错位**：
    #   把第 k 条正文的相似度安到第 k 条（其实是空的）条目上，且不报任何错。
    #   这里用占位正文保证**一一对应**（空条目的向量无意义，但不会污染别人的分数）。
    raw = [item_text(it)[:1200] for it in items]
    texts = [t if t else "(no content)" for t in raw]
    qv = Embedder().embed_query(ctx["query"] or "query")
    vecs = Embedder().embed_docs(texts) if texts else []
    if len(vecs) != len(items):
        logger.warning(f"[WebPipe] L4 向量数 {len(vecs)} ≠ 条目数 {len(items)}，"
                       f"整层 fail-open（不剔任何条目）")
        return items, {"embed_min": f.embed_min,
                       "error_len_mismatch": [len(items), len(vecs)]}
    kept, dropped = [], []
    for it, v, t in zip(items, vecs, raw):
        st = ctx["state"][id(it)]
        if not t:
            # 空条目不给相似度（避免用占位文本的向量冒充证据相关性）
            st["cos"] = None
            kept.append(it)
            continue
        st["cos"] = round(cosine(qv, v), 4)
        if st["cos"] < f.embed_min:
            st["dropped_by"] = "embed"
            st["reason"] = f"余弦 {st['cos']} < {f.embed_min}"
            dropped.append(it)
        else:
            kept.append(it)
    scores = sorted(s["cos"] for s in (ctx["state"][id(i)] for i in items)
                    if s.get("cos") is not None)
    q = lambda p: scores[min(len(scores) - 1, int(p * (len(scores) - 1)))] if scores else None
    return kept, {"embed_min": f.embed_min, "n_scored": len(scores),
                  "cos_min": q(0.0), "cos_p10": q(0.10), "cos_p50": q(0.50),
                  "cos_max": q(1.0)}


def _layer_rerank(items: list[dict], ctx: dict):
    """L6：cross-encoder 全文精排（对**抓到的正文**），并按分数阈值/池大小截断。"""
    f = WEB_PIPELINE
    if not items:
        return items, {}
    from src.engine.reranker import Reranker
    chunks = [{"ref_id": it.get("ref_id"), "text": item_text(it)[:2000] or str(it.get("title") or ""),
               "_it": it} for it in items]
    top_k = f.rerank_top_k or len(chunks)
    ranked = Reranker().rerank(ctx["query"], chunks, top_k=max(1, top_k))
    kept, dropped = [], []
    for c in ranked:
        it = c["_it"]
        st = ctx["state"][id(it)]
        st["rerank"] = round(float(c.get("rerank_score") or 0.0), 4)
        if f.rerank_min > 0 and st["rerank"] < f.rerank_min:
            st["dropped_by"], st["reason"] = "rerank", f"精排分 {st['rerank']} < {f.rerank_min}"
            dropped.append(it)
        else:
            kept.append(it)
    # 未进入 top_k 的条目（被截断）
    ranked_ids = {id(c["_it"]) for c in ranked}
    for it in items:
        if id(it) not in ranked_ids:
            st = ctx["state"][id(it)]
            st["dropped_by"], st["reason"] = "rerank", f"未进入 top_k={top_k}"
            dropped.append(it)
    return kept, {"top_k": top_k, "rerank_min": f.rerank_min}


async def _layer_adjudicate(items: list[dict], ctx: dict):
    """L5：复用生产裁决（fast 模型，异步）。无 ``adjudicate_fn`` 时跳过（只记 note）。"""
    fn: Optional[Callable] = ctx.get("adjudicate_fn")
    if fn is None or not items:
        return items, {"note": "未提供 adjudicate_fn（跳过）"}
    res = await fn(ctx["query"], ctx.get("main_items") or [], items)
    drops = set(str(x) for x in (res or {}).get("dropped_keys") or [])
    kept, dropped = [], []
    for it in items:
        st = ctx["state"][id(it)]
        key = str(it.get("ref_id") or it.get("url") or "")
        if key and key in drops:
            st["dropped_by"], st["reason"] = "adjudicate", "fast 模型判无关"
            dropped.append(it)
        else:
            kept.append(it)
    return kept, {"n_dropped": len(dropped),
                  "raw": {k: v for k, v in (res or {}).items() if k != "raw_text"}}


async def run_web_pipeline(items: list[dict], *, query: str,
                           main_items: Optional[list] = None,
                           adjudicate_fn: Optional[Callable] = None) -> tuple[list[dict], dict]:
    """跑六层管线。返回 ``(kept_items, report)``；关闭时原样返回。

    ``adjudicate_fn(intent, main_items, web_items) -> {"dropped_keys": [...]}``（异步）。
    """
    items = list(items or [])
    flags = WEB_PIPELINE
    if not flags.any_on():
        return items, {"enabled": False}

    ctx: dict[str, Any] = {
        "query": query or "",
        "terms": entity_terms(query or ""),
        "main_items": main_items or [],
        "adjudicate_fn": adjudicate_fn,
        "state": {id(it): {"ref_id": it.get("ref_id"), "url": it.get("url"),
                           "title": it.get("title")} for it in items},
    }
    report: dict = {"enabled": True, "layers_enabled": list(flags.layers),
                    "flags": flags.as_dict(), "n_in": len(items),
                    "per_layer": [], "unknown_layers": list(flags.unknown_layers),
                    "terms": ctx["terms"][:10]}
    cur = items
    for name in flags.layers:
        t0 = time.perf_counter()
        fn = {"fetch": _layer_fetch, "tiers": _layer_tiers, "content": _layer_content,
              "meta": _layer_meta, "embed": _layer_embed,
              "adjudicate": _layer_adjudicate, "rerank": _layer_rerank}[name]
        try:
            out = fn(cur, ctx)
            cur, note = await out if hasattr(out, "__await__") else out
            report["per_layer"].append({
                "layer": name, "n_in": len(cur) + 0, "n_out": len(cur),
                "ms": round((time.perf_counter() - t0) * 1000, 1), "note": note})
        except Exception as e:  # noqa: BLE001 —— fail-open，绝不阻断回答
            logger.warning(f"[WebPipe] 层 {name} 失败（fail-open，保持输入不变）: {e}")
            report["per_layer"].append({
                "layer": name, "error": f"{type(e).__name__}: {e}"[:200],
                "ms": round((time.perf_counter() - t0) * 1000, 1)})
    # n_in/n_out 必须按"进入该层时的条数"记，故重算一遍
    n = len(items)
    for rec in report["per_layer"]:
        if "n_out" in rec:
            rec["n_in"] = n
            n = rec["n_out"]
    report["n_out"] = len(cur)
    report["n_dropped"] = len(items) - len(cur)
    report["items"] = [ctx["state"][id(it)] for it in items]
    report["n_unverifiable"] = sum(1 for s in report["items"]
                                   if s.get("state") == "unverifiable")
    report["drop_reasons"] = {}
    for s in report["items"]:
        r = s.get("dropped_by")
        if r:
            report["drop_reasons"][r] = report["drop_reasons"].get(r, 0) + 1
    logger.info(f"[WebPipe] {report['n_in']} → {report['n_out']} 条"
                f"（layers={'+'.join(flags.layers)}，剔除原因 {report['drop_reasons']}）")
    return cur, report
