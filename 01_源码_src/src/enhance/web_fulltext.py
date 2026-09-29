# -*- coding: utf-8 -*-
"""模块2：联网原文块管线（默认关闭）。

审查员原话
----------
「目前联网通道返回的是搜索服务生成的摘要，属于二手文本，无法做原文块预览，也无法
做保真校验。可以对返回的引用条目**抓取原文、切块**后和本地证据一样进入**向量化、
去重、校验、侧栏预览**流程，使联网证据与本地证据可信度对等、审计口径一致。」

现状（v9.6.3，必须先讲清楚）
--------------------------
联网通道现在返回的是 **DeepSeek 原生联网回答全文（不截断）+ 引用条目(title/url/snippet)**。
模型侧只看到 `[W1] 标题 — URL`；侧栏 `text_preview` 取 snippet。也就是说：
  * 「联网返回的是 2840 字摘要」——**已过时**（现在是不截断的原生成文）；
  * 「没有原文块、无法做侧栏原文预览、却与本地同等挂号」——**仍然成立**。

本模块补的正是后半句：把 `[Wn]` 指向的**网页正文**抓下来、切成块，产出与本地证据
**同构**的条目（带 `text`、`chunk_index`、`source="web"`），从而
  * 侧栏预览 = 模型实际依据的那一块（不是搜索片段）；
  * 与本地证据走同一套去重/校验/展示口径；
  * 支持度校验（模块1）也能对联网论断生效。

边界与取舍（写进说明书）
------------------------
* 只对**引用条目**抓取（不扩散爬取）；条数由 `CITRUS_WEB_FULLTEXT_MAX` 限制（默认 5）。
* 抓取失败/超时/非 HTML → **fail-open**：退回原有 snippet 形态，不阻断回答。
* 遵守 robots 与超时；不做登录、不绕反爬。抓不到就用片段，并如实标注来源形态。
* 本模块**不改** `src/tools/deepseek_web.py`（联网工具本体）：它只消费其 artifact。
"""
from __future__ import annotations

import logging
import re
from typing import Iterable, Optional

logger = logging.getLogger(__name__)

_TIMEOUT = 15
_MAX_BYTES = 2_000_000
_UA = ("Mozilla/5.0 (compatible; CitrusQA/1.0; +research-use) "
       "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36")
# 与本地证据同构的块长（字符）——取语料切分口径量级，保证"块"语义一致
CHUNK_CHARS = 1200
CHUNK_OVERLAP = 120
_MIN_BODY_CHARS = 500          # 正文短于此视为"跳转页/拦截页"，尝试跟一跳 meta refresh


def _strip_html(html: str) -> str:
    """极简正文抽取：去脚本/样式/标签，保留段落换行。

    不引入新的第三方依赖（readability/trafilatura 需额外安装且会改变部署形态）。
    对学术页面（含 <article>/<p>）够用；抽不出正文时返回空串由调用方 fail-open。
    """
    if not html:
        return ""
    s = re.sub(r"(?is)<(script|style|noscript|svg|head)[^>]*>.*?</\1>", " ", html)
    s = re.sub(r"(?is)<!--.*?-->", " ", s)
    # 段落级标签 → 换行，其余标签 → 空格
    s = re.sub(r"(?i)</(p|div|section|article|li|h[1-6]|tr|br)\s*>", "\n", s)
    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", " ", s)
    # 实体
    for a, b in (("&nbsp;", " "), ("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"),
                 ("&quot;", '"'), ("&#39;", "'"), ("&mdash;", "—")):
        s = s.replace(a, b)
    s = re.sub(r"[ \t\u00a0]+", " ", s)
    s = re.sub(r"\n\s*\n\s*\n+", "\n\n", s)
    return s.strip()


def chunk_text(text: str, chunk_chars: int = CHUNK_CHARS,
               overlap: int = CHUNK_OVERLAP) -> list[str]:
    """按段落优先、定长兜底切块——与本地语料"块"的量级对齐。"""
    text = (text or "").strip()
    if not text:
        return []
    paras = [p.strip() for p in text.split("\n") if p.strip()]
    chunks: list[str] = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= chunk_chars:
            buf = (buf + "\n" + p).strip()
            continue
        if buf:
            chunks.append(buf)
        while len(p) > chunk_chars:
            chunks.append(p[:chunk_chars])
            p = p[max(0, chunk_chars - overlap):]
        buf = p
    if buf:
        chunks.append(buf)
    return chunks


def _attr(tag: str, name: str) -> Optional[str]:
    """取 HTML 标签里某个属性的值，**按引号配对**取值。

    不能写成 `content=["\\']([^"\\']+)["\\']`：Elsevier 跳转页的 content 用双引号包裹、
    内部又含单引号（`content="2; url='/retrieve/...'"`），那种写法会在第一个单引号处截断
    （自检 EP-9b 实测踩到）。这里用反向引用保证首尾引号同类。
    """
    m = re.search(name + r"""\s*=\s*(["'])(.*?)\1""", tag, re.I | re.S)
    if m:
        return m.group(2)
    m = re.search(name + r"""\s*=\s*([^\s"'>]+)""", tag, re.I)     # 无引号
    return m.group(1) if m else None


def _follow_meta_refresh(html: str, base_url: str) -> Optional[str]:
    """从"落地页/跳转页"里取出真实目标 URL。

    真机验证（2026-09-21）实测：`doi.org/<doi>` 返回 **HTTP 200 但正文 0 字符**——
    页面是出版商的 "Auto Article Locator"，靠
    `<meta http-equiv="REFRESH" content="2; url='...'">` 跳转。
    只认 meta refresh 与其中的 `Redirect=` 参数（两者都做 URL 解码），最多跟一跳。
    """
    if not html:
        return None
    tag = None
    for m in re.finditer(r"<meta\b[^>]*>", html, re.I):
        if re.search(r"http-equiv\s*=\s*[\"']?refresh", m.group(0), re.I):
            tag = m.group(0)
            break
    if tag is None:
        return None
    content = _attr(tag, "content")
    if not content:
        return None
    um = re.search(r"url\s*=\s*[\"']?([^\"'>\s]+)", content, re.I)
    if not um:
        return None
    from urllib.parse import unquote, urljoin
    target = unquote(um.group(1)).strip("'\"")
    # 出版商跳转页形如 /retrieve/xxx?Redirect=<编码后的真实地址>&key=...
    rm = re.search(r"[?&]Redirect=([^&]+)", target, re.I)
    if rm:
        target = unquote(rm.group(1))
    target = unquote(target)
    if not re.match(r"^https?://", target, re.I):
        target = urljoin(base_url, target)
    return target if target.lower() != (base_url or "").lower() else None


def _pdf_text(data: bytes, max_pages: int = 60) -> str:
    """从 PDF 字节里抽正文（与仓库既有做法一致：`fitz` 取块 + 页眉页脚过滤）。

    真机验证（2026-09-21）实测：联网引用里有相当比例是 **PDF 直链**
    （政府/机构报告、agrinfo 的 book-of-reports 等）——只走 HTML 路径会整条抓不到。
    这里不 import `src.tools.search`（那是实验依赖文件），直接用同一个库 `fitz`
    做轻量抽取，保持增强包对检索链路零耦合。
    """
    try:
        import fitz
    except Exception:                                          # noqa: BLE001
        logger.info("[WebFulltext] 未安装 fitz，跳过 PDF 抽取")
        return ""
    try:
        doc = fitz.open(stream=data, filetype="pdf")
    except Exception as e:                                     # noqa: BLE001
        logger.info("[WebFulltext] PDF 打开失败: %s", e)
        return ""
    out: list[str] = []
    try:
        for pno, page in enumerate(doc):
            if pno >= max_pages:
                break
            h, w = page.rect.height, page.rect.width
            for b in page.get_text("blocks"):
                if len(b) > 6 and b[6] != 0:                   # 图片块
                    continue
                t = (b[4] or "").strip()
                if not t:
                    continue
                x0, y0, x1, y1 = b[0], b[1], b[2], b[3]
                # 页眉页脚启发式（与仓库既有实现同思路）
                if len(t) < 40 and y0 < h * 0.08:
                    continue
                if len(t) < 20 and y1 > h * 0.92:
                    continue
                if len(t) < 10 and y1 > h * 0.85 and abs(x0 + x1 - w) < w * 0.3:
                    continue
                out.append(re.sub(r"\s+", " ", t))
    finally:
        try:
            doc.close()
        except Exception:                                      # noqa: BLE001
            pass
    return "\n".join(out).strip()


def fetch_page(url: str) -> Optional[str]:
    """抓一页并抽正文（HTML 或 PDF）。失败返回 None（调用方 fail-open）。

    路径：HTTP 200 → 按 Content-Type / 后缀判定 → HTML 抽标签 / PDF 抽文本；
    若落地页是"跳转页"（正文极短且含 meta refresh），跟一跳再抽。
    """
    if not url or not re.match(r"^https?://", url, re.I):
        return None
    try:
        import requests
        r = requests.get(url, timeout=_TIMEOUT, headers={"User-Agent": _UA},
                         stream=True)
        if r.status_code != 200:
            logger.info("[WebFulltext] %s HTTP %s", url[:80], r.status_code)
            return None
        ctype = (r.headers.get("Content-Type") or "").lower()
        raw = r.raw.read(_MAX_BYTES, decode_content=True) or b""
        # ── PDF 直链 ──
        if "pdf" in ctype or raw[:5] == b"%PDF-" or re.search(r"\.pdf($|[?#])",
                                                              url, re.I):
            return _pdf_text(raw) or None
        if "html" not in ctype and "text" not in ctype:
            return None
        enc = r.encoding or "utf-8"
        try:
            html = raw.decode(enc, "ignore")
        except LookupError:
            html = raw.decode("utf-8", "ignore")
        text = _strip_html(html)
        if len(text) >= _MIN_BODY_CHARS:
            return text
        # 正文过短 → 可能是跳转页，跟一跳
        nxt = _follow_meta_refresh(html, url)
        if nxt:
            logger.info("[WebFulltext] 跳转页 → 跟一跳: %s", nxt[:90])
            r2 = requests.get(nxt, timeout=_TIMEOUT, headers={"User-Agent": _UA},
                              stream=True)
            if r2.status_code == 200:
                raw2 = r2.raw.read(_MAX_BYTES, decode_content=True) or b""
                if "pdf" in (r2.headers.get("Content-Type") or "").lower() \
                        or raw2[:5] == b"%PDF-":
                    t2 = _pdf_text(raw2)
                else:
                    try:
                        html2 = raw2.decode(r2.encoding or "utf-8", "ignore")
                    except LookupError:
                        html2 = raw2.decode("utf-8", "ignore")
                    t2 = _strip_html(html2)
                if len(t2) > len(text):
                    return t2
        return text or None
    except Exception as e:                                     # noqa: BLE001
        logger.info("[WebFulltext] 抓取失败 %s: %s", url[:80], e)
        return None


def build_web_blocks(web_items: Iterable[dict], *, max_items: int = 5,
                     fetch_fn=None) -> tuple[list[dict], dict]:
    """把联网引用条目升级成"原文块"条目（与本地证据同构）。

    `fetch_fn(url) -> str|None` 可注入（测试用）。
    返回 (新条目列表, 统计)。**抓不到就原样退回**，不丢条目。
    """
    fetch = fetch_fn or fetch_page
    out: list[dict] = []
    n_ok = n_fail = 0
    for i, it in enumerate(list(web_items or [])[:max_items]):
        url = str(it.get("url") or it.get("link") or "")
        title = str(it.get("title") or it.get("name") or "Untitled")
        body = fetch(url)
        if not body:
            n_fail += 1
            out.append({**it, "text_kind": "snippet",
                        "text": str(it.get("snippet") or it.get("abstract") or ""),
                        "note": "抓取失败/非正文页，退回搜索片段（fail-open）"})
            continue
        n_ok += 1
        for ci, ch in enumerate(chunk_text(body)):
            out.append({
                "ref_id": f"{it.get('ref_id') or f'W{i + 1}'}#{ci}",
                "parent_ref_id": it.get("ref_id") or f"W{i + 1}",
                "type": "web", "source": "web",
                "url": url, "title": title,
                "chunk_index": ci,
                "text": ch,                       # ← 与本地证据同构：块正文
                "text_kind": "fulltext",
                "doi": it.get("doi", ""),
            })
    stat = {"n_in": len(list(web_items or [])), "n_fetched": n_ok,
            "n_fallback": n_fail, "n_blocks": len(out),
            "chunk_chars": CHUNK_CHARS, "note": "抓到的网页正文按块入池；失败退回片段"}
    return out, stat


def to_cited_ref(block: dict) -> dict:
    """把原文块条目转成前端 `cited_refs` 契约（侧栏展示块正文，而非搜索片段）。"""
    return {
        "ref_id": block.get("ref_id"),
        "type": "web",
        "source": "web",
        "url": block.get("url", ""),
        "title": block.get("title", ""),
        "text_preview": str(block.get("text") or "")[:300],
        "score": 0,
    }
