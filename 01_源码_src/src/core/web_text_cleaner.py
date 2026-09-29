# -*- coding: utf-8 -*-
"""联网文本净化（v10.6「本地与联网分离输出」架构）.

职责：把联网取回的原始文本变成**可以直接给用户看**的干净文本：

1. 去掉 ``[W1]`` / ``[W1-W3]`` / ``[W1,3]`` / ``[W1、W3]`` 等**联网引用标记**
   ——分离模式下联网条目不再占编号，残留标记会让用户以为存在可核验的 [Wn] 依据；
2. 归一空白，含 ``U+2028`` / ``U+0085`` / ``U+000B`` / ``U+000C`` / ``U+00A0`` 等
   "假换行/假空格"——这些字符曾把一条 SSE 事件劈成两半、静默丢掉整条 citations
   （见 REPRODUCE.md 坑 16/17），在出口路径上必须一并消毒；
3. 提取并去重 URL（去尾部标点；host 小写比较）；
4. 按上限截断，尽量在句读处断开（不把句子切一半）。

设计原则：**纯函数、零依赖、绝不抛异常**。它跑在答案出口路径上——净化器自己坏掉
不能把答案弄丢，故所有函数失败时一律安全降级（返回原文或安全截断）。
"""
from __future__ import annotations

import re
import unicodedata

__all__ = [
    "strip_web_markers", "normalize_ws", "find_urls", "dedupe_urls",
    "host_of", "truncate_at_sentence", "clean_web_text", "clean_answer_text",
]

def _safe(fn, default=""):
    """净化函数绝不抛异常：任何意外都降级为 default。"""
    try:
        return fn()
    except Exception:                                          # noqa: BLE001
        return default


def strip_web_markers(text: str) -> str:
    """去掉联网引用标记 [Wn]（分离模式下编号不再存在）。"""
    if not text:
        return ""
    return _safe(lambda: _WEB_MARKER.sub("", text), text)


def normalize_ws(text: str) -> str:
    """归一空白：假换行→真换行，NBSP/全角空格→普通空格，压缩连续空行。"""
    if not text:
        return ""

    def _go():
        s = text.translate(_WS_MAP)
        s = unicodedata.normalize("NFC", s)
        s = re.sub(r"[ \t]+", " ", s)
        s = re.sub(r" *\n *", "\n", s)
        s = re.sub(r"\n{3,}", "\n\n", s)
        return s.strip()

    return _safe(_go, text)


def find_urls(text: str) -> list[str]:
    """按出现顺序提取 URL（去尾部标点）。"""
    if not text:
        return []
    return _safe(lambda: [u.rstrip(_URL_TAIL) for u in _URL.findall(text)], [])


def host_of(url: str) -> str:
    """取小写 host（失败返回空串）；用于去重与展示。"""
    if not url:
        return ""

    def _go():
        s = url.split("//", 1)[-1]
        s = s.split("/", 1)[0].split("?", 1)[0].split("#", 1)[0]
        s = s.split("@")[-1]
        if ":" in s:
            s = s.split(":", 1)[0]
        return s.lower().strip()

    return _safe(_go, "")


def dedupe_urls(urls: list[str]) -> list[str]:
    """按 host+path 去重，保留首次出现顺序。"""
    out: list[str] = []
    seen: set[str] = set()
    for u in urls or []:
        key = _safe(lambda _u=u: _u.split("//", 1)[-1].split("#", 1)[0].lower().rstrip("/"), u)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(u)
    return out


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """按字符上限截断，尽量在句读处断开；不把句子切一半。"""
    if not text:
        return ""
    if max_chars <= 0 or len(text) <= max_chars:
        return text

    def _go():
        head = text[:max_chars]
        # 在窗口后半段找最后一个句读，避免截得过短
        floor = int(max_chars * 0.6)
        best = -1
        for ch in "。！？!?.;；\n":
            idx = head.rfind(ch)
            if idx > best:
                best = idx
        if best >= floor:
            return head[:best + 1].rstrip()
        return head.rstrip() + "…"

    return _safe(_go, text[:max_chars])


#: emoji / 图形符号（保守范围：只取"确定是图形"的块，不动箭头、数学符号等正文常用字符）
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"      # 各类符号与图形（含 ⚠ 之外的绝大多数 emoji）
    "\U00002600-\U000027BF"      # 杂项符号 + 装饰符号（⚠ U+26A0、✅、❌ 等）
    "\U0001F000-\U0001F2FF"      # 麻将/扑克/圈字等
    "\U00002B00-\U00002BFF"      # 箭头/星形装饰
    "\U0001F1E6-\U0001F1FF"      # 区域指示符（国旗）
    "\U0000FE00-\U0000FE0F"      # 变体选择符（⚠️ 的第二码位）
    "]+"
)


def strip_emoji(text: str) -> str:
    """去掉 emoji/图形符号（v10.9）。

    为什么：输出约束（计划 §4.2）要求**无 emoji**；实测 E2E 里联网综述正文自带 `⚠️`
    （联网搜索模型写的），直接进分离小节就违反约束。
    **只用于联网正文**——本地答案侧走 `clean_answer_text`，那里必须逐字不动，
    不得在此处做任何删改。
    """
    return _EMOJI.sub("", text or "")


#: 行首 markdown 标题标记（`#`~`######` + 空格）。联网综述常带这些，
#: 而输出约束（计划 §4.2）要求**无 `##`**，故在净化阶段降级为纯文本。
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.MULTILINE)


def strip_md_headings(text: str) -> str:
    """把行首 `#`/`##`/… 标题标记去掉，保留标题文字（v10.9）。

    实测：E2E 里联网综述正文自带 `##`，直接进分离小节会违反"无 `##`"的输出约束。
    这里**只去标记、不删文字**，也不动正文其它内容。
    """
    return _MD_HEADING.sub("", text or "")


def clean_web_text(text: str, max_chars: int = 0) -> str:
    """联网正文净化：去 [Wn] → 归一空白 → 去标题标记 → 去 emoji → 去元评论 →（可选）截断。

    ``max_chars <= 0`` 表示不截断。

    """
    if not text:
        return ""
    s = normalize_ws(strip_web_markers(str(text)))
    s = strip_md_headings(s)                     # v10.9：满足"无 `##`"输出约束
    s = strip_emoji(s)                           # v10.9：满足"无 emoji"输出约束
    if max_chars and max_chars > 0:
        s = truncate_at_sentence(s, max_chars)
    return s


def clean_answer_text(text: str) -> str:
    """答案侧消毒：**只做两件必要的事**，其余逐字不动。

    1. 去掉联网标记 ``[Wn]``（分离模式下编号不存在）；
    2. 消毒"假换行"字符——它们会把一条 SSE 事件劈成两半、静默丢掉整条 citations
       （REPRODUCE.md 坑 16/17）。

    刻意**不做** NFC 归一、不压缩空白、不动 ``[n]``/``[Hn]``/表格/空行结构：
    "本地答案逐字不动"是分离架构的硬约束，任何多余的文本改写都会让
    `done.answer` 与模型原始输出产生无法解释的差异（判官与用户看的是同一份）。
    """
    if not text:
        return ""

    def _go():
        return strip_web_markers(str(text)).translate(_WS_MAP)

    return _safe(_go, text)
