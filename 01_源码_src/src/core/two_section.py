# -*- coding: utf-8 -*-
"""Supervisor 两段输出的**解析 + 验证 + 流式过滤**（v10.14 · 阶段 2 · 计划 §六 新增 3/5/6）。

依《CitrusQA 输出架构重构 · 最终确认文档》：

* §六 新增 3：两段分隔符 `===== 联网小节 =====`
* §六 新增 5：两段代码验证（第一段无 `[Wn]`/URL；第二段无重复 URL）
* §六 新增 6：输出路径唯一化（**流式只第一段**，done 含两段）
* §八 8.2：违规 → **拒绝重试一次**；**不做清洗**（不清洗、不改写、不剥离）

**设计原则**：本模块**只判断与切分，绝不改写模型输出**。
上一轮"正则清洗"（`_to_search_query` / `strip_meta_commentary`）已按 §四 撤销；
这里的验证**只返回违规理由**，由调用方决定重试。

纯函数、零依赖、绝不抛异常（跑在答案出口路径上，坏掉不能把答案弄丢）。
"""
from __future__ import annotations

import re

__all__ = [
    "SEPARATOR", "split_two_sections", "validate_local_section",
    "validate_web_section", "StreamSectionFilter", "has_separator",
]

#: 两段分隔符（计划 §六 新增 3，必须逐字一致）
SEPARATOR = "===== 联网小节 ====="

#: 容错匹配：整行由 `=` 包裹、且含「联网小节」（允许空格/全角/多等号差异）。
#   为什么容错：模型可能输出 `==== 联网小节 ====` 或 `===== 联网小节 =====`；
#   容错**只影响切分位置**，不改写任何内容。
_SEP_RE = re.compile(r"^[ \t]*={2,}[ \t]*联网小节[ \t]*={2,}[ \t]*$", re.M)

#: 第一段违规判据（计划 §八 8.2）
_WN_RE = re.compile(r"\[W\s*\d+\]", re.IGNORECASE)
_URL_LABEL_RE = re.compile(r"URL\s*[:：]", re.IGNORECASE)
_HTTP_RE = re.compile(r"https?://", re.IGNORECASE)

#: 用于"与第一段高度重合"判定的**子句**切分。
#   ⚠️ 实测教训：首版只按句末标点切，遇到 `补充：<整句>` 这种"前缀 + 与第一段
#   完全相同的句子"就漏报了（实测 FAIL）。故：
#     · 同时按**句末**与**句中**标点切（含中文逗号/顿号）；
#     · 再对每段剥掉"标签式前缀"（`xxx：` / `xxx:`）后重试。
#   仍保持**保守**：只认"完整子串出现"，不做模糊相似度（误报会白白触发一次重试）。
_SENT_SPLIT_RE = re.compile(r"(?<=[。！？!?.;；，,、])\s*")
_LABEL_PREFIX_RE = re.compile(r"^[^：:]{0,24}[：:]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")


def _is_substantial(s: str) -> bool:
    """判定一个子句是否"足够实质"，值得作为重合证据。

    ⚠️ 实测教训：首版用固定 20 字符阈值，遇到 19 字符的中文子句就**漏报**了
    （`黄龙病由韧皮部杆菌属细菌引起 [1]，` = 19 字符）。改为**内容感知**：

    * 中文：CJK ≥ 8 字（中文信息密度高，8 字已是很具体的短语）
    * 英文：≥ 5 个词（避免 "the results show that" 这类常见短语误报）
    """
    t = str(s or "").strip()
    if len(t) < 12:
        return False
    if len(_CJK_RE.findall(t)) >= 8:
        return True
    return len(t.split()) >= 5


def _overlap_candidates(piece: str) -> list[str]:
    """给一个子句生成"可比较"的候选串（原文 + 剥标签前缀后的剩余）。"""
    t = piece.strip()
    out = [t]
    m = _LABEL_PREFIX_RE.match(t)
    if m:
        rest = t[m.end():].strip()
        if rest:
            out.append(rest)
    return out


def has_separator(text: str) -> bool:
    """文本里是否出现两段分隔符（容错）。"""
    try:
        return bool(_SEP_RE.search(str(text or "")))
    except Exception:                                          # noqa: BLE001
        return False


def split_two_sections(text: str) -> tuple[str, str]:
    """按分隔符切成 ``(第一段=本地答案, 第二段=联网小节)``。

    * 无分隔符 → ``(全文, "")``（**不报错、不改写**：向后兼容模型偶尔漏写分隔符）。
    * 多个分隔符 → 以**第一个**为界（其余留在第二段，由验证报"重复"）。
    """
    s = str(text or "")
    if not s:
        return "", ""
    m = _SEP_RE.search(s)
    if not m:
        return s, ""
    return s[:m.start()].rstrip(), s[m.end():].strip()


def validate_local_section(local: str) -> list[str]:
    """第一段验证（计划 §八 8.2）：含 `[Wn]` / `URL:` / `http` → 违规。

    返回违规理由列表；空列表 = 通过。**只判断，不清洗。**
    """
    out: list[str] = []
    s = str(local or "")
    if not s.strip():
        out.append("第一段为空")
        return out
    m = _WN_RE.search(s)
    if m:
        out.append(f"第一段含联网编号 {m.group(0)!r}（引用池隔离：联网条目不进 [n]/[Wn]）")
    if _URL_LABEL_RE.search(s):
        out.append("第一段含 `URL:` 标签（联网痕迹）")
    m2 = _HTTP_RE.search(s)
    if m2:
        out.append(f"第一段含明文链接 {m2.group(0)!r}（应折叠为 🔗 并只在联网小节）")
    return out


def validate_web_section(web: str, local: str = "") -> list[str]:
    """第二段验证（计划 §八 8.2）：重复 URL / 与第一段高度重合的句子 → 违规。

    "高度重合"判据（保守）：第二段某个 ≥20 字的句子，**完整出现**在第一段里
    ⇒ 视为重复。用"完整子串"而非模糊相似度：**宁可漏报也不误报**（误报会白白
    触发一次重试，浪费 token 且可能让模型改坏内容）。
    """
    out: list[str] = []
    s = str(web or "")
    if not s.strip():
        return out                      # 第二段可为空（计划 §六 新增 4 的边界情形）
    # ① 重复 URL（同一 URL 出现 >1 次）
    urls = re.findall(r"https?://[^\s)\]】]+", s)
    seen: dict[str, int] = {}
    for u in urls:
        k = u.rstrip(".,;:!?）)】]").lower()
        seen[k] = seen.get(k, 0) + 1
    dup = [u for u, c in seen.items() if c > 1]
    if dup:
        out.append(f"第二段重复 URL {len(dup)} 个（示例 {dup[0][:60]!r}）")
    # ② 与第一段高度重合的句子
    if str(local or "").strip():
        loc = str(local)
        hits: list[str] = []
        for piece in _SENT_SPLIT_RE.split(s):
            for cand in _overlap_candidates(piece):
                if _is_substantial(cand) and cand in loc:
                    hits.append(cand)
                    break
        if hits:
            out.append(f"第二段有 {len(hits)} 句与第一段完全重合"
                       f"（示例 {hits[0][:50]!r}）")
    return out


class StreamSectionFilter:
    """流式过滤器：**只放行第一段**（计划 §六 新增 6「输出路径唯一化」）。

    用法：``on_text=StreamSectionFilter(emit_text)``。

    为什么需要它：Supervisor 现在**一次 LLM 调用输出两段**，而该调用的 token 是
    逐块 `emit_text` 上屏的。若不拦，第二段（联网小节）会**先在流式里出现一次**、
    再由 `done.answer` 出现一次 ⇒ 正是用户实测的"重复输出"。
    本过滤器在流式侧**截断**于分隔符：一旦看到分隔符（或其前缀），**此后不再放行**。

    **不做清洗**：放行的内容**逐字**就是模型输出，只控制"发不发"。
    """

    def __init__(self, sink, separator: str = SEPARATOR) -> None:
        self._sink = sink
        self._sep = separator
        self._buf = ""              # 尚未判定归属的尾部缓冲
        self._done = False          # 已见到分隔符 ⇒ 停止放行
        self._held = 0              # 因缓冲/截断而未放行的字符数（观测用）

    def __call__(self, content: str) -> None:
        try:
            if self._done or not content:
                if content:
                    self._held += len(content)
                return
            self._buf += str(content)
            # 判断缓冲里是否已出现分隔符（或其"可能成为分隔符"的前缀）
            idx = self._find_sep(self._buf)
            if idx is not None:
                head = self._buf[:idx]
                if head:
                    self._sink(head)
                self._held += len(self._buf) - len(head)
                self._buf = ""
                self._done = True
                return
            # 未出现分隔符：把**确定安全**的部分放行，尾部可能是不完整分隔符 ⇒ 留缓冲
            keep = self._partial_tail_len(self._buf)
            if len(self._buf) > keep:
                emit = self._buf[:len(self._buf) - keep]
                self._buf = self._buf[len(self._buf) - keep:]
                if emit:
                    self._sink(emit)
        except Exception:                                      # noqa: BLE001
            # 过滤器坏掉绝不能吞掉答案：直接把本块放行（fail-open）
            try:
                if content and not self._done:
                    self._sink(content)
            except Exception:                                  # noqa: BLE001
                pass

    def flush(self) -> None:
        """收尾：把仍留在缓冲里的第一段内容放行（未见到分隔符时）。"""
        try:
            if not self._done and self._buf:
                self._sink(self._buf)
            self._buf = ""
        except Exception:                                      # noqa: BLE001
            pass

    # ── 内部 ────────────────────────────────────────────────────────────
    def _find_sep(self, s: str) -> int | None:
        """返回分隔符在 ``s`` 中的起始下标（容错匹配整行）。"""
        m = _SEP_RE.search(s)
        if m:
            return m.start()
        # 无完整分隔符：找 `联网小节` 所在行的行首（模型可能还没写完 `=`）
        for m2 in re.finditer(r"联网小节", s):
            ls = s.rfind("\n", 0, m2.start()) + 1
            line = s[ls:m2.end()]
            if re.fullmatch(r"[ \t]*={2,}[ \t]*联网小节[ \t]*", line):
                return ls
        return None

    def _partial_tail_len(self, s: str) -> int:
        """尾部有多少字符**可能是分隔符的开头** ⇒ 必须留在缓冲里等待。"""
        # 取最后一行；若它看起来像"分隔符的开头"（全是 = 和空格，或含"联网"前缀），
        # 就保留整行；否则只保留很短的尾巴（防跨块分隔符被劈开）。
        nl = s.rfind("\n")
        last = s[nl + 1:] if nl >= 0 else s
        if re.fullmatch(r"[ \t=]*", last) and last.strip():
            return len(last)
        if "联网" in last or "小节" in last:
            return len(last)
        return min(len(last), 12)      # 允许分隔符跨 SSE 块：留 12 字符尾巴
