"""回答审计（架构级约束）v2 —— 治 P1 意图偏差 / P2 引用失真 / P3 时效缺口 / 口径混淆。

v2 修订（基于 beta pack 110 条真实请求回放）：
  - CALIBER_MIX 改「行级引用映射 + 子句级判定」：引用常挂在行尾，按句切分会丢；
    免责词只在违规子句的 ±1 子句窗口内才豁免（防长句尾部免责词遮蔽整句）。
  - WEAK_NUMBER 只认「统计模式内」的数字（百分数 / 每X株 等），并先剥离引用标记，
    避免把列表序号、引用编号 [7] 当成数字（原版 124 条噪声 → 收窄）。
  - RECENCY 只认「强时效词」（最新/20xx年/行情/产量/市场/政策/新品种…），
    裸「目前/最近」不再触发（原版 165 条噪声 → 收窄）。

设计原则：**架构级约束，不是提示词约束**。
  Stage A 代码硬门（本模块，确定性，零 LLM 成本，先跑）
  Stage B LLM 回看（review_with_llm，fast 模型 rubric，输出结构化 verdict）
  行动策略：critical -> 1 次修复循环；minor -> 标注 + ledger，不重生成。
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

# ================= 统计口径词典（词族，可由调用方覆盖扩展） =================
CALIBER_FAMILIES: dict = {
    "seedling":  ["实生", "seedling", "珠心胚", "nucellar"],
    "hybrid":    ["杂交", "杂种", "hybrid", "cross", "杂合"],
    "mutation":  ["芽变", "突变", "mutation", "sport"],
    "graft":     ["嫁接", "graft"],
    "polyploid": ["多倍体", "倍性", "polyploid", "ploid"],
}

# 统计断言（子句级）
STAT_RE = re.compile(
    r"\d+(?:\.\d+)?\s*%|获得率|占比|比例|比率|平均|每\s*[\d,，.]+\s*(?:株|粒|个)"
    r"|\brate\b|\bratio\b", re.IGNORECASE)

# 统计数字提取（只取统计模式内的数字；先剥离引用标记）
STAT_NUM_RES = (
    re.compile(r"(\d+(?:\.\d+)?)\s*%"),
    re.compile(r"每\s*([\d,，.]+)\s*(?:株|粒|个)"),
    re.compile(r"(?:获得率|占比|比例|比率|平均)[^\d，。；;]{0,10}(\d+(?:\.\d+)?)"),
)

# 口径免责声明（子句窗口级）
DISCLAIMER_RE = re.compile(
    r"不是|而非|并非|不等于|≠|严格区分|口径|不能直接|未在文献|未给出|未拆分|"
    r"仅代表|单独机构|非文献原文|推导|换算|标注|来自不同|并不统一|需与.{0,12}区分")

# 强时效词（裸「目前/近年」不算）
RECENCY_STRONG_RE = re.compile(
    r"最新|20[2-3]\d\s*年|行情|走势|产量|种植面积|价格|市场|政策|新品种|新进展",
    re.IGNORECASE)
GAP_DECL_RE = re.compile(
    r"未检索到|无最新|缺口|无法|未能|未见|未获得|不可用|仅.{0,6}本地|本地文献")

CIT_RE = re.compile(r"\[(\d{1,3}|[WH]\d{1,3})\]", re.IGNORECASE)
CIT_STRIP_RE = re.compile(r"\[(?:\d{1,3}|[WH]\d{1,3})\]", re.IGNORECASE)
CLAUSE_SPLIT_RE = re.compile(r"[，,；;。：:|]")


# ======================== 数据结构 ========================
@dataclass
class Issue:
    code: str
    severity: str      # "critical" | "minor"
    sentence: str
    detail: str


@dataclass
class AuditReport:
    severity: str = "ok"
    issues: list = field(default_factory=list)
    stats: dict = field(default_factory=dict)

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)
        if issue.severity == "critical":
            self.severity = "critical"
        elif issue.severity == "minor" and self.severity == "ok":
            self.severity = "minor"


# ======================== 工具 ========================
def _families(text: str, families: dict) -> set:
    low = (text or "").lower()
    return {fam for fam, words in families.items()
            if any(w.lower() in low for w in words)}


def _lines(answer: str):
    for line in (answer or "").splitlines():
        if line.strip():
            yield line


def _clauses(line: str) -> list:
    return [c.strip() for c in CLAUSE_SPLIT_RE.split(line) if c.strip()]


def _norm_num(s: str) -> str:
    return (s or "").replace(",", "").replace("，", "")


def _stat_numbers(text: str) -> list:
    """只提取统计模式内的数字（剥离引用标记后）。"""
    t = CIT_STRIP_RE.sub(" ", text or "")
    out = []
    for rx in STAT_NUM_RES:
        for m in rx.finditer(t):
            for g in m.groups():
                if g:
                    out.append(_norm_num(g))
    return [x for x in out if x]


# ==================== Stage A：代码硬门 ====================
def check_citation_ids(answer: str, valid_ref_ids: set) -> list:
    """A1：引用编号必须在证据池内（越界 = 捏造引用，critical）。
    注意：集成时应传入【完整证据池】（cited+uncited+historical），
    而不是仅 cited，否则该检查恒不触发。"""
    issues = []
    valid = {str(v).upper() for v in valid_ref_ids}
    for m in CIT_RE.finditer(answer or ""):
        rid = m.group(1).upper()
        if valid and rid not in valid:
            issues.append(Issue("BAD_CITATION", "critical", m.group(0),
                                f"引用 [{rid}] 不在证据池 {sorted(valid)[:12]}"))
    return issues


def check_caliber(answer: str, cited_refs: list, query: str,
                  families: Optional[dict] = None) -> list:
    """A2：口径一致性（词表 + 引用片段核对），行级映射 / 子句级判定 v2。

    CALIBER_MIX：子句含统计断言 + ≥2 口径词族（含问题口径 F_q），
    且违规子句 ±1 邻域内无口径声明 → minor（LLM 回看可升级 critical）。
    WEAK_NUMBER：统计模式内数字在其所属行的引用片段中找不到 → minor。
    """
    fams = families or CALIBER_FAMILIES
    q_fams = _families(query or "", fams)

    ref_low: dict = {}
    for r in cited_refs or []:
        rid = str(r.get("ref_id", "")).upper()
        snippet = str(r.get("text_preview", "")) + " " + str(r.get("title", ""))
        ref_low[rid] = _norm_num(snippet)

    # 数字核对护栏：引用片段过短（导出包 preview 截断）时禁用 WEAK_NUMBER，
    # 避免把"片段截断"误报为"引用错位"；口径词族判定不受此影响。
    _max_prev = max((len(str(r.get("text_preview") or "")) for r in (cited_refs or [])), default=0)
    _num_check_ok = _max_prev >= 300

    issues: list = []
    for line in _lines(answer):
        ids = [m.group(1).upper() for m in CIT_RE.finditer(line)]
        if not ids:
            continue
        clauses = _clauses(line)
        for ci in range(len(clauses)):
            window = " ".join(clauses[max(0, ci - 1): ci + 2])
            if not STAT_RE.search(window) or not re.search(r"\d", window):
                continue
            w_fams = _families(window, fams)
            if q_fams and (w_fams & q_fams) and len(w_fams) >= 2:
                if DISCLAIMER_RE.search(window):
                    continue
                ratio = bool(re.search(r"\d+(?:\.\d+)?\s*%|每\s*[\d,，.]+\s*(?:株|粒|个)", window))
                issues.append(Issue(
                    "CALIBER_MIX", "critical" if ratio else "minor",
                    clauses[ci][:120] if ci < len(clauses) else window[:120],
                    f"±1 子句窗口口径混用 {sorted(w_fams)}（含问题口径 {sorted(w_fams & q_fams)}），"
                    f"窗口内无口径声明——数字归属需复核（LLM 回看确认）"))
                break  # 每行最多报 1 处（同根因去噪；细粒度判定交 Stage B）
        nums = _stat_numbers(line)
        # 仅当本行所有被引条目的片段都足够长（非截断）时才做数字核对
        if nums and _num_check_ok and all(len(ref_low.get(i, "")) >= 300 for i in ids):
            support = any(n in ref_low.get(i, "") for i in ids for n in nums)
            if not support:
                issues.append(Issue(
                    "WEAK_NUMBER", "minor", line[:120],
                    f"统计数字 {nums[:4]} 未出现在引用 {ids} 的证据片段中"))
    return issues


def check_recency(answer: str, recency_required: bool, web_available: bool) -> list:
    """A3：强时效断言必须挂 [Wn]，或行内显式声明缺口。"""
    if not recency_required:
        return []
    issues = []
    for line in _lines(answer):
        if not RECENCY_STRONG_RE.search(line):
            continue
        has_w = bool(re.search(r"\[W\d{1,3}\]", line, re.IGNORECASE))
        has_gap = bool(GAP_DECL_RE.search(line))
        if not has_w and not has_gap:
            issues.append(Issue(
                "RECENCY_NO_WEB", "minor", line[:120],
                "含强时效词但无 [Wn] 引用也无缺口声明"
                + ("" if web_available else "（本轮联网未开启/无 [Wn]，应声明缺口）")))
    return issues


def run_code_audit(answer: str, *, cited_refs: list, valid_ref_ids: set,
                   query: str, recency_required: bool = False,
                   web_available: bool = False) -> "AuditReport":
    """Stage A 汇总入口。severity: ok / minor / critical。"""
    rep = AuditReport()
    for i in check_citation_ids(answer, valid_ref_ids):
        rep.add(i)
    for i in check_caliber(answer, cited_refs, query):
        rep.add(i)
    for i in check_recency(answer, recency_required, web_available):
        rep.add(i)
    seen, uniq = set(), []
    for i in rep.issues:
        k = (i.code, i.sentence.strip()[:80])
        if k in seen:
            continue
        seen.add(k)
        uniq.append(i)
    rep.issues = uniq
    rep.severity = "ok"
    for i in uniq:
        if i.severity == "critical":
            rep.severity = "critical"
            break
        rep.severity = "minor"
    rep.stats = {
        "n_issues": len(rep.issues),
        "critical": sum(1 for i in rep.issues if i.severity == "critical"),
        "minor": sum(1 for i in rep.issues if i.severity == "minor"),
    }
    return rep


# ==================== Stage B：LLM 回看 ====================
REVIEW_SYSTEM = """你是农业科研问答的质检员。对照【用户意图】与【回答】复核，只输出 JSON：
{"requirement_met": true/false,
 "relevance": "high|medium|low",
 "citation_entailment": [
   {"claim": "抽查的数值/结论句(<=5条)", "ref_id": "1", "entailed": true,
    "note": "证据片段是否真实支撑该句"}],
 "caliber_issues": ["口径混淆点，如：实生选种获得率!=杂交育种比例"],
 "structure": "ok|verbose|missing_table|missing_diagram",
 "severity": "ok|minor|critical",
 "fixes": ["最小修改建议，每条一句话，禁止重写全文"]}
规则：只依据给出的证据片段判断；不确定标 false 并在 note 说明；不得臆造。
severity 判定：事实错误/捏造引用/答非所问/口径硬伤 = critical；冗长、排版 = minor。"""


def build_review_messages(query: str, answer: str, cited_refs: list,
                          intent_summary: str = "") -> list:
    previews = "\n".join(
        f"[{r.get('ref_id')}] {str(r.get('title', ''))[:80]} :: "
        f"{str(r.get('text_preview', ''))[:300]}"
        for r in (cited_refs or [])[:14])
    return [
        {"role": "system", "content": REVIEW_SYSTEM},
        {"role": "user", "content":
            f"【用户意图】{intent_summary or query}\n\n【证据片段】\n{previews}\n\n"
            f"【回答】\n{(answer or '')[:8000]}"},
    ]


def parse_review_json(text: str):
    import json
    m = re.search(r"\{.*\}", (text or ""), re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


async def review_with_llm(answer: str, *, query: str, intent_summary: str,
                          cited_refs: list, llm_call):
    """Stage B。llm_call(messages)->str 由集成方传入（fast 模型，temperature=0）。"""
    try:
        raw = await llm_call(
            build_review_messages(query, answer, cited_refs, intent_summary))
        return parse_review_json(raw)
    except Exception as e:
        logger.warning("[audit] llm review failed: %s", e)
        return None
