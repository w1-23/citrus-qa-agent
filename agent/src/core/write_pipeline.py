"""Write Pipeline — Plan-Execute 长文写作流水线 (v8.3.2).

架构: 写任务四路路由 (direct_write / plan_execute / react / modify) +
      Plan(结构化大纲) → Execute(逐章独立生成) + 中断续传。

核心原则:
  - 每章独立 LLM 调用，输出预算 100% 聚焦一章（内容充实、永不截断）
  - 章间上下文 = running_context（前章 <summary> 标签，代码提取，非累积）
  - 材料按大纲 refs 语义分配（DOI 索引优先，标题模糊 fallback）
  - 单章失败 → 缺章占位 + 部分成功返回（优雅降级）
  - 断点续传: pipeline_tasks 表持久化 Plan 与已完成章节
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import sqlite3
import time
import uuid
from pathlib import Path
from typing import Optional

from langchain_core.messages import SystemMessage, HumanMessage

from src.config import settings, PROJECT_ROOT
from src.prompts.loader import _read_prompt

logger = logging.getLogger(__name__)

# 章节标题契约: 所有写入文件的章节必须以此开头（modify 切分依赖）
SECTION_HEADING_RE = re.compile(r"^##\s+(.+)$", re.MULTILINE)
SECTION_SPLIT_RE = re.compile(r"^(?=##\s)", re.MULTILINE)
SUMMARY_TAG_RE = re.compile(r"<summary>(.*?)</summary>", re.DOTALL)
CHAR_TARGET_RE = re.compile(r"(\d{3,})\s*(?:字|字符)")

_WORKSPACE_ROOT = (PROJECT_ROOT / "workspace" / "output").resolve()

# v8.4: 固定系统提示（字节级稳定，动态任务要求一律放 HumanMessage——
# 书 2.3 铁律: system 前缀不动，动态内容追加末尾；此前全部内容塞单条
# SystemMessage，前缀随任务变化）
_WRITE_SYSTEM_PROMPT = (
    "你是柑橘科研领域的中文写作专家。"
    "请严格遵循用户消息中的任务要求、材料引用规则与输出格式。"
)


def _call_llm(llm, prompt: str, max_tokens=None):
    """[System(固定) + Human(动态)] 消息结构统一入口。"""
    kw = {}
    if max_tokens:
        kw["max_tokens"] = max_tokens
    return llm.ainvoke([
        SystemMessage(content=_WRITE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ], **kw)

# ─────────────────────────────────────────────
# 1. 写任务分类（四路路由）
# ─────────────────────────────────────────────

_PLAN_EXECUTE_FAST = re.compile(r"综述|报告|文献回顾|research review|系统阐述|多章节|全面|展开论述")
_REACT_FAST = re.compile(r"先写|草稿|看看风格|分步|一步步|初稿|我改改|先给我看")
_DIRECT_WRITE_FAST = re.compile(r"保存|存到|写入文件|preserve|store")


def classify_write_task(goal: str, context: str, file_exists: bool,
                        llm=None) -> dict:
    """四路路由: direct_write / plan_execute / react / modify。

    快筛（覆盖 80% 明确场景）+ LLM 结构化分类兜底。
    Returns: {mode, target_section, target_chars}
    """
    goal = goal or ""
    context = context or ""
    out = {"mode": "react", "target_section": "", "target_chars": 0}

    # 第一层: 代码快筛
    if file_exists:
        out["mode"] = "modify"
        m = re.search(r"(?:第\s*([一二三四五六七八九十\d]+)\s*[章节部分]|“?([^”\s]{2,20}?)[章节部分]?\")", goal)
        # 粗略提取: 目标章节标题或序号（LLM 兜底会细化）
        if m:
            out["target_section"] = m.group(1) or m.group(2) or ""
    elif _DIRECT_WRITE_FAST.search(goal) and not re.search(r"写一[篇份]|撰写", goal):
        out["mode"] = "direct_write"
    elif _PLAN_EXECUTE_FAST.search(goal):
        out["mode"] = "plan_execute"
    elif _REACT_FAST.search(goal):
        out["mode"] = "react"
    else:
        # 第二层: LLM 结构化分类兜底
        if llm is not None:
            try:
                resp = llm.invoke([
                    SystemMessage(content=(
                        "Classify a writing task into exactly one mode. Reply with a JSON object "
                        '{"mode": "plan_execute" | "react" | "direct_write" | "modify", '
                        '"target_section": "" | "chapter title or number if modifying an existing file", '
                        '"target_chars": number}. '
                        "plan_execute: long structured document (review/report, many sections); "
                        "react: incremental/iterative writing (draft, adjust style); "
                        "direct_write: save EXISTING finished content verbatim; "
                        "modify: change a specific section of an existing file.")),
                    HumanMessage(content=f"Goal: {goal[:500]}\nFile exists: {file_exists}"),
                ])
                content = (resp.content or "").strip()
                content = re.sub(r"```(?:json)?\n?", "", content).replace("\n```", "").strip()
                data = json.loads(content)
                mode = str(data.get("mode", "react")).strip().lower()
                if mode in ("plan_execute", "react", "direct_write", "modify"):
                    out["mode"] = mode
                out["target_section"] = str(data.get("target_section", ""))[:80]
                out["target_chars"] = int(data.get("target_chars") or 0)
            except Exception as e:
                logger.warning(f"[WritePipeline] classify LLM failed, default react: {e}")

    # 目标字数提取（正则优先，LLM 兜底值其次；0 = 由 Plan 依据材料丰富度自主决定）
    m = CHAR_TARGET_RE.search(goal)
    if m:
        out["target_chars"] = int(m.group(1))
    elif not out.get("target_chars"):
        out["target_chars"] = 0

    # v8.12: 综述/长文档（plan_execute）不应被过小的目标字数约束——"300字"这类
    # 对一篇综述明显失真，会连带 validate 阈值、单章容量判据全部偏小（实测
    # target_chars=300 vs 实际产出 7341 字，差 24 倍）。< 800 视为未指定，
    # 转自主模式（篇幅由材料丰富度决定，validate 用 1500 下限口径）。
    if out["mode"] == "plan_execute" and out["target_chars"] and out["target_chars"] < 800:
        out["target_chars"] = 0

    logger.info(f"[WritePipeline] classify -> {out['mode']} (target_chars={out['target_chars']}, "
                f"section={out['target_section'] or 'none'}, file_exists={file_exists})")
    # v8.13: 结构化诊断事件（分类决策快照）
    try:
        from src.core.diag import diag
        diag("classify", mode=out["mode"], target_chars=out.get("target_chars", 0),
             target_section=out.get("target_section") or "", file_exists=file_exists)
    except Exception:
        pass
    return out


# ─────────────────────────────────────────────
# 2. Stage 1: Plan（结构化大纲）
# ─────────────────────────────────────────────

def _build_plan_prompt(material_pack: list[dict], target_chars: int, retry_info: str = "",
                       skill_catalog: str = "") -> str:
    prompt_file = _read_prompt("write-plan.md")
    material_text = _format_material_pack(material_pack, max_entries=25)
    if target_chars and target_chars > 0:
        size_line = f"目标字数: {target_chars} 字"
    else:
        # v8.3.2: 自主模式——篇幅由材料丰富度决定
        size_line = ("目标字数: 由你依据【检索材料】的丰富程度自主决定——"
                     "材料充分则深入展开（多章节、每章充实）；材料有限则精简成文（不灌水、不编造），"
                     "并如实说明哪些方面因材料不足而省略。")
    extra = ""
    if retry_info:
        extra = f"\n\n【上次大纲未通过校验，失败项: {retry_info}】请修正后重新输出完整 JSON。"
    # v8.6 (书 §2.5/4.8.2): 渐进式披露目录——Plan 阶段只见技能目录，全文按需加载
    catalog = f"\n\n{skill_catalog}" if skill_catalog else ""
    # v8.12: skills_used 声明引导前置到材料之前——此前引导只在目录末尾，被长
    # 材料列表淹没，模型常忽略 → 全量回退注入（渐进式披露失效，白费上下文）
    skill_declare = ""
    if skill_catalog:
        skill_declare = (
            "\n\n【重要】下方【可用写作技能】列出的技能若与本次写作相关，请务必在大纲 JSON 中声明"
            ' "skills_used": ["<技能id>", ...]（只列你真正会用到的技能 id）；确定不需要则声明空数组 []。'
            "未声明会默认注入可能不相关的技能全文，浪费上下文。"
        )
    return (f"{prompt_file}\n\n---\n{size_line}{skill_declare}\n"
            f"检索材料:\n{material_text}{catalog}{extra}")


# v8.3.8: 证据保真——chunk 最大 1992 字符，3000 为安全阀（当前语料零截断）；
# 总量由条数与累计预算控制，不砍单条正文
MATERIAL_EVIDENCE_MAX_CHARS = 3000
MATERIAL_TOTAL_MAX_CHARS = 60000


def _material_evidence(r: dict) -> str:
    """取材料证据文本：chunk 正文（text）优先（含机制/数字细节），摘要次之。"""
    text = str(r.get("text", "") or "").strip()
    if text:
        return text[:MATERIAL_EVIDENCE_MAX_CHARS]
    return str(r.get("abstract", r.get("snippet", "")))[:MATERIAL_EVIDENCE_MAX_CHARS]


def _format_material_pack(material_pack: list[dict], max_entries: int = 25) -> str:
    """材料包 → 文本（供 Plan 阶段 LLM 阅读）。总量受累计预算控制，不截断单条正文。"""
    if not material_pack:
        return "(无检索材料)"
    lines = []
    total = 0
    truncated_by_budget = False
    for i, r in enumerate(material_pack[:max_entries], 1):
        doi = r.get("doi", "")
        title = r.get("title", r.get("name", "Untitled"))[:120]
        evidence = _material_evidence(r)
        line = f"[{i}] {title} | DOI: {doi or 'N/A'}\n    {evidence}"
        total += len(line)
        if total > MATERIAL_TOTAL_MAX_CHARS:
            truncated_by_budget = True
            break
        lines.append(line)
    out = "\n".join(lines)
    if truncated_by_budget:
        out += (f"\n\n[材料总量达 {MATERIAL_TOTAL_MAX_CHARS} 字符预算，"
                f"已截断条数（共 {len(material_pack)} 篇，仅列前 {len(lines)} 篇）]")
    return out


def validate_plan(plan: dict, target_chars: int) -> tuple[bool, dict]:
    """动态阈值校验（v8.3.2 双模式 + 单章防截断上限）。

    - 用户指定字数: 严格校验（下限按目标联动）
    - 自主模式 (target_chars=0): 宽松下限（材料少可精简），只保证成文质量
    - 单章上限: target_chars ≤ 单章安全容量，防止 API 截断
    Returns: (ok, failed_checks)
    """
    sections = plan.get("sections") or []
    if not isinstance(sections, list) or not sections:
        return False, {"section_count": "sections 为空"}

    # 单章安全容量: section_max_tokens / 1.2 * 0.9（留余量防截断）
    max_per_section = int(settings.PIPELINE_SECTION_MAX_TOKENS / 1.2 * 0.9)
    if target_chars and target_chars > 0:
        target = max(target_chars, 300)
        checks = {
            "section_count": len(sections) >= max(3, target // 1500),
            "min_per_section": all(int(s.get("target_chars") or 0) >= 250 for s in sections),
            "max_per_section": all(int(s.get("target_chars") or 0) <= max_per_section for s in sections),
            "total_chars": sum(int(s.get("target_chars") or 0) for s in sections) >= target * 0.8,
            "ref_coverage": (sum(1 for s in sections if s.get("refs"))
                             / len(sections)) >= settings.PIPELINE_REFS_COVERAGE_RATIO,
        }
    else:
        # 自主模式: 材料有限时允许精简，但保证成文最低质量
        checks = {
            "section_count": len(sections) >= 2,
            "min_per_section": all(int(s.get("target_chars") or 0) >= 200 for s in sections),
            "max_per_section": all(int(s.get("target_chars") or 0) <= max_per_section for s in sections),
            "total_chars": sum(int(s.get("target_chars") or 0) for s in sections) >= 1500,
            "ref_coverage": (sum(1 for s in sections if s.get("refs"))
                             / len(sections)) >= 0.3,
        }
    failed = {k: v for k, v in checks.items() if not v}
    return (not failed), failed


async def run_stage1_plan(llm, material_pack: list[dict], target_chars: int,
                          skill_catalog: str = "") -> tuple[Optional[dict], str]:
    """Stage 1: 生成并校验大纲。返回 (plan_dict, plan_text)；失败返回 (None, plan_text) 供 ReAct 回退。"""
    prompt = _build_plan_prompt(material_pack, target_chars, skill_catalog=skill_catalog)
    retries = settings.PIPELINE_MAX_PLAN_RETRIES
    plan_text = ""
    for attempt in range(retries + 1):
        try:
            resp = await _call_llm(llm, prompt)
        except Exception as e:
            logger.warning(f"[WritePipeline] plan LLM failed (attempt {attempt+1}): {e}")
            await asyncio.sleep(2)
            continue
        plan_text = (resp.content or "").strip()
        content = re.sub(r"```(?:json)?\n?", "", plan_text).replace("\n```", "").strip()
        try:
            plan = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(f"[WritePipeline] plan JSON parse failed: {e}")
            continue
        ok, failed = validate_plan(plan, target_chars)
        if ok:
            logger.info(f"[WritePipeline] plan ok: {len(plan.get('sections', []))} sections, "
                        f"total={sum(int(s.get('target_chars') or 0) for s in plan.get('sections', []))}")
            # v8.13: 结构化诊断事件（大纲决策快照）
            try:
                from src.core.diag import diag
                diag("plan", ok=True, attempt=attempt,
                     sections=len(plan.get("sections", [])),
                     total=sum(int(s.get("target_chars") or 0) for s in plan.get("sections", [])),
                     target_chars=target_chars,
                     skills_used=str(plan.get("skills_used", []))[:200])
            except Exception:
                pass
            return plan, plan_text
        logger.warning(f"[WritePipeline] plan validation failed: {failed}")
        if attempt < retries:
            prompt = _build_plan_prompt(material_pack, target_chars,
                                        retry_info=json.dumps(failed, ensure_ascii=False),
                                        skill_catalog=skill_catalog)
    # v8.13: 结构化诊断事件（大纲失败快照——供定位 validate 卡点）
    try:
        from src.core.diag import diag
        diag("plan", ok=False, target_chars=target_chars,
             failed=str(locals().get("failed", "unknown"))[:300])
    except Exception:
        pass
    return None, plan_text


async def _react_fallback_write(llm, goal: str, plan_text: str, material_pack: list[dict],
                                output_path: str, gap: bool, skill_prompt: str = "") -> dict:
    """Plan 失败回退: 带大纲一次性生成全文并落盘（v8.3.3）。

    单次 LLM 调用；内容超单章容量时按 ## 章节切块 write+append 分批写盘。
    LLM 也失败 → 返回提示文本（兜底告知 supervisor）。
    v8.4.4: 注入写作 skill（追加语义，不影响静态缓存）。
    """
    if not output_path:
        return {"result": f"[plan_failed] 大纲生成失败，已回退常规写作。\n{plan_text[:500]}",
                "mode": "react_fallback", "chapters": 0, "total_chars": 0,
                "missing_sections": [], "truncated_sections": [], "material_gap": gap}
    prompt_file = _read_prompt("write-section.md")
    prompt = (f"{prompt_file}\n\n---\n任务: 撰写完整文档 — {goal[:300]}\n"
              f"参考大纲（用于结构参考）:\n{plan_text[:2000]}\n\n"
              f"检索材料:\n{_format_material_pack(material_pack, max_entries=25)}\n"
              f"{_format_skill_block(skill_prompt)}"
              f"要求: 一次输出完整 Markdown 文档（# 标题 + 摘要 + 分章节）。"
              f"长度不限、不要省略章节与证据细节——系统会自动分块写盘，"
              f"无需自行压缩篇幅。")
    try:
        resp = await asyncio.wait_for(
            _call_llm(llm, prompt, max_tokens=8000),
            timeout=settings.PIPELINE_SECTION_TIMEOUT * 3)
        content, _ = extract_summary((resp.content or "").strip())
    except Exception as e:
        logger.warning(f"[WritePipeline] react_fallback LLM failed: {e}")
        content = ""
    if not content:
        return {"result": f"[plan_failed] 大纲生成失败，已回退常规写作。\n{plan_text[:500]}",
                "mode": "react_fallback", "chapters": 0, "total_chars": 0,
                "missing_sections": [], "truncated_sections": [], "material_gap": gap}

    from src.tools.file_ops import write_local_file
    from src.tools.registry import run_tool_checked
    max_per_section = int(settings.PIPELINE_SECTION_MAX_TOKENS / 1.2 * 0.9)
    blocks = [b.strip() for b in SECTION_SPLIT_RE.split(content) if b.strip()]
    total = 0
    draft = _draft_path(output_path)
    try:
        if len(content) <= max_per_section or len(blocks) <= 1:
            msg = await run_tool_checked(write_local_file,
                                         {"path": draft, "content": content, "mode": "write"})
            if msg.startswith("Error") or msg.startswith("[ERR"):
                raise RuntimeError(msg[:200])
            total = len(content)
        else:
            for i, b in enumerate(blocks):
                mode = "write" if i == 0 else "append"
                msg = await run_tool_checked(write_local_file,
                                             {"path": draft, "content": b, "mode": mode})
                if msg.startswith("Error") or msg.startswith("[ERR"):
                    raise RuntimeError(msg[:200])
                total += len(b)
    except Exception as e:
        logger.error(f"[WritePipeline] react_fallback write failed: {e}")
        return {"result": f"[plan_failed] 大纲生成失败，回退写作也未能保存。\n{plan_text[:300]}",
                "mode": "react_fallback", "chapters": 0, "total_chars": 0,
                "missing_sections": [], "truncated_sections": [], "material_gap": gap}

    published = _publish_draft(output_path)
    logger.info(f"[WritePipeline] react_fallback saved: {total} chars -> {output_path}"
                f" (published={published})")
    loc = f"已保存到 {output_path}" if published \
        else f"草稿已保存到 {draft}（发布失败，可手动核对）"
    return {"result": f"{loc}（{total} 字符，大纲失败回退模式）",
            "mode": "react_fallback", "chapters": len(blocks), "total_chars": total,
            "missing_sections": [], "truncated_sections": [], "material_gap": gap}


# ─────────────────────────────────────────────
# 3. Stage 2: Execute（逐章生成）
# ─────────────────────────────────────────────

def _update_job(step: str, summary: str = "") -> None:
    """v8.3.7 M2: 同步任务进度到 task_jobs（断连保活后可查询）。"""
    try:
        from src.core.jobs import update_job
        from src.core.tracing import get_job_id
        fields = {"current_step": step}
        if summary:
            fields["progress_summary"] = summary
        update_job(get_job_id(), **fields)
    except Exception:
        pass


def extract_summary(resp: str) -> tuple[str, str]:
    """分离正文与 <summary> 标签。返回 (body, summary)。"""
    m = SUMMARY_TAG_RE.search(resp)
    if m:
        summary = m.group(1).strip()[:100]
        body = SUMMARY_TAG_RE.sub("", resp).rstrip()
        return body, summary
    return resp.rstrip(), ""


# ── v8.4.6 F7: 草稿-发布机制（在确认无法恢复之前不暴露中间态）──

def _draft_path(output_path: str) -> str:
    """写作期间的草稿文件名（同目录，.draft.md 后缀）。"""
    return f"{output_path}.draft.md"


def _publish_draft(output_path: str) -> bool:
    """草稿校验通过后原子发布为正式文件（os.replace）。"""
    draft = (_WORKSPACE_ROOT / _draft_path(output_path)).resolve()
    final = (_WORKSPACE_ROOT / output_path).resolve()
    try:
        if not draft.exists():
            return False
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(draft, final)
        logger.info(f"[WritePipeline] 草稿已发布: {output_path}")
        return True
    except Exception as e:
        logger.warning(f"[WritePipeline] 草稿发布失败（草稿保留）: {e}")
        return False


def _verify_section_written(output_path: str, heading: str) -> bool:
    """写后 read-back: 文件存在且含本章节标题（防静默写失败/路径逃逸）。

    容错匹配: 与文件中 ## 标题做 精确/包含 双向比对（LLM 可能微调标题措辞）。
    """
    target = (_WORKSPACE_ROOT / output_path).resolve()
    try:
        if not target.exists():
            return False
        content = target.read_text(encoding="utf-8")
        if not heading:
            return bool(content.strip())
        for m in SECTION_HEADING_RE.finditer(content):
            h = m.group(1).strip()
            if heading == h or heading in h or h in heading:
                return True
        return False
    except Exception:
        return False


def verify_reference_integrity(output_path: str) -> list:
    """引用完整性校验（轻量、非 LLM）: 正文 [n] 编号 vs 参考文献区编号。

    Returns: 问题列表（空 = 无问题）。
    """
    target = (_WORKSPACE_ROOT / output_path).resolve()
    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        return [f"read_back_failed: {e}"]
    refs_m = re.search(r"(?:^|\n)#{1,3}\s*参考文献\s*", content)
    body = content[:refs_m.start()] if refs_m else content
    cited = {int(n) for n in re.findall(r"\[(\d{1,3})\]", body)}
    refs_text = content[refs_m.start():] if refs_m else ""
    listed = {int(n) for n in re.findall(r"\[(\d{1,3})\]\s*[\.\]\s\S]", refs_text)}
    issues = []
    if not refs_m:
        issues.append("未找到参考文献区")
    for n in sorted(cited - listed):
        issues.append(f"正文引用[{n}] 无对应文献条目")
    for n in sorted(listed - cited):
        issues.append(f"文献[{n}] 未被正文引用")
    return issues


# ── v8.4.1 分章写作后统一引用（代码提取暂存 → 全局重编号 → 文末合并）──

_REF_MARKER_RE = re.compile(
    r"^(?:#{1,3}\s*)?\*{0,2}(?:本章)?参考文献\*{0,2}\s*$")
_REF_ENTRY_RE = re.compile(
    r"^\s*(?:\[(\d{1,3})\]|(\d{1,3})[.、）)])\s*(.*)$")
_CITE_MARKER_RE = re.compile(r"\[(\d{1,3}(?:\s*[,，]\s*\d{1,3})*)\]")
_DOI_RE = re.compile(r"10\.\d{4,9}/[^\s\]\[)）,，;；]+")


def _unify_references(output_path: str) -> dict:
    """分章写作后统一引用: 提取各章"本章参考文献"→ 全局重编号 → 文末合并。

    背景: 每章 LLM 独立编号（各章都从 [1] 起），正文 [n] 与本章引用区一一对应。
    统一策略: 按正文引用出现顺序分配全局号；跨章重复条目（DOI 或文本相同）合并
    为同一全局号；重写正文标记；删除各章引用区；文末追加统一 "## 参考文献"。

    Returns: {"unified": int, "chapters": int, "dropped": [str]}（unified=0 表示
    未找到任何引用区，文件保持不变）。
    """
    target = (_WORKSPACE_ROOT / output_path).resolve()
    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[WritePipeline] unify refs read failed: {e}")
        return {"unified": 0, "chapters": 0, "dropped": [str(e)]}

    lines = content.split("\n")
    preamble: list[str] = []          # # 标题 / ## 摘要 / ## 关键词 等前置块
    chapters: list[dict] = []         # {"heading": str, "body": [str], "refs": [str]}
    current: dict | None = None
    in_refs = False

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("## "):
            if current is None and (stripped.startswith("## 摘要")
                                    or stripped.startswith("## 关键词")):
                preamble.append(line)
                continue
            if _REF_MARKER_RE.match(stripped) and current is not None:
                # "## 本章参考文献" 这类带标题级标记的引用区 → 并入当前章
                in_refs = True
                continue
            if current is not None:
                chapters.append(current)
            current = {"heading": line, "body": [], "refs": []}
            in_refs = False
            continue
        if stripped.startswith("# ") and current is None:
            # 文档标题（## 章级以下才分段）
            preamble.append(line)
            continue
        if current is None:
            preamble.append(line)
            continue
        if not in_refs and _REF_MARKER_RE.match(stripped):
            in_refs = True
            continue
        if in_refs:
            current["refs"].append(line)
        else:
            current["body"].append(line)
    if current is not None:
        chapters.append(current)

    # 解析每章引用条目: {local_num: entry_text}
    chapter_entries: list[dict] = []
    for ch in chapters:
        entries: dict[int, str] = {}
        cur_num: int | None = None
        cur_text: list[str] = []
        for line in ch["refs"]:
            m = _REF_ENTRY_RE.match(line)
            if m and (m.group(1) or m.group(2)):
                if cur_num is not None:
                    entries[cur_num] = "\n".join(cur_text).strip()
                cur_num = int(m.group(1) or m.group(2))
                cur_text = [m.group(3)]
            elif cur_num is not None:
                cur_text.append(line)
        if cur_num is not None:
            entries[cur_num] = "\n".join(cur_text).strip()
        chapter_entries.append(entries)

    # 全局编号: 按正文引用出现顺序；跨章重复引用（DOI/文本）合并
    doi_map: dict[str, int] = {}
    text_map: dict[str, int] = {}
    global_entries: list[str] = []
    global_counter = 0
    dropped: list[str] = []

    new_chapter_bodies: list[str] = []
    for ch, entries in zip(chapters, chapter_entries):
        local_map: dict[int, int] = {}
        body_text = "\n".join(ch["body"])

        def _assign(local: int) -> int:
            nonlocal global_counter
            if local in local_map:
                return local_map[local]
            entry = entries.get(local)
            if not entry:
                # v8.13 A1 配套: 悬空引用（正文 [n] 无条目）删除标记而非保留原编号——
                # 保留会把悬空编号写回正文，与全局重排后的合法编号撞号（语义漂移）
                dropped.append(f"{ch['heading'][:30]}: 引用[{local}] 无条目，已删除")
                local_map[local] = 0  # 0 = 删除哨兵
                return 0
            doi_m = _DOI_RE.search(entry)
            doi = doi_m.group(0).rstrip(".,;。，；") if doi_m else ""
            norm = re.sub(r"\s+", " ", entry)[:120]
            if doi and doi in doi_map:
                g = doi_map[doi]
            elif norm and norm in text_map:
                g = text_map[norm]
            else:
                global_counter += 1
                g = global_counter
                global_entries.append(entry)
                if doi:
                    doi_map[doi] = g
                if norm:
                    text_map[norm] = g
            local_map[local] = g
            return g

        def _rewrite(m: re.Match) -> str:
            nums = [int(x) for x in re.split(r"[,，\s]+", m.group(1)) if x.strip()]
            mapped = [str(_assign(n)) for n in nums]
            mapped = [s for s in mapped if s != "0"]  # 悬空引用删除标记
            if not mapped:
                return ""
            return "[" + ",".join(mapped) + "]"

        new_body = _CITE_MARKER_RE.sub(_rewrite, body_text)
        # 去掉引用区前残留的分隔线
        new_body = new_body.rstrip()
        new_body = re.sub(r"\n*-{3,}\s*$", "", new_body)
        new_chapter_bodies.append(ch["heading"] + "\n\n" + new_body.strip())

    if global_counter == 0:
        # 未找到任何引用条目 → 文件保持原样
        return {"unified": 0, "chapters": len(chapters), "dropped": dropped}

    out = "\n".join(preamble).rstrip() + "\n\n"
    out += "\n\n".join(new_chapter_bodies)
    out += "\n\n---\n\n## 参考文献\n\n"
    out += "\n\n".join(f"[{i+1}] {e}" for i, e in enumerate(global_entries))
    out += "\n"

    try:
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_text(out, encoding="utf-8")
        os.replace(tmp, target)
        logger.info(
            f"[WritePipeline] 引用统一: {len(global_entries)} 条全局引用, "
            f"{len(chapters)} 章合并, dropped={len(dropped)}")
        # v8.13: 结构化诊断事件（引用统一快照）
        try:
            from src.core.diag import diag
            diag("unify", unified=len(global_entries), chapters=len(chapters),
                 dropped=len(dropped))
        except Exception:
            pass
        return {"unified": len(global_entries), "chapters": len(chapters),
                "dropped": dropped}
    except Exception as e:
        logger.warning(f"[WritePipeline] unify refs write failed: {e}")
        return {"unified": 0, "chapters": len(chapters), "dropped": [str(e)]}


def _prune_unreferenced_refs(output_path: str) -> dict:
    """v8.12: 裁剪文末参考文献中未被正文引用的条目，并重排为连续编号。

    在 _unify_references 之后运行，保证"参考文献列表 == 正文实际引用集合"，
    消除 verify_reference_integrity 报出的"文献[n] 未被正文引用"僵尸引用
    （此前只告警不处理，正式发布文件里残留未引用条目）。
    纯规则、无 LLM 调用；正文 [n] 重写复用 _CITE_MARKER_RE 正则。
    Returns: {"pruned": int, "before": int, "after": int}
    """
    target = (_WORKSPACE_ROOT / output_path).resolve()
    try:
        content = target.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"[WritePipeline] prune refs read failed: {e}")
        return {"pruned": 0, "before": 0, "after": 0}

    refs_m = re.search(r"(?:^|\n)#{1,3}\s*参考文献\s*", content)
    if not refs_m:
        return {"pruned": 0, "before": 0, "after": 0}
    head = content[:refs_m.start()]
    refs_tail = content[refs_m.start():]

    # 正文引用的编号集合
    cited = {int(n) for n in re.findall(r"\[(\d{1,3})\]", head)}

    # 解析参考文献区条目（复用 _REF_ENTRY_RE 的条目行解析）
    entries: list[tuple[int, str]] = []
    cur_num: int | None = None
    cur_text: list[str] = []
    for line in refs_tail.split("\n"):
        m = _REF_ENTRY_RE.match(line)
        if m and (m.group(1) or m.group(2)):
            if cur_num is not None:
                entries.append((cur_num, "\n".join(cur_text).strip()))
            cur_num = int(m.group(1) or m.group(2))
            cur_text = [m.group(3)]
        elif cur_num is not None:
            cur_text.append(line)
    if cur_num is not None:
        entries.append((cur_num, "\n".join(cur_text).strip()))

    if not entries:
        return {"pruned": 0, "before": 0, "after": 0}

    kept = [(n, t) for n, t in entries if n in cited]
    pruned = len(entries) - len(kept)
    if pruned == 0:
        return {"pruned": 0, "before": len(entries), "after": len(kept)}

    # old -> new 连续编号映射（未引用条目被删除后，后续编号前移）
    remap = {n: i + 1 for i, (n, _t) in enumerate(kept)}

    def _rewrite_cite(m: re.Match) -> str:
        nums = [int(x) for x in re.split(r"[,，\s]+", m.group(1)) if x.strip()]
        # v8.13 A1 配套: 正文引用了但文献区无条目的悬空编号一并删除标记
        # （remap.get(n,n) 保留原编号会撞上重排后的合法编号）
        mapped = [str(remap[n]) for n in nums if n in remap]
        if not mapped:
            return ""
        return "[" + ",".join(mapped) + "]"

    new_head = _CITE_MARKER_RE.sub(_rewrite_cite, head)
    kept_block = "\n\n".join(f"[{i + 1}] {t}" for i, (_n, t) in enumerate(kept))
    new_content = new_head.rstrip() + "\n\n## 参考文献\n\n" + kept_block + "\n"

    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(new_content, encoding="utf-8")
    os.replace(tmp, target)
    # v8.13: 悬空引用同步计数（dangling = 正文引用但无条目，随重排一并删除）
    dangling = sum(1 for n in re.findall(r"\[(\d{1,3})\]", head)
                   if int(n) not in remap)
    logger.info(f"[WritePipeline] 引用裁剪: 删除 {pruned} 条未引用文献 "
                f"({len(entries)} -> {len(kept)}), 悬空引用删除 {dangling} 处")
    # v8.13: 结构化诊断事件（引用裁剪快照——追踪"僵尸引用"发生率）
    try:
        from src.core.diag import diag
        diag("prune", pruned=pruned, before=len(entries), after=len(kept))
    except Exception:
        pass
    return {"pruned": pruned, "before": len(entries), "after": len(kept)}


def _extract_material_subsets(plan_section: dict, material_pack: list[dict]) -> str:
    """按 refs 从材料包抽取子集（DOI 精确 → 标题模糊 fallback），累计 ≤8000 字符。

    v8.3.9: 单条正文 3000 安全阀（当前语料零截断），累计 8000 超限截断条数并标记；
    refs 未匹配项显式反馈（LLM 明确证据缺口）。
    """
    refs = plan_section.get("refs") or []
    by_doi = {}
    for r in material_pack:
        doi = (r.get("doi") or "").strip().lower()
        if doi:
            by_doi[doi] = r
    picked = []
    unmatched_refs = []
    for ref in refs:
        ref_s = str(ref).strip()
        if not ref_s:
            continue
        doi = ref_s.lower()
        r = by_doi.get(doi)
        if r is None:
            # 标题模糊匹配
            for cand in material_pack:
                title = str(cand.get("title", "")).lower()
                if title and (ref_s.lower() in title or title[:40] in ref_s.lower()):
                    r = cand
                    break
        if r and r not in picked:
            picked.append(r)
        elif r is None:
            # v8.3.7 G3: 未匹配 refs 显式反馈——LLM 明确知道证据缺口
            unmatched_refs.append(ref_s[:80])
    if not picked:
        # 兜底: 取材料包前 3 篇
        picked = material_pack[:3]
    lines = []
    total = 0
    for r in picked:
        evidence = _material_evidence(r)
        line = f"- {r.get('title', 'Untitled')[:100]} | DOI: {r.get('doi', 'N/A')}\n  {evidence}"
        total += len(line)
        if total > 8000:
            break
        lines.append(line)
    out = "\n".join(lines) or "(无匹配材料)"
    if unmatched_refs:
        out += (f"\n\n⚠️ 以下大纲 refs 未在材料库中匹配到: {'; '.join(unmatched_refs)}"
                f"——相关内容需标注 [模型知识] 或省略，不得虚构文献支撑。")
    return out


def _format_skill_block(skill_prompt: str) -> str:
    """写作 skill 注入块（v8.4.3/8.4.4）: 前 3 块、≤4000 字符（控制单章 prompt 成本）。"""
    if not skill_prompt:
        return ""
    blocks = [b.strip() for b in skill_prompt.split("\n---\n") if b.strip()]
    picked = "\n---\n".join(blocks[:3])[:4000]
    return f"\n\n## 写作技能参考\n{picked}"


# ── v8.6 渐进式披露（书 §2.5/4.8.2，O6 落地）──
# 第一层：目录常驻（Plan 阶段只给技能名+一句话用途，≤800 字符）；
# 第二层：模型在大纲 JSON 中声明 skills_used → 按需注入全文到单章 prompt。
# 质量兜底：模型未声明时自动注入匹配度最高的第 1 个技能全文（不劣于旧行为）。

def _build_skill_catalog(skill_map: dict) -> str:
    """渐进式披露目录：技能 id + 名称 + 首行用途（≤800 字符）。"""
    if not skill_map:
        return ""
    lines: list[str] = []
    total = 0
    for sid, meta in list(skill_map.items())[:6]:
        name = str(meta.get("name") or sid)[:60]
        content = str(meta.get("content") or "")
        first_line = next(
            (l.strip() for l in content.splitlines()
             if l.strip() and len(l.strip()) > 8), "")[:120]
        line = f"- [{sid}] {name}" + (f": {first_line}" if first_line else "")
        if total + len(line) > 800:
            break
        lines.append(line)
        total += len(line)
    if not lines:
        return ""
    return ("## 可用写作技能（渐进式披露：仅目录，按需加载全文）\n"
            + "\n".join(lines)
            + '\n如写作需要某项技能的完整指导，请在返回的大纲 JSON 中附加 '
              '"skills_used": ["<技能id>", ...]（可空数组；不附加则使用默认技能）。')


def _resolve_skills_used(plan: dict, plan_text: str, skill_map: dict) -> list[str]:
    """从大纲解析模型声明要使用的技能 id（JSON 字段优先，文本标记兜底）。"""
    if not skill_map:
        return []
    ids: list[str] = []
    raw = plan.get("skills_used") if isinstance(plan, dict) else None
    if isinstance(raw, list):
        ids = [str(x) for x in raw if str(x) in skill_map]
    if not ids and plan_text:
        ids = re.findall(r'use_skill\("([^"]+)"\)', plan_text)
        ids = [x for x in ids if x in skill_map]
    return ids


def _build_selected_skill_prompt(skill_map: dict, skills_used: list[str]) -> str:
    """按声明拼装选中技能全文；未声明 → 保持旧行为（匹配顺序全部注入，
    _format_skill_block 截前 3 块 ≤4000 字符）——渐进式披露只增不减。

    输出格式与旧 skill_prompt 一致（"## Skill: name\ncontent" 块，\n---\n 连接）。
    """
    if not skill_map:
        return ""
    if skills_used:
        ordered = [sid for sid in skills_used if sid in skill_map]
    else:
        ordered = list(skill_map.keys())
    parts = []
    for sid in ordered[:5]:
        meta = skill_map[sid]
        parts.append(f"## Skill: {meta.get('name', sid)}\n\n{meta.get('content', '')}")
    return "\n---\n".join(parts)


def _build_section_prompt(plan: dict, idx: int, section: dict, running_context: str,
                          material_pack: list[dict], skill_prompt: str = "") -> str:
    prompt_file = _read_prompt("write-section.md")
    outline_summary = {
        "title": plan.get("title", ""),
        "total_sections": len(plan.get("sections", [])),
        "all_headings": [s.get("heading", "") for s in plan.get("sections", [])],
    }
    material = _extract_material_subsets(section, material_pack)
    skill_block = _format_skill_block(skill_prompt)
    return (f"{prompt_file}\n\n---\n"
            f"全文大纲: {json.dumps(outline_summary, ensure_ascii=False)[:400]}\n"
            f"已完成章节概要: {running_context[:500] or '(本文第一章)'}\n"
            f"本章: {idx+1}/{len(plan.get('sections', []))} — {section.get('heading', '')}\n"
            f"本章要点: {json.dumps(section.get('points', []), ensure_ascii=False)[:400]}\n"
            f"本章目标字数: {section.get('target_chars', 600)}\n"
            f"相关材料:\n{material}"
            f"{skill_block}")


async def run_stage2_execute(llm, plan: dict, material_pack: list[dict],
                             output_path: str, task_id: str = "",
                             resume_completed: Optional[list] = None,
                             skill_prompt: str = "",
                             skill_map: Optional[dict] = None,
                             skills_used: Optional[list] = None) -> dict:
    """Stage 2: 逐章生成并写盘。返回 {chapters, total_chars, missing_sections, truncated_sections}。

    v8.6 (书 §2.5/4.8.2 渐进式披露): 传 skill_map + skills_used 时按模型声明
    注入选中技能全文（未声明 → 默认第 1 个匹配技能，质量兜底）；不传 skill_map
    时保持旧行为（skill_prompt 原样注入）。
    """
    sections = plan.get("sections", [])
    if not sections:
        return {"chapters": 0, "total_chars": 0, "missing_sections": [],
                "truncated_sections": []}

    from src.core.progress_bus import emit_encoded
    from src.tools.file_ops import write_local_file
    from src.tools.registry import run_tool_checked
    from src.core.write_pipeline_state import mark_section_done

    # v8.6: 渐进式披露——模型声明的技能全文（未声明 → 旧行为全量注入，零回归）
    if skill_map:
        selected = _build_selected_skill_prompt(skill_map, skills_used or [])
        if selected:
            skill_prompt = selected
            logger.info(f"[WritePipeline] progressive disclosure: "
                        f"skills_used={skills_used or ['<未声明: 全量(旧行为)>']} "
                        f"({len(skill_prompt)} chars)")
    resume_completed = resume_completed or []
    running_context = ""
    missing = []
    truncated = []
    total_chars = 0
    # 与 validate_plan 同一公式的单章安全容量（防 API 截断）
    max_per_section = int(settings.PIPELINE_SECTION_MAX_TOKENS / 1.2 * 0.9)

    # v8.4.6 F7: 分章写入草稿文件（.draft.md），全部完成+引用统一+完整性校验后
    # 原子发布为正式文件——中断/缺章时工作区不留半成品
    draft = _draft_path(output_path)

    # v8.4.6: 单章生成并发（parallel_sections，默认 3）——各章 prompt 共享
    # 完整大纲（标题+全部章节名）与本章材料，章节间无强顺序依赖；
    # 生成结果仍按章节序号顺序写盘（append 语义与引用统一不受影响）。
    # 质量说明: 放弃"前章摘要"衔接（running_context），但大纲全貌保留，
    # 综述章节独立性高、正文科学内容不受影响；1=串行可回退旧行为。
    sem = asyncio.Semaphore(max(1, int(getattr(
        settings, "PIPELINE_PARALLEL_SECTIONS", 3) or 3)))

    async def _gen_section(idx: int, section: dict):
        """生成单章（并发执行；返回 (body, summary) 或 None）。"""
        async with sem:
            prompt = _build_section_prompt(plan, idx, section, "",
                                           material_pack, skill_prompt)
            resp_content = ""
            for attempt in range(3):
                try:
                    resp = await asyncio.wait_for(
                        _call_llm(llm, prompt),
                        timeout=settings.PIPELINE_SECTION_TIMEOUT)
                    resp_content = (resp.content or "").strip()
                    break
                except Exception as e:
                    logger.warning(
                        f"[WritePipeline] section {idx+1} LLM failed "
                        f"(attempt {attempt+1}/3): {e}")
                    await asyncio.sleep(2 ** attempt + 0.5 * ((uuid.uuid4().int >> 32) % 1000) / 1000)
            if not resp_content:
                return idx, None, ""
            body, summary = extract_summary(resp_content)
            if len(body) > max_per_section:
                logger.warning(f"[WritePipeline] section {idx+1} {len(body)} chars > "
                               f"{max_per_section}, condensed retry")
                try:
                    # v8.12: 把上一版正文喂进压缩 prompt——此前只挂一句"请压缩"，
                    # 模型看不到待压内容 → 等于重新生成一遍，长度不受控（实测
                    # section 3442→3420 几乎无效）。喂回原文才能真正压缩。
                    condensed_prompt = (
                        f"下面是本章已生成的正文（{len(body)} 字符），超出安全容量 {max_per_section} 字符。"
                        f"请将它压缩到 {max_per_section} 字符以内：保留核心观点、关键数据、"
                        "所有引用编号 [n] 与本章结论，删除冗余修饰与重复表述，并在末尾照常输出 "
                        '<summary>摘要</summary> 标签。\n\n'
                        f"=== 待压缩正文 ===\n{body}\n=== 结束 ==="
                    )
                    resp = await asyncio.wait_for(
                        _call_llm(llm, condensed_prompt),
                        timeout=settings.PIPELINE_SECTION_TIMEOUT)
                    condensed, summary2 = extract_summary((resp.content or "").strip())
                    if condensed and len(condensed) <= max_per_section:
                        # v8.13: 结构化诊断事件（压缩前后长度——评估压缩有效性）
                        try:
                            from src.core.diag import diag
                            diag("section_condensed", idx=idx + 1,
                                 before=len(body), after=len(condensed))
                        except Exception:
                            pass
                        body, summary = condensed, summary2 or summary
                        logger.info(f"[WritePipeline] section {idx+1} condensed ok: {len(body)} chars")
                    else:
                        # v8.13: 结构化诊断事件（压缩失败——仍超容量）
                        try:
                            from src.core.diag import diag
                            diag("section_condensed", idx=idx + 1,
                                 before=len(body), after=len(condensed), ok=False)
                        except Exception:
                            pass
                        logger.warning(f"[WritePipeline] section {idx+1} still {len(condensed)} chars "
                                       f"after condensed retry, writing as-is")
                        truncated.append(section.get("heading", f"第{idx+1}章"))
                except Exception as e:
                    logger.warning(f"[WritePipeline] section {idx+1} condensed retry failed: {e}")
                    truncated.append(section.get("heading", f"第{idx+1}章"))
            if not SECTION_HEADING_RE.search(body):
                body = f"## {section.get('heading', '')}\n\n{body}"
            return idx, body, summary

    pending = [i for i in range(len(sections)) if i not in resume_completed]
    for idx in resume_completed:
        logger.info(f"[WritePipeline] resume: 跳过已完成章节 {idx+1} {sections[idx].get('heading', '')}")
    gen_results: dict = {}
    if pending:
        results = await asyncio.gather(*[_gen_section(i, sections[i]) for i in pending])
        gen_results = {idx: (body, summary) for idx, body, summary in results}

    # 按章节序号顺序写盘（append 语义）
    for idx, section in enumerate(sections):
        if idx in resume_completed:
            continue
        try:
            emit_encoded("section_start", {"heading": section.get("heading", ""),
                                           "index": idx + 1, "total": len(sections)})
        except Exception:
            pass
        _update_job(f"writing {idx+1}/{len(sections)}", section.get("heading", ""))
        gen = gen_results.get(idx)
        if gen is None or not gen[0]:
            missing.append(section.get("heading", f"第{idx+1}章"))
            continue
        body, summary = gen
        mode = "write" if idx == 0 else "append"
        if idx == 0 and plan.get("title"):
            header = (f"# {plan['title']}\n\n"
                      f"## 摘要\n{plan.get('abstract_draft', '')}\n\n"
                      f"## 关键词\n{', '.join(plan.get('keywords', []) or [])}\n\n")
            body = header + body
        try:
            # v8.13 第四批: 统一工具执行出口（沙箱/超时/offload 一致）
            msg = await run_tool_checked(write_local_file,
                                         {"path": draft, "content": body, "mode": mode})
            if msg.startswith("Error") or msg.startswith("[ERR"):
                # 失败返回错误字符串而非抛异常，必须显式检测
                logger.error(f"[WritePipeline] write failed section {idx+1}: {msg[:200]}")
                missing.append(section.get("heading", f"第{idx+1}章"))
                continue
            total_chars += len(body)
            logger.info(f"[WritePipeline] section {idx+1}/{len(sections)} done: "
                        f"{len(body)} chars | {msg[:80]}")
        except Exception as e:
            logger.error(f"[WritePipeline] write failed section {idx+1}: {e}")
            missing.append(section.get("heading", f"第{idx+1}章"))
            continue

        # v8.3.3 写后 read-back: 确认章节已落盘（草稿）
        if not _verify_section_written(draft, section.get("heading", "")):
            logger.error(f"[WritePipeline] read-back FAILED section {idx+1} "
                         f"{section.get('heading', '')}")
            missing.append(section.get("heading", f"第{idx+1}章"))
            continue

        mark_section_done(task_id, idx)
        points = section.get('points') or ['']
        running_context += (f"第{idx+1}章({section.get('heading', '')}): "
                            f"{summary or points[0]}；")
        running_context = running_context[-500:]
        try:
            emit_encoded("section_done", {"heading": section.get("heading", ""),
                                          "index": idx + 1, "total": len(sections)})
        except Exception:
            pass
        _update_job(f"done {idx+1}/{len(sections)}", "")
        # v8.4.1: 业务日志——单章完成
        try:
            from src.core.business_logger import blog
            blog("section_done", idx=idx + 1, total=len(sections),
                 heading=str(section.get("heading", ""))[:40],
                 chars=len(body))
        except Exception:
            pass
        # v8.13: 结构化诊断事件（单章完成快照——443s 耗时问题的分解入口）
        try:
            from src.core.diag import diag
            diag("section", idx=idx + 1, total=len(sections),
                 heading=str(section.get("heading", ""))[:40],
                 chars=len(body), target=section.get("target_chars", 0))
        except Exception:
            pass

    return {"chapters": len(sections) - len(missing) - len(resume_completed),
            "total_chars": total_chars, "missing_sections": missing,
            "truncated_sections": truncated}


# ─────────────────────────────────────────────
# 4. Modify: 定向修改已有文档
# ─────────────────────────────────────────────

def _split_sections(markdown_text: str) -> list[dict]:
    """按 `## ` 二级标题安全切分（正则前瞻，不切碎 ###）。"""
    parts = SECTION_SPLIT_RE.split(markdown_text)
    sections = []
    for p in parts:
        if not p.strip():
            continue
        m = SECTION_HEADING_RE.search(p)
        heading = m.group(1).strip() if m else ""
        sections.append({"heading": heading, "body": p})
    return sections


def _locate_section(sections: list[dict], target: str) -> int:
    """定位目标章节: 标题精确匹配 → 序数 fallback。返回索引或 -1。

    注意: 头部段（heading=""，如 # 标题+摘要）占用索引 0，序数定位须基于真实章节列表。
    """
    t = (target or "").strip()
    if not t:
        return -1
    real_idx = [i for i, s in enumerate(sections) if s["heading"]]
    # 序数: "3"/"三" → 第 N 个真实章节
    num_map = {"一": 1, "二": 2, "三": 3, "四": 4, "五": 5, "六": 6, "七": 7, "八": 8, "九": 9, "十": 10}
    if t.isdigit():
        n = int(t) - 1
        if 0 <= n < len(real_idx):
            return real_idx[n]
    elif t in num_map:
        n = num_map[t] - 1
        if 0 <= n < len(real_idx):
            return real_idx[n]
    # 标题精确/包含匹配
    for i, s in enumerate(sections):
        if s["heading"] and (t == s["heading"] or t in s["heading"] or s["heading"] in t):
            return i
    return -1


async def modify_document(llm, output_path: str, target_section: str, user_goal: str,
                          material_pack: list[dict]) -> dict:
    """modify 分支: 章节重写 / 追加新章节。返回 {action, result}。"""
    from src.tools.file_ops import write_local_file
    from src.tools.registry import run_tool_checked
    target = (PROJECT_ROOT / "workspace" / "output" / output_path).resolve()
    if not target.exists():
        return {"action": "error", "result": f"[ERR_FILE_NOT_FOUND] 文件不存在: {output_path}"}
    text = target.read_text(encoding="utf-8")
    sections = _split_sections(text)
    idx = _locate_section(sections, target_section)

    # 追加新章节（目标不存在且用户要求"补充/加一节"）
    if idx < 0 and re.search(r"加[一节章]|补充|新增", user_goal):
        prompt_file = _read_prompt("write-section.md")
        prompt = (f"{prompt_file}\n\n---\n全文已有章节: {[s['heading'] for s in sections]}\n"
                  f"任务: 新增章节 — {user_goal}\n材料: {_format_material_pack(material_pack, 8)}")
        resp = await asyncio.wait_for(_call_llm(llm, prompt),
                                      timeout=settings.PIPELINE_SECTION_TIMEOUT)
        body, _ = extract_summary((resp.content or "").strip())
        if not SECTION_HEADING_RE.search(body):
            body = f"## {target_section or '补充章节'}\n\n{body}"
        # v8.13 A15: 检查写结果（此前吞错误，写失败也报"成功"）
        msg = await run_tool_checked(write_local_file,
                                     {"path": output_path, "content": body, "mode": "append"})
        if msg.startswith("Error") or msg.startswith("[ERR"):
            return {"action": "error", "result": msg[:200]}
        return {"action": "append_section", "result": f"已追加章节: {target_section or '补充章节'}"}

    if idx < 0:
        return {"action": "error",
                "result": f"未找到目标章节「{target_section}」。现有章节: {[s['heading'] for s in sections]}"}

    # 章节重写: 原章节 + 新材料 → 单章生成 → 替换
    old_body = sections[idx]["body"]
    prompt_file = _read_prompt("write-section.md")
    prompt = (f"{prompt_file}\n\n---\n任务: 重写以下章节（保持 `## {sections[idx]['heading']}` 标题）\n"
              f"用户要求: {user_goal}\n"
              f"原章节内容:\n{old_body[:2000]}\n\n"
              f"相关材料:\n{_format_material_pack(material_pack, 8)}")
    resp = await asyncio.wait_for(_call_llm(llm, prompt),
                                  timeout=settings.PIPELINE_SECTION_TIMEOUT)
    new_body, _ = extract_summary((resp.content or "").strip())
    if not SECTION_HEADING_RE.search(new_body):
        new_body = f"## {sections[idx]['heading']}\n\n{new_body}"

    new_text = text.replace(old_body, new_body, 1)
    if new_text == text:
        # 替换失败（原文被 normalize），回退: 整文重建
        rebuilt = "".join(
            (new_body if i == idx else s["body"]) for i, s in enumerate(sections)
        )
        new_text = rebuilt
    # v8.13 A15: 检查写结果（此前吞错误，写失败也报"成功"）
    msg = await run_tool_checked(write_local_file,
                                 {"path": output_path, "content": new_text, "mode": "write"})
    if msg.startswith("Error") or msg.startswith("[ERR"):
        return {"action": "error", "result": msg[:200]}
    return {"action": "rewrite_section",
            "result": f"已重写章节: {sections[idx]['heading']}"}


# ─────────────────────────────────────────────
# 5. 总入口
# ─────────────────────────────────────────────

async def run_write_pipeline(task: dict, material_pack: list[dict],
                             llm_factory=None, session_id: str = "",
                             skill_prompt: str = "",
                             skill_map: Optional[dict] = None,
                             cls: Optional[dict] = None) -> dict:
    """Plan-Execute 流水线总入口。

    Args:
        task: {"goal", "query", "output_path"}
        material_pack: 检索材料包（结构化 list[dict]）
        llm_factory: 可注入（测试用），默认构造 main 模型 ChatOpenAI
        skill_prompt: v8.4.3 写作 skill 内容（追加到单章 prompt，不影响静态缓存）
        skill_map: v8.6 渐进式披露 {skill_id: {name, content}}——Plan 阶段只注入
            目录，模型在 skills_used 中声明后再注入全文（未声明兜底 top-1）
        cls: v8.12 调用方已分类结果（expert_graph 路由时已调 classify），传入则
            跳过重复分类；None 时内部补一次（向后兼容测试/独立调用）
    Returns:
        {"result": str, "mode": str, "chapters": int, "total_chars": int,
         "missing_sections": list, "truncated_sections": list,
         "reference_issues": list, "material_gap": bool}
    """
    from src.core.progress_bus import emit_encoded

    goal = task.get("goal", "")
    output_path = task.get("output_path", "")
    if llm_factory is None:
        from langchain_openai import ChatOpenAI
        def llm_factory():
            return ChatOpenAI(
                model=settings.MAIN_MODEL,
                api_key=settings.RESOLVED_MAIN_API_KEY,
                base_url=settings.MAIN_BASE_URL,
                temperature=0,
                timeout=settings.PIPELINE_SECTION_TIMEOUT,
            )
    llm = llm_factory()

    # 材料包充分性检查
    gap = len(material_pack) < settings.PIPELINE_MATERIAL_MIN_COUNT
    if gap:
        logger.info(f"[WritePipeline] material gap: {len(material_pack)} < "
                    f"{settings.PIPELINE_MATERIAL_MIN_COUNT}")

    file_exists = bool(output_path) and (_WORKSPACE_ROOT / output_path).resolve().exists()
    # v8.12: 复用调用方已分类结果（expert_graph 路由时已调过一次 classify），避免
    # 重复分类——LLM 兜底场景下会多一次 LLM 调用 + 延迟（正则快筛命中时仅日志重复）
    if cls is None:
        cls = classify_write_task(goal, "", file_exists, llm=llm)
    # v8.13: 结构化诊断事件（流水线入口快照）
    try:
        from src.core.diag import diag
        diag("pipeline_start", mode=cls.get("mode", ""),
             target_chars=cls.get("target_chars", 0),
             pack=len(material_pack), gap=gap)
    except Exception:
        pass

    if cls["mode"] == "direct_write":
        return {"result": "[direct_write] 现成内容应由 supervisor 直写，不应进入流水线",
                "mode": "direct_write", "chapters": 0, "total_chars": 0,
                "missing_sections": [], "material_gap": gap}

    if cls["mode"] == "modify" and file_exists:
        r = await modify_document(llm, output_path, cls["target_section"], goal, material_pack)
        return {"result": r["result"], "mode": "modify", "chapters": 0, "total_chars": 0,
                "missing_sections": [], "material_gap": gap}

    if cls["mode"] == "react":
        # 渐进式写作由调用方（supervisor）转入 ReAct 路径，此处仅做结构化交接
        return {"result": "[react] 已转入渐进式写作流程", "mode": "react",
                "chapters": 0, "total_chars": 0, "missing_sections": [],
                "truncated_sections": [], "material_gap": gap}

    # plan_execute
    target_chars = cls["target_chars"]
    try:
        emit_encoded("status", {"stage": "planning", "message": "正在规划综述结构..."})
    except Exception:
        pass

    # 断点续传: 同 session+path 的未完成任务 → 复用已完成的 Plan 与材料
    # (fix: plan 必须前置初始化，否则非续传路径引用未绑定变量 → UnboundLocalError)
    resume_completed = []
    task_id = ""
    plan = None
    try:
        if settings.PIPELINE_RESUME_ENABLED and output_path:
            # v8.10r: 原子领取（置 resumed）防同 session 并发双续传
            from src.core.write_pipeline_state import claim_resumable_task
            resumable = claim_resumable_task(session_id=session_id, output_path=output_path)
            if resumable:
                plan = resumable["plan"]
                resume_completed = resumable["completed"]
                task_id = resumable["task_id"]
                # v8.3.7 G2: 材料持久化复用——断点续传不再依赖重新检索
                saved_materials = resumable.get("materials") or []
                if saved_materials and len(saved_materials) >= len(material_pack):
                    material_pack = saved_materials
                    logger.info(f"[WritePipeline] resume: 复用持久化材料 {len(material_pack)} 篇")
                logger.info(f"[WritePipeline] resume task {task_id}: "
                            f"{len(resume_completed)}/{len(plan.get('sections', []))} 章已完成")
    except Exception as e:
        logger.warning(f"[WritePipeline] resume check failed: {e}")

    if plan is None:
        # v8.6: 渐进式披露——Plan 阶段只注入技能目录（≤800 字符）
        catalog = _build_skill_catalog(skill_map) if skill_map else ""
        # v8.13 A3: 续传任务在 plan 阶段被取消同样置 aborted（task_id 来自 claim）
        try:
            plan, plan_text = await run_stage1_plan(
                llm, material_pack, target_chars, skill_catalog=catalog)
        except BaseException:
            if task_id:
                try:
                    from src.core.write_pipeline_state import finish_task
                    finish_task(task_id, "aborted")
                except Exception:
                    pass
            raise
    else:
        plan_text = ""
    if plan is None:
        # ReAct 回退: 带大纲一次性生成全文并落盘（材料已在 pack，单次调用成本可控）
        return await _react_fallback_write(llm, goal, plan_text, material_pack,
                                            output_path, gap, skill_prompt=skill_prompt)

    # v8.4.13 第三步: 写作计划事件（前端「📋 执行计划」折叠块）——大纲就绪即展示
    try:
        from src.core.progress_bus import emit_encoded
        # v8.10q: 字段修正——plan 的 sections 项字段是 heading（非 title），
        # 此前取 title 恒空 → 前端执行计划折叠块"内部为空"（只显示章节数）
        sections = [str(s.get("heading", "") or s.get("title", ""))
                    for s in (plan.get("sections") or []) if s][:20]
        emit_encoded("plan", {
            "title": str(plan.get("title") or "")[:80],
            "sections": sections,
            "chapters": len(sections),
            "target_chars": int(target_chars or 0),
            "mode": "plan_execute",
        })
    except Exception:
        pass

    # v8.3.7: output_path 缺失兜底——supervisor 可能不传路径（LLM 漏参），
    # 此时路径解析会落到 workspace/output 目录本身 → os.replace 覆盖目录失败。
    # 用大纲标题生成默认文件名。
    if not output_path:
        safe_title = re.sub(r'[\\/:*?"<>|\r\n]+', '_',
                            (plan.get("title") or "综述").strip())[:40] or "综述"
        output_path = f"{safe_title}_{uuid.uuid4().hex[:6]}.md"
        logger.info(f"[WritePipeline] output_path 缺失，生成默认路径: {output_path}")

    if not task_id:
        try:
            from src.core.write_pipeline_state import start_task
            task_id = start_task(session_id, output_path, plan, material_pack)
        except Exception as e:
            logger.warning(f"[WritePipeline] start_task failed: {e}")

    try:
        emit_encoded("plan_ready", {
            "title": plan.get("title", ""),
            "sections": [s.get("heading", "") for s in plan.get("sections", [])],
            "total_chars": sum(int(s.get("target_chars") or 0) for s in plan.get("sections", [])),
        })
    except Exception:
        pass
    _update_job(f"plan_ready {len(plan.get('sections', []))}sections",
                plan.get("title", ""))

    # v8.6 (书 §2.5/4.8.2): 解析模型声明使用的技能（JSON skills_used / use_skill 标记）
    skills_used: list = []
    if skill_map:
        skills_used = _resolve_skills_used(plan, plan_text, skill_map)
        logger.info(f"[WritePipeline] skills_used={skills_used or ['<未声明: 全量(旧行为)>']}")

    # v8.13 A3: 执行阶段兜底——用户取消/进程中断（CancelledError 等）时置 aborted，
    # 配合 state 层"陈旧 resumed 可重领"，续传不再永久卡死
    try:
        exec_result = await run_stage2_execute(llm, plan, material_pack, output_path,
                                               task_id=task_id, resume_completed=resume_completed,
                                               skill_prompt=skill_prompt,
                                               skill_map=skill_map, skills_used=skills_used)
    except BaseException:
        try:
            from src.core.write_pipeline_state import finish_task
            finish_task(task_id, "aborted")
        except Exception:
            pass
        raise
    # v8.4.1: 分章写作后统一引用——各章"本章参考文献"提取合并为文末全局引用
    # v8.4.6 F7: 统一引用与完整性校验作用于草稿，通过后原子发布正式文件
    draft = _draft_path(output_path)
    # v8.13 A3: 全量续传（全部章节已完成）时 chapters=0 但草稿完整——unify/
    # verify/publish 仍应作用于草稿（此前跳过 → "未产生可保存内容"且草稿永不发布）
    has_draft = bool(output_path) and (_WORKSPACE_ROOT / draft).resolve().exists()
    content_ready = (exec_result["chapters"] > 0
                     or (has_draft and not exec_result["missing_sections"]))
    unify_info = {"unified": 0}
    if content_ready and output_path:
        try:
            unify_info = _unify_references(draft)
        except Exception as e:
            logger.warning(f"[WritePipeline] unify refs failed: {e}")
        # v8.12: 统一引用后裁剪未引用条目（僵尸引用）并重排编号，保证文末参考
        # 文献列表 == 正文实际引用集合（此前 verify 只告警不拦截，正式文件里
        # 残留"文献[n] 未被正文引用"条目；裁剪后 verify 无引用问题 → 正常发布）
        try:
            _prune_unreferenced_refs(draft)
        except Exception as e:
            logger.warning(f"[WritePipeline] prune refs failed: {e}")
    try:
        from src.core.business_logger import blog
        blog("pipeline_done", chapters=exec_result["chapters"],
             total_chars=exec_result["total_chars"],
             missing=len(exec_result["missing_sections"]),
             refs_unified=unify_info.get("unified", 0),
             output=output_path[:80])
    except Exception:
        pass
    try:
        from src.core.write_pipeline_state import finish_task
        finish_task(task_id, "done" if not exec_result["missing_sections"] else "partial")
    except Exception:
        pass
    # v8.3.3 写后引用完整性校验（正文 [n] vs 参考文献列表；统一引用后再校验）
    ref_issues = []
    if content_ready and output_path:
        ref_issues = verify_reference_integrity(draft)
        for issue in ref_issues[:5]:
            logger.warning(f"[WritePipeline] ref integrity: {issue}")
        # v8.13: 结构化诊断事件（引用完整性快照——裁剪后正常应为 0）
        try:
            from src.core.diag import diag
            diag("ref_verify", issues=len(ref_issues),
                 sample="; ".join(ref_issues[:3])[:300])
        except Exception:
            pass
    # v8.4.6 F7: 校验通过 → 原子发布；缺章/失败 → 保留草稿并明确告知
    # v8.13: 悬空引用同样拦截发布——此前只拦缺章，"正文引用[n] 无对应文献条目"
    # 会发布进正式文件。仅拦截悬空引用类 issue（"未找到参考文献区"是质量提示
    # 不阻断；"未引用条目"已由引用裁剪消除），保留草稿供补写/人工核对。
    published = False
    _dangling = [i for i in ref_issues if "无对应文献条目" in i]
    if (content_ready
            and not exec_result["missing_sections"]
            and not _dangling):
        published = _publish_draft(output_path)
    # v8.13: 结构化诊断事件（发布决策快照）
    try:
        from src.core.diag import diag
        diag("publish", published=published, chapters=exec_result["chapters"],
             total_chars=exec_result["total_chars"],
             missing=len(exec_result["missing_sections"]),
             ref_issues=len(ref_issues))
    except Exception:
        pass
    if published:
        location = f"已保存到 {output_path}"
    elif content_ready:
        location = f"草稿已保存到 {draft}（存在缺章/未发布，可续写或手动核对）"
    else:
        location = f"未产生可保存内容（{output_path}）"
    summary = (f"综述{location}（{exec_result['total_chars']} 字符，"
               f"{exec_result['chapters']} 章）")
    # v8.4.6 B9: 写作自检由代码校验（对照书 §1.2.2 验证 / 实验7-17"口头声称去验证"教训）
    code_checks = {
        "sections_ok": exec_result["chapters"],
        "sections_total": len(plan.get("sections", [])),
        "ref_issues": len(ref_issues),
        "published": published,
    }
    if exec_result["chapters"] == len(plan.get("sections", [])) and not ref_issues:
        summary += "\n代码自检: 章节完整、引用配对通过。"
    else:
        _issues = []
        if exec_result["missing_sections"]:
            _issues.append(f"缺章 {len(exec_result['missing_sections'])} 个")
        if ref_issues:
            _issues.append(f"引用问题 {len(ref_issues)} 项")
        summary += f"\n代码自检: 未全部通过（{'；'.join(_issues)}）。"
    if exec_result["missing_sections"]:
        summary += f"\n⚠️ 缺章: {', '.join(exec_result['missing_sections'])}（生成失败）"
    if exec_result.get("truncated_sections"):
        summary += (f"\n⚠️ 章节超容量有截断风险: "
                    f"{', '.join(exec_result['truncated_sections'])}（精简重试后仍超限）")
    if ref_issues:
        summary += f"\n⚠️ 引用完整性: {'；'.join(ref_issues[:3])}"
    if gap:
        summary += f"\n⚠️ 材料不足（{len(material_pack)} 篇），部分章节可能缺少文献支撑"
    summary += f"\n\n摘要: {plan.get('abstract_draft', '')}"

    return {"result": summary, "mode": "plan_execute",
            "chapters": exec_result["chapters"], "total_chars": exec_result["total_chars"],
            "missing_sections": exec_result["missing_sections"],
            "truncated_sections": exec_result.get("truncated_sections", []),
            "reference_issues": ref_issues, "material_gap": gap,
            # v8.4.6 B8: 结构化状态（缺章/引用问题 → partial）
            # v8.13: 改为发布结果驱动——全量续传等"发布成功但含提示级引用问题"
            # （如未写参考文献区）的场景 status 与发布结果一致，不再矛盾
            "status": "ok" if published else "partial"}
