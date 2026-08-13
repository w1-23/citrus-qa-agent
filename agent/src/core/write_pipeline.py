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


def _call_llm(llm, prompt: str):
    """[System(固定) + Human(动态)] 消息结构统一入口。"""
    return llm.ainvoke([
        SystemMessage(content=_WRITE_SYSTEM_PROMPT),
        HumanMessage(content=prompt),
    ])

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

    logger.info(f"[WritePipeline] classify -> {out['mode']} (target_chars={out['target_chars']}, "
                f"section={out['target_section'] or 'none'}, file_exists={file_exists})")
    return out


# ─────────────────────────────────────────────
# 2. Stage 1: Plan（结构化大纲）
# ─────────────────────────────────────────────

def _build_plan_prompt(material_pack: list[dict], target_chars: int, retry_info: str = "") -> str:
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
    return (f"{prompt_file}\n\n---\n{size_line}\n"
            f"检索材料:\n{material_text}{extra}")


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


async def run_stage1_plan(llm, material_pack: list[dict], target_chars: int) -> tuple[Optional[dict], str]:
    """Stage 1: 生成并校验大纲。返回 (plan_dict, plan_text)；失败返回 (None, plan_text) 供 ReAct 回退。"""
    prompt = _build_plan_prompt(material_pack, target_chars)
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
            return plan, plan_text
        logger.warning(f"[WritePipeline] plan validation failed: {failed}")
        if attempt < retries:
            prompt = _build_plan_prompt(material_pack, target_chars,
                                        retry_info=json.dumps(failed, ensure_ascii=False))
    return None, plan_text


async def _react_fallback_write(llm, goal: str, plan_text: str, material_pack: list[dict],
                                output_path: str, gap: bool) -> dict:
    """Plan 失败回退: 带大纲一次性生成全文并落盘（v8.3.3）。

    单次 LLM 调用；内容超单章容量时按 ## 章节切块 write+append 分批写盘。
    LLM 也失败 → 返回提示文本（兜底告知 supervisor）。
    """
    if not output_path:
        return {"result": f"[plan_failed] 大纲生成失败，已回退常规写作。\n{plan_text[:500]}",
                "mode": "react_fallback", "chapters": 0, "total_chars": 0,
                "missing_sections": [], "truncated_sections": [], "material_gap": gap}
    prompt_file = _read_prompt("write-section.md")
    prompt = (f"{prompt_file}\n\n---\n任务: 撰写完整文档 — {goal[:300]}\n"
              f"参考大纲（用于结构参考）:\n{plan_text[:2000]}\n\n"
              f"检索材料:\n{_format_material_pack(material_pack, max_entries=25)}\n"
              f"要求: 一次输出完整 Markdown 文档（# 标题 + 摘要 + 分章节），"
              f"控制在 3000 字以内，宁精勿滥。")
    try:
        resp = await asyncio.wait_for(_call_llm(llm, prompt),
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
    max_per_section = int(settings.PIPELINE_SECTION_MAX_TOKENS / 1.2 * 0.9)
    blocks = [b.strip() for b in SECTION_SPLIT_RE.split(content) if b.strip()]
    total = 0
    try:
        if len(content) <= max_per_section or len(blocks) <= 1:
            msg = write_local_file.func(output_path, content, "write")
            if msg.startswith("Error"):
                raise RuntimeError(msg[:200])
            total = len(content)
        else:
            for i, b in enumerate(blocks):
                mode = "write" if i == 0 else "append"
                msg = write_local_file.func(output_path, b, mode)
                if msg.startswith("Error"):
                    raise RuntimeError(msg[:200])
                total += len(b)
    except Exception as e:
        logger.error(f"[WritePipeline] react_fallback write failed: {e}")
        return {"result": f"[plan_failed] 大纲生成失败，回退写作也未能保存。\n{plan_text[:300]}",
                "mode": "react_fallback", "chapters": 0, "total_chars": 0,
                "missing_sections": [], "truncated_sections": [], "material_gap": gap}

    logger.info(f"[WritePipeline] react_fallback saved: {total} chars -> {output_path}")
    return {"result": f"已保存到 {output_path}（{total} 字符，大纲失败回退模式）",
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


def _build_section_prompt(plan: dict, idx: int, section: dict, running_context: str,
                          material_pack: list[dict]) -> str:
    prompt_file = _read_prompt("write-section.md")
    outline_summary = {
        "title": plan.get("title", ""),
        "total_sections": len(plan.get("sections", [])),
        "all_headings": [s.get("heading", "") for s in plan.get("sections", [])],
    }
    material = _extract_material_subsets(section, material_pack)
    return (f"{prompt_file}\n\n---\n"
            f"全文大纲: {json.dumps(outline_summary, ensure_ascii=False)[:400]}\n"
            f"已完成章节概要: {running_context[:500] or '(本文第一章)'}\n"
            f"本章: {idx+1}/{len(plan.get('sections', []))} — {section.get('heading', '')}\n"
            f"本章要点: {json.dumps(section.get('points', []), ensure_ascii=False)[:400]}\n"
            f"本章目标字数: {section.get('target_chars', 600)}\n"
            f"相关材料:\n{material}")


async def run_stage2_execute(llm, plan: dict, material_pack: list[dict],
                             output_path: str, task_id: str = "",
                             resume_completed: Optional[list] = None) -> dict:
    """Stage 2: 逐章生成并写盘。返回 {chapters, total_chars, missing_sections, truncated_sections}。"""
    sections = plan.get("sections", [])
    if not sections:
        return {"chapters": 0, "total_chars": 0, "missing_sections": [],
                "truncated_sections": []}

    from src.core.progress_bus import emit_encoded
    from src.tools.file_ops import write_local_file
    from src.core.write_pipeline_state import mark_section_done

    resume_completed = resume_completed or []
    running_context = ""
    missing = []
    truncated = []
    total_chars = 0
    # 与 validate_plan 同一公式的单章安全容量（防 API 截断）
    max_per_section = int(settings.PIPELINE_SECTION_MAX_TOKENS / 1.2 * 0.9)

    for idx, section in enumerate(sections):
        if idx in resume_completed:
            logger.info(f"[WritePipeline] resume: 跳过已完成章节 {idx+1} {section.get('heading', '')}")
            continue
        try:
            emit_encoded("section_start", {"heading": section.get("heading", ""),
                                           "index": idx + 1, "total": len(sections)})
        except Exception:
            pass
        _update_job(f"writing {idx+1}/{len(sections)}", section.get("heading", ""))
        prompt = _build_section_prompt(plan, idx, section, running_context, material_pack)
        resp_content = ""
        for attempt in range(3):
            try:
                resp = await asyncio.wait_for(_call_llm(llm, prompt),
                                              timeout=settings.PIPELINE_SECTION_TIMEOUT)
                resp_content = (resp.content or "").strip()
                break
            except Exception as e:
                logger.warning(f"[WritePipeline] section {idx+1} LLM failed (attempt {attempt+1}/3): {e}")
                await asyncio.sleep(2 ** attempt + 0.5 * ((uuid.uuid4().int >> 32) % 1000) / 1000)
        if not resp_content:
            missing.append(section.get("heading", f"第{idx+1}章"))
            continue

        body, summary = extract_summary(resp_content)
        # v8.3.3 运行时容量兜底: 实际输出超单章安全容量 → 精简重写一次防截断
        if len(body) > max_per_section:
            logger.warning(f"[WritePipeline] section {idx+1} {len(body)} chars > "
                           f"{max_per_section}, condensed retry")
            try:
                condensed_prompt = prompt + (
                    f"\n\n【上一版输出 {len(body)} 字，超过安全容量 {max_per_section} 字（API 会截断）。"
                    f"请压缩到 {max_per_section} 字以内：保留核心内容与要点、删除冗余修饰与重复表述，"
                    f"并照常输出 <summary> 标签。】")
                resp = await asyncio.wait_for(
                    _call_llm(llm, condensed_prompt),
                    timeout=settings.PIPELINE_SECTION_TIMEOUT)
                condensed, summary2 = extract_summary((resp.content or "").strip())
                if condensed and len(condensed) <= max_per_section:
                    body, summary = condensed, summary2 or summary
                    logger.info(f"[WritePipeline] section {idx+1} condensed ok: {len(body)} chars")
                else:
                    logger.warning(f"[WritePipeline] section {idx+1} still {len(condensed)} chars "
                                   f"after condensed retry, writing as-is")
                    truncated.append(section.get("heading", f"第{idx+1}章"))
            except Exception as e:
                logger.warning(f"[WritePipeline] section {idx+1} condensed retry failed: {e}")
                truncated.append(section.get("heading", f"第{idx+1}章"))
        # 标题契约: 缺失则自动补
        if not SECTION_HEADING_RE.search(body):
            body = f"## {section.get('heading', '')}\n\n{body}"
        mode = "write" if idx == 0 else "append"
        if idx == 0 and plan.get("title"):
            header = (f"# {plan['title']}\n\n"
                      f"## 摘要\n{plan.get('abstract_draft', '')}\n\n"
                      f"## 关键词\n{', '.join(plan.get('keywords', []) or [])}\n\n")
            body = header + body
        try:
            msg = write_local_file.func(output_path, body, mode)
            if msg.startswith("Error"):
                # write_local_file 失败时返回错误字符串而非抛异常，必须显式检测
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

        # v8.3.3 写后 read-back: 确认章节已落盘
        if not _verify_section_written(output_path, section.get("heading", "")):
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
        write_local_file.func(output_path, body, "append")
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
    write_local_file.func(output_path, new_text, "write")
    return {"action": "rewrite_section",
            "result": f"已重写章节: {sections[idx]['heading']}"}


# ─────────────────────────────────────────────
# 5. 总入口
# ─────────────────────────────────────────────

async def run_write_pipeline(task: dict, material_pack: list[dict],
                             llm_factory=None, session_id: str = "") -> dict:
    """Plan-Execute 流水线总入口。

    Args:
        task: {"goal", "query", "output_path"}
        material_pack: 检索材料包（结构化 list[dict]）
        llm_factory: 可注入（测试用），默认构造 main 模型 ChatOpenAI
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
    cls = classify_write_task(goal, "", file_exists, llm=llm)

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
            from src.core.write_pipeline_state import find_resumable_task
            resumable = find_resumable_task(session_id=session_id, output_path=output_path)
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
        plan, plan_text = await run_stage1_plan(llm, material_pack, target_chars)
    else:
        plan_text = ""
    if plan is None:
        # ReAct 回退: 带大纲一次性生成全文并落盘（材料已在 pack，单次调用成本可控）
        return await _react_fallback_write(llm, goal, plan_text, material_pack,
                                           output_path, gap)

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

    exec_result = await run_stage2_execute(llm, plan, material_pack, output_path,
                                           task_id=task_id, resume_completed=resume_completed)
    try:
        from src.core.write_pipeline_state import finish_task
        finish_task(task_id, "done" if not exec_result["missing_sections"] else "partial")
    except Exception:
        pass
    # v8.3.3 写后引用完整性校验（正文 [n] vs 参考文献列表）
    ref_issues = []
    if exec_result["chapters"] > 0 and output_path:
        ref_issues = verify_reference_integrity(output_path)
        for issue in ref_issues[:5]:
            logger.warning(f"[WritePipeline] ref integrity: {issue}")
    summary = (f"综述已保存: {output_path}（{exec_result['total_chars']} 字符，"
               f"{exec_result['chapters']} 章）")
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
            "reference_issues": ref_issues, "material_gap": gap}
