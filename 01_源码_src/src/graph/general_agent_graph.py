# -*- coding: utf-8 -*-
"""Single general-purpose tool-using Agent graph.

The model owns retrieval planning and stopping. Code enforces one wall-clock
deadline, executes selected tools, and stores the unmodified final answer.
"""
from __future__ import annotations

import asyncio
import hashlib
import html
import json
import logging
import os
import re
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from src.config import get_deepseek_model, settings
from src.core.agent_loop import emit_llm_usage
from src.core.model_capabilities import model_supports_images
from src.core.progress_bus import (
    emit_status, emit_text,
    emit_tool_call_start, emit_tool_executing, emit_tool_result,
)
from src.core.stream_llm import stream_llm_response
from src.graph.state import AgentState
from src.prompts.loader import assemble_system_prompt

logger = logging.getLogger(__name__)

_FINAL_INSTRUCTION = (
    "现在请根据用户的原始问题，以及本轮已取得的全部网页和本地证据，"
    "直接给出最终回答。用户明确指定的长度、数量和格式优先；默认简洁，"
    "不要把单篇论文请求扩成综述。按问题需要自然组织篇幅和结构；重要事实可直接附上"
    "工具实际返回的 Markdown URL 或 DOI 链接。不要声称做过未完成的核验，"
    "不要输出检索过程或固定格式说明。"
)
_FINAL_RESERVE_MIN_SEC = 35.0
_FINAL_RESERVE_MAX_SEC = 60.0
_NO_PROGRESS_BATCHES_TO_STOP = 2
_MAX_MODEL_TOOL_TURNS = 8
_MAX_TOOL_CALLS_PER_BATCH = 8
_MAX_SOURCE_ENFORCEMENT_ATTEMPTS = 2
_COMPLETION_TOKEN_BUCKETS = (512, 1024, 1536, 2048, 4096, 8192, 12288, 16384, 32768)
_MAX_PARALLEL_AGENT_TOOLS = 8
_DSML_PREFIX = r"(?:[|｜]{1,2}\s*DSML\s*[|｜]{1,2}\s*)?"
_DSML_PROTOCOL_RE = re.compile(
    r"<\s*" + _DSML_PREFIX + r"(?:calls|invoke|parameter)\b|"
    r"\binvoke\s+name\s*=|\bparameter\s+name\s*=|"
    r"</?\s*(?:tool_call|tool_calls|function_call|function_calls)\b",
    re.I,
)
_DSML_CALLS_OPEN_RE = re.compile(r"<\s*" + _DSML_PREFIX + r"calls\s*>", re.I)
_DSML_CALLS_CLOSE_RE = re.compile(r"</\s*" + _DSML_PREFIX + r"calls\s*>", re.I)
_DSML_INVOKE_OPEN_RE = re.compile(
    r"<\s*" + _DSML_PREFIX + r"invoke\b(?P<attrs>[^>]*)>", re.I)
_DSML_INVOKE_CLOSE_RE = re.compile(r"</\s*" + _DSML_PREFIX + r"invoke\s*>", re.I)
_DSML_PARAMETER_RE = re.compile(
    r"<\s*" + _DSML_PREFIX + r"parameter\b(?P<attrs>[^>]*)>"
    r"(?P<value>.*?)"
    r"</\s*" + _DSML_PREFIX + r"parameter\s*>", re.I | re.S)
_DSML_ATTR_RE = re.compile(r"([\w:-]+)\s*=\s*([\"'])(.*?)\2", re.S)
_RETRIEVAL_CUES = re.compile(
    r"最新|进展|论文|文献|研究|证据|来源|核实|检索|查找|查询|搜索|查一下|查一查|搜一下|找一下|查查|资料|数据库|知识库|"
    r"详细|机制|过程|为什么|如何|比较|对比|设计|数据|年份|doi|pmid|"
    r"审查|检查数据|数据质量|异常值|离群|重复|缺失值|表格|参考文献核验|doi核验|图像复核|图片重复|补充数据|补充材料|"
    r"课题记录|项目记录|实验方案|实验设计|复现|导出|csv|tsv|xlsx|xlsm?|pdf|docx|"
    r"记住|记一下|记下来|加入记忆|长期记忆|保存|存一下|备忘|回顾课题|之前记录|以前保存|语料缺口|覆盖范围|"
    r"recent|latest|paper|literature|study|evidence|source|verify|search|"
    r"mechanism|process|compare|design|data|doi|pmid|outlier|audit|csv|xlsx|protocol", re.I)
_LOCAL_CITRUS_CORPUS_CUES = re.compile(
    r"柑橘|柑桔|橘子|柚子|柠檬|金柑|枳壳|枳橙|甜橙|脐橙|砂糖橘|温州蜜柑|"
    r"黄龙病|柑橘溃疡|柑橘品种|柑橘砧木|"
    r"\bcitrus\b|\bmandarins?\b|\bpomelos?\b|\blemons?\b|\bkumquats?\b|"
    r"\btrifoliate orange\b|\bcitrus rootstocks?\b|\bHLB\b|"
    r"\b(?:sweet|navel|blood|bitter|sour) oranges?\b", re.I)


def _source_scope(query: str) -> str:
    """Respect explicit user source constraints before exposing tools."""
    q = str(query or "").lower()
    if re.search(r"(?:融合搜索|融合检索|本地.{0,8}联网|(?:仅|只).{0,4}本地.{0,8}(?:搜索|检索))"
                 r".{0,8}(?:是什么|什么意思|含义|如何理解)", q):
        return "auto"
    if re.search(r"(?:只|仅)(?:(?:查|用|看|进行|搜索|检索)(?:一下|一次)?)?(?:本地|本地库|知识库)"
                 r"|(?:只|仅)(?:在|从)(?:本地|本地库|知识库)(?:查|检索|搜索)"
                 r"|(?:只|仅)(?:在|从)(?:本地|本地库|知识库).{0,24}(?:查|检索|搜索)"
                 r"|不要联网|不许联网|仅限本地", q):
        return "local"
    if re.search(r"(?:只|仅)(?:(?:查|用|看|进行|搜索|检索)(?:一下|一次)?)?(?:联网|网络|网页)"
                 r"|(?:只|仅)(?:在|从)(?:联网|网络|网页)(?:查|检索|搜索)"
                 r"|(?:只|仅)(?:在|从)(?:联网|网络|网页).{0,24}(?:查|检索|搜索)"
                 r"|不要查本地|不查本地|不要用知识库|仅限联网", q):
        return "web"
    mentions_both = (re.search(r"(?:联网|網絡|网络|网页).{0,16}(?:本地|知识库)|"
                               r"(?:本地|知识库).{0,16}(?:联网|网络|网页)", q)
                     or re.search(r"(?:融合|结合).{0,8}(?:本地|知识库).{0,8}(?:联网|网络|网页)|"
                                  r"(?:融合|结合).{0,8}(?:联网|网络|网页).{0,8}(?:本地|知识库)", q))
    fusion_action = re.search(r"融合|结合|同时查|一起查|交叉核对|对照本地", q)
    if ((mentions_both and fusion_action)
            or (fusion_action and re.search(r"搜索|检索|查询|核对", q))
            or re.search(r"同时查|一起查|交叉核对|对照本地", q)):
        return "fusion"
    if re.search(r"(?:请|帮我|要求|需要|进行).{0,10}(?:本地|本地库|知识库).{0,8}(?:查|检索|搜索)", q):
        return "local"
    if re.search(r"(?:请|帮我|要求|需要|进行).{0,10}(?:联网|网络|网页).{0,8}(?:查|检索|搜索)", q):
        return "web"
    return "auto"


def _explicit_literature_request(query: str) -> bool:
    return bool(re.search(
        r"文献|论文|参考文献|引用核验|文献核验|查证引用|检索.{0,8}(?:研究|文章|论文)|"
        r"找.{0,6}(?:论文|文献)|查.{0,6}(?:论文|文献)|"
        r"literature|references?|papers?|studies|scholarly sources|doi verification",
        str(query or ""), re.I))


def _local_corpus_matches(query: str) -> bool:
    """Whether this request is in scope for CitrusQA's citrus-focused corpus."""
    return bool(_LOCAL_CITRUS_CORPUS_CUES.search(str(query or "")))


def _explicit_data_audit(query: str) -> bool:
    text = str(query or "")
    intent = re.compile(
        r"(?:审查|检查|核查|分析).{0,10}(?:原始)?(?:实验)?(?:数据|数值|表格|工作簿|源数据|CSV|TSV|Excel|XLSX|XLSM?)|"
        r"(?:实验)?(?:数据|数值|表格|工作簿|源数据|CSV|TSV|Excel|XLSX|XLSM?).{0,10}(?:异常|质量|合理性|审查|检查|核查|离群|重复|缺失)|"
        r"\b(?:audit|check|review|inspect)\b.{0,18}\b(?:data|spreadsheet|table|workbook)\b",
        re.I,
    )
    negation = re.compile(r"(?:不要|不需要|无需|不用|不必|不想|不希望|暂不|先不|别|请勿|勿|do not|don't|not)\s*$", re.I)
    # Evaluate each clause independently so a later explicit request can override
    # an earlier negation, while “不要审查数据” never triggers the audit tool.
    clauses = re.split(r"[，,。！？!?；;\n]+|但是|不过|然而|\bbut\b|\bhowever\b", text, flags=re.I)
    for clause in clauses:
        for match in intent.finditer(clause):
            if not negation.search(clause[:match.start()].rstrip()):
                return True
    return False


def _explicit_reference_check(query: str) -> bool:
    return bool(re.search(
        r"参考文献.{0,8}(?:真实|存在|核验|核查|检查)|(?:核验|核查|检查).{0,8}(?:doi|引用|参考文献)|"
        r"doi.{0,8}(?:核验|核查|真实)|(?:verify|check).{0,12}(?:references?|citations?|doi)",
        str(query or ""), re.I))


def _main_model_supports_image_input() -> bool:
    """Resolve image support from the actually selected provider/model pair."""
    return model_supports_images(
        getattr(settings, "RESOLVED_PROVIDER_ID", "deepseek"),
        str(get_deepseek_model() or ""),
    )


def _is_simple_query(query: str, scope: str) -> bool:
    if scope != "auto":
        return False
    q = str(query or "").strip()
    if not q or _RETRIEVAL_CUES.search(q):
        return False
    return len(q) <= 48 and not re.search(r"[?？].*[?？]|以及|并且|分别|综述|全面", q)


def _scoped_toolset(scope: str):
    tools, _schemas = _toolset()
    research_tools = {
        "inspect_uploaded_table", "audit_uploaded_data",
        "verify_uploaded_references", "fetch_and_audit_open_supplements", "export_literature_evidence",
        "export_paper_analysis", "export_experiment_design", "save_research_note",
        "recall_research_notes", "export_research_notes", "export_corpus_gap_report",
    }
    if scope == "local":
        # Source scope must also apply to network-capable research tools.
        network_research_tools = {"verify_uploaded_references", "fetch_and_audit_open_supplements"}
        tools = [t for t in tools if t.name.startswith("local_") or
                 (t.name in research_tools and t.name not in network_research_tools)]
    elif scope == "web":
        tools = [t for t in tools if t.name.startswith("deepseek_") or t.name in research_tools]
    return tools, []


def _answer_profile(query: str) -> tuple[int, str]:
    """Choose a bounded completion budget and concise task-specific guidance.

    Output caps are rounded to a small set of buckets so user-specified lengths
    do not create an unbounded number of cached HTTP clients.
    """
    q = " ".join(str(query or "").lower().split())
    paper_recommendation = (
        bool(re.search(r"推荐|找|检索|提供|给我|列出", q))
        and bool(re.search(r"论文|文献|paper|article", q, re.I))
        and not bool(re.search(r"撰写|写作|写一篇|分析全文|详细介绍|深入分析|逐篇比较", q))
        and not bool(re.search(
            r"\d+\s*篇|[两二三四五六七八九十几多]+\s*篇|多篇|若干篇", q))
    )
    explicit = re.search(
        r"(\d+)\s*(tokens?|字|字符|词|句|句话|条|点|段|篇)(?:以内|左右|上下|最多|不超过|内)?",
        q, re.I)
    if explicit:
        amount = max(1, int(explicit.group(1)))
        unit = explicit.group(2).lower()
        factor = (1 if unit.startswith("token") else
                  1.7 if unit in ("字", "字符") else
                  2.2 if unit == "词" else
                  240 if unit in ("句", "句话") else
                  260 if unit in ("条", "点") else
                  320 if unit == "段" else 1400)
        target = max(256, int(amount * factor))
        guidance = (f"严格遵守用户指定的篇幅：约 {amount}{explicit.group(2)}以内；"
                    "只保留完成请求所需的信息。")
        if paper_recommendation:
            guidance += ("这是单篇现有论文推荐请求：只给一篇真实且匹配的论文，"
                         "列标题、作者、期刊/年份和实际 DOI/URL，不附综述。")
    else:
        long_form = (not paper_recommendation) and bool(re.search(
            r"深度研究|深入分析|详细(?:解释|说明|分析)?|全面(?:分析|梳理)?|"
            r"系统梳理|文献综述|综述|比较(?:分析)?|对比(?:分析)?|逐篇|逐项|"
            r"完整报告|长文|多章节|写一篇|撰写|comprehensive|in depth|detailed review",
            q, re.I))
        concise = bool(re.search(
            r"只要|只需|仅需|只给|简短|简要|简洁|一句话|直接给|不要展开|"
            r"无需解释|不必解释|简答|tl;?dr|briefly|in one sentence",
            q, re.I))
        if long_form:
            target = 12288
            guidance = "用户要求深入或长篇输出；按任务需要展开，但不要重复或凑章节。"
        elif paper_recommendation:
            target = 2048
            guidance = ("用户只要推荐一篇现有论文：只提供一篇最匹配且可核实的文献，"
                        "列出标题、作者、期刊/年份及实际 DOI 或 URL，最多用一句话说明相关性；"
                        "不要附领域综述或检索过程。")
        elif concise:
            target = 1024
            guidance = "用户要求简短或只要指定内容；直接给结果，不补充未请求的背景。"
        elif len(q) <= 120 and not re.search(r"为什么|如何|机制|比较|对比|分析|设计", q):
            target = 2048
            guidance = "按普通问答简洁直接回答；仅补充理解答案必需的信息。"
        else:
            target = 4096
            guidance = "默认简洁直接回答；只有解决问题所必需的细节才展开。"

    budget = next((b for b in _COMPLETION_TOKEN_BUCKETS if b >= target),
                  _COMPLETION_TOKEN_BUCKETS[-1])
    return budget, guidance


def _final_reserve_sec() -> float:
    """Scale synthesis time with the request deadline, with safe bounds."""
    request_sec = max(1.0, float(settings.AGENT_REQUEST_TIMEOUT_SEC))
    return min(_FINAL_RESERVE_MAX_SEC,
               max(_FINAL_RESERVE_MIN_SEC, request_sec * 0.30))


def _evidence_fingerprints(messages: list[ToolMessage]) -> set[str]:
    """Identify genuinely new tool output without relying on a fixed turn cap.

    URLs make repeated search results easy to recognize. A normalized content
    digest also catches local BM25 results and fetched pages without URLs.
    Error messages are deliberately excluded: retrying with a changed query
    after a transient failure can still make progress.
    """
    found: set[str] = set()
    for message in messages:
        content = str(getattr(message, "content", "") or "")
        if not content or content.startswith(("[WEB_TOOL_ERROR]", "[TOOL_ERROR]",
                                               "[TOOL_DUPLICATE_SKIPPED]", "[ERR_REQUEST_DEADLINE]")):
            continue
        found.update("url:" + url.rstrip(".,);]")
                     for url in re.findall(r"https?://[^\s<>\]\[\"']+", content))
        normalized = " ".join(content.split())
        if normalized:
            found.add("text:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest())
    return found


def _safe_persisted_trace(messages: list) -> list:
    """Keep local evidence in the live model context, not conversation history."""
    safe = []
    for message in messages:
        if isinstance(message, ToolMessage) and getattr(message, "name", "") in {
                "local_bm25_search", "local_dense_search", "local_hybrid_search"}:
            rows = ((getattr(message, "artifact", None) or {}).get("main_results") or [])
            safe.append(ToolMessage(
                content=f"本地检索已执行并读取 {len(rows)} 条去重候选；文献原文已从持久化轨迹中省略。",
                tool_call_id=message.tool_call_id, name=message.name, artifact={}))
        elif isinstance(message, ToolMessage) and getattr(message, "name", "") == "inspect_uploaded_table":
            # Keep only schema-level hints in session history; sample cell values
            # and previews are request-scoped and must not persist with the trace.
            try:
                profile = json.loads(str(message.content or "{}"))
                profile.pop("raw_preview", None)
                profile.pop("sample_rows", None)
                for field in profile.get("fields", []):
                    field.pop("samples", None)
                    for key in ("min", "median", "max"):
                        field.pop(key, None)
                safe.append(ToolMessage(
                    content=json.dumps(profile, ensure_ascii=False, default=str),
                    tool_call_id=message.tool_call_id, name=message.name, artifact={}))
            except Exception:
                safe.append(ToolMessage(
                    content="表格结构已在本轮检查；原始单元格样例未写入会话历史。",
                    tool_call_id=message.tool_call_id, name=message.name, artifact={}))
        elif isinstance(message, ToolMessage) and getattr(message, "name", "") == "audit_uploaded_table":
            # Keep the download capability but omit row-level findings from the
            # durable tool trace; the assistant's user-facing answer remains.
            links = re.findall(r"\[([^\]]+)\]\((/api/v2/session/[A-Za-z0-9_-]+/artifacts/[a-f0-9]{32})\)",
                               str(message.content or ""))
            link_text = "\n".join(f"- [{label}]({url})" for label, url in links)
            safe.append(ToolMessage(
                content="本轮数据审查已执行；行级发现仅保存在限时报告中，原始表格未修改。\n" + link_text,
                tool_call_id=message.tool_call_id, name=message.name, artifact={}))
        elif isinstance(message, HumanMessage) and isinstance(message.content, list):
            # Vision payloads are request-scoped; keep a trace placeholder, not
            # base64 image bytes, in session history.
            blocks = []
            for block in message.content:
                if isinstance(block, dict) and block.get("type") == "image_url":
                    blocks.append({"type": "text", "text": "[本轮图片附件已用于分析；图像数据未写入会话历史]"})
                else:
                    blocks.append(block)
            safe.append(HumanMessage(content=blocks, additional_kwargs=message.additional_kwargs))
        else:
            safe.append(message)
    return safe


def _parse_dsml_tool_calls(content: str, tools_by_name: dict) -> tuple[bool, list[dict]]:
    """Recover provider-emitted DSML text into calls from this request's allowlist.

    Some OpenAI-compatible endpoints return ``<｜｜DSML｜｜ invoke ...>`` as
    assistant text instead of populating ``AIMessage.tool_calls``. The loop must
    never treat that control text as a user-facing answer. Only fully parsed,
    schema-compatible calls to currently exposed tools are recoverable.
    """
    text = str(content or "")
    if not _DSML_PROTOCOL_RE.search(text):
        return False, []

    calls: list[dict] = []
    cursor = 0
    while True:
        opening = _DSML_INVOKE_OPEN_RE.search(text, cursor)
        if opening is None:
            break
        closing = _DSML_INVOKE_CLOSE_RE.search(text, opening.end())
        if closing is None:
            return True, []

        attrs = {k.lower(): html.unescape(v) for k, _, v in _DSML_ATTR_RE.findall(
            opening.group("attrs") or "")}
        name = attrs.get("name", "").strip()
        tool = tools_by_name.get(name)
        if tool is None:
            return True, []

        args: dict = {}
        block = text[opening.end():closing.start()]
        for parameter in _DSML_PARAMETER_RE.finditer(block):
            param_attrs = {k.lower(): html.unescape(v)
                           for k, _, v in _DSML_ATTR_RE.findall(
                               parameter.group("attrs") or "")}
            param_name = param_attrs.get("name", "").strip()
            if not param_name or param_name in args:
                return True, []
            value = html.unescape(parameter.group("value").strip())
            if param_attrs.get("string", "true").lower() != "true":
                try:
                    value = json.loads(value)
                except (TypeError, ValueError):
                    pass
            args[param_name] = value

        # Reject malformed or schema-incompatible calls before dispatch. The
        # normal tool executor still performs its own validation and sandboxing.
        input_model = getattr(tool, "args_schema", None)
        fields = (getattr(input_model, "model_fields", None)
                  or getattr(input_model, "__fields__", None))
        if fields:
            allowed_fields = set(fields)
            if not args or set(args) - allowed_fields:
                return True, []
            required = {
                field_name for field_name, field in fields.items()
                if (getattr(field, "is_required", lambda: False)()
                    if callable(getattr(field, "is_required", None))
                    else getattr(field, "required", False))
            }
            if required - set(args):
                return True, []

        calls.append({"name": name, "args": args,
                      "id": "dsml_" + uuid.uuid4().hex,
                      "type": "tool_call"})
        cursor = closing.end()

    return True, calls


def _sanitize_history_tool_protocol(messages: list) -> list:
    """Prevent previously persisted DSML leaks from being echoed on follow-ups."""
    cleaned = []
    for message in messages:
        if (isinstance(message, AIMessage)
                and not getattr(message, "tool_calls", None)
                and _DSML_PROTOCOL_RE.search(str(getattr(message, "content", "") or ""))):
            logger.warning("Removed unexecuted DSML tool-call text from conversation history")
            cleaned.append(AIMessage(content="[上一轮工具调用未被执行，因此没有形成有效回答。]"))
        else:
            cleaned.append(message)
    return cleaned


def _toolset():
    """Expose retrieval plus task-specific research workflow tools."""
    from src.tools import _TOOL_REGISTRY_BY_NAME

    allowed = {"deepseek_search", "deepseek_fetch", "deepseek_fetch_paper",
               "local_hybrid_search"}
    research_enabled = os.getenv("CITRUS_RESEARCH_TOOLS_ENABLED", "true").strip().lower() not in {
        "0", "false", "no", "off"}
    if research_enabled:
        allowed.update({
            "inspect_uploaded_table", "audit_uploaded_data",
            "verify_uploaded_references", "fetch_and_audit_open_supplements", "export_literature_evidence",
            "export_paper_analysis", "export_experiment_design", "save_research_note",
            "recall_research_notes", "export_research_notes", "export_corpus_gap_report",
        })
    return [tool for name, tool in _TOOL_REGISTRY_BY_NAME.items()
            if name in allowed], []


def _tool_call_dict(tc) -> dict:
    if isinstance(tc, dict):
        return {"id": tc.get("id", ""), "name": tc.get("name", ""),
                "args": dict(tc.get("args") or {})}
    return {"id": getattr(tc, "id", ""), "name": getattr(tc, "name", ""),
            "args": dict(getattr(tc, "args", None) or {})}


async def _run_regular_calls(calls: list[dict], tool_by_name: dict) -> list[ToolMessage]:
    """Run independent read-only tools concurrently with a small fan-out cap."""
    from src.tools.registry import _run_single_tool

    semaphore = asyncio.Semaphore(_MAX_PARALLEL_AGENT_TOOLS)

    async def run_one(call):
        tool = tool_by_name.get(call.get("name", ""))
        if tool is None:
            return ToolMessage(content="[ERR_UNKNOWN_TOOL] Tool is not available.",
                               tool_call_id=call.get("id", ""),
                               name=call.get("name", ""), artifact={})
        async with semaphore:
            return await _run_single_tool(tool, call)

    return await asyncio.gather(*(run_one(call) for call in calls))


async def _execute_calls(response, *, tools_by_name: dict,
                         deadline: float, seen_call_signatures: set[str]) -> list[ToolMessage]:
    calls = [_tool_call_dict(tc) for tc in (getattr(response, "tool_calls", None) or [])]
    fresh_calls = []
    completed: dict[str, ToolMessage] = {}
    for index, call in enumerate(calls):
        if index >= _MAX_TOOL_CALLS_PER_BATCH:
            completed[call["id"]] = ToolMessage(
                content=(f"[TOOL_BATCH_LIMIT] 本轮最多执行 {_MAX_TOOL_CALLS_PER_BATCH} 项工具；"
                         "此项未执行，请先综合已有结果。"),
                tool_call_id=call["id"], name=call["name"], artifact={})
            continue
        signature = hashlib.sha256(json.dumps(
            [call["name"], call.get("args") or {}], ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if signature in seen_call_signatures:
            completed[call["id"]] = ToolMessage(
                content="[TOOL_DUPLICATE_SKIPPED] This identical tool request was already attempted; use its earlier result or choose a new target.",
                tool_call_id=call["id"], name=call["name"], artifact={})
        else:
            seen_call_signatures.add(signature)
            fresh_calls.append(call)
        try:
            emit_tool_call_start(call["name"], call["args"], call["id"])
            state = "相同请求已跳过" if call["id"] in completed else "调用中"
            emit_tool_executing(f"{call['name']}：{state}", call["name"], call["id"])
        except Exception:
            pass

    if not fresh_calls:
        return [completed[c["id"]] for c in calls]
    remaining = max(0.01, deadline - time.monotonic() - _final_reserve_sec())
    try:
        results = await asyncio.wait_for(
            _run_regular_calls(fresh_calls, tools_by_name), timeout=remaining)
    except asyncio.TimeoutError:
        results = [ToolMessage(
            content="[ERR_REQUEST_DEADLINE] The request is nearing its time limit. Use evidence already received and answer now.",
            tool_call_id=call["id"], name=call["name"], artifact={}) for call in fresh_calls]
    except Exception as exc:  # noqa: BLE001
        logger.warning("Tool batch failed: %s", exc)
        results = [ToolMessage(content=f"[TOOL_ERROR] {type(exc).__name__}: {exc}",
                               tool_call_id=call["id"], name=call["name"], artifact={})
                   for call in fresh_calls]

    completed.update({str(message.tool_call_id): message for message in results})
    results = [completed[call["id"]] for call in calls]

    for call, message in zip(calls, results):
        try:
            if call["name"] in {"local_bm25_search", "local_dense_search", "local_hybrid_search"}:
                artifact = getattr(message, "artifact", None) or {}
                rows = artifact.get("main_results") or []
                families = call.get("args", {}).get("corpus_types") or []
                if call["name"] == "local_hybrid_search":
                    summary = (f"本地混合检索完成：类别 {', '.join(families)}；"
                               f"BM25 {artifact.get('bm25_candidates', 0)} + "
                               f"Dense {artifact.get('dense_candidates', 0)} → "
                               f"RRF {artifact.get('rrf_candidates', 0)} → "
                               f"重排后 {len(rows)} 条；{artifact.get('elapsed_ms', 0)} ms。")
                    if artifact.get("reranker_fallback"):
                        summary += " 重排降级为 RRF 顺序。"
                    summary += (f" 相邻分数衰减阈值 {artifact.get('adjacent_decay_ratio', 0.6):.2f}；"
                                f"剔除尾部 {artifact.get('decay_filtered', 0)} 条。")
                else:
                    summary = (f"本地检索完成：类别 {', '.join(families)}；"
                               f"命中 {len(rows)} 条记录；原文只进入模型上下文。")
                emit_tool_result(call["name"], summary, call["id"], summary=summary)
            else:
                emit_tool_result(call["name"], str(message.content), call["id"])
        except Exception:
            pass
    return results

async def _call_llm(llm, messages, *, timeout: float):
    # Buffer assistant text until the response is classified. Providers can
    # leak DSML control blocks through ``content``; streaming those chunks
    # before checking ``tool_calls`` would expose executable protocol as an
    # answer even if the loop later recovers it.
    return await asyncio.wait_for(
        stream_llm_response(llm, messages),
        timeout=max(0.1, timeout))


def _provider_failure_notice(exc: Exception) -> str:
    """Expose actionable provider failures without labeling ordinary timeouts as quota errors."""
    status = getattr(exc, "status_code", None) or getattr(exc, "http_status", None)
    pieces = [str(exc)]
    for attr in ("body", "message", "code", "type"):
        value = getattr(exc, attr, None)
        if value:
            pieces.append(str(value))
    response = getattr(exc, "response", None)
    if response is not None:
        status = status or getattr(response, "status_code", None)
        try:
            pieces.append(str(response.text))
        except Exception:
            pass
    detail = " ".join(pieces).lower()
    quota_markers = ("insufficient_quota", "quota_exceeded", "quota exceeded", "insufficient balance",
                     "balance is insufficient", "billing_hard_limit", "out of credits", "no credits",
                     "usagelimiterror", "usage limit exceeded", "monthly limit", "余额不足", "额度不足",
                     "欠费", "充值")
    if status == 402 or any(marker in detail for marker in quota_markers):
        return "本轮未完成：模型服务商返回额度或余额不足。请补充额度后重试。"
    if status == 429:
        return "本轮未完成：模型服务商触发请求频率限制（不是额度不足提示）。请稍后重试。"
    if status in (500, 502, 503, 504):
        return f"本轮未完成：模型服务商暂时不可用（HTTP {status}）；这不是额度不足结论，请稍后重试。"
    if status in (401, 403) or any(token in detail for token in ("invalid api key", "unauthorized", "authentication")):
        return "本轮未完成：模型服务商拒绝了认证。请检查所选服务商的 API Key。"
    if isinstance(exc, (asyncio.TimeoutError, TimeoutError)) or "timeout" in detail:
        return "本轮请求超时，未检测到额度不足错误；请稍后重试或缩小问题范围。"
    if any(token in detail for token in ("connectionerror", "connecterror", "connection refused", "name or service not known")):
        return "本轮未完成：无法连接模型服务商；没有收到额度不足响应，请检查网络或服务商状态。"
    return "本轮模型请求失败，服务商未返回明确的额度不足信息；请检查服务连接后重试。"


def _tool_failure_notice(messages: list) -> str:
    """Preserve a failed retrieval/tool status when the run expires before synthesis."""
    failed = []
    for message in messages:
        content = str(getattr(message, "content", "") or "")
        if content.startswith(("[WEB_TOOL_ERROR]", "[WEB_FETCH_FAILED]", "[LOCAL_SEARCH_ERROR]",
                               "[LOCAL_SEARCH_TRANSLATION_ERROR]", "[ERR_TIMEOUT]", "[ERR_NETWORK]")):
            failed.append(content)
    if not failed:
        return ""
    detail = " ".join(failed).lower()
    quota_markers = ("insufficient_quota", "quota_exceeded", "quota exceeded", "insufficient balance",
                     "out of credits", "no credits", "usage limit exceeded", "余额不足", "额度不足", "欠费")
    if any(marker in detail for marker in quota_markers) or "http 402" in detail:
        return "本轮未完成：检索或模型服务返回额度/余额不足。请补充额度后重试。"
    if "http 429" in detail or "rate limit" in detail:
        return "本轮未完成：检索或模型服务触发频率限制，不是额度不足提示；请稍后重试。"
    if "http 401" in detail or "http 403" in detail or "invalid api key" in detail or "unauthorized" in detail:
        return "本轮未完成：检索或模型服务认证失败；请管理员核对对应服务的 API Key。"
    if any(code in detail for code in ("http 500", "http 502", "http 503", "http 504")):
        return "本轮未完成：检索或模型服务暂时不可用；未将服务错误当作‘没有检索结果’。请稍后重试。"
    if any(token in detail for token in ("timeout", "timed out", "err_timeout", "request_deadline")):
        return "本轮未完成：检索或模型请求超时；未检测到额度不足响应。请稍后重试或缩小范围。"
    if any(token in detail for token in ("local_search", "bm25", "dense", "rerank")):
        return "本轮未完成：本地检索链路发生错误；这不等于本地语料没有命中。请稍后重试。"
    return "本轮未完成：联网检索或网页读取发生错误；这不等于没有相关网页结果。请稍后重试。"


async def general_agent_node(state: AgentState) -> dict:
    query = str(state.get("query") or "")
    session_id = str(state.get("session_id") or "")
    deadline = float(state.get("request_deadline") or
                     (time.monotonic() + max(1, settings.AGENT_REQUEST_TIMEOUT_SEC)))
    started = time.monotonic()
    scope = _source_scope(query)
    retrieval_disabled = _is_simple_query(query, scope)
    explicit_literature = _explicit_literature_request(query)
    explicit_reference_check = _explicit_reference_check(query)
    explicit_data_audit = _explicit_data_audit(query)
    try:
        from src.research.context import get_request_tables
        has_table_attachments = bool(get_request_tables())
    except Exception:
        has_table_attachments = False
    required_tool_groups: list[set[str]] = []
    if scope == "local":
        required_tool_groups.append({"local_hybrid_search"})
    elif scope in {"web", "fusion"}:
        required_tool_groups.append({"deepseek_search", "deepseek_fetch", "deepseek_fetch_paper"})
    elif scope == "auto" and (explicit_literature or explicit_reference_check):
        required_tool_groups.append({"deepseek_search"})
    local_literature_candidate = (
        explicit_literature and not explicit_reference_check and _local_corpus_matches(query))
    if scope == "fusion" and local_literature_candidate:
        required_tool_groups.append({"local_hybrid_search"})
    elif scope == "auto" and local_literature_candidate:
        # Local corpus is a complementary candidate source for explicit literature requests.
        required_tool_groups.append({"local_hybrid_search"})
    if explicit_reference_check and scope != "local":
        required_tool_groups.append({"verify_uploaded_references"})
    if explicit_data_audit and has_table_attachments:
        required_tool_groups.append({"audit_uploaded_data"})
    tools, _ = _scoped_toolset(scope)
    available_names = {tool.name for tool in tools}
    required_tool_groups = [group for group in required_tool_groups
                            if group & available_names]
    visualization_enabled = bool(state.get("visualization_enabled"))
    if visualization_enabled:
        from src.tools.draw_svg import draw_svg
        tools.append(draw_svg)
    image_attachments = state.get("image_attachments") or []
    disease_image_query = bool(image_attachments and re.search(
        r"病害|病症|什么病|识别.{0,6}病|诊断|症状|黄龙病|溃疡病|黑斑病|病斑|disease|diagnos|symptom",
        query, re.I))
    main_model_can_see_images = bool(image_attachments) and _main_model_supports_image_input()
    if retrieval_disabled:
        keep = {"draw_svg"}
        tools = [tool for tool in tools if tool.name in keep]
    tools_by_name = {tool.name: tool for tool in tools}
    system_prompt = assemble_system_prompt(mode="expert", format_hint=state.get("format_hint"),
                                           query=query)

    from src.core.context_manager import (
        assemble_supervisor_messages, build_human_message, build_loaded_context,
    )
    loaded = build_loaded_context(state, mode="expert", format_hint=state.get("format_hint"))
    human = build_human_message(loaded, retrieval_context="")
    completion_budget, answer_guidance = _answer_profile(query)
    if disease_image_query:
        completion_budget = min(completion_budget, 512)
        answer_guidance += (
            "\n病害图片问题请用极简用户格式回答：最多 3 行，依次写“最可能判断 + 置信度（高/中/低）”、"
            "“鉴别（仅 1 个关键点）”、以及“建议（仅在把握中低时给 1 条优先筛查）”。"
            "不要写长篇机制、多种诱因、候选排名、重复总结或多个检查清单。"
            "把握不足时明确说明不确定；不得声称确诊或给出农药用量。"
        )
    scope_guidance = {
        "local": "用户明确限定仅使用本地语料；不要调用联网工具。",
        "web": "用户明确限定仅使用联网来源；不要调用本地语料。",
        "fusion": "用户要求融合检索；联网优先，并且只有本地语料目录覆盖问题时才调用本地检索。",
        "auto": "按系统提示中的来源规则选择工具；简单稳定问题直接回答。",
    }.get(scope, "按系统提示中的来源规则选择工具。")
    request_policy = (
        "\n\n<request_policy>\n" + scope_guidance + "\n"
        + "系统提示中的工具说明是本轮任务路由的唯一文字规则；代码会强制执行用户指定的来源范围。\n"
        + ("可使用 draw_svg 生成图表；仅当图示能实质帮助理解时调用，并在验证后原样返回 SVG。\n"
           if visualization_enabled else "")
        + "</request_policy>"
    )
    human.content = (str(human.content or "")
                     + "\n\n<answer_length_and_scope>\n" + answer_guidance
                     + "\n</answer_length_and_scope>" + request_policy)
    attachment_context = str(state.get("attachment_context") or "").strip()
    if attachment_context:
        human.content += "\n\n" + attachment_context
    image_attachments = state.get("image_attachments") or []
    if image_attachments:
        image_scope = f"用户本轮上传了 {len(image_attachments)} 张图片；图片仅用于本轮，不写入知识库。"
        if disease_image_query:
            if main_model_can_see_images:
                vision_rules = (
                    "\n\n" + image_scope
                    + "你已收到原图，请由主模型直接判断最可能的病害或生理问题。"
                    + "回答最多 3 行：最可能判断及把握（高/中/低，非校准概率）；"
                    + "1 个关键鉴别点；仅在把握中低时给 1 条优先筛查建议。"
                    + "照片不足以判断时明确说不确定，并指出最有价值的一项补充观察或检测；"
                    + "不得声称确诊或给出农药用量。"
                )
            else:
                vision_rules = (
                    "\n\n" + image_scope
                    + "当前所选主模型不支持图片输入；不要声称看到了原图或给出图像诊断。"
                    + "请说明当前无法读取图片，并请用户补充症状描述或切换支持图像输入的主模型。"
                )
        elif main_model_can_see_images:
            vision_rules = "\n\n" + image_scope + "请按用户问题检查原图；不得仅凭照片断言品种。"
        else:
            vision_rules = (
                "\n\n" + image_scope
                + "当前所选主模型不支持图片输入，不要声称看到了原图；请依据文字和已有工具结果回答，"
                + "并请用户补充必要的图片描述。"
            )
        human.content += vision_rules
        text_block = {"type": "text", "text": str(human.content)}
        human.content = [text_block]
        if main_model_can_see_images:
            human.content.extend([
                {"type": "image_url", "image_url": {"url": image["url"]}}
                for image in image_attachments if image.get("url")
            ])
    messages, trace_start = assemble_supervisor_messages(system_prompt, state, human)
    messages = _sanitize_history_tool_protocol(messages)

    from src.core.llm_pool import get_llm
    llm_base = get_llm(
        model=get_deepseek_model(), api_key=settings.RESOLVED_MAIN_API_KEY,
        base_url=settings.RESOLVED_MAIN_BASE_URL,
        temperature=settings.TEMPERATURE_MAIN, max_tokens=completion_budget,
        timeout=min(120, max(5, settings.AGENT_REQUEST_TIMEOUT_SEC)))
    llm_tools = llm_base.bind_tools(tools) if tools else llm_base
    tool_names_called: list[str] = []
    seen_evidence: set[str] = set()
    stagnant_batches = 0
    tool_turns = 0
    main_results: list[dict] = []
    web_results: list[dict] = []
    answer = ""
    stop_reason = "model_finished"
    failure_notice = ""
    source_enforcement_attempts = 0
    protocol_recovery_attempts = 0
    seen_call_signatures: set[str] = set()

    try:
        emit_status("step_done", step_id="load")
        emit_status("step_active", step_id="retrieve")
    except Exception:
        pass

    while True:
        if tool_turns >= _MAX_MODEL_TOOL_TURNS:
            stop_reason = "tool_turn_limit"
            break
        final_reserve = _final_reserve_sec()
        remaining = deadline - time.monotonic()
        if remaining <= final_reserve:
            stop_reason = "request_deadline"
            break
        try:
            emit_status("agent_progress", message="正在分析当前问题与已有证据…")
            response = await _call_llm(
                llm_tools, messages, timeout=min(remaining - final_reserve, 120))
        except asyncio.TimeoutError:
            logger.warning("General Agent model call reached reserved finalization window")
            stop_reason = "model_timeout"
            failure_notice = _provider_failure_notice(asyncio.TimeoutError())
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("General Agent model call failed")
            stop_reason = f"model_error:{type(exc).__name__}"
            failure_notice = _provider_failure_notice(exc)
            break

        emit_llm_usage(session_id, "general_agent", response)
        calls = getattr(response, "tool_calls", None) or []
        if not calls:
            detected, recovered_calls = _parse_dsml_tool_calls(
                str(getattr(response, "content", "") or ""), tools_by_name)
            if recovered_calls:
                logger.warning("Recovered %d DSML tool call(s) emitted as assistant text",
                               len(recovered_calls))
                response = AIMessage(content="", tool_calls=recovered_calls)
                calls = recovered_calls
                emit_status("agent_progress",
                            message="已恢复模型返回的工具调用格式，继续执行检索…")
            elif detected:
                protocol_recovery_attempts += 1
                logger.error("Rejected unparseable or unavailable DSML tool-call text")
                messages.extend([
                    AIMessage(content="[上一轮返回的是未执行的工具协议，不是有效回答。]"),
                    HumanMessage(content=(
                        "上一轮输出了未执行的工具调用协议。不要复述或输出任何 DSML、XML 或工具调用标记。"
                        "请根据用户原始问题与已有证据继续：若当前工具列表含有合适工具，请通过标准结构化工具调用；"
                        "否则直接给出正常答案，并简要说明确实缺少的证据。")),
                ])
                if protocol_recovery_attempts < 2:
                    emit_status("agent_progress",
                                message="检测到未执行的工具协议，正在安全恢复回答…")
                    continue
                stop_reason = "invalid_tool_protocol"
                break

        messages.append(response)
        if not calls:
            missing_groups = [group for group in required_tool_groups
                              if not group.intersection(tool_names_called)]
            if (missing_groups and
                    source_enforcement_attempts < _MAX_SOURCE_ENFORCEMENT_ATTEMPTS and tools):
                source_enforcement_attempts += 1
                missing = [" / ".join(sorted(group)) for group in missing_groups]
                messages.append(HumanMessage(content=(
                    "用户明确要求的任务或来源尚未完成。请调用以下尚未尝试的工具组："
                    f"{'; '.join(missing)}。完成一次尝试后检查回执；若工具失败或无结果，"
                    "如实说明状态并继续回答，不要把失败写成没有相关资料。")))
                emit_status("agent_progress", message="正在补齐用户明确要求的检索或审查步骤…")
                continue
            answer = str(response.content or "")
            # No raw tool protocol is ever accepted as an answer, even if a
            # provider emits it during an unbound final response.
            if _DSML_PROTOCOL_RE.search(answer):
                logger.error("Blocked DSML tool-call text from final answer")
                answer = "本轮模型返回了未执行的工具调用格式，无法据此形成可靠回答。请重试。"
            else:
                emit_text(answer)
            break

        emit_status("agent_progress", message=f"已选择 {len(calls)} 项工具操作，开始执行…")

        tool_turns += 1
        results = await _execute_calls(response, tools_by_name=tools_by_name,
                                       deadline=deadline,
                                       seen_call_signatures=seen_call_signatures)
        messages.extend(results)
        emit_status("agent_progress", message="工具结果已并入上下文，正在判断是否还需补充…")
        new_evidence = _evidence_fingerprints(results) - seen_evidence
        seen_evidence.update(new_evidence)
        if new_evidence:
            stagnant_batches = 0
        else:
            stagnant_batches += 1
        for call, result in zip(calls, results):
            d = _tool_call_dict(call)
            if (not str(getattr(result, "content", "") or "").startswith("[TOOL_BATCH_LIMIT]")
                    and d["name"] not in tool_names_called):
                tool_names_called.append(d["name"])
            artifact = getattr(result, "artifact", None) or {}
            if isinstance(artifact, dict):
                main_results.extend(artifact.get("main_results") or [])
                web_results.extend(artifact.get("web_results") or [])

        if stagnant_batches >= _NO_PROGRESS_BATCHES_TO_STOP:
            stop_reason = "no_new_evidence"
            logger.info("General Agent stopped retrieval after %d consecutive tool batches "
                        "without new evidence", stagnant_batches)
            break

    # Reserve a final model pass without tools so the Agent can synthesize all
    # evidence already in its transcript, while keeping a strict total deadline.
    if not answer:
        remaining = max(0.0, deadline - time.monotonic() - 1.0)
        if remaining > 1:
            missing_groups = [group for group in required_tool_groups
                              if not group.intersection(tool_names_called)]
            missing_notice = ""
            if missing_groups:
                missing_names = [" / ".join(sorted(group)) for group in missing_groups]
                missing_notice = (
                    "\n明确要求的步骤仍未实际调用：" + "; ".join(missing_names) + ". "
                    "不得声称该步骤已完成，也不得把未执行说成无结果；请明确标记未完成。"
                )
            final_messages = messages + [HumanMessage(content=_FINAL_INSTRUCTION + missing_notice)]
            try:
                final_response = await _call_llm(
                    llm_base, final_messages, timeout=remaining)
                emit_llm_usage(session_id, "general_agent", final_response)
                final_text = str(final_response.content or "")
                if _DSML_PROTOCOL_RE.search(final_text):
                    logger.error("Blocked DSML tool-call text from final synthesis")
                    answer = "本轮模型返回了未执行的工具调用格式，无法据此形成可靠回答。请重试。"
                else:
                    answer = final_text
                if answer:
                    if answer == final_text:
                        messages.append(final_response)
                    emit_text(answer)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Final synthesis failed (%s): %s", stop_reason, exc)
                failure_notice = _provider_failure_notice(exc)
        if not answer:
            answer = failure_notice or _tool_failure_notice(messages) or (
                "本轮检索或生成超出请求时限，未检测到额度不足错误。请稍后重试，或缩小问题范围。")
    missing_groups = [group for group in required_tool_groups
                      if not group.intersection(tool_names_called)]
    if missing_groups and answer and not answer.startswith("本轮未完成"):
        missing_names = [" / ".join(sorted(group)) for group in missing_groups]
        answer = ("本轮未完成：以下明确要求的步骤未实际调用：" + "; ".join(missing_names) + "。\n\n" + answer)

    elapsed = (time.monotonic() - started) * 1000
    logger.info("General Agent finished: %d chars, %d tool types, %.0fms, stop=%s",
                len(answer), len(tool_names_called), elapsed, stop_reason)
    try:
        emit_status(
            "agent_progress",
            message=(f"检索摘要：来源模式 {scope}；本地类别检索按语料目录由模型选择；"
                     f"{'简单问题直接回答' if retrieval_disabled else '未检索类别不启动本地索引查询'}。"),
        )
    except Exception:
        pass
    return {
        "answer": answer,
        "gen_time_ms": elapsed,
        "main_results": main_results,
        "web_results": web_results,
        "tools_called": tool_names_called,
        "turn_trace": _safe_persisted_trace(messages[trace_start:]),
        "_trace": {"node": "supervisor", "elapsed_ms": elapsed,
                   "summary": f"{len(answer)} chars, {len(tool_names_called)} tool types, {stop_reason}",
                   "tool_turns": tool_turns,
                   "source_scope": scope,
                   "required_tools_unattempted": [sorted(group) for group in missing_groups],
                   "local_route": "called" if any(n.startswith("local_") for n in tool_names_called) else "not_called",
                   "local_tool_types": [n for n in tool_names_called if n.startswith("local_")],
                   "retrieval_disabled_simple_query": retrieval_disabled},
    }


def build_general_agent_graph():
    from src.graph.session_nodes import load_session_context, save_session_context

    graph = StateGraph(AgentState)
    graph.add_node("session_load", load_session_context)
    graph.add_node("supervisor", general_agent_node)
    graph.add_node("session_save", save_session_context)
    graph.set_entry_point("session_load")
    graph.add_edge("session_load", "supervisor")
    graph.add_edge("supervisor", "session_save")
    graph.add_edge("session_save", END)
    return graph.compile()
