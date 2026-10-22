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
import re
import time
import uuid

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, StateGraph

from src.config import get_deepseek_model, settings
from src.core.agent_loop import emit_llm_usage
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
    r"recent|latest|paper|literature|study|evidence|source|verify|search|"
    r"mechanism|process|compare|design|data|doi|pmid", re.I)


def _source_scope(query: str) -> str:
    """Respect explicit user source constraints before exposing tools."""
    q = str(query or "").lower()
    if re.search(r"(?:融合搜索|融合检索|本地.{0,8}联网).{0,8}(?:是什么|什么意思|含义|如何理解)", q):
        return "auto"
    if re.search(r"(?:只|仅)(?:(?:查|用|看|进行|搜索|检索)(?:一下|一次)?)?(?:本地|本地库|知识库)"
                 r"|(?:只|仅)(?:在|从)(?:本地|本地库|知识库)(?:查|检索|搜索)"
                 r"|不要联网|不许联网|仅限本地", q):
        return "local"
    if re.search(r"(?:只|仅)(?:(?:查|用|看|进行|搜索|检索)(?:一下|一次)?)?(?:联网|网络|网页)"
                 r"|(?:只|仅)(?:在|从)(?:联网|网络|网页)(?:查|检索|搜索)"
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


def _is_simple_query(query: str, scope: str) -> bool:
    if scope != "auto":
        return False
    q = str(query or "").strip()
    if not q or _RETRIEVAL_CUES.search(q):
        return False
    return len(q) <= 48 and not re.search(r"[?？].*[?？]|以及|并且|分别|综述|全面", q)


def _scoped_toolset(scope: str):
    tools, _schemas = _toolset()
    if scope == "local":
        tools = [t for t in tools if t.name.startswith("local_")]
    elif scope == "web":
        tools = [t for t in tools if t.name.startswith("deepseek_")]
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
    """Expose only web research and local knowledge-retrieval tools."""
    from src.tools import _TOOL_REGISTRY_BY_NAME

    allowed = {"deepseek_search", "deepseek_fetch", "deepseek_fetch_paper",
               "local_hybrid_search"}
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
    for call in calls:
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


async def general_agent_node(state: AgentState) -> dict:
    query = str(state.get("query") or "")
    session_id = str(state.get("session_id") or "")
    deadline = float(state.get("request_deadline") or
                     (time.monotonic() + max(1, settings.AGENT_REQUEST_TIMEOUT_SEC)))
    started = time.monotonic()
    scope = _source_scope(query)
    retrieval_disabled = _is_simple_query(query, scope)
    tools, _ = _scoped_toolset(scope)
    visualization_enabled = bool(state.get("visualization_enabled"))
    if visualization_enabled:
        from src.tools.draw_svg import draw_svg
        tools.append(draw_svg)
    if retrieval_disabled:
        tools = [tool for tool in tools if tool.name == "draw_svg"]
    tools_by_name = {tool.name: tool for tool in tools}
    system_prompt = assemble_system_prompt(mode="expert", format_hint=state.get("format_hint"),
                                           query=query)

    from src.core.context_manager import (
        assemble_supervisor_messages, build_human_message, build_loaded_context,
    )
    loaded = build_loaded_context(state, mode="expert", format_hint=state.get("format_hint"))
    human = build_human_message(loaded, retrieval_context="")
    completion_budget, answer_guidance = _answer_profile(query)
    human.content = (str(human.content or "") +
                     "\n\n<answer_length_and_scope>\n" + answer_guidance +
                     "\n</answer_length_and_scope>" +
                     "\n\n<source_execution_policy>\n" +
                     (f"User explicitly requested source scope={scope}. Use only that source.\n"
                      if scope in {"local", "web"} else
                     "User requested fusion. Use web evidence and consult local evidence only if the corpus catalog below covers the requested fact type. If no listed category applies, skip local search and record that it is out of corpus scope.\n"
                      if scope == "fusion" else
                      "Choose sources based on the question. Do not retrieve for a simple stable question.\n") +
                     "Local corpus catalog (choose only a matching family; do not search the whole corpus):\n"
                     "- literature: classified citrus literature: pathology/diseases and resistance; genes/genomics/evolution; breeding/biotechnology; physiology/metabolism; cultivation/agroecology; nutrition/postharvest. Paper metadata usually includes DOI.\n"
                     "- ucr_varieties: UCR citrus variety records, including cultivar names, parentage, origin, traits, and registry fields; typically no DOI.\n"
                     "- deng_variety_book: digitized records from Deng Xiuxin, Chinese Citrus Varieties; taxonomy and variety descriptions; no DOI.\n"
                     "- experiment_methods: CGI experimental methods such as transformation, extraction, reporter assays, and inoculation; no DOI.\n"
                     "Literature l1_categories values, when useful: agriculture/ecology, pathology/resistance, physiology/metabolism, nutrition/health, breeding/biotechnology, genomics/evolution.\n"
                     "If no family covers the requested detail, do not call local_hybrid_search. If relevant, choose corpus_types explicitly; local_hybrid_search runs BM25 and dense retrieval in parallel, merges by RRF, reranks candidates, and returns the selected original passages. For Chinese questions about English literature, make query a concise English gap/title/DOI expression. Web failure with a known DOI/title is a reason to check only the literature family.\n"
                     "Never put private local material into Search or Fetch arguments.\n"
                     + ("Visualization is enabled for this request. Use draw_svg only when a diagram materially clarifies the requested answer; do not add a chart for simple requests. Use only facts and numbers supported by the current answer/evidence, label illustrative examples as illustrative, and after validation include the returned fenced SVG block verbatim in your final answer.\n"
                        if visualization_enabled else "") +
                     "</source_execution_policy>")
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
    main_results: list[dict] = []
    web_results: list[dict] = []
    answer = ""
    stop_reason = "model_finished"
    source_enforcement_attempts = 0
    protocol_recovery_attempts = 0
    seen_call_signatures: set[str] = set()

    try:
        emit_status("step_done", step_id="load")
        emit_status("step_active", step_id="retrieve")
    except Exception:
        pass

    while True:
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
            break
        except Exception as exc:  # noqa: BLE001
            logger.exception("General Agent model call failed")
            stop_reason = f"model_error:{type(exc).__name__}"
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
            used_web = any(n in tool_names_called for n in ("deepseek_search", "deepseek_fetch", "deepseek_fetch_paper"))
            source_requirements_met = (
                scope in {"auto", "local"} or
                (scope == "web" and used_web) or
                # Fusion explicitly requires web; the model decides whether any
                # listed local corpus family can contribute to this question.
                (scope == "fusion" and used_web)
            )
            if (not source_requirements_met and source_enforcement_attempts < 2 and tools):
                source_enforcement_attempts += 1
                missing = ("联网来源" if scope == "web" and not used_web else
                           "联网搜索" if scope == "fusion" and not used_web else "指定来源")
                messages.append(HumanMessage(content=(
                    f"用户明确要求的来源范围尚未满足：还没有实际调用{missing}。"
                    "请先调用相应检索工具，检查返回内容后再回答。若工具失败，"
                    "将失败回执作为来源尝试记录，再基于可用证据回答。")))
                emit_status("agent_progress", message=f"正在补齐用户指定的来源：{missing}…")
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
            if d["name"] not in tool_names_called:
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
            final_messages = messages + [HumanMessage(content=_FINAL_INSTRUCTION)]
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
        if not answer:
            answer = "本轮检索或生成未能在请求时限内完成。请稍后重试，或缩小问题范围。"

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
                   "source_scope": scope,
                   "local_route": "called" if any(n.startswith("local_") for n in tool_names_called) else "not_called",
                   "local_tool_types": [n for n in tool_names_called if n.startswith("local_")],
                   "retrieval_disabled_simple_query": retrieval_disabled},
    }


def build_general_agent_graph():
    from src.graph.expert_graph import expert_load_node, expert_save_node

    graph = StateGraph(AgentState)
    graph.add_node("expert_load", expert_load_node)
    graph.add_node("supervisor", general_agent_node)
    graph.add_node("expert_save", expert_save_node)
    graph.set_entry_point("expert_load")
    graph.add_edge("expert_load", "supervisor")
    graph.add_edge("supervisor", "expert_save")
    graph.add_edge("expert_save", END)
    return graph.compile()
