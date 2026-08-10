"""Expert Graph — Supervisor Agent ReAct.

v8.3.0: structured SSE events (thinking, tool_call_start, tool_executing, tool_result, text).
"""
import asyncio
import json
import logging
import time
import uuid
from typing import AsyncIterator

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.config import settings, get_deepseek_model
from src.prompts.loader import assemble_system_prompt

logger = logging.getLogger(__name__)

from src.core.progress_bus import (
    emit_encoded, emit_thinking, emit_text,
    emit_tool_call_start, emit_tool_executing, emit_tool_result,
    emit_status, mark_tool_start, mark_tool_end,
)

SUPERVISOR_MAX_TURNS = 4

_AGENT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "call_retrieve_agent",
            "description": "Search academic literature for citrus research. "
                           "Use for factual queries, mechanisms, comparisons, reviews. "
                           "Query must be English keywords (5-15 words), NOT a full sentence. "
                           "If first search is insufficient, call again with synonyms or "
                           "narrower terms (up to 3 different angles).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "English search keywords (5-15 words)",
                    },
                    "goal": {
                        "type": "string",
                        "description": "What to retrieve and why",
                    },
                },
                "required": ["query", "goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_write_agent",
            "description": "Write or save content to a file. MUST call this when user asks to "
                           "'save', 'write', 'preserve', 'store' any content to a file — "
                           "even simple content like 'save 111 to test.md'. "
                           "Also use for academic reviews, reports, or structured answers. "
                           "For complex writing, ensure sufficient literature has been retrieved first. "
                           "context can be a finished document (will be saved directly) or "
                           "raw material to synthesize.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Writing goal",
                    },
                    "context": {
                        "type": "string",
                        "description": "Previous retrieval results or document content to base writing on",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "File path to save (only if user requested saving)",
                    },
                },
                "required": ["goal", "context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_analyze_agent",
            "description": "Perform statistical analysis or design experiments. "
                           "Call when user asks for data analysis, statistics, "
                           "or experiment planning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Analysis or experiment design goal",
                    },
                    "data_context": {
                        "type": "string",
                        "description": "Data or background for analysis",
                    },
                },
                "required": ["goal", "data_context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": "Read a local file from disk (.pdf, .md, .txt, .csv, .xlsx). "
                           "PDFs default to first 30000 chars (~8-12 pages). "
                           "For full PDF text, set max_chars=0. "
                           "Use FIRST when user asks to read/open/view a local file. "
                           "Absolute paths (e.g. E:/data/paper.pdf) work for files anywhere. "
                           "Relative paths resolve from workspace/. "
                           "For academic paper content extraction, prefer pdf_read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path (e.g. E:/docs/paper.pdf) or relative path (e.g. upload/data.csv)",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max chars to read (default 0 = auto: PDF 30000, others unlimited). Set 0 for full text.",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdf_read",
            "description": "Extract abstract and key sections from an academic research PDF. "
                           "Use for literature PAPERS found via DOI, search results, or online sources. "
                           "Do NOT use for local files (use read_local_file instead). "
                           "Suitable for extracting structured content like Abstract, Introduction, Methods, Results, Conclusion. "
                           "Returns plain text with section headers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path to the PDF paper (absolute or relative to workspace)",
                    },
                    "cross_reference": {
                        "type": "boolean",
                        "description": "Whether to validate metadata via CrossRef (default false)",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
]


def _make_tool_call(tc: dict) -> dict:
    name = tc.get("name", "")
    args = tc.get("args", {})
    return {"name": name, "args": args}


def _build_full_retrieval_context(main_results: list, web_results: list) -> str:
    parts = []
    for i, r in enumerate(main_results[:20]):
        parts.append(
            f"[{i+1}] {r.get('title', r.get('name', 'Untitled'))}\n"
            f"    Authors: {r.get('authors', 'N/A')}\n"
            f"    Year: {r.get('year', 'N/A')}  DOI: {r.get('doi', 'N/A')}\n"
            f"    Abstract: {str(r.get('abstract', r.get('snippet', '')))[:500]}"
        )
    for i, wr in enumerate(web_results[:10]):
        parts.append(
            f"[Web-{i+1}] {wr.get('title', wr.get('name', 'Untitled'))}\n"
            f"    URL: {wr.get('url', wr.get('link', 'N/A'))}\n"
            f"    Snippet: {str(wr.get('snippet', wr.get('content', '')))[:300]}"
        )
    return "\n\n".join(parts) if parts else ""


def _normalize_output_path(path: str) -> str:
    """Strip redundant output/ nesting and extract filename from absolute paths."""
    if not path or not path.strip():
        return ""
    path = path.strip().replace("\\", "/")
    # Remove redundant output/output nesting anywhere in the path
    while "/output/output/" in path:
        path = path.replace("/output/output/", "/output/")
    # If path is absolute and under workspace/output, extract just the filename
    if path.startswith("E:/") or path.startswith("D:/") or path.startswith("C:/"):
        if "/workspace/output/" in path:
            idx = path.rfind("/workspace/output/")
            rest = path[idx + len("/workspace/output/"):]
            path = rest.lstrip("/")
    # Strip leading output/ so relative paths work correctly
    while path.startswith("output/"):
        path = path[7:]
    return path


def _looks_like_document(text: str) -> bool:
    """Document structure heuristic: has heading/section markers, no raw-material features."""
    positive = ["# ", "## ", "### ", "摘要", "引言", "结论", "参考文献", "References", "关键词"]
    negative = ["检索结果:", "以下是相关文献:", "来源:", "chunk_id:", "confidence:"]
    has_pos = any(m in text for m in positive)
    has_neg = any(m in text for m in negative)
    return has_pos and not has_neg


async def _execute_tool_call(tc: dict, tc_id: str = "") -> dict:
    from src.core.agent_runner import run_agent

    name = tc.get("name", "")
    args = tc.get("args", {})

    agent_map = {
        "call_retrieve_agent": "retrieve-agent",
        "call_write_agent": "write-agent",
        "call_analyze_agent": "analyze-agent",
    }
    agent_name = agent_map.get(name)
    if not agent_name:
        return {"error": f"Unknown tool: {name}", "result": ""}

    task = {
        "goal": args.get("goal", ""),
        "query": args.get("query", ""),
        "output_path": _normalize_output_path(args.get("output_path", "")),
    }
    context = args.get("context", "") or args.get("data_context", "")
    skill_prompt = ""

    if agent_name == "write-agent":
        all_retrieved = args.get("_all_retrieved", "")
        if all_retrieved:
            context = (f"{context}\n\n=== 检索结果 ===\n{all_retrieved}" if context
                       else all_retrieved)
        try:
            from src.core.skill_tree import SkillTree
            st = SkillTree()
            query_for_skill = task.get("goal", task.get("query", ""))
            matches = st.search(query_for_skill, top_k=5)
            skill_parts = []
            for skill_id, score, meta in matches:
                content = st.load_content(skill_id)
                if content:
                    skill_parts.append(f"## Skill: {meta.get('name', skill_id)}\n\n{content}")
            if skill_parts:
                skill_prompt = "\n\n---\n\n".join(skill_parts)
                logger.info(f"[ExpertGraph] matched {len(skill_parts)} skills for write-agent")
        except Exception as e:
            logger.warning(f"[ExpertGraph] skill match failed: {e}")

    try:
        emit_status("agent_start", agent=agent_name[:20])
        emit_tool_executing(f"启动子Agent: {agent_name}...", agent_name, tc_id)
    except Exception:
        pass
    try:
        t_agent = time.perf_counter()
        # Direct-write shortcut: context is a finished document (structure markers, no raw material)
        if (agent_name == "write-agent" and len(context) > 2000
                and task.get("output_path") and _looks_like_document(context)):
            from src.tools.file_ops import write_local_file
            save_path = task["output_path"]
            save_msg = write_local_file.func(save_path, context, "write")
            logger.info(f"[ExpertGraph] direct write (context={len(context)}chars): {save_msg}")
            result = {"agent": "write-agent", "result": f"已保存到 {save_path}", "artifacts": {},
                      "tools_called": 0, "file_saved": True}
        else:
            result = await run_agent(
                agent_name, task, context=context,
                system_prompt_extra=skill_prompt, timeout_sec=120,
            )
        dt_agent = (time.perf_counter() - t_agent) * 1000
    except Exception as e:
        logger.error(f"[ExpertGraph] sub-agent {agent_name} failed: {e}")
        return {"agent": agent_name, "result": f"[Error: {e}]", "artifacts": {}}

    if agent_name == "write-agent":
        output_path = task.get("output_path", "")
        answer_text = result.get("result", "")
        file_saved = result.get("file_saved", False)
        if output_path and answer_text and len(answer_text) > 50 and not file_saved:
            try:
                from src.tools.file_ops import write_local_file
                save_msg = write_local_file.func(output_path, answer_text)
                logger.info(f"[ExpertGraph] forced save: {save_msg}")
                result["result"] = f"[已保存到 {output_path}]\n\n{answer_text}"
            except Exception as e:
                logger.warning(f"[ExpertGraph] forced save failed: {e}")

    logger.info(
        f"[ExpertGraph] {agent_name} -> {len(result.get('result', ''))} chars, "
        f"{result.get('tools_called', 0)} tools"
    )
    try:
        emit_status("agent_done", agent=agent_name[:20],
                     tools_called=result.get("tools_called", 0))
    except Exception:
        pass
    return result


async def expert_load_node(state: AgentState) -> dict:
    query = state.get("query", "")
    session_id = state.get("session_id", "default")
    mode = state.get("mode", "expert")
    logger.info(f"[ExpertGraph:load] session={session_id[:8]}...")

    result: dict = {}
    try:
        from src.session.manager import session_manager
        from src.core.context_budget import ContextBudget, ContextBudgetConfig
        from src.core.context_manager import ContextManager
        from src.guardrails.memory import memory_store

        budget_config = ContextBudgetConfig(
            max_tokens=settings.CONTEXT_BUDGET_MAX_TOKENS,
            soft_threshold=settings.CONTEXT_BUDGET_SOFT_THRESHOLD,
            hard_threshold=settings.CONTEXT_BUDGET_HARD_THRESHOLD,
        )
        budget = ContextBudget(budget_config)

        ctx_mgr = ContextManager(
            session_manager=session_manager,
            memory_store=memory_store,
            budget=budget,
        )

        ctx = await ctx_mgr.load(session_id, query, mode)
        result["session_id"] = ctx.session_id
        result["history_summary"] = ctx.history_summary
        result["long_term_memory"] = ctx.long_term_memory
        result["search_suggestions"] = ctx.search_suggestions
        result["format_hint"] = ctx.format_hint

        history_msgs = list(ctx.history_messages) if ctx.history_messages else []
        if history_msgs:
            result["messages"] = history_msgs

        load_summary = f"history={len(history_msgs)}msgs"
        if ctx.format_hint:
            load_summary += f", fmt={ctx.format_hint}"
        result["_trace"] = {"node": "expert_load", "elapsed_ms": 0, "summary": load_summary}

        hist_chars = sum(len(m.content or "") for m in history_msgs if hasattr(m, 'content'))
        est_tokens = budget.estimate_tokens(history_msgs)
        try:
            emit_encoded("context_status", {
                "history_msgs": len(history_msgs),
                "history_chars": hist_chars,
                "ltm_recalled": bool(ctx.long_term_memory),
                "ltm_chars": len(ctx.long_term_memory or ""),
                "suggestions": ctx.search_suggestions[:3] if ctx.search_suggestions else [],
                "format_hint": ctx.format_hint or "",
                "estimated_tokens": est_tokens,
                "max_tokens": budget_config.max_tokens,
                "soft_threshold": budget_config.soft_threshold,
                "hard_threshold": budget_config.hard_threshold,
                "compressed": bool(ctx.history_summary),
                "compression_len": len(ctx.history_summary or ""),
            })
        except Exception:
            pass
        return result

    except Exception as e:
        logger.warning(f"[ExpertGraph:load] failed: {e}")
        result["_trace"] = {"node": "expert_load", "elapsed_ms": 0, "summary": "unavailable"}
        return result


async def supervisor_node(state: AgentState) -> dict:
    query = state.get("query", "")
    format_hint = state.get("format_hint")
    logger.info(f"[ExpertGraph:supervisor] query={query[:60]}...")

    system_prompt = assemble_system_prompt(
        mode="expert",
        format_hint=format_hint,
        query=query,
    )

    from src.core.context_manager import LoadedContext, build_human_message
    ctx = LoadedContext(
        session_id=state.get("session_id", ""),
        mode="expert",
        query=query,
        history_summary=state.get("history_summary"),
        long_term_memory=state.get("long_term_memory"),
        search_suggestions=state.get("search_suggestions", []),
        format_hint=format_hint,
    )
    current_human = build_human_message(ctx)

    messages: list = [SystemMessage(content=system_prompt)]
    history_msgs = list(state.get("messages", []))
    if history_msgs:
        messages.extend(history_msgs)
    messages.append(current_human)

    llm_base = ChatOpenAI(
        model=get_deepseek_model(),
        api_key=settings.RESOLVED_MAIN_API_KEY,
        base_url=settings.MAIN_BASE_URL,
        temperature=settings.TEMPERATURE_MAIN,
        max_tokens=32768,
        timeout=120,
    )
    llm_with_tools = llm_base.bind_tools(_AGENT_TOOLS)

    answer = ""
    all_main_results = []
    all_web_results = []
    tool_call_count = 0
    t0 = time.perf_counter()

    try:
        emit_status("step_done", step_id="load")
        emit_status("step_active", step_id="retrieve")
    except Exception:
        pass

    try:
        for turn in range(SUPERVISOR_MAX_TURNS):
            t_llm = time.perf_counter()
            for attempt in range(3):
                try:
                    response = await llm_with_tools.ainvoke(messages)
                    break
                except Exception as e:
                    if attempt < 2:
                        logger.warning(f"[ExpertGraph:supervisor] LLM retry {attempt+1}/3: {e}")
                        await asyncio.sleep(3)
                    else:
                        raise
            dt_llm = (time.perf_counter() - t_llm) * 1000
            messages.append(response)

            if not getattr(response, "tool_calls", None):
                answer = response.content or ""
                logger.info(
                    f"[ExpertGraph:supervisor] turn{turn} done: "
                    f"{len(answer)}c / {dt_llm:.0f}ms"
                )
                break

            # Emit thinking (LLM's internal reasoning before calling tools)
            if response.content:
                try:
                    emit_thinking(response.content[:800])
                except Exception:
                    pass

            for tc in response.tool_calls:
                tc_dict = _make_tool_call(tc)
                tc_id = getattr(tc, "id", str(uuid.uuid4()))
                if tc_dict["name"] == "call_write_agent":
                    full_ctx = _build_full_retrieval_context(all_main_results, all_web_results)
                    if full_ctx:
                        tc_dict["args"]["_all_retrieved"] = full_ctx
                logger.info(
                    f"[ExpertGraph:supervisor] turn{turn}: "
                    f"call {tc_dict['name']}({str(tc_dict['args'])[:80]})"
                )
                try:
                    emit_tool_call_start(tc_dict["name"], tc_dict.get("args", {}), tc_id)
                    emit_tool_executing(f"正在执行 {tc_dict['name']}...", tc_dict["name"], tc_id)
                    if tc_dict["name"] in ("call_retrieve_agent", "call_write_agent", "call_analyze_agent"):
                        agent_map_short = {
                            "call_retrieve_agent": "retrieve-agent",
                            "call_write_agent": "write-agent",
                            "call_analyze_agent": "analyze-agent",
                        }
                        aname = agent_map_short.get(tc_dict["name"], "agent")
                        emit_encoded("agent_switch", {
                            "agent_name": aname,
                            "task": tc_dict["args"].get("goal", "")[:80],
                            "turn": turn + 1,
                        })
                except Exception:
                    pass

                if tc_dict["name"] == "read_local_file":
                    try:
                        mark_tool_start(tc_id, "read_local_file")
                    except Exception:
                        pass
                    from src.tools.readfile import read_local_file
                    content = await read_local_file.ainvoke(tc_dict["args"])
                    sub_result = {"agent": "read_local_file", "result": content or "", "artifacts": {}}
                    try:
                        emit_tool_result("read_local_file", str(content)[:100000], tc_id,
                                         summary=f"读取完成 ({len(str(content))} 字符)")
                    except Exception:
                        pass
                elif tc_dict["name"] == "pdf_read":
                    try:
                        mark_tool_start(tc_id, "pdf_read")
                    except Exception:
                        pass
                    from src.tools.search import pdf_read as pdf_read_func
                    file_path = tc_dict["args"].get("file_path", "")
                    content, artifact = await asyncio.to_thread(pdf_read_func.func, file_path, False)
                    sub_result = {"agent": "pdf_read", "result": content or "", "artifacts": artifact or {}}
                    try:
                        emit_tool_result("pdf_read", str(content)[:100000], tc_id,
                                         summary=f"文献提取完成 ({len(str(content))} 字符)")
                    except Exception:
                        pass
                else:
                    sub_result = await _execute_tool_call(tc_dict, tc_id)
                    try:
                        emit_tool_result(
                            tc_dict["name"],
                            sub_result.get("result", "")[:100000],
                            tc_id,
                            summary=(
                                f"子Agent {sub_result.get('agent', '?')} 完成, "
                                f"{sub_result.get('tools_called', 0)} 次工具调用"
                            ),
                        )
                    except Exception:
                        pass

                artifacts = sub_result.get("artifacts", {}) or {}
                all_main_results.extend(artifacts.get("main_results", []))
                all_web_results.extend(artifacts.get("web_results", []))
                tool_call_count += 1

                caps = getattr(settings, "TOOL_RESULT_CAPS", {}) or {}
                agent_display = sub_result.get("agent", "?")
                cap = caps.get(agent_display, caps.get("default", 100000))
                result_text = sub_result.get("result", "") or ""
                truncated = result_text[:cap]
                if len(result_text) > cap:
                    truncated += "\n\n[... 已截断 ...]"
                    logger.warning(
                        f"[ExpertGraph] truncated {agent_display} result "
                        f"{len(result_text)} chars > cap {cap}"
                    )
                tool_msg_content = f"[{agent_display} result]\n{truncated}"
                messages.append(ToolMessage(
                    content=tool_msg_content,
                    tool_call_id=tc.get("id", "unknown"),
                    name=tc_dict["name"],
                ))

            logger.info(
                f"[ExpertGraph:supervisor] turn{turn}: "
                f"LLM {dt_llm:.0f}ms "
                f"(total {time.perf_counter()-t0:.1f}s)"
            )

        else:
            logger.info("[ExpertGraph:supervisor] max turns, forcing final")
            messages.append(HumanMessage(content=(
                "You have reached the maximum number of turns. "
                "Do NOT call any more tools. "
                "Provide your final answer now. "
                "If information is insufficient, explain what is missing."
            )))
            final_resp = await llm_base.ainvoke(messages)
            answer = final_resp.content or ""

    except Exception as e:
        logger.error(f"[ExpertGraph:supervisor] error: {e}")
        answer = f"An error occurred: {e}"

    elapsed = (time.perf_counter() - t0) * 1000

    seen_doi = set()
    deduped_main = []
    for r in all_main_results:
        doi = (r.get("doi") or "").strip()
        if doi and doi in seen_doi:
            continue
        seen_doi.add(doi)
        deduped_main.append(r)

    cited_refs = []
    for i, r in enumerate(deduped_main[:20]):
        cited_refs.append({
            "ref_id": i + 1,
            "type": "main",
            "doi": r.get("doi", "N/A"),
            "title": r.get("title", r.get("name", "Untitled")),
            "section_name": r.get("section_name", ""),
            "text_preview": (r.get("abstract") or r.get("snippet") or "")[:300],
            "score": r.get("score", r.get("rerank_score", 0)) or 0,
            "year": r.get("year", ""),
            "authors": r.get("authors", ""),
        })
    for i, wr in enumerate(all_web_results[:10]):
        cited_refs.append({
            "ref_id": f"W{i+1}",
            "type": "web",
            "url": wr.get("url", wr.get("link", "")),
            "title": wr.get("title", wr.get("name", "Untitled")),
            "text_preview": (wr.get("snippet") or wr.get("content") or "")[:300],
            "score": 0,
        })

    references_data = {
        "cited": cited_refs,
        "uncited": [],
        "total": len(cited_refs),
    }

    if answer:
        try:
            emit_status("step_done", step_id="retrieve")
            emit_status("step_done", step_id="supervise")
            emit_status("step_active", step_id="answer")
            emit_status("final_answer", message="生成最终答案...")
        except Exception:
            pass
        chunk_size = 8
        for i in range(0, len(answer), chunk_size):
            chunk = answer[i:i + chunk_size]
            try:
                emit_text(chunk)
            except Exception:
                pass
            await asyncio.sleep(0.012)
        try:
            emit_status("step_done", step_id="answer")
        except Exception:
            pass

    logger.info(
        f"[ExpertGraph:supervisor] done: {len(answer)}c, "
        f"{tool_call_count} tool calls, "
        f"{len(deduped_main)} main + {len(all_web_results)} web results, "
        f"{elapsed:.0f}ms"
    )

    return {
        "answer": answer,
        "gen_time_ms": elapsed,
        "main_results": deduped_main[:20],
        "web_results": all_web_results[:10],
        "references_data": references_data,
        "_trace": {
            "node": "supervisor",
            "elapsed_ms": elapsed,
            "summary": f"{len(answer)} chars, {tool_call_count} tools",
        },
    }


async def expert_save_node(state: AgentState) -> dict:
    query = state.get("query", "")
    answer = state.get("answer", "")
    session_id = state.get("session_id", "default")
    if not answer:
        return {"_trace": {"node": "save", "elapsed_ms": 0, "summary": "no answer"}}

    try:
        from langchain_core.messages import HumanMessage, AIMessage
        from src.session.manager import session_manager
        msgs_to_save = [
            HumanMessage(content=query),
            AIMessage(content=answer),
        ]
        asyncio.create_task(session_manager.save_messages(session_id, msgs_to_save))
    except Exception as e:
        logger.warning(f"[ExpertGraph:save] failed: {e}")

    try:
        from src.guardrails.memory import memory_store
        if len(answer) > 500:
            is_substantial = any(
                kw in answer
                for kw in ("###", "结论", "摘要", "引言", "核心结论", "局限与边界")
            )
            if is_substantial:
                facts = await asyncio.to_thread(memory_store.extract_key_facts, query, answer)
                for f in facts:
                    memory_store.save_long_term_fact(
                        f.get("key", ""),
                        f.get("value", ""),
                        f.get("confidence", 0.5),
                        owner_session=session_id,
                        source_query=query,
                    )
    except Exception as e:
        logger.debug(f"[ExpertGraph:save] LTM skip: {e}")

    return {"_trace": {"node": "save", "elapsed_ms": 0, "summary": "saved"}}


def build_expert_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("expert_load", expert_load_node)
    graph.add_node("supervisor", supervisor_node)
    graph.add_node("expert_save", expert_save_node)

    graph.set_entry_point("expert_load")
    graph.add_edge("expert_load", "supervisor")
    graph.add_edge("supervisor", "expert_save")
    graph.add_edge("expert_save", END)

    return graph.compile()
