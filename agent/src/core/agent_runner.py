"""Agent Runner — 统一 ReAct 子 Agent 执行器.

v8.3.0: structured SSE events during tool execution (tool_call_start,
tool_executing, tool_result) with content and progress tracking.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from typing import Optional

from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
)
from langchain_openai import ChatOpenAI

from src.config import settings, get_deepseek_model
from src.prompts.loader import assemble_agent_prompt
from src.tools import _TOOL_REGISTRY_BY_NAME
from src.tools.registry import PartitionedToolNode

logger = logging.getLogger(__name__)

from src.core.progress_bus import (
    emit_encoded, emit_thinking,
    emit_tool_call_start, emit_tool_executing, emit_tool_result,
    emit_usage_delta,
)


def _truncate_context_blocks(context: str, max_chars: int = 24000) -> str:
    """按完整条目截断检索证据，避免把单篇文献切尾 (AG-8)。

    v8.3.5 截断透明 (规范 4.2.5 参数保真): 截断时显式标记缺失内容，
    模型不得误以为看到了全部上下文。
    """
    if len(context) <= max_chars:
        return context
    blocks = context.split("\n\n")
    out, total = [], 0
    for b in blocks:
        if total + len(b) + 2 > max_chars:
            break
        out.append(b)
        total += len(b) + 2
    if not out:
        return context[:max_chars] + \
            f"\n\n[上下文已截断: 原始 {len(context)} 字符，仅显示前 {max_chars} 字符]"
    return "\n\n".join(out) + \
        f"\n\n[上下文已截断: 共 {len(blocks)} 条内容，仅显示前 {len(out)} 条，其余未提供]"


def _count_unique_docs(main_results: list) -> int:
    """按 DOI 去重计数文献（无 DOI 按条计数）— v8.3.4 收敛判断用。"""
    seen = set()
    n = 0
    for r in main_results:
        doi = (r.get("doi") or "").strip().lower()
        if doi:
            if doi in seen:
                continue
            seen.add(doi)
        n += 1
    return n


def _tc_id(tc) -> str:
    """兼容 dict/对象两种 tool_calls 形态取 id（v8.3.5 id 单一来源）。"""
    if isinstance(tc, dict):
        return tc.get("id", "") or str(uuid.uuid4())
    return getattr(tc, "id", "") or str(uuid.uuid4())


async def run_agent(
    agent_name: str,
    task: dict,
    *,
    context: str = "",
    system_prompt_extra: str = "",
    skills: Optional[list[str]] = None,
    timeout_sec: int = 120,
    session_id: str = "",
) -> dict:
    """Run a sub-agent in ReAct loop. Returns result dict.

    Args:
        agent_name: "retrieve-agent" | "write-agent" | "analyze-agent"
        task: {"goal": "...", "query": "...", "output_path": "..."}
        context: upstream dependency results
        system_prompt_extra: additional instructions appended to system prompt
        skills: skill IDs for write-agent
        timeout_sec: LLM call timeout
        session_id: 会话 ID（context_usage 增量追踪用）

    Returns:
        {"agent": str, "result": str, "artifacts": dict, "tools_called": int, "turns": int}
    """
    try:
        system_content = assemble_agent_prompt(agent_name, skills=skills)
    except ValueError:
        system_content = f"You are {agent_name}. Complete the assigned task."
    except Exception as e:
        logger.warning(f"[AgentRunner] prompt assembly failed: {e}")
        system_content = f"You are {agent_name}."

    if system_prompt_extra:
        system_content += f"\n\n---\n{system_prompt_extra}"

    tool_names = _resolve_tool_names(agent_name)
    tools = [_TOOL_REGISTRY_BY_NAME[n] for n in tool_names if n in _TOOL_REGISTRY_BY_NAME]

    goal = task.get("goal", "")
    query = task.get("query", "")
    output_path = task.get("output_path", "")

    human_content = (
        f"<task_goal>\n{goal}\n</task_goal>\n\n"
        f"<task_query>\n{query}\n</task_query>\n\n"
    )
    if output_path:
        human_content += (
            f"<output_path>\n{output_path}\n</output_path>\n"
            f"IMPORTANT: You MUST call write_local_file to save the final output to this path.\n\n"
        )
    if context:
        human_content += (
            f"<context>\n{_truncate_context_blocks(context)}\n</context>\n\n"
        )
    human_content += "Please complete the task. Output the final result when done."

    messages: list = [
        SystemMessage(content=system_content),
        HumanMessage(content=human_content),
    ]

    # v8.3.1: write-agent 单轮输出上限 12000（约 1-2 章节），防止 32768 硬截断浪费；
    # 配合 prompt"每轮 1-2 章节、后续 append 续写"，6 轮上限覆盖完整综述
    max_t = 12000 if agent_name == "write-agent" else settings.MAX_TOKENS
    llm_base = ChatOpenAI(
        model=get_deepseek_model(),
        api_key=settings.RESOLVED_MAIN_API_KEY,
        base_url=settings.MAIN_BASE_URL,
        temperature=settings.TEMPERATURE_MAIN,
        max_tokens=max_t,
        timeout=timeout_sec,
    )
    llm_with_tools = llm_base.bind_tools(tools) if tools else llm_base

    max_turns = _get_max_turns(agent_name)
    tool_count = 0
    collected_artifacts = {"main_results": [], "web_results": []}
    llm_error: str = ""
    t_start = time.perf_counter()

    result_content = ""
    turns_taken = 0

    for turn in range(max_turns):
        # v8.3.4: retrieve-agent 代码级收敛——去重文献数达标 → 强制收尾输出报告，
        # 不再进入下一轮检索（"max turns forcing final" 只在确有必要时出现）
        if (agent_name == "retrieve-agent" and turn > 0
                and _count_unique_docs(collected_artifacts["main_results"])
                >= settings.RETRIEVE_CONVERGE_MIN_DOCS):
            logger.info(
                f"[AgentRunner] {agent_name} 提前收敛: 去重文献 ≥ "
                f"{settings.RETRIEVE_CONVERGE_MIN_DOCS} 篇，强制输出检索报告"
            )
            messages.append(HumanMessage(content=(
                "已收集到足够的相关文献，立即停止检索，"
                "直接输出最终结构化检索报告（检索结论 / 关键文献 / 信息缺口）。")))
            try:
                final_resp = await llm_base.ainvoke(messages)
                if final_resp.content:
                    result_content = final_resp.content
            except Exception as e:
                logger.warning(f"[AgentRunner] {agent_name} converge-final failed: {e}")
            break
        turns_taken = turn + 1
        t_llm = time.perf_counter()
        response = None
        for attempt in range(3):
            try:
                response = await llm_with_tools.ainvoke(messages)
                break
            except Exception as e:
                llm_error = str(e)
                if attempt < 2:
                    logger.warning(f"[AgentRunner] {agent_name} LLM retry {attempt+1}/3: {e}")
                    await asyncio.sleep(2)
                else:
                    logger.error(f"[AgentRunner] {agent_name} LLM error (retries exhausted): {e}")
        if response is None:
            break

        dt_llm = (time.perf_counter() - t_llm) * 1000
        messages.append(response)
        # v8.3.3: 子 Agent 真实 token 增量推送（前端上下文面板实时刷新，避免累计值重复计数）
        try:
            usage = (getattr(response, "usage_metadata", None)
                     or (getattr(response, "response_metadata", {}) or {}).get("usage", {}))
            if isinstance(usage, dict) and usage.get("total_tokens"):
                emit_usage_delta(session_id, agent_name,
                                 usage.get("input_tokens", 0),
                                 usage.get("output_tokens", 0),
                                 usage.get("total_tokens", 0))
        except Exception:
            pass

        if hasattr(response, "content") and response.content:
            logger.debug(
                f"[AgentRunner] {agent_name} turn{turn}: "
                f"content={response.content[:120]}"
            )

        if not getattr(response, "tool_calls", None):
            result_content = response.content or ""
            logger.info(
                f"[AgentRunner] {agent_name} turn{turn} done: "
                f"{dt_llm:.0f}ms, {len(result_content)} chars"
            )
            break

        # Emit thinking content (LLM reasoning before tool calls)
        if response.content:
            try:
                emit_thinking(response.content[:800])
            except Exception:
                pass

        # Emit structured tool events for each tool call
        # v8.3.5: id 单一来源——每个 tool_call 只提取一次 id，
        # 协议 ToolMessage / 计时器 mark_tool_start-end / SSE 三处复用同一 id
        tc_ids = [_tc_id(tc) for tc in response.tool_calls]
        for idx, tc in enumerate(response.tool_calls):
            tc_dict = _make_tool_call_dict(tc)
            tc_id = tc_ids[idx]
            tc_name = tc_dict.get("name", "?")
            try:
                emit_tool_call_start(tc_name[:30], tc_dict.get("args", {}), tc_id)
                emit_tool_executing(f"子Agent {agent_name} 调用 {tc_name}...", tc_name[:30], tc_id)
            except Exception:
                pass

        t_tool = time.perf_counter()
        try:
            tn = PartitionedToolNode(tools)
            tool_results = await tn.execute_tools(list(response.tool_calls))
        except Exception as e:
            logger.error(f"[AgentRunner] {agent_name} tool exec failed: {e}")
            for idx, tc in enumerate(response.tool_calls):
                tc_id = tc_ids[idx] if idx < len(tc_ids) else _tc_id(tc)
                tc_name = getattr(tc, "name", "") or (tc.get("name", "") if isinstance(tc, dict) else "")
                try:
                    emit_tool_result(tc_name, str(e)[:500], tc_id,
                                     is_error=True, summary="工具执行失败")
                except Exception:
                    pass
            break

        dt_tool = (time.perf_counter() - t_tool) * 1000

        for idx, tr in enumerate(tool_results):
            messages.append(tr)
            art = getattr(tr, "artifact", {}) or {}
            if isinstance(art, dict):
                collected_artifacts["main_results"].extend(
                    art.get("main_results", [])
                )
                collected_artifacts["web_results"].extend(
                    art.get("web_results", [])
                )

            tc = response.tool_calls[idx] if idx < len(response.tool_calls) else None
            # v8.3.5: 复用 tc_ids 单一来源（此前两次独立提取 → dict 形态 UUID 不匹配，
            # mark_tool_start/end 失配 → 心跳计时器泄漏 → tool_executing 事件风暴）
            tc_id = tc_ids[idx] if idx < len(tc_ids) else _tc_id(tc)
            tc_name = (tc.get("name", "") if isinstance(tc, dict)
                       else getattr(tc, "name", "") if tc else "")
            # AG-8: 子 Agent 内工具结果分档截断（与 supervisor 的 TOOL_RESULT_CAPS 一致）
            caps = getattr(settings, "TOOL_RESULT_CAPS", {}) or {}
            cap = caps.get(agent_name, caps.get("default", 100000))
            full_content = str(getattr(tr, "content", ""))
            if len(full_content) > cap:
                tr.content = full_content[:cap] + "\n\n[... 已截断 ...]"
                logger.info(f"[AgentRunner] truncated tool result ({len(full_content)} chars > cap {cap})")
            try:
                result_text = str(getattr(tr, "content", ""))[:100000]
                result_count = len(collected_artifacts["main_results"]) + len(collected_artifacts["web_results"])
                emit_tool_result(
                    tc_name[:30],
                    result_text,
                    tc_id,
                    summary=f"完成 ({dt_tool:.0f}ms, {result_count} 条结果)",
                )
            except Exception:
                pass

        tool_count += len(response.tool_calls)
        logger.info(
            f"[AgentRunner] {agent_name} turn{turn}: "
            f"LLM {dt_llm:.0f}ms + tools {dt_tool:.0f}ms "
            f"({time.perf_counter()-t_start:.1f}s total)"
        )

    else:
        if result_content and not getattr(
            messages[-1] if messages else None, "tool_calls", None
        ):
            pass
        else:
            logger.info(f"[AgentRunner] {agent_name} max turns, forcing final")
            messages.append(HumanMessage(content=(
                "You have reached the maximum number of turns. "
                "Do NOT call any more tools. "
                "Provide your final answer based on the information you have. "
                "If information is insufficient, state what is missing."
            )))
            try:
                final_resp = await llm_base.ainvoke(messages)
                if final_resp.content:
                    result_content = final_resp.content
            except Exception:
                for m in reversed(messages):
                    if hasattr(m, "content") and m.content and not isinstance(m, SystemMessage):
                        result_content = m.content
                        break

    total_time = (time.perf_counter() - t_start) * 1000

    unique_dois = []
    for r in collected_artifacts.get("main_results", []):
        doi = (r.get("doi") or "").strip()
        if doi and doi not in unique_dois:
            unique_dois.append(doi)

    try:
        emit_encoded("agent_summary", {
            "agent_name": agent_name,
            "doc_count": len(collected_artifacts.get("main_results", [])),
            "doi_count": len(unique_dois),
            "doi_list": unique_dois[:10],
            "result_len": len(result_content),
            "tools_called": tool_count,
        })
    except Exception:
        pass

    logger.info(
        f"[AgentRunner] {agent_name} complete: "
        f"{len(result_content)} chars, {tool_count} tools, "
        f"{total_time:.0f}ms"
    )

    return {
        "agent": agent_name,
        "result": result_content or (f"[AgentError] {agent_name} LLM 调用失败(重试耗尽): {llm_error}" if llm_error else "(no output)"),
        "artifacts": collected_artifacts,
        "tools_called": tool_count,
        # v8.3.3: 返回实际轮数（此前恒等于上限，误导调用方）
        "turns": turns_taken,
    }


def _make_tool_call_dict(tc) -> dict:
    name = getattr(tc, "name", "") or ""
    args = {}
    try:
        if callable(getattr(tc, "get", None)):
            args = dict(tc)
        elif hasattr(tc, "__dict__"):
            args = {k: v for k, v in tc.__dict__.items() if k not in ("id", "name")}
    except Exception:
        pass
    return {"name": name, "args": args}


def _resolve_tool_names(agent_name: str) -> list[str]:
    mapping = {
        "retrieve-agent": [
            "citrus_rag_search", "academic_search",
        ],
        "write-agent": ["write_local_file"],
        "analyze-agent": [
            "statistical_analysis", "experimental_design",
        ],
    }
    return mapping.get(agent_name, [])


def _get_max_turns(agent_name: str) -> int:
    # v8.3.3: 轮次上限接线 config.yaml subagents.<name>.max_turns（此前硬编码）
    sub = (getattr(settings, "SUBAGENT_MAX_TURNS", None) or {}).get(agent_name, {})
    if isinstance(sub, dict):
        return int(sub.get("max_turns") or 3)
    return int(sub or 3)
