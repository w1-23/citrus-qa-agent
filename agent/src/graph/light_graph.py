"""Light Graph — LLM 自主路由 Supervisor.

v8.3.0: 移除 hardcoded pre-retrieve/ReAct fallback.
LLM bind_tools 自主决定是否调用搜索工具, max 3 轮.
"""
import asyncio
import logging
import time

from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END

from src.graph.state import AgentState
from src.config import settings, get_deepseek_model
from src.core.agent_loop import (
    dedup_by_doi, emit_llm_usage, FINAL_ANSWER_PROMPT,
    invoke_llm_with_retry, force_final_answer,
    build_cited_refs, renumber_and_sync_trace,
    run_save_node,
)
from src.prompts.loader import assemble_system_prompt

logger = logging.getLogger(__name__)

from src.core.progress_bus import (
    emit_text, emit_reasoning,
    emit_tool_call_start, emit_tool_executing, emit_tool_result,
    emit_status,
)
from src.core.stream_llm import stream_llm_response

# v8.3.3: 轮次上限接线 config.yaml light.max_turns（此前硬编码 2）
LIGHT_MAX_TURNS = settings.LIGHT_MAX_TURNS
LIGHT_TOOL_NAMES = ("citrus_rag_search", "read_local_file")


def _build_light_llm(bind_tools=None):
    # v8.4: 客户端进程级复用（llm_pool），避免每请求新建 ChatOpenAI/连接池
    from src.core.llm_pool import get_llm as _pool_get_llm
    client = _pool_get_llm(
        model=get_deepseek_model(),
        api_key=settings.RESOLVED_MAIN_API_KEY,
        base_url=settings.RESOLVED_MAIN_BASE_URL,
        temperature=settings.TEMPERATURE_MAIN,
        max_tokens=16384,
        timeout=60,
    )
    if bind_tools:
        client = client.bind_tools(bind_tools)
    return client


async def load_context_node(state: AgentState) -> dict:
    query = state.get("query", "")
    session_id = state.get("session_id", "default")
    mode = state.get("mode", "light")
    logger.info(f"[LightGraph:load] session={session_id[:8]}...")

    try:
        emit_status("step_active", step_id="load", message="加载会话上下文...")
    except Exception:
        pass

    result: dict = {}
    try:
        from src.session.manager import session_manager
        from src.core.context_budget import ContextBudget, ContextBudgetConfig
        from src.core.context_manager import ContextManager, finalize_load_result
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
        # v8.4: 收尾装配收敛至 context_manager.finalize_load_result
        # （与 expert 图共用，消除双实现漂移）
        result = finalize_load_result(
            ctx,
            session_manager=session_manager,
            session_id=session_id,
            budget=budget,
            node_label="load_context",
            log_prefix="LightGraph",
        )
        # v8.15.3e: load 完成状态推送（与 expert 图一致，消除静默加载期）
        try:
            from src.core.progress_bus import emit_status, emit_progress
            emit_status("step_done", step_id="load")
            emit_progress("tool_progress", {
                "message": f"已加载历史 {len(ctx.history_messages)} 条，正在理解问题…",
                "tool_call_id": ""})
        except Exception:
            pass
        # v8.17.15: 草稿全链删除（用户决策）——light 模式不再有草稿先行；
        # 联网（若开启）由 light_retrieve_node 后的 supervisor 工具链承担。
        return result

    except Exception as e:
        logger.warning(f"[LightGraph:load] failed: {e}")
        result["_trace"] = {"node": "load_context", "elapsed_ms": 0, "summary": "unavailable"}
        return result


async def light_retrieve_node(state: AgentState) -> dict:
    """v8.4.6 F1: light 模式代码级预检索——科研问题保证至少一次本地检索。

    此前 light 由 LLM 自主决定是否调用工具，实测经常 0 次工具直接作答
    （无执行日志、无文献引用面板）。此处代码强制预检索：
      - 走基础检索（无 HyDE，light 模式速度优先，省 ~6s/次）
      - 结果经 <retrieval_context> 注入当前轮 HumanMessage（light_rules 已预留）
      - artifacts 合流 references_data → 侧栏引用正常显示
    LLM 仍可在此基础上自行补充工具调用（追问/缺口）。
    """
    query = state.get("query", "")
    session_id = state.get("session_id", "default")
    logger.info(f"[LightGraph:retrieve] pre-retrieve for session={session_id[:8]}...")
    try:
        from src.tools.search import format_rag_context
        from src.retrieval.multi_retriever import MultiBatchRetriever
        rag = MultiBatchRetriever()
        results = await asyncio.to_thread(rag.search, query)
        main = list(results or [])
        content = format_rag_context(main, "main") if main else "未检索到相关文献。"
        emit_status("retrieval_done", main_count=len(main))
        logger.info(f"[LightGraph:retrieve] pre-retrieve done: {len(main)} items")
        return {
            "retrieval_context": content,
            "main_results": main,
            "web_results": [],
            "_trace": {"node": "light_retrieve", "elapsed_ms": 0,
                       "summary": f"pre-retrieve {len(main)} items"},
        }
    except Exception as e:
        logger.warning(f"[LightGraph:retrieve] pre-retrieve failed: {e}")
        return {"retrieval_context": "", "main_results": [], "web_results": [],
                "_trace": {"node": "light_retrieve", "elapsed_ms": 0,
                           "summary": "unavailable"}}


async def _light_force_final(messages: list) -> str:
    """light 收尾统一函数（v8.4.6 B1：预算超限与跑满轮次共用）。

    v8.13-b5a: 收尾骨架收敛至 src.core.agent_loop.force_final_answer，
    这里保留 light 差异点（未绑工具客户端、不记日志、fallback=any）。
    """
    return await force_final_answer(
        messages,
        stream_call=lambda: stream_llm_response(
            _build_light_llm(),
            messages + [HumanMessage(content=FINAL_ANSWER_PROMPT)],
            on_text=emit_text, on_reasoning=emit_reasoning),
        fallback_mode="any",
    )


async def light_supervisor_node(state: AgentState) -> dict:
    """LLM autonomous routing supervisor.

    Binds citrus_rag_search.
    LLM receives light-mode system prompt and decides whether to call tools.
    Max 3 turns, timeout 60s per LLM call.
    """
    query = state.get("query", "")
    format_hint = state.get("format_hint")
    logger.info(f"[LightGraph:supervisor] query={query[:60]}...")

    system_prompt = assemble_system_prompt(
        mode="light", format_hint=format_hint, query=query,
    )

    from src.core.context_manager import LoadedContext, build_human_message
    ctx = LoadedContext(
        session_id=state.get("session_id", ""),
        mode="light",
        query=query,
        history_summary=state.get("history_summary"),
        long_term_memory=state.get("long_term_memory"),
        resident_cards=state.get("resident_cards"),
        search_suggestions=state.get("search_suggestions", []),
        format_hint=format_hint,
    )
    # v8.4.6 F1: 代码级预检索结果注入 <retrieval_context>（light_rules 的既定通道）
    human_msg = build_human_message(
        ctx, retrieval_context=state.get("retrieval_context") or "")

    messages: list = [SystemMessage(content=system_prompt)]
    history_msgs = list(state.get("messages", []))
    if history_msgs:
        messages.extend(history_msgs)
    # v8.3.8: 历史检索证据块
    if state.get("history_evidence_block"):
        messages.append(HumanMessage(content=state["history_evidence_block"]))
    messages.append(human_msg)
    # v8.3.8: 本轮轨迹起点
    trace_start_index = len(messages) - 1

    from src.tools import _TOOL_REGISTRY_BY_NAME
    tools = [t for t_name in LIGHT_TOOL_NAMES
             if t_name in _TOOL_REGISTRY_BY_NAME
             for t in [_TOOL_REGISTRY_BY_NAME[t_name]]]
    # v8.15.2: light 模式不分配联网工具（用户决定：轻量模式走纯本地检索）

    llm = _build_light_llm(bind_tools=tools) if tools else _build_light_llm()

    answer = ""
    # v8.4.6 F1: 预检索节点的 artifacts 合流（否则 side 栏引用为空）
    all_main_results = list(state.get("main_results") or [])
    all_web_results = list(state.get("web_results") or [])
    tool_call_count = 0
    tool_names_called: list[str] = []   # v9.2: 已调用工具名（main.py done 分支消费）
    t0 = time.perf_counter()

    try:
        for turn in range(LIGHT_MAX_TURNS):
            # v8.4.6 B1: 预算前移（与 expert 一致）——每次模型调用前检查上下文占用，
            # 超硬阈值直接统一收尾（light 无独立熔断，窗口 1M 下风险低但须兜底）
            try:
                from src.core.context_budget import ContextBudget, ContextBudgetConfig
                _budget = ContextBudget(ContextBudgetConfig(
                    max_tokens=settings.CONTEXT_BUDGET_MAX_TOKENS,
                    soft_threshold=settings.CONTEXT_BUDGET_SOFT_THRESHOLD,
                    hard_threshold=settings.CONTEXT_BUDGET_HARD_THRESHOLD,
                ))
                _est = _budget.estimate_tokens(messages)
                if _est / _budget.config.max_tokens >= _budget.config.hard_threshold:
                    logger.warning(
                        f"[LightGraph:supervisor] 上下文占用 "
                        f"{_est / _budget.config.max_tokens:.1%} ≥ 硬阈值，强制收尾")
                    answer = await _light_force_final(messages)
                    break
            except Exception:
                pass
            t_llm = time.perf_counter()
            # v8.13-b5a: LLM 重试收敛至 invoke_llm_with_retry（3 次/3s/失败上抛）
            response, _, _ = await invoke_llm_with_retry(
                lambda: stream_llm_response(
                    llm, messages,
                    on_text=emit_text, on_reasoning=emit_reasoning),
                label="[LightGraph:supervisor]", sleep_s=3.0)
            dt_llm = (time.perf_counter() - t_llm) * 1000
            messages.append(response)
            # v8.3.3: 推送真实 token 增量（前端上下文面板实时刷新，避免累计值重复计数）
            emit_llm_usage(state.get("session_id", ""), "light_supervisor", response)

            if not getattr(response, "tool_calls", None):
                answer = response.content or ""
                logger.info(f"[LightGraph:supervisor] turn{turn} done: {len(answer)}c / {dt_llm:.0f}ms")
                break

            # v8.4.13: 工具轮中间文本已由流式实时上屏，不再 emit_thinking

            # Emit tool_call_start events for all tools
            for tc in response.tool_calls:
                # tool_calls 元素为 dict（OpenAI 兼容格式），兼容对象形式取值
                if isinstance(tc, dict):
                    tc_id = tc.get("id", "")
                    tc_name = tc.get("name", "?")
                    tc_args = dict(tc.get("args") or {})
                else:
                    tc_id = getattr(tc, "id", "")
                    tc_name = getattr(tc, "name", "?")
                    tc_args = dict(getattr(tc, "args", None) or {})
                logger.info(f"[LightGraph:supervisor] turn{turn}: call {tc_name}({str(tc_args)[:80]})")
                if tc_name and tc_name not in tool_names_called:
                    tool_names_called.append(tc_name)
                try:
                    emit_tool_call_start(tc_name[:30], tc_args, tc_id)
                    emit_tool_executing(f"调用 {tc_name}...", tc_name[:30], tc_id)
                except Exception:
                    pass

            # v8.4.6 B4: light 补充检索同样代码级去重（与预检索 query 及本轮
            # 已执行角度共享 seen 集）——防"预检索后模型又搜同一角度"
            from src.core.agent_runner import check_query_redundant
            from langchain_core.messages import ToolMessage
            _seen = [query]
            exec_calls: list = []
            placeholder_results: dict = {}
            for _idx, _tc in enumerate(response.tool_calls):
                _td = (_tc if isinstance(_tc, dict)
                       else {"id": getattr(_tc, "id", ""), "name": getattr(_tc, "name", ""),
                             "args": dict(getattr(_tc, "args", None) or {})})
                if _td.get("name") == "citrus_rag_search":
                    _q = str((_td.get("args") or {}).get("query", "") or "")
                    _reason = check_query_redundant(_q, _seen)
                    if _reason:
                        placeholder_results[_idx] = ToolMessage(
                            content=f"[DEDUP] 检索角度重复: {_reason}。该检索未执行，请更换关键词/角度。",
                            tool_call_id=_td.get("id", ""), name="citrus_rag_search",
                            artifact={"main_results": [], "web_results": []})
                        logger.info(f"[LightGraph:supervisor] 重复检索角度跳过: {_reason}")
                        continue
                    _seen.append(_q)
                exec_calls.append((_idx, _tc))

            # Batch execute all tool calls (preserves original API call IDs)
            t_tool = time.perf_counter()
            from src.tools.registry import PartitionedToolNode
            tn = PartitionedToolNode(tools)
            exec_results = (await tn.execute_tools([tc for _, tc in exec_calls])
                            if exec_calls else [])
            dt_tool = (time.perf_counter() - t_tool) * 1000
            tool_call_count += len(response.tool_calls)
            tool_results: list = [None] * len(response.tool_calls)
            for (_idx, _tc), tr in zip(exec_calls, exec_results):
                tool_results[_idx] = tr
            for _idx, tr in placeholder_results.items():
                tool_results[_idx] = tr

            for idx, tr in enumerate(tool_results):
                messages.append(tr)
                art = getattr(tr, "artifact", {}) or {}
                if isinstance(art, dict):
                    all_main_results.extend(art.get("main_results", []))
                    all_web_results.extend(art.get("web_results", []))
                tc = response.tool_calls[idx] if idx < len(response.tool_calls) else None
                tc_id = getattr(tc, "id", "") if tc else ""
                tc_name = getattr(tc, "name", "") if tc else "tool"
                try:
                    result_text = str(getattr(tr, "content", ""))[:2000]
                    emit_tool_result(tc_name[:30], result_text, tc_id,
                                     summary=f"完成 ({dt_tool:.0f}ms)")
                except Exception:
                    pass

            logger.info(f"[LightGraph:supervisor] turn{turn}: LLM {dt_llm:.0f}ms (total {time.perf_counter()-t0:.1f}s)")

        else:
            logger.info("[LightGraph:supervisor] max turns, forcing final")
            answer = await _light_force_final(messages)

    except Exception as e:
        logger.error(f"[LightGraph:supervisor] error: {e}")
        answer = f"An error occurred: {e}"

    elapsed = (time.perf_counter() - t0) * 1000

    deduped_main = dedup_by_doi(all_main_results)

    # v9.2: 引用回执装配 + 统一重排收敛共享原语（agent_loop，web 槽位 light=5）
    cited_refs = build_cited_refs(deduped_main, all_web_results, web_slot=5)
    answer, cited_refs, ref_remap = renumber_and_sync_trace(
        messages, answer, cited_refs)

    references_data = {"cited": cited_refs, "uncited": [],
                       "remap": ref_remap, "total": len(cited_refs)}

    # v8.15.2: 不再注入历史证据引用（H1..Hn）——侧栏只显示本轮回答真实引用的证据，
    # 防止侧栏膨胀。（原 v8.4.6 F2 行为已移除）

    # v8.4.13: 回答已由流式逐 token 上屏（text 事件），不再模拟打字机推送
    logger.info(
        f"[LightGraph:supervisor] done: {len(answer)}c, "
        f"{tool_call_count} tool calls, "
        f"{len(deduped_main)} main + {len(all_web_results)} web results, "
        f"{elapsed:.0f}ms"
    )

    return {
        "answer": answer,
        "gen_time_ms": elapsed,
        "main_results": deduped_main[:20],
        "web_results": all_web_results[:5],
        "references_data": references_data,
        # v9.2: 工具名列表（main.py done 分支对齐 expert——request_done 日志真实计数）
        "tools_called": tool_names_called,
        # v8.3.8: 本轮完整轨迹（save 节点持久化）
        "turn_trace": messages[trace_start_index:],
        "_trace": {"node": "light_supervisor", "elapsed_ms": elapsed,
                   "summary": f"{len(answer)} chars, {tool_call_count} tools"},
    }


async def save_context_node(state: AgentState) -> dict:
    """v9.2: save 核心收敛至 agent_loop.run_save_node（light 差异：
    无 web 账本并入、LTM 仅长度门槛；行为与 v9.2 前逐位一致）。"""
    return await run_save_node(state, log_tag="LightGraph", include_web=False)



def build_light_graph() -> StateGraph:
    graph = StateGraph(AgentState)

    graph.add_node("load_context", load_context_node)
    # v8.4.6 F1: 代码级预检索（科研问题保证 ≥1 次本地检索，日志/引用面板可见）
    graph.add_node("light_retrieve", light_retrieve_node)
    graph.add_node("light_supervisor", light_supervisor_node)
    graph.add_node("save_context", save_context_node)

    graph.set_entry_point("load_context")
    graph.add_edge("load_context", "light_retrieve")
    graph.add_edge("light_retrieve", "light_supervisor")
    graph.add_edge("light_supervisor", "save_context")
    graph.add_edge("save_context", END)

    return graph.compile()
