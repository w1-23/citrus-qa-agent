# -*- coding: utf-8 -*-
"""AgentLoop 基座（第四批第三步·第一步）——收敛三层 ReAct 循环的重复原语。

expert_graph（专家 supervisor）/ light_graph（轻量 supervisor）/ agent_runner（子 Agent）
三层各自维护了几乎相同的：工具调用 id 提取、DOI 去重计数、LLM 调用重试、
强制收尾（force-final）与空答兜底。此处收敛为公共原语；三层只保留差异点
（max_turns、工具集、执行策略、llm 客户端、收尾 label 等），后续步骤再抽循环骨架。

设计纪律：原语为「纯函数/注入式骨架」，不硬编码任何一层的策略——
        - tc_id / count_unique_docs / last_message_content 为纯函数；
        - invoke_llm_with_retry 注入 `invoke` 可调用对象（调用方绑定 llm+消息+流式回调）；
        - force_final_answer 注入 `stream_call` 可调用对象（调用方绑定各自的 llm/prompt）。
        保证三层各自的差异点（3s vs 2s、raise vs None、aimessage vs any 兜底）可精确表达。
"""
import asyncio
import logging
import uuid

from langchain_core.messages import AIMessage, SystemMessage

from src.core.evidence import (src_of, renumber_refs,
                               render_evidence, EVIDENCE_SNIPPET_MAX_CHARS)

logger = logging.getLogger(__name__)

# 三层收尾 prompt 字节一致（原 expert `_FINAL_PROMPT` 与 light 内联 `final_prompt` 相同）
FINAL_ANSWER_PROMPT = (
    "请立即给出最终回答：基于已检索的全部证据，完整、详尽、结构化地作答；"
    "不要精简、不要省略、不要提及工具或轮次限制；信息不足请逐条说明缺口。"
)


def tc_id(tc) -> str:
    """兼容 dict/对象两种 tool_calls 形态取 id（v8.3.4/8.3.5 单一来源）。"""
    if isinstance(tc, dict):
        return tc.get("id", "") or str(uuid.uuid4())
    return getattr(tc, "id", "") or str(uuid.uuid4())


def count_unique_docs(main_results: list) -> int:
    """按 DOI 去重计数文献（无 DOI 按条计数）——状态栏/引用统计/收敛判断用。"""
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


def dedup_by_doi(rows) -> list:
    """按 DOI 去重文献列表（保留首次出现顺序；无 DOI 条目原样保留）。

    与 count_unique_docs 的差异：此处不做 lower（贴合 expert/light 收尾去重原行为）。
    """
    seen = set()
    out = []
    for r in rows:
        doi = (r.get("doi") or "").strip()
        if doi and doi in seen:
            continue
        seen.add(doi)
        out.append(r)
    return out


def build_cited_refs(deduped_main: list, all_web_results: list,
                     *, web_slot: int = 10) -> list:
    """引用回执装配（v9.2 抽取，expert/light 两图共用——消除 40 行双实现）。

    - 数字编号 [n] = deduped_main 顺序 i+1（rag/ucr 共用编号池，与侧栏同构）；
    - 联网 [Wn] = all_web_results 顺序 W{i+1}，上限 web_slot（expert 10 / light 5）；
    - 字段结构与前端 renderCitationItem 契约严格一致（含 UCR 专属字段）。
    """
    cited_refs: list = []
    for i, r in enumerate(deduped_main[:20]):
        cited_refs.append({
            "ref_id": i + 1,
            "type": "main",
            "source": src_of(r),                       # v8.15: rag|ucr
            "doi": r.get("doi", "N/A"),
            "title": r.get("title", r.get("name", "Untitled")),
            "section_name": r.get("section_name", ""),
            "text_preview": (r.get("abstract") or r.get("snippet") or "")[:300],
            "score": r.get("score", r.get("rerank_score", 0)) or 0,
            "year": r.get("year", ""),
            "authors": r.get("authors", ""),
            # v8.15: UCR 品种条目专属字段（无 DOI 时前端显示登记号/品种名）
            "variety_name": r.get("variety_name", ""),
            "registry_id": r.get("registry_id", ""),
        })
    for i, wr in enumerate(all_web_results[:web_slot]):
        cited_refs.append({
            "ref_id": f"W{i+1}",
            "type": "web",
            "source": "web",
            "url": wr.get("url", wr.get("link", "")),
            "title": wr.get("title", wr.get("name", "Untitled")),
            "text_preview": (wr.get("snippet") or wr.get("content")
                             or wr.get("abstract") or "")[:300],
            "score": 0,
        })
    return cited_refs


def renumber_and_sync_trace(messages: list, answer: str, cited_refs: list) -> tuple:
    """引用重排 + 轨迹同步（v9.2 抽取，expert/light 共用）。

    对 answer/cited_refs 执行 renumber_refs（数字 [n]→1..k、[Wn]→W1..Wm、
    [Hn]→H1..Hp）；若编号发生变化，把 messages 中承载原始回答的
    AIMessage.content 更新为重排后文本——save 节点按
    ``trace[-1].content == state.answer`` 判重，不同步会因编号差异追加重复消息。

    返回 (new_answer, new_cited, remap)。
    """
    ref_remap: dict = {}
    _orig = answer
    try:
        answer, cited_refs, ref_remap = renumber_refs(answer, cited_refs)
    except Exception as e:
        logger.debug(f"renumber_refs skipped: {e}")
    if answer != _orig:
        try:
            for _m in messages:
                if (isinstance(_m, AIMessage)
                        and not getattr(_m, "tool_calls", None)
                        and getattr(_m, "content", None) == _orig):
                    _m.content = answer
                    break
        except Exception:
            pass
    return answer, cited_refs, ref_remap


def last_message_content(messages: list, mode: str = "aimessage") -> str:
    """收尾空答兜底：从历史消息取最后一段可用文本。

    mode:
      - "aimessage": 仅最后一条 AIMessage.content（专家 supervisor 收尾）
      - "any":       最后一条含 content 的消息，不区分类型（轻量 supervisor 收尾）
      - "nonsystem": 优先 AIMessage，其次任意非 system 消息（子 Agent 收尾）
    """
    if mode in ("aimessage", "nonsystem"):
        for m in reversed(messages):
            if isinstance(m, AIMessage) and getattr(m, "content", None):
                return m.content
        if mode == "aimessage":
            return ""
    if mode == "nonsystem":
        for m in reversed(messages):
            if not isinstance(m, SystemMessage) and getattr(m, "content", None):
                return str(m.content)
        return ""
    # mode == "any"
    for m in reversed(messages):
        if getattr(m, "content", None):
            return m.content
    return ""


async def invoke_llm_with_retry(invoke, *, retries: int = 3, sleep_s: float = 3.0,
                                label: str = "", on_exhausted: str = "raise"):
    """统一 LLM 调用重试（三层 flake/超时兜底收敛为一份）。

    invoke: async callable（无参，调用方绑定 llm + 消息 + 流式回调）。
    on_exhausted: "raise" 最终失败向上抛（专家/轻量）；"none" 最终失败返回
                  (None, retries, last_error)（子 Agent，由循环 break 兜底）。
    返回 (response, attempts_used, last_error)：attempts_used 为 1-based 实际尝试次数，
    last_error 为最近一次异常字符串（无异常则为 ""）。
    """
    last_error = ""
    for attempt in range(retries):
        try:
            return await invoke(), attempt + 1, last_error
        except Exception as e:
            last_error = str(e)
            if attempt < retries - 1:
                logger.warning(f"{label} LLM retry {attempt+1}/{retries}: {e}")
                await asyncio.sleep(sleep_s)
            elif on_exhausted == "raise":
                raise
            else:
                logger.error(f"{label} LLM error (retries exhausted): {e}")
    return None, retries, last_error


async def force_final_answer(messages: list, *, stream_call, label: str = "",
                             fallback_mode: str = "aimessage") -> str:
    """统一强制收尾：注入的 stream_call 流式生成最终回答，异常/空答回退到历史消息。

    label 非空才记日志（专家模式 label 含 reason，轻量模式为空不记）。
    fallback_mode 见 last_message_content。
    """
    if label:
        logger.info(f"{label}, forcing final")
    resp = None
    try:
        resp = await stream_call()
    except Exception as e:
        if label:
            logger.warning(f"{label} 收尾调用失败: {e}")
    if getattr(resp, "content", None):
        return resp.content
    return last_message_content(messages, mode=fallback_mode)


def emit_llm_usage(session_id: str, source: str, response) -> None:
    """推送真实 token 增量（统一经 cache_metrics 提取，含 prompt_cache 命中字段）。

    三层（expert/light/agent_runner）LLM 调用后各复制一份 try/except 上报，收敛为此。
    """
    try:
        from src.core.cache_metrics import emit_usage_from_response
        emit_usage_from_response(session_id, source, response)
    except Exception:
        pass


_LTM_SUBSTANTIAL_KWS = ("###", "结论", "摘要", "引言", "核心结论", "局限与边界")


def default_ltm_gate(answer: str) -> bool:
    """expert 原 LTM 提取门槛：回答含结构化章节关键词（v9.2 常量化）。"""
    return any(kw in answer for kw in _LTM_SUBSTANTIAL_KWS)


async def run_save_node(state: dict, *, log_tag: str,
                        include_web: bool = False,
                        ltm_gate=None) -> dict:
    """save 节点二合一（v9.2，expert/light 共用——消除 85% 同构双实现）。

    原 expert_graph.expert_save_node 与 light_graph.save_context_node 约 85%
    相同（轨迹过滤 → _validate_trace → answer 末尾判重 → save_messages →
    报告拼装 → 证据账本 → save_evidence → 后台 LTM 提取），且已漂移过：
    v8.17.17 web 证据并入账本只改了 expert。此处收敛为单实现，差异显式参数化：

      - log_tag      失败日志标签（"ExpertGraph"/"LightGraph"）
      - include_web  是否把 web_results 并入证据账本（expert=True / light=False）
      - ltm_gate     额外 LTM 门槛函数（入参 answer → bool；expert 默认结构化
                     章节关键词门，light 传 None 仅保留长度 >500 门槛）

    行为不变量逐条保留：无 answer 早退、[历史检索证据] 注入消息不入历史、
    trace[-1].content == answer 判重（防重复消息）、幂等键透传、后台 spawn
    不阻塞主链路、save 失败不抛（仅 warning 日志）。
    """
    query = state.get("query", "")
    answer = state.get("answer", "")
    session_id = state.get("session_id", "default")
    if not answer:
        return {"_trace": {"node": "save", "elapsed_ms": 0, "summary": "no answer"}}

    try:
        from langchain_core.messages import AIMessage, ToolMessage
        from src.session.manager import session_manager, _validate_trace
        from src.core.background import spawn
        # v8.3.8: 完整轨迹持久化（含 tool_calls/ToolMessage 配对）
        trace = list(state.get("turn_trace") or [])
        # 过滤 system 与注入的证据块（它们不属于会话轨迹）；expert 原实现
        # 额外排除 None 元素——保留逐图原行为
        if log_tag == "ExpertGraph":
            trace = [m for m in trace
                     if not isinstance(m, type(None))
                     and getattr(m, "type", "") != "system"]
        else:
            trace = [m for m in trace if getattr(m, "type", "") != "system"]
        # 历史证据块是注入消息，不重复入历史（由 session_evidence 表承载）
        trace = [m for m in trace
                 if not str(getattr(m, "content", "")).startswith("[历史检索证据]")]
        trace = _validate_trace(trace)
        # 最终回答保证在轨迹末尾（v9.2 重排后轨迹同步正为此服务）
        if not (trace and isinstance(trace[-1], AIMessage)
                and not getattr(trace[-1], "tool_calls", None)
                and trace[-1].content == answer):
            trace.append(AIMessage(content=answer))
        spawn(session_manager.save_messages(
            session_id, trace, state.get("idempotency_key", "")))
        # v8.3.8: 证据账本（检索报告 + 结构化清单，跨轮复用）
        # v8.3.9: 合并全部检索报告（多轮检索不丢） + evidence 保留 chunk_id 可回查原文
        report_parts = []
        for m in trace:
            if isinstance(m, ToolMessage) and getattr(m, "name", "") == "call_search_both":
                report_parts.append(str(m.content))
        report_text = "\n\n---\n\n".join(report_parts)
        main_results = state.get("main_results") or []
        web_results = state.get("web_results") or []
        # v8.17.17: 纯联网轮（0 main_results + N web_results）不再存 0 items——
        # web_results [Wn] 条目按联网来源并入 evidence（doi 空/url 带完整地址）；
        # 该并入仅 expert 开启（light 保持原行为：仅 report/main 触发账本）
        if report_text or main_results or (include_web and web_results):
            evidence = [
                {
                    "doi": r.get("doi", ""),
                    "chunk_id": f"{r.get('paper_id', '')}:{r.get('chunk_index', '')}",
                    "title": str(r.get("title", ""))[:150],
                    "score": r.get("score", r.get("rerank_score", 0)) or 0,
                    "year": str(r.get("year", "")),
                    # v8.15: 来源随账本持久化（历史证据 H1..Hn 保留来源徽标）
                    "source": src_of(r),
                    # v8.13-b4c: 账本片段经 render_evidence 单一渲染（2000 安全阀）
                    "snippet": render_evidence(r, max_chars=EVIDENCE_SNIPPET_MAX_CHARS),
                }
                for r in main_results[:30]
            ]
            if include_web:
                evidence += [
                    {
                        "doi": "",
                        "chunk_id": str(w.get("url", ""))[:200],
                        "title": str(w.get("title", ""))[:150],
                        "score": float(w.get("score", 0) or 0),
                        "year": "",
                        "source": "web",
                        "snippet": str(w.get("abstract", w.get("snippet", "")))[:500],
                        "url": str(w.get("url", "")),
                    }
                    for w in web_results[:30]
                ]
            spawn(session_manager.save_evidence(
                session_id, query, evidence, report_text))
    except Exception as e:
        logger.warning(f"[{log_tag}:save] failed: {e}")

    try:
        # v8.4: LTM 提取转后台（spawn），不与响应抢时间；写入走 ADD-only + 置信度门槛
        # （书 3.1 记忆生命周期: 后台提取候选 → 核验 → 更新；提取器不阻塞主链路）
        from src.core.background import spawn as _spawn

        def _extract_and_save_ltm(q: str, a: str, sid: str):
            try:
                from src.guardrails.memory import memory_store
                facts = memory_store.extract_key_facts(q, a)
                for f in facts:
                    # v8.6 (书 §3.1 偏好追踪): type=preference 走 preference_memory
                    # （消费点见 build_human_message 的 <user_preferences> 块）
                    # v8.9: 偏好写入全局域（用户级偏好跨会话生效）
                    if f.get("type") == "preference":
                        memory_store.set_preference(
                            memory_store.GLOBAL_PREF_DOMAIN,
                            f.get("key", ""), f.get("value", ""))
                        continue
                    memory_store.save_long_term_fact(
                        f.get("key", ""),
                        f.get("value", ""),
                        f.get("confidence", 0.5),
                        owner_session=sid,
                        source_query=q,
                    )
            except Exception as e:
                logger.debug(f"[{log_tag}:save] LTM background extract failed: {e}")

        ltm_ok = len(answer) > 500
        if ltm_gate is not None:
            ltm_ok = ltm_ok and ltm_gate(answer)
        if ltm_ok:
            _spawn(asyncio.to_thread(
                _extract_and_save_ltm, query, answer, session_id))
    except Exception as e:
        logger.debug(f"[{log_tag}:save] LTM skip: {e}")

    return {"_trace": {"node": "save", "elapsed_ms": 0, "summary": "saved"}}