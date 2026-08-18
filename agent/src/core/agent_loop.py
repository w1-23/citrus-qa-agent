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