# -*- coding: utf-8 -*-
"""deepseek_web_search — DeepSeek 原生联网搜索工具（v8.15 P3 脚手架）。

机制：DeepSeek API 开放 Responses 接口，请求 tools 声明 {"type": "web_search"}，
由 DeepSeek 服务端自行决定是否联网检索、检索哪些网站（不限制站点），
同一请求内返回最终回答 + web_search_call 引用条目。本工具把一次这样的
调用封装成检索链中的一个工具：模型自主决定何时需要实时信息并调用它，
调用内仍是 DeepSeek 自己联网检索，我们只把引用映射为 web 证据组条目。

⚠️ 未实测标记（v8.15）：沙箱无外网、无有效 API key，本文件按 OpenAI
Responses API web_search 契约编写，端点路径/引用字段形态以 DeepSeek 官方
文档为准。上线前请先运行配套探测（python -m src.tools.websearch_probe
或手动 curl）核对：① 端点精确路径（config web_search.responses_path）
② 模型 deepseek-v4-flash 是否在该能力支持列表 ③ 返回 output 里
web_search_call 的精确字段名。探测通过前保持 config web_search.enabled=false。

设计（与 PIPELINE 结构一致）：
  - 主循环（chat/completions + langchain 工具调用）完全不动；
  - 本工具仅在内部分支开一次 Responses 调用（非流式，聚合引用）；
  - 返回 content=模型检索摘要 + 引用清单；artifact.web_results=引用条目
    （source="web"），汇入 cited_refs type="web" → 侧栏「联网搜索」组。
"""
import logging
import time
from typing import Tuple

import requests

from langchain_core.tools import tool

from src.config import settings, get_deepseek_model

logger = logging.getLogger(__name__)

# 引用条目默认上限
_WEB_MAX_ITEMS = 8
_HTTP_TIMEOUT = 30


def _responses_endpoint() -> str:
    """Responses 端点：{base}{path}。base 与主模型一致（api.deepseek.com）。"""
    base = (getattr(settings, "RESOLVED_MAIN_BASE_URL", None)
            or settings.MAIN_BASE_URL or "https://api.deepseek.com").rstrip("/")
    path = getattr(settings, "WEB_SEARCH_RESPONSES_PATH", "/v1/responses")
    return f"{base}{path}"


def _parse_response_output(output: list) -> Tuple[str, list]:
    """解析 Responses 输出：text 块拼接 + web_search_call 引用条目。

    契约（OpenAI Responses API web_search 形态，DeepSeek 待核对）：
      - message 块: {"type": "message", "content": [{"type": "output_text", "text": ...}]}
      - 引用块:   {"type": "web_search_call", "status": ..., "title": ..., "url": ...,
                   "content": {...}}
    防御式解析：字段缺失时跳过该条目，绝不因引用格式差异抛异常。
    """
    text_parts: list[str] = []
    calls: list[dict] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "")
        if itype == "message":
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    text_parts.append(str(c.get("text", "")))
        elif itype == "web_search_call":
            title = str(item.get("title", "") or "").strip()
            url = str(item.get("url", "") or "").strip()
            if not url:
                # 兼容部分实现把引用挂在 status/content 上
                content = item.get("content") or {}
                url = str((content.get("url") if isinstance(content, dict) else "") or "").strip()
                title = title or str((content.get("title") if isinstance(content, dict) else "") or "")
            snippet = str((item.get("content") or ""))
            if isinstance(item.get("content"), dict):
                snippet = str(item["content"].get("snippet")
                              or item["content"].get("content") or "")
            if url:
                calls.append({"title": title or url, "url": url,
                              "abstract": (snippet or "")[:500]})
    return "\n".join(p for p in text_parts if p).strip(), calls[: _WEB_MAX_ITEMS]


@tool(response_format="content_and_artifact")
def deepseek_web_search(query: str) -> Tuple[str, dict]:
    """联网搜索（DeepSeek 原生 Responses web_search tool）。由 DeepSeek 自行联网检索并返回带引用的最新信息。

    仅当本地文献库覆盖不足、需要最新/实时信息（近期事件、最新品种上线、政策、市场行情等）时使用；
    每次检索预算每轮 ≤1 次。返回引用条目会进入侧栏「联网搜索」证据组。

    Args:
        query: 要联网检索的问题/关键词（英文或中文均可，5-20 词为佳）
    """
    empty = {"main_results": [], "web_results": []}
    # v8.15: 前端开关即总开关（config web_search.enabled 仅部署默认值，不作门槛）。
    # 每次请求由 chat_v2 写入 web_search_enabled contextvar，工具执行层据此强制短路——
    # 开关关闭时即使模型误调也不会产生任何网络请求。
    from src.core.tracing import web_search_enabled as _req_web_on
    if not _req_web_on():
        logger.info("[deepseek_web_search] 本次请求未开启联网搜索，请求被短路")
        return ("[DISABLED] 联网搜索未开启。\n"
                "建议: 需要实时/最新信息时，点击输入框左下的「联网」开关后重新提问；"
                "本地文献类问题直接使用 citrus_rag_search。",
                empty)
    query = (query or "").strip()
    if not query:
        return "[ERR_PARSE] 查询词不能为空", empty
    if len(query) > 500:
        return f"[ERR_PARSE] 查询词过长 ({len(query)}字符)，请精简至 500 字符以内", empty

    t0 = time.perf_counter()
    try:
        payload = {
            "model": get_deepseek_model(),
            "input": query,
            "tools": [{"type": "web_search"}],   # 服务端内置 web_search：模型自主决定是否联网
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {settings.RESOLVED_MAIN_API_KEY or settings.MAIN_API_KEY}",
            "Content-Type": "application/json",
        }
        url = _responses_endpoint()
        logger.info(f"[deepseek_web_search] Responses 调用: {url} model={payload['model']}")
        resp = requests.post(url, headers=headers, json=payload, timeout=_HTTP_TIMEOUT)
        resp.raise_for_status()
        body = resp.json()
        summary, calls = _parse_response_output(body.get("output") or [])
        elapsed = (time.perf_counter() - t0) * 1000

        if not summary and not calls:
            logger.warning(f"[deepseek_web_search] 无返回内容: {str(body)[:300]}")
            return ("[ERR_EMPTY] 联网搜索未返回内容（模型判断无需联网或请求参数有误）。"
                    "建议: 检查 config web_search.responses_path 是否与官方文档一致。", empty)

        items = []
        for idx, c in enumerate(calls, 1):
            item = {
                "ref_id": f"W{idx}",
                "type": "web",
                "source": "web",
                "url": c["url"],
                "title": c["title"],
                "abstract": c.get("abstract", ""),
                "snippet": c.get("abstract", ""),
            }
            items.append(item)

        text_result = (
            f"联网检索摘要（DeepSeek 原生搜索返回）:\n{summary[:2000] or '(模型判断无需联网，未检索)'}"
            + ("\n\n引用:\n" + "\n".join(f"[W{i}] {c['title']} — {c['url']}"
                                          for i, c in enumerate(calls, 1)) if calls else "")
        )
        content = _format_web_tool_result("deepseek_web_search", query, text_result,
                                          status="ok", results_count=len(items),
                                          elapsed_ms=elapsed)
        logger.info(f"[deepseek_web_search] done: {len(calls)} 引用, {len(summary)} 字摘要, {elapsed:.0f}ms")
        return content, {"main_results": [], "web_results": items}
    except Exception as e:
        logger.error(f"[deepseek_web_search] 调用失败: {e}")
        return (f"[ERR_NETWORK] 联网搜索调用失败: {e}\n"
                "建议: 检查 config web_search.responses_path（Responses 端点路径）"
                "与 API key 是否有效。", empty)


def _format_web_tool_result(tool_name, query, content, status="ok",
                            results_count=0, elapsed_ms=0) -> str:
    """与 search._format_tool_result 同构的工具回执头（避免循环依赖，本地实现）。"""
    header = (
        f"## [ToolResult] {tool_name}\n"
        f"**status**: {status}\n"
        f"**query**: \"{query[:120]}\"\n"
        f"**results_count**: {results_count}"
    )
    if elapsed_ms > 0:
        header += f"\n**elapsed_ms**: {elapsed_ms:.0f}"
    return f"{header}\n\n{content}"