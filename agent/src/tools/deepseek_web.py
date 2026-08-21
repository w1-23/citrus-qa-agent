# -*- coding: utf-8 -*-
"""deepseek_web_search — DeepSeek 原生联网搜索工具（v8.15，真机联调后定稿）。

机制：DeepSeek API 开放 Responses 接口，请求 tools 声明 {"type": "web_search"}，
由 DeepSeek 服务端自行决定是否联网检索、检索哪些网站（不限制站点），
同一请求内返回最终回答 + 引用。本工具把一次这样的调用封装成检索链中的一个
工具：模型自主决定何时需要实时信息并调用它，调用内仍是 DeepSeek 自己联网检索，
我们只把引用映射为 web 证据组条目。

2026-08-21 真机探测结论（HTTP 200）：DeepSeek 该版本响应的
  - message 块: {"type":"message","content":[{"type":"output_text","text":...}]}
  - web_search_call 块只记录搜索动作: {"action":{"type":"search","queries":[...]}}
    → 不含结构化来源 URL；URL 依赖模型在回答文本里写出（MD 链接/裸 URL），
    故工具在 input 中附"请标注真实网址"指令，并用多形态防御式解析提取。

设计（与 PIPELINE 结构一致）：
  - 主循环（chat/completions + langchain 工具调用）完全不动；
  - 本工具仅在内部分支开一次 Responses 调用（非流式，聚合引用）；
  - 返回 content=模型检索摘要 + 检索词 + 引用清单；artifact.web_results=引用条目
    （source="web"），汇入 cited_refs type="web" → 侧栏「联网搜索」组。
前端开关即总开关：开关否由请求级 contextvar（web_search_enabled）决定，
工具执行层据此短路，关闭时零网络请求。
"""
import logging
import re
import time
from typing import Tuple

import requests

from langchain_core.tools import tool

from src.config import settings, get_deepseek_model

logger = logging.getLogger(__name__)

# 引用条目默认上限
_WEB_MAX_ITEMS = 8
_HTTP_TIMEOUT = 30   # 兼容旧引用；实际值由 _web_http_timeout() 从 config web_search.timeout_sec 读取


def _web_http_timeout() -> int:
    """联网调用 HTTP 读取超时（秒）。

    v8.15.3b: DeepSeek 原生联网实测响应 33-50s——旧 30s 会把"本会成功的慢响应"
    自掐成伪失败（7×30s 白等根因）。取 config web_search.timeout_sec（默认 90），
    钳制在 30~300s（低于 30 视为误配）。
    """
    t = int(getattr(settings, "WEB_SEARCH_TIMEOUT", 90) or 90)
    return min(max(t, 30), 300)

# 引用/URL 提取正则（防御式兜底：Markdown 链接 + 裸 URL）
_MD_LINK_RE = re.compile(r"\[([^\]]{1,200})\]\s*\(\s*(https?://[^)\s]+)\s*\)", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"https?://[^\s\)\]\<>\",。；】》』！？]+", re.IGNORECASE)
# 内部锚点（2026-08 实测：API 在 web_search_call.action.url 后追加 #ws_call_id=call_xx，须剥离）
_WS_CALL_FRAG_RE = re.compile(r"#ws_call.*$", re.IGNORECASE)


def _clean_url(u: str) -> str:
    u = _WS_CALL_FRAG_RE.sub("", u or "").strip()
    return u.rstrip("，。；：！？、/.,;:!?)]}>")


def _responses_endpoint() -> str:
    """Responses 端点：{base}{path}。base 与主模型一致（api.deepseek.com）。"""
    base = (getattr(settings, "RESOLVED_MAIN_BASE_URL", None)
            or settings.MAIN_BASE_URL or "https://api.deepseek.com").rstrip("/")
    path = getattr(settings, "WEB_SEARCH_RESPONSES_PATH", "/v1/responses")
    return f"{base}{path}"


def _extract_urls_deep(obj, acc: list, depth: int = 0) -> None:
    """递归深扫任意嵌套 dict/list 找含 http 的 url/title（不确定层级时的通用兜底）。"""
    if depth > 8 or obj is None:
        return
    if isinstance(obj, dict):
        url = str(obj.get("url") or "").strip()
        title = str(obj.get("title") or "").strip()
        if url.lower().startswith("http"):
            acc.append({
                "title": title or url,
                "url": url,
                "abstract": str(obj.get("snippet") or obj.get("abstract") or "")[:500],
            })
        for v in obj.values():
            _extract_urls_deep(v, acc, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            _extract_urls_deep(v, acc, depth + 1)


def _parse_response_output(output: list):
    """解析 Responses 输出（DeepSeek 实测结构）。
    返回 (summary, calls, meta)；防御式：任何格式差异不抛异常，尽力提取 URL。
    """
    text_parts: list[str] = []
    queries: list[str] = []
    url_items: list[dict] = []
    for item in output or []:
        if not isinstance(item, dict):
            continue
        itype = item.get("type", "")
        if itype == "message":
            for c in item.get("content") or []:
                if isinstance(c, dict) and c.get("type") == "output_text":
                    t = str(c.get("text", ""))
                    text_parts.append(t)
                    # 行内 Markdown 链接 [title](url)
                    for m in _MD_LINK_RE.finditer(t):
                        url_items.append({"title": (m.group(1).strip() or m.group(2).strip()),
                                          "url": m.group(2).strip(), "abstract": ""})
                    # 裸 URL 兜底
                    for u in _BARE_URL_RE.findall(t):
                        url_items.append({"title": u, "url": u, "abstract": ""})
        elif itype == "web_search_call":
            act = item.get("action") or {}
            if isinstance(act, dict) and act.get("type") == "search":
                # 过滤 API 自己追加的伪查询项（ws_call_id=call_xx）
                queries.extend(str(q) for q in (act.get("queries") or [])
                               if not str(q).lower().startswith("ws_call_id="))
        # 任意深层字段扫描（action.url / annotations / file 块等未知结构）
        _extract_urls_deep(item, url_items)
    # 按 URL 去重（剥离 #ws_call_id 内部锚点 + 行内残留尾符）
    seen: set = set()
    calls: list[dict] = []
    for it in url_items:
        u = _clean_url(it["url"])
        if not u or u in seen:
            continue
        seen.add(u)
        calls.append({"title": (it["title"] or u)[:200], "url": u,
                      "abstract": it.get("abstract", "")[:500]})
    summary = "\n".join(p for p in text_parts if p).strip()
    return summary, calls[: _WEB_MAX_ITEMS], {"queries": queries[:8]}


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

    # v8.15: 输入附加"带网址引用"指令——DeepSeek 该版本 web_search_call 只回搜索动作、
    # 不返回结构化来源，须让模型在回答里明确写出真实网址（[标题](URL)），
    # 才能被解析进「联网搜索」证据组并在侧栏点击。
    _input_prompt = (
        f"{query}\n\n"
        "如果使用了联网搜索，请在回答中对引用的信息来源标注真实网址，"
        "格式如：[来源标题](https://...)。只列你实际引用且真实存在的网页地址。"
    )
    t0 = time.perf_counter()
    try:
        payload = {
            "model": get_deepseek_model(),
            "input": _input_prompt,
            "tools": [{"type": "web_search"}],   # 服务端内置 web_search：模型自主决定是否联网
            "stream": False,
        }
        headers = {
            "Authorization": f"Bearer {settings.RESOLVED_MAIN_API_KEY or settings.MAIN_API_KEY}",
            "Content-Type": "application/json",
        }
        url = _responses_endpoint()
        _to = _web_http_timeout()
        logger.info(f"[deepseek_web_search] Responses 调用: {url} model={payload['model']} timeout={_to}s")
        resp = requests.post(url, headers=headers, json=payload, timeout=_to)
        if resp.status_code != 200:
            # v8.15.3: 失败详情落日志（HTTP 状态码 + 响应体前 600 字）——
            # "7 次 30s 盲重试"的根因是失败原因全被吞（只看到 err=True）；
            # 403/429/500/网络/超时 由此一眼可辨，且工具回执带状态码让模型不再盲试
            logger.error(
                f"[deepseek_web_search] HTTP {resp.status_code} 响应体: {resp.text[:600]}")
            return (f"[ERR_NETWORK] 联网搜索失败 HTTP {resp.status_code}"
                    "（详情已记录日志）。连续失败时系统会自动熔断停止重试；"
                    "建议检查 config web_search.responses_path 与 API key 是否有效。",
                    empty)
        body = resp.json()
        summary, calls, meta = _parse_response_output(body.get("output") or [])
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

        _queries_line = ("\n\n本次模型联网检索了这些关键词:\n" + "、".join(
            f"「{q[:80]}」" for q in meta.get("queries", [])[:5])) if meta.get("queries") else ""
        text_result = (
            f"联网检索摘要（DeepSeek 原生搜索返回）:\n{summary[:2000] or '(模型判断无需联网，未检索)'}"
            + _queries_line
            + ("\n\n引用:\n" + "\n".join(f"[W{i}] {c['title']} — {c['url']}"
                                          for i, c in enumerate(calls, 1)) if calls else "")
        )
        content = _format_web_tool_result("deepseek_web_search", query, text_result,
                                          status="ok", results_count=len(items),
                                          elapsed_ms=elapsed)
        logger.info(f"[deepseek_web_search] done: {len(calls)} 引用, {len(summary)} 字摘要, "
                    f"{len(meta.get('queries', []))} 检索词, {elapsed:.0f}ms")
        return content, {"main_results": [], "web_results": items}
    except requests.exceptions.Timeout as e:
        # v8.15.3: 超时单独分支（超时 ≠ 403/429——盲重试与误改配置都源于此混为一谈）
        _to = _web_http_timeout()
        logger.error(f"[deepseek_web_search] 请求超时 ({_to}s): {e}")
        return (f"[ERR_NETWORK] 联网搜索超时（{_to}s 限制）。"
                "DeepSeek 原生联网响应本就 30-50s；若仍持续超时说明服务端异常或"
                "网络受限（需代理），连续失败时系统会自动熔断停止重试。",
                empty)
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
