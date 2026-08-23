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
import asyncio
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


def _to_web_items(calls: list[dict]) -> list[dict]:
    """引用条目 → web 证据组条目（与 deepseek_web_search 内部同构，供草稿路径复用）。"""
    items = []
    for idx, c in enumerate(calls, 1):
        items.append({
            "ref_id": f"W{idx}",
            "type": "web",
            "source": "web",
            "url": c["url"],
            "title": c["title"],
            "abstract": c.get("abstract", ""),
            "snippet": c.get("abstract", ""),
        })
    return items


def _responses_web_search(input_prompt: str) -> tuple[str, list[dict]]:
    """v8.17: 原生联网回答（草稿路径）——Responses + 原生 web_search 工具。

    与 deepseek_web_search 同端点/同超时/同解析，但为「用户原始问题直答」形态：
    返回 (summary 正文, 去重引用清单)；失败抛异常（调用方 fail-soft 跳过草稿）。
    """
    payload = {
        "model": get_deepseek_model(),
        "input": input_prompt,
        "tools": [{"type": "web_search"}],
        "stream": False,
    }
    headers = {
        "Authorization": f"Bearer {settings.RESOLVED_MAIN_API_KEY or settings.MAIN_API_KEY}",
        "Content-Type": "application/json",
    }
    url = _responses_endpoint()
    _to = _web_http_timeout()
    logger.info(f"[draft-web] Responses 调用: {url} model={payload['model']} timeout={_to}s")
    resp = requests.post(url, headers=headers, json=payload, timeout=_to)
    if resp.status_code != 200:
        raise RuntimeError(
            f"Responses HTTP {resp.status_code}: {resp.text[:300]}")
    body = resp.json()
    summary, calls, _meta = _parse_response_output(body.get("output") or [])
    if not summary and not calls:
        raise RuntimeError("联网无返回内容")
    return summary, calls


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

    # v8.15.3d: 输入构造——**用户原始问题优先**（chat_v2 经 contextvar 直传）：
    # DeepSeek 原生联网围绕原始问题作答（output_text 即针对原始问题的综述回答）；
    # 模型给的检索词降级为"搜索参考关键词"（双保险）。原始问题缺失时回退纯检索词。
    # 同时附加"带网址引用"指令——DeepSeek 该版本 web_search_call 只回搜索动作、
    # 不返回结构化来源，须让模型在回答里明确写出真实网址（[标题](URL)），
    # 才能被解析进「联网搜索」证据组并在侧栏点击。
    _orig_q = ""
    try:
        from src.core.tracing import original_query as _orig_query
        _orig_q = (_orig_query() or "").strip()
    except Exception:
        _orig_q = ""
    _ref_cmd = ("如果使用了联网搜索，请在回答中对引用的信息来源标注真实网址，"
                "格式如：[来源标题](https://...)。只列你实际引用且真实存在的网页地址。")
    if _orig_q and _orig_q != query:
        _input_prompt = (f"{_orig_q}\n\n"
                         f"（搜索参考关键词：{query}）\n\n{_ref_cmd}")
        logger.info(f"[deepseek_web_search] 原始问题直传：{_orig_q[:80]}...")
    else:
        _input_prompt = f"{query}\n\n{_ref_cmd}"
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
        # v8.15.3f: 摘要必须进 artifact——此前 summary 只进 content（retrieve-agent 自己看得到，
        # 但 call_retrieve_agent 的回执是代码用 collected_artifacts 组装的，只读 web_results=
        # 纯 URL 条目 → supervisor 只看到 8 个网址、无正文，落得 cited=0 的短回答）。
        # web_summary 单列一字段，经 collect→build_evidence_report 渲染进「网络综述」段。
        return content, {
            "main_results": [],
            "web_results": items,
            "web_summary": summary[:4000],
        }
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


# ═══════════════════════════════════════════════════════════════════════
# v8.16.1 草稿先行：分隔符结构化输出（非 JSON）
#   DRAFT_ZH（中文展示） / DRAFT_EN（HyDE 复用） / MULTI_QUERY / SUMMARY
# ═══════════════════════════════════════════════════════════════════════

_STRUCTURED_START = "===STRUCTURED_START==="
_STRUCTURED_END = "===STRUCTURED_END==="


class StructuredParseError(ValueError):
    """分隔符结构化区块解析失败（字段缺失/格式不符）。"""


def _strip_code_fence(block: str) -> str:
    """防御: 模型偶发包 ``` 围栏时剥离（提示词已约束，此仅为容错）。"""
    s = block.strip()
    if s.startswith("```"):
        s = s[3:]
        # 去掉语言标记行（如 ```text）
        nl = s.find("\n")
        if 0 < nl < 20:
            s = s[nl + 1:]
        if s.endswith("```"):
            s = s[:-3]
    return s.strip()


def _parse_structured_response(raw_text: str, require_answer: bool = True) -> dict:
    """从 DeepSeek 响应中提取结构化字段（分隔符定位，不依赖 JSON 解析）。

    规划 v8.16.1 §5.2 实现：
      - 联网回答正文 = 结构化区块之前的所有文本（草稿调用中通常为空）
      - 区块内按行 "字段名: 值"，MULTI_QUERY/SUMMARY 为竖线分隔列表
      - 必需字段缺失/为空 → 抛 StructuredParseError（调用方 fail-soft 跳过草稿）
    v8.17 修复（日志实证三次降级 parse_error=结构化区块未找到 而 raw 含完整回答）：
      - **容忍缺失 END 标记**——模型偶发在 ANSWER 后提前收尾（未写
        ===STRUCTURED_END===），此时取 START 之后全部文本为区块；
      - **ANSWER 字段 = DRAFT_ZH 别名**（v8.17 模板用 ANSWER，旧 DRAFT_ZH 兼容）；
      - **require_answer=False**：联网路径的二级提取块只有 MULTI_QUERY/SUMMARY
        （无 ANSWER），不强制草稿字段。

    Returns:
        {"answer": str, "draft_zh": str, "draft_en": str,
         "multi_query": list[str], "summary": list[str]}
    """
    if not raw_text or not raw_text.strip():
        raise StructuredParseError("结构化区块未找到（空响应）")
    start_idx = raw_text.find(_STRUCTURED_START)
    if start_idx == -1:
        raise StructuredParseError("结构化区块未找到")
    end_idx = raw_text.find(_STRUCTURED_END)
    if end_idx != -1 and end_idx <= start_idx:
        raise StructuredParseError("结构化区块未找到")
    answer_text = raw_text[:start_idx].strip()
    if end_idx == -1:
        # v8.17: 缺 END 标记 → 取 START 之后全部文本（DRAFT_ZH/ANSWER 是唯一必需字段，
        # 无歧义；MULTI_QUERY/SUMMARY 在其后按行解析，多余尾部行忽略）
        block = _strip_code_fence(
            raw_text[start_idx + len(_STRUCTURED_START):].strip())
    else:
        block = _strip_code_fence(
            raw_text[start_idx + len(_STRUCTURED_START):end_idx].strip())

    result: dict = {}
    current_field: str | None = None
    for line in block.split("\n"):
        line = line.strip()
        if not line:
            continue
        known = False
        key, value = "", ""
        if ":" in line:
            key, value = line.split(":", 1)
            key = key.strip().upper()
            value = value.strip()
            # v8.17: ANSWER 为原生回答草稿字段（别名归一为 DRAFT_ZH）
            if key == "ANSWER":
                key = "DRAFT_ZH"
            if key in ("DRAFT_ZH", "DRAFT_EN", "MULTI_QUERY", "SUMMARY"):
                known = True
                current_field = key
        if known:
            if key in ("MULTI_QUERY", "SUMMARY"):
                items = [v.strip() for v in value.split("|") if v.strip()]
                if items:
                    result[key] = items
            else:
                result[key] = value
        elif current_field in ("DRAFT_ZH", "DRAFT_EN"):
            # 续行容错：模型可能把 DRAFT_EN/DRAFT_ZH 段落折行
            # （无已知字段前缀的行归属当前字段；提示词已约束单行，此为防御）
            _prev = result.get(current_field, "")
            result[current_field] = (_prev + "\n" + line).strip() if _prev else line

    required = ("DRAFT_ZH",) if require_answer else ()
    for field in required:
        if field not in result:
            raise StructuredParseError(f"缺少字段: {field}")
        if not result[field]:
            raise StructuredParseError(f"字段为空: {field}")

    return {
        "answer": answer_text,
        "draft_zh": result.get("DRAFT_ZH", ""),
        # v8.17: 草稿=原生回答（ANSWER）；DRAFT_EN 为旧模板兼容字段
        "draft_en": result.get("DRAFT_EN", ""),
        "multi_query": result.get("MULTI_QUERY", []),
        "summary": result.get("SUMMARY", []),
    }


def _fast_llm_call(system_prompt: str, user_content: str) -> str | None:
    """统一 fast 模型调用（草稿 / 检索素材提取共用）。

    防线（v8.15.3c / v8.16.3 / v8.16.4 实战沉淀）：
      - 空 content 重试一次（v4-flash 思维链偶发吃光预算）
      - v8.16.4: 默认关思维链（thinking:disabled）；厂商参数被拒时第 1 次
        自动退回默认参数重试（fail-soft，防"无输出"）
      - 异常不吞掉（call_exception 标签由调用方区分）
    """
    if not system_prompt:
        return None
    from openai import OpenAI
    client = OpenAI(
        api_key=settings.RESOLVED_FAST_API_KEY,
        base_url=settings.RESOLVED_FAST_BASE_URL,
        timeout=max(int(getattr(settings, "DRAFT_TIMEOUT_SEC", 25) or 25), 5),
    )
    max_tokens = max(int(getattr(settings, "DRAFT_MAX_TOKENS", 2048) or 2048), 256)
    extra: dict = {}
    if getattr(settings, "DRAFT_THINKING_OFF", True):
        extra["extra_body"] = {"thinking": {"type": "disabled"}}
    answer = ""
    for attempt in (1, 2):
        try:
            resp = client.chat.completions.create(
                model=settings.RESOLVED_FAST_MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_content},
                ],
                temperature=0.2,
                max_tokens=max_tokens,
                **extra,
            )
        except Exception as e:
            if extra and attempt == 1:
                logger.warning(
                    f"[fast] thinking:disabled 参数被拒（{type(e).__name__}: {e}），"
                    f"退回默认参数重试一次")
                extra = {}
                continue
            raise
        answer = (resp.choices[0].message.content or "").strip()
        if answer:
            break
        logger.warning(f"[fast] 第 {attempt} 次调用返回为空，重试一次")
    return answer or None


def _call_structured_draft(query: str) -> str | None:
    """非联网 fast 模型调用：一次性产出分隔符结构化区块（ANSWER + MQ + SUMMARY）。

    v8.17: 模板升级为「原生回答草稿（ANSWER）+ 检索素材（MULTI_QUERY/SUMMARY）一体」——
    "后续提取也从里面提取"：检索素材与草稿同一次生成，不额外调用。
    防线与草稿历史一致（空 content 重试一次 / 异常不吞 / v8.16.4 关思维链 fail-soft）。
    """
    from src.prompts.loader import assemble_structured_output_prompt
    prompt = assemble_structured_output_prompt()
    if not prompt:
        logger.warning("[draft] structured_output.md 模板缺失，草稿跳过")
        return None
    return _fast_llm_call(prompt, f"用户问题：\n{query}")


def _call_extract_from_answer(query: str, answer: str) -> str | None:
    """v8.17: 联网模式专用二级提取——从原生联网回答中提炼 MULTI_QUERY/SUMMARY。

    满足"后续提取也从里面提取"（联网路径的原生回答是自由文本，需一次读取
    该回答的提取调用）；提取失败不阻塞草稿展示（调用方 fail-soft）。
    """
    from src.prompts.loader import assemble_structured_extract_prompt
    prompt = assemble_structured_extract_prompt()
    if not prompt:
        logger.warning("[draft] structured_extract.md 模板缺失，跳过检索素材提取")
        return None
    user_content = f"用户问题：\n{query}\n\n原生联网回答（草稿）：\n{answer[:4000]}"
    return _fast_llm_call(prompt, user_content)


def _fallback_draft_zh(raw: str) -> str:
    """v8.16.3: 结构化解析失败时的降级草稿文本（纯函数,worker 侧调用）。

    模板保持严格不弱化——降级只发生在解析层：
    v8.17 改进（日志实证 raw 含完整回答仅缺 END 标记）——优先取 START 之后
    的正文（剥去 ANSWER:/DRAFT_ZH: 等字段前缀与分隔符），上限 300 字，
    避免 raw[:200] 拦腰截断残句。
    """
    if not raw:
        return ""
    text = raw
    idx = text.find("===STRUCTURED_START===")
    if idx > 0:
        # 区块前正文优先（模型偶发"开头回答+区块"形态，旧行为兼容）
        text = text[:idx]
    elif idx == 0:
        # 仅区块形态（v8.17 日志实证）：取 START 之后内容并剥字段前缀
        text = text[len("===STRUCTURED_START==="):]
    for pat in ("===STRUCTURED_END===", "ANSWER:", "DRAFT_ZH:", "DRAFT_EN:"):
        text = text.replace(pat, "")
    # 按行过滤代码围栏（首字符 ``` 的整行），避免 replace("```") 破坏围栏后文本
    lines = [ln.strip() for ln in text.splitlines()
             if ln.strip() and not ln.lstrip().startswith("```")]
    return re.sub(r"[ \t]+", " ", "\n".join(lines)).strip()[:300]


def _draft_search_multi(queries: list) -> list:
    """草稿通道多路检索（独立包装，测试可打桩）。

    v8.17 恢复草稿检索并入：从原生回答提取的 MULTI_QUERY/SUMMARY 喂 search_multi，
    结果并入 retrieve-agent 证据（draft_extra_count）；种子查询=原始问题，覆盖
    citrus_rag_search 自身 HyDE 结构化未首轮覆盖的角度。
    """
    if not queries:
        return []
    from src.tools.search import _get_rag
    rag = _get_rag()
    return rag.search_multi(queries)


async def draft_worker(query: str, session_id: str) -> None:
    """v8.17 草稿先行后台任务（load 节点 create_task 启动，fire-and-forget）。

    用户要求（v8.17 决策）：
      - **原生回答用于草稿**：一次 DeepSeek API 调用直接回答原始问题 → 草稿展示；
      - **联网模式只决定该调用是否带原生联网**：web 开 → Responses+原生
        web_search（回答即联网综述，附 [Wn] 引用）；web 关 → 非联网 fast 调用
        （ANSWER + MULTI_QUERY + SUMMARY 一体分隔符输出）；
      - **后续提取也从里面提取**：检索素材（MULTI_QUERY/SUMMARY）从原生回答
        提炼（web 关同一调用内提取；web 开为读取回答的二级提取调用）；
      - **HyDE 部分额外调用**：由 citrus_rag_search 内部独立结构化生成（本链路不动）；
      - **最后融合原生回答进行生成**：草稿（原生回答）经 draft_store → agent_runner
        并入确定性回执「原生回答参考」段，supervisor 融合生成最终回答；
      - 草稿检索并入恢复：提取出的 MQ/SUMMARY 喂 search_multi，结果进 [n] 证据。

    注意: 本任务执行严格 fail-soft——任何异常只记日志，绝不阻塞主链路。
    v8.16.2: 业务日志 draft_done / draft_skipped——"草稿未显示"类问题第一现场。
    """
    if not getattr(settings, "DRAFT_ENABLED", True):
        return
    query = (query or "").strip()
    if not query:
        return
    from src.core.progress_bus import emit_draft
    try:
        from src.core.tracing import web_search_enabled as _req_web_on
        web_mode = bool(_req_web_on())
    except Exception:
        web_mode = False

    draft_zh: str = ""
    parsed: dict = {}
    web_items: list = []

    if web_mode:
        # ── 联网路径：原生联网回答（草稿）→ 二级提取检索素材 ──
        try:
            summary, calls = await asyncio.to_thread(_responses_web_search, query)
        except Exception as e:
            logger.warning(f"[draft-web] 原生联网回答失败（跳过草稿）: {e}")
            _draft_blog("draft_skipped",
                        reason=f"call_exception:{type(e).__name__}:{str(e)[:120]}",
                        mode="web")
            return
        draft_zh = (summary or "").strip()
        web_items = _to_web_items(calls)
        if not draft_zh:
            logger.info("[draft-web] 联网回答无正文，跳过")
            _draft_blog("draft_skipped", reason="call_empty", mode="web")
            return
        try:
            raw_extract = await asyncio.to_thread(
                _call_extract_from_answer, query, draft_zh)
            if raw_extract:
                # v8.17: 提取块仅 MULTI_QUERY/SUMMARY（无 ANSWER），不强制草稿字段
                parsed = _parse_structured_response(raw_extract, require_answer=False)
        except StructuredParseError as e:
            logger.warning(f"[draft-web] 检索素材提取格式异常（跳过并入）: {e}")
        except Exception as e:
            logger.warning(f"[draft-web] 检索素材提取异常（跳过并入）: {e}")
    else:
        # ── 非联网路径：一次 fast 调用产出 原生回答(ANSWER) + 检索素材(MQ/SUMMARY) ──
        try:
            raw = await asyncio.to_thread(_call_structured_draft, query)
        except Exception as e:
            logger.warning(f"[draft] 结构化草稿调用异常（跳过）: {e}")
            _draft_blog("draft_skipped",
                        reason=f"call_exception:{type(e).__name__}:{str(e)[:120]}")
            return
        if not raw:
            logger.info("[draft] 两次调用均返回空（fast 模型无 content）")
            _draft_blog("draft_skipped", reason="call_empty")
            return
        try:
            parsed = _parse_structured_response(raw)
            draft_zh = parsed["draft_zh"]
        except StructuredParseError as e:
            # 降级路径：非空但格式断裂 → 剥标记取正文作草稿；检索并入跳过
            fallback = _fallback_draft_zh(raw)
            if len(fallback) >= 10:
                draft_zh = fallback
                try:
                    emit_draft(draft_zh, str(
                        getattr(settings, "DRAFT_LABEL", "预检索草稿·验证中")
                        or "预检索草稿·验证中"),
                        source="web" if web_mode else "local")
                    logger.info(f"[draft] 降级草稿已推送前端 ({len(draft_zh)} 字, q={query[:40]!r})")
                except Exception as _e2:
                    logger.debug(f"[draft] emit_draft 失败: {_e2}")
                _draft_blog("draft_fallback", zh_len=len(draft_zh),
                            parse_error=str(e)[:120], raw_preview=raw[:200])
            else:
                _draft_blog("draft_skipped", reason=f"parse_error:{e}",
                            raw_preview=raw[:200])
            return
        except Exception as e:
            logger.warning(f"[draft] 草稿解析异常（跳过）: {e}")
            _draft_blog("draft_skipped", reason=f"parse_exception:{str(e)[:120]}")
            return

    # ── 草稿展示（原生回答）──
    # v8.17 修订（修正2）: 草稿=API 原生返回，成功路径**不截断/不降级/不摘要**
    # （完整内容上屏；模板约束约 300-600 字，前端面板高度自适应）；
    # source 标识随事件下发（修正4：前端展示 draft.source 徽标）。
    _source = "web" if web_mode else "local"
    try:
        emit_draft(draft_zh, str(
            getattr(settings, "DRAFT_LABEL", "预检索草稿·验证中")
            or "预检索草稿·验证中"), source=_source)
        logger.info(f"[draft] 草稿已推送前端 ({len(draft_zh)} 字, "
                    f"{_source} 模式, q={query[:40]!r})")
    except Exception as e:
        logger.debug(f"[draft] emit_draft 失败: {e}")

    # ── 检索素材并入：MQ/SUMMARY 从原生回答提取 → 多路检索（种子=原始问题）──
    extra_retrieval = bool(getattr(settings, "DRAFT_EXTRA_RETRIEVAL", True))
    results: list = []
    queries_n = 0
    if extra_retrieval and parsed:
        mq = [q for q in (parsed.get("multi_query") or [])[:3] if q and q.strip()]
        sm = [s for s in (parsed.get("summary") or [])[:3] if s and s.strip()]
        queries = [query] + mq + sm
        queries_n = len(queries)
        if queries_n > 1:
            try:
                results = await asyncio.to_thread(_draft_search_multi, queries)
            except Exception as e:
                logger.warning(f"[draft] 草稿多路检索失败（跳过并入）: {e}")
                results = []

    # ── 落仓（原生回答 + 检索结果 + web 引用）→ agent_runner 并入回执/融合 ──
    try:
        from src.core.draft_store import draft_store
        from src.core.tracing import get_request_id
        draft_store.put(session_id, get_request_id(), {
            "answer_text": draft_zh,          # 原生回答 → 「原生回答参考」段（融合）
            "results": results,               # 草稿多路检索 → [n] 证据
            "web_items": web_items,           # 联网路径引用 → [Wn] 证据
            "queries_n": queries_n,
            "web_mode": web_mode,
        })
    except Exception as e:
        logger.debug(f"[draft] draft_store.put 失败: {e}")

    _draft_blog("draft_done", zh_len=len(draft_zh), queries_n=queries_n,
                items=len(results), web=len(web_items), mode="web" if web_mode else "local")


def _draft_blog(event: str, **fields) -> None:
    """v8.16.2: 草稿业务日志（fail-soft：日志失败绝不影响草稿链路）。"""
    try:
        from src.core.business_logger import blog
        blog(event, **fields)
    except Exception:
        pass
