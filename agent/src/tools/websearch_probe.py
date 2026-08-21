# -*- coding: utf-8 -*-
"""websearch_probe — DeepSeek Responses API web_search 能力探测（v8.15 P3）。

联调前先跑本脚本核对官方文档三要素：
  1. 端点精确路径（config web_search.responses_path，默认 /v1/responses）
  2. 当前模型（deepseek-v4-flash / deepseek-chat）是否在支持列表
  3. 返回 output 里 web_search_call 的精确字段名（title/url/status/content）

用法（真机，需有效 API key）：
  python -m src.tools.websearch_probe --run
  python -m src.tools.websearch_probe --query "最新柑橘黄龙病防治 2026" --model deepseek-chat

不带 --run 时仅打印当前配置与端点推导，不发请求（离线安全）。
"""
import argparse
import json


def _dry_run() -> None:
    from src.config import settings, get_deepseek_model
    from src.tools.deepseek_web import _responses_endpoint
    print("── websearch_probe 配置检查（离线）──")
    print(f"  web_search.enabled       = {settings.WEB_SEARCH_ENABLED}")
    print(f"  web_search.provider      = {settings.WEB_SEARCH_PROVIDER}")
    print(f"  web_search.responses_path= {settings.WEB_SEARCH_RESPONSES_PATH}")
    print(f"  model(当前解析)          = {get_deepseek_model()}")
    print(f"  推导端点                 = {_responses_endpoint()}")
    print("  有 API key               =",
          bool(settings.RESOLVED_MAIN_API_KEY or settings.MAIN_API_KEY))
    print("\n提示: 请到 DeepSeek 官方文档核对端点/支持模型/引用字段后，再以 --run 实测。")


def _run(query: str, model: str) -> None:
    from src.config import settings
    from src.tools.deepseek_web import _responses_endpoint, _parse_response_output
    import requests, time
    key = settings.RESOLVED_MAIN_API_KEY or settings.MAIN_API_KEY
    if not key:
        print("[ERR] 无 API key（state/api_key 或 .env DEEPSEEK_API_KEY）")
        return
    url = _responses_endpoint()
    payload = {"model": model, "input": query, "tools": [{"type": "web_search"}], "stream": False}
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    t0 = time.perf_counter()
    print(f"POST {url}")
    print(f"  payload.model = {model}")
    resp = requests.post(url, headers=headers, json=payload, timeout=60)
    print(f"  HTTP {resp.status_code} ({time.perf_counter()-t0:.1f}s)")
    if resp.status_code != 200:
        print("  响应体:", resp.text[:2000])
        return
    body = resp.json()
    out = body.get("output") or []
    print(f"  output item 数量 = {len(out)}")
    for it in out:
        if isinstance(it, dict):
            print(f"    - type={it.get('type','?')}")
    summary, calls = _parse_response_output(out)
    print(f"  解析总结文本长度 = {len(summary)}")
    print(f"  解析引用条目数   = {len(calls)}")
    for c in calls[:8]:
        print(f"    [W] {c['title'][:60]} — {c['url'][:80]}")
    if not calls:
        print("  ⚠️ 未解析到 web_search_call 条目——请对照官方文档核对字段名，")
        print("     并在 src/tools/deepseek_web.py:_parse_response_output 中修正。")
    # 打印原始结构供人工核对（截断）
    print("\n── 原始 output 精简结构（前 3 项，校验用）──")
    for it in out[:3]:
        print(json.dumps(it, ensure_ascii=False)[:600])


def main() -> None:
    ap = argparse.ArgumentParser(description="DeepSeek Responses web_search 探测")
    ap.add_argument("--run", action="store_true", help="实际发请求实测（需 API key）")
    ap.add_argument("--query", default="最新柑橘黄龙病防治研究进展", help="探测查询")
    ap.add_argument("--model", default=None, help="模型名（默认取当前解析模型）")
    args = ap.parse_args()
    if not args.run:
        _dry_run()
        return
    from src.config import get_deepseek_model
    model = args.model or get_deepseek_model()
    _run(args.query, model)


if __name__ == "__main__":
    main()