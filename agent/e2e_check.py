# -*- coding: utf-8 -*-
"""v8.13-b5b HTTP 端到端检查脚本：起服务后逐条提问，核对答案与引用。

用法（服务先起，rag-agent 环境）:
    rag-agent\\python.exe e2e_check.py --query "..." [--query "..."]

流程:
  1. 轮询 GET /health 直至就绪（最多 --wait 秒）
  2. 对每个 --query POST /api/v2/chat（expert 模式）读 SSE 流
  3. 打印 done 事件（answer/gen_time_ms/tools_called）与 citations 事件（引用条数+前几条）
  4. 全部成功 exit 0；任一查询无 answer 或流内出现 error -> exit 1
"""
import argparse
import json
import sys
import time

try:
    import httpx
except ImportError:
    print("httpx missing (rag-agent env required)")
    sys.exit(2)

EVENT_TYPES_OK = {"status", "progress", "tool_executing", "heartbeat", "citations"}


def wait_health(base: str, timeout: int) -> dict:
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            r = httpx.get(f"{base}/health", timeout=5)
            if r.status_code == 200:
                return r.json()
        except Exception:
            pass
        time.sleep(2)
    raise SystemExit(f"[FAIL] /health 未就绪（{timeout}s 内）")


def chat_once(base: str, query: str, session_id: str, timeout: int):
    payload = {"query": query, "session_id": session_id, "light_mode": False}
    answer = None
    done_meta = {}
    citations = []
    errors = []
    seen_raw = []

    def handle(etype, inner):
        nonlocal answer, done_meta
        if etype == "done":
            answer = inner.get("answer", "") if isinstance(inner, dict) else ""
            done_meta = {k: inner.get(k) for k in
                         ("session_id", "job_id", "gen_time_ms",
                          "tools_called", "citation_count",
                          "has_references") if isinstance(inner, dict) and k in inner}
        elif etype == "citations":
            items = inner if isinstance(inner, list) else inner.get("items", [])
            citations.extend(items or [])
        elif etype == "error":
            errors.append(json.dumps(inner, ensure_ascii=False)[:300])

    try:
        with httpx.stream("POST", f"{base}/api/v2/chat", json=payload,
                          timeout=timeout) as resp:
            if resp.status_code != 200:
                return None, [], [f"HTTP {resp.status_code}"], []
            cur_event = ""
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("event:"):
                    cur_event = line[6:].strip()
                    continue
                if not line.startswith("data:"):
                    continue
                raw = line[5:].strip()
                if len(seen_raw) < 3:
                    seen_raw.append(f"e={cur_event or '?'} d={raw[:120]}")
                try:
                    obj = json.loads(raw)
                except Exception:
                    obj = raw
                etype = cur_event
                inner = obj
                if isinstance(obj, dict) and ("event" in obj or "type" in obj or "data" in obj):
                    etype = obj.get("event") or obj.get("type") or etype
                    inner = obj.get("data", obj)
                    if isinstance(inner, str):
                        try:
                            inner = json.loads(inner)
                        except Exception:
                            pass
                cur_event = ""
                handle(etype, inner)
    except Exception as e:
        errors.append(f"stream error: {e}")
    return answer, citations, errors, seen_raw, done_meta


def main():
    ap = argparse.ArgumentParser(description="HTTP 端到端检查")
    ap.add_argument("--query", action="append", required=True)
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--session", default=None, help="固定会话 ID（默认自动）")
    ap.add_argument("--wait", type=int, default=180)
    ap.add_argument("--timeout", type=int, default=600)
    args = ap.parse_args()

    health = wait_health(args.base, args.wait)
    print(f"[OK] /health: {health.get('status', health)}")

    session_id = args.session or f"e2e-{int(time.time())}"
    all_ok = True
    for i, q in enumerate(args.query, 1):
        print(f"\n=== Q{i} (session={session_id}) ===")
        print(f"    {q[:100]}")
        answer, citations, errors, raw_sample, done_meta = chat_once(args.base, q, session_id, args.timeout)
        if raw_sample:
            print(f"    [raw sample] {' || '.join(raw_sample)}")
        if errors:
            print(f"    [ERR] {'; '.join(errors)}")
            all_ok = False
        if not answer:
            print("    [FAIL] 无 answer")
            all_ok = False
            continue
        print(f"    --- answer ({len(answer)} chars) ---")
        print("    " + answer.replace("\n", "\n    ")[:700])
        if citations:
            print(f"    --- citations ({len(citations)}) ---")
            for c in citations[:6]:
                if isinstance(c, dict):
                    title = str(c.get("title", c.get("paper_id", c.get("id", ""))))[:70]
                    batch = str(c.get("_batch", c.get("batch", "")))
                    print(f"      * {title}" + (f" [batch={batch}]" if batch and batch != "None" else ""))
                else:
                    print(f"      * {str(c)[:90]}")
        print("    --- meta ---")
        print(f"    {json.dumps(done_meta, ensure_ascii=False)}")

    print(f"\n{'PASS: 全部查询完成且有答案' if all_ok else 'FAIL: 存在异常'}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())