# -*- coding: utf-8 -*-
"""真机 probe：api.deepseek.com 对思维链关闭字段的接受度（v8.17.19 验证工具）。

直接用项目 .env 的真实 key/base_url/model 发最小请求：
  基线（无字段） → 4 种关闭字段逐一对比 + tools（retrieve-agent 形态）组合。
判断标准（用户口径）：HTTP 200 + 响应无 reasoning_content + 耗时明显下降 = 字段有效。
验证结论（2026-08-24 实测）：extra_body thinking:disabled 有效（官方端点，
响应无 reasoning_content）；reasoning/enable_thinking 无效；reasoning_effort:none
亦表现为关闭但官方首选仍是 thinking。config.yaml model.reasoning_off_body 默认值
即验证通过字段，零改动。
用法: cd agent && python -m src.tools.thinking_probe
"""
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request


def load_env():
    d = {}
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
    with io.open(p, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            d[k.strip().upper()] = v.strip()
    return d


def main():
    env = load_env()
    key = env.get("DEEPSEEK_API_KEY") or env.get("MAIN_API_KEY")
    base = (env.get("MAIN_BASE_URL") or "https://api.deepseek.com").rstrip("/")
    model = env.get("MAIN_MODEL") or "deepseek-v4-flash"
    if not key:
        print("NO KEY in .env")
        return 1

    # 1) 模型清单（确认 deepseek-v4-flash 在该端点真实存在）
    try:
        req = urllib.request.Request(
            base + "/v1/models",
            headers={"Authorization": "Bearer " + key})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode("utf-8"))
        names = [m.get("id") for m in data.get("data", [])]
        print("models:", names)
        print("target in list:", model in names)
    except Exception as e:
        print("models list failed:", e)

    # 2) 字段矩阵（基线 + 4 种关闭字段，各发 1 次）
    cases = [
        ("baseline(no field)", None),
        ("thinking disabled", {"thinking": {"type": "disabled"}}),
        ("reasoning enabled=false", {"reasoning": {"enabled": False}}),
        ("reasoning_effort none", {"reasoning_effort": "none"}),
        ("enable_thinking false", {"enable_thinking": False}),
    ]
    for name, extra in cases:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "1+1等于几？请直接给出答案。"}],
            "max_tokens": 400,
        }
        if extra:
            body.update(extra)
        t0 = time.time()
        try:
            req = urllib.request.Request(
                base + "/chat/completions",
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            msg = data["choices"][0]["message"]
            rc = msg.get("reasoning_content")
            dt = time.time() - t0
            print("[%s] HTTP 200  %.1fs  reasoning_content=%s  answer=%r" % (
                name, dt, "YES" if rc else "no", str(msg.get("content") or "")[:40]))
        except urllib.error.HTTPError as e:
            print("[%s] HTTP %s  %s" % (name, e.code, e.read().decode("utf-8", "ignore")[:200]))
        except Exception as e:
            print("[%s] ERROR %s" % (name, e))
    # 3) retrieve-agent 真实形态：tools 数组 + thinking:disabled（bind_tools 场景）
    try:
        body = {
            "model": model,
            "messages": [{"role": "user", "content": "检索柑橘黄龙病相关文献。"}],
            "max_tokens": 400,
            "thinking": {"type": "disabled"},
            "tools": [{
                "type": "function",
                "function": {
                    "name": "citrus_rag_search",
                    "description": "本地检索",
                    "parameters": {
                        "type": "object",
                        "properties": {"query": {"type": "string"}},
                        "required": ["query"],
                    },
                },
            }],
        }
        t0 = time.time()
        req = urllib.request.Request(
            base + "/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": "Bearer " + key,
                     "Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=90) as r:
            data = json.loads(r.read().decode("utf-8"))
        msg = data["choices"][0]["message"]
        rc = msg.get("reasoning_content")
        tcs = msg.get("tool_calls")
        dt = time.time() - t0
        print("[tools+thinking disabled] HTTP 200  %.1fs  reasoning_content=%s  tool_calls=%s" % (
            dt, "YES" if rc else "no", [t.get("function", {}).get("name") for t in (tcs or [])]))
        print("    FINISH_REASON:", data["choices"][0].get("finish_reason"))
    except urllib.error.HTTPError as e:
        print("[tools+thinking disabled] HTTP %s  %s" % (e.code, e.read().decode("utf-8", "ignore")[:200]))
    except Exception as e:
        print("[tools+thinking disabled] ERROR %s" % (e,))
    # 4) Responses 端点（deepseek_web_search 内部调用形态）：纯对话（不带 web_search 工具）
    #    验证 thinking 关闭字段是否被接受 + reasoning 是否消失。
    try:
        base2 = base
        resp_path = (getattr(__import__("src.config", fromlist=["settings"]).settings,
                             "WEB_SEARCH_RESPONSES_PATH", "/v1/responses") or "/v1/responses")
        for rname, rbody_extra in (
            ("responses baseline", None),
            ("responses thinking disabled", {"thinking": {"type": "disabled"}}),
        ):
            body = {
                "model": model,
                "input": "1+1等于几？请直接给出答案。",
                "stream": False,
            }
            if rbody_extra:
                body.update(rbody_extra)
            t0 = time.time()
            req = urllib.request.Request(
                base2 + resp_path,
                data=json.dumps(body).encode("utf-8"),
                headers={"Authorization": "Bearer " + key,
                         "Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=90) as r:
                data = json.loads(r.read().decode("utf-8"))
            raw = json.dumps(data, ensure_ascii=False)
            has_reasoning = ("reasoning" in raw.lower() or "reasoning_content" in raw.lower())
            dt = time.time() - t0
            print("[%s] HTTP 200  %.1fs  reasoning_field=%s" % (
                rname, dt, "YES" if has_reasoning else "no"))
    except urllib.error.HTTPError as e:
        print("[responses] HTTP %s  %s" % (e.code, e.read().decode("utf-8", "ignore")[:200]))
    except Exception as e:
        print("[responses] ERROR %s" % (e,))
    return 0


if __name__ == "__main__":
    sys.exit(main())