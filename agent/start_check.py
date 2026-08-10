# -*- coding: utf-8 -*-
"""启动验证：后台起 uvicorn → 轮询 /health → 打印关键启动日志 → 关闭。"""
import subprocess
import sys
import time
import urllib.request
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(os.environ.get("TEMP", "/tmp"), "citrus_uvicorn_test.log")
ERR = LOG + ".err"

proc = subprocess.Popen(
    [sys.executable, "-m", "uvicorn", "src.api.main:app", "--port", "8011"],
    cwd=ROOT,
    stdout=open(LOG, "w", encoding="utf-8"),
    stderr=open(ERR, "w", encoding="utf-8"),
)

t0 = time.time()
ok = False
try:
    while time.time() - t0 < 300:
        time.sleep(3)
        try:
            with urllib.request.urlopen("http://127.0.0.1:8011/health", timeout=3) as r:
                body = r.read().decode()
                if r.status == 200:
                    print(f"[OK] /health -> {body} (after {time.time()-t0:.0f}s)")
                    ok = True
                    break
        except Exception:
            pass
    if ok:
        # 应用日志写入 agent/logs/agent.log（非 stdout），等待 RAG 预热完成检查 idx_map（AG-11）
        app_log = os.path.join(ROOT, "logs", "agent.log")
        deadline = time.time() + 90
        idx_ok = 0
        idx_bad = 0
        degraded = False
        while time.time() < deadline:
            time.sleep(2)
            try:
                log_text = open(app_log, encoding="utf-8", errors="replace").read()
            except Exception:
                log_text = ""
            idx_ok = log_text.count("idx_map ok")
            idx_bad = log_text.count("idx_map match rate LOW")
            degraded = ("降级为 BM25" in log_text
                        or "Failed to load Qdrant" in log_text
                        or "批次: 0" in log_text)
            if idx_ok >= 5 or degraded:
                break
        print(f"[RAG] idx_map ok: {idx_ok} 批次 | match LOW: {idx_bad} | BM25降级: {degraded}")
        if idx_ok >= 5 and not degraded:
            print("[OK] 向量检索就绪（5 批次全部加载）")
        else:
            print("[WARN] 向量检索异常 — 请检查 data/ 与是否有其他实例占用")
finally:
    proc.terminate()
    try:
        proc.wait(timeout=10)
    except Exception:
        proc.kill()
    time.sleep(1)

if not ok:
    print("[FAIL] /health 未在 300s 内就绪")
    print("--- stderr tail ---")
    if os.path.exists(ERR):
        lines = open(ERR, encoding="utf-8", errors="replace").read().splitlines()
        print("\n".join(lines[-40:]))
    sys.exit(1)
