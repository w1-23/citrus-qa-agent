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
