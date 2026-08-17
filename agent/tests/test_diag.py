# -*- coding: utf-8 -*-
"""diag 结构化诊断层测试（v8.13）——事件写入/计时区间/永不抛异常。

CITRUS_LOG_DIR 由 conftest 指向 tests/.tmp_runner/logs，
diag 的 JSONL 落在其 diag/ 子目录，与生产 logs 隔离。
"""
import json
import os
from pathlib import Path

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def _diag_path() -> Path:
    import time
    root = Path(os.environ.get("CITRUS_LOG_DIR", "")) / "diag"
    return root / f"diag_{time.strftime('%Y-%m-%d')}.jsonl"


def _read_tail_events():
    p = _diag_path()
    if not p.exists():
        return []
    events = []
    with open(p, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return events


def test_diag_event():
    print("[DIAG-1] diag 事件写入 JSONL + 关联字段注入")
    from src.core.diag import diag
    diag("diag_test_event", a=1, text="你好")
    events = _read_tail_events()
    hit = [e for e in events if e.get("event") == "diag_test_event"]
    check("事件落盘", len(hit) >= 1, f"hits={len(hit)}")
    if hit:
        e = hit[-1]
        check("字段完整", e.get("a") == 1 and e.get("text") == "你好")
        check("关联字段注入", "req" in e and "session" in e and "job" in e)


def test_diag_span():
    print("[DIAG-2] diag_span 正常/异常路径")
    from src.core.diag import diag_span
    with diag_span("diag_test_span", tag="ok"):
        pass
    try:
        with diag_span("diag_test_span_err", tag="boom"):
            raise ValueError("boom")
    except ValueError:
        pass
    events = _read_tail_events()
    names = [e.get("event") for e in events]
    check("span start", "diag_test_span_start" in names)
    done = [e for e in events if e.get("event") == "diag_test_span_done"]
    check("span done 含 dur_ms", bool(done) and "dur_ms" in done[-1],
          f"dur_ms={done[-1].get('dur_ms') if done else None}")
    err = [e for e in events if e.get("event") == "diag_test_span_err_error"]
    check("span error 事件", bool(err)
          and err[-1].get("err") == "ValueError" and "dur_ms" in err[-1],
          str(err[-1] if err else None))


def test_diag_never_raises():
    print("[DIAG-3] 永不抛异常（不可序列化值/超长字段截断）")
    from src.core.diag import diag

    class _Unserializable:
        def __repr__(self):
            return "<unserializable>"

    try:
        diag("diag_test_safe", obj=_Unserializable(), long="x" * 2000)
        check("不可序列化值兜底", True)
    except Exception as e:
        check("不可序列化值兜底", False, str(e))
    events = _read_tail_events()
    hit = [e for e in events if e.get("event") == "diag_test_safe"]
    check("长字段截断", hit and len(str(hit[-1].get("long", ""))) <= 510,
          f"len={len(str(hit[-1].get('long', ''))) if hit else 0}")


print()
if __name__ == "__main__":
    test_diag_event()
    test_diag_span()
    test_diag_never_raises()
    print(f"diag tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        raise SystemExit(1)
