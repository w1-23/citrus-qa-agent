"""缓存观测适配测试（v8.4.4 项1）——extract_usage 三种 usage 形态提取。"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.core.cache_metrics import extract_usage, reset_cache_stats, _usage_samples_logged

passed, failed = [], []


def check(name, cond, detail=""):
    if cond:
        passed.append(name)
    else:
        failed.append(name)
        if os.environ.get("PYTEST_CURRENT_TEST"):
            raise AssertionError(name + (f" {detail}" if detail else ""))
    print(f"  {'PASS' if cond else 'FAIL'}  {name}  {detail}")


def _mk_response(um=None, rm=None):
    class R:
        usage_metadata = um or {}
        response_metadata = rm or {}
    return R()


def test_extract_usage_forms():
    print("[CU-1] usage 三种形态提取")
    # 形态1: usage_metadata 标准字段 + 无缓存字段
    r1 = _mk_response(um={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150})
    u1 = extract_usage(r1)
    check("usage_metadata 形态", u1 and u1["total_tokens"] == 150
          and u1["cache_hit"] == 0 and u1["cache_miss"] == 0, str(u1))

    # 形态2: DeepSeek raw usage 在 response_metadata.token_usage（含缓存字段）
    r2 = _mk_response(rm={"token_usage": {
        "prompt_tokens": 200, "completion_tokens": 50, "total_tokens": 250,
        "prompt_cache_hit_tokens": 180, "prompt_cache_miss_tokens": 20,
    }})
    u2 = extract_usage(r2)
    check("token_usage 形态 + 缓存字段", u2 and u2["total_tokens"] == 250
          and u2["cache_hit"] == 180 and u2["cache_miss"] == 20
          and u2["input_tokens"] == 200, str(u2))

    # 形态3: response_metadata.usage（部分 provider 路径）
    r3 = _mk_response(rm={"usage": {
        "prompt_tokens": 90, "total_tokens": 110,
        "prompt_cache_hit_tokens": 90, "prompt_cache_miss_tokens": 0,
    }})
    u3 = extract_usage(r3)
    check("usage 形态", u3 and u3["total_tokens"] == 110
          and u3["cache_hit"] == 90 and u3["cache_miss"] == 0, str(u3))

    # 无 total → None
    r4 = _mk_response(um={})
    check("无 total 返回 None", extract_usage(r4) is None)

    # 无响应对象 → None（异常兜底）
    check("异常兜底", extract_usage(None) is None)


def test_usage_sample_logging():
    print("[CU-2] 首次样本日志开关（不改变提取结果）")
    reset_cache_stats()
    global _usage_samples_logged
    saved = _usage_samples_logged
    r = _mk_response(um={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2})
    for _ in range(8):
        extract_usage(r)
    check("样本日志限量 5 次", _usage_samples_logged - saved <= 5,
          f"logged={_usage_samples_logged - saved}")


print()
if __name__ == "__main__":
    test_extract_usage_forms()
    test_usage_sample_logging()
    print(f"cache metrics tests: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("FAILED:", failed)
        sys.exit(1)
