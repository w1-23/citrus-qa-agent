"""Prompt Cache 命中率观测（阶段0 · 上下文工程改造）.

DeepSeek 上下文缓存按块自动启用，usage 返回:
  prompt_cache_hit_tokens / prompt_cache_miss_tokens
LangChain 只把标准字段放进 usage_metadata；provider 原始 usage 保留在
response_metadata["token_usage"]（或 "usage"）。

本模块统一提取缓存字段:
  - 向 context_usage SSE 事件追加 cache_hit / cache_miss / cache_ratio
  - 按 source 累计统计，定期输出 INFO 日志
作为静态前缀重构（阶段1）前后的对比基线。
"""
from __future__ import annotations

import logging
import threading

logger = logging.getLogger(__name__)

_AGG_INTERVAL = 100  # 每 N 次调用输出一次聚合日志

_lock = threading.Lock()
_agg: dict = {}          # source -> {"hit": int, "miss": int, "calls": int}
_global_calls = 0


# 首次 N 次调用输出原始 usage 键集合（命中率恒 0 时一眼定位 provider 字段位置）
_USAGE_SAMPLE_LIMIT = 5
_usage_samples_logged = 0


def extract_usage(response) -> dict | None:
    """归一化 usage，含 prompt cache 字段。取不到 total 时返回 None。

    v8.4.4: 首次 5 次调用打印原始 usage 键集合（DeepSeek 的 prompt_cache
    字段可能不在 usage_metadata 而在 response_metadata 的 token_usage/usage，
    命中率恒 0 时据此定位适配点）。
    """
    try:
        um = getattr(response, "usage_metadata", None) or {}
        rm = getattr(response, "response_metadata", {}) or {}
        raw = rm.get("token_usage") or rm.get("usage") or {}
        merged: dict = {}
        # v8.4.4: DeepSeek raw usage 用 prompt_tokens/completion_tokens 命名，
        # usage_metadata 用 input/output_tokens——两者都要读
        for key, alt in (("input_tokens", "prompt_tokens"),
                         ("output_tokens", "completion_tokens"),
                         ("total_tokens", "total_tokens")):
            val = um.get(key) or raw.get(key) or raw.get(alt) or 0
            merged[key] = int(val or 0)
        merged["cache_hit"] = int(
            raw.get("prompt_cache_hit_tokens")
            or um.get("prompt_cache_hit_tokens") or 0)
        merged["cache_miss"] = int(
            raw.get("prompt_cache_miss_tokens")
            or um.get("prompt_cache_miss_tokens") or 0)
        if not merged["total_tokens"]:
            return None
        # v8.4.4: 首次样本输出 usage 键集合（诊断用，不改变正常路径）
        global _usage_samples_logged
        if _usage_samples_logged < _USAGE_SAMPLE_LIMIT:
            _usage_samples_logged += 1
            logger.info(
                f"[CacheMetrics] usage 样本#{_usage_samples_logged}: "
                f"usage_metadata_keys={sorted(um.keys())} "
                f"raw_keys={sorted(raw.keys())} "
                f"cache_hit={merged['cache_hit']} cache_miss={merged['cache_miss']}"
            )
        return merged
    except Exception as e:
        logger.debug(f"[CacheMetrics] extract_usage failed: {e}")
        return None


def _record(source: str, cache_hit: int, cache_miss: int) -> None:
    global _global_calls
    with _lock:
        entry = _agg.setdefault(source, {"hit": 0, "miss": 0, "calls": 0})
        entry["hit"] += cache_hit
        entry["miss"] += cache_miss
        entry["calls"] += 1
        _global_calls += 1
        if _global_calls % _AGG_INTERVAL == 0:
            total_hit = sum(e["hit"] for e in _agg.values())
            total_miss = sum(e["miss"] for e in _agg.values())
            denom = total_hit + total_miss
            ratio = (total_hit / denom) if denom else 0.0
            per_source = ", ".join(
                f"{s}={e['hit']}/{e['hit']+e['miss']}"
                for s, e in sorted(_agg.items())
            )
            logger.info(
                f"[CacheMetrics] 累计 {_global_calls} 次调用: "
                f"前缀缓存命中率 {ratio:.1%} "
                f"(hit={total_hit}, miss={total_miss}) | 按来源: {per_source}"
            )



def reset_cache_stats() -> None:
    global _agg, _global_calls
    with _lock:
        _agg.clear()
        _global_calls = 0


def emit_usage_from_response(session_id: str, source: str, response) -> None:
    """从 LLM 响应提取 usage（含缓存字段）并推送 context_usage 增量。

    替代各调用点重复的 usage_metadata 内联提取（expert_graph/light_graph/agent_runner）。
    """
    try:
        merged = extract_usage(response)
        if not merged:
            return
        from src.core.progress_bus import emit_usage_delta
        emit_usage_delta(
            session_id, source,
            input_tokens=merged.get("input_tokens", 0),
            output_tokens=merged.get("output_tokens", 0),
            total=merged.get("total_tokens", 0),
            cache_hit=merged.get("cache_hit", 0),
            cache_miss=merged.get("cache_miss", 0),
        )
        _record(source, merged.get("cache_hit", 0), merged.get("cache_miss", 0))
    except Exception:
        pass
