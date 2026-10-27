"""Normalize per-call token usage and provider prompt-cache telemetry.

Cache counts are reported only when the provider supplies cache fields. Missing
fields remain ``unavailable``; they are never converted into a zero-hit claim.
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


def _mapping(value) -> dict:
    if isinstance(value, dict):
        return value
    if hasattr(value, "model_dump"):
        try:
            return value.model_dump()
        except Exception:
            pass
    if hasattr(value, "__dict__"):
        return {k: v for k, v in vars(value).items() if not k.startswith("_")}
    return {}


def _first_present(containers: list[dict], names: tuple[str, ...]) -> tuple[bool, int]:
    for container in containers:
        for name in names:
            if name in container and container[name] is not None:
                try:
                    return True, max(0, int(container[name]))
                except (TypeError, ValueError):
                    continue
    return False, 0


def _first_int(first: dict, second: dict, names: tuple[str, ...]) -> int:
    found, value = _first_present([first, second], names)
    return value if found else 0


def extract_usage(response) -> dict | None:
    """Normalize one provider call; missing cache fields remain explicitly unknown."""
    try:
        response_map = _mapping(response)
        um = _mapping(getattr(response, "usage_metadata", None)
                       or response_map.get("usage_metadata") or {})
        rm = _mapping(getattr(response, "response_metadata", None)
                      or response_map.get("response_metadata") or {})
        raw = _mapping(rm.get("token_usage") or rm.get("usage")
                       or response_map.get("token_usage") or response_map.get("usage") or {})
        # OpenAI Python SDK objects expose usage separately from response_metadata.
        raw_usage = _mapping(getattr(response, "usage", None)
                             or response_map.get("usage") or {})
        if raw_usage:
            raw = {**raw, **raw_usage}
        containers = [raw, um]
        for container in (raw, um):
            for key in ("prompt_tokens_details", "input_token_details", "prompt_token_details"):
                nested = _mapping(container.get(key))
                if nested:
                    containers.append(nested)

        input_tokens = _first_int(um, raw, ("input_tokens", "prompt_tokens"))
        output_tokens = _first_int(um, raw, ("output_tokens", "completion_tokens"))
        total_tokens = _first_int(um, raw, ("total_tokens",)) or input_tokens + output_tokens
        hit_found, cache_hit = _first_present(
            containers, ("cached_tokens", "prompt_cache_hit_tokens", "cache_read_tokens", "cache_read"))
        miss_found, cache_miss = _first_present(
            containers, ("prompt_cache_miss_tokens", "cache_creation_input_tokens",
                         "cache_creation_tokens", "cache_miss"))
        cache_status = "reported" if hit_found or miss_found else "unavailable"
        if cache_status == "reported" and not miss_found and input_tokens:
            cache_miss = max(input_tokens - cache_hit, 0)
        if not (input_tokens or output_tokens or total_tokens):
            return None
        merged = {"input_tokens": input_tokens, "output_tokens": output_tokens,
                  "total_tokens": total_tokens, "cache_hit": cache_hit,
                  "cache_miss": cache_miss, "cache_status": cache_status}
        global _usage_samples_logged
        if _usage_samples_logged < _USAGE_SAMPLE_LIMIT:
            _usage_samples_logged += 1
            logger.info("[CacheMetrics] usage sample#%s input=%s output=%s cache=%s hit=%s miss=%s",
                        _usage_samples_logged, input_tokens, output_tokens,
                        cache_status, cache_hit, cache_miss)
        return merged
    except Exception as e:
        logger.debug("[CacheMetrics] extract_usage failed: %s", e)
        return None

def _record(source: str, cache_hit: int, cache_miss: int, cache_status: str) -> None:
    global _global_calls
    with _lock:
        entry = _agg.setdefault(source, {"hit": 0, "miss": 0, "calls": 0, "unknown": 0})
        if cache_status == "reported":
            entry["hit"] += cache_hit
            entry["miss"] += cache_miss
            entry["calls"] += 1
        else:
            entry["unknown"] += 1
        _global_calls += 1
        if _global_calls % _AGG_INTERVAL == 0:
            total_hit = sum(e["hit"] for e in _agg.values())
            total_miss = sum(e["miss"] for e in _agg.values())
            denom = total_hit + total_miss
            ratio = (total_hit / denom) if denom else 0.0
            per_source = ", ".join(
                f"{s}={e['hit']}/{e['hit']+e['miss']} (unknown={e['unknown']})"
                for s, e in sorted(_agg.items())
            )
            logger.info(
                f"[CacheMetrics] 累计 {_global_calls} 次调用: "
                f"服务商已报告缓存命中率 {ratio:.1%} "
                f"(hit={total_hit}, miss={total_miss}; unknown={sum(e['unknown'] for e in _agg.values())}) | 按来源: {per_source}"
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
            cache_status=merged.get("cache_status", "unavailable"),
        )
        _record(source, merged.get("cache_hit", 0), merged.get("cache_miss", 0),
                merged.get("cache_status", "unavailable"))
        # v8.13: 结构化诊断事件（llm_usage）——一次调用一行 JSONL，覆盖所有 LLM 调用点
        from src.core.diag import diag
        diag("llm_usage", source=source,
             input_tokens=merged.get("input_tokens", 0),
             output_tokens=merged.get("output_tokens", 0),
             total_tokens=merged.get("total_tokens", 0),
             cache_hit=merged.get("cache_hit", 0),
             cache_miss=merged.get("cache_miss", 0),
             cache_status=merged.get("cache_status", "unavailable"))
    except Exception:
        pass
