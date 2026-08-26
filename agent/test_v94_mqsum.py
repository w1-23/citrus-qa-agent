# -*- coding: utf-8 -*-
"""mq_sum 模式单测：枚举合法、组合=M3Q+SUM、未知降级 full、raw 跳过 HyDE。"""
import sys
from pathlib import Path

AGENT = Path(r"E:\codex_WORKSPACES\Citrus_QA_Agent\agent")
sys.path.insert(0, str(AGENT))

from src.tools.search import _VALID_QUERY_MODES, _compose_queries  # noqa

hid = {"hyde": "H", "multi_query": ["q1", "q2", "q3", "q4"],
       "summary": ["s1", "s2", "s3", "s4", "s5", "s6"]}

# 1. 枚举包含 mq_sum
assert "mq_sum" in _VALID_QUERY_MODES, "mq_sum 未注册"
print("1) _VALID_QUERY_MODES:", _VALID_QUERY_MODES)

# 2. mq_sum = MQx3 + SUMx5，无 HyDE
qs = _compose_queries("Q", hid, "mq_sum")
assert qs == ["q1", "q2", "q3", "s1", "s2", "s3", "s4", "s5"], qs
assert "H" not in qs
print("2) mq_sum ->", qs)

# 3. 未知模式降级 full（含 HyDE 首批）
qs2 = _compose_queries("Q", hid, "nonsense")
assert qs2 == ["H", "q1", "q2", "q3", "s1", "s2", "s3", "s4", "s5"], qs2
print("3) unknown -> full:", qs2)

# 4. hyde_parsed=None -> None（调方走单路）
assert _compose_queries("Q", None, "mq_sum") is None
print("4) hyde_parsed=None -> None OK")

# 5. mq_only/sum_only 不回归
assert _compose_queries("Q", hid, "mq_only") == ["q1", "q2", "q3"]
assert _compose_queries("Q", hid, "sum_only") == ["s1", "s2", "s3", "s4", "s5"]
print("5) mq_only/sum_only 回归 OK")

# 6. full 模式空路源 -> 原始查询保底
assert _compose_queries("Q", {"hyde": "", "multi_query": [], "summary": []},
                        "full") == ["Q"]
print("6) full 空路源保底 OK")

print("\nALL PASS")