"""Root StateGraph — routes light/expert based on mode flag.

v8.1.1: simple mode dispatch, no LLM routing.
v8.10r: 删除死代码 run_query（无调用，入口在 api/main.py）。
v9.2: 删除未使用的 logger（grep 证实全文件零引用）。
"""
from langgraph.graph import StateGraph

from src.graph.light_graph import build_light_graph
from src.graph.expert_graph import build_expert_graph

_light_graph = build_light_graph()
_expert_graph = build_expert_graph()


def build_graph(mode: str = "light") -> StateGraph:
    """Return compiled graph for given mode."""
    if mode == "expert":
        return _expert_graph
    return _light_graph
