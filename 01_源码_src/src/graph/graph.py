"""Root StateGraph — one general tool-using agent for both UI modes.

v8.1.1: simple mode dispatch, no LLM routing.
v8.10r: 删除死代码 run_query（无调用，入口在 api/main.py）。
v9.2: 删除未使用的 logger（grep 证实全文件零引用）。
"""
from langgraph.graph import StateGraph

from src.graph.general_agent_graph import build_general_agent_graph

_general_graph = build_general_agent_graph()


def build_graph(mode: str = "light") -> StateGraph:
    """Return the shared general-agent graph; ``mode`` is kept for API compatibility."""
    return _general_graph
