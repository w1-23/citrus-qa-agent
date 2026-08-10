"""Root StateGraph — routes light/expert based on mode flag.

v8.1.1: simple mode dispatch, no LLM routing.
"""
import logging
from typing import Optional

from langgraph.graph import StateGraph

from src.graph.state import AgentState
from src.graph.light_graph import build_light_graph
from src.graph.expert_graph import build_expert_graph

logger = logging.getLogger(__name__)

_light_graph = build_light_graph()
_expert_graph = build_expert_graph()


def build_graph(mode: str = "light") -> StateGraph:
    """Return compiled graph for given mode."""
    if mode == "expert":
        return _expert_graph
    return _light_graph


async def run_query(
    query: str, session_id: str, mode: str = "light"
) -> str:
    """Simple async API."""
    graph = build_graph(mode)
    state: AgentState = {
        "query": query,
        "session_id": session_id,
        "mode": mode,
        "messages": [],
        "answer": "",
    }
    final_state = None
    async for event in graph.astream(state, stream_mode="values"):
        final_state = event
    return final_state.get("answer", "") if final_state else ""
