"""AgentState TypedDict — v8.1.1 streamlined."""
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    session_id: str
    mode: str
    # v8.3.7: 幂等键（save 节点写入历史时用，防重发重复）
    idempotency_key: str
    # v8.3.8: 本轮新增轨迹（含 tool_calls/ToolMessage 配对），save 节点完整持久化
    turn_trace: list[BaseMessage]
    # v8.3.8: 历史检索证据块（跨轮复用，load 注入）
    history_evidence_block: Optional[str]

    answer: str
    gen_time_ms: float

    history_summary: Optional[str]
    long_term_memory: Optional[str]
    search_suggestions: list[str]
    format_hint: Optional[str]

    retrieval_context: Optional[str]
    references_data: Optional[dict]

    _trace: Optional[dict]
