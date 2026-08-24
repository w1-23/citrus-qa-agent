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
    # v8.3.8 修复: main_results/web_results 必须声明——langgraph 对未声明键不合并进
    # 下游节点 state，导致 save 节点证据账本为空（复测发现）
    main_results: list
    web_results: list
    # v9.1.3（用户日志: request_done tools=0 统计失真）: tools_called 同样必须声明——
    # supervisor 返回的 tool_names_called 列表未经声明会被 langgraph 丢弃，
    # 下游 request_done 的 tools 统计恒 0（与 supervisor_done 口径不一致误导排查）。
    tools_called: list

    answer: str
    gen_time_ms: float

    history_summary: Optional[str]
    long_term_memory: Optional[str]
    resident_cards: Optional[str]     # v8.4: 常驻卡片（双层记忆"概览"层）
    search_suggestions: list[str]
    format_hint: Optional[str]

    retrieval_context: Optional[str]
    references_data: Optional[dict]

    # v8.15: 联网搜索开关（前端逐请求下发；有效值 = 请求开关 && config web_search.enabled）
    web_search_enabled: Optional[bool] = None

    _trace: Optional[dict]
