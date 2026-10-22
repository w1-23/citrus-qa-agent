"""AgentState TypedDict — v8.1.1 streamlined."""
from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage


class AgentState(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    query: str
    visualization_enabled: bool
    request_deadline: float
    agent_no_prefetch_hints: bool
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

    # v10.0（卡点① Intake）: 意图判定结果（结构化）与用户确认后的意图（全链路锚点）
    intent_profile: Optional[dict]
    intent_confirmed: Optional[str]

    retrieval_context: Optional[str]
    references_data: Optional[dict]

    # v10.0（卡点②③）: audit_report / adjudication / citation_info 必须声明——
    # supervisor_node 明明 return 了这三个键（expert_graph.py:1207-1209），但 langgraph
    # 对未声明键不并入节点 update，于是 api/main.py:431-440 的
    #   output.get("citation_info") / output.get("audit_report") / output.get("adjudication")
    # 恒为 falsy，done 事件里这三个字段永远缺失。实测（v10_3 全部 79 条记录 done.audit 均为空）
    # 与 raw SSE 抓包一致。后果：
    #   * index.html:2767 的 _repaired 恒 false → 修复循环产出的**最终答案永远不上屏**，
    #     用户看到的是回审判为 critical 的那版草稿（修复过程按设计不流式上屏）；
    #   * "✓ 质检发现问题并已自动修正" 横幅、citation_unsupported/citation_mismatch
    #     假完成告警、"🧹 证据裁决：已舍弃 N 条" 提示全部不出现。
    # 与 tools_called 完全同一根因（见上 v9.1.3 注释），此处补齐声明。
    # 影响面仅限 done 事件载荷与前端展示：save 节点及其余节点均不读这三个键，
    # 故不改变任何生成内容。
    audit_report: dict
    adjudication: list
    citation_info: dict

    _trace: Optional[dict]
