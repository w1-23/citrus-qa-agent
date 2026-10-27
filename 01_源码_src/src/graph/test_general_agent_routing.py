from src.graph.general_agent_graph import (
    _explicit_data_audit,
    _explicit_export_request,
    _explicit_note_save_request,
    _local_corpus_matches,
    _scoped_toolset,
    _source_scope,
)


def test_local_corpus_match_is_limited_to_citrus_topics():
    assert _local_corpus_matches("Find citrus papers on Huanglongbing resistance")
    assert _local_corpus_matches("检索柑橘黄龙病抗性相关文献")
    assert not _local_corpus_matches("检索 PaperConan 的数据审查方法")
    assert not _local_corpus_matches("Why is this orange-colored?")


def test_source_scope_still_respects_explicit_local_and_web_constraints():
    assert _source_scope("只查本地柑橘文献") == "local"
    assert _source_scope("只查网页上的柑橘文献") == "web"
    assert _source_scope("只从本地柑橘文献语料检索关于黄龙病抗性的英文文献") == "local"
    assert _source_scope("只从网页搜索黄龙病抗性相关论文") == "web"


def test_data_audit_router_requires_positive_data_review_intent():
    assert _explicit_data_audit("请帮我审查这份实验数据")
    assert _explicit_data_audit("检查表格有没有异常值")
    assert not _explicit_data_audit("只告诉我附件中的 DOI 和标题；不要审查数值或生成报告")
    assert not _explicit_data_audit("附件只是参考，不需要分析里面的表格")
    assert _explicit_data_audit("不要审查附件，但请检查这份数据是否有异常")


def _tool_names(query: str, scope: str = "auto", **kwargs) -> set[str]:
    tools, _ = _scoped_toolset(scope, query=query, **kwargs)
    return {tool.name for tool in tools}


def test_tool_capabilities_are_reduced_to_the_requested_workflow():
    assert _tool_names("你好") == set()

    audit = _tool_names("请审查这份实验表格有没有异常", has_table=True)
    assert audit == {"inspect_uploaded_table", "audit_uploaded_data"}

    literature = _tool_names("检索柑橘黄龙病抗性相关文献")
    assert {"deepseek_search", "deepseek_fetch", "deepseek_fetch_paper",
            "local_hybrid_search"}.issubset(literature)
    assert not (literature & {"audit_uploaded_data", "save_research_note",
                             "export_experiment_design", "export_research_notes"})

    web_only = _tool_names("仅联网回答柑橘黄龙病相关研究", scope="web")
    assert "local_hybrid_search" not in web_only
    assert "deepseek_search" in web_only

    local_only = _tool_names("只查本地柑橘文献", scope="local")
    assert local_only == {"local_hybrid_search"}

    assert _tool_names("画一个数据流程图", visualization_enabled=True) == {"draw_svg"}


def test_project_notes_and_exports_are_only_available_on_explicit_intent():
    assert _explicit_note_save_request("请把这个决定保存到柑橘课题记录")
    assert _tool_names("请把这个决定保存到柑橘课题记录") == {"save_research_note"}
    assert _tool_names("导出柑橘课题记录") == {"export_research_notes"}
    assert "export_literature_evidence" in _tool_names("把检索到的文献导出为 RIS")
    assert "export_literature_evidence" not in _tool_names("检索柑橘抗病相关文献")
    assert _explicit_export_request("导出为 BibTeX")
