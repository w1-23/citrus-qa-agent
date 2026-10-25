from src.graph.general_agent_graph import (
    _explicit_data_audit,
    _local_corpus_matches,
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
