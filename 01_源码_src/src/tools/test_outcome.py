from src.tools.outcome import normalize_tool_outcome


def test_tool_outcome_distinguishes_complete_partial_no_result_failure_and_not_run():
    assert normalize_tool_outcome("audit_uploaded_data", "数据审查已完成：重复值 1 条。详情见异常报告。[打开异常报告](/api/v2/session/s/artifacts/aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa)").status == "complete"
    partial = normalize_tool_outcome("audit_uploaded_data", "数据审查未完整完成，未完成范围不能判断为无异常。")
    assert partial.status == "partial"
    not_run = normalize_tool_outcome("audit_uploaded_data", "审查未启动：当前轮没有选择附件。")
    assert not_run.status == "not_run"
    assert normalize_tool_outcome("local_hybrid_search", "", {"local_status": "hit", "main_results": []}).status == "no_results"
    assert normalize_tool_outcome("deepseek_search", "[WEB_TOOL_ERROR] Search failed", {"web_results": []}).status == "failed"
    # Natural-language “not found” from a source is not promoted to a failure.
    assert normalize_tool_outcome("verify_uploaded_references", "登记库未查到记录。", {}).status == "complete"
    mixed = normalize_tool_outcome(
        "verify_uploaded_references",
        "已核验 2 条 DOI：metadata_matched 1 条、unverified 1 条。", {})
    assert mixed.status == "partial"
    assert normalize_tool_outcome(
        "fetch_and_audit_open_supplements", "公开补充材料获取未完成：TimeoutException。", {}).status == "failed"
