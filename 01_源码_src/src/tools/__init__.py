"""Active tool registry for the single General Agent architecture."""
from src.tools.registry import register_tool, get_tool_spec, PartitionedToolNode, init_tool_registry
from src.tools.registry import get_tools_by_category, cleanup_offload_files, get_offload_file_list
from src.tools.deepseek_search_fetch import (
    deepseek_search, deepseek_fetch, deepseek_fetch_paper, local_hybrid_search,
)
from src.tools.draw_svg import draw_svg
from src.research.tools import (
    inspect_uploaded_table, audit_uploaded_data, export_literature_evidence,
    verify_uploaded_references, fetch_and_audit_open_supplements,
    export_paper_analysis, export_experiment_design, save_research_note,
    recall_research_notes, export_research_notes, export_corpus_gap_report,
)

_TOOL_REGISTRY = [
    deepseek_search, deepseek_fetch, deepseek_fetch_paper, local_hybrid_search,
    draw_svg, inspect_uploaded_table, audit_uploaded_data,
    verify_uploaded_references, fetch_and_audit_open_supplements,
    export_literature_evidence, export_paper_analysis, export_experiment_design,
    save_research_note, recall_research_notes, export_research_notes,
    export_corpus_gap_report,
]
_TOOL_REGISTRY_BY_NAME: dict[str, object] = {t.name: t for t in _TOOL_REGISTRY}


def get_tool_names():
    return [t.name for t in _TOOL_REGISTRY]


# Register the code-owned tool set and retain categories for execution tracing.
init_tool_registry()

# Active read-only tools are code-owned; stale YAML entries from earlier releases
# are ignored by the registry loader.
for _tool, _category, _key in (
    (deepseek_search, "web", "deepseek_search"),
    (deepseek_fetch, "web", "deepseek_fetch"),
    (deepseek_fetch_paper, "web", "deepseek_fetch_paper"),
    (local_hybrid_search, "rag", "local_hybrid_search"),
    (inspect_uploaded_table, "research", "research_data"),
    (audit_uploaded_data, "research", "research_data"),
    (verify_uploaded_references, "research", "research_references"),
    (fetch_and_audit_open_supplements, "research", "research_web_fetch"),
    (export_literature_evidence, "research", "research_artifacts"),
    (export_paper_analysis, "research", "research_artifacts"),
    (export_experiment_design, "research", "research_artifacts"),
    (save_research_note, "research", "research_notes"),
    (recall_research_notes, "research", "research_notes"),
    (export_research_notes, "research", "research_artifacts"),
    (export_corpus_gap_report, "research", "research_artifacts"),
):
    if get_tool_spec(_tool.name) is None:
        register_tool(_tool, category=_category, is_readonly=True, concurrency_key=_key)
