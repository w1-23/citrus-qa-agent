"""Tool Registry & Unified exports — v8.3.1"""
from src.tools.registry import register_tool, get_tool_spec, PartitionedToolNode, init_tool_registry
from src.tools.registry import get_tools_by_category, cleanup_offload_files, get_offload_file_list
from src.tools.search import (citrus_rag_search, pdf_read, academic_search)
from src.tools.analyze import (statistical_analysis, experimental_design)
from src.tools.readfile import read_local_file
from src.tools.file_ops import write_local_file

_TOOL_REGISTRY = [
    citrus_rag_search, pdf_read, academic_search,
    read_local_file, write_local_file,
    statistical_analysis, experimental_design,
]
_TOOL_REGISTRY_BY_NAME: dict[str, object] = {t.name: t for t in _TOOL_REGISTRY}


def get_tool_names():
    return [t.name for t in _TOOL_REGISTRY]
