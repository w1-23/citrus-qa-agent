"""Tool Registry & Unified exports — v8.1.1"""
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

from src.core.registries import LIGHT_MODE_ALLOWED, EXPERT_MODE_ALLOWED


def get_tools_for_mode(mode: str, intent: str = "retrieve") -> list:
    """Dynamically assemble tools based on mode."""
    from src.config import FeatureFlags
    if FeatureFlags.is_enabled("light_mode") or mode == "light":
        base = LIGHT_MODE_ALLOWED
    else:
        base = EXPERT_MODE_ALLOWED
    return [t for t in _TOOL_REGISTRY if t.name in base]


def get_all_tools():
    from src.config import FeatureFlags
    tools = [t for t in _TOOL_REGISTRY]
    if FeatureFlags.is_enabled("light_mode"):
        tools = [t for t in tools if t.name in LIGHT_MODE_ALLOWED]
    return tools


def get_tools_for_agent(agent_name: str) -> list:
    from src.core.registries import AGENT_REGISTRY
    config = AGENT_REGISTRY.get(agent_name)
    if not config:
        return list(_TOOL_REGISTRY)
    whitelist = set(config.get("tools", []))
    return [t for t in _TOOL_REGISTRY if t.name in whitelist]


def get_tool_names():
    return [t.name for t in _TOOL_REGISTRY]
