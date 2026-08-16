"""Agent / Skill registries — v8.3.1 simplified."""
from typing import Dict

AGENT_REGISTRY: Dict = {
    "retrieve-agent": {
        "description": "Multi-source literature search: local RAG + academic DBs.",
        "tools": ["citrus_rag_search", "academic_search"],
        "max_turns": 1,
        "react": True,
        "system_prompt": "agents/retrieve-agent.md",
    },
    "write-agent": {
        "description": "Academic writing: reviews, reports, structured answers. Context-based only, no search.",
        "tools": ["write_local_file"],
        "max_turns": 6,
        "react": True,
        "system_prompt": "agents/write-agent.md",
    },
    "analyze-agent": {
        "description": "Statistical analysis and experiment design.",
        "tools": ["statistical_analysis", "experimental_design"],
        "max_turns": 2,
        "react": False,
        "system_prompt": "agents/analyze-agent.md",
    },
}


def get_agent_names():
    return list(AGENT_REGISTRY.keys())

