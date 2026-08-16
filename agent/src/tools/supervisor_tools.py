"""Supervisor tool schemas — expert 图 bind_tools 的单一来源.

阶段1(静态前缀)/阶段3(工具治理): 从 expert_graph.py 内联 _AGENT_TOOLS 提取。
schema 顺序固定、内容字节级稳定 —— 工具定义是静态前缀的一部分，
重排或改写会使变动点之后的 Prompt Cache 失效（书 2.3 铁律）。

注意: 本文件的描述文本是"何时用/何时不用/边界"的决策条件，
修改须同步审查 decision_guide.md 与其一致性（阶段3 工具描述治理）。
"""
from __future__ import annotations

_SUPERVISOR_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "call_retrieve_agent",
            "description": "Search academic literature for citrus research. "
                           "Use for factual queries, mechanisms, comparisons, reviews. "
                           "Query must be English keywords (5-15 words), NOT a full sentence. "
                           "If first search is insufficient, call again with synonyms or "
                           "narrower terms (up to 3 different angles).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "English search keywords (5-15 words)",
                    },
                    "goal": {
                        "type": "string",
                        "description": "What to retrieve and why",
                    },
                },
                "required": ["query", "goal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_write_agent",
            "description": "Write, compose, or synthesize NEW content (reviews, reports, structured "
                           "answers) from material and save it to a file. MUST call this when the user "
                           "asks to WRITE/COMPOSE/DRAFT content like a review, report or article — "
                           "even simple content like 'write 111 to test.md' when it requires writing. "
                           "Do NOT use when the content ALREADY EXISTS in the conversation "
                           "(e.g. 'save this answer'): use write_local_file directly instead. "
                           "For complex writing, ensure sufficient literature has been retrieved first. "
                           "context can be a finished document (will be saved directly) or "
                           "raw material to synthesize.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Writing goal",
                    },
                    "context": {
                        "type": "string",
                        "description": "Previous retrieval results or document content to base writing on",
                    },
                    "output_path": {
                        "type": "string",
                        "description": "File path to save (only if user requested saving)",
                    },
                },
                "required": ["goal", "context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "call_analyze_agent",
            "description": "Perform statistical analysis or design experiments. "
                           "Call when user asks for data analysis, statistics, "
                           "or experiment planning.",
            "parameters": {
                "type": "object",
                "properties": {
                    "goal": {
                        "type": "string",
                        "description": "Analysis or experiment design goal",
                    },
                    "data_context": {
                        "type": "string",
                        "description": "Data or background for analysis",
                    },
                },
                "required": ["goal", "data_context"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_local_file",
            "description": "Read a local file from disk (.pdf, .md, .txt, .csv, .xlsx). "
                           "Reads the FULL file by default (no truncation). "
                           "Use FIRST when user asks to read/open/view a local file. "
                           "Absolute paths resolve under the project directory; "
                           "relative paths resolve from workspace/. "
                           "For academic paper content extraction, prefer pdf_read.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute path (e.g. E:/docs/paper.pdf) or relative path (e.g. upload/data.csv)",
                    },
                    "max_chars": {
                        "type": "integer",
                        "description": "Max chars to read (default 0 = full file, no truncation). Pass a positive number to limit (e.g. 5000).",
                    },
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "pdf_read",
            "description": "Extract abstract and key sections from an academic research PDF. "
                           "Use for literature PAPERS found via DOI, search results, or online sources. "
                           "Do NOT use for local files (use read_local_file instead). "
                           "Suitable for extracting structured content like Abstract, Introduction, Methods, Results, Conclusion. "
                           "Returns plain text with section headers.",
            "parameters": {
                "type": "object",
                "properties": {
                    "file_path": {
                        "type": "string",
                        "description": "File path to the PDF paper (absolute or relative to workspace)",
                    },
                    "cross_reference": {
                        "type": "boolean",
                        "description": "Whether to validate metadata via CrossRef (default false)",
                    },
                },
                "required": ["file_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_local_file",
            "description": "Directly save EXISTING finished content to a file (workspace/output only). "
                           "Use when the user asks to save/preserve text that ALREADY EXISTS in the "
                           "conversation (e.g. save this answer, save this paragraph) — content is "
                           "written verbatim WITHOUT rewriting. "
                           "Do NOT use for writing new content: when the user asks to WRITE, COMPOSE, "
                           "or synthesize a review/report/article from material, use call_write_agent instead.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "File path relative to workspace/output/ (e.g. 'notes/answer.md')",
                    },
                    "content": {
                        "type": "string",
                        "description": "The exact finished content to save (verbatim, no rewriting)",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["write", "append"],
                        "description": "'write' overwrites, 'append' appends (default write)",
                    },
                },
                "required": ["path", "content"],
            },
        },
    },
]


def get_supervisor_tool_schemas() -> list:
    """返回 supervisor 工具 schema 列表（固定顺序，调用方不得重排/修改）。"""
    return _SUPERVISOR_TOOL_SCHEMAS


def get_supervisor_tool_names() -> list[str]:
    return [t["function"]["name"] for t in _SUPERVISOR_TOOL_SCHEMAS]
