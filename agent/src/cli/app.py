"""CLI — Rich terminal interface for Citrus QA Agent v8.1.1"""
import asyncio
import logging
import re
import time

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.markdown import Markdown
from rich.live import Live
from rich.prompt import Prompt

console = Console(force_terminal=True)

from src.logger import setup_logging
setup_logging()


def _strip_tags(text: str) -> str:
    """Strip XML-like tags and known garbled artifacts from LLM output."""
    # XML/HTML tags only
    text = re.sub(r"<[/]?\??[A-Za-z_]+\s*[^>]*>", "", text)
    # Remaining angle-bracket artifacts (not Chinese text)
    text = re.sub(r"<[^>\u4e00-\u9fff]*>", "", text)
    return text


async def run_query(query: str, mode: str = "light"):
    from src.graph.graph import build_graph
    from src.graph.state import AgentState
    import os

    graph = build_graph(mode)
    state: AgentState = {
        "query": query,
        "session_id": os.environ.get("CITRUS_SESSION", "default"),
        "mode": mode,
        "messages": [],
        "main_results": [],
        "web_results": [],
        "answer": "",
    }

    t0 = time.perf_counter()
    console.print(Panel(f"[bold]Citrus QA Agent v8.1.1[/bold]  Mode: [cyan]{mode.upper()}[/cyan]  Query: {query[:80]}", expand=False))

    # Live progress display using Rich Text (not Table — avoids Row API issues)
    final = {}
    trace_lines = []
    start_wall = time.perf_counter()

    def _render_trace():
        wall = time.perf_counter() - start_wall
        header = f"T+{wall:5.1f}s"
        lines = "\n".join(trace_lines) if trace_lines else "  [waiting...]"
        return f"{header}\n{lines}"

    with Live(Panel(_render_trace(), title="Execution", border_style="dim"), refresh_per_second=4, console=console) as live:
        async for node_output in graph.astream(state, stream_mode="updates"):
            for node_name, output in node_output.items():
                trace = output.get("_trace") if isinstance(output, dict) else None
                if trace:
                    wall = time.perf_counter() - start_wall
                    dt = trace.get("elapsed_ms", 0)
                    node = trace.get("node", node_name)
                    summary = trace.get("summary", "")
                    icon = "W" if "W" in str(summary) else ">"
                    trace_lines.append(f"{wall:5.1f}s {icon} {node:<20} {summary}")
                    live.update(Panel(_render_trace(), title="Execution", border_style="dim"))

                # Collect answer and references for final display (v8.1.1 node names)
                if node_name in ("supervisor", "light_synthesize", "light_react", "chitchat"):
                    if output.get("answer"):
                        final.update(output)
                elif node_name in ("build_references", "validate_citations"):
                    ref = (output or {}).get("references_data") or output.get("citations_data") if output else {}
                    if ref and isinstance(ref, dict):
                        final["references_data"] = ref

    elapsed_graph = time.perf_counter() - t0
    answer = final.get("answer", "(no answer)")
    answer = _strip_tags(answer)

    console.print()
    t_render = time.perf_counter()
    console.print(Markdown(answer))
    dt_render = (time.perf_counter() - t_render) * 1000

    console.print()
    console.print(Panel("[bold]References[/bold]", expand=False))
    t_ref = time.perf_counter()
    _render_references(final.get("references_data", final.get("citations_data", {})))
    dt_ref = (time.perf_counter() - t_ref) * 1000

    elapsed_total = time.perf_counter() - t0
    console.print(f"[dim]Total: {elapsed_total:.1f}s (graph {elapsed_graph:.1f}s + ans {dt_render:.0f}ms + refs {dt_ref:.0f}ms)[/dim]")


_MAX_REF_ROWS = {
    "rag": 8,
    "multi_search": 10,
    "web": 6,
}


def _render_references(ref_data: dict):
    if not ref_data or not isinstance(ref_data, dict):
        console.print("  [dim](no references)[/dim]")
        return
    rag = ref_data.get("rag", []) or []
    ms = ref_data.get("multi_search", []) or []
    web = ref_data.get("web", []) or []
    total = ref_data.get("total", len(rag) + len(ms) + len(web))

    if total == 0:
        console.print("  [dim](no references)[/dim]")
        return

    table = Table(show_header=True, header_style="bold", expand=True, box=None, padding=(0, 1))
    table.add_column("#", width=4, justify="right")
    table.add_column("Source", width=16)
    table.add_column("DOI / URL", width=36)
    table.add_column("Details", width=46)

    def _add_section(title, items, max_rows):
        if not items:
            return
        table.add_section()
        table.add_row(f"[bold]{title} ({len(items)})[/bold]", "", "", "")
        shown = 0
        for item in items:
            if shown >= max_rows:
                remaining = len(items) - shown
                table.add_row("...", f"[dim]+{remaining} more[/dim]", "", "")
                break
            ref_id = item.get("ref_id", "?")
            tag = item.get("type", "?")
            is_rag = "RAG" in str(tag) or "rag" in str(tag).lower()
            is_web = any(kw in str(tag).lower() for kw in ["web", "联网", "百科", "tavily", "serper"])
            if is_rag:
                doi = item.get("doi", "")[:35]
                title_text = (item.get("title", "") or "")[:60]
                section_name = (item.get("section_name", "") or "")[:24]
                table.add_row(ref_id, f"[green]{tag}[/green]", f"[cyan]{doi}[/cyan]", f"[bold]{title_text}[/bold]\n[dim]{section_name}[/dim]")
            elif is_web:
                url = item.get("url", "")[:50]
                title = (item.get("title", "") or "")[:60]
                snippet = (item.get("snippet", "") or "")[:60]
                details = title
                if snippet:
                    details += f"\n[dim]{snippet}[/dim]"
                table.add_row(ref_id, f"[orange3]{tag}[/orange3]", url, details)
            else:
                doi = item.get("doi", "")[:35]
                authors = (item.get("authors", "") or "")[:24]
                year = item.get("year", "")
                journal = (item.get("journal", "") or "")[:24]
                table.add_row(ref_id, f"[blue]{tag}[/blue]", f"[cyan]{doi}[/cyan]", f"{authors}\n[dim]{year} {journal}[/dim]")
            shown += 1

    _add_section("RAG 文献库", rag, _MAX_REF_ROWS["rag"])
    _add_section("学术论文", ms, _MAX_REF_ROWS["multi_search"])
    _add_section("网络来源", web, _MAX_REF_ROWS["web"])

    console.print(table)
    if total > sum(_MAX_REF_ROWS.values()):
        console.print(f"  [dim]显示 {sum(_MAX_REF_ROWS.values())}/{total} 条引用，其余已折叠[/dim]")


def main():
    # ── Preload models first, with progress spinner ──
    from rich.spinner import Spinner
    with console.status("[bold green]Loading models...[/bold green]", spinner="dots") as status:
        try:
            from src.retrieval.init import eager_load_rag
            eager_load_rag()
            console.print("[green][OK] Models loaded[/green]")
        except Exception as e:
            console.print(f"[red][WARN] Load failed: {e} — will retry on first query[/red]")

    console.print(Panel(
        "[bold cyan]Citrus QA Agent v8.1.1[/bold cyan]\n"
        "  • [green]light mode[/green]: local RAG — fast (1-3s)\n"
        "  • [orange3]expert mode[/orange3]: RAG + academic databases + sub-agents\n"
        "  • Log file: [dim]logs/agent.log[/dim]\n\n"
        "Type [bold]/expert[/bold] to switch mode, [bold]/quit[/bold] to exit.",
        border_style="cyan",
    ))

    mode = "light"
    while True:
        try:
            query = Prompt.ask(f"\n[{mode}] >>>")
        except (KeyboardInterrupt, EOFError):
            break

        query = query.strip()
        if not query:
            continue
        if query.lower() in ("/quit", "/exit", "/q"):
            break
        if query.lower() == "/expert":
            mode = "expert"
            console.print("[orange3]Switched to EXPERT mode[/orange3]")
            continue
        if query.lower() == "/light":
            mode = "light"
            console.print("[green]Switched to LIGHT mode[/green]")
            continue

        try:
            asyncio.run(run_query(query, mode))
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")

    console.print("\n[dim]Goodbye.[/dim]")


if __name__ == "__main__":
    main()
