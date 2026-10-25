"""Render the active General Agent prompt and its source files for review.

用法:
    python -m src.prompts.snapshot                # 渲染到 src/prompts/snapshots/
    python -m src.prompts.snapshot --dir <path>   # 指定输出目录
"""
from __future__ import annotations

import argparse
from pathlib import Path

def render_all() -> dict[str, str]:
    """Render active prompt sources as deterministic review snapshots."""
    from src.prompts.loader import (
        assemble_system_prompt,
        ROLE_SOURCE_FILES,
        PROMPT_DIR,
    )

    out = {"system_general_agent.txt": assemble_system_prompt(mode="general")}
    # Source files used by the active prompt, deduplicated in declared order.
    seen: set[str] = set()
    for role, filenames in ROLE_SOURCE_FILES.items():
        for filename in filenames:
            if filename in seen:
                continue
            seen.add(filename)
            content = (PROMPT_DIR / "source" / filename).read_text(encoding="utf-8")
            out[f"source_{filename.removesuffix('.md')}.txt"] = content
    return out


def write_snapshots(target: Path) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in render_all().items():
        p = target / name
        p.write_text((content or "(empty)") + "\n", encoding="utf-8")
        written.append(p)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="渲染提示词快照")
    ap.add_argument("--dir", default=str(Path(__file__).resolve().parent / "snapshots"))
    args = ap.parse_args()
    paths = write_snapshots(Path(args.dir))
    for p in paths:
        print(f"snapshot: {p} ({p.stat().st_size} bytes)")
    print(f"OK: {len(paths)} snapshots written")
