"""Prompt 版本化快照（书 §6.10.4 提示词敏感性评估，O5 落地）。

把当前全部提示词装配结果渲染为确定性 .txt 快照，提示词文件变更后
可 diff 快照审查（配合评估集做回归）。快照不参与运行时逻辑，仅审查工具。

用法:
    python -m src.prompts.snapshot                # 渲染到 src/prompts/snapshots/
    python -m src.prompts.snapshot --dir <path>   # 指定输出目录
"""
from __future__ import annotations

import argparse
from pathlib import Path

SAMPLE_QUERY = "示例问题：柑橘黄龙病的综合防治措施"


def render_all(include_strategy_cards: bool = True) -> dict[str, str]:
    """渲染全部提示词装配结果为 {文件名: 内容}。"""
    from src.prompts.loader import (
        assemble_system_prompt,
        assemble_agent_prompt,
        build_dynamic_blocks,
        build_agent_extra_block,
        VALID_AGENTS,
        VALID_FORMATS,
    )

    out: dict[str, str] = {}
    out["system_expert.txt"] = assemble_system_prompt(
        mode="expert", format_hint="fallback", query=SAMPLE_QUERY,
        include_strategy_cards=include_strategy_cards)
    out["system_light.txt"] = assemble_system_prompt(
        mode="light", format_hint="fallback", query=SAMPLE_QUERY,
        include_strategy_cards=include_strategy_cards)
    for agent in sorted(VALID_AGENTS):
        out[f"agent_{agent}.txt"] = assemble_agent_prompt(agent)
    for fmt in sorted(VALID_FORMATS):
        out[f"dynamic_{fmt}.txt"] = build_dynamic_blocks(
            format_hint=fmt, query=SAMPLE_QUERY,
            include_strategy_cards=include_strategy_cards)
    out["agent_extra_block.txt"] = build_agent_extra_block(
        system_prompt_extra="（示例：系统附加指令块）")
    return out


def write_snapshots(target: Path, include_strategy_cards: bool = True) -> list[Path]:
    target.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for name, content in render_all(include_strategy_cards).items():
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
