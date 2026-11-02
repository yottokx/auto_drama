"""Compare whole plans made with fix `order` (run_allocation_order.sh) to their original runs.

Usage: allocation_order_report.py WORKS_DIR  (writes WORKS_DIR/alloc-plan/compare.md)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def most_prompt(run: Path, stage: str) -> int | None:
    """The largest input of ``stage``; the expansions are the script-outline calls after the allocation."""
    rows = [row for row in load(run / "llm" / "llm-metrics.json") if row.get("type") == "context_budget"]
    if stage == "script-expansion":
        first = next(n for n, row in enumerate(rows) if row["stage"] == "script-allocation")
        rows, stage = rows[first:], "script-outline"
    found = [row["prompt_tokens"] for row in rows if row["stage"] == stage]
    return max(found) if found else None


def summary(run: Path) -> dict:
    chain = load(run / "story-chain.json")
    ends = [row["last_event"] for row in load(run / "chapter-allocation.json")["chapters"]]
    sizes = [len(event["steps"]) for event in chain["events"]]
    skeleton = [sum(sizes[start:end]) for start, end in zip([0, *ends[:-1]], ends)]
    plot = load(run / "plot.json")
    steps = [sum(len(event["steps"]) for event in chapter["events"]) for chapter in plot["chapters"]]
    return {"ends": ends, "skeleton": skeleton, "steps": steps, "plot": plot,
            "allocation": most_prompt(run, "script-allocation"), "expansion": most_prompt(run, "script-expansion")}


def main() -> int:
    works = Path(sys.argv[1])
    lines = ["# 案3（章ごとに最後の出来事を先に書く）の計画への影響", "",
             "| 計画 | 区切り 元→案3 | 骨組みの手順/章 元→案3 | 肉付け後の手順/章 元→案3 | 章分け入力 元→案3 | 最大の肉付け入力 元→案3 |",
             "|---|---|---|---|---|---|"]
    details = []
    for run in sorted((works / "alloc-plan").glob("*-order")):
        if not (run / "plot.json").exists():
            lines.append(f"| {run.name} | 計画なし | | | | |")
            continue
        chain = run.name.removesuffix("-order")
        old, new = summary(works / chain), summary(run)
        lines.append(f"| {chain} | {old['ends']}→{new['ends']} | {old['skeleton']}→{new['skeleton']} | "
                     f"{old['steps']}→{new['steps']} | {old['allocation']}→{new['allocation']} | "
                     f"{old['expansion']}→{new['expansion']} |")
        details += ["", f"## {chain}", ""]
        for number, (a, b) in enumerate(zip(old["plot"]["chapters"], new["plot"]["chapters"]), 1):
            details += [f"- 第{number}章 元「{a['title']}」{a['role']}",
                        f"  - 案3「{b['title']}」{b['role']}"]
    (works / "alloc-plan" / "compare.md").write_text("\n".join(lines + details) + "\n", encoding="utf-8")
    print(works / "alloc-plan" / "compare.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
