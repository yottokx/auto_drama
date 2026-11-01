"""Length, inputs and time of probe_script_drama.py runs, and a readable script for each.

Usage: script_drama_report.py WORKS_DIR WORK RUN...
Prints a table and writes RUN.md (names in place of character IDs) next to each run.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_plot_drama as drama


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def readable(run: Path, value: dict) -> Path:
    who = drama.names(value, load(run / "cast-plan.json")["plan"])
    for path in sorted((run / "chapters").glob("chapter-0*.plan.json")):
        who.update({row["id"]: row["name"] for row in load(path).get("new_characters", [])})
    lines = [f"# {value['title']}（{run.name}）", ""]
    for path in sorted((run / "chapters").glob("chapter-[0-9][0-9][0-9].md")):
        lines += [f"## {path.stem}", ""]
        for line in path.read_text(encoding="utf-8").splitlines():
            match = re.match(r"([\w-]+): (.*)", line)
            if match and match[1] == "NARRATOR":
                line = f"（{match[2]}）"
            elif match:
                line = f"**{who.get(match[1], match[1])}**「{match[2]}」"
            if line.strip():
                lines.append(line)
        lines.append("")
    out = run.with_suffix(".md")
    out.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    works, work = Path(sys.argv[1]), sys.argv[2]
    value = load(works / "inputs" / f"{work}.json")
    print("| 台本 | 場面数/章 | 文字数/章（計） | 章計画の入力 最大 | 場面執筆の入力 最大 | 所要（台本） |")
    print("|---|---|---|---|---|---|")
    for name in sys.argv[3:]:
        run = works / name
        plans = sorted((run / "chapters").glob("chapter-[0-9][0-9][0-9].plan.json"))
        texts = sorted((run / "chapters").glob("chapter-[0-9][0-9][0-9].md"))
        scenes = [len(load(path)["scenes"]) for path in plans]
        chars = [len(path.read_text(encoding="utf-8")) for path in texts]
        rows = [row for path in (run / "llm").glob("session-*-metrics.json") for row in load(path)
                if row.get("type") == "context_budget"]
        most = {stage: max((row["prompt_tokens"] for row in rows if row.get("stage") == stage), default=0)
                for stage in ("script-plan", "script-scene")}
        minutes = sum(row.get("elapsed_seconds", 0) for row in load(run / "draft-state.json")["sessions"]
                      if row.get("requested_phase") == "chapters") / 60
        readable(run, value)
        print(f"| {name} | {scenes} | {chars}（{sum(chars)}） | {most['script-plan']} | {most['script-scene']} "
              f"| {minutes:.0f}分 |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
