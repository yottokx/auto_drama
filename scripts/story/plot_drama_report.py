"""Side-by-side reading of probe_plot_drama.py runs: each work's plans and a few counts.

The counts only help reading; none is a target. Per run: steps in all, each person's steps per
chapter, steps whose result someone goes on not knowing, and steps that only pass something on.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ARMS = ("base", "a", "b", "a2", "a3l", "a3i", "a3s", "a4")
PASSING_ON = re.compile("伝え|報告|指示|確認|説明|共有|教え")


def counts(run: Path, who: dict) -> dict:
    plot = json.loads((run / "plot.json").read_text(encoding="utf-8"))
    per = {}
    steps = hidden = passing = 0
    for chapter in plot["chapters"]:
        for event in chapter["events"]:
            for step in event["steps"]:
                steps += 1
                hidden += bool(step.get("hidden_from"))
                passing += bool(PASSING_ON.search(step["action"]))
                name = who.get(step["character_id"], step["character_id"])
                per.setdefault(name, [0] * len(plot["chapters"]))[chapter["number"] - 1] += 1
    return {"chapters": len(plot["chapters"]), "steps": steps, "hidden": hidden, "passing": passing, "per": per}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--works", type=Path, required=True)
    args = parser.parse_args()
    summary = ["# 力関係プロット実験の比較", "",
               "| 作品 | 条件 | 章 | 手順 | 知らない人がいる手順 | 伝える系の手順 |", "|---|---|---|---|---|---|"]
    for source in sorted((args.works / "inputs").glob("*.json")):
        work = source.stem
        value = json.loads(source.read_text(encoding="utf-8"))
        who = {row["result"]["id"]: row["result"]["name"] for row in value["approval_snapshot"]["characters"]}
        parts = [f"# {value['title']}", ""]
        # Each arm's first run, then its runs with other seeds (WORK-ARM-sN).
        runs = [args.works / f"{work}-{arm}" for arm in ARMS] + sorted(
            path for arm in ARMS for path in args.works.glob(f"{work}-{arm}-s*") if path.is_dir())
        for run in runs:
            if not (run / "plot.json").exists():
                continue
            arm = run.name[len(work) + 1:]
            cast = json.loads((run / "cast-plan.json").read_text(encoding="utf-8"))["plan"]
            who.update({row["id"]: row["name"] for row in cast["supporting_characters"]})
            row = counts(run, who)
            summary.append(f"| {value['title']} | {arm} | {row['chapters']} | {row['steps']} | {row['hidden']} | {row['passing']} |")
            parts += [f"## 条件 {arm}", "", "| 人物 | " + " | ".join(f"{n}章" for n in range(1, row["chapters"] + 1)) + " |",
                      "|---|" + "---|" * row["chapters"]]
            parts += [f"| {name} | " + " | ".join(str(n) for n in values) + " |" for name, values in row["per"].items()]
            plan = (run / "drama-plan.md").read_text(encoding="utf-8")
            parts += ["", plan.split("\n", 1)[1].replace("\n## ", "\n### "), ""]
        (args.works / f"report-{work}.md").write_text("\n".join(parts), encoding="utf-8")
    (args.works / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(args.works / "summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
