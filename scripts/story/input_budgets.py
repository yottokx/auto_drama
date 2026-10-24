"""List each request of a script run with its input tokens and what the context fitting dropped.

Reads RUN/requests/<key>-<attempt>.json (the fitted request and its selection) and prints one
row per request: input tokens against the input limit, whether the optional example stayed,
how many future chapter routes were cut to their purpose and end state, and how each earlier
chapter was carried (full source, note and excerpt, or note only).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPRESENTATION = {"full_source": "原文", "note_and_source_excerpt": "要約+抜粋", "note": "要約"}


def row(path: Path) -> str | None:
    selection = json.loads(path.read_text(encoding="utf-8")).get("selection") or {}
    budget = selection.get("budget") or {}
    if "prompt_tokens" not in budget:
        return None
    example = selection.get("example")
    future = selection.get("future") or {}
    compact = sum(1 for route in future.get("routes", []) if route["representation"] != "full")
    history = [f"{item['number']}:{REPRESENTATION.get(item['representation'], item['representation'])}"
               for item in selection.get("history_coverage", [])
               if item["number"] != selection.get("current_chapter")]
    parts = [f"{path.stem:<24}", f"{budget['prompt_tokens']:>6} / {budget['input_token_limit']:>6}",
             "作例" + ("あり" if example and example["included"] else "なし" if example else "-"),
             f"予定の圧縮{compact}"]
    if history:
        parts.append("前の章 " + " ".join(history))
    return "  ".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path, help="A check_text_long.py output directory")
    args = parser.parse_args()
    paths = sorted(path for path in (args.run / "requests").glob("*.json")
                   if not path.name.endswith(".validation.json"))
    for path in paths:
        line = row(path)
        if line:
            print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
