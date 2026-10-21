"""Copy a finished check_text_long.py run without its last chapter, so only that chapter is written again.

The copy keeps the plan, every earlier chapter and the history notes. It drops the last
chapter's text, scene plan, scenes and their request records, rebuilds the location registry
from the earlier chapters, and rewinds the request counter to the last chapter's first request,
so the rewritten chapter is asked with the same seeds. Resume the copy with the same options.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("--shift", type=int, default=0,
                        help="Move the request counter this far, so the rewritten chapter gets other seeds")
    args = parser.parse_args()
    if args.target.exists():
        parser.error(f"{args.target} already exists.")
    shutil.copytree(args.source, args.target, ignore=shutil.ignore_patterns(".text-debug.lock"))
    path = args.target / "draft-state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    number = state["chapters"][-1]["number"]
    prefixes = (f"plan-{number:03d}", f"voices-{number:03d}", f"route-done-{number:03d}", f"c{number:03d}-")
    dropped = [key for key in state["steps"] if key.startswith(prefixes)]
    first = min(row["request"] for key in dropped for row in state["steps"][key]["attempts"])
    for key in dropped:
        del state["steps"][key]
    state["chapters"].pop()
    state["chapter_plans"].pop(str(number), None)
    for name in ("scene_commits", "scene_budgets"):
        state[name] = {key: value for key, value in state.get(name, {}).items()
                       if not key.startswith(f"c{number:03d}-")}
    for name in ("outfit_plan_normalizations", "chapter_plan_normalizations",
                 "chapter_boundary_corrections", "staging_normalizations"):
        if isinstance(state.get(name), dict):
            state[name].pop(str(number), None)
    locations = {}
    for chapter in range(1, number):
        plan = state["chapter_plans"][str(chapter)]["plan"]
        locations.update({row["id"]: row for row in plan["locations"]})
    state["locations"] = locations
    state["partial"] = None
    state["request_ordinal"] = first - 1 + args.shift
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    for pattern in (f"chapters/chapter-{number:03d}*", f"sources/c{number:03d}-*", f"requests/plan-{number:03d}*",
                    f"requests/voices-{number:03d}*", f"requests/c{number:03d}-*"):
        for item in args.target.glob(pattern):
            item.unlink()
    shutil.rmtree(args.target / "exports" / f"chapter-{number:03d}", ignore_errors=True)
    print(json.dumps({"dropped_chapter": number, "dropped_steps": len(dropped), "request_ordinal": first - 1 + args.shift}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
