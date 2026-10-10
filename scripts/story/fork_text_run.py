"""Copy a text-only script experiment and reopen its chapters from a given one.

Resuming the copy with check_text.py reuses every saved model reply whose request is
unchanged and sends only the requests that current code adds or changes, so one stage
can be tried against an existing script without regenerating the others. The source
directory is never modified. A changed request for an already saved chapter plan is
still refused by the engine; fork from a chapter whose plan should stay as it is.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


def fork(source: Path, output: Path, chapter: int) -> dict:
    source, output = source.resolve(strict=True), output.resolve()
    if output.exists() or output == source or source in output.parents:
        raise ValueError("Use a new output directory outside the source experiment.")
    shutil.copytree(source, output, ignore=shutil.ignore_patterns(".text-debug.lock"))
    path = output / "draft-state.json"
    state = json.loads(path.read_text(encoding="utf-8"))
    kept = [row for row in state["chapters"] if row["number"] < chapter]
    reopened = [row["number"] for row in state["chapters"] if row["number"] >= chapter]
    state["chapters"] = kept
    prefixes = tuple(f"c{number:03d}-" for number in reopened)
    for key in ("scene_commits", "outfit_moments", "staging_normalizations"):
        state[key] = {scope: value for scope, value in state.get(key, {}).items()
                      if not scope.startswith(prefixes)}
    # A later chapter's plan record names the chapters before it; it is rebuilt on resume.
    state["chapter_plans"] = {key: value for key, value in state.get("chapter_plans", {}).items()
                              if int(key) <= max(chapter, 1) and int(key) <= len(kept) + 1}
    # Restore what the reopened chapter saw: no later notes, only earlier chapters' places.
    state["notes"] = [row for row in state.get("notes", []) if row["number"] < chapter]
    if "locations" in state:
        state["locations"] = {location["id"]: location for row in kept
                              for location in row["narrative"]["locations"]}
    state["partial"] = None
    path.write_bytes(json.dumps(state, ensure_ascii=False, indent=2).encode())
    for number in reopened:
        shutil.rmtree(output / "exports" / f"chapter-{number:03d}", ignore_errors=True)
        (output / "chapters" / f"chapter-{number:03d}.md").unlink(missing_ok=True)
    return {"output": str(output), "kept_chapters": [row["number"] for row in kept],
            "reopened_chapters": reopened}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--from-chapter", type=int, default=1)
    args = parser.parse_args()
    print(json.dumps(fork(args.source, args.output_dir, args.from_chapter), ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
