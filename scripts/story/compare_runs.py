"""Compare two script runs request by request: the messages and answer formats sent, the seeds,
and the saved plot and chapters.

Made to check that the product's long-form plan (check_text_long.py --product) sends what the
experiments sent: with the same input, cast (--cast-from) and seed, every request should match
and so should every answer. Reads RUN/requests/<key>-<attempt>.json, RUN/plot.json and
RUN/chapters/*.md; prints one line per request and file.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def requests(run: Path) -> dict[str, dict]:
    return {path.stem: json.loads(path.read_text(encoding="utf-8"))["request"]
            for path in sorted((run / "requests").glob("*.json")) if not path.name.endswith(".validation.json")}


def first_difference(a: str, b: str) -> str:
    index = next((n for n, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))
    return f"at {index}: {a[max(0, index - 40):index + 40]!r} / {b[max(0, index - 40):index + 40]!r}"


def compare_request(a: dict, b: dict) -> str:
    notes = []
    for number, (x, y) in enumerate(zip(a.get("messages", []), b.get("messages", []))):
        if x != y:
            notes.append(f"message {number} ({x.get('role')}) differs " + first_difference(
                str(x.get("content")), str(y.get("content"))))
    if len(a.get("messages", [])) != len(b.get("messages", [])):
        notes.append("message count differs")
    if a.get("response_format") != b.get("response_format"):
        notes.append("answer format differs")
    if a.get("seed") != b.get("seed"):
        notes.append(f"seed {a.get('seed')} / {b.get('seed')}")
    return "; ".join(notes) or "same"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("second", type=Path)
    args = parser.parse_args()
    first, second = requests(args.first), requests(args.second)
    for key in sorted(set(first) | set(second)):
        if key not in first or key not in second:
            print(f"{key:<26} only in {'first' if key in first else 'second'}")
            continue
        print(f"{key:<26} {compare_request(first[key], second[key])}")
    for name in ["plot.json", *sorted({path.name for run in (args.first, args.second)
                                        for path in (run / "chapters").glob("*.md")})]:
        paths = [run / ("" if name == "plot.json" else "chapters") / name for run in (args.first, args.second)]
        if not all(path.is_file() for path in paths):
            print(f"{name:<26} missing in {'first' if not paths[0].is_file() else 'second'}")
            continue
        texts = [path.read_text(encoding="utf-8") for path in paths]
        print(f"{name:<26} {'same' if texts[0] == texts[1] else 'differs ' + first_difference(*texts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
