"""Run a text-only story experiment without coordinator writes or media generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.worker.generation.text_debug import run_text_debug


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True,
                        help="Approval snapshot JSON or saved m3_narrative job-request.json")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workflow", choices=("legacy", "causal"), default="causal")
    parser.add_argument("--policy", choices=("chapter_editor_v1", "story_draft_v1",
                                             "script_continuation_v1"),
                        help="Select an experiment; script_continuation_v1 preserves scripts and Tyrano output")
    parser.add_argument("--context-size", type=int,
                        help="Override profile context tokens; otherwise use configured size")
    parser.add_argument("--seed", type=int, help="Override the input seed (snapshot default: 1)")
    parser.add_argument("--chapter-limit", type=int,
                        help="Stop after N chapters without changing the approved story length")
    parser.add_argument("--plot-only", action="store_true",
                        help="Generate and save cast and plot without writing chapters")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    try:
        value = json.loads(args.input.read_text(encoding="utf-8-sig"))
        report = run_text_debug(value, args.output_dir, workflow=args.workflow,
                                context_size=args.context_size, chapter_limit=args.chapter_limit,
                                resume=args.resume, seed=args.seed, policy=args.policy,
                                plot_only=args.plot_only)
    except KeyboardInterrupt:
        print("Interrupted. Saved chapters and request caches remain resumable.", file=sys.stderr)
        return 130
    except (OSError, RuntimeError, ValueError, TypeError, KeyError) as exc:
        print(f"Text experiment failed: {exc}", file=sys.stderr)
        print(f"Inspect {args.output_dir.resolve()} for report and partial requests.", file=sys.stderr)
        return 1
    chapter_count = (report["saved_chapter_count"] if "saved_chapter_count" in report
                     else report["adopted_chapter_count"])
    print(json.dumps({"status": report["status"], "chapters": chapter_count,
                      "story": str(args.output_dir.resolve() / "story.html")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
