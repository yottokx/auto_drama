"""Which model writes the scene text: the same plot and chapter plan, written by another model.

A new run takes over a finished probe_plot_drama.py plan (its plot, chain and cast) and a saved
chapter plan, so chapter planning is skipped and only the scene text (and the chapter's
bookkeeping calls) comes from the model in --input's profile. Staging is left out. Run it once
with the model under test and once with the plan's own model as the control, with the same seed.

Nothing in the product changes.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_plot_drama as drama
from probe_plot_drama import pipeline, script_continuation as sc

from services.worker.generation.script_cast import CastPlan
from services.worker.generation.script_chapter_plan import ChapterScriptPlan
from services.worker.generation.text_debug import run_text_debug
from services.worker.model_config import select_config

# The plan's state the plot check needs: the plot and the chain and allocation it came from. The cast
# plan is recorded afresh under this run's identity (probe_plot_drama hands over the same plan).
PLAN_KEYS = ("story_forces", "story_chain", "story_chain_sha256", "chapter_allocation",
             "skeleton_plot", "skeleton_plot_sha256", "plot", "plot_sha256", "outline")


def use_saved(plan_from: Path, chapter_plans: Path) -> None:
    saved = json.loads((plan_from / "draft-state.json").read_text(encoding="utf-8"))
    init = sc.ScriptRun.__init__

    def taken_over(self, *args, **kwargs):
        init(self, *args, **kwargs)
        if "plot" not in self.state:
            self.state.update({key: saved[key] for key in PLAN_KEYS if key in saved})
            self.persist()

    sc.ScriptRun.__init__ = taken_over

    def saved_plan(self, number, context, supporting):
        path = chapter_plans / "chapters" / f"chapter-{number:03d}.plan.json"
        return ChapterScriptPlan.model_validate_json(path.read_text(encoding="utf-8"))

    sc.ScriptRun.chapter_plan = saved_plan
    options = sc.ScriptRun.options.fget
    sc.ScriptRun.options = property(lambda self: options(self).model_copy(update={"skip_staging": True}))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Its profile names the writing model")
    parser.add_argument("--plan-from", type=Path, required=True, help="A finished probe_plot_drama.py plan run")
    parser.add_argument("--plan-arm", choices=drama.ARMS, required=True)
    parser.add_argument("--chapter-plans-from", type=Path, required=True,
                        help="A script run whose chapters/chapter-NNN.plan.json are reused")
    parser.add_argument("--chapters", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    plan = json.loads((args.plan_from / "cast-plan.json").read_text(encoding="utf-8"))["plan"]
    drama.use_arm(args.plan_arm, CastPlan.model_validate(plan))
    use_saved(args.plan_from, args.chapter_plans_from)
    profile = value["profile"]
    load_config = pipeline.load_config
    pipeline.load_config = lambda: select_config(load_config(), {"profile": profile}, pipeline.ROOT)
    report = run_text_debug({key: value[key] for key in ("approval_snapshot", "profile", "script_options")},
                            args.output_dir, workflow="causal", resume=args.output_dir.exists(),
                            seed=args.seed, chapter_limit=args.chapters, policy="script_continuation_v1")
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
