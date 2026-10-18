"""Run check_text.py's script experiment with a cast plan taken from probe_cast_size.py.

The cast plan step is not sent to the model: it returns the chosen saved answer instead, so
the plot and the chapters can be tried with a larger cast than the current instruction makes.
Every later step runs as in check_text.py. Nothing in the product changes.

--scene-cast open lifts the three-person limit on a scene (to the cast's capacity of eight) in
the scene contract, the chapter scene plan's wording and the heard-voices check; acting also
asks the plan to list only the people who speak or act in a scene.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.worker.generation import pipeline, script_continuation
from services.worker.generation.script_cast import CastPlan
from services.worker.generation.text_debug import run_text_debug
from services.worker.model_config import select_config


def chosen_cast(results: Path, title: str, arm: str) -> CastPlan:
    for row in json.loads(results.read_text(encoding="utf-8")):
        if title in row["title"] and "answer" in row["arms"].get(arm, {}):
            answer = dict(row["arms"][arm]["answer"])
            answer.pop("surroundings", None)
            return CastPlan.model_validate(answer)
    raise ValueError(f"No saved cast answer for {title} / {arm}.")


SCENE_CASTS = ("capped", "open", "acting")
CAP = "character_idsは登場人物を最大3人、"
OPEN = "character_idsは登場人物を、"
ACTING = "character_idsは当場面で発言・行動する人物を、"
ACTING_AFTER = "短縮・改名しません。"
ACTING_RULE = "居合わせるだけで発言も行動もしない人物はcharacter_idsに入れず、必要なら地の文で触れます。"
WHOLE_CAST = 8  # the cast plan's capacity


def open_scene_cast() -> None:
    """Let a scene hold up to the whole cast: the scene contract, its holders and the voice check."""
    from annotated_types import MaxLen
    from pydantic import BaseModel

    from packages.contracts import m3
    from services.worker.generation import narrative, script_chapter_plan, script_voices

    for cls in (m3.ScenePlan, script_chapter_plan.WeightedScene):
        field = copy.copy(cls.model_fields["character_ids"])
        field.metadata = [MaxLen(WHOLE_CAST) if isinstance(item, MaxLen) else item for item in field.metadata]
        cls.model_fields["character_ids"] = field
        cls.model_rebuild(force=True)
    # Models holding a scene plan keep the old limit until rebuilt.
    for module in (m3, narrative, script_chapter_plan):
        for value in vars(module).values():
            if isinstance(value, type) and issubclass(value, BaseModel) and value.__module__ == module.__name__:
                value.model_rebuild(force=True)
    script_voices.MAX_CAST = WHOLE_CAST


def scene_cast_instruction(prompt: str, mode: str) -> str:
    if prompt.count(CAP) != 1:
        raise ValueError("The chapter scene plan instruction differs from the expected wording.")
    if mode == "open":
        return prompt.replace(CAP, OPEN)
    head, _, tail = prompt.partition(CAP)
    rest, found, after = tail.partition(ACTING_AFTER)
    if not found:
        raise ValueError("The chapter scene plan instruction has no end to the character_ids sentence.")
    return head + ACTING + rest + ACTING_AFTER + ACTING_RULE + after


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cast-results", type=Path, required=True, help="probe_cast_size.py results.json")
    parser.add_argument("--title", required=True)
    parser.add_argument("--arm", required=True, help="for example surroundings@1879534044")
    parser.add_argument("--context-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--chapter-limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scene-cast", choices=SCENE_CASTS, default="capped",
                        help="open: a scene may hold up to the whole cast (8), not three; "
                             "acting: open, and a scene lists only the people who speak or act in it")
    args = parser.parse_args()
    cast = chosen_cast(args.cast_results, args.title, args.arm)
    if args.scene_cast != "capped":
        open_scene_cast()
    original = script_continuation.ScriptRun.structured

    def structured(self, key, purpose, number, prompt, context, model, validate=None, schema=None):
        if purpose == "script-plan" and args.scene_cast != "capped":
            prompt = scene_cast_instruction(prompt, args.scene_cast)
        if key != "cast-plan":
            return original(self, key, purpose, number, prompt, context, model, validate, schema)
        return validate(cast) if validate else cast

    script_continuation.ScriptRun.structured = structured
    value = json.loads(args.input.read_text(encoding="utf-8-sig"))
    profile = value.get("profile", {})
    if profile.get("common_settings_version"):
        # As a production with the common settings does: the chosen model and context for every step.
        load_config = pipeline.load_config
        pipeline.load_config = lambda: select_config(load_config(), {"profile": profile}, pipeline.ROOT)
    report = run_text_debug(value, args.output_dir, workflow="causal", context_size=args.context_size,
                            chapter_limit=args.chapter_limit, resume=args.resume, seed=args.seed,
                            policy="script_continuation_v1")
    print(json.dumps({"status": report["status"], "chapters": report.get("saved_chapter_count"),
                      "story": str(args.output_dir.resolve() / "story.html")}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
