"""Script-stage changes for drama, run from a saved probe_plot_drama.py plan.

The plan's chapters are written into scripts (text only) with one of:

  now  the product's chapter planning and scene writing (the control)
  sub  each scene plan says what people do and show, apart from a new `unspoken` field (aims and
       secrets nobody says aloud), which reaches the scene writer as material it must not explain;
       NARRATOR no longer explains inner states or what an action achieves
  frc  the plan's forces (each person's want, means, last resort, unknowns) reach chapter planning
       and scene writing, so people opposed in a scene act on their own means there too
  frcl frc with the forces in chapter planning only, and no scenes added for the responses: the
       scene writer gets the same material as now, so scripts and inputs stay near now's length
  frcsub frc and sub together: chapter planning gets both rules and the forces, and the scene
       writer gets the forces and the scene's unspoken aims

--skip-staging leaves out the staging calls (directions); the script text is what gets read.

The run copies the plan folder and resumes it with --script, so the plan is the same for every arm.
Nothing in the product changes.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

import probe_plot_drama as drama
from probe_plot_drama import pipeline, replaced, script_continuation as sc

from services.worker.generation import script_chapter_plan
from services.worker.generation.script_cast import CastPlan
from services.worker.generation.text_debug import run_text_debug
from services.worker.model_config import select_config

ARMS = ("now", "sub", "frc", "frcl", "frcsub")

NARRATOR_RULE = "必要な状況・動作・内心はNARRATOR、人物の言葉は台詞で、応答を十分に展開します。"
SUB_NARRATOR = ("必要な状況・動作はNARRATOR、人物の言葉は台詞で、応答を十分に展開します。"
                "内心や、行動の意味・相手への効果はNARRATORで解説せず、言葉の選び方・間・動作と相手の反応で示します。"
                "人物に自分の狙いや心理の仕組みを解説させません。")
SUB_PLAN = (
    "required_eventsは、人物がその場で実際にすること・言うことの要点と、相手から見える反応で書きます。"
    "相手の心がどう変わるかという狙いや効果の言葉、相手に伝える主張の文面をそのまま書かず、"
    "何を見せ、何を差し出し、相手が何を返すかにします。"
    "各場面のunspokenには、その場面で人物が口にしない狙い・隠していること・相手の知らないことを1〜2文で書きます。"
    "なければ空文字です。"
    "計画の相談・報告・状況の確認だけで進む場面は、その内容を後の場面で行動として見せられるなら作らず、"
    "仕掛けや決断は実行する場面で初めて見せます。"
)
SUB_BRIEF = ("\nこの場面で人物が口にしない狙い・隠していること（台詞やNARRATORで説明せず、"
             "行動と相手の反応から読み取れるようにします）:\n")
FRC_PLAN = (
    "briefのforcesは人物ごとの望み・手段・最後の手・知らないこと・食い違いで、プロットを作った前提です。"
    "場面に登場する人物は、割り当てた出来事の間も自分の望みと手段で動きます。"
    "食い違う相手がいる場面では、相手の働きかけに気づく・妨げる・様子を見るなど、"
    "本人の判断としての応答をrequired_eventsに含めます。主筋の出来事と結果は変えません。"
)
FRCL_PLAN = FRC_PLAN + "対立側の応答は既存の場面の出来事に含め、そのために場面を増やしません。"
FRC_SCENE = ("briefのforcesにある人物は、この場面でも自分の望みと手段から動きます。"
             "何もしない場合も本人の判断として描き、場面計画の出来事と結果は変えません。")
FORCES_BRIEF = "\n力関係forces（人物ごとの望み・手段・最後の手・知らないこと・食い違い。プロットの前提）:\n"


class WeightedScene(script_chapter_plan.WeightedScene):
    unspoken: str = Field(default="", max_length=300,
                          description="この場面で人物が口にしない狙い・隠していること・相手の知らないこと。なければ空文字。")


class FirstChapterPlan(script_chapter_plan.FirstChapterPlan):
    scenes: list[WeightedScene] = Field(min_length=1, max_length=8)


class ContinuedChapterPlan(script_chapter_plan.ContinuedChapterPlan):
    scenes: list[WeightedScene] = Field(min_length=1, max_length=8)


class SplitChapterPlan(script_chapter_plan.SplitChapterPlan):
    scenes: list[WeightedScene] = Field(min_length=1, max_length=8)


ORIGINAL = {FirstChapterPlan: script_chapter_plan.FirstChapterPlan,
            ContinuedChapterPlan: script_chapter_plan.ContinuedChapterPlan,
            SplitChapterPlan: script_chapter_plan.SplitChapterPlan}


def use_sub() -> None:
    sc.SCENE_INSTRUCTION = replaced(sc.SCENE_INSTRUCTION, [(NARRATOR_RULE, SUB_NARRATOR)])
    sc.FirstChapterPlan, sc.ContinuedChapterPlan, sc.SplitChapterPlan = (
        FirstChapterPlan, ContinuedChapterPlan, SplitChapterPlan)
    pending = {}
    project_plan = sc.project_plan

    def without_unspoken(draft, options):
        pending.clear()
        # Gemma sometimes runs the next field's text into the string after a blank line.
        pending.update({scene.id: scene.unspoken.split("\n\n")[0].strip() for scene in draft.scenes})
        plain = ORIGINAL[type(draft)].model_validate({
            **draft.model_dump(exclude={"scenes"}),
            "scenes": [scene.model_dump(exclude={"unspoken"}) for scene in draft.scenes]})
        return project_plan(plain, options)

    sc.project_plan = without_unspoken
    open_scene_schema = sc.open_scene_schema

    def opened(schema):
        open_scene_schema(schema)
        unspoken_first(schema)

    sc.open_scene_schema = opened
    chapter_plan = sc.ScriptRun.chapter_plan

    def kept_unspoken(self, number, context, supporting):
        plan = chapter_plan(self, number, context, supporting)
        saved = self.state.setdefault("unspoken", {})
        saved.update({f"c{number:03d}-{key}": value for key, value in pending.items() if value})
        self.persist()
        return plan

    sc.ScriptRun.chapter_plan = kept_unspoken


def forces_text(state: dict) -> str:
    keys = ("character_id", "want", "means", "last_resort", "unknown", "at_odds")
    return json.dumps([{key: row[key] for key in keys if key in row} for row in state["story_forces"]],
                      ensure_ascii=False)


def unspoken_first(schema: dict) -> None:
    """Ask for each scene's unspoken aims right after its objectives, before the events."""
    scene = schema["$defs"]["WeightedScene"]
    keys = list(scene["properties"])
    keys.remove("unspoken")
    keys.insert(keys.index("objectives") + 1, "unspoken")
    scene["required"] = [*scene.get("required", []), "unspoken"]
    drama.ordered(scene, keys)


def use_arm(arm: str) -> None:
    sub, frc = arm in ("sub", "frcsub"), arm in ("frc", "frcl", "frcsub")
    if sub:
        use_sub()
    if arm in ("frc", "frcsub"):
        sc.SCENE_INSTRUCTION = sc.SCENE_INSTRUCTION + FRC_SCENE
    if arm == "now":
        return
    plan_rule = FRCL_PLAN if arm == "frcl" else FRC_PLAN
    call = sc.ScriptRun.call

    def changed(self, key, purpose, number, system, context, *args, **kwargs):
        brief = ""
        if purpose == "script-plan" and key.startswith("plan-"):
            if sub:
                system = system + SUB_PLAN
            if frc:
                system = system + plan_rule
                brief = FORCES_BRIEF + forces_text(self.state)
        elif arm != "frcl" and purpose == "script-scene" and (key.endswith("-text") or key.endswith("-continue")):
            if frc:
                brief = FORCES_BRIEF + forces_text(self.state)
            hidden = self.state.get("unspoken", {}).get(key.rsplit("-", 1)[0]) if sub else None
            if hidden:
                brief = brief + SUB_BRIEF + hidden
        if brief:
            context = {**context, "brief": context.get("brief", "") + brief}
        return call(self, key, purpose, number, system, context, *args, **kwargs)

    sc.ScriptRun.call = changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--plan-from", type=Path, required=True, help="A finished probe_plot_drama.py plan run")
    parser.add_argument("--plan-arm", choices=drama.ARMS, required=True, help="The arm that made the plan")
    parser.add_argument("--arm", choices=ARMS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--skip-staging", action="store_true", help="Leave out the staging calls")
    args = parser.parse_args()
    if not args.output_dir.exists():
        shutil.copytree(args.plan_from, args.output_dir, ignore=shutil.ignore_patterns(".text-debug.lock"))
    value = json.loads(args.input.read_text(encoding="utf-8"))
    plan = json.loads((args.output_dir / "cast-plan.json").read_text(encoding="utf-8"))["plan"]
    drama.use_arm(args.plan_arm, CastPlan.model_validate(plan))
    use_arm(args.arm)
    if args.skip_staging:
        # Set at run time: the saved plan's script_options are part of the experiment identity.
        options = sc.ScriptRun.options.fget
        sc.ScriptRun.options = property(lambda self: options(self).model_copy(update={"skip_staging": True}))
    profile = value["profile"]
    load_config = pipeline.load_config
    pipeline.load_config = lambda: select_config(load_config(), {"profile": profile}, pipeline.ROOT)
    report = run_text_debug({key: value[key] for key in ("approval_snapshot", "profile", "script_options")},
                            args.output_dir, workflow="causal", resume=True, seed=args.seed,
                            policy="script_continuation_v1")
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
