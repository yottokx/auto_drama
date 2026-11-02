"""How chapters are planned and written from a plan with forces (ScriptOptions.script_drama).

Chosen in docs/experiments/plot-drama-20261030.md: the plan's forces reach chapter planning and
scene writing ("forces"), and with "forces_unspoken" each scene plan also names what its people do
not say aloud, which the writer shows by action and response instead of NARRATOR's explanation.
scripts/story/probe_script_drama.py ran the same wording as the arms frc and frcsub.
"""

from __future__ import annotations

import json

from pydantic import Field

from . import script_chapter_plan

FORCES_PLAN = (
    "briefのforcesは人物ごとの望み・手段・最後の手・知らないこと・食い違いで、プロットを作った前提です。"
    "場面に登場する人物は、割り当てた出来事の間も自分の望みと手段で動きます。"
    "食い違う相手がいる場面では、相手の働きかけに気づく・妨げる・様子を見るなど、"
    "本人の判断としての応答をrequired_eventsに含めます。主筋の出来事と結果は変えません。"
)
FORCES_SCENE = ("briefのforcesにある人物は、この場面でも自分の望みと手段から動きます。"
                "何もしない場合も本人の判断として描き、場面計画の出来事と結果は変えません。")
FORCES_BRIEF = "\n力関係forces（人物ごとの望み・手段・最後の手・知らないこと・食い違い。プロットの前提）:\n"
UNSPOKEN_PLAN = (
    "required_eventsは、人物がその場で実際にすること・言うことの要点と、相手から見える反応で書きます。"
    "相手の心がどう変わるかという狙いや効果の言葉、相手に伝える主張の文面をそのまま書かず、"
    "何を見せ、何を差し出し、相手が何を返すかにします。"
    "各場面のunspokenには、その場面で人物が口にしない狙い・隠していること・相手の知らないことを1〜2文で書きます。"
    "なければ空文字です。"
    "計画の相談・報告・状況の確認だけで進む場面は、その内容を後の場面で行動として見せられるなら作らず、"
    "仕掛けや決断は実行する場面で初めて見せます。"
)
NARRATOR_RULE = "必要な状況・動作・内心はNARRATOR、人物の言葉は台詞で、応答を十分に展開します。"
UNSPOKEN_NARRATOR = ("必要な状況・動作はNARRATOR、人物の言葉は台詞で、応答を十分に展開します。"
                     "内心や、行動の意味・相手への効果はNARRATORで解説せず、言葉の選び方・間・動作と相手の反応で示します。"
                     "人物に自分の狙いや心理の仕組みを解説させません。")
UNSPOKEN_BRIEF = ("\nこの場面で人物が口にしない狙い・隠していること（台詞やNARRATORで説明せず、"
                  "行動と相手の反応から読み取れるようにします）:\n")


# A scene plan with what its people do not say aloud. Named as the scene it extends, so the schema
# helpers that look up WeightedScene apply to it too; a docstring would enter the answer format.
class WeightedScene(script_chapter_plan.WeightedScene):
    unspoken: str = Field(default="", max_length=300,
                          description="この場面で人物が口にしない狙い・隠していること・相手の知らないこと。なければ空文字。")


class FirstChapterPlan(script_chapter_plan.FirstChapterPlan):
    scenes: list[WeightedScene] = Field(min_length=1, max_length=8)


class ContinuedChapterPlan(script_chapter_plan.ContinuedChapterPlan):
    scenes: list[WeightedScene] = Field(min_length=1, max_length=8)


class SplitChapterPlan(script_chapter_plan.SplitChapterPlan):
    scenes: list[WeightedScene] = Field(min_length=1, max_length=8)


UNSPOKEN_MODELS = {script_chapter_plan.FirstChapterPlan: FirstChapterPlan,
                   script_chapter_plan.ContinuedChapterPlan: ContinuedChapterPlan,
                   script_chapter_plan.SplitChapterPlan: SplitChapterPlan}
PLAIN_MODELS = {value: key for key, value in UNSPOKEN_MODELS.items()}


def forces_text(forces: list[dict]) -> str:
    return json.dumps(forces, ensure_ascii=False)


def unspoken_schema(schema: dict) -> None:
    """Ask for each scene's unspoken aims right after its objectives, before the events."""
    scene = schema["$defs"]["WeightedScene"]
    keys = [key for key in scene["properties"] if key != "unspoken"]
    keys.insert(keys.index("objectives") + 1, "unspoken")
    scene["properties"] = {key: scene["properties"][key] for key in keys}
    scene["required"] = [key for key in keys if key in {*scene.get("required", []), "unspoken"}]


def split_unspoken(draft) -> tuple[object, dict[str, str]]:
    """The plan without its unspoken notes, and the notes by scene ID (empty ones left out)."""
    # Gemma sometimes runs the next field's text into the string after a blank line.
    notes = {scene.id: scene.unspoken.split("\n\n")[0].strip() for scene in draft.scenes}
    plain = PLAIN_MODELS[type(draft)].model_validate({
        **draft.model_dump(exclude={"scenes"}),
        "scenes": [scene.model_dump(exclude={"unspoken"}) for scene in draft.scenes]})
    return plain, {key: value for key, value in notes.items() if value}


def scene_instruction(instruction: str, method: str) -> str:
    """The scene writer's instruction for ``method`` (ScriptOptions.script_drama)."""
    if method == "forces_unspoken":
        if instruction.count(NARRATOR_RULE) != 1:
            raise ValueError("The scene instruction's NARRATOR rule differs from the expected text.")
        instruction = instruction.replace(NARRATOR_RULE, UNSPOKEN_NARRATOR)
    return instruction + (FORCES_SCENE if method != "plain" else "")
