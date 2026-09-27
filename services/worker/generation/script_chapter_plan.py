"""Direct scene planning; actual scripts supply history, the plot supplies intent."""

from pydantic import Field, model_validator

from packages.contracts.m2 import CharacterResult
from packages.contracts.m3 import Location, ScenePlan
from packages.contracts.script import Contract

from .script_budget import SceneSize
from .script_cast import ScriptOptions


class WeightedScene(ScenePlan):
    length_weight: int = Field(default=1, ge=1, le=10, strict=True)


class FirstChapterPlan(Contract):
    new_characters: list[CharacterResult] = Field(max_length=3)
    locations: list[Location] = Field(min_length=1, max_length=8)
    scenes: list[WeightedScene] = Field(min_length=1, max_length=8)


class Continuation(Contract):
    continuation: str = Field(min_length=1, max_length=240,
        description="前章の最後の反応を受け、誰が最初に何を新しく行うかを1〜2文で書く。第1場面の最初の出来事へ直接渡す。過去の要約や章全体の結末ではない。")


class ContinuedChapterPlan(FirstChapterPlan, Continuation):
    pass


class ChapterScriptPlan(Contract):
    continuation: str = ""
    new_characters: list[CharacterResult] = Field(max_length=3)
    locations: list[Location] = Field(min_length=1, max_length=8)
    scenes: list[ScenePlan] = Field(min_length=1, max_length=8)
    scene_sizes: dict[str, SceneSize]

    @model_validator(mode="after")
    def sizes_match_scenes(self):
        if set(self.scene_sizes) != {scene.id for scene in self.scenes}:
            raise ValueError("Scene size metadata must match the planned scenes.")
        return self


def project_plan(draft: FirstChapterPlan, options: ScriptOptions) -> ChapterScriptPlan:
    """Add soft volume guidance once; preserve the planner's actual scene contents."""
    total = sum(scene.length_weight for scene in draft.scenes)
    scenes, sizes = [], {}
    opening = getattr(draft, "continuation", "")
    for index, scene in enumerate(draft.scenes):
        body = round(options.target_body_characters * scene.length_weight / total)
        dialogue = round(options.target_dialogue_characters * scene.length_weight / total)
        sizes[scene.id] = SceneSize(length_weight=scene.length_weight, body_characters=body,
                                   dialogue_characters=dialogue)
        data = scene.model_dump(exclude={"length_weight"})
        if index == 0 and opening:
            # The generated opening has a single execution site. Preserve the
            # rest of the first event and the public eight-event limit.
            first = data["required_events"][0]
            first["description"] = ("【最初の新行動】" + opening
                                    + "\n【続く行動・応答】" + first["description"])
        data["objectives"] += (f"\n【分量の補助目安】この場面の本文約{body}文字、うち台詞約{dialogue}文字。"
                               "会話と応答を十分に描く。字数検査や機械的な水増しはしない。")
        scenes.append(ScenePlan.model_validate(data))
    return ChapterScriptPlan(continuation=opening,
        new_characters=draft.new_characters, locations=draft.locations, scenes=scenes,
        scene_sizes=sizes)
