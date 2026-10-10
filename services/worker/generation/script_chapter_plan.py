"""Direct scene planning; actual scripts supply history, the plot supplies intent."""

from pydantic import Field, model_validator

from packages.contracts.m2 import CharacterResult
from packages.contracts.m3 import Location, ScenePlan
from packages.contracts.outfits import DEFAULT_OUTFIT, SceneOutfit
from packages.contracts.script import Contract, Identifier

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


OUTFIT_PLAN = (
    "各sceneのoutfitsには、別の姿を持つ人物が当場面に登場する場合の姿を記します。"
    "outfit_idは場面の開始時の姿、change_toは場面の途中で姿が変わる場合の変化後、変わらなければnullです。"
    "開始時の姿は、直前の場面や前章の終わりの姿と、その間の時間経過・場所から決めます。"
    "場面の途中で着替え・変身する場合は場面を分けず、change_toで示して経緯をrequired_eventsに書きます。"
    "当章のoutfit_appearancesに割り当てられた姿は、当章のいずれかの場面で必ず登場させます。"
    "割り当てのない姿も状況に合えば使えます。姿の名前や説明を台詞で解説させる必要はありません。"
    "別の姿を持つ人物が登場しない場面は空配列です。"
    "outfitsに指定できない人物は姿が1種類だけです。その人物が着替えた・普段と違う服装だとは書きません。"
)


OUTFIT_MOMENT = (
    "確定した台本の発話にIDが付いています。本文は変更・再入力しません。"
    "changesの各人物は、この場面の中で姿（服装・変身など）がfromからtoへ変わる予定です。"
    "各人物について、変化後の姿が初めて画面に映るべき発話のIDをutterance_idに記します。"
    "その発話が表示される直前に立ち絵が切り替わります。"
    "着替え終える、変身の光に包まれて姿が変わるなど、変化そのものを描く発話があれば、"
    "その次の発話ではなく、変化を描いているその発話を選びます。"
    "変化の予告・準備・きっかけだけの発話は選びません。"
    "本文がその変化を実際に描いていなければnullにします。推測で選ばず、描写のある発話だけを根拠にします。"
    "expected=falseの人物は、次の場面までに姿が変わる予定です。この場面の本文が変化を描いている場合だけ"
    "発話を選び、描いていなければnullにします。"
)


class OutfitMoment(Contract):
    character_id: str
    utterance_id: Identifier | None


class OutfitMoments(Contract):
    moments: list[OutfitMoment] = Field(max_length=6)


def pending_outfit_changes(plan, following_plans, outfits: dict) -> list[dict]:
    """Changes this scene may show: those it declares, and those due before a later scene."""
    changes = []
    chosen = {row.character_id: row for row in plan.outfits}
    for cid in plan.character_ids:
        if cid not in outfits:
            continue
        row = chosen.get(cid)
        start = row.outfit_id if row else DEFAULT_OUTFIT
        if row and row.change_to:
            changes.append({"character_id": cid, "from": start, "to": row.change_to, "expected": True})
            continue
        following = next((scene for scene in following_plans if cid in scene.character_ids), None)
        if following is None:
            continue
        later = next((item.outfit_id for item in following.outfits if item.character_id == cid), DEFAULT_OUTFIT)
        if later != start:
            changes.append({"character_id": cid, "from": start, "to": later, "expected": False})
    return changes


def outfit_moment_schema(changes: list[dict], utterance_ids: list[str]) -> dict:
    schema = OutfitMoments.model_json_schema()
    item = schema["$defs"]["OutfitMoment"]["properties"]
    item["character_id"]["enum"] = [row["character_id"] for row in changes]
    item["utterance_id"] = {"anyOf": [{"type": "string", "enum": utterance_ids}, {"type": "null"}]}
    schema["properties"]["moments"].update(minItems=len(changes), maxItems=len(changes))
    return schema


def add_scene_outfits(schema: dict, outfits: dict) -> None:
    """Ask for each scene's looks right after its cast, before its events are written."""
    # One alternative per character, offering only that character's own looks.
    item = {"anyOf": [
        {"type": "object", "additionalProperties": False,
         "properties": {"character_id": {"type": "string", "enum": [cid]},
                        "outfit_id": {"type": "string", "enum": [row.id for row in rows],
                                      "description": SceneOutfit.model_fields["outfit_id"].description},
                        "change_to": {"anyOf": [{"type": "string", "enum": [row.id for row in rows]},
                                                {"type": "null"}],
                                      "description": SceneOutfit.model_fields["change_to"].description}},
         "required": ["character_id", "outfit_id", "change_to"]} for cid, rows in outfits.items()]}
    scene = schema["$defs"]["WeightedScene"]
    properties = {}
    for key, value in scene["properties"].items():
        properties[key] = value
        if key == "character_ids":
            properties["outfits"] = {"type": "array", "maxItems": len(outfits), "items": item,
                                     "description": "別の姿を持つ人物の、当場面での姿。該当者がいなければ空配列。"}
    scene["properties"] = properties
    scene["required"].append("outfits")


def settle_scene_outfits(plan, outfits: dict, required: list[tuple[str, str]]):
    """Keep only the choices that differ from the base look; reject unusable ones.

    Returns the plan and a note for each choice that was dropped rather than rejected.
    """
    scenes, notes = [], []
    for scene in plan.scenes:
        kept, seen = [], set()
        for row in scene.outfits:
            known = {item.id for item in outfits.get(row.character_id, [])}
            if not {row.outfit_id, row.change_to or row.outfit_id} <= known:
                raise ValueError(f"Scene {scene.id}: {row.character_id} has no such look. "
                                 f"Use one of {sorted(known)} for this character.")
            if row.character_id in seen:
                raise ValueError(f"Scene {scene.id}: give {row.character_id} one outfits entry.")
            seen.add(row.character_id)
            if row.character_id not in scene.character_ids:
                notes.append({"scene_id": scene.id, "character_id": row.character_id,
                              "dropped": "character_not_in_scene"})
                continue
            change = row.change_to if row.change_to != row.outfit_id else None
            if row.outfit_id != DEFAULT_OUTFIT or change:
                kept.append(SceneOutfit(character_id=row.character_id, outfit_id=row.outfit_id,
                                        change_to=change))
        scenes.append(scene.model_copy(update={"outfits": kept}))
    shown = {(row.character_id, look) for scene in scenes for row in scene.outfits
             for look in (row.outfit_id, row.change_to)}
    missing = [pair for pair in required if pair not in shown]
    if missing:
        raise ValueError("This chapter's outfit_appearances must be shown in at least one scene's outfits "
                         f"(as outfit_id or change_to, with that character in character_ids). Missing: {missing}.")
    return plan.model_copy(update={"scenes": scenes}), notes
