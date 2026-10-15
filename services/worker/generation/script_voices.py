"""Who is heard in each planned scene, so a voice the plan left out still gets its own speaker.

A chapter plan decides its cast before anyone speaks. A grandfather heard only on a voicemail,
or a voice over a radio, is planned as an event rather than as a person, and the writer, which
may only use the planned cast's IDs, then gives those lines to someone on stage. Without
reasoning, and even with it, a plan almost never adds such a person on its own.

So after the plan, one light call names everyone heard in each scene. Anyone heard who is not
yet a character is defined in a call of its own and cast in the scenes where they are heard.
"""

from __future__ import annotations

import json
from typing import Literal

from pydantic import Field

from packages.contracts.m2 import CharacterResult
from packages.contracts.script import Contract

PURPOSE = "script-voices"
VOICES_INSTRUCTION = (
    "確定した場面計画について、各場面で台詞として声が聞こえる人物を全員挙げます。"
    "その場にいる人物だけでなく、電話・無線・スピーカー・録音や留守電・映像・回想の声だけで"
    "姿を見せない人物や、亡くなった人物の声も含めます。一言でも台詞を話すなら、名前のない人物も含めます。"
    "地の文で話題に出るだけで台詞のない人物は含めません。"
    "資料の人物ならcharacter_idにそのIDを、資料にいない人物ならcharacter_idを空文字にします。"
    "nameは呼び名、howはその声の届き方です（present: その場で話す、remote: 機器や念話を通して届く、"
    "recollection: 思い出している過去の言葉）。")
DEFINE_INSTRUCTION = (
    "次の人物は、この章の場面で台詞の声が聞こえますが、まだ人物として定義されていません。"
    "資料の世界観と既存の人物との関係に合う設定を作ります。idには指定のIDをそのまま使います。"
    "故人や声だけの人物でも、声の演技と外見を具体的に書きます。")
# A scene shows at most this many people; someone heard in a full scene cannot be added to it.
MAX_CAST = 3


class Voice(Contract):
    name: str = Field(min_length=1, max_length=40)
    character_id: str
    how: Literal["present", "remote", "recollection"]


class SceneVoices(Contract):
    scene_id: str
    voices: list[Voice] = Field(max_length=8)


class ChapterVoices(Contract):
    scenes: list[SceneVoices]


def _scene(scene) -> dict:
    # The volume guidance is for the writer; who speaks is decided by what happens.
    return {"id": scene.id, "character_ids": scene.character_ids,
            "objectives": scene.objectives.split("\n【分量の補助目安】")[0], "start_state": scene.start_state,
            "required_events": [event.description for event in scene.required_events],
            "end_state": scene.end_state}


def voices_schema(plan, names: dict[str, str]) -> dict:
    schema = ChapterVoices.model_json_schema()
    schema["$defs"]["SceneVoices"]["properties"]["scene_id"]["enum"] = [scene.id for scene in plan.scenes]
    schema["$defs"]["Voice"]["properties"]["character_id"]["enum"] = ["", *names]
    schema["properties"]["scenes"].update(minItems=len(plan.scenes), maxItems=len(plan.scenes))
    return schema


def voices_context(plan, names: dict[str, str]) -> dict:
    brief = ("資料の人物（IDと名前）: " + json.dumps(names, ensure_ascii=False)
             + "\n確定した場面計画: " + json.dumps([_scene(scene) for scene in plan.scenes], ensure_ascii=False))
    return {"setting": {}, "outline": "", "brief": brief, "chapters": [], "notes": []}


def unheard(answer: ChapterVoices, plan) -> dict[str, list[str]]:
    """The scenes each voice without a character is heard in, by name, in order of appearance."""
    scene_ids = [scene.id for scene in plan.scenes]
    if [row.scene_id for row in answer.scenes] != scene_ids:
        raise ValueError("Name the voices of every planned scene once, in order.")
    heard: dict[str, list[str]] = {}
    for row in answer.scenes:
        for voice in row.voices:
            if not voice.character_id:
                scenes = heard.setdefault(voice.name.strip(), [])
                if row.scene_id not in scenes:
                    scenes.append(row.scene_id)
    return heard


def define_schema(identifier: str) -> dict:
    schema = CharacterResult.model_json_schema()
    schema["properties"]["id"]["enum"] = [identifier]
    return schema


def define_context(context: dict, plan, scene_ids: list[str], name: str, identifier: str) -> dict:
    scenes = [_scene(scene) for scene in plan.scenes if scene.id in scene_ids]
    return {**context, "brief": context["brief"] + "\n当章の場面計画のうち、この人物の声が聞こえる場面: "
            + json.dumps(scenes, ensure_ascii=False) + f"\n定義する人物: {name}（id: {identifier}）"}


def cast(plan, added: list[tuple[CharacterResult, list[str]]]):
    """The plan with each new person defined and cast where heard, if the scene has room."""
    scenes, used = [], set()
    for scene in plan.scenes:
        ids = list(scene.character_ids)
        for person, heard_in in added:
            if scene.id in heard_in and person.id not in ids and len(ids) < MAX_CAST:
                ids.append(person.id)
                used.add(person.id)
        scenes.append(scene.model_copy(update={"character_ids": ids}))
    return plan.model_copy(update={
        "new_characters": [*plan.new_characters, *(person for person, _ in added if person.id in used)],
        "scenes": scenes})
