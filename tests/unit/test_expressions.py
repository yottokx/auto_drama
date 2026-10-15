"""Staging states a face for each line; the script keeps it, and a picture is shown where one exists."""

import hashlib
import json

import pytest
from pydantic import ValidationError

from packages.contracts.expressions import (
    EXPRESSION_FADE_MS,
    EXPRESSION_NAMES,
    EXPRESSIONS,
    expression_changes,
)
from packages.contracts.m3 import NarrativeDirection, ScenePlan
from packages.contracts.script import Script
from packages.narrative import parse_scene_text
from packages.tyrano_export.compiler import compile_scenario, entry_states
from packages.tyrano_export.demo import demo_content
from services.worker.generation import narrative
from tests.unit.test_m3_narrative import FakeLLM
from tests.unit.test_script_continuation_run import FIRST, chapter_plan


def test_the_kinds_are_fixed_and_each_has_a_name():
    assert EXPRESSIONS[0] == "neutral" and len(set(EXPRESSIONS)) == len(EXPRESSIONS) == 12
    assert set(EXPRESSION_NAMES) == set(EXPRESSIONS)
    assert NarrativeDirection.model_json_schema()["properties"]["expression"]["anyOf"][0]["enum"] == list(EXPRESSIONS)


def test_a_face_stays_until_it_changes_and_starts_ordinary():
    lines = [("u1", "a", "smile"), ("u2", None, "neutral"), ("u3", "b", "neutral"), ("u4", "a", "smile"),
             ("u5", "a", "neutral"), ("u6", "b", "angry")]
    listed = {"u2": {"b": "surprised"}, "u4": {"b": "surprised", "a": "cry"}, "u6": {"a": "neutral"}}
    assert expression_changes(lines, listed) == [
        ("u1", "a", "smile"),
        # A line of narration changes only the people it names.
        ("u2", "b", "surprised"),
        # Speaking in an ordinary face ends the surprise; the smile two lines later is not repeated,
        # and what the line says of its own speaker gives way to that speaker's annotation.
        ("u3", "b", "neutral"), ("u4", "b", "surprised"),
        ("u5", "a", "neutral"), ("u6", "b", "angry"),
    ]
    assert expression_changes([("u1", "a", "neutral"), ("u2", None, "smile")], {}) == []


def scene():
    plan = ScenePlan.model_validate(chapter_plan("s1")["scenes"][0])
    return plan, parse_scene_text(FIRST, plan.id, set(plan.character_ids))


def answer(utterances, faces, directions=()):
    return {"emotions": [{"utterance_id": row.id, "inner_emotion": "落ち着き", "voice_emotion": "calm" if row.speaker_id
                          else "neutral", "delivery": "", **({"expression": face} if face else {})}
                         for row, face in zip(utterances, faces, strict=True)],
            "directions": list(directions)}


def test_staging_asks_for_one_face_per_line_and_only_real_people():
    plan, utterances = scene()
    llm = FakeLLM([{"content": json.dumps(answer(utterances, ["smile", "neutral", "serious"]), ensure_ascii=False)}])
    narrative._staging(llm, "context", plan, utterances)
    (_, extra), = llm.calls
    schema = extra["response_format"]["json_schema"]["schema"]
    emotion = schema["$defs"]["EmotionAnnotation"]
    assert "expression" in emotion["required"]
    assert emotion["properties"]["expression"]["enum"] == list(EXPRESSIONS)
    # Neither an empty answer nor a repeating one fits the schema.
    assert (schema["properties"]["emotions"]["minItems"], schema["properties"]["emotions"]["maxItems"]) == (3, 3)
    listed = schema["$defs"]["ExpressionAnnotation"]["properties"]
    assert listed["character_id"]["enum"] == plan.character_ids
    assert listed["utterance_id"]["enum"] == [row.id for row in utterances]


def test_faces_become_directions_only_where_they_change():
    plan, utterances = scene()
    staged = narrative.Staging.model_validate(answer(
        utterances, ["smile", "neutral", "smile"],
        [{"utterance_id": utterances[1].id, "kind": "expression", "character_id": "ren", "expression": "surprised"}]))
    directions = narrative._directions(plan, utterances, staged)
    faces = [(row.utterance_id, row.character_id, row.expression, row.timing, row.duration_ms)
             for row in directions if row.kind == "expression"]
    assert faces == [
        (utterances[0].id, "aoi", "smile", "before", EXPRESSION_FADE_MS),
        (utterances[1].id, "ren", "surprised", "before", EXPRESSION_FADE_MS),
        (utterances[2].id, "ren", "smile", "before", EXPRESSION_FADE_MS),
    ]
    assert all(row.expression is None and "expression" not in row.model_dump() for row in directions
               if row.kind != "expression")
    unknown = narrative.Staging.model_validate(answer(utterances, [None] * 3, [
        {"utterance_id": utterances[0].id, "kind": "expression", "character_id": "nobody", "expression": "smile"}]))
    with pytest.raises(ValueError, match="unknown character"):
        narrative._directions(plan, utterances, unknown)


def test_an_answer_saved_before_expressions_existed_reads_and_stages_as_it_did():
    plan, utterances = scene()
    old = answer(utterances, [None] * 3)
    staged = narrative.Staging.model_validate(old)
    # Saved stages are compared and stored by this dump.
    assert staged.model_dump() == old
    assert [row.kind for row in narrative._directions(plan, utterances, staged)] == ["enter", "enter"]


def with_faces(directions, pictures=(("aki", None, "smile"),), outfit=False):
    script, assets = demo_content()
    data = script.model_dump(mode="json")
    aki = next(row for row in data["characters"] if row["id"] == "aki")
    base = assets["aki_sprite"]

    def picture(asset_id, salt):
        # Any decodable picture will do; only its name reaches the scenario.
        content = base + salt.encode()
        data["assets"].append({"id": asset_id, "kind": "character", "artifact_id": "fixture",
                               "filename": asset_id + ".png", "sha256": hashlib.sha256(content).hexdigest()})
        assets[asset_id] = base
        data["assets"][-1]["sha256"] = hashlib.sha256(base).hexdigest()

    if outfit:
        picture("aki_coat", "coat")
        aki["outfits"] = [{"id": "coat", "image_asset_id": "aki_coat"}]
    aki["expressions"] = []
    for character, look, expression in pictures:
        asset_id = f"{character}_{look or 'base'}_{expression}"
        picture(asset_id, asset_id)
        aki["expressions"].append({"expression": expression, "image_asset_id": asset_id,
                                   **({"outfit_id": look} if look else {})})
    for index, row in enumerate(directions, 1):
        data["directions"].append({"id": f"face_{index}", "timing": "before", "duration_ms": 150, **row})
    return Script.model_validate(data), assets


def face(utterance, expression, character="aki"):
    return {"kind": "expression", "utterance_id": utterance, "character_id": character, "expression": expression}


def test_a_visible_face_changes_in_place_without_holding_the_line():
    script, assets = with_faces([face("line_002", "smile"), face("line_003", "angry"), face("line_004", "neutral")])
    scenario = compile_scenario(script, assets)
    # Beside the two changes of face, the entrance states which picture the character comes on with.
    assert scenario.count('wait="false"]') == 2 and scenario.count("[chara_mod ") == 3
    smile = scenario.index('[chara_mod name="ad_aki" storage="aki_base_smile.png" time="150" cross="false" wait="false"]')
    assert scenario.index("*utterance_line_002") < smile < scenario.index('[ad_say id="line_002"]')
    # There is no picture of anger: the script still asks for it, and the ordinary portrait is shown.
    back = scenario.index('[chara_mod name="ad_aki" storage="aki_sprite.png" time="150" cross="false" wait="false"]')
    assert scenario.index("*utterance_line_003") < back < scenario.index('[ad_say id="line_003"]')
    # Returning to the ordinary face from a face without a picture changes nothing on screen.
    assert "[chara_mod" not in scenario[scenario.index("*utterance_line_004"):]
    # A jump to a later line restores the face that was showing when it began.
    states = entry_states(script, assets)["entries"]
    shown = {line: next(row["storage"] for row in value["characters"] if row["id"] == "aki")
             for line, value in states.items() if value["characters"]}
    assert shown == {"line_002": "aki_sprite.png", "line_003": "aki_base_smile.png", "line_004": "aki_sprite.png"}


def test_a_script_without_expression_pictures_plays_exactly_as_before():
    plain, assets = demo_content()
    script, _ = with_faces([face("line_002", "smile"), face("line_003", "cry", "ren")], pictures=())
    assert not script.characters[0].expressions
    assert compile_scenario(script, assets) == compile_scenario(plain, assets)


def test_a_face_belongs_to_the_look_being_worn_and_an_entrance_resets_it():
    script, assets = with_faces(
        [face("line_002", "smile"),
         {"kind": "outfit", "utterance_id": "line_003", "character_id": "aki", "outfit_id": "coat"},
         {"kind": "exit", "utterance_id": "line_004", "character_id": "aki"},
         {"kind": "enter", "utterance_id": "line_004", "character_id": "aki", "position": "left"}],
        pictures=(("aki", None, "smile"), ("aki", "coat", "smile")), outfit=True)
    scenario = compile_scenario(script, assets)
    second, third, fourth = (scenario.index(f"*utterance_line_00{n}") for n in (2, 3, 4))
    assert 'storage="aki_base_smile.png"' in scenario[second:third]
    # Changing clothes keeps the smile, in the picture made for that look.
    assert 'storage="aki_coat_smile.png"' in scenario[third:fourth]
    # Coming on again, the character is in the base look and an ordinary face.
    assert '[chara_mod name="ad_aki" storage="aki_sprite.png" time="0"]' in scenario[fourth:]


def test_pictures_must_be_of_a_look_the_character_has_and_never_of_the_ordinary_face():
    for pictures in ((("aki", "coat", "smile"),), (("aki", None, "neutral"),),
                     (("aki", None, "smile"), ("aki", None, "smile"))):
        with pytest.raises(ValidationError):
            with_faces([], pictures=pictures)
    script, _ = demo_content()
    data = script.model_dump(mode="json")
    data["directions"].append({"id": "face", "kind": "expression", "utterance_id": "line_002", "timing": "before",
                               "character_id": "nobody", "expression": "smile"})
    with pytest.raises(ValidationError):
        Script.model_validate(data)
