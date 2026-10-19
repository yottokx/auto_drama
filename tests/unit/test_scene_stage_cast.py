"""A scene of more than three people: its central people stand on stage, everyone else speaks from a frame."""

import json

import pytest

from packages.contracts.m3 import SCENE_CAST_MAX, ScenePlan
from packages.contracts.script import RemoteVoice
from packages.narrative import parse_scene_text
from packages.narrative.validation import validate_narrative
from services.worker.generation import narrative, script_voices
from services.worker.generation.event_cg_lenient import _standing, fit
from services.worker.generation.script_cast import (
    CAST_ANCHOR,
    CAST_LIMIT_MAX,
    CastPlan,
    ScriptOptions,
    SurroundedCastPlan,
    cast_instruction,
    cast_limit_guide,
    cast_schema,
)
from services.worker.generation.script_chapter_plan import FirstChapterPlan, open_scene_schema
from tests.unit.test_event_cg_lenient import cg
from tests.unit.test_event_cg_lenient import payload as cg_payload
from tests.unit.test_m3_narrative import FakeLLM
from tests.unit.test_script_continuation_run import chapter_plan

CAST = ["aoi", "ren", "mio", "sora"]
RAW = ("NARRATOR: 教室は賑やかだ。\naoi: おはよう。\nNARRATOR: 蓮が入ってきた。\nren: 遅れた。\n"
       "mio: 静かにして。\nNARRATOR: 蓮は出ていった。\naoi: またね。\nsora: ……。\n")


def crowded():
    plan = ScenePlan.model_validate({**chapter_plan("s1")["scenes"][0], "character_ids": CAST})
    return plan, parse_scene_text(RAW, plan.id, set(CAST))


def staged(utterances, directions=()):
    return narrative.SceneStaging.model_validate({"emotions": [
        {"utterance_id": row.id, "inner_emotion": "落ち着き", "voice_emotion": "neutral", "delivery": "",
         "expression": "smile" if row.speaker_id else "neutral"} for row in utterances],
        "directions": list(directions)})


def test_a_scene_may_hold_the_whole_cast_but_earlier_schemas_still_ask_for_three():
    plan, _ = crowded()
    assert plan.character_ids == CAST
    schema = FirstChapterPlan.model_json_schema()
    assert schema["$defs"]["WeightedScene"]["properties"]["character_ids"]["maxItems"] == 3
    open_scene_schema(schema)
    assert schema["$defs"]["WeightedScene"]["properties"]["character_ids"]["maxItems"] == SCENE_CAST_MAX


def test_central_people_come_and_go_on_the_lines_named_and_slips_are_fixed():
    plan, utterances = crowded()
    ids = [row.id for row in utterances]
    reply = {"central": [
        {"character_id": "aoi", "enters_at": ids[0], "leaves_at": "", "reason": "主人公。"},
        {"character_id": "ren", "enters_at": ids[2], "leaves_at": ids[5], "reason": "来て去る。"},
        {"character_id": "aoi", "enters_at": "", "leaves_at": "", "reason": "重複。"}]}
    llm = FakeLLM([{"content": json.dumps(reply, ensure_ascii=False)}])
    spans = narrative.central_cast(llm, "context", plan, utterances, {})
    # Coming in on the first line is being there from the start; leaving shows the line, then goes.
    assert spans == [("aoi", 0, len(ids)), ("ren", 2, 6)]
    ((_, messages), extra), = llm.calls
    item = extra["response_format"]["json_schema"]["schema"]["$defs"]["CentralCharacter"]["properties"]
    assert item["character_id"]["enum"] == CAST and item["enters_at"]["enum"] == ["", *ids]
    assert "4人" in messages[-1]["content"]


def test_someone_only_heard_is_not_central_and_going_before_coming_stays_to_the_end():
    plan, utterances = crowded()
    ids = [row.id for row in utterances]
    reply = {"central": [
        {"character_id": "mio", "enters_at": ids[4], "leaves_at": ids[1], "reason": "逆。"},
        {"character_id": "ren", "enters_at": "", "leaves_at": "", "reason": "電話。"}]}
    sources = {row.id: "remote" for row in utterances if row.speaker_id == "ren"}
    spans = narrative.central_cast(FakeLLM([{"content": json.dumps(reply)}]), "c", plan, utterances, sources)
    assert spans == [("mio", 4, len(ids))]


def test_someone_silent_and_never_named_is_not_central():
    plan, utterances = crowded()
    reply = {"central": [
        {"character_id": "aoi", "enters_at": "", "leaves_at": "", "reason": "主人公。"},
        {"character_id": "mio", "enters_at": "", "leaves_at": "", "reason": "見守る。"},
        {"character_id": "sora", "enters_at": "", "leaves_at": "", "reason": "黙っている。"}]}
    silent = [row for row in utterances if row.speaker_id not in ("mio", "sora")]
    names = {"aoi": "青井 葵（あおい あおい）", "mio": "白石・澪", "sora": "教室（きょうしつ） 空"}
    # Mio never speaks nor is named; Sora never speaks but the narration says the classroom.
    spans = narrative.central_cast(FakeLLM([{"content": json.dumps(reply, ensure_ascii=False)}]),
                                   "c", plan, silent, {}, names)
    assert [cid for cid, _, _ in spans] == ["aoi", "sora"]
    assert narrative._name_parts(names["aoi"]) == ["青井", "葵"]
    assert narrative._name_parts(names["mio"]) == ["白石", "澪"]


def test_those_who_stay_keep_their_place_while_others_come_and_go():
    plan, utterances = crowded()
    stage = [("aoi", 0, 8), ("ren", 2, 6), ("mio", 4, 8)]
    rows = [(row.kind, row.timing, [u.id for u in utterances].index(row.utterance_id), row.character_id, row.position)
            for row in narrative._stage_directions(plan, utterances, stage)]
    assert rows == [
        ("enter", "before", 0, "aoi", "center"),
        ("position", "before", 2, "aoi", "left"), ("enter", "before", 2, "ren", "right"),
        ("position", "before", 4, "ren", "center"), ("enter", "before", 4, "mio", "right"),
        # Ren goes after the line that tells of it; Mio keeps the right.
        ("exit", "after", 5, "ren", None)]


def test_a_speaker_without_a_portrait_lights_no_one_and_listeners_off_stage_change_no_face():
    plan, utterances = crowded()
    ids = [row.id for row in utterances]
    staging = staged(utterances, [
        {"utterance_id": ids[4], "kind": "focus", "character_id": "mio", "timing": "start", "duration_ms": 0},
        {"utterance_id": ids[4], "kind": "expression", "character_id": "sora", "expression": "sad"},
        {"utterance_id": ids[4], "kind": "expression", "character_id": "aoi", "expression": "sad"}])
    directions = narrative._directions(plan, utterances, staging, stage=[("aoi", 0, 8)])
    assert {row.character_id for row in directions if row.kind == "enter"} == {"aoi"}
    assert not [row for row in directions if row.kind == "focus"]
    faces = {(ids.index(row.utterance_id), row.character_id) for row in directions if row.kind == "expression"}
    # Aoi's own smile, her change while listening and her smile again; nothing for the others.
    assert faces == {(1, "aoi"), (4, "aoi"), (6, "aoi")}


def test_a_crowded_scene_needs_entrances_only_for_those_on_stage():
    import copy

    from tests.unit.test_m3_narrative import narrative_fixture

    value, snapshot = narrative_fixture()
    snapshot["characters"] += [{"result": {"id": "mio"}}, {"result": {"id": "sora"}}]
    value["outline"]["character_arcs"] += [{"character_id": cid, "change": "見守る"} for cid in ("mio", "sora")]
    scene = value["scenes"][0]
    # The keeper speaks from a frame, Mio and Sora never speak: only Hero stands on stage.
    for line in scene["utterances"]:
        if line["speaker_id"] == "keeper":
            line["remote"] = RemoteVoice(source="offstage").model_dump(mode="json")
    scene["directions"] = [row for row in scene["directions"] if row["character_id"] == "Hero"]
    scene["plan"]["character_ids"] = ["Hero", "keeper", "mio", "sora"]
    validate_narrative(value, snapshot)
    # A scene of three needs none for Mio, silent, but the keeper speaking in person needs one.
    small = copy.deepcopy(value)
    small["scenes"][0]["plan"]["character_ids"] = ["Hero", "keeper", "mio"]
    validate_narrative(small, snapshot)
    for line in small["scenes"][0]["utterances"]:
        line["remote"] = None
    with pytest.raises(ValueError, match="entrance"):
        validate_narrative(small, snapshot)


def test_an_offstage_voice_is_framed_without_a_channel():
    assert RemoteVoice(source="offstage", expression="smile").model_dump() == {"source": "offstage", "expression": "smile"}


def test_heard_voices_join_a_scene_up_to_the_limit_given():
    scene = ScenePlan.model_validate({**chapter_plan("s1")["scenes"][0], "character_ids": ["aoi", "ren", "mio"]})

    class Plan:
        def __init__(self, scenes, new_characters=()):
            self.scenes, self.new_characters = scenes, list(new_characters)

        def model_copy(self, update):
            return Plan(update["scenes"], update["new_characters"])

    full = script_voices.cast(Plan([scene]), [], {"sora": ["s1"]})
    assert full.scenes[0].character_ids == ["aoi", "ren", "mio"]
    opened = script_voices.cast(Plan([scene]), [], {"sora": ["s1"]}, SCENE_CAST_MAX)
    assert opened.scenes[0].character_ids == ["aoi", "ren", "mio", "sora"]


def test_a_chosen_character_count_drops_the_size_anchor_and_lists_the_people_around_first():
    text = cast_instruction(3, 4)
    assert CAST_ANCHOR not in text and "最大4人は容量です" in text and "surroundings" in text
    assert "空配列" in cast_instruction(3, 0) and "最大0人" not in cast_instruction(3, 0)
    schema = cast_schema(SurroundedCastPlan, 4)
    assert next(iter(schema["properties"])) == "surroundings" and schema["required"][0] == "surroundings"
    assert schema["properties"]["supporting_characters"]["maxItems"] == 4
    assert cast_schema(CastPlan, None) == CastPlan.model_json_schema()
    value = SurroundedCastPlan(surroundings=["担任"], supporting_characters=[], everyday_context=[], connections=[])
    assert value.plan() == CastPlan(supporting_characters=[], everyday_context=[], connections=[])


def test_the_options_are_absent_from_earlier_plans():
    assert ScriptOptions().model_dump() == ScriptOptions.model_validate({}).model_dump()
    assert "cast_limit" not in ScriptOptions().model_dump() and "open_scene_cast" not in ScriptOptions().model_dump()
    assert ScriptOptions(cast_limit=7, open_scene_cast=True).model_dump()["cast_limit"] == 7
    with pytest.raises(ValueError):
        ScriptOptions(cast_limit=CAST_LIMIT_MAX + 1)


@pytest.mark.parametrize(("profile", "guide"), [
    ({"context_size": 16384, "reasoning_level": "none"}, 5),
    ({"context_size": 32768, "reasoning_level": "none"}, 8),
    ({"context_size": 32768, "reasoning_level": "medium"}, 7),
    ({}, 5),
])
def test_the_guideline_follows_the_context_and_reasoning(profile, guide):
    assert cast_limit_guide(profile) == guide


def test_an_event_cg_keeps_three_people_and_those_on_stage_first():
    value = cg_payload()
    value["references"] = [{"character_id": cid} for cid in ("alice", "bob", "carol", "dave")]
    scene = value["context"]["narrative"]["scenes"][0]
    scene["plan"]["character_ids"] = ["alice", "bob", "carol", "dave"]
    scene["directions"] = [
        {"kind": "enter", "utterance_id": "s1-u1", "timing": "before", "character_id": "bob"},
        {"kind": "enter", "utterance_id": "s1-u1", "timing": "before", "character_id": "carol"},
        {"kind": "enter", "utterance_id": "s1-u3", "timing": "before", "character_id": "dave"},
        {"kind": "exit", "utterance_id": "s1-u3", "timing": "after", "character_id": "carol"}]
    assert _standing(scene, "s1-u3") == {"bob", "carol", "dave"}
    assert _standing(scene, "s1-u4") == {"bob", "dave"}
    cgs, _ = fit({"cgs": [cg("s1-u4", "s1-u6", ("alice", "bob", "carol", "dave"))]}, value, 3)
    assert cgs[0]["character_ids"] == ["bob", "dave", "alice"]


def test_an_offstage_line_is_framed_and_lights_no_one():
    from packages.contracts.script import Script
    from packages.tyrano_export import compile_scenario
    from packages.tyrano_export.compiler import remote_frames
    from tests.unit.test_remote_voice import remote_script

    script, assets = remote_script()
    data = script.model_dump(mode="json")
    for row in data["utterances"]:
        if row.get("remote"):
            row["remote"] = {"source": "offstage", "expression": "smile"}
    script = Script.model_validate(data)
    frames = remote_frames(script, assets)["lines"]
    assert {frame["source"] for frame in frames.values()} == {"offstage"}
    assert all(frame["channel"] is None for frame in frames.values())
    scenario = compile_scenario(script, assets)
    start = scenario.index("*utterance_line_002")
    assert '[chara_ptext name=""]' in scenario[start:scenario.index('[ad_say id="line_002"]')]
