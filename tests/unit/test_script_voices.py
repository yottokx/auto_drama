import copy

import pytest

from packages.contracts.m2 import CharacterResult
from services.worker.generation import script_continuation as runner
from services.worker.generation import script_voices as voices
from services.worker.generation.script_chapter_plan import FirstChapterPlan
from tests.unit.test_script_continuation_run import (
    FIRST,
    chapter_plan,
    outline,
    read_json,
    snapshot,
    staging,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


def plan(*casts):
    value = chapter_plan(*(f"s{number}" for number in range(1, len(casts) + 1)))
    for scene, ids in zip(value["scenes"], casts):
        scene["character_ids"] = list(ids)
    return FirstChapterPlan.model_validate(value)


def person(identifier, name="祖父"):
    value = copy.deepcopy(snapshot()["characters"][0]["result"])
    value.update(id=identifier, name=name)
    return CharacterResult.model_validate(value)


def heard(*scenes):
    return voices.ChapterVoices.model_validate({"scenes": [
        {"scene_id": scene_id, "voices": [{"name": name, "character_id": cid, "how": how}
                                          for name, cid, how in rows]} for scene_id, rows in scenes]})


def test_only_voices_without_a_character_are_unheard_in_order_of_appearance():
    answer = heard(("s1", [("葵", "aoi", "present"), ("祖父", "", "remote"), ("祖父", "", "recollection")]),
                   ("s2", [("ラジオの声", "", "remote")]), ("s3", [(" 祖父 ", "", "remote")]))
    assert voices.unheard(answer, plan(["aoi"], ["aoi"], ["ren"])) == {
        "祖父": ["s1", "s3"], "ラジオの声": ["s2"]}


def test_every_planned_scene_must_be_answered_once_in_order():
    with pytest.raises(ValueError, match="in order"):
        voices.unheard(heard(("s2", []), ("s1", [])), plan(["aoi"], ["aoi"]))


def test_schema_limits_answers_to_the_planned_scenes_and_known_people():
    schema = voices.voices_schema(plan(["aoi"], ["ren"]), {"aoi": "葵", "ren": "蓮"})
    assert schema["$defs"]["SceneVoices"]["properties"]["scene_id"]["enum"] == ["s1", "s2"]
    assert schema["$defs"]["Voice"]["properties"]["character_id"]["enum"] == ["", "aoi", "ren"]
    assert schema["properties"]["scenes"]["minItems"] == schema["properties"]["scenes"]["maxItems"] == 2
    assert voices.define_schema("voice-1-1")["properties"]["id"]["enum"] == ["voice-1-1"]


def test_light_context_carries_the_plan_without_the_volume_guidance():
    value = plan(["aoi"])
    scene = value.scenes[0].model_copy(update={"objectives": value.scenes[0].objectives + "\n【分量の補助目安】長め"})
    context = voices.voices_context(value.model_copy(update={"scenes": [scene]}), {"aoi": "葵"})
    assert context["setting"] == {} and "分量" not in context["brief"]
    assert "未確定の写真" in context["brief"] and '"aoi": "葵"' in context["brief"]


def test_cast_adds_the_new_person_where_heard_unless_the_scene_is_full():
    grandfather, stranger = person("voice-1-1"), person("voice-1-2", "見知らぬ声")
    result = voices.cast(plan(["aoi"], ["aoi", "ren", "helper"], ["ren"]),
                         [(grandfather, ["s1", "s2"]), (stranger, ["s2"])])
    assert [scene.character_ids for scene in result.scenes] == [
        ["aoi", "voice-1-1"], ["aoi", "ren", "helper"], ["ren"]]
    # Someone heard only in full scenes has nowhere to speak, so they are not added.
    assert [row.id for row in result.new_characters] == ["voice-1-1"]


def test_unplanned_voice_is_defined_and_cast_in_the_chapter_plan(runtime, tmp_path):
    calls, control = runtime
    grandfather = person("voice-1-1").model_dump(mode="json")
    control["responses"] = [outline(), chapter_plan("s1"), FIRST, staging(FIRST)]
    control["voices"] = [
        {"scenes": [{"scene_id": "s1", "voices": [
            {"name": "葵", "character_id": "aoi", "how": "present"},
            {"name": "祖父", "character_id": "", "how": "remote"}]}]},
        grandfather]
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    voice_calls = [call for call in calls if call["purpose"] == voices.PURPOSE]
    assert len(voice_calls) == 2 and not control["voices"]
    assert "定義する人物: 祖父（id: voice-1-1）" in voice_calls[1]["messages"][-1]["content"]
    saved = read_json(tmp_path / "chapters/chapter-001.plan.json")
    assert saved["scenes"][0]["character_ids"] == ["aoi", "ren", "voice-1-1"]
    assert [row["id"] for row in saved["new_characters"]] == ["voice-1-1"]
    traces = [row for path in tmp_path.rglob("llm-metrics.json") for row in read_json(path)]
    assert [row["characters"] for row in traces if row.get("type") == "heard_voices_cast"] == [
        [{"id": "voice-1-1", "name": "祖父", "scenes": ["s1"], "cast": True}]]


def test_no_unheard_voice_leaves_the_plan_unchanged(runtime, tmp_path):
    _calls, control = runtime
    control["responses"] = [outline(), chapter_plan("s1"), FIRST, staging(FIRST)]
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    assert len(control["voice_calls"]) == 1
    saved = read_json(tmp_path / "chapters/chapter-001.plan.json")
    expected = chapter_plan("s1")
    assert saved["scenes"][0]["character_ids"] == expected["scenes"][0]["character_ids"]
    assert saved["new_characters"] == []
