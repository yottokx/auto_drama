"""Alternate looks of a main character reach the plot, each scene plan and the exported stage."""

import copy
import json
import zipfile

import pytest

from packages.contracts.outfits import outfits_after, snapshot_outfits
from services.worker.generation import script_continuation as runner
from services.worker.generation.draft_story import DraftExecutionError
from services.worker.generation.script_chapter_plan import FirstChapterPlan, settle_scene_outfits
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    chapter_plan,
    empty_cast,
    prompt_text,
    read_json,
    staging,
    story_chain,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as runtime  # noqa: PLC0414

OUTFITS = [{"id": "default", "name": "仕事着", "description": "紺色のシャツと灰色のパンツ。"},
           {"id": "festival", "name": "浴衣", "description": "藍色の浴衣に白い帯。", "occasion": "開場日の夜だけ着る。"}]


def snapshot():
    source = three_chapter_snapshot()
    source["characters"][0]["outfits"] = copy.deepcopy(OUTFITS)
    return source


def presentation(appearances):
    value = {f"chapter_{n}": {"title": f"章{n}", "role": "結果を受けて進む。", "conversation_topics": [],
                              "outfit_appearances": appearances.get(n, [])} for n in (1, 2, 3)}
    value["foreshadowing"] = []
    return value


def plan(*outfits, continued=False):
    value = chapter_plan("s1", continued=continued)
    value["scenes"][0]["outfits"] = list(outfits)
    return value


def moments(*utterance_ids):
    return {"moments": [{"character_id": "aoi", "utterance_id": uid} for uid in utterance_ids]}


def chain():
    value = story_chain()
    value["events"] = value["events"][:3]
    return story_chain_draft(value)


WORN = {"character_id": "aoi", "outfit_id": "festival", "occasion": "開場の夜に浴衣で受付に立つ。"}


def test_look_is_allocated_planned_and_shown_by_its_own_portrait(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [
        empty_cast(), chain(), presentation({2: [WORN]}),
        plan({"character_id": "aoi", "outfit_id": "default", "change_to": None}), FIRST, staging(FIRST),
        HANDOFF, plan({"character_id": "aoi", "outfit_id": "default", "change_to": "festival"}, continued=True),
        SECOND, staging(SECOND), moments("s1-u2")]
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=2)
    assert report["saved_chapter_count"] == 2
    by_purpose = {}
    for call in calls:
        by_purpose.setdefault(call["purpose"], []).append(call)
    assert "character_outfits" in prompt_text(by_purpose["script-outline"][0])
    schema = by_purpose["script-allocation"][0]["extra"]["response_format"]["json_schema"]["schema"]
    (item,) = schema["$defs"]["ChapterPresentation"]["properties"]["outfit_appearances"]["items"]["anyOf"]
    assert item["properties"]["character_id"]["enum"] == ["aoi"]
    assert item["properties"]["outfit_id"]["enum"] == ["default", "festival"]
    first_plan, second_plan = by_purpose["script-plan"]
    scene = first_plan["extra"]["response_format"]["json_schema"]["schema"]["$defs"]["WeightedScene"]
    assert list(scene["properties"])[list(scene["properties"]).index("character_ids") + 1] == "outfits"
    (choice,) = scene["properties"]["outfits"]["items"]["anyOf"]
    assert choice["properties"]["character_id"]["enum"] == ["aoi"]
    assert choice["properties"]["outfit_id"]["enum"] == ["default", "festival"]
    assert "前章の終了時点での各人物の姿" not in prompt_text(first_plan)
    assert '"aoi": "default"' in prompt_text(second_plan)
    first_scene, second_scene = by_purpose["script-scene"]
    assert "仕事着" in prompt_text(first_scene) and "藍色の浴衣" not in prompt_text(first_scene)
    assert "changes_during_scene_to" in prompt_text(second_scene) and "藍色の浴衣" in prompt_text(second_scene)
    assert "開場日の夜だけ着る" not in prompt_text(second_scene)

    plot = read_json(tmp_path / "plot.json")
    assert [row.get("outfit_appearances", []) for row in plot["chapters"]] == [[], [WORN], []]
    first = read_json(tmp_path / "exports" / "chapter-001" / "narrative.json")
    second = read_json(tmp_path / "exports" / "chapter-002" / "narrative.json")
    assert "outfits" not in first["scenes"][0]["plan"]
    assert second["scenes"][0]["plan"]["outfits"] == [
        {"character_id": "aoi", "outfit_id": "default", "change_to": "festival"}]
    moment = by_purpose["script-staging"][-1]
    assert '"from": "仕事着", "to": "浴衣", "expected": true' in prompt_text(moment)
    choices = moment["extra"]["response_format"]["json_schema"]["schema"]["$defs"]["OutfitMoment"]["properties"]
    assert choices["utterance_id"]["anyOf"][0]["enum"] == ["s1-u1", "s1-u2", "s1-u3"]
    # She enters in her base look and changes just before the line that shows the change.
    exported = tmp_path / "exports" / "chapter-002"
    script = read_json(exported / "script.json")
    aoi = next(row for row in script["characters"] if row.get("outfits"))
    assert [row["id"] for row in aoi["outfits"]] == ["festival"]
    assert [(row["kind"], row["utterance_id"], row["timing"], row["outfit_id"])
            for row in script["directions"] if "outfit_id" in row] == [("outfit", "s1-u2", "before", "festival")]
    files = {row["id"]: row["filename"] for row in script["assets"]}
    worn = files[aoi["outfits"][0]["image_asset_id"]]
    with zipfile.ZipFile(exported / "tyrano-source.zip") as archive:
        scenario = archive.read("data/scenario/first.ks").decode("utf-8")
        entries = json.loads(archive.read("data/others/auto_drama_states.json"))["entries"]
    changed = f'[chara_mod name="ad_{aoi["id"]}" storage="{worn}" time="400"]'
    assert scenario.index('[ad_line id="s1-u2"]') < scenario.index(changed) < scenario.index('[ad_say id="s1-u2"]')
    stored = {uid: next(row["storage"] for row in entry["characters"] if row["id"] == aoi["id"])
              for uid, entry in entries.items() if entry["characters"]}
    assert stored["s1-u2"] == files[aoi["image_asset_id"]] and stored["s1-u3"] == worn


def test_scene_that_opens_in_a_look_enters_with_that_portrait(runtime, tmp_path):
    _, control = runtime
    control["responses"] = [
        empty_cast(), chain(), presentation({1: [WORN]}),
        plan({"character_id": "aoi", "outfit_id": "festival", "change_to": None}), FIRST, staging(FIRST)]
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    exported = tmp_path / "exports" / "chapter-001"
    script = read_json(exported / "script.json")
    aoi = next(row for row in script["characters"] if row.get("outfits"))
    entrance = next(row for row in script["directions"] if row["kind"] == "enter" and row["character_id"] == aoi["id"])
    assert entrance["outfit_id"] == "festival"
    files = {row["id"]: row["filename"] for row in script["assets"]}
    with zipfile.ZipFile(exported / "tyrano-source.zip") as archive:
        scenario = archive.read("data/scenario/first.ks").decode("utf-8")
        assert f"data/fgimage/{files[aoi['outfits'][0]['image_asset_id']]}" in archive.namelist()
    worn = f'[chara_mod name="ad_{aoi["id"]}" storage="{files[aoi["outfits"][0]["image_asset_id"]]}" time="0"]'
    assert scenario.index(worn) < scenario.index(f'[chara_show name="ad_{aoi["id"]}"')


def test_unallocated_look_is_rejected_then_accepted_after_feedback(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [empty_cast(), chain(), presentation({}), presentation({3: [WORN]})]
    runner.run_script_debug(snapshot(), tmp_path, plot_only=True)
    retry = [call for call in calls if call["purpose"] == "script-allocation"][1]
    assert "Not assigned to any chapter: [('aoi', 'festival')]" in prompt_text(retry)
    assert read_json(tmp_path / "plot.json")["chapters"][2]["outfit_appearances"] == [WORN]


def test_stating_the_base_look_for_a_chapter_is_not_an_appearance(runtime, tmp_path):
    _, control = runtime
    kept = {"character_id": "aoi", "outfit_id": "default", "occasion": "なし"}
    control["responses"] = [empty_cast(), chain(), presentation({1: [kept], 3: [kept, WORN, {**WORN, "occasion": "別の機会。"}]})]
    runner.run_script_debug(snapshot(), tmp_path, plot_only=True)
    plot = read_json(tmp_path / "plot.json")
    assert [row.get("outfit_appearances", []) for row in plot["chapters"]] == [[], [], [WORN]]


def test_chapter_must_show_the_look_it_was_allocated(runtime, tmp_path):
    _, control = runtime
    control["responses"] = [empty_cast(), chain(), presentation({1: [WORN]}), plan(), plan()]
    with pytest.raises(DraftExecutionError, match="Missing: \\[\\('aoi', 'festival'\\)\\]"):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)


def test_settling_keeps_only_real_choices_and_rejects_unknown_looks():
    outfits = snapshot_outfits(snapshot())
    draft = plan({"character_id": "aoi", "outfit_id": "festival", "change_to": "festival"})
    draft["scenes"][0]["length_weight"] = 1
    draft["scenes"].append({**copy.deepcopy(draft["scenes"][0]), "id": "s2", "character_ids": ["ren"],
                            "outfits": [{"character_id": "aoi", "outfit_id": "festival", "change_to": None}]})
    draft["scenes"].append({**copy.deepcopy(draft["scenes"][0]), "id": "s3",
                            "outfits": [{"character_id": "aoi", "outfit_id": "default", "change_to": None}]})
    settled, notes = settle_scene_outfits(FirstChapterPlan.model_validate(draft), outfits, [("aoi", "festival")])
    assert [json.loads(scene.model_dump_json()).get("outfits") for scene in settled.scenes] == [
        [{"character_id": "aoi", "outfit_id": "festival"}], None, None]
    assert notes == [{"scene_id": "s2", "character_id": "aoi", "dropped": "character_not_in_scene"}]
    assert outfits_after(settled.scenes) == {"aoi": "default", "ren": "default"}
    draft["scenes"][0]["outfits"][0]["outfit_id"] = "armor"
    with pytest.raises(ValueError, match="has no such look"):
        settle_scene_outfits(FirstChapterPlan.model_validate(draft), outfits, [])


@pytest.mark.parametrize("answers", [[moments(None)], [{"moments": []}, {"moments": []}]])
def test_change_the_text_does_not_show_leaves_the_portrait_until_the_next_entrance(runtime, tmp_path, answers):
    _, control = runtime
    control["responses"] = [
        empty_cast(), chain(), presentation({1: [WORN]}),
        plan({"character_id": "aoi", "outfit_id": "default", "change_to": "festival"}), FIRST, staging(FIRST),
        *answers]
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    assert report["exported_chapter_count"] == 1
    script = read_json(tmp_path / "exports" / "chapter-001" / "script.json")
    assert all(row["kind"] != "outfit" for row in script["directions"])
    record = read_json(tmp_path / "draft-state.json")["outfit_moments"]["c001-s1"]
    assert record["moments"] == ({"aoi": None} if len(answers) == 1 else {}) and ("error" in record) == (len(answers) == 2)


def test_change_due_before_the_next_scene_is_asked_about_and_may_return_to_the_base_look(runtime, tmp_path):
    calls, control = runtime
    draft = plan({"character_id": "aoi", "outfit_id": "festival", "change_to": None})
    draft["scenes"].append({**copy.deepcopy(draft["scenes"][0]), "id": "s2", "outfits": []})
    control["responses"] = [
        empty_cast(), chain(), presentation({1: [WORN]}), draft,
        FIRST, staging(FIRST), moments("s1-u3"), SECOND, staging(SECOND, "s2")]
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    asked = [call for call in calls if '"expected": false' in prompt_text(call)]
    assert len(asked) == 1 and '"from": "浴衣", "to": "仕事着"' in prompt_text(asked[0])
    script = read_json(tmp_path / "exports" / "chapter-001" / "script.json")
    assert [(row["utterance_id"], row["outfit_id"]) for row in script["directions"] if row["kind"] == "outfit"] == [
        ("s1-u3", "default")]
    aoi = next(row for row in script["characters"] if row.get("outfits"))
    entrances = [row.get("outfit_id") for row in script["directions"]
                 if row["kind"] == "enter" and row["character_id"] == aoi["id"]]
    assert entrances == ["festival", None]
