"""Ensemble planning preserves identities, stage scope and public script contracts."""
import copy

import pytest

from packages.narrative import parse_scene_text
from services.worker.generation import script_continuation as runner
from services.worker.generation.script_cast import (
    CastPlan,
    ScriptOptions,
    check_cast,
    script_metrics,
)
from services.worker.generation.script_plot import StoryChain, check_chain
from services.worker.generation.script_realization import (
    ChapterRealizationResponse,
    realize_chapter,
)
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    allocation,
    chapter_plan,
    prompt_text,
    read_json,
    realization_plan,
    snapshot,
    staging,
    story_chain,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


def ensemble():
    people = []
    for cid in ("neighbor", "visitor"):
        character = copy.deepcopy(snapshot()["characters"][0]["result"])
        character.update(id=cid, name=cid, role="近所の友人", settings="夕飯の献立が気になる友人。")
        people.append(character)
    return {"supporting_characters": people,
            "everyday_context": [{"character_id": p["id"], "personal_concern": "夕飯の買い物。",
                                  "contact": "仕事帰りに立ち寄れる。", "initial_knowledge": "二人とは旧知。展示の進捗は知らない。"}
                                 for p in people],
            "connections": [{"character_ids": ["aoi", "neighbor"], "relationship": "昔からの友人で気軽に話す。"}]}


def test_main_arcs_and_available_actors_are_independent():
    value = CastPlan.model_validate(ensemble())
    check_cast(value, {"aoi", "ren"})
    data = story_chain()
    data["events"][0]["steps"][0]["character_id"] = "neighbor"
    chain = StoryChain.model_validate(data)
    check_chain(chain, {"aoi", "ren"}, {"aoi", "ren", "neighbor", "visitor"})
    with pytest.raises(ValueError, match="registered"):
        check_chain(chain, {"aoi", "ren"})
    data["core"]["characters"][0]["character_id"] = "neighbor"
    chain = StoryChain.model_validate(data)
    with pytest.raises(ValueError, match="main cast"):
        check_chain(chain, {"aoi", "ren"}, {"aoi", "ren", "neighbor"})


@pytest.mark.parametrize("mutation", ["duplicate", "main", "unknown_pair", "main_pair", "missing_life"])
def test_cast_registry_rejects_only_invalid_references(mutation):
    data = ensemble()
    if mutation == "duplicate":
        data["supporting_characters"][1]["id"] = "neighbor"
    elif mutation == "main":
        data["supporting_characters"][0]["id"] = "aoi"
    elif mutation == "unknown_pair":
        data["connections"][0]["character_ids"][1] = "unknown"
    elif mutation == "main_pair":
        data["connections"][0]["character_ids"] = ["aoi", "ren"]
    else:
        data["everyday_context"].pop()
    with pytest.raises(ValueError):
        check_cast(CastPlan.model_validate(data), {"aoi", "ren"})


def test_interaction_scenes_connect_without_fabricating_main_steps():
    draft = realization_plan(2)
    main = draft["scenes"][0]
    opening = {**main, "id": "chat", "interaction": "昼ご飯の好みをからかい合う。",
               "interaction_end": "休憩を終えて作業台へ戻る。", "length_weight": 1}
    ending = {**opening, "id": "bye", "interaction_end": "挨拶を終える。"}
    main["length_weight"] = 2
    draft["scenes"] = [opening, main, ending]
    for row in draft["placements"]:
        row["scene_number"] = 2
    source = StoryChain.model_validate(story_chain()).events[:1]
    result = realize_chapter(ChapterRealizationResponse.model_validate(draft).as_realization(), source)
    assert result["scenes"][1].start_state == opening["interaction_end"]
    assert result["scenes"][2].start_state == source[0].steps[-1].result
    assert result["destination"] == source[0].steps[-1].result
    assert result["scenes"][2].end_state == "挨拶を終える。"
    assert [step for event in result["realized_events"] for step in event.steps] == source[0].steps
    assert result["realized_events"][0].start_condition == opening["interaction_end"]
    assert result["scenes"][0].required_events[0].description == opening["interaction"]
    assert "本文約2000文字、うち台詞約1000文字" in result["scenes"][1].objectives
    assert opening["interaction"] in result["scenes"][0].objectives


def test_all_done_actions_allow_a_conversation_without_replay():
    draft = realization_plan(2)
    for row in draft["placements"]:
        row.pop("scene_number")
        row.update(handling="already_done", reason_from_source="前章で実施済み。")
    draft["scenes"][0]["interaction_end"] = "休憩を終える。"
    result = realize_chapter(ChapterRealizationResponse.model_validate(draft).as_realization(), StoryChain.model_validate(story_chain()).events[:1])
    assert result["realized_events"] == []
    assert result["destination"] == draft["current_facts"]["situation"]
    assert "EVENT_1" not in result["scenes"][0].model_dump_json()


def test_planned_cast_can_debut_later_and_resume_without_unused_assets(runtime, tmp_path):
    calls, control = runtime
    cast = ensemble()
    raw = "aoi: 夕飯は決めた？\nneighbor: まだ。帰りに一緒に考えよう。\nNARRATOR: 二人は笑った。\n"
    utterances = parse_scene_text(raw, "s1", {"aoi", "neighbor"})
    stage = {"emotions": [{"utterance_id": u.id, "inner_emotion": "平静", "voice_emotion": "neutral", "delivery": ""}
                          for u in utterances], "directions": []}
    second_plan = chapter_plan("s1", continued=True)
    second_plan["scenes"][0]["character_ids"].append("neighbor")
    control["responses"] = [cast, story_chain_draft(), allocation(), chapter_plan("s1"), FIRST, staging(FIRST)]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    first = read_json(tmp_path / "exports/chapter-001/narrative.json")
    assert first["supporting_characters"] == []
    assert "neighbor" not in (tmp_path / "exports/chapter-001/asset-requirements.json").read_text(encoding="utf-8")
    assert "neighbor" in calls[1]["extra"]["response_format"]["json_schema"]["schema"]["$defs"]["ChainStep"]["properties"]["character_id"]["enum"]
    assert "4000" not in calls[0]["messages"][-1]["content"] and "2000" not in calls[1]["messages"][-1]["content"]
    assert "4000" in prompt_text(calls[3]) and "2000" in prompt_text(calls[4])
    assert "尺合わせはしません" not in calls[4]["messages"][0]["content"]
    control["responses"] = [HANDOFF, second_plan, raw, stage]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=2, resume=True)
    second = read_json(tmp_path / "exports/chapter-002/narrative.json")
    assert second["supporting_characters"] == [cast["supporting_characters"][0]]
    requirements = read_json(tmp_path / "exports/chapter-002/asset-requirements.json")
    assert any(r["target_id"] == "neighbor" for r in requirements["requirements"])
    assert not any(r["target_id"] == "visitor" for r in requirements["requirements"])
    assert "neighbor" not in prompt_text(calls[6])  # Handoff sees actual chapter cast, not future definitions.
    assert "visitor" in prompt_text(calls[7])  # The planner can still choose the future character.
    assert len([c for c in calls if c["purpose"] == "script-cast"]) == 1
    count = len(calls)
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=2, resume=True)
    assert len(calls) == count
    options = {"approval_snapshot": three_chapter_snapshot(), "script_options": {"target_body_characters": 5000}}
    with pytest.raises(ValueError, match="changed"):
        runner.run_script_debug(options, tmp_path, chapter_limit=2, resume=True)


def test_planned_cast_is_available_without_inventing_past_events(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [ensemble(), story_chain_draft(), allocation(), chapter_plan("s1"), FIRST, staging(FIRST)]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    planning = prompt_text(calls[3])
    assert "neighbor" in planning and "introduced_character_ids" in planning
    schema = calls[3]["extra"]["response_format"]["json_schema"]["schema"]
    assert "current_facts" not in schema["properties"]


def test_size_metrics_are_diagnostic_not_a_content_gate():
    result = script_metrics("aoi: 短い。\nNARRATOR: 間。\nNARRATOR: 終わり。\n")
    assert result["body_characters"] == 9 and result["dialogue_characters"] == 3
    assert result["longest_narrator_run"] == 2 and result["playback_seconds"] is None
    assert result["acceptance"] == "not_evaluated"
    with pytest.raises(ValueError):
        ScriptOptions(target_body_characters=100, target_dialogue_characters=200)
