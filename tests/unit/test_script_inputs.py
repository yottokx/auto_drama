"""Scene input projection retains references without mutating saved registries."""

import json
from copy import deepcopy

import pytest

from packages.contracts.m3 import ScenePlan
from services.worker.generation.script_inputs import scene_material


def scene(**updates):
    return ScenePlan.model_validate({"id": "scene-1", "location_id": "hall", "character_ids": ["aoi"],
        "objectives": "仲間の手掛かりを調べる。", "start_state": "廊下にいる。",
        "required_events": [{"id": "event-1", "description": "手帳を読む。"}],
        "end_state": "CURRENT_ENDPOINT_ONLY_ONCE", "atmosphere": "静か", **updates})


def context():
    return {"setting": {"world": {"rule": "RULE_MUST_SURVIVE"}, "main_character_ids": ["aoi", "ren", "mio"],
        "introduced_character_ids": ["aoi", "ren"],
        "characters": [{"id": cid, "name": name, "role": "生徒", "settings": {"voice": f"DETAIL_{cid}"}}
                       for cid, name in [("aoi", "葵"), ("ren", "蓮"), ("mio", "美緒"), ("sora", "空")]],
        "locations": [{"id": "hall", "name": "廊下", "description": "ACTIVE_LOCATION_DESCRIPTION_ONCE"},
                      {"id": "roof", "name": "屋上", "description": "UNNEEDED_FUTURE_LOCATION_DETAIL"}],
        "relationships": {"pairs": [{"characterIds": ["aoi", "ren"], "relationship": "ACTIVE_RELATION"},
                                    {"characterIds": ["ren", "mio"], "relationship": "OFFSCENE_RELATION"},
                                    {"characterIds": ["mio", "sora"], "relationship": "UNRELATED_PAIR"}]},
        "planned_connections_not_events": [{"character_ids": ["aoi", "ren"], "relationship": "友人"},
                                           {"character_ids": ["mio", "sora"], "relationship": "近所"}],
        "initial_cast_context_not_events": [{"character_id": "ren", "initial_knowledge": "REN_CONTEXT"},
                                             {"character_id": "sora", "initial_knowledge": "SORA_CONTEXT"}]},
        "brief": "この場面を書く。", "planned_connection": "既出の約束から調査を続ける。",
        "future": {"core": {"external_resolution": "WHOLE_STORY_DESTINATION"}, "core_version": "plot",
                   "current_chapter": 2, "routes": [
                       {"number": 2, "title": "現在章", "role": "CURRENT_CHAPTER_ROLE", "version": "plot",
                        "events": ["CURRENT_EVENT_DUP"], "route": "CURRENT_ROUTE_DUP",
                        "conversation_topics": ["CURRENT_CONVERSATION_DUP"]},
                       {"number": 3, "title": "最終章", "role": "FINAL_CHAPTER_PURPOSE", "version": "plot",
                        "route": {"next_state": "FINAL_CHAPTER_DESTINATION"}}]},
        "chapters": [{"number": 1, "text": "aoi: PREVIOUS_ACTUAL_SOURCE"}],
        "notes": [{"number": 1, "text": "SOURCE_DERIVED_NOTE"}]}


def test_projection_is_deeply_non_destructive_and_location_and_endpoint_occur_once():
    plan, original = scene(), context()
    before = deepcopy(original)
    output = scene_material(2, plan, original, [])
    serialized = json.dumps(output, ensure_ascii=False)
    assert original == before
    assert serialized.count("ACTIVE_LOCATION_DESCRIPTION_ONCE") == 1
    assert serialized.count("CURRENT_ENDPOINT_ONLY_ONCE") == 1
    assert "UNNEEDED_FUTURE_LOCATION_DETAIL" not in serialized
    assert "RULE_MUST_SURVIVE" in serialized
    assert output["current_chapter"] == 2
    assert output["chapters"] == original["chapters"] and output["notes"] == original["notes"]
    output["setting"]["characters"][0]["settings"]["voice"] = "changed"
    output["chapters"][0]["text"] = "changed"
    assert original == before


@pytest.mark.parametrize("reference", ["蓮に聞いたことを確かめる。", "renが残した手掛かりを確かめる。"])
def test_offscene_person_referred_to_by_name_or_id_keeps_details_and_relationship(reference):
    plan = scene(objectives=reference)
    output = scene_material(2, plan, context(), [])
    characters = {row["id"]: row for row in output["setting"]["characters"]}
    assert characters["ren"]["settings"]["voice"] == "DETAIL_ren"
    assert characters["aoi"]["settings"]["voice"] == "DETAIL_aoi"
    assert characters["mio"] == {"id": "mio", "name": "美緒", "role": "生徒"}
    relations = output["setting"]["relationships"]["pairs"]
    assert {row["relationship"] for row in relations} == {"ACTIVE_RELATION", "OFFSCENE_RELATION"}
    assert output["setting"]["initial_cast_context_not_events"] == [
        {"character_id": "ren", "initial_knowledge": "REN_CONTEXT"}]


def test_offscene_person_in_chapter_connection_keeps_their_detail():
    original = context()
    original["planned_connection"] = "蓮の依頼を受けて、これから葵が調べる。"
    output = scene_material(2, scene(), original, [])
    ren = next(row for row in output["setting"]["characters"] if row["id"] == "ren")
    assert ren["settings"]["voice"] == "DETAIL_ren"
    assert output["planned_connection"] == original["planned_connection"]


def test_following_scene_and_chapter_boundaries_remain_but_current_route_duplicates_are_removed():
    following = scene(id="scene-2", location_id="roof", objectives="LONG_LATER_OBJECTIVES",
                      atmosphere="LATER_ATMOSPHERE", start_state="FOLLOWING_START",
                      end_state="FOLLOWING_END", required_events=[{"id": "later-event", "description": "LATER_ACTION"}])
    original = context()
    output = scene_material(2, scene(), original, [following])
    serialized = json.dumps(output, ensure_ascii=False)
    assert "FOLLOWING_START" in serialized and "FOLLOWING_END" in serialized and "LATER_ACTION" in serialized
    assert "LONG_LATER_OBJECTIVES" not in serialized and "LATER_ATMOSPHERE" not in serialized
    assert "CURRENT_EVENT_DUP" not in serialized and "CURRENT_ROUTE_DUP" not in serialized
    assert "CURRENT_CONVERSATION_DUP" not in serialized
    assert "CURRENT_CHAPTER_ROLE" in serialized
    assert "FINAL_CHAPTER_PURPOSE" in serialized and "FINAL_CHAPTER_DESTINATION" in serialized
    assert "WHOLE_STORY_DESTINATION" in serialized
    assert original["future"]["routes"][0]["events"] == ["CURRENT_EVENT_DUP"]
