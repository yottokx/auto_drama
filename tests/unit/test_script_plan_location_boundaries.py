"""Pruning unreferenced backgrounds preserves used places and chapter history."""

from copy import deepcopy

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.draft_story import DraftExecutionError
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    chapter_plan,
    outline,
    read_json,
    snapshot,
    staging,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


def garden():
    return {"id": "garden", "name": "中庭", "description": "芝生と石のベンチがある。",
            "time_of_day": "夜", "atmosphere": "静かな安堵",
            "image_prompt": "quiet courtyard, stone bench, moonlight, no people"}


def unused_location():
    return {**garden(), "id": "unused", "name": "未訪問の庭"}


def test_pruning_keeps_all_used_location_attributes_and_declaration_order(runtime, tmp_path):
    _, control = runtime
    plan = chapter_plan("s1", "s2")
    room = deepcopy(plan["locations"][0])
    plan["locations"] = [garden(), unused_location(), room]
    plan["scenes"][1]["location_id"] = "garden"
    original = deepcopy(plan)
    control["responses"] = [outline(), plan, FIRST, staging(FIRST), SECOND, staging(SECOND, "s2")]

    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)

    assert report["status"] == "chapter_limit_reached"
    expected = [garden(), room]
    adopted = read_json(tmp_path / "chapters" / "chapter-001.plan.json")
    narrative = read_json(tmp_path / "exports" / "chapter-001" / "narrative.json")
    assert adopted["locations"] == narrative["locations"] == expected
    assert [row["location_id"] for row in adopted["scenes"]] == ["room", "garden"]
    assert [row["plan"]["location_id"] for row in narrative["scenes"]] == ["room", "garden"]
    assert read_json(tmp_path / "draft-state.json")["locations"] == {
        location["id"]: location for location in expected}
    assert plan == original


def test_pruning_does_not_merge_duplicate_used_location_ids(runtime, tmp_path):
    calls, control = runtime
    plan = chapter_plan("s1")
    duplicate = {**plan["locations"][0], "description": "同じIDを持つ別の場所定義。"}
    plan["locations"].extend([unused_location(), duplicate])
    control["responses"] = [outline(), plan, deepcopy(plan)]

    with pytest.raises(DraftExecutionError, match="Duplicate location identifiers"):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)

    assert [call["purpose"] for call in calls] == ["script-outline", "script-plan", "script-plan"]
    state = read_json(tmp_path / "draft-state.json")
    assert not state.get("chapter_plans")
    assert not state.get("chapter_plan_normalizations")


def test_pruning_does_not_allow_a_used_registered_location_to_be_renamed(runtime, tmp_path):
    calls, control = runtime
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    before = read_json(tmp_path / "draft-state.json")["locations"]
    plan = chapter_plan("s1", continued=True)
    plan["locations"][0]["name"] = "登録済みIDに別の場所名"
    plan["locations"].append(unused_location())
    control["responses"] = [HANDOFF, plan, deepcopy(plan)]

    with pytest.raises(DraftExecutionError, match="must retain its registered name"):
        runner.run_script_debug(snapshot(), tmp_path, resume=True)

    assert len(calls) == 7
    state = read_json(tmp_path / "draft-state.json")
    assert state["locations"] == before
    assert "2" not in state.get("chapter_plans", {})
    assert "2" not in state.get("chapter_plan_normalizations", {})


def test_unused_previous_location_remains_in_registry_and_completed_chapter(runtime, tmp_path):
    _, control = runtime
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    first_narrative = tmp_path / "exports" / "chapter-001" / "narrative.json"
    original_bytes = first_narrative.read_bytes()
    previous_registry = read_json(tmp_path / "draft-state.json")["locations"]
    plan = chapter_plan("s1", continued=True)
    plan["locations"].append(garden())
    plan["scenes"][0]["location_id"] = "garden"
    control["responses"] = [HANDOFF, plan, SECOND, staging(SECOND)]

    report = runner.run_script_debug(snapshot(), tmp_path, resume=True)

    assert report["status"] == "script_complete"
    state = read_json(tmp_path / "draft-state.json")
    assert state["locations"] == {**previous_registry, "garden": garden()}
    assert state["chapter_plan_normalizations"]["2"]["removed_unused_location_ids"] == ["room"]
    second = read_json(tmp_path / "exports" / "chapter-002" / "narrative.json")
    assert second["locations"] == [garden()]
    assert first_narrative.read_bytes() == original_bytes
