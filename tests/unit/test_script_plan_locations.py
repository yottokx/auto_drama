"""Unused background declarations are pruned without another planning request."""
import copy
import json

import pytest

from packages.contracts.m3 import Location
from services.worker.generation import narrative
from services.worker.generation.causal_runtime import digest
from services.worker.generation.draft_story import DraftExecutionError
from tests.unit.test_script_continuation_run import (
    FIRST,
    chapter_plan,
    outline,
    read_json,
    staging,
)
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import generate, job
from tests.unit.test_script_production import runtime as _runtime

runtime = _runtime
app_runtime = _app_runtime


def plan_with_unused_location():
    value = chapter_plan("s1")
    value["locations"].append({**value["locations"][0], "id": "unused", "name": "未使用の待合室"})
    return value


def test_unused_location_does_not_retry_or_reach_media_requirements(app_runtime, tmp_path):
    calls, control = app_runtime
    plan = plan_with_unused_location()
    control["responses"] = [outline(), plan, FIRST, staging(FIRST)]
    envelope = generate(job(), tmp_path)
    assert len(calls) == 4
    assert envelope["result"]["locations"] == plan["locations"][:1]
    assert envelope["result"]["scenes"][0]["plan"]["location_id"] == "room"
    state = read_json(tmp_path / "script/draft-state.json")
    source = json.loads(state["steps"]["plan-001"]["attempts"][0]["reply"]["content"])
    assert source == plan  # The model's original reply remains untouched.
    assert list(state["locations"]) == ["room"]
    assert [row["id"] for row in state["chapter_plans"]["1"]["plan"]["locations"]] == ["room"]
    assert state["chapter_plan_normalizations"]["1"]["removed_unused_location_ids"] == ["unused"]
    assert any(row["type"] == "chapter_plan_normalization" for row in envelope["trace"])
    assert generate(job(), tmp_path) == envelope
    assert len(calls) == 4


def test_cached_invalid_ids_then_unused_location_recovers_without_replanning(app_runtime, tmp_path, monkeypatch):
    calls, control = app_runtime
    valid = plan_with_unused_location()
    bad_ids = copy.deepcopy(valid)
    bad_ids["scenes"][0]["character_ids"] = ["s1", "s2"]
    control["responses"] = [outline(), bad_ids, valid]
    original_check = narrative._validate_chapter_plan

    def old_check(plan, characters):
        # Recreate the old plan validation before unused declarations were pruned.
        plan.locations.append(Location.model_validate(valid["locations"][1]))
        original_check(plan, characters)

    with monkeypatch.context() as patch:
        patch.setattr(narrative, "_validate_chapter_plan", old_check)
        with pytest.raises(DraftExecutionError, match="unused.*unused"):
            generate(job(), tmp_path)
    assert len(calls) == 3
    before = read_json(tmp_path / "script/draft-state.json")
    saved_attempts = copy.deepcopy(before["steps"]["plan-001"]["attempts"])
    sources = {path: path.read_bytes() for path in (tmp_path / "script/requests").glob("plan-001-*")}
    control["responses"] = [FIRST, staging(FIRST)]
    envelope = generate(job(), tmp_path)
    assert envelope["result"]["locations"] == valid["locations"][:1]
    assert len(calls) == 5
    assert [row["purpose"] for row in calls].count("script-plan") == 2
    state = read_json(tmp_path / "script/draft-state.json")
    assert state["steps"]["plan-001"]["attempts"] == saved_attempts
    assert all(path.read_bytes() == content for path, content in sources.items())
    assert state["request_ordinal"] == 5
    report = read_json(tmp_path / "script/report.json")
    assert report["metrics"]["charged_tokens"] == 5 * 530
    assert len(report["chapter_plan_normalizations"]) == 1
    assert digest(state["outline"]) == digest(before["outline"])


@pytest.mark.parametrize("damage, message", [
    ("location", "unknown location"),
    ("character", "unknown characters"),
    ("duplicate_location", "Duplicate location identifiers"),
])
def test_pruning_does_not_hide_broken_scene_references(app_runtime, tmp_path, damage, message):
    calls, control = app_runtime
    plan = plan_with_unused_location()
    if damage == "location":
        plan["scenes"][0]["location_id"] = "missing"
    elif damage == "character":
        plan["scenes"][0]["character_ids"] = ["missing"]
    else:
        plan["locations"].append(copy.deepcopy(plan["locations"][0]))
    control["responses"] = [outline(), plan, plan]
    with pytest.raises(DraftExecutionError, match=message):
        generate(job(), tmp_path)
    assert len(calls) == 3
    state = read_json(tmp_path / "script/draft-state.json")
    assert not state.get("chapter_plan_normalizations")
    assert not state["chapters"]
