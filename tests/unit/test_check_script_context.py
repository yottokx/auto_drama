"""Capacity replay must preserve baseline plans while measuring current prompts."""

import copy
import json
from types import SimpleNamespace

import pytest

from scripts.story import check_script_context as probe
from services.worker.generation.causal_runtime import digest
from services.worker.generation.script_cast import ScriptOptions
from services.worker.generation.script_chapter_plan import FirstChapterPlan, project_plan
from services.worker.generation.script_continuation import ScriptRun
from tests.unit.test_script_continuation_run import chapter_plan, outline, snapshot


def saved_plan(continuation=""):
    plan = project_plan(FirstChapterPlan.model_validate(chapter_plan("s1", "s2")), ScriptOptions()).model_dump()
    plan["continuation"] = continuation
    return {"number": 2, "plan": plan, "sha256": digest(plan)}


def test_saved_plan_is_read_verbatim_without_new_opening_projection_or_reply_validation():
    record = saved_plan("旧章全体の接続方針。" * 40)
    state = {"chapter_plans": {"2": record}, "steps": {
        "plan-002": {"attempts": [{"status": "completed", "reply": {"content": "not JSON"}}]}}}
    original = copy.deepcopy(state)
    result = probe.saved_chapter_plan(state, 2, ScriptOptions(target_body_characters=8000))
    assert result.model_dump() == record["plan"]
    assert state == original
    assert "【最初の新行動】" not in str(result.model_dump())
    result.scenes[0].required_events.clear()
    assert state == original


@pytest.mark.parametrize("changed", ["hash", "number", "contents"])
def test_changed_saved_plan_is_rejected_instead_of_falling_back_to_old_reply(changed):
    record = saved_plan()
    if changed == "hash":
        record["sha256"] = "modified"
    elif changed == "number":
        record["number"] = 3
    else:
        record["plan"]["scenes"][0]["objectives"] = "modified"
    with pytest.raises(ValueError, match="Saved chapter plan hash or number mismatch"):
        probe.saved_chapter_plan({"chapter_plans": {"2": record}, "steps": {}}, 2, ScriptOptions())


def test_legacy_reply_keeps_long_continuation_without_injecting_it_as_an_opening():
    old = chapter_plan("s1", continued=True)
    old["continuation"] = "昔の長い章全体の説明。" * 40
    state = {"steps": {"plan-002": {"attempts": [{"status": "completed",
             "reply": {"content": json.dumps(old, ensure_ascii=False)}}]}}}
    original = copy.deepcopy(state)
    result = probe.saved_chapter_plan(state, 2, ScriptOptions())
    assert result.continuation == old["continuation"]
    assert result.scenes[0].required_events[0].description == old["scenes"][0]["required_events"][0]["description"]
    assert result.scenes[0].objectives.count("【分量の補助目安】") == 1
    assert state == original


def test_unsaved_plan_is_not_invented():
    assert probe.saved_chapter_plan({"steps": {}}, 3, ScriptOptions()) is None


@pytest.mark.parametrize("number", [2, 3])
def test_current_plan_request_is_measured_without_validating_baseline_reply(monkeypatch, tmp_path, number):
    reader = object.__new__(ScriptRun)
    reader.payload = {"approval_snapshot": snapshot(), "script_options": ScriptOptions().model_dump()}
    reader.manifest = {"approved_chapter_count": 3}
    reader.state = {"plot": outline(), "plot_sha256": "fixed", "chapters": [], "notes": [], "locations": {}}
    old_reply = {"steps": {f"plan-{number:03d}": {"attempts": [
        {"status": "completed", "reply": {"content": "not valid under any current schema"}}]}}}
    captures = []

    def measure(llm, output, key, purpose, system, context, extra, baseline):
        captures.append({"key": key, "purpose": purpose, "schema": extra["response_format"]["json_schema"]["schema"],
                         "context": context, "baseline": baseline})
        return {"step": key, "status": "fits"}

    monkeypatch.setattr(probe, "measure", measure)
    result = probe.measure_chapter_plan(reader, number, [], None, tmp_path, old_reply)
    assert result == {"step": f"plan-{number:03d}", "status": "fits"}
    assert len(captures) == 1
    assert captures[0]["schema"]["properties"]["continuation"]["maxLength"] == 240
    assert captures[0]["baseline"]["reply"]["content"] == "not valid under any current schema"
    assert "structured" not in reader.__dict__


def test_future_chapters_notes_and_locations_are_excluded_without_mutating_saved_material():
    chapters = {number: {"number": number, "text": f"chapter {number}", "narrative": {
        "locations": [{"id": f"location-{number}", "name": f"Place {number}"}]}} for number in (1, 2, 3)}
    notes = [{"number": number, "text": f"note {number}"} for number in (1, 2, 3)]
    originals = copy.deepcopy((chapters, notes))
    reader = SimpleNamespace(state={})
    probe.prior_history(reader, chapters, notes, 2)
    assert [row["number"] for row in reader.state["chapters"]] == [1]
    assert [row["number"] for row in reader.state["notes"]] == [1]
    assert list(reader.state["locations"]) == ["location-1"]
    reader.state["chapters"][0]["text"] = "changed copy"
    reader.state["notes"][0]["text"] = "changed copy"
    assert (chapters, notes) == originals


def test_frozen_output_budget_is_reused_and_hash_changes_are_rejected():
    size = project_plan(FirstChapterPlan.model_validate(chapter_plan("s1")), ScriptOptions()).scene_sizes["s1"]
    budget = {"max_tokens": 2304, "scene_size": size.model_dump(), "fixed": "saved estimate"}
    state = {"scene_budgets": {"c002-s1": {"budget": budget, "sha256": digest(budget)}}}
    llm = SimpleNamespace(profile={"max_tokens": 8192}, _output_limit=lambda profile: 8192)
    result, source = probe.probe_output_budget(state, "c002-s1", size, ["aoi", "ren"],
                                               ScriptOptions(), llm, [{"unavailable_future_sample": True}])
    assert result == budget and result is not budget
    assert source == "saved_scene_budgets"
    budget["max_tokens"] = 3072
    with pytest.raises(ValueError, match="Saved output budget hash or scene size mismatch"):
        probe.probe_output_budget(state, "c002-s1", size, ["aoi", "ren"], ScriptOptions(), llm, [])
