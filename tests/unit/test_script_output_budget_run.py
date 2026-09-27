"""The chosen scene budget survives dispatch, truncation and interrupted replay."""

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.causal_runtime import digest
from services.worker.generation.script_budget import SceneSize, SceneTokenPolicy
from tests.unit.test_script_continuation_run import (
    FIRST,
    SECOND,
    chapter_plan,
    outline,
    prompt_text,
    read_json,
    snapshot,
    staging,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


def test_estimate_preflight_request_and_journal_use_the_same_output_tokens(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [outline(), chapter_plan("s1", "s2"), FIRST, staging(FIRST),
                            SECOND, staging(SECOND, "s2")]
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    state = read_json(tmp_path / "draft-state.json")
    writers = [call for call in calls if call["purpose"] == "script-scene"]
    for number, call in enumerate(writers, 1):
        scope = f"c001-s{number}"
        budget = state["scene_budgets"][scope]["budget"]
        saved = read_json(tmp_path / "requests" / f"{scope}-text-1.json")
        attempt = state["steps"][scope + "-text"]["attempts"][0]
        assert budget == saved["output_budget"] == attempt["output_budget"]
        assert (budget["max_tokens"] == call["extra"]["max_tokens"]
                == saved["request"]["max_tokens"] == saved["selection"]["budget"]["output_tokens"])
        assert attempt["reserved_tokens"] == saved["selection"]["budget"]["prompt_tokens"] + budget["max_tokens"]
        assert budget["max_tokens"] < call["profile"]["max_tokens"]
        assert saved["selection"]["material_selection"]["full_character_ids"] == ["aoi", "ren"]
        assert saved["selection"]["material_selection"]["included_location_ids"] == ["room"]
    assert state["scene_budgets"]["c001-s1"]["budget"]["samples"] == []
    sample = state["scene_budgets"]["c001-s2"]["budget"]["samples"][0]
    assert sample["step"] == "c001-s1-text" and sample["source_sha256"] == digest(FIRST)
    assert report["scene_output_budgets"] == state["scene_budgets"]
    assert state["chapter_plans"]["1"]["plan"]["scene_sizes"]["s1"]["body_characters"] == 2000


def test_budget_is_frozen_before_cancelled_dispatch_and_reused_on_resume(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [outline(), chapter_plan("s1", "s2"), FIRST, staging(FIRST),
                            GenerationCancelled("interrupted scene")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    before = read_json(tmp_path / "draft-state.json")["scene_budgets"]
    control["responses"] = [SECOND, staging(SECOND, "s2")]
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    after = read_json(tmp_path / "draft-state.json")["scene_budgets"]
    assert after == before
    retry_calls = [call for call in calls if call["purpose"] == "script-scene"][1:]
    assert len(retry_calls) == 2
    assert retry_calls[0]["extra"] == retry_calls[1]["extra"]
    assert FIRST in prompt_text(retry_calls[1])


def test_bounded_continuation_measures_partial_source_anchor_and_grammar(runtime, tmp_path):
    calls, control = runtime
    raw = FIRST + SECOND
    anchor = FIRST[-min(160, len(FIRST)):]
    control["responses"] = [outline(), chapter_plan("s1"),
        {"content": FIRST, "_finish_reason": "length"}, anchor + SECOND, staging(raw)]
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    first = read_json(tmp_path / "requests/c001-s1-text-1.json")
    second = read_json(tmp_path / "requests/c001-s1-continue-1.json")
    writer = [call for call in calls if call["purpose"] == "script-scene"][-1]
    assert second["request"]["messages"] == writer["messages"]
    assert FIRST in second["request"]["messages"][1]["content"]
    assert anchor in second["request"]["messages"][0]["content"]
    assert second["request"]["grammar"] == writer["extra"]["grammar"]
    assert second["selection"]["budget"]["output_tokens"] == writer["extra"]["max_tokens"]
    assert second["output_budget"] == first["output_budget"]
    assert read_json(tmp_path / "draft-state.json")["chapters"][0]["text"] == raw


def test_source_note_records_availability_and_source_hash(runtime, tmp_path):
    _, control = runtime
    control["responses"][4:5] = ["", ""]
    runner.run_script_debug(snapshot(), tmp_path)
    state = read_json(tmp_path / "draft-state.json")
    assert state["notes"][0]["available"] is False
    assert state["notes"][0]["source_sha256"] == digest(FIRST)
    request = read_json(tmp_path / "requests/plan-002-1.json")
    coverage = request["selection"]["history_coverage"][0]
    assert coverage["note_status"] == "unavailable"
    assert coverage["representation"] == "full_source"


def test_connection_is_future_only_and_scene_target_metadata_is_validated(runtime, tmp_path):
    calls, control = runtime
    planned = "写真と札を配置し、葵の応答を受けて距離を選び直す。"
    control["responses"][5]["continuation"] = planned
    runner.run_script_debug(snapshot(), tmp_path)
    writer = [call for call in calls if call["purpose"] == "script-scene"][-1]
    assert "【最初の新行動】" + planned in prompt_text(writer)
    assert prompt_text(writer).count(planned) == 1
    assert "章間の接続メモ" not in prompt_text(writer)
    with pytest.raises(ValueError, match="cannot exceed"):
        SceneSize(length_weight=1, body_characters=5, dialogue_characters=6)
    with pytest.raises(ValueError):
        SceneTokenPolicy(safety_factor=float("inf"))
