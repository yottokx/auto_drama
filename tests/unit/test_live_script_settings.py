"""Live LLM settings preserve source, accepted stages and resource history."""
import copy

import pytest

from services.worker import model_config
from services.worker.generation import script_continuation
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.llm import ContextBudgetError
from tests.unit.test_common_plan_worker import generate as generate_plan
from tests.unit.test_common_plan_worker import plan_job
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    allocation,
    chapter_plan,
    empty_cast,
    outline,
    read_json,
    staging,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import following, generate, job

runtime = _runtime
app_runtime = _app_runtime


@pytest.fixture
def configured(app_runtime, monkeypatch):
    """Inference is mocked; exercise request/config/manifest/journal binding."""
    def select(config, payload, _root):
        config = copy.deepcopy(config)
        config["model_routing"] = {}
        profile = payload.get("profile", {})
        for key in ("model_id", "temperature", "top_p", "reasoning_level"):
            if key in profile:
                config["llm"][key] = profile[key]
        context = profile.get("context_size", 16384)
        config["llm"].update(context_size=context, max_context_size=context,
                             model_context_size=context)
        return config

    monkeypatch.setattr(model_config, "select_config", select)
    return app_runtime


def refresh(request, context=32768, *, model="new-model", effort="none"):
    request = copy.deepcopy(request)
    request["retry_generation"] = request.get("retry_generation", 0) + 1
    request["payload"].update(execution_settings_version=1,
        execution_settings_revision=request["retry_generation"],
        profile={"common_settings_version": 1, "model_id": model,
                 "context_size": context, "reasoning_level": effort,
                 "temperature": 0.3, "top_p": 0.8})
    return request


@pytest.mark.parametrize("model", ["gemma-4-31B-it-UD-Q4_K_XL", "new-model"])
def test_legacy_third_chapter_context_failure_retries_32k_preserving_history(
        configured, monkeypatch, tmp_path, model):
    calls, control = configured
    control["responses"] = [empty_cast(), story_chain_draft(), allocation(),
        chapter_plan("s1"), FIRST, staging(FIRST), HANDOFF,
        chapter_plan("s1", continued=True), SECOND, staging(SECOND), HANDOFF]
    first_job = job(three_chapter_snapshot())
    first = generate(first_job, tmp_path / "first")
    second_job = following(first_job, first)
    second = generate(second_job, tmp_path / "second")
    third_job = following(second_job, second)
    before_sources = copy.deepcopy(third_job["payload"]["script_checkpoint"])
    original_check = script_continuation.RoutedLLM.check_context

    def check(self, stage, request):
        if self.payload.get("chapter_number") != 3 or stage != "script-plan":
            return original_check(self, stage, request)
        budget = self.context_budget(prompt_tokens=9976, output_tokens=request["max_tokens"])
        self.trace.append({"type": "context_budget", "stage": stage, **budget})
        if not budget["fits"]:
            error = ContextBudgetError("9976 + 6144 + 512 exceeds 16384")
            error.budget = budget
            raise error
        return budget

    monkeypatch.setattr(script_continuation.RoutedLLM, "check_context", check)
    with pytest.raises(ContextBudgetError):
        generate(third_job, tmp_path / "third")
    old = read_json(tmp_path / "third/script/draft-state.json")
    assert old["steps"]["plan-003"]["preflight_failures"]
    new_job = refresh(third_job, model=model)
    new_job["payload"]["script_checkpoint_source"] = copy.deepcopy(second_job["payload"])
    control["responses"] = [chapter_plan("s1", continued=True), SECOND, staging(SECOND)]
    result = generate(new_job, tmp_path / "third")
    state = read_json(tmp_path / "third/script/draft-state.json")
    assert third_job["payload"]["script_checkpoint"] == before_sources
    assert state["chapters"][:2] == old["chapters"]
    assert state["carried_usage"] == old["carried_usage"]
    assert state["steps"]["handoff-003"] == old["steps"]["handoff-003"]
    assert state["steps"]["plan-003"]["preflight_failures"] == old["steps"]["plan-003"]["preflight_failures"]
    assert len(calls) == 14
    assert all(row["profile"]["context_size"] == 32768 for row in calls[-3:])
    assert calls[-3]["profile"]["model_id"] == model
    assert result["provenance"]["script_checkpoint"]["execution_settings_version"] == 1
    assert read_json(tmp_path / "third/script/report.json")["metrics"]["charged_tokens"] == 14 * 530
    assert generate(new_job, tmp_path / "third") == result
    assert len(calls) == 14


def test_settings_retry_preserves_committed_scene_and_rebudgets_unfinished_scene(configured, tmp_path):
    calls, control = configured
    request = job()
    request["payload"].update(execution_settings_version=1, execution_settings_revision=0)
    control["responses"] = [outline(), chapter_plan("s1", "s2"), FIRST, staging(FIRST),
                              GenerationCancelled("stop second scene")]
    with pytest.raises(GenerationCancelled):
        generate(request, tmp_path)
    old = read_json(tmp_path / "script/draft-state.json")
    raw = (tmp_path / "script/sources/c001-s1.raw.txt").read_bytes()
    control["responses"] = [SECOND, staging(SECOND, "s2")]
    result = generate(refresh(request), tmp_path)
    state = read_json(tmp_path / "script/draft-state.json")
    assert state["scene_commits"]["c001-s1"] == old["scene_commits"]["c001-s1"]
    assert state["steps"]["c001-s1-text"] == old["steps"]["c001-s1-text"]
    assert (tmp_path / "script/sources/c001-s1.raw.txt").read_bytes() == raw
    assert state["scene_budget_history"]["c001-s2"] == [old["scene_budgets"]["c001-s2"]]
    assert state["scene_budgets"]["c001-s2"]["budget"]["max_tokens"] == old["scene_budgets"]["c001-s2"]["budget"]["max_tokens"] + 8192
    assert state["steps"]["c001-s2-text"]["attempts"][0] == old["steps"]["c001-s2-text"]["attempts"][0]
    assert result["result"]["scenes"][0]["raw_text"] == FIRST
    assert len(calls) == 7


def test_model_and_effort_change_reuses_raw_source_waiting_for_staging(configured, tmp_path):
    calls, control = configured
    request = job()
    request["payload"].update(execution_settings_version=1, execution_settings_revision=0)
    control["responses"] = [outline(), chapter_plan("s1"), FIRST, GenerationCancelled("staging")]
    with pytest.raises(GenerationCancelled):
        generate(request, tmp_path)
    old = read_json(tmp_path / "script/draft-state.json")
    control["responses"] = [staging(FIRST)]
    result = generate(refresh(request, effort="medium"), tmp_path)
    state = read_json(tmp_path / "script/draft-state.json")
    assert state["steps"]["c001-s1-text"] == old["steps"]["c001-s1-text"]
    assert state["scene_budgets"] == old["scene_budgets"]
    assert result["result"]["scenes"][0]["raw_text"] == FIRST
    assert len(calls) == 5
    assert calls[-1]["profile"]["model_id"] == "new-model"
    assert calls[-1]["profile"]["reasoning_level"] == "medium"


def test_legacy_planning_retry_preserves_accepted_cast(configured, tmp_path):
    calls, control = configured
    request = plan_job()
    control["responses"] = [empty_cast(), GenerationCancelled("plot")]
    with pytest.raises(GenerationCancelled):
        generate_plan(request, tmp_path)
    old = read_json(tmp_path / "script/draft-state.json")
    control["responses"] = [outline()]
    result = generate_plan(refresh(request), tmp_path)
    state = read_json(tmp_path / "script/draft-state.json")
    assert state["steps"]["cast-plan"] == old["steps"]["cast-plan"]
    assert state["cast_plan"]["plan"] == old["cast_plan"]["plan"]
    assert result["result"]["cast_plan"] == empty_cast()
    assert len(calls) == 3


def test_smaller_context_rebudgets_unfinished_continuation_without_rewriting_source(configured, tmp_path):
    calls, control = configured
    request = refresh(job(), context=32768, model="first-model")
    request["retry_generation"] = 0
    request["payload"]["execution_settings_revision"] = 0
    control["responses"] = [outline(), chapter_plan("s1"),
        {"content": FIRST, "_finish_reason": "length"}, GenerationCancelled("continuation")]
    with pytest.raises(GenerationCancelled):
        generate(request, tmp_path)
    old = read_json(tmp_path / "script/draft-state.json")
    anchor = FIRST[-min(160, len(FIRST)):]
    control["responses"] = [anchor + SECOND, staging(FIRST + SECOND)]
    result = generate(refresh(request, context=16384), tmp_path)
    state = read_json(tmp_path / "script/draft-state.json")
    assert state["steps"]["c001-s1-text"] == old["steps"]["c001-s1-text"]
    assert result["result"]["scenes"][0]["raw_text"] == FIRST + SECOND
    assert len(calls) == 6
    assert calls[-2]["extra"]["max_tokens"] <= 8192
    assert state["scene_budget_history"]["c001-s1-continuation"] == [
        old["scene_budgets"]["c001-s1-continuation"]]


def test_live_next_chapter_uses_new_model_and_context_and_replays_without_regeneration(configured, tmp_path):
    calls, _ = configured
    request = job()
    request["payload"].update(execution_settings_version=1, execution_settings_revision=0)
    first = generate(request, tmp_path / "first")
    first_source = (tmp_path / "first/script/sources/c001-s1.raw.txt").read_bytes()
    next_job = following(request, first)
    next_job["payload"]["profile"] = refresh(next_job, effort="medium")["payload"]["profile"]
    second = generate(next_job, tmp_path / "second")
    assert len(calls) == 8
    assert all(call["profile"]["model_id"] == "new-model" and
               call["profile"]["context_size"] == 32768 and
               call["profile"]["reasoning_level"] == "medium" for call in calls[4:])
    assert second["provenance"]["script_checkpoint"]["generation_identity"] == first["provenance"]["script_checkpoint"]["generation_identity"]
    assert (tmp_path / "first/script/sources/c001-s1.raw.txt").read_bytes() == first_source
    assert generate(next_job, tmp_path / "second") == second
    assert len(calls) == 8


def test_live_checkpoint_rejects_changed_seed_even_with_new_settings(configured, tmp_path):
    calls, _ = configured
    request = job()
    request["payload"].update(execution_settings_version=1, execution_settings_revision=0)
    first = generate(request, tmp_path / "first")
    next_job = following(request, first)
    next_job["payload"]["profile"] = {"context_size": 32768}
    next_job["payload"]["seed"] += 1
    before = len(calls)
    with pytest.raises(ValueError, match="checkpoint"):
        generate(next_job, tmp_path / "second")
    assert len(calls) == before
