"""Resume real request caches without losing stage identities or repair limits."""

import json

import pytest
from pydantic import BaseModel, ConfigDict, ValidationError

from services.worker.generation.cancellation import GenerationCancelled, cancellation_scope
from services.worker.generation.causal_runtime import (
    CausalRun,
    PersistentUsage,
    RepairLimitError,
    ResourceBudgetError,
)
from services.worker.generation.llm import LocalLLM
from services.worker.generation.pipeline import ROOT, load_config


class StageValue(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    value: int


class FakeLLM:
    def __init__(self, directory):
        self.output = directory / "llm"
        self.config = {"llm": {"temperature": 0.6}}
        self.profile = dict(self.config["llm"])
        self.requests = 0
        self.trace = []
        self.server_context_size = 16384
        self.responses = []

    def runtime_identity(self):
        return {"profile": dict(self.profile), "context": self.server_context_size}

    def set_profile(self, profile):
        self.profile = dict(profile)

    def chat(self, *_args, **_kwargs):
        self.requests += 1
        response = self.responses.pop(0) if self.responses else {"content": str(self.requests)}
        self.trace.append({"type": "llm_generation", "request": self.requests, "cache_hit": False,
                           "usage": {"prompt_tokens": 10, "completion_tokens": 2}})
        if isinstance(response, Exception):
            raise response
        return response


def generated(llm, value=1):
    llm.chat("test", [{"role": "user", "content": str(value)}])
    return StageValue(value=value)


def forbidden():
    pytest.fail("The successful stage must replay without generation")


def test_restart_restores_profile_and_request_ordinals_for_following_stages(tmp_path):
    payload = {"seed": 7}
    first = FakeLLM(tmp_path)

    def first_stage():
        first.set_profile({"temperature": 0.2})
        return generated(first, 1)

    with CausalRun(first, payload) as run:
        assert run.node("first", {}, StageValue, first_stage).value == 1
        assert run.node("second", {}, StageValue, lambda: generated(first, 2)).value == 2
        assert run.state["calls"] == 2
    resumed = FakeLLM(tmp_path)
    with CausalRun(resumed, payload) as run:
        assert run.node("first", {}, StageValue, forbidden).value == 1
        assert resumed.profile == {"temperature": 0.2}
        assert run.node("second", {}, StageValue, forbidden).value == 2
        assert resumed.requests == 2
        assert run.state["calls"] == 2
        assert run.node("third", {}, StageValue, lambda: generated(resumed, 3)).value == 3
        assert resumed.requests == 3
        assert run.state["calls"] == 3


@pytest.mark.parametrize("changed", ["input", "runtime", "profile", "schema"])
def test_changed_stage_inputs_or_runtime_do_not_reuse_previous_stage(tmp_path, changed):
    llm = FakeLLM(tmp_path)
    with CausalRun(llm, {}) as run:
        run.node("stage", {"source": 1}, StageValue, lambda: generated(llm))
    before = {p.name: p.read_bytes() for p in (tmp_path / "causal-stages").glob("stage-*.json")}
    resumed = FakeLLM(tmp_path)
    inputs = {"source": 2 if changed == "input" else 1}
    model = StageValue
    if changed == "runtime":
        resumed.server_context_size = 32768
    elif changed == "profile":
        resumed.set_profile({"temperature": 0.3})
    elif changed == "schema":
        class NewStageValue(StageValue):
            label: str = "new schema"
        model = NewStageValue
    with CausalRun(resumed, {}) as run:
        result = run.node("stage", inputs, model,
                          lambda: model.model_validate(generated(resumed, 2).model_dump()))
        assert result.value == 2
        assert run.state["calls"] == 2
    assert all((tmp_path / "causal-stages" / name).read_bytes() == data
               for name, data in before.items())


def test_changed_run_payload_cannot_reset_budget_in_existing_work_directory(tmp_path):
    with CausalRun(FakeLLM(tmp_path), {"seed": 1}):
        pass
    with pytest.raises(ValueError, match="different inputs/profile"):
        CausalRun(FakeLLM(tmp_path), {"seed": 2})


def test_failed_generation_is_not_a_successful_stage_and_calls_survive_restart(tmp_path):
    payload = {"workflow_limits": {"max_calls": 2}}
    llm = FakeLLM(tmp_path)
    llm.responses = [RuntimeError("offline"), RuntimeError("still offline")]
    original = llm.chat
    with pytest.raises(RuntimeError, match="offline"), CausalRun(llm, payload) as run:
        run.node("stage", {}, StageValue, lambda: generated(llm))
    assert llm.chat == original
    assert not list((tmp_path / "causal-stages").glob("stage-*.json"))
    with pytest.raises(RuntimeError, match="still offline"), CausalRun(llm, payload) as run:
        run.node("stage", {}, StageValue, lambda: generated(llm))
    resumed = FakeLLM(tmp_path)
    with CausalRun(resumed, payload) as run:
        with pytest.raises(RepairLimitError, match="LLM call budget"):
            run.node("different-stage", {}, StageValue, lambda: generated(resumed))
        assert resumed.requests == 0
        assert run.state["calls"] == 2


def reply(text):
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 2}}


def test_partial_stage_resume_replays_exact_llm_requests_without_spending_budget_twice(tmp_path):
    payload = {"seed": 8, "workflow_limits": {"max_calls": 3}}
    output = tmp_path / "llm"
    output.mkdir()
    first = LocalLLM(ROOT, load_config(), payload, output)
    calls = []

    def initial_request(_path, request):
        calls.append(request["seed"])
        if len(calls) == 2:
            first.trace.append({"type": "llm_generation", "cache_hit": False,
                                "usage": {"prompt_tokens": 10, "completion_tokens": 0}})
            raise TimeoutError("lost second response")
        return reply("one")

    first.request = initial_request

    def generate(llm):
        llm.chat("first", [{"role": "user", "content": "one"}])
        llm.chat("second", [{"role": "user", "content": "two"}])
        return StageValue(value=2)

    with pytest.raises(TimeoutError), CausalRun(first, payload) as run:
        run.node("both", {}, StageValue, lambda: generate(first))
    assert calls == [9, 10]
    second = LocalLLM(ROOT, load_config(), payload, output)
    second.request = lambda _path, request: (calls.append(request["seed"]) or reply("two"))
    with CausalRun(second, payload) as run:
        assert run.node("both", {}, StageValue, lambda: generate(second)).value == 2
        assert run.state["calls"] == 3
        assert second.requests == 2
    assert calls == [9, 10, 10]
    third = LocalLLM(ROOT, load_config(), payload, output)
    third.request = lambda *_: pytest.fail("The stage is already complete")
    with CausalRun(third, payload) as run:
        assert run.node("both", {}, StageValue, forbidden).value == 2
        assert third.requests == 2
        with pytest.raises(RepairLimitError):
            third.chat("new", [{"role": "user", "content": "new"}])


def test_repairs_are_persistent_but_reconstructing_the_same_feedback_is_not_charged_again(tmp_path):
    payload = {"workflow_limits": {"max_repairs": 2}}
    first = FakeLLM(tmp_path)
    with CausalRun(first, payload) as run:
        run.node("bad-gate", {}, StageValue, lambda: generated(first))
        run.repair("scene", "missing causal bridge")
        run.node("still-bad-gate", {}, StageValue, lambda: generated(first))
        run.repair("scene", "missing causal bridge")
    resumed = FakeLLM(tmp_path)
    with CausalRun(resumed, payload) as run:
        run.node("bad-gate", {}, StageValue, forbidden)
        run.repair("scene", "missing causal bridge")
        run.node("still-bad-gate", {}, StageValue, forbidden)
        run.repair("scene", "missing causal bridge")
        assert run.state["repairs"] == 2
        assert len([t for t in resumed.trace if t["type"] == "causal_repair_replay"]) == 2
        run.node("third-bad-gate", {}, StageValue, lambda: generated(resumed))
        with pytest.raises(RepairLimitError, match="repair budget"):
            run.repair("extraction", "moving scope must not reset the chapter budget")


def test_same_issue_without_another_llm_call_is_still_bounded(tmp_path):
    payload = {"workflow_limits": {"max_repairs": 2}}
    with CausalRun(FakeLLM(tmp_path), payload) as run:
        run.repair("scene", "same problem")
        run.repair("scene", "same problem")
        with pytest.raises(RepairLimitError):
            run.repair("scene", "same problem")
    with CausalRun(FakeLLM(tmp_path), payload) as resumed:
        resumed.repair("scene", "same problem")
        resumed.repair("scene", "same problem")
        with pytest.raises(RepairLimitError):
            resumed.repair("scene", "same problem")


@pytest.mark.parametrize("field,value", [
    ("max_calls", 0), ("max_calls", 1001), ("max_calls", True),
    ("max_repairs", 0), ("max_repairs", 31), ("max_repairs", "6"),
])
def test_invalid_limits_are_rejected_without_overwriting_existing_budget(tmp_path, field, value):
    with pytest.raises(ValueError, match="Invalid workflow"):
        CausalRun(FakeLLM(tmp_path), {"workflow_limits": {field: value}})
    assert not (tmp_path / "causal-stages" / "budget.json").exists()


def test_largest_supported_call_limit_is_preserved(tmp_path):
    run = CausalRun(FakeLLM(tmp_path), {"workflow_limits": {"max_calls": 1000}})
    assert run.max_calls == 1000
    assert run.max_repairs == 6


@pytest.mark.parametrize("corruption", ["fingerprint", "value", "requests"])
def test_cached_values_and_request_counts_are_revalidated_before_replay(tmp_path, corruption):
    llm = FakeLLM(tmp_path)
    with CausalRun(llm, {}) as run:
        run.node("stage", {}, StageValue, lambda: generated(llm))
    path = next((tmp_path / "causal-stages").glob("stage-*.json"))
    cached = json.loads(path.read_text("utf-8"))
    if corruption == "fingerprint":
        cached["key"] = "0" * 64
    elif corruption == "value":
        cached["value"] = {"value": "not an integer"}
    else:
        cached["requests"] = -1
    path.write_text(json.dumps(cached), encoding="utf-8")
    resumed = FakeLLM(tmp_path)
    with CausalRun(resumed, {}) as run:
        with pytest.raises((ValueError, ValidationError)):
            run.node("stage", {}, StageValue, forbidden)
        assert resumed.requests == 0


def test_invalid_fresh_result_is_never_written_as_success(tmp_path):
    with CausalRun(FakeLLM(tmp_path), {}) as run, pytest.raises(ValidationError):
        run.node("stage", {}, StageValue, lambda: {"value": "invalid"})
    assert not list((tmp_path / "causal-stages").glob("stage-*.json"))


def test_cancellation_after_generation_prevents_stage_adoption_and_restores_chat(tmp_path):
    llm = FakeLLM(tmp_path)
    original = llm.chat
    with cancellation_scope() as cancellation:
        def finish_after_cancellation():
            value = generated(llm)
            cancellation.cancel()
            return value
        with pytest.raises(GenerationCancelled), CausalRun(llm, {}) as run:
            run.node("stage", {}, StageValue, finish_after_cancellation)
    assert llm.chat == original
    assert not list((tmp_path / "causal-stages").glob("stage-*.json"))
    state = json.loads((tmp_path / "causal-stages" / "budget.json").read_text("utf-8"))
    assert state["calls"] == 1


def test_invalid_persisted_budget_cannot_reset_usage(tmp_path):
    with CausalRun(FakeLLM(tmp_path), {}):
        pass
    path = tmp_path / "causal-stages" / "budget.json"
    state = json.loads(path.read_text("utf-8"))
    state["calls"] = -1
    path.write_text(json.dumps(state), encoding="utf-8")
    with pytest.raises(ValueError, match="persisted causal budget"):
        CausalRun(FakeLLM(tmp_path), {})


def test_stage_names_cannot_escape_the_cache_directory(tmp_path):
    with CausalRun(FakeLLM(tmp_path), {}) as run, pytest.raises(ValueError, match="stage name"):
        run.node("../elsewhere", {}, StageValue, forbidden)


def test_token_budget_is_persistent_and_cached_stages_do_not_charge_again(tmp_path):
    payload = {"workflow_limits": {"max_tokens": 12}}
    first = FakeLLM(tmp_path)
    with CausalRun(first, payload) as run:
        run.node("first", {}, StageValue, lambda: generated(first))
        assert run.usage.totals()["prompt_tokens"] == 10
        assert run.usage.totals()["completion_tokens"] == 2
    second = FakeLLM(tmp_path)
    with CausalRun(second, payload) as run:
        run.node("first", {}, StageValue, forbidden)
        with pytest.raises(ResourceBudgetError, match="max_tokens"):
            run.node("next", {}, StageValue, lambda: generated(second))
        assert len(run.usage.data["requests"]) == 1
        assert second.requests == 1


def test_story_token_budget_spans_chapters_and_resume_cannot_reset_it(tmp_path):
    payload = {"storyline_id": "shared", "workflow_limits": {"max_tokens": 12, "max_story_tokens": 24}}
    for chapter in (1, 2):
        llm = FakeLLM(tmp_path / f"chapter-{chapter}")
        with CausalRun(llm, {**payload, "chapter_number": chapter}) as run:
            run.node("scene", {}, StageValue, lambda llm=llm: generated(llm))
            assert run.usage.totals(run.usage.chapter)["prompt_tokens"] == 10
            assert run.usage.totals()["prompt_tokens"] == 10 * chapter
    third = FakeLLM(tmp_path / "chapter-3")
    with CausalRun(third, {**payload, "chapter_number": 3}) as run:
        with pytest.raises(ResourceBudgetError, match="max_story_tokens"):
            run.node("scene", {}, StageValue, lambda: generated(third))
        assert third.requests == 0


def test_measured_failed_request_is_saved_and_an_over_budget_response_is_not_adopted(tmp_path):
    payload = {"workflow_limits": {"max_tokens": 20}}
    first = FakeLLM(tmp_path)
    first.responses = [RuntimeError("failed after measured inference")]
    with pytest.raises(RuntimeError, match="measured inference"), CausalRun(first, payload) as run:
        generated(first)
    assert run.usage.data["requests"][0]["status"] == "measured"
    assert "RuntimeError" in run.usage.data["requests"][0]["error"]
    second = FakeLLM(tmp_path)
    with CausalRun(second, payload) as run:
        with pytest.raises(ResourceBudgetError, match="max_tokens"):
            run.node("oversized", {}, StageValue, lambda: generated(second))
        assert run.usage.totals()["prompt_tokens"] == 20
        assert run.usage.totals()["completion_tokens"] == 4
    assert not list((tmp_path / "causal-stages").glob("oversized-*.json"))


@pytest.mark.parametrize("pending", [False, True])
def test_unknown_request_usage_blocks_resume_instead_of_becoming_zero(tmp_path, pending):
    llm = FakeLLM(tmp_path)
    if pending:
        run = CausalRun(llm, {})
        run.usage.begin_request()  # Simulate process death before the finally block.
    else:
        def unmeasured(*_args, **_kwargs):
            llm.requests += 1
            return {"content": "No measured usage"}
        llm.chat = unmeasured
        with CausalRun(llm, {}) as run, pytest.raises(ResourceBudgetError, match="unmeasured"):
            generated(llm)
    with pytest.raises(ResourceBudgetError, match="unmeasured"), CausalRun(FakeLLM(tmp_path), {}):
        pass


def test_job_time_includes_requests_once_and_persists_across_chapters(tmp_path, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr("services.worker.generation.causal_runtime.time.monotonic", lambda: clock[0])
    payload = {"storyline_id": "timed", "chapter_number": 1,
               "workflow_limits": {"max_elapsed_seconds": 20, "max_story_elapsed_seconds": 15}}
    config = {"llm": {"temperature": 0.6}}
    usage = PersistentUsage(tmp_path / "chapter-1", payload, config)
    with usage.job_attempt():
        clock[0] += 3  # Server startup.
        key = usage.begin_request()
        clock[0] += 5
        usage.finish_request(key, [{"type": "llm_generation", "usage": {
            "prompt_tokens": 8, "completion_tokens": 2}}], 5, None)
        clock[0] += 2  # Release.
    assert usage.summary()["story"]["elapsed_seconds"] == 10  # Not 15.
    resumed = PersistentUsage(tmp_path / "chapter-2", {**payload, "chapter_number": 2}, config)
    with resumed.job_attempt():
        clock[0] += 6
    assert resumed.summary()["story"]["elapsed_seconds"] == 16
    assert resumed.totals("1")["elapsed_seconds"] == 10
    with pytest.raises(ResourceBudgetError, match="max_story_elapsed_seconds"):
        resumed.check()


def test_unknown_interrupted_job_time_is_not_billed_as_offline_wall_time(tmp_path):
    usage = PersistentUsage(tmp_path, {}, {})
    usage.data["attempts"].append({"id": "interrupted", "chapter": "1", "status": "pending",
                                   "elapsed_seconds": None})
    usage.save()
    restarted = PersistentUsage(tmp_path, {}, {})
    assert restarted.totals()["elapsed_seconds"] == 0
    assert restarted.totals()["unfinished_attempts"] == ["interrupted"]
    with pytest.raises(ResourceBudgetError, match="unmeasured"):
        restarted.check()


@pytest.mark.parametrize("field,value", [("max_tokens", 0), ("max_tokens", True),
    ("max_story_tokens", "100"), ("max_elapsed_seconds", -1), ("max_story_elapsed_seconds", float("inf"))])
def test_invalid_resource_limits_are_rejected(field, value, tmp_path):
    with pytest.raises(ValueError, match="Invalid workflow"):
        CausalRun(FakeLLM(tmp_path), {"workflow_limits": {field: value}})


def test_old_budget_without_measurements_cannot_claim_fresh_zero_usage(tmp_path):
    with CausalRun(FakeLLM(tmp_path), {}):
        pass
    path = tmp_path / "causal-stages" / "budget.json"
    saved = json.loads(path.read_text("utf-8"))
    saved["calls"] = 1
    path.write_text(json.dumps(saved), encoding="utf-8")
    with pytest.raises(ValueError, match="no resource measurements"):
        CausalRun(FakeLLM(tmp_path), {})
