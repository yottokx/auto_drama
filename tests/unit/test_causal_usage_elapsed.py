"""Charge live chapter wall time, including model switches, exactly once."""

import pytest

from services.worker.generation.causal_runtime import PersistentUsage, ResourceBudgetError


@pytest.fixture
def clock(monkeypatch):
    value = [0.0]
    monkeypatch.setattr("services.worker.generation.causal_runtime.time.monotonic", lambda: value[0])
    return value


def measured_request(usage, clock, seconds):
    key = usage.begin_request()
    clock[0] += seconds
    usage.finish_request(key, [{"type": "llm_generation", "usage": {
        "prompt_tokens": 8, "completion_tokens": 2}}], seconds, None)


def test_active_elapsed_includes_switches_without_double_counting_requests(tmp_path, clock):
    payload = {"workflow_limits": {"max_elapsed_seconds": 10}}
    usage = PersistentUsage(tmp_path, payload, {})
    with usage.job_attempt():
        clock[0] += 3  # First model load.
        measured_request(usage, clock, 4)
        clock[0] += 2  # Model switch outside the request.
        shared_view = PersistentUsage(tmp_path, payload, {})
        assert shared_view.summary()["chapter"]["elapsed_seconds"] == 9
        shared_view.check()
        clock[0] += 2
        with pytest.raises(ResourceBudgetError, match="max_elapsed_seconds"):
            shared_view.begin_request()
        assert len(shared_view.data["requests"]) == 1
    assert usage.summary()["chapter"]["elapsed_seconds"] == 11
    clock[0] += 500  # No elapsed charge after the invocation ends.
    assert usage.summary()["story"]["elapsed_seconds"] == 11


def test_live_load_time_at_exact_limit_prevents_new_request(tmp_path, clock):
    usage = PersistentUsage(tmp_path, {"workflow_limits": {"max_elapsed_seconds": 5}}, {})
    with usage.job_attempt():
        clock[0] += 5
        usage.check()  # Existing measured work may finish exactly at the limit.
        with pytest.raises(ResourceBudgetError, match="max_elapsed_seconds"):
            usage.begin_request()
        assert usage.data["requests"] == []
    assert usage.summary()["chapter"]["elapsed_seconds"] == 5


def test_restarted_chapter_adds_live_time_to_prior_attempt_only(tmp_path, clock):
    payload = {"workflow_limits": {"max_elapsed_seconds": 5}}
    first = PersistentUsage(tmp_path, payload, {})
    with first.job_attempt():
        clock[0] += 3
    clock[0] += 1000
    resumed = PersistentUsage(tmp_path, payload, {})
    with resumed.job_attempt():
        clock[0] += 3
        with pytest.raises(ResourceBudgetError, match="max_elapsed_seconds"):
            resumed.begin_request()
        assert resumed.summary()["chapter"]["elapsed_seconds"] == 6
    assert PersistentUsage(tmp_path, payload, {}).summary()["story"]["elapsed_seconds"] == 6


def test_story_limit_includes_live_next_chapter_without_charging_previous_chapter(tmp_path, clock):
    payload = {"storyline_id": "timed", "workflow_limits": {
        "max_elapsed_seconds": 100, "max_story_elapsed_seconds": 15}}
    first = PersistentUsage(tmp_path / "chapter-1", {**payload, "chapter_number": 1}, {})
    with first.job_attempt():
        clock[0] += 8
    clock[0] += 100
    second = PersistentUsage(tmp_path / "chapter-2", {**payload, "chapter_number": 2}, {})
    with second.job_attempt():
        measured_request(second, clock, 2)
        clock[0] += 6
        assert second.totals("1")["elapsed_seconds"] == 8
        assert second.totals("2")["elapsed_seconds"] == 8
        assert second.totals()["elapsed_seconds"] == 16
        with pytest.raises(ResourceBudgetError, match="max_story_elapsed_seconds"):
            second.begin_request()
    assert second.summary()["story"]["elapsed_seconds"] == 16


def test_other_experiment_does_not_inherit_live_attempt_with_same_id(tmp_path, clock):
    first = PersistentUsage(tmp_path / "first", {"storyline_id": "first"}, {})
    with first.job_attempt():
        clock[0] += 8
        other = PersistentUsage(tmp_path / "other", {"storyline_id": "other"}, {})
        other.data["attempts"].append({"id": "attempt-1", "chapter": "1", "status": "pending",
                                       "elapsed_seconds": None})
        other.save()
        assert other.summary()["story"]["elapsed_seconds"] == 0
        with pytest.raises(ResourceBudgetError, match="unmeasured"):
            other.check()
