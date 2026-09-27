"""A known plot-schema defect gets one bounded recovery, without erasing history."""
import copy

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.draft_story import DraftExecutionError
from tests.unit.test_script_continuation_run import (
    allocation,
    empty_cast,
    read_json,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


def legacy_core(self, key, purpose, number, prompt, context, model, validate, schema):
    old = copy.deepcopy(schema)
    old["$defs"]["ChainCore"]["properties"]["characters"].update(minItems=1, maxItems=10)
    old["$defs"]["ChainCore"]["properties"]["characters"].pop("description", None)
    return self.structured(key, purpose, number, prompt, context, model, validate, old)


def duplicate_cast():
    value = story_chain_draft()
    value["core"]["characters"] = [copy.deepcopy(value["core"]["characters"][0]) for _ in range(3)]
    return value


def fail_legacy(runtime, monkeypatch, output):
    calls, control = runtime
    control["responses"] = [empty_cast(), duplicate_cast(), duplicate_cast()]
    with monkeypatch.context() as patch:
        patch.setattr(runner.ScriptRun, "structured_core", legacy_core)
        with pytest.raises(DraftExecutionError, match="approved main cast"):
            runner.run_script_debug(three_chapter_snapshot(), output, plot_only=True)
    assert len(calls) == 3
    state = read_json(output / "draft-state.json")
    assert "core_cast_schema" not in state["steps"]["story-chain"]
    return state


def test_old_cast_failure_keeps_original_requests_and_resumes_plot_once(runtime, tmp_path, monkeypatch):
    calls, control = runtime
    old = fail_legacy(runtime, monkeypatch, tmp_path)
    files = {path: path.read_bytes() for path in (tmp_path / "requests").glob("*")}
    cast = (tmp_path / "cast-plan.json").read_bytes()
    manifest = (tmp_path / "experiment.json").read_bytes()
    control["responses"] = [story_chain_draft(), allocation()]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
    assert report["status"] == "plot_complete"
    assert report["metrics"]["requests"] == 5
    assert report["metrics"]["charged_tokens"] == 5 * 530
    assert all(path.read_bytes() == content for path, content in files.items())
    assert (tmp_path / "cast-plan.json").read_bytes() == cast
    assert (tmp_path / "experiment.json").read_bytes() == manifest
    state = read_json(tmp_path / "draft-state.json")
    assert state["steps"]["story-chain"]["attempts"] == old["steps"]["story-chain"]["attempts"]
    assert state["steps"]["story-chain"]["core_cast_replacement"] == "story-chain-main-cast-v1"
    assert state["steps"]["story-chain-main-cast-v1"]["attempts"][0]["request"] == 4
    assert state["request_ordinal"] == 5
    schema = calls[3]["extra"]["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["ChainCore"]["properties"]["characters"]["maxItems"] == 2
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
    assert len(calls) == 5


def test_replacement_cannot_gain_more_attempts_on_repeated_resume(runtime, tmp_path, monkeypatch):
    calls, control = runtime
    fail_legacy(runtime, monkeypatch, tmp_path)
    control["responses"] = [duplicate_cast(), duplicate_cast()]
    for _ in range(3):
        with pytest.raises(DraftExecutionError, match="approved main cast"):
            runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
        assert len(calls) == 5
        state = read_json(tmp_path / "draft-state.json")
        assert len(state["steps"]["story-chain-main-cast-v1"]["attempts"]) == 2
        assert state["request_ordinal"] == 5
        assert read_json(tmp_path / "report.json")["metrics"]["charged_tokens"] == 5 * 530


def test_new_schema_failures_never_get_legacy_recovery(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [empty_cast(), duplicate_cast(), duplicate_cast()]
    for resume in (False, True):
        with pytest.raises(DraftExecutionError, match="approved main cast"):
            runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=resume)
        assert len(calls) == 3
        assert "story-chain-main-cast-v1" not in read_json(tmp_path / "draft-state.json")["steps"]


def test_valid_legacy_chain_survives_schema_change_and_allocation_resume(runtime, tmp_path, monkeypatch):
    calls, control = runtime
    control["responses"] = [empty_cast(), story_chain_draft(), GenerationCancelled("allocation stopped")]
    with monkeypatch.context() as patch:
        patch.setattr(runner.ScriptRun, "structured_core", legacy_core)
        with pytest.raises(GenerationCancelled):
            runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True)
    source = (tmp_path / "story-chain.json").read_bytes()
    control["responses"] = [allocation()]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
    assert len(calls) == 4
    assert (tmp_path / "story-chain.json").read_bytes() == source
    state = read_json(tmp_path / "draft-state.json")
    assert state["steps"]["story-chain"]["core_cast_legacy_reused_request"] == 2
    assert "story-chain-main-cast-v1" not in state["steps"]
