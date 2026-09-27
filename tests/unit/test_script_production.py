"""Real app worker entrypoint, mocked inference, portable cross-job continuation."""
import copy
from contextlib import nullcontext

import pytest

from packages.narrative import validate_narrative
from packages.narrative.continuity import narrative_hash
from services.coordinator.m3_bundle import validate_bundle
from services.coordinator.m3_service import _validate_script_checkpoint
from services.worker.generation import m3_pipeline, script_continuation
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.causal_runtime import digest
from services.worker.generation.draft_story import DraftBudgetError
from services.worker.generation.script_production import ProductionScriptRun
from services.worker.generation.workflow_version import generator_protocol
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    allocation,
    chapter_plan,
    empty_cast,
    prompt_text,
    read_json,
    snapshot,
    staging,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


@pytest.fixture
def app_runtime(runtime, monkeypatch):
    monkeypatch.setattr(m3_pipeline, "RoutedLLM", script_continuation.RoutedLLM)
    monkeypatch.setattr(m3_pipeline, "gpu_lock", lambda *_args, **_kwargs: nullcontext())
    monkeypatch.setattr(m3_pipeline.pipeline.voice_session, "prepare_job", lambda *_: None)
    return runtime


def job(source=None):
    return {"kind": "m3_narrative", "payload": {"schema_version": 1, "seed": 123,
        "chapter_number": 1, "storyline_id": "app-storyline", "production_id": "app-production-1",
        "m4": True, "approval_snapshot": source or snapshot(), "profile": {},
        "story_workflow_version": 2, "workflow_policy": "script_continuation_v1",
        "generator_protocol": generator_protocol("causal", "script_continuation_v1")}}


def generate(request, work):
    bundle = m3_pipeline.generate_job(request, work)
    return validate_bundle(bundle, "m3_narrative")[0]


def following(request, envelope):
    number = request["payload"]["chapter_number"] + 1
    return {**request, "payload": {**request["payload"], "chapter_number": number,
        "production_id": f"app-production-{number}", "previous_narrative": envelope["result"],
        "previous_narrative_artifact_id": f"coordinator-artifact-{number - 1}",
        "script_checkpoint": copy.deepcopy(envelope["provenance"]["script_checkpoint"])}}


def test_three_separate_jobs_preserve_plot_sources_lineage_budgets_and_cache(app_runtime, tmp_path):
    calls, control = app_runtime
    control["responses"] = [empty_cast(), story_chain_draft(), allocation(), chapter_plan("s1"), FIRST, staging(FIRST),
        HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND),
        HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND)]
    request = job(three_chapter_snapshot())
    previous = None
    totals = []
    for number in range(1, 4):
        work = tmp_path / f"independent-worker-{number}"
        envelope = generate(request, work)
        result = validate_narrative(envelope["result"], request["payload"]["approval_snapshot"], previous,
            expected_previous_artifact_id=request["payload"].get("previous_narrative_artifact_id"))
        _validate_script_checkpoint(request["payload"], result, envelope["provenance"])
        assert result.storyline_id == "app-storyline" and result.chapter_number == number
        assert result.scenes[0].directions and result.scenes[0].utterances
        assert all(scene.review.policy == "not_evaluated" for scene in result.scenes)
        assert not (work / "script" / "exports").exists()
        checkpoint = envelope["provenance"]["script_checkpoint"]
        assert checkpoint["narrative_hash"] == narrative_hash(result)
        assert checkpoint["sha256"] == digest({k: v for k, v in checkpoint.items() if k != "sha256"})
        assert all("narrative" not in row and "export" not in row for row in checkpoint["state"]["chapters"])
        report = read_json(work / "script" / "report.json")
        totals.append(report["metrics"]["requests"])
        assert report["execution_mode"] == "production" and report["exported_chapter_count"] == 0
        assert generate(request, work) == envelope  # uncertain upload retries do no inference
        request = following(request, envelope)
        previous = result
    assert totals == [6, 10, 14] and len(calls) == 14
    assert FIRST in prompt_text(calls[7]) and FIRST in prompt_text(calls[11])
    assert SECOND in prompt_text(calls[11])
    assert [call["purpose"] for call in calls].count("script-outline") == 1
    assert all(call["profile"]["reasoning_level"] == "none" for call in calls)
    assert calls[4]["profile"]["max_tokens"] > 3072  # M2's old cap must not replace writer budgets
    assert request["payload"]["script_checkpoint"]["state"]["writer_samples"]


@pytest.mark.parametrize("change", ["missing", "source", "storyline", "settings", "seed", "checksum", "note"])
def test_predecessor_mismatch_stops_before_model_calls(app_runtime, tmp_path, change):
    calls, _ = app_runtime
    first = job()
    request = following(first, generate(first, tmp_path / "first"))
    if change == "missing":
        del request["payload"]["script_checkpoint"]
    elif change == "source":
        request["payload"]["previous_narrative"]["title"] = "別の本文"
    elif change in {"storyline", "seed"}:
        key = "storyline_id" if change == "storyline" else "seed"
        request["payload"][key] = "another-story" if change == "storyline" else 456
    elif change == "settings":
        request["payload"]["profile"]["context_size"] = 32768
    else:
        checkpoint = request["payload"]["script_checkpoint"]
        if change == "checksum":
            checkpoint["sha256"] = "bad"
        else:
            checkpoint["state"]["notes"] = [{"number": 1, "source_sha256": "wrong", "text": "別の章のメモ"}]
            checkpoint["sha256"] = digest({k: v for k, v in checkpoint.items() if k != "sha256"})
    before = len(calls)
    with pytest.raises(ValueError, match="checkpoint"):
        generate(request, tmp_path / "second")
    assert len(calls) == before


def test_cancellation_resumes_saved_raw_script_without_rewriting_it(app_runtime, tmp_path):
    calls, control = app_runtime
    control["responses"].insert(3, GenerationCancelled("stop"))
    request = job()
    with pytest.raises(GenerationCancelled):
        generate(request, tmp_path)
    assert (tmp_path / "script/sources/c001-s1.raw.txt").read_text(encoding="utf-8") == FIRST
    result = generate(request, tmp_path)
    assert result["result"]["scenes"][0]["raw_text"] == FIRST
    assert [call["purpose"] for call in calls].count("script-scene") == 1
    assert [call["purpose"] for call in calls].count("script-staging") == 2


def test_story_budget_is_carried_into_a_fresh_worker_directory(app_runtime, tmp_path):
    calls, _ = app_runtime
    request = job()
    request["payload"]["workflow_limits"] = {"max_story_tokens": 24000}
    first = generate(request, tmp_path / "first")
    request = following(request, first)
    checkpoint = request["payload"]["script_checkpoint"]
    checkpoint["state"]["carried_usage"][0]["charged_tokens"] = 25000
    checkpoint["sha256"] = digest({k: v for k, v in checkpoint.items() if k != "sha256"})
    before = len(calls)
    with pytest.raises(DraftBudgetError, match="story tokens"):
        generate(request, tmp_path / "second")
    assert len(calls) == before


def test_context_can_expand_for_a_new_production(app_runtime, tmp_path, monkeypatch):
    calls, _ = app_runtime
    config = m3_pipeline.pipeline.load_config()
    config["llm"]["context_size"] = 32768
    monkeypatch.setattr(m3_pipeline.pipeline, "load_config", lambda: config)
    request = job()
    request["payload"]["profile"]["context_size"] = 32768
    generate(request, tmp_path)
    assert all(call["profile"]["context_size"] == 32768 for call in calls)


def test_single_main_missing_relationships_retries_original_job_and_continues(app_runtime, tmp_path, monkeypatch):
    calls, control = app_runtime
    source = three_chapter_snapshot()
    source["characters"] = source["characters"][:1]
    del source["relationships"]  # Exactly how the app stores single-main approvals.
    request = job(source)
    original = copy.deepcopy(request)
    real_setting = ProductionScriptRun.setting

    def old_setting(self, *args, **kwargs):
        return self.payload["approval_snapshot"]["relationships"]["result"]

    monkeypatch.setattr(ProductionScriptRun, "setting", old_setting)
    with pytest.raises(KeyError, match="relationships"):
        generate(request, tmp_path / "chapter-1")
    assert calls == []
    saved_identity = read_json(tmp_path / "chapter-1/script/experiment.json")["input_sha256"]
    monkeypatch.setattr(ProductionScriptRun, "setting", real_setting)

    chain = story_chain_draft()
    chain["core"]["characters"] = chain["core"]["characters"][:1]
    for event in chain["events"]:
        for step in event["steps"]:
            step["character_id"] = "aoi"
    plan = chapter_plan("s1")
    plan["scenes"][0]["character_ids"] = ["aoi"]
    later_plan = {**copy.deepcopy(plan), "continuation": "葵が札の位置を変え、展示を確かめる。"}
    first = FIRST.replace("ren:", "aoi:")
    second = SECOND.replace("ren:", "aoi:")
    control["responses"] = [empty_cast(), chain, allocation(), plan, first, staging(first),
        HANDOFF, later_plan, second, staging(second), HANDOFF, later_plan, second, staging(second)]
    for number in range(1, 4):
        envelope = generate(request, tmp_path / f"chapter-{number}")
        result = envelope["result"]
        assert result["chapter_number"] == number
        assert "relationships" not in request["payload"]["approval_snapshot"]
        assert envelope["provenance"]["script_checkpoint"]["approval_sha256"] == digest(source)
        assert generate(request, tmp_path / f"chapter-{number}") == envelope
        request = following(request, envelope)
    assert len(calls) == 14
    assert original["payload"]["approval_snapshot"] == source
    assert read_json(tmp_path / "chapter-1/script/experiment.json")["input_sha256"] == saved_identity
