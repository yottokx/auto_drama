"""Real app worker entrypoint, mocked inference, portable cross-job continuation."""
import copy
import json
from contextlib import nullcontext

import pytest

from packages.narrative import validate_narrative
from packages.narrative.continuity import narrative_hash
from services.coordinator.m3_bundle import validate_bundle
from services.coordinator.m3_service import _validate_script_checkpoint
from services.worker.generation import m3_pipeline, script_continuation
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.causal_runtime import digest
from services.worker.generation.draft_story import DraftBudgetError, DraftExecutionError
from services.worker.generation.script_production import (
    ProductionScriptRun,
    _generation_identity_config,
)
from services.worker.generation.workflow_version import generator_protocol
from services.worker.model_config import REGISTRY, entries
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


def test_common_model_continues_on_worker_with_different_local_paths(app_runtime, tmp_path, monkeypatch):
    calls, _ = app_runtime
    workers = [tmp_path / "worker-a", tmp_path / "worker-b"]
    for index, worker in enumerate(workers):
        model_dir = worker / f"models-{index}"
        model_dir.mkdir(parents=True)
        (model_dir / "portable.gguf").write_bytes(b"GGUFtest")
        server = worker / f"server-{index}.exe"
        server.touch()
        registry = worker / REGISTRY
        registry.parent.mkdir(parents=True)
        registry.write_text(json.dumps({"schema_version": 1, "model_dirs": [str(model_dir)],
                                       "server_executable": str(server)}), encoding="utf-8")
    # One worker uses discovery, the other an equivalent existing explicit registry.
    entry = entries(workers[1])["portable"]
    metadata = workers[1] / "worker-specific-llm.json"
    metadata.write_text(json.dumps(entry.pop("llm_base")), encoding="utf-8")
    entry["llm_config"] = metadata.name
    (workers[1] / REGISTRY).write_text(json.dumps({"schema_version": 1,
        "models": {"portable": entry}}), encoding="utf-8")
    request = job()
    request["payload"]["profile"] = {"common_settings_version": 1, "model_id": "portable",
        "temperature": 0.25, "top_p": 0.7, "reasoning_level": "none", "context_size": 32768}
    monkeypatch.setattr(m3_pipeline.pipeline, "ROOT", workers[0])
    first = generate(request, tmp_path / "first")
    monkeypatch.setattr(m3_pipeline.pipeline, "ROOT", workers[1])
    second = generate(following(request, first), tmp_path / "second")
    assert second["result"]["chapter_number"] == 2
    assert len(calls) == 8
    for work, worker in (("first", workers[0]), ("second", workers[1])):
        actual = read_json(tmp_path / work / "job-request.json")["generation_config"]
        assert str(worker) in actual["llm_base"]["model"]["relative_path"]
        identity = read_json(tmp_path / work / "script/experiment.json")["generation_config"]
        assert "llm_config" not in identity
        assert "relative_path" not in identity["llm_base"]["model"]
        assert "executable" not in identity["llm_base"]["server"]


def test_legacy_generation_identity_keeps_existing_checkpoint_shape():
    config = {"llm_config": "config/m0-llm.json", "llm": {"context_size": 16384}}
    assert _generation_identity_config(config, {"profile": {}}) == config


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


def invalid_plan():
    value = chapter_plan("s1")
    value["scenes"][0]["character_ids"] = ["unknown-character"]
    return value


def test_explicit_retry_replaces_rejected_legacy_plan_and_preserves_valid_work(app_runtime, tmp_path):
    calls, control = app_runtime
    control["responses"][1:2] = [invalid_plan(), invalid_plan()]
    request = job()
    with pytest.raises(DraftExecutionError, match="unknown characters"):
        generate(request, tmp_path)
    assert len(calls) == 3
    state_path = tmp_path / "script/draft-state.json"
    old = read_json(state_path)
    for row in old["steps"]["plan-001"]["attempts"]:
        row.pop("validation_feedback", None)  # Existing failed production journals.
    state_path.write_text(json.dumps(old, ensure_ascii=False), encoding="utf-8")
    manifest = (tmp_path / "script/experiment.json").read_bytes()
    job_input = (tmp_path / "job-request.json").read_bytes()
    saved_files = {path: path.read_bytes() for path in (tmp_path / "script/requests").glob("*")}
    with pytest.raises(DraftExecutionError, match="unknown characters"):
        generate(request, tmp_path)
    assert len(calls) == 3  # A lease resume grants no additional calls.

    control["responses"] = [chapter_plan("s1"), FIRST, staging(FIRST)]
    request["retry_generation"] = 1
    envelope = generate(request, tmp_path)
    state = read_json(state_path)
    assert len(calls) == 6
    assert [call["purpose"] for call in calls[3:]] == ["script-plan", "script-scene", "script-staging"]
    assert calls[3]["profile"] == calls[1]["profile"]
    assert state["steps"]["outline"] == old["steps"]["outline"]
    assert state["steps"]["plan-001"]["attempts"][:2] == old["steps"]["plan-001"]["attempts"]
    assert state["steps"]["plan-001"]["retry_attempt_start"] == 2
    assert state["request_ordinal"] == 6
    assert read_json(tmp_path / "script/report.json")["metrics"]["charged_tokens"] == 6 * 530
    assert all(path.read_bytes() == content for path, content in saved_files.items())
    assert (tmp_path / "script/experiment.json").read_bytes() == manifest
    assert (tmp_path / "job-request.json").read_bytes() == job_input
    before = read_json(tmp_path / "script/requests/plan-001-1.json")["request"]
    after = read_json(tmp_path / "script/requests/plan-001-3.json")["request"]
    assert after["seed"] != before["seed"]
    assert envelope["result"]["scenes"][0]["raw_text"] == FIRST
    assert generate(request, tmp_path) == envelope
    assert len(calls) == 6


def test_each_explicit_retry_grants_only_two_fresh_plan_calls(app_runtime, tmp_path):
    calls, control = app_runtime
    control["responses"][1:2] = [invalid_plan(), invalid_plan()]
    request = job()
    with pytest.raises(DraftExecutionError, match="unknown characters"):
        generate(request, tmp_path)
    for generation in (1, 2):
        request["retry_generation"] = generation
        control["responses"] = [invalid_plan(), invalid_plan()]
        with pytest.raises(DraftExecutionError, match="unknown characters"):
            generate(request, tmp_path)
        assert len(calls) == 3 + generation * 2
        with pytest.raises(DraftExecutionError, match="unknown characters"):
            generate(request, tmp_path)
        assert len(calls) == 3 + generation * 2
        state = read_json(tmp_path / "script/draft-state.json")
        assert len(state["steps"]["plan-001"]["attempts"]) == 2 + generation * 2
        assert state["request_ordinal"] == len(calls)
        assert read_json(tmp_path / "script/report.json")["metrics"]["charged_tokens"] == len(calls) * 530


def test_resume_reuses_valid_repair_without_new_plan_request(app_runtime, tmp_path):
    calls, control = app_runtime
    control["responses"][1:2] = [invalid_plan(), chapter_plan("s1"), GenerationCancelled("stop scene")]
    request = job()
    with pytest.raises(GenerationCancelled):
        generate(request, tmp_path)
    assert len(calls) == 4
    state_path = tmp_path / "script/draft-state.json"
    before = read_json(state_path)
    # A crash before committing the valid plan must still reuse the repaired answer.
    before.pop("chapter_plans", None)
    state_path.write_text(json.dumps(before, ensure_ascii=False), encoding="utf-8")
    generate(request, tmp_path)
    assert [call["purpose"] for call in calls].count("script-plan") == 2
    assert [call["purpose"] for call in calls].count("script-outline") == 1
    assert read_json(state_path)["steps"]["plan-001"]["attempts"] == before["steps"]["plan-001"]["attempts"]


@pytest.mark.parametrize("rejected,attempts", [
    ("{malformed json", 2),
    ({"content": "{unfinished json", "_finish_reason": "length"}, 1),
])
def test_explicit_retry_can_replace_malformed_and_truncated_answers(app_runtime, tmp_path, rejected, attempts):
    calls, control = app_runtime
    control["responses"][1:2] = [copy.deepcopy(rejected) for _ in range(attempts)]
    request = job()
    with pytest.raises(DraftExecutionError):
        generate(request, tmp_path)
    assert len(calls) == attempts + 1
    with pytest.raises(DraftExecutionError):
        generate(request, tmp_path)
    assert len(calls) == attempts + 1
    control["responses"] = [chapter_plan("s1"), FIRST, staging(FIRST)]
    request["retry_generation"] = 1
    envelope = generate(request, tmp_path)
    assert envelope["result"]["scenes"][0]["raw_text"] == FIRST
    assert len(calls) == attempts + 4
    assert [call["purpose"] for call in calls].count("script-outline") == 1


def test_explicit_retry_of_child_reference_repair_preserves_source_cast(app_runtime, tmp_path):
    from tests.unit.test_script_cast_references import assignments, broken_cast

    calls, control = app_runtime
    bad = assignments()
    bad["assignments"]["R1"] = "unresolved"
    control["responses"] = [broken_cast(), bad, copy.deepcopy(bad)]
    request = job(three_chapter_snapshot())
    with pytest.raises(DraftExecutionError, match="unresolved"):
        generate(request, tmp_path)
    old = read_json(tmp_path / "script/draft-state.json")
    assert len(calls) == 3
    control["responses"] = [assignments(), story_chain_draft(), allocation(), chapter_plan("s1"), FIRST, staging(FIRST)]
    request["retry_generation"] = 1
    generate(request, tmp_path)
    state = read_json(tmp_path / "script/draft-state.json")
    assert state["steps"]["cast-plan"] == old["steps"]["cast-plan"]
    assert state["cast_connection_repair"]["source_request"] == old["cast_connection_repair"]["source_request"]
    assert state["steps"]["cast-plan-connections-v1"]["retry_attempt_start"] == 2
    assert len(calls) == 9
