"""Text-only experiments keep production validation, source lineage and resumability."""

import copy
import io
import json
import zipfile
from contextlib import nullcontext

import pytest

from packages.narrative import parse_scene_text
from services.worker.generation import text_debug
from tests.unit.test_m2_generation import character, world
from tests.unit.test_m4_narrative import chapter_fixture


def snapshot():
    return {
        "world": {"result": world()},
        "characters": [{"result": {**character(), "id": cid, "name": cid}}
                       for cid in ("Hero", "keeper")],
        "relationships": {"result": {"pairs": [{"characterIds": ["Hero", "keeper"],
            "summary": "門を隔てて対面する", "firstToSecond": "通してほしい", "secondToFirst": "警戒する"}]}},
    }


def bundle(envelope, *, media=False):
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("result.json", json.dumps(envelope, ensure_ascii=False))
        if media:
            archive.writestr("image.png", b"not-allowed")
    return output.getvalue()


@pytest.fixture
def runtime(monkeypatch):
    calls = []
    control = {"fail_chapter": None, "wrong_lineage": False, "media": False,
               "bad_provenance": None, "timings": False, "crash_after_bundle": False}
    monkeypatch.setattr(text_debug.pipeline, "load_config", lambda: {
        "llm": {"model_id": "test-model", "context_size": 16384}})

    def generate(job, work):
        calls.append(copy.deepcopy(job))
        if (work / "result.zip").exists():
            return (work / "result.zip").read_bytes()
        payload = job["payload"]
        number = payload["chapter_number"]
        cache = work / "llm-v6"
        cache.mkdir(exist_ok=True)
        text_debug.write_json(cache / "01-scene-text-s1-example.json", {
            "request": {}, "request_sha256": "example",
            "response": {"choices": [{"message": {"content": "Hero: 未採用の途中台詞"}}],
                         "usage": {"prompt_tokens": 31, "completion_tokens": 7}},
        })
        timing = [{"type": "model_load", "elapsed_seconds": 2.2},
                  {"type": "llm_generation", "elapsed_seconds": 3.1},
                  {"type": "model_release", "elapsed_seconds": 0.4}]
        if control["timings"]:
            text_debug.write_json(work / "generation-trace.json", timing[:-1])
            text_debug.write_json(cache / "llm-metrics.json", timing)
        if number == control["fail_chapter"]:
            raise RuntimeError("scene review rejected")
        result, _ = chapter_fixture(number)
        result.update(storyline_id=payload["storyline_id"],
                      previous_narrative_artifact_id=payload.get("previous_narrative_artifact_id"),
                      previous_state_hash=payload.get("previous_state_hash"))
        if number > 1:
            result["start_state"] = payload["previous_narrative"]["end_state"]
        if control["wrong_lineage"]:
            result["storyline_id"] = "another-story"
        result["title"] = "閉じた扉<script>alert(1)</script>"
        raw = result["scenes"][0]["raw_text"].replace("扉を開けてくれ。", "扉を開けてくれ。<img src=x>")
        result["scenes"][0]["raw_text"] = raw
        result["scenes"][0]["utterances"] = [item.model_dump() for item in parse_scene_text(
            raw, "s1", {"Hero", "keeper"})]
        provenance = {"seed": payload["seed"], "profile": payload["profile"],
                      "input_sha256": text_debug._hash({"kind": job["kind"], "payload": payload}),
                      "generator_protocol": payload["generator_protocol"],
                      "prompt_version": payload["generator_protocol"]["prompt"],
                      "llm": {"requests": 1}}
        if control["bad_provenance"]:
            provenance[control["bad_provenance"]] = "wrong"
        content = bundle({"schema_version": 1, "kind": job["kind"], "result": result,
                          "provenance": provenance,
                          "trace": timing if control["timings"] else [
                              {"type": "test_review", "passed": True}]}, media=control["media"])
        if control["crash_after_bundle"]:
            (work / "result.zip").write_bytes(content)
            raise RuntimeError("Crash after worker cache before experimental adoption")
        return content

    monkeypatch.setattr(text_debug.m3_pipeline, "generate_job", generate)
    return calls, control


@pytest.fixture
def metered_runtime(runtime, monkeypatch):
    """Exercise real budget persistence with the model/result boundary stubbed."""
    calls, control = runtime
    control["usage_missing"] = False
    generate = text_debug.m3_pipeline.generate_job

    def measured(job, work):
        if (work / "result.zip").exists():
            return generate(job, work)
        usage = text_debug.PersistentUsage(work, job["payload"], text_debug.pipeline.load_config())
        key = usage.begin_request()
        error = None
        try:
            return generate(job, work)
        except RuntimeError as exc:
            error = str(exc)
            raise
        finally:
            trace = [{"type": "llm_generation", "cache_hit": False, "usage":
                {} if control["usage_missing"] else {"prompt_tokens": 31, "completion_tokens": 7}}]
            usage.finish_request(key, trace, 0, error)

    monkeypatch.setattr(text_debug.m3_pipeline, "generate_job", measured)
    # Production validation has its own real-schema tests. These fixtures target
    # the runner's commit boundary and use the existing small legacy narrative.
    monkeypatch.setattr(text_debug, "_validate_result", lambda envelope, *_args, **_kwargs: envelope["result"])
    return calls, control


def test_all_chapters_use_only_narrative_and_exact_predecessor(runtime, tmp_path):
    calls, _ = runtime
    source = snapshot()
    saved_source = copy.deepcopy(source)
    report = text_debug.run_text_debug(source, tmp_path, workflow="legacy", context_size=32768)
    assert report["status"] == "text_verified"
    assert report["adopted_chapter_count"] == 3
    assert report["public_build_created"] is False
    assert report["development_stage"] == "poc"
    assert report["automated_checks"] == "passed"
    assert report["quality_acceptance"] == "not_evaluated"
    assert report["human_review"] == "not_performed"
    assert source == saved_source
    assert [call["kind"] for call in calls] == ["m3_narrative"] * 3
    assert [call["payload"]["seed"] for call in calls] == [1, 2, 3]
    for number, call in enumerate(calls, 1):
        payload = call["payload"]
        assert payload["chapter_number"] == number
        assert payload["profile"]["context_size"] == 32768
        assert payload["execution_mode"] == "text_only"
        if number > 1:
            previous = text_debug._read(tmp_path / "chapters" / f"chapter-{number - 1:03d}.json")
            assert payload["previous_narrative"] == previous["envelope"]["result"]
            assert payload["previous_state_hash"] == previous["end_state_hash"]
            assert payload["previous_narrative_artifact_id"] == previous["artifact_id"]
    html = (tmp_path / "story.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in html and "&lt;img src=x&gt;" in html
    assert "<script>" not in html and "<img " not in html
    assert "品質受入は未実施" in html
    story = (tmp_path / "story.md").read_text(encoding="utf-8")
    assert "第3章" in story and "未採用の途中台詞" not in story
    assert "秘密の刻印" not in story
    assert "秘密の刻印" in (tmp_path / "author-notes.md").read_text(encoding="utf-8")
    assert report["attempts"][0]["metrics"]["prompt_tokens_new"] == 31


def test_limit_does_not_rewrite_approved_length_and_resume_can_extend(runtime, tmp_path):
    calls, _ = runtime
    first = text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    committed_bytes = (tmp_path / "chapters/chapter-001.json").read_bytes()
    assert first["status"] == "chapter_limit_reached"
    assert first["approved_chapter_count"] == 3
    assert calls[0]["payload"]["approval_snapshot"]["world"]["result"]["chapterCount"] == 3
    result = text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", resume=True)
    assert result["status"] == "text_verified"
    assert [call["payload"]["chapter_number"] for call in calls] == [1, 2, 3]
    assert (tmp_path / "chapters/chapter-001.json").read_bytes() == committed_bytes
    text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", resume=True)
    assert len(calls) == 3


def test_failed_chapter_retains_prefix_and_partial_output_then_resumes(runtime, tmp_path):
    calls, control = runtime
    control["fail_chapter"] = 2
    with pytest.raises(RuntimeError, match="scene review"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy")
    report = text_debug._read(tmp_path / "report.json")
    assert report["status"] == "failed" and report["adopted_chapter_count"] == 1
    assert not (tmp_path / "chapters/chapter-002.json").exists()
    assert "未採用の途中台詞" in (tmp_path / "jobs/chapter-002/draft.md").read_text(encoding="utf-8")
    assert "未採用の途中台詞" not in (tmp_path / "story.md").read_text(encoding="utf-8")
    assert report["attempts"][-1]["metrics"]["prompt_tokens_new"] == 31
    control["fail_chapter"] = None
    result = text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", resume=True)
    assert result["status"] == "text_verified" and "error" not in result
    assert [call["payload"]["chapter_number"] for call in calls] == [1, 2, 2, 3]
    assert calls[1] == calls[2]
    assert result["attempts"][2]["metrics"]["new_request_cache_records"] == 0


def test_causal_story_budget_spans_chapters_and_resume_instead_of_resetting(metered_runtime, tmp_path):
    calls, _ = metered_runtime
    source = {"approval_snapshot": snapshot(), "workflow_limits": {"max_story_tokens": 76}}
    first = text_debug.run_text_debug(source, tmp_path, chapter_limit=1)
    assert first["resource_budget"]["story"]["prompt_tokens"] == 31
    assert first["resource_budget"]["story"]["completion_tokens"] == 7
    with pytest.raises(text_debug.ResourceBudgetError, match="max_story_tokens"):
        text_debug.run_text_debug(source, tmp_path, resume=True)
    report = text_debug._read(tmp_path / "report.json")
    assert report["adopted_chapter_count"] == 2
    assert report["resource_budget"]["story"]["prompt_tokens"] == 62
    assert report["resource_budget"]["story"]["completion_tokens"] == 14
    assert [job["payload"]["chapter_number"] for job in calls] == [1, 2]
    assert calls[1]["payload"]["workflow_limits"] == {"max_story_tokens": 76}
    assert not (tmp_path / "chapters/chapter-003.json").exists()


def test_failed_measured_generation_counts_against_persistent_story_budget(metered_runtime, tmp_path):
    calls, control = metered_runtime
    control["fail_chapter"] = 2
    source = {"approval_snapshot": snapshot(), "workflow_limits": {"max_story_tokens": 100}}
    with pytest.raises(RuntimeError, match="scene review"):
        text_debug.run_text_debug(source, tmp_path)
    before = text_debug._read(tmp_path / "report.json")["resource_budget"]["story"]
    assert before["prompt_tokens"] + before["completion_tokens"] == 76
    control["fail_chapter"] = None
    with pytest.raises(text_debug.ResourceBudgetError, match="max_story_tokens"):
        text_debug.run_text_debug(source, tmp_path, resume=True)
    report = text_debug._read(tmp_path / "report.json")
    assert report["adopted_chapter_count"] == 1
    assert report["resource_budget"]["story"]["prompt_tokens"] == 93
    assert report["resource_budget"]["story"]["completion_tokens"] == 21
    assert [job["payload"]["chapter_number"] for job in calls] == [1, 2, 2]


def test_successful_bundle_without_usage_cannot_be_adopted_or_resumed_as_zero(metered_runtime, tmp_path):
    _, control = metered_runtime
    control["usage_missing"] = True
    with pytest.raises(text_debug.ResourceBudgetError, match="unmeasured"):
        text_debug.run_text_debug(snapshot(), tmp_path)
    report = text_debug._read(tmp_path / "report.json")
    assert report["adopted_chapter_count"] == 0
    assert report["resource_budget"]["story"]["unmeasured_requests"] == ["request-1"]
    assert report["resource_budget"]["story"]["elapsed_seconds"] >= 0
    assert report["resource_budget"]["story"]["unfinished_attempts"] == []
    control["usage_missing"] = False
    with pytest.raises(text_debug.ResourceBudgetError, match="unmeasured"):
        text_debug.run_text_debug(snapshot(), tmp_path, resume=True)


def test_reusing_completed_bundle_does_not_charge_its_tokens_twice(metered_runtime, tmp_path):
    _, control = metered_runtime
    control["crash_after_bundle"] = True
    with pytest.raises(RuntimeError, match="Crash after worker cache"):
        text_debug.run_text_debug(snapshot(), tmp_path, chapter_limit=1)
    before = text_debug._read(tmp_path / "report.json")["resource_budget"]["story"]
    control["crash_after_bundle"] = False
    result = text_debug.run_text_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    after = result["resource_budget"]["story"]
    assert after["prompt_tokens"] == before["prompt_tokens"] == 31
    assert after["completion_tokens"] == before["completion_tokens"] == 7
    assert after["elapsed_seconds"] >= before["elapsed_seconds"]
    assert result["attempts"][-1]["metrics"]["result_cache_hit"] is True


def test_workflow_limits_are_immutable_experiment_inputs(runtime, tmp_path):
    source = {"approval_snapshot": snapshot(), "workflow_limits": {"max_tokens": 1000}}
    text_debug.run_text_debug(source, tmp_path, workflow="legacy", chapter_limit=1)
    source["workflow_limits"]["max_tokens"] = 2000
    with pytest.raises(ValueError, match="changed"):
        text_debug.run_text_debug(source, tmp_path, workflow="legacy", resume=True)


@pytest.mark.parametrize("change", ["seed", "context_size", "workflow", "snapshot", "config"])
def test_resume_rejects_changed_experiment_identity(runtime, tmp_path, monkeypatch, change):
    calls, _ = runtime
    text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    value, kwargs = snapshot(), {"workflow": "legacy", "resume": True}
    if change in {"seed", "context_size", "workflow"}:
        kwargs[change] = {"seed": 2, "context_size": 32768, "workflow": "causal"}[change]
    elif change == "snapshot":
        value["world"]["result"]["title"] = "違う作品"
    else:
        monkeypatch.setattr(text_debug.pipeline, "load_config", lambda: {"changed": True})
    with pytest.raises(ValueError, match="changed"):
        text_debug.run_text_debug(value, tmp_path, **kwargs)
    assert len(calls) == 1


def test_resume_rejects_tampered_adoption_before_generation(runtime, tmp_path):
    calls, _ = runtime
    text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    path = tmp_path / "chapters/chapter-001.json"
    commit = text_debug._read(path)
    commit["envelope"]["result"]["end_state"]["summary"] = "変わった本文実績"
    text_debug.write_json(path, commit)
    with pytest.raises(ValueError, match="changed"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", resume=True)
    assert len(calls) == 1


def test_protocol_change_requires_new_experiment(runtime, tmp_path, monkeypatch):
    calls, _ = runtime
    text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    original = text_debug.generator_protocol("legacy")
    monkeypatch.setattr(text_debug, "generator_protocol", lambda workflow: {
        **original, "implementation": original["implementation"] + 1})
    with pytest.raises(ValueError, match="changed"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", resume=True)
    assert len(calls) == 1


@pytest.mark.parametrize("field", ["seed", "profile", "input_sha256", "generator_protocol", "prompt_version"])
def test_provenance_must_match_actual_experiment_input(runtime, tmp_path, field):
    _, control = runtime
    control["bad_provenance"] = field
    with pytest.raises(ValueError, match="provenance"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    assert not list((tmp_path / "chapters").glob("*.json"))


def test_failure_report_uses_complete_llm_trace_with_release_timing(runtime, tmp_path):
    _, control = runtime
    control.update(fail_chapter=1, timings=True)
    with pytest.raises(RuntimeError, match="scene review"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy")
    report = text_debug._read(tmp_path / "report.json")
    metric = report["attempts"][0]["metrics"]
    assert metric["trace_origin"] == "current_file:llm-v6/llm-metrics.json"
    assert metric["model_load_seconds_this_run"] == 2.2
    assert metric["model_release_seconds_this_run"] == 0.4
    assert metric["timing_trace_this_run"][-1]["type"] == "model_release"
    assert report["automated_checks"] == "failed"
    assert report["quality_acceptance"] == "not_evaluated"


def test_reused_result_retains_original_trace_without_counting_load_twice(runtime, tmp_path):
    _, control = runtime
    control.update(timings=True, crash_after_bundle=True)
    with pytest.raises(RuntimeError, match="Crash after worker"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    control["crash_after_bundle"] = False
    report = text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1,
                                       resume=True)
    metric = report["attempts"][-1]["metrics"]
    assert metric["result_cache_hit"]
    assert metric["trace_origin"] == "saved_result"
    assert metric["model_load_seconds_this_run"] == 0
    assert metric["model_release_seconds_this_run"] == 0
    assert metric["timing_trace_this_run"] == []
    assert metric["timing_trace"][0]["elapsed_seconds"] == 2.2
    assert report["metrics"]["model_load_seconds_this_run"] == 2.2


@pytest.mark.parametrize("problem", ["wrong_lineage", "media", "legacy_fallback"])
def test_rejects_bad_result_without_adopting(runtime, tmp_path, problem):
    _, control = runtime
    if problem != "legacy_fallback":
        control[problem] = True
    workflow = "causal" if problem == "legacy_fallback" else "legacy"
    with pytest.raises(ValueError):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow=workflow)
    assert not list((tmp_path / "chapters").glob("*.json"))
    assert text_debug._read(tmp_path / "report.json")["status"] == "failed"


def test_extract_job_preserves_settings_profile_seed_and_does_not_read_media(runtime, tmp_path):
    calls, _ = runtime
    source = snapshot()
    source["characters"][0]["image"] = {"path": "missing.png"}
    source["characters"][0]["voice"] = {"path": "missing.wav"}
    job = {"kind": "m3_narrative", "payload": {"approval_snapshot": source,
           "profile": {"temperature": 0.4}, "seed": 82, "chapter_number": 3,
           "previous_narrative": {"not": "used as an experimental predecessor"}}}
    text_debug.run_text_debug(job, tmp_path, workflow="legacy", chapter_limit=1)
    payload = calls[0]["payload"]
    assert payload["profile"] == {"temperature": 0.4}
    assert payload["seed"] == 82 and payload["chapter_number"] == 1
    assert "previous_narrative" not in payload
    assert payload["approval_snapshot"] == source


def test_purpose_profiles_are_preserved_for_every_chapter_and_recorded(runtime, tmp_path):
    calls, _ = runtime
    profiles = {"scene_planning": {"temperature": 0.25, "max_tokens": 2048},
                "story_extraction": {"temperature": 0.1},
                "information_extraction": {"temperature": 0.15, "max_tokens": 1024}}
    job = {"kind": "m3_narrative", "payload": {"approval_snapshot": snapshot(),
           "profile": {"temperature": 0.6}, "profiles": profiles, "seed": 12}}
    original = copy.deepcopy(job)
    # The public extraction API keeps its three-value return contract.
    assert len(text_debug.extract_input(job)) == 3
    text_debug.run_text_debug(job, tmp_path, workflow="legacy")
    assert job == original
    assert text_debug._read(tmp_path / "experiment.json")["profiles"] == profiles
    for call in calls:
        assert call["payload"]["profiles"] == profiles
        assert call["payload"]["profile"] == {"temperature": 0.6}
    # Payload ownership is separate from the manifest and other chapter jobs.
    assert calls[0]["payload"]["profiles"] is not calls[1]["payload"]["profiles"]


@pytest.mark.parametrize("change", ["modify", "add", "remove"])
def test_resume_rejects_changed_purpose_profiles(runtime, tmp_path, change):
    calls, _ = runtime
    job = {"kind": "m3_narrative", "payload": {"approval_snapshot": snapshot(),
           "profiles": {"continuity_review": {"temperature": 0.1}}}}
    text_debug.run_text_debug(job, tmp_path, workflow="legacy", chapter_limit=1)
    if change == "modify":
        job["payload"]["profiles"]["continuity_review"]["temperature"] = 0.3
    elif change == "add":
        job["payload"]["profiles"]["story_blueprint"] = {"max_tokens": 4096}
    else:
        job["payload"].pop("profiles")
    with pytest.raises(ValueError, match="changed"):
        text_debug.run_text_debug(job, tmp_path, workflow="legacy", resume=True)
    assert len(calls) == 1


def test_empty_purpose_profiles_keep_default_payload_and_existing_resume(runtime, tmp_path):
    calls, _ = runtime
    text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    manifest_path = tmp_path / "experiment.json"
    manifest = text_debug._read(manifest_path)
    assert manifest["profiles"] == {}
    # Earlier default-profile manifests omitted this field; semantics are equal.
    manifest.pop("profiles")
    text_debug.write_json(manifest_path, manifest)
    job = {"kind": "m3_narrative", "payload": {"approval_snapshot": snapshot(), "profiles": {}}}
    text_debug.run_text_debug(job, tmp_path, workflow="legacy", resume=True)
    assert len(calls) == 3
    assert all("profiles" not in call["payload"] for call in calls)


@pytest.mark.parametrize("profiles", [None, [], {"scene_planning": 1}, {"": {}}])
def test_malformed_purpose_profiles_are_rejected_before_generation(runtime, tmp_path, profiles):
    calls, _ = runtime
    job = {"kind": "m3_narrative", "payload": {"approval_snapshot": snapshot(), "profiles": profiles}}
    with pytest.raises(TypeError, match="Purpose profiles"):
        text_debug.run_text_debug(job, tmp_path, workflow="legacy")
    assert calls == []


def test_tampered_empty_manifest_cannot_add_unfingerprinted_profile(runtime, tmp_path):
    calls, _ = runtime
    text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    manifest_path = tmp_path / "experiment.json"
    manifest = text_debug._read(manifest_path)
    manifest["profiles"] = {"scene_planning": {"temperature": 1.9}}
    text_debug.write_json(manifest_path, manifest)
    with pytest.raises(ValueError, match="changed"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", resume=True)
    assert len(calls) == 1


def test_missing_relationships_rejected_before_generation(runtime, tmp_path):
    calls, _ = runtime
    value = snapshot()
    value.pop("relationships")
    with pytest.raises(ValueError, match="pairs"):
        text_debug.run_text_debug(value, tmp_path, workflow="legacy")
    assert calls == []


def test_existing_output_and_concurrent_owner_are_not_overwritten(runtime, tmp_path):
    with text_debug._output_lock(tmp_path), pytest.raises(RuntimeError, match="Another"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy")
    marker = tmp_path / "existing.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy")
    assert marker.read_text(encoding="utf-8") == "keep"


def test_requires_resume_even_after_success(runtime, tmp_path):
    text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    with pytest.raises(ValueError, match="--resume"):
        text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy")


def test_real_worker_pipeline_is_used_with_fake_llm_only(monkeypatch, tmp_path):
    """Exercise job wrapping, standard prompts and content validation without a server."""
    from tests.unit.test_m3_narrative import FakeLLM, narrative_responses
    from tests.unit.test_m4_narrative import continuity_review, response

    result, _ = chapter_fixture()
    responses = narrative_responses(result)
    responses.insert(2, response(result["start_state"]))
    responses.extend([response(result["end_state"]), response(continuity_review())])
    captured = []

    class OfflineLLM(FakeLLM):
        def __init__(self, root, config, payload, output, **kwargs):
            super().__init__(responses)
            self.payload = payload
            self.base = {"model": {"revision": "fixture", "publisher_sha256": "0" * 64}}
            self.retry_seed_segments = []
            captured.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    config = {"llm": {"model_id": "fake", "context_size": 16384},
              "gpu_lock_timeout_seconds": 1, "max_zip_bytes": 1_000_000}
    monkeypatch.setattr(text_debug.pipeline, "load_config", lambda: config)
    monkeypatch.setattr(text_debug.m3_pipeline, "LocalLLM", OfflineLLM)
    monkeypatch.setattr(text_debug.m3_pipeline, "gpu_lock", lambda *args: nullcontext())
    monkeypatch.setattr(text_debug.m3_pipeline, "generate_background",
                        lambda *args: pytest.fail("No image runtime in text-only execution"))
    monkeypatch.setattr(text_debug.m3_pipeline, "_media_job",
                        lambda *args: pytest.fail("No voice/image runtime in text-only execution"))
    report = text_debug.run_text_debug(snapshot(), tmp_path, workflow="legacy", chapter_limit=1)
    assert report["status"] == "chapter_limit_reached"
    assert len(captured) == 1
    assert [call[0][0] for call in captured[0].calls] == [
        "story_outline", "supporting_character", "story_state", "scene_plan", "scene-text-s1",
        "staging", "quality_review", "story_state", "continuity_review"]
    assert (tmp_path / "jobs/chapter-001/job-request.json").exists()
    assert (tmp_path / "jobs/chapter-001/result.zip").exists()
