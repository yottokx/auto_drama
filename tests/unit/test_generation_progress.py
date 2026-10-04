"""Real model boundaries emit progress without changing cache or inference identity."""
import copy
import json
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from packages.contracts.job_progress import JobProgress
from packages.contracts.planning import validate_plan_content
from services.worker.client import WorkerClient
from services.worker.generation import script_continuation
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.causal_runtime import digest
from services.worker.generation.draft_story import DraftExecutionError
from services.worker.generation.progress import progress_scope, report_progress
from tests.unit.test_common_plan_worker import (
    generate,
    plan_job,
    semantic_plan,
    supporting_cast,
)
from tests.unit.test_script_continuation_run import (
    FIRST,
    allocation,
    chapter_plan,
    outline,
    read_json,
    staging,
    story_chain_draft,
)
from tests.unit.test_script_continuation_run import runtime as _runtime
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import job

runtime = _runtime
app_runtime = _app_runtime


@pytest.mark.parametrize("count", [1, 2, 3, 9])
def test_planning_reports_only_actual_accepted_stages_and_reuses_bundle(app_runtime, tmp_path, count):
    calls, control = app_runtime
    planned = semantic_plan(count)["plot"]
    if count == 3:
        replies = [story_chain_draft(), allocation()]
        expected = ["cast_plan", "story_core", "plot", "validation"]
    elif count <= 8:
        replies = [planned]
        expected = ["cast_plan", "plot_plan", "validation"]
    else:
        replies = [planned["core"], {"chapters": planned["chapters"][:8]}, {"chapters": planned["chapters"][8:]}]
        expected = ["cast_plan", "story_core", "plot", "plot", "validation"]
    control["responses"] = [supporting_cast(), *replies]
    updates = []
    request = plan_job(count)
    with progress_scope(updates.append):
        generate(request, tmp_path)
    assert [row["stage"] for row in updates[-1]["steps"]] == expected
    assert all(row["status"] == "completed" for row in updates[-1]["steps"])
    assert [event["sequence"] for event in updates] == list(range(1, len(updates) + 1))
    assert all(JobProgress.model_validate(value).phase == "planning" for value in updates)
    with progress_scope(updates.append):
        generate(request, tmp_path)
    assert len(calls) == 1 + len(replies)  # Completed ZIP adds no model or fake progress.
    if count == 9:
        batches = [row for row in updates[-1]["steps"] if row["stage"] == "plot"]
        assert [(row["completed"], row["total"]) for row in batches] == [(8, 9), (9, 9)]


def test_chapter_scene_counts_and_validated_staging_are_actual(app_runtime, tmp_path):
    calls, control = app_runtime
    request = job()
    content = semantic_plan()
    request["payload"]["approved_plan"] = {"content": content, "approval_id": "accepted-1", "sha256": digest(content)}
    plan = chapter_plan("s1")
    second = copy.deepcopy(plan["scenes"][0])
    second["id"] = "s2"
    plan["scenes"].append(second)
    control["responses"] = [plan, FIRST, staging(FIRST), FIRST, staging(FIRST, "s2")]
    updates = []
    with progress_scope(updates.append):
        result = generate(request, tmp_path)
    script = [row for row in updates[-1]["steps"] if row["stage"] == "script"]
    assert [(row["scene_number"], row["completed"], row["total"]) for row in script] == [(1, 1, 2), (2, 2, 2)]
    assert all(row["status"] == "completed" for row in updates[-1]["steps"])
    metadata = {"chapter_number": 1, "scene_count": 2, "supporting_character_count": 0}
    planned = [event for event in updates if any(row["stage"] == "chapter_scene_plan"
                and row["status"] == "completed" for row in event["steps"])]
    assert planned[0]["chapter_plan"] == metadata
    assert not any(row["stage"] == "script" for row in planned[0]["steps"])
    assert all(event["chapter_plan"] == metadata for event in planned)
    assert read_json(tmp_path / "script/draft-state.json")["progress"]["chapter_plan"] == metadata
    assert "cast_plan" not in {row["stage"] for row in updates[-1]["steps"]}
    assert "progress" not in result["provenance"]["script_checkpoint"]["state"]
    assert len(calls) == 5


def test_chapter_layout_counts_used_existing_and_new_supporting_characters_once(app_runtime, tmp_path):
    _calls, control = app_runtime
    request = job()
    content = semantic_plan()
    content["cast_plan"] = supporting_cast()
    unused = copy.deepcopy(content["cast_plan"]["supporting_characters"][0])
    unused.update(id="unused", name="未登場の友人")
    content["cast_plan"]["supporting_characters"].append(unused)
    content["cast_plan"]["everyday_context"].append({
        **content["cast_plan"]["everyday_context"][0], "character_id": "unused"})
    content = validate_plan_content(content, request["payload"]["approval_snapshot"]).model_dump(mode="json")
    request["payload"]["approved_plan"] = {"content": content, "approval_id": "accepted-1", "sha256": digest(content)}
    plan = chapter_plan("s1", "s2")
    new = copy.deepcopy(unused)
    new.update(id="newcomer", name="新登場の友人")
    plan["new_characters"] = [new]
    plan["scenes"][0]["character_ids"] = ["aoi", "ren", "helper"]
    plan["scenes"][1]["character_ids"] = ["aoi", "helper", "newcomer"]
    control["responses"] = [plan, GenerationCancelled("stop before body")]
    updates = []
    with progress_scope(updates.append), pytest.raises(GenerationCancelled):
        generate(request, tmp_path)
    assert updates[-1]["chapter_plan"] == {"chapter_number": 1, "scene_count": 2,
                                           "supporting_character_count": 2}


def test_chapter_layout_metadata_survives_cancel_and_cached_plan_resume(app_runtime, tmp_path):
    calls, control = app_runtime
    request = job()
    content = semantic_plan()
    request["payload"]["approved_plan"] = {"content": content, "approval_id": "accepted-1", "sha256": digest(content)}
    control["responses"] = [chapter_plan("s1"), GenerationCancelled("stop before body")]
    with progress_scope(lambda _value: None), pytest.raises(GenerationCancelled):
        generate(request, tmp_path)
    metadata = read_json(tmp_path / "script/draft-state.json")["progress"]["chapter_plan"]
    control["responses"] = [FIRST, staging(FIRST)]
    resumed = []
    with progress_scope(resumed.append):
        result = generate(request, tmp_path)
    assert resumed[0]["chapter_plan"] == metadata
    assert all(event["chapter_plan"] == metadata for event in resumed)
    assert [call["purpose"] for call in calls].count("script-plan") == 1
    assert "progress" not in result["provenance"]["script_checkpoint"]["state"]


def test_chapter_layout_completion_waits_for_final_projection(app_runtime, tmp_path, monkeypatch):
    calls, control = app_runtime
    request = job()
    content = semantic_plan()
    request["payload"]["approved_plan"] = {"content": content, "approval_id": "accepted-1", "sha256": digest(content)}
    control["responses"] = [chapter_plan("s1")]

    def reject(_draft, _options):
        raise ValueError("final layout projection failed")

    monkeypatch.setattr(script_continuation, "project_plan", reject)
    updates = []
    with progress_scope(updates.append), pytest.raises(ValueError, match="final layout"):
        generate(request, tmp_path)
    assert all("chapter_plan" not in event for event in updates)
    assert not any(row["status"] == "completed" for event in updates for row in event["steps"])
    assert updates[-1]["steps"][0]["status"] == "failed" and len(calls) == 1


def test_rejected_answer_is_failed_until_an_explicit_retry_accepts_it(app_runtime, tmp_path):
    calls, control = app_runtime
    request = plan_job()
    control["responses"] = [supporting_cast(), "{bad", "{bad"]
    failed = []
    with progress_scope(failed.append), pytest.raises(DraftExecutionError):
        generate(request, tmp_path)
    assert failed[-1]["steps"][-1]["status"] == "failed"
    accepted = []
    control["responses"] = [outline()]
    with progress_scope(accepted.append):
        generate({**request, "retry_generation": 1}, tmp_path)
    assert accepted[0]["steps"][0]["status"] == "completed"  # Accepted cast survives resume.
    assert all(row["status"] == "completed" for row in accepted[-1]["steps"])
    assert len(calls) == 4
    assert read_json(tmp_path / "script/draft-state.json")["progress"]["steps"][-1]["status"] == "completed"


@pytest.mark.parametrize("outcome", ["ok", "disconnected", "stale"])
def test_worker_sends_lease_progress_and_telemetry_outage_does_not_rerun(outcome, tmp_path):
    updates, completed, failed, executed = [], [], [], []
    expiry = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()

    def protocol(request):
        path = request.url.path
        if path == "/api/workers":
            return httpx.Response(200, json={"id": "worker"})
        if path.endswith("/claim"):
            return httpx.Response(200, json={"job": {"id": "progress-job", "kind": "m3_plan", "payload": {},
                "lease_id": "lease", "lease_expires_at": expiry, "attempt": 1}})
        if path.endswith("/heartbeat"):
            return httpx.Response(200, json={"lease_expires_at": expiry})
        if path.endswith("/progress"):
            updates.append(json.loads(request.content))
            if outcome == "disconnected":
                raise httpx.ReadError("progress disconnected", request=request)
            return httpx.Response(409 if outcome == "stale" else 200, json={})
        if path.endswith("/complete"):
            completed.append(request.content)
            return httpx.Response(200, json={})
        if path.endswith("/fail"):
            failed.append(request.content)
            return httpx.Response(200, json={})
        raise AssertionError(path)

    def generation(_job, _path):
        executed.append(True)
        report_progress({"schema_version": 1, "phase": "planning", "current_step": "cast-plan",
            "steps": [{"id": "cast-plan", "stage": "cast_plan", "status": "running"}]})
        return b"one accepted bundle"

    with httpx.Client(base_url="http://coordinator", transport=httpx.MockTransport(protocol)) as client:
        worker = WorkerClient(client, generation_runner=generation, generation_kinds=["m3_plan"], work_dir=tmp_path)
        assert worker.run_once() == ("stale" if outcome == "stale" else "completed")
    assert len(executed) == 1 and not failed
    assert bool(completed) == (outcome != "stale")
    assert updates[0]["worker_id"] == "worker" and updates[0]["lease_id"] == "lease"
    if outcome == "disconnected":
        assert updates[0] == updates[1]  # Retry the exact observation once.
