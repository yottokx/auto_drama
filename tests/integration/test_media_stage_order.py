"""Chapter media stages stay contiguous across workers, failures and old queues."""

import copy

import pytest
from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from tests.integration.test_m2 import claim, report_voice_inventory
from tests.integration.test_m3 import (
    approved,
    complete,
    finish,
    narrative,
    output,
    pin_legacy_production,
    production,
)
from tests.integration.test_project_history import history, restore


def start_three_supporting_characters(client):
    project, worker = approved(client)
    job = claim(client, worker)
    result = narrative(job["payload"]["approval_snapshot"])
    original = result["supporting_characters"][0]
    for number in (2, 3):
        result["supporting_characters"].append(
            {**copy.deepcopy(original), "id": f"support-{number}", "name": f"同行者{number}"}
        )
    for number, scene in enumerate(result["scenes"], 2):
        scene["plan"]["character_ids"].append(f"support-{number}")
        scene["directions"].append({
            **scene["directions"][0],
            "id": f"enter-support-{number}", "utterance_id": scene["utterances"][0]["id"],
            "kind": "enter", "character_id": f"support-{number}", "position": "center", "duration_ms": 0,
        })
    response = complete(client, worker, job, output(job, result=result))
    assert response.status_code == 200, response.text
    return project, worker


def register_worker(client, kinds):
    worker = client.post("/api/workers", json={"name": "stage fixture", "capabilities": [*kinds, "tts_download"]}).json()["id"]
    if "m3_voice" in kinds:
        report_voice_inventory(client, worker)
    return worker


def test_three_portraits_then_backgrounds_then_references_then_dialogue(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = start_three_supporting_characters(client)
        # Emulate an already queued production with later-stage jobs ahead in
        # priority. The barrier applies regardless of insertion or job priority.
        with client.app.state.coordinator.db.transaction() as connection:
            connection.execute("UPDATE job SET priority=100 WHERE kind IN ('m3_background','m3_voice','m3_voice_clone')")
        sequence = []
        while job := claim(client, worker):
            if job["payload"]["chapter_number"] != 1:
                break
            sequence.append(job["kind"])
            response = complete(client, worker, job)
            assert response.status_code == 200, response.text
        assert sequence == ["m3_image"] * 3 + ["m3_background"] * 2 + ["m3_voice"] * 3 + ["m3_voice_clone"] * 8
        assert production(client, project)["chapters"][0]["status"] == "published"


def test_later_stage_waits_for_all_portraits_and_retry(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        _, worker = start_three_supporting_characters(client)
        images = register_worker(client, ["m3_image"])
        backgrounds = register_worker(client, ["m3_background"])
        voices = register_worker(client, ["m3_voice", "m3_voice_clone"])
        first = claim(client, worker)
        second = claim(client, images)
        assert first["kind"] == second["kind"] == "m3_image"
        assert claim(client, backgrounds) is None
        assert claim(client, voices) is None
        assert complete(client, worker, first).status_code == 200
        # A failed job is not a completed stage. Other kinds cannot skip it.
        response = client.post(f"/api/jobs/{second['id']}/fail", json={
            "worker_id": images, "lease_id": second["lease_id"], "error": "fixture retry",
        })
        assert response.status_code == 200, response.text
        assert claim(client, backgrounds) is None
        assert claim(client, voices) is None
        remaining = []
        while job := claim(client, images):
            remaining.append(job["id"])
            assert complete(client, images, job).status_code == 200
        assert second["id"] in remaining and len(remaining) == 2
        background = claim(client, backgrounds)
        assert background["kind"] == "m3_background"
        assert claim(client, voices) is None


@pytest.mark.parametrize("failure_kind", ["m3_image", "m3_background", "m3_voice"])
def test_legacy_shared_stage_failure_settles_history_and_can_retry_or_restore(tmp_path, failure_kind):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        pin_legacy_production(client, project)
        writer = register_worker(client, ["m3_narrative"])
        # Old productions could write all chapters before chapter one's media
        # finished. Later chapters wait on shared requirements with no own job.
        for number in (1, 2, 3):
            narrative_job = claim(client, writer)
            assert narrative_job["kind"] == "m3_narrative"
            assert narrative_job["payload"]["chapter_number"] == number
            response = complete(client, writer, narrative_job)
            assert response.status_code == 200, response.text
        while media := claim(client, worker):
            if media["kind"] == failure_kind:
                break
            assert complete(client, worker, media).status_code == 200
        else:
            pytest.fail(f"Expected the first chapter's {failure_kind} job")
        failed_id = media["id"]
        for _ in range(media["max_attempts"]):
            response = client.post(f"/api/jobs/{media['id']}/fail", json={
                "worker_id": worker, "lease_id": media["lease_id"], "error": "shared material failed",
            })
            assert response.status_code == 200, response.text
            if response.json()["status"] == "failed":
                break
            media = claim(client, worker)
            assert media["id"] == failed_id
        else:
            pytest.fail("The fixture must reach terminal failure")
        finish(client, worker)  # Independent work in the same stage can settle.
        stalled = production(client, project)
        assert stalled["status"] == "failed"
        assert all(chapter["narrative_artifact_id"] for chapter in stalled["chapters"])
        for child in stalled["chapters"][1:]:
            shared = [row for row in child["requirements"] if row["kind"] == failure_kind
                      and row["artifact_id"] is None and row["job_id"] is None]
            assert shared, "Later chapters must be waiting on the failed shared source without a local job"
            assert any(job["kind"] == "m3_voice_clone" and job["status"] == "pending" for job in child["jobs"])
        assert claim(client, worker) is None, "Blocked queued jobs must not bypass the shared material failure"
        failed_history = history(client, project)
        assert not failed_history["busy"]
        assert failed_history["pending_operation"] is None

        response = client.post(f"/api/jobs/{failed_id}/retry")
        assert response.status_code == 200, response.text
        assert history(client, project)["busy"]
        finish(client, worker)
        assert production(client, project)["status"] == "published"

        restore(client, project, failed_history["current_revision_id"])
        restored = production(client, project)
        assert restored["history_frozen"]
        assert not history(client, project)["busy"]
        assert next(job for job in restored["jobs"] if job["id"] == failed_id)["status"] == "failed"
        assert claim(client, worker) is None
