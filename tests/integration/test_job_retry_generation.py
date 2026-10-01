"""Explicit retries are distinct from lease recovery and production interruption."""

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from tests.integration.test_coordinator import claim, register, sample
from tests.integration.test_m4 import approve


def fail(client, job, worker):
    response = client.post(f"/api/jobs/{job['id']}/fail", json={
        "worker_id": worker, "lease_id": job["lease_id"], "error": "fixture failure",
    })
    assert response.status_code == 200, response.text


def test_only_explicit_retry_increments_generation_and_preserves_inputs(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        original = sample(client)["job"]
        assert original["retry_generation"] == 0
        worker = register(client)
        for _ in range(original["max_attempts"]):
            job = claim(client, worker)
            assert job["retry_generation"] == 0
            fail(client, job, worker)
        assert claim(client, worker) is None

    # The counter is durable, independent of process and worker identity.
    with TestClient(create_app(tmp_path)) as client:
        worker = register(client)
        for generation in (1, 2):
            response = client.post(f"/api/jobs/{original['id']}/retry")
            assert response.status_code == 200, response.text
            queued = response.json()
            assert queued["retry_generation"] == generation
            assert queued["payload"] == original["payload"]
            assert queued["settings_snapshot"] == original["settings_snapshot"]
            assert client.post(f"/api/jobs/{original['id']}/retry").status_code == 409
            job = claim(client, worker)
            assert job["retry_generation"] == generation
            fail(client, job, worker)
            assert claim(client, worker) is None
        saved = client.get(f"/api/jobs/{original['id']}").json()["job"]
        assert saved["retry_generation"] == 2


def test_stop_resume_compensates_attempt_without_granting_retry_generation(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approve(client)
        original = claim(client, worker)
        assert original["kind"] == "m3_narrative"
        assert original["retry_generation"] == 0
        response = client.post(f"/api/m3/projects/{project}/stop", json={"mode": "immediate"})
        assert response.status_code == 200, response.text
        stopped = client.get(f"/api/jobs/{original['id']}").json()["job"]
        assert stopped["max_attempts"] == original["max_attempts"] + 1
        assert stopped["retry_generation"] == 0
        assert client.post(f"/api/m3/projects/{project}/resume").status_code == 200
        resumed = claim(client, worker)
        assert resumed["id"] == original["id"]
        assert resumed["retry_generation"] == 0
        assert resumed["payload"] == original["payload"]


def test_narrative_retry_preserves_story_model_profile_and_seed(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        _, worker = approve(client)
        original = claim(client, worker)
        assert original["kind"] == "m3_narrative"
        fail(client, original, worker)
        for _ in range(original["max_attempts"] - 1):
            automatic = claim(client, worker)
            assert automatic["id"] == original["id"]
            assert automatic["retry_generation"] == 0
            fail(client, automatic, worker)
        response = client.post(f"/api/jobs/{original['id']}/retry")
        assert response.status_code == 200, response.text
        retried = claim(client, worker)
        assert retried["id"] == original["id"]
        assert retried["retry_generation"] == 1
        assert retried["payload"] == original["payload"]
        assert retried["settings_snapshot"] == original["settings_snapshot"]


def test_migration_gives_existing_failed_jobs_zero_retry_generation(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        original = sample(client)["job"]
        worker = register(client)
        for _ in range(original["max_attempts"]):
            fail(client, claim(client, worker), worker)
        with client.app.state.coordinator.db.transaction(migration=True) as connection:
            connection.execute("ALTER TABLE job DROP COLUMN retry_generation")
            connection.execute("DELETE FROM schema_migration WHERE version=8")

    with TestClient(create_app(tmp_path)) as client:
        migrated = client.get(f"/api/jobs/{original['id']}").json()["job"]
        assert migrated["status"] == "failed"
        assert migrated["retry_generation"] == 0
        assert migrated["payload"] == original["payload"]
        assert migrated["settings_snapshot"] == original["settings_snapshot"]
        response = client.post(f"/api/jobs/{original['id']}/retry")
        assert response.status_code == 200, response.text
        assert response.json()["retry_generation"] == 1
