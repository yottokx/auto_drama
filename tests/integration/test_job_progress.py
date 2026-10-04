"""Job progress is durable, monotonic within a lease, and inactive across retries."""
import copy

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from services.coordinator.service import Coordinator, public
from tests.integration.test_coordinator import bundle, claim, complete, register, sample


def progress(sequence=1, status="running"):
    return {"schema_version": 1, "sequence": sequence, "phase": "planning", "current_step": "cast-plan",
            "steps": [{"id": "cast-plan", "stage": "cast_plan", "status": status}]}


def send(client, job, worker, value):
    return client.post(f"/api/jobs/{job['id']}/progress", json={"worker_id": worker,
                      "lease_id": job["lease_id"], "progress": value})


def test_progress_is_durable_monotonic_and_idempotent(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        sample(client)
        worker = register(client)
        job = claim(client, worker)
        saved = send(client, job, worker, progress())
        assert saved.status_code == 200, saved.text
        assert saved.json()["progress"]["attempt"] == 1
        assert saved.json()["progress"]["active"] is True
        assert send(client, job, worker, progress()).json() == saved.json()
        assert send(client, job, worker, progress(2, "completed")).status_code == 200
        assert send(client, job, worker, progress(1)).status_code == 409
        assert send(client, job, worker, progress(2, "running")).status_code == 409
    with TestClient(create_app(tmp_path)) as client:
        restored = client.get(f"/api/jobs/{job['id']}").json()["job"]["progress"]
        assert restored["sequence"] == 2 and restored["steps"][0]["status"] == "completed"
        assert restored["active"] is True
        assert complete(client, job, worker, bundle(client, job)).status_code == 200
        assert client.get(f"/api/jobs/{job['id']}").json()["job"]["progress"]["active"] is False
        assert send(client, job, worker, progress(3)).status_code == 409


def test_expired_progress_cannot_cross_attempt_identity(tmp_path):
    now = [1800000000.0]
    service = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=service)) as client:
        sample(client)
        worker = register(client)
        old = claim(client, worker)
        assert send(client, old, worker, progress(5)).status_code == 200
        now[0] += 11
        assert send(client, old, worker, progress(6)).status_code == 409
        assert client.get(f"/api/jobs/{old['id']}").json()["job"]["progress"]["active"] is False
        replacement = register(client)
        current = claim(client, replacement)
        assert current["attempt"] == 2 and current["progress"]["active"] is False
        assert send(client, old, worker, progress(9)).status_code == 409
        new = send(client, current, replacement, progress(1))
        assert new.status_code == 200, new.text
        assert new.json()["progress"]["attempt"] == 2 and new.json()["progress"]["active"] is True


def test_failure_and_explicit_retry_preserve_inactive_evidence(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        created = sample(client)
        # Exhaust one permitted attempt, retaining its factual progress record.
        with client.app.state.coordinator.db.transaction() as connection:
            connection.execute("UPDATE job SET max_attempts=1 WHERE id=?", (created["job"]["id"],))
        worker = register(client)
        old = claim(client, worker)
        assert send(client, old, worker, progress(4)).status_code == 200
        failed = client.post(f"/api/jobs/{old['id']}/fail", json={"worker_id": worker,
                            "lease_id": old["lease_id"], "error": "failure"})
        assert failed.json()["status"] == "failed" and failed.json()["progress"]["active"] is False
        retry = client.post(f"/api/jobs/{old['id']}/retry", json={})
        assert retry.status_code == 200 and retry.json()["progress"]["sequence"] == 4
        current = claim(client, worker)
        assert current["progress"]["active"] is False
        assert send(client, current, worker, progress(1)).json()["progress"]["attempt"] == 2
        assert send(client, old, worker, progress(10)).status_code == 409


def test_progress_contract_rejects_invalid_references_and_counts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        sample(client)
        worker = register(client)
        job = claim(client, worker)
        invalid = progress()
        invalid["current_step"] = "absent"
        assert send(client, job, worker, invalid).status_code == 422
        invalid = progress()
        invalid["steps"][0].update(completed=2, total=1)
        assert send(client, job, worker, invalid).status_code == 422
        invalid = progress()
        invalid["sequence"] = True
        assert send(client, job, worker, invalid).status_code == 422
        assert client.get(f"/api/jobs/{job['id']}").json()["job"]["progress"] is None


def test_frozen_progress_is_never_active_or_from_another_attempt():
    record = {"status": "running", "attempt_count": 1, "progress": {**progress(), "attempt": 1},
              "progress_active": False}
    assert public(copy.deepcopy(record))["progress"]["active"] is False
    record.pop("progress_active")
    record["attempt_count"] = 2
    assert public(record)["progress"]["active"] is False


def test_chapter_layout_metadata_is_saved_and_returned_after_restart(tmp_path):
    value = {"schema_version": 1, "sequence": 1, "phase": "chapter", "current_step": None,
             "steps": [{"id": "plan-001", "stage": "chapter_scene_plan", "status": "completed",
                        "chapter_number": 1}],
             "chapter_plan": {"chapter_number": 1, "scene_count": 4, "supporting_character_count": 0}}
    with TestClient(create_app(tmp_path)) as client:
        sample(client)
        worker = register(client)
        job = claim(client, worker)
        saved = send(client, job, worker, value)
        assert saved.status_code == 200, saved.text
        assert saved.json()["progress"]["chapter_plan"] == value["chapter_plan"]
        value.update(sequence=2, current_step="c001-s1-text")
        value["steps"].append({"id": "c001-s1-text", "stage": "script", "status": "running",
                               "chapter_number": 1, "scene_number": 1, "completed": 0, "total": 4})
        assert send(client, job, worker, value).status_code == 200
    with TestClient(create_app(tmp_path)) as client:
        restored = client.get(f"/api/jobs/{job['id']}").json()["job"]["progress"]
        assert restored["chapter_plan"] == value["chapter_plan"]
        assert restored["steps"][0]["status"] == "completed" and restored["current_step"] == "c001-s1-text"
        invalid = copy.deepcopy(value)
        invalid["sequence"] = 3
        invalid["chapter_plan"]["scene_count"] = 0
        assert send(client, job, worker, invalid).status_code == 422
