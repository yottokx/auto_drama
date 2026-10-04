from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts import Script
from packages.tyrano_export import compile_bundle
from services.coordinator.app import create_app
from services.coordinator.service import Coordinator, required
from services.worker.client import WorkerClient


def sample(client):
    response = client.post("/api/projects/demo", json={"title": "灯台の約束"})
    assert response.status_code == 201, response.text
    return response.json()


def register(client):
    return client.post(
        "/api/workers", json={"name": "test", "capabilities": ["tyrano_export"]}
    ).json()["id"]


def claim(client, worker_id):
    return client.post(f"/api/workers/{worker_id}/claim", json={}).json()["job"]


def bundle(client, job):
    raw = client.get(f"/api/artifacts/{job['payload']['script_artifact_id']}/content").content
    script = Script.model_validate_json(raw)
    assets = {
        asset.id: client.get(f"/api/artifacts/{asset.artifact_id}/content").content
        for asset in script.assets
    }
    return compile_bundle(script, assets)


def complete(client, job, worker_id, data):
    return client.post(
        f"/api/jobs/{job['id']}/complete",
        params={"worker_id": worker_id, "lease_id": job["lease_id"]},
        content=data,
        headers={"Content-Type": "application/zip"},
    )


def test_worker_export_survives_restart_and_reproduces(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        created = sample(client)
        project_id = created["project"]["id"]
        job_id = created["job"]["id"]
        # A queued job survives before any worker executes it.
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(f"/api/jobs/{job_id}").json()["job"]["status"] == "pending"
        worker = WorkerClient(client)
        assert worker.run_once() == "completed"
        detail = client.get(f"/api/projects/{project_id}").json()
        job = detail["jobs"][0]
        assert job["status"] == "completed"
        output = client.get(f"/api/artifacts/{job['result_artifact_id']}/content")
        assert output.status_code == 200
        expected = bundle(client, job)
        assert output.content == expected
        assert len([a for a in detail["artifacts"] if a["kind"] == "tyrano_export"]) == 1
        assert all("storage_key" not in a for a in detail["artifacts"])
        with ZipFile(BytesIO(expected)) as archive:
            assert "data/scenario/first.ks" in archive.namelist()
            assert not any(name.startswith("tyrano/") for name in archive.namelist())
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(f"/api/artifacts/{job['result_artifact_id']}/content").content == expected
        assert WorkerClient(client).run_once() == "idle"
        assert client.get(f"/api/jobs/{job_id}").json()["job"]["attempt_count"] == 1


def test_expired_attempt_recovered_stale_rejected_duplicate_idempotent(tmp_path):
    now = [1_800_000_000.0]
    service = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=service)) as client:
        created = sample(client)
        old_worker = register(client)
        old = claim(client, old_worker)
        data = bundle(client, old)
    now[0] += 11
    recovered = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=recovered)) as client:
        detail = client.get(f"/api/jobs/{created['job']['id']}").json()
        assert detail["job"]["status"] == "pending"
        assert detail["attempts"][0]["status"] == "expired"
        worker_id = register(client)
        current = claim(client, worker_id)
        assert current["attempt"] == 2
        assert current["retry_generation"] == old["retry_generation"] == 0
        assert complete(client, old, old_worker, data).status_code == 409
        stale = {"worker_id": old_worker, "lease_id": old["lease_id"]}
        assert client.post(f"/api/jobs/{old['id']}/heartbeat", json=stale).status_code == 409
        assert (
            client.post(f"/api/jobs/{old['id']}/fail", json={**stale, "error": "late"}).status_code
            == 409
        )
        accepted = complete(client, current, worker_id, data)
        assert accepted.status_code == 200, accepted.text
        duplicate = complete(client, current, worker_id, data)
        assert duplicate.json() == accepted.json()
        assert complete(client, current, worker_id, data + b"altered").status_code == 409
        detail = client.get(f"/api/projects/{created['project']['id']}").json()
        assert len([a for a in detail["artifacts"] if a["kind"] == "tyrano_export"]) == 1


def test_unexpired_lease_survives_coordinator_restart(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        sample(client)
        worker = register(client)
        job = claim(client, worker)
        data = bundle(client, job)
    with TestClient(create_app(tmp_path)) as client:
        assert claim(client, register(client)) is None
        assert complete(client, job, worker, data).status_code == 200


def test_partial_and_uncommitted_files_never_publish(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        created = sample(client)
        worker_id = register(client)
        job = claim(client, worker_id)
        data = bundle(client, job)
        assert complete(client, job, worker_id, data[:-12]).status_code == 422
        service = app.state.coordinator
        original = service._register_artifact

        def interrupted(*args, **kwargs):
            if kwargs.get("source_job_id"):
                raise OSError("simulated failure after file write before DB commit")
            return original(*args, **kwargs)

        monkeypatch.setattr(service, "_register_artifact", interrupted)
        assert complete(client, job, worker_id, data).status_code == 503
        (service.store.staging / "interrupted.part").write_bytes(b"incomplete")
        detail = client.get(f"/api/projects/{created['project']['id']}").json()
        assert detail["jobs"][0]["status"] == "running"
        assert not any(a["kind"] == "tyrano_export" for a in detail["artifacts"])
    with TestClient(create_app(tmp_path)) as client:
        detail = client.get(f"/api/projects/{created['project']['id']}").json()
        assert not any(a["kind"] == "tyrano_export" for a in detail["artifacts"])
        assert complete(client, job, worker_id, data).status_code == 200


def test_dependency_waits_and_retry_budget_is_persistent(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        created = sample(client)
        project_id, parent = created["project"]["id"], created["job"]
        child_response = client.post(
            f"/api/projects/{project_id}/jobs",
            json={
                "script_artifact_id": parent["payload"]["script_artifact_id"],
                "depends_on": [parent["id"]],
                "priority": 100,
            },
        )
        assert child_response.status_code == 201
        child = child_response.json()
        worker_id = register(client)
        for expected in range(1, 4):
            current = claim(client, worker_id)
            assert current["id"] == parent["id"]
            assert current["attempt"] == expected
            response = client.post(
                f"/api/jobs/{current['id']}/fail",
                json={
                    "worker_id": worker_id,
                    "lease_id": current["lease_id"],
                    "error": "fixture failure",
                },
            )
            assert response.status_code == 200
        assert claim(client, worker_id) is None
        assert client.get(f"/api/jobs/{parent['id']}").json()["job"]["status"] == "failed"
        assert client.get(f"/api/jobs/{child['id']}").json()["job"]["status"] == "pending"
    with TestClient(create_app(tmp_path)) as client:
        assert client.post(f"/api/jobs/{parent['id']}/retry").status_code == 200
        assert client.post(f"/api/jobs/{parent['id']}/retry").status_code == 409
        worker = WorkerClient(client)
        assert worker.run_once() == "completed"
        assert worker.run_once() == "completed"
        detail = client.get(f"/api/jobs/{parent['id']}").json()
        assert len(detail["attempts"]) == 4
        assert detail["job"]["max_attempts"] == 4


def test_claim_is_atomic_and_capability_filtered(tmp_path):
    service = Coordinator(tmp_path)
    service.initialize()
    created = service.create_demo("同時取得")
    unsupported = service.register_worker("other", ["image_generate"])
    assert service.claim(unsupported["id"]) is None
    workers = [service.register_worker(str(i), ["tyrano_export"]) for i in range(4)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda worker: service.claim(worker["id"]), workers))
    assert [r["id"] for r in results if r] == [created["job"]["id"]]


def test_changed_script_is_new_version_and_cross_project_refs_rejected(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        created = sample(client)
        project_id = created["project"]["id"]
        original = created["job"]["payload"]["script_artifact_id"]
        script = client.get(f"/api/artifacts/{original}/content").json()
        script["utterances"][0]["display_text"] = "改稿した冒頭。"
        saved = client.post(f"/api/projects/{project_id}/scripts", json=script)
        assert saved.status_code == 201, saved.text
        assert saved.json()["version"] == 2
        assert client.get(f"/api/artifacts/{original}/content").json() != script
        other = client.post("/api/projects", json={"title": "別作品"}).json()
        assert client.post(f"/api/projects/{other['id']}/scripts", json=script).status_code == 422
        assert (
            client.post(
                f"/api/projects/{other['id']}/jobs", json={"script_artifact_id": original}
            ).status_code
            == 422
        )


def test_corrupt_completed_file_is_not_served(tmp_path):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        created = sample(client)
        WorkerClient(client).run_once()
        job = client.get(f"/api/jobs/{created['job']['id']}").json()["job"]
        service = app.state.coordinator
        with service.db.transaction() as connection:
            record = required(connection, "artifact", job["result_artifact_id"])
        (service.store.root / record["storage_key"]).write_bytes(b"truncated")
        response = client.get(f"/api/artifacts/{record['id']}/content")
        assert response.status_code == 503
        assert str(tmp_path) not in response.text


def test_result_expiring_during_validation_cannot_be_adopted(tmp_path, monkeypatch):
    import services.coordinator.service as module

    now = [1_800_000_000.0]
    service = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=service)) as client:
        created = sample(client)
        worker_id = register(client)
        job = claim(client, worker_id)
        data = bundle(client, job)
        original = module.validate_bundle

        def slow(*args):
            result = original(*args)
            now[0] += 11
            return result

        monkeypatch.setattr(module, "validate_bundle", slow)
        assert complete(client, job, worker_id, data).status_code == 409
        detail = client.get(f"/api/projects/{created['project']['id']}").json()
        assert detail["jobs"][0]["status"] == "pending"
        assert not any(a["kind"] == "tyrano_export" for a in detail["artifacts"])


def test_migration_idempotence_and_drift_rejected(tmp_path):
    service = Coordinator(tmp_path)
    service.initialize()
    service.initialize()
    with service.db.transaction() as connection:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("SELECT COUNT(*) FROM schema_migration").fetchone()[0] == 13
        connection.execute("UPDATE schema_migration SET sha256='changed'")
    with pytest.raises(RuntimeError, match="has changed"):
        service.initialize()


@pytest.mark.parametrize("body", [b'{"title":"bad\\u0001"}', b'{"title":"bad\\ud800"}'])
def test_invalid_unicode_is_rejected_at_request_boundary(tmp_path, body):
    with TestClient(create_app(tmp_path)) as client:
        response = client.post(
            "/api/projects/demo", content=body, headers={"Content-Type": "application/json"}
        )
        assert response.status_code == 422
        assert client.get("/api/projects").json() == {"projects": []}


def test_migration_line_endings_and_failed_migration_rollback(tmp_path, monkeypatch):
    import services.coordinator.database as module

    migrations = tmp_path / "migrations"
    migrations.mkdir()
    source = (module.MIGRATIONS / "001_initial.sql").read_bytes().replace(b"\r\n", b"\n")
    initial = migrations / "001_initial.sql"
    initial.write_bytes(source)
    monkeypatch.setattr(module, "MIGRATIONS", migrations)
    service = Coordinator(tmp_path / "data")
    service.db.migrate()
    initial.write_bytes(source.replace(b"\n", b"\r\n"))
    service.db.migrate()
    (migrations / "002_broken.sql").write_text(
        "CREATE TABLE must_rollback (id TEXT);\nINVALID SQL;\n", encoding="utf-8"
    )
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        service.db.migrate()
    with service.db.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM schema_migration").fetchone()[0] == 1
        assert (
            connection.execute("SELECT 1 FROM sqlite_master WHERE name='must_rollback'").fetchone()
            is None
        )
