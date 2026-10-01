"""Management API tests use tiny manifests and never fetch model weights."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from services.coordinator import tts_settings
from services.coordinator.app import create_app
from services.coordinator.service import Coordinator

MODELS = ("irodori-v4.1-small", "irodori-v4-large")
PRECISIONS = ("fp32", "bf16", "int8", "int4")


@pytest.fixture(autouse=True)
def tiny_catalog(monkeypatch):
    revision = ["test-pinned-revision"]

    def bundle(model_id, precision):
        if model_id not in MODELS or precision not in PRECISIONS:
            raise ValueError("Unsupported test model")
        weight = "base" if precision in ("fp32", "bf16") else precision
        return {
            "manifest_id": f"{model_id}-{weight}-{revision[0]}",
            "provider_id": "irodori", "model_id": model_id, "precision": precision,
            "weight_id": weight,
            "total_bytes": 100, "prepared_only": True,
            "files": [{"repo_id": "test/tts", "revision": revision[0],
                       "path": "model.safetensors", "size": 100, "sha256": "a" * 64}],
        }

    monkeypatch.setattr(tts_settings, "_bundle", bundle)
    monkeypatch.setattr(tts_settings, "_catalog", lambda: [
        {"model_id": model_id, "provider_id": "irodori", "precisions": list(PRECISIONS)}
        for model_id in MODELS
    ])
    return revision


def worker(client, capabilities=None):
    response = client.post("/api/workers", json={
        "name": "test-download-worker", "capabilities": capabilities or ["tts_download"],
    })
    assert response.status_code == 201, response.text
    return response.json()["id"]


def enqueue(client, worker_id, model_id=MODELS[1], precision="bf16"):
    response = client.post(f"/api/workers/{worker_id}/tts-downloads", json={
        "model_id": model_id, "precision": precision,
    })
    assert response.status_code == 201, response.text
    return response.json()


def claim(client, worker_id):
    response = client.post(f"/api/workers/{worker_id}/tts-downloads/claim", json={})
    assert response.status_code == 200, response.text
    return response.json()["operation"]


def progress(client, operation, status="downloading", **patch):
    return client.post(f"/api/tts-downloads/{operation['id']}/progress", json={
        "worker_id": operation["worker_id"], "lease_id": operation["lease_id"],
        "status": status, "done_bytes": 20, "total_bytes": 100, **patch,
    })


@pytest.mark.parametrize(("failure", "expected"), [
    ("Network access blocked (WinError 10013)", "外部通信が拒否"),
    ("Network connection failed", "配布元に接続できません"),
    ("Model download timed out", "タイムアウト"),
    ("TLS verification failed", "証明書"),
    ("Local storage permission denied", "書き込み権限"),
    ("Model source returned HTTP 404", "配布元にありません"),
])
def test_download_failure_categories_are_actionable_without_private_details(tmp_path, failure, expected):
    with TestClient(create_app(tmp_path)) as client:
        worker_id = worker(client)
        enqueue(client, worker_id)
        operation = claim(client, worker_id)
        private = " https://example.test/download?token=secret C:\\private\\model.bin"
        result = progress(client, operation, "failed", error=failure + private)
        assert result.status_code == 200
        public = result.json()["error"]
        assert expected in public
        assert all(value not in public for value in ("secret", "example.test", "private"))


def test_active_settings_persist_independent_ready_selections_without_changing_existing_jobs(tmp_path):
    from tests.integration.test_coordinator import sample

    with TestClient(create_app(tmp_path)) as client:
        original = client.get("/api/settings/tts").json()
        assert original["generation_active"] is True
        assert original["stage"] == "active"
        settings = copy.deepcopy(original["settings"])
        settings["voice_design"].update(model_id=MODELS[1], precision="bf16")
        settings["voice_clone"].update(model_id=MODELS[0], precision="int4")
        target = worker(client, ["tts_download", "m2_voice", "m2_voice_clone"])
        models = [{**settings[purpose], "manifest_id": tts_settings._bundle(
            settings[purpose]["model_id"], settings[purpose]["precision"],
        )["manifest_id"], "file_download_ready": True, "generation_ready": True,
                   "generation_purposes": [purpose]}
                  for purpose in ("voice_design", "voice_clone")]
        for item in models:
            item.pop("provider_id")
        assert client.post(f"/api/workers/{target}/tts-models", json={"models": models}).status_code == 200
        # Common settings never rewrite an existing job's frozen inputs.
        created = sample(client)
        before = client.get(f"/api/jobs/{created['job']['id']}").json()
        assert client.post("/api/settings/tts", json=settings).json()["settings"] == settings
        assert client.get(f"/api/jobs/{created['job']['id']}").json() == before
        assert client.get("/api/tts/models").json()["models"][1]["precisions"] == list(PRECISIONS)
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/settings/tts").json()["settings"] == settings


@pytest.mark.parametrize("patch", [
    {"generation_active": False}, {"schema_version": 2}, {"unknown": "secret"},
    {"voice_design": {"provider_id": "missing", "model_id": MODELS[1], "precision": "bf16"}},
    {"voice_design": {"model_id": MODELS[1], "precision": "fp16"}},
])
def test_invalid_settings_do_not_replace_saved_selections(tmp_path, patch):
    with TestClient(create_app(tmp_path)) as client:
        before = client.get("/api/settings/tts").json()["settings"]
        assert client.post("/api/settings/tts", json={**before, **patch}).status_code == 422
        assert client.get("/api/settings/tts").json()["settings"] == before


@pytest.mark.parametrize("model_id", MODELS)
@pytest.mark.parametrize("precision", PRECISIONS)
def test_all_requested_models_and_precisions_enqueue_without_download(tmp_path, model_id, precision):
    with TestClient(create_app(tmp_path)) as client:
        operation = enqueue(client, worker(client), model_id, precision)
        assert (operation["model_id"], operation["precision"]) == (model_id, precision)
        assert operation["status"] == "queued"
        assert operation["file_download_ready"] is False
        assert operation["generation_ready"] is False
        assert "manifest" not in operation and "lease_id" not in operation


def test_operations_persist_target_worker_and_deduplicate_shared_checkpoint(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        target = worker(client)
        other = worker(client)
        operation = enqueue(client, target, precision="fp32")
        assert enqueue(client, target, precision="bf16")["id"] == operation["id"]
        assert claim(client, other) is None
    with TestClient(create_app(tmp_path)) as client:
        claimed = claim(client, target)
        assert claimed["id"] == operation["id"]
        assert claim(client, target) is None
        assert claimed["manifest"]["files"][0]["revision"] == "test-pinned-revision"
        assert progress(client, claimed).status_code == 200
        completed = progress(client, claimed, "completed").json()
        assert completed["status"] == "completed"
        assert completed["file_download_ready"] is True
        assert completed["generation_ready"] is False
        state = client.get("/api/tts-downloads").json()
        inventory = next(item for item in state["workers"] if item["id"] == target)["inventory"]
        assert {item["precision"] for item in inventory} == {"fp32", "bf16"}
        assert all(item["file_download_ready"] and not item["generation_ready"] for item in inventory)
        assert "manifest" not in state["operations"][0]


def test_cancel_running_blocks_completion_and_resume_keeps_manifest(tmp_path, tiny_catalog):
    with TestClient(create_app(tmp_path)) as client:
        target = worker(client)
        queued = enqueue(client, target)
        operation = claim(client, target)
        assert progress(client, operation, done_bytes=40).status_code == 200
        cancelled = client.post(f"/api/tts-downloads/{queued['id']}/cancel").json()
        assert cancelled["status"] == "cancelling"
        assert cancelled["cancel_requested"] is True
        assert client.post(f"/api/tts-downloads/{queued['id']}/resume").status_code == 409
        # If verification finished concurrently with cancellation, it still cannot mark ready.
        assert progress(client, operation, "completed").json()["status"] == "cancelled"
        tiny_catalog[0] = "new-catalog-revision"
        resumed = client.post(f"/api/tts-downloads/{queued['id']}/resume").json()
        assert resumed["status"] == "queued"
        next_claim = claim(client, target)
        assert next_claim["manifest"] == operation["manifest"]
        assert next_claim["lease_id"] != operation["lease_id"]
        assert progress(client, operation).status_code == 409
        assert progress(client, next_claim, "completed").status_code == 200
        inventory = client.get("/api/tts-downloads").json()["workers"][0]["inventory"]
        assert any(item["manifest_id"] == operation["manifest_id"]
                   and item["precision"] == operation["precision"] for item in inventory)
        assert not any("new-catalog-revision" in item["manifest_id"] for item in inventory)


def test_expired_worker_restart_can_retarget_same_pinned_operation(tmp_path, tiny_catalog):
    now = [1_800_000_000.0]
    coordinator = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=coordinator)) as client:
        old_worker = worker(client)
        enqueue(client, old_worker)
        old = claim(client, old_worker)
        new_worker = worker(client)
        assert client.post(f"/api/tts-downloads/{old['id']}/resume", json={
            "worker_id": new_worker,
        }).status_code == 409
    now[0] += 11
    tiny_catalog[0] = "catalog-after-restart"
    restarted = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=restarted)) as client:
        state = client.get("/api/tts-downloads").json()
        assert state["operations"][0]["status"] == "interrupted"
        assert progress(client, old).status_code == 409
        resumed = client.post(f"/api/tts-downloads/{old['id']}/resume", json={
            "worker_id": new_worker,
        })
        assert resumed.status_code == 200, resumed.text
        assert claim(client, old_worker) is None
        current = claim(client, new_worker)
        assert current["id"] == old["id"]
        assert current["manifest"] == old["manifest"]


def test_download_management_does_not_take_generation_worker_slot(tmp_path):
    from tests.integration.test_coordinator import sample

    with TestClient(create_app(tmp_path)) as client:
        target = worker(client, ["tts_download", "tyrano_export"])
        created = sample(client)
        enqueue(client, target)
        assert claim(client, target) is not None
        generation = client.post(f"/api/workers/{target}/claim", json={}).json()["job"]
        assert generation["id"] == created["job"]["id"]


def test_invalid_target_progress_and_private_errors_are_safe(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        unsupported = worker(client, ["tyrano_export"])
        assert client.post(f"/api/workers/{unsupported}/tts-downloads", json={
            "model_id": MODELS[1], "precision": "bf16",
        }).status_code == 409
        target = worker(client)
        enqueue(client, target)
        operation = claim(client, target)
        assert progress(client, operation, worker_id=unsupported).status_code == 409
        assert progress(client, operation, done_bytes=101).status_code == 422
        assert progress(client, operation, total_bytes=999).status_code == 422
        assert progress(client, operation, current_file="PRIVATE_TOKEN").json()["current_file"] is None
        failed = progress(client, operation, "failed", error=(
            "https://files.invalid/model?token=PRIVATE_TOKEN at C:\\PRIVATE_PATH\\model"
        ), current_file="C:\\PRIVATE_PATH\\model.safetensors").json()
        assert failed["status"] == "failed"
        assert failed["current_file"] == "model.safetensors"
        assert "PRIVATE" not in json.dumps(client.get("/api/tts-downloads").json())


def test_inventory_validates_manifest_and_allows_missing_files(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        target = worker(client)
        manifest = tts_settings._bundle(MODELS[1], "int8")
        item = {"model_id": MODELS[1], "precision": "int8",
                "manifest_id": manifest["manifest_id"], "file_download_ready": True}
        assert client.post(f"/api/workers/{target}/tts-models", json={
            "models": [item],
        }).status_code == 200
        invalid = client.post(f"/api/workers/{target}/tts-models", json={
            "models": [{**item, "manifest_id": "unknown"}],
        })
        assert invalid.status_code == 422
        assert client.post(f"/api/workers/{target}/tts-models", json={
            "models": [item, item],
        }).status_code == 422
        state = client.get("/api/tts-downloads").json()
        assert state["workers"][0]["inventory"][0]["file_download_ready"] is True
        assert client.post(f"/api/workers/{target}/tts-models", json={"models": []}).status_code == 200
        assert client.get("/api/tts-downloads").json()["workers"][0]["inventory"] == []
