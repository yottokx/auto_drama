"""Exercise the real acquisition manager against an isolated Coordinator API."""

import hashlib
import threading

import httpx
import pytest
from fastapi.testclient import TestClient

from packages.contracts.tts_catalog import manifest_fingerprint
from services.coordinator import tts_settings
from services.coordinator.app import create_app
from services.worker import tts_downloads
from services.worker.tts_download_client import TTSDownloadAgent

MODEL = "irodori-v4-large"
PRECISIONS = ("fp32", "bf16", "int8", "int4")
DATA = b"tiny mock checkpoint: no actual model weights"
DIRECTORY = "services/worker/runtimes/irodori/models/integration-test"


@pytest.fixture
def tiny_sources(monkeypatch):
    revision = ["a" * 40]

    def bundle(model_id, precision):
        assert model_id == MODEL
        assert precision in PRECISIONS
        weight = "full" if precision in {"fp32", "bf16"} else precision
        value = {
            "provider_id": "irodori", "model_id": model_id, "precision": precision,
            "weight_id": weight, "prepared_only": True, "total_bytes": len(DATA),
            "files": [{
                "repo_id": "test/tts", "revision": revision[0],
                "local_dir": f"{DIRECTORY}/{weight}", "path": "model.safetensors",
                "size": len(DATA), "sha256": hashlib.sha256(DATA).hexdigest(),
            }],
        }
        value["manifest_id"] = manifest_fingerprint(value)
        return value

    def catalog():
        return [{"model_id": MODEL, "provider_id": "irodori", "precisions": list(PRECISIONS)}]

    monkeypatch.setattr(tts_settings, "_bundle", bundle)
    monkeypatch.setattr(tts_settings, "_catalog", catalog)
    monkeypatch.setattr(tts_downloads, "catalog", catalog)
    return bundle, revision


def enqueue(client):
    response = client.post("/api/workers", json={
        "name": "isolated-download-worker", "capabilities": ["tts_download"],
    })
    assert response.status_code == 201, response.text
    worker_id = response.json()["id"]
    response = client.post(f"/api/workers/{worker_id}/tts-downloads", json={
        "model_id": MODEL, "precision": "bf16",
    })
    assert response.status_code == 201, response.text
    return worker_id, response.json()


def test_real_worker_reports_progress_and_reuses_verified_full_checkpoint(tmp_path, tiny_sources):
    bundle, _ = tiny_sources
    started, release, finished = threading.Event(), threading.Event(), threading.Event()
    requests = []

    def download(request):
        requests.append(request)
        started.set()
        assert release.wait(5), "The test did not release the mock model stream"
        return httpx.Response(200, content=DATA)

    def report(snapshot):
        if snapshot["status"] in {"completed", "failed", "cancelled"}:
            finished.set()

    worker_root = tmp_path / "worker"
    manager = tts_downloads.DownloadManager(
        worker_root, report, scan_existing=False, reserve_bytes=0, bundle_resolver=bundle,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(download)),
    )
    with TestClient(create_app(tmp_path / "coordinator")) as client:
        settings_before = client.get("/api/settings/tts").json()
        worker_id, operation = enqueue(client)
        agent = TTSDownloadAgent(client, worker_id, worker_root, manager=manager)
        try:
            agent.tick()
            assert started.wait(5)
            # A blocked network stream still renews the Coordinator lease and
            # publishes progress through the same protocol used by the Worker.
            agent.tick()
            running = client.get("/api/tts-downloads").json()["operations"][0]
            assert running["status"] == "downloading"
            assert running["done_bytes"] == 0
            assert running["total_bytes"] == len(DATA)
            assert running["current_file"] == "model.safetensors"
            assert running["file_download_ready"] is False
            release.set()
            assert finished.wait(5)
            agent.tick()
            state = client.get("/api/tts-downloads").json()
            completed = state["operations"][0]
            assert completed["id"] == operation["id"]
            assert completed["status"] == "completed"
            assert completed["done_bytes"] == len(DATA)
            assert completed["file_download_ready"] is True
            assert completed["generation_ready"] is False
            inventory = state["workers"][0]["inventory"]
            assert {item["precision"] for item in inventory} == {"fp32", "bf16"}
            assert {item["manifest_id"] for item in inventory} == {operation["manifest_id"]}
            assert all(item["file_download_ready"] and not item["generation_ready"]
                       for item in inventory)
            assert client.get("/api/settings/tts").json() == settings_before
            assert "manifest" not in completed and str(worker_root) not in str(state)
            target = worker_root / f"{DIRECTORY}/full/model.safetensors"
            assert target.read_bytes() == DATA
            assert not target.with_name(target.name + ".part").exists()
            assert len(requests) == 1
            assert requests[0].url.host == "huggingface.co"
            assert requests[0].url.path == f"/test/tts/resolve/{'a' * 40}/model.safetensors"
        finally:
            release.set()
            agent.close()
            manager.close(wait=True)

    # A new manager reads the durable verification receipt without fetching or
    # hashing the checkpoint again, and advertises both execution precisions.
    def no_network(request):
        pytest.fail(f"A verified checkpoint should not be fetched again: {request.url}")

    restored = tts_downloads.DownloadManager(
        worker_root, scan_existing=False, reserve_bytes=0, bundle_resolver=bundle,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(no_network)),
    )
    try:
        assert {item["precision"] for item in restored.inventory()} == {"fp32", "bf16"}
    finally:
        restored.close(wait=True)


def test_queued_download_uses_frozen_manifest_after_catalog_changes(tmp_path, tiny_sources):
    bundle, revision = tiny_sources
    finished = threading.Event()
    requests = []

    def download(request):
        requests.append(request)
        return httpx.Response(200, content=DATA)

    def report(snapshot):
        if snapshot["status"] in {"completed", "failed", "cancelled"}:
            finished.set()

    worker_root = tmp_path / "worker"
    manager = tts_downloads.DownloadManager(
        worker_root, report, scan_existing=False, reserve_bytes=0, bundle_resolver=bundle,
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(download)),
    )
    with TestClient(create_app(tmp_path / "coordinator")) as client:
        worker_id, operation = enqueue(client)
        revision[0] = "b" * 40
        current_manifest = bundle(MODEL, "bf16")["manifest_id"]
        assert current_manifest != operation["manifest_id"]
        agent = TTSDownloadAgent(client, worker_id, worker_root, manager=manager)
        try:
            agent.tick()
            assert finished.wait(5)
            agent.tick()
            state = client.get("/api/tts-downloads").json()
            completed = state["operations"][0]
            assert completed["status"] == "completed"
            assert completed["manifest_id"] == operation["manifest_id"]
            assert completed["generation_ready"] is False
            assert len(requests) == 1
            assert requests[0].url.path == f"/test/tts/resolve/{'a' * 40}/model.safetensors"
            assert all(item["manifest_id"] != current_manifest
                       for item in state["workers"][0]["inventory"])
            # Receipts bind file identity to revision; downloaded old files
            # cannot report the newer catalog as ready.
            assert manager.inventory() == []
        finally:
            agent.close()
            manager.close(wait=True)
