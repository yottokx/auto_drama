"""The download control loop must remain alive during independent generation work."""

import json
from pathlib import Path

import httpx

from services.worker.tts_download_client import TTSDownloadAgent


class Manager:
    def __init__(self):
        self.started = []
        self.cancelled = []
        self.state = []
        self.closed = False

    def snapshots(self):
        return self.state

    def inventory(self):
        return [{"model_id": "irodori-v4-large", "precision": "bf16",
                 "manifest_id": "pinned", "file_download_ready": True,
                 "local_path": "must-not-leave-worker"}]

    def start(self, operation):
        self.started.append(operation)
        self.state = [{"operation_id": operation["id"], "status": "downloading",
                       "completed_bytes": 4, "total_bytes": 10}]

    def cancel(self, identifier):
        self.cancelled.append(identifier)

    def close(self, *, wait=False):
        self.closed = True


def test_claim_uses_frozen_manifest_and_renews_without_new_bytes():
    manager = Manager()
    requests = []

    def protocol(request):
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path.endswith("/tts-models"):
            assert "local_path" not in body["models"][0]
            return httpx.Response(200, json={})
        if request.url.path.endswith("/claim"):
            return httpx.Response(200, json={"operation": {
                "id": "download-1", "lease_id": "lease-1", "model_id": "irodori-v4-large",
                "precision": "bf16", "manifest": {"manifest_id": "original-revision"}
            }})
        assert body["worker_id"] == "worker-1" and body["lease_id"] == "lease-1"
        assert body["done_bytes"] == 4
        return httpx.Response(200, json={"operation": {"cancel_requested": False}})

    with httpx.Client(base_url="http://coordinator", transport=httpx.MockTransport(protocol)) as client:
        agent = TTSDownloadAgent(client, "worker-1", Path("."), manager=manager)
        agent.tick()
        agent.tick()
        agent.tick()
        assert manager.started[0]["manifest"]["manifest_id"] == "original-revision"
        assert len(manager.started) == 1
        assert len([path for path, _ in requests if path.endswith("/progress")]) == 2
        agent.close()
    assert manager.closed and "download-1" in manager.cancelled


def test_cancel_instruction_stops_local_stream_and_stale_lease_cannot_report_again():
    manager = Manager()
    claim = True
    progress_count = 0

    def protocol(request):
        nonlocal claim, progress_count
        if request.url.path.endswith("/tts-models"):
            return httpx.Response(200, json={})
        if request.url.path.endswith("/claim"):
            operation = {"id": "download-1", "lease_id": "lease-1",
                         "model_id": "irodori-v4-large", "precision": "bf16"} if claim else None
            claim = False
            return httpx.Response(200, json={"operation": operation})
        progress_count += 1
        if progress_count == 1:
            return httpx.Response(200, json={"cancel_requested": True})
        return httpx.Response(409, json={"detail": "expired"})

    with httpx.Client(base_url="http://coordinator", transport=httpx.MockTransport(protocol)) as client:
        agent = TTSDownloadAgent(client, "worker-1", Path("."), manager=manager)
        for _ in range(4):
            agent.tick()
        assert progress_count == 2
        assert manager.cancelled == ["download-1", "download-1"]
        assert not agent._leases
