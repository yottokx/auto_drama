from __future__ import annotations

import json
import threading
from datetime import UTC, datetime, timedelta

import httpx

from packages.tyrano_export import compile_bundle, demo_content
from services.worker.client import WorkerClient


class CoordinatorProtocol:
    def __init__(self):
        self.script, self.assets = demo_content()
        self.expiry = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
        self.job = {
            "id": "job-1",
            "kind": "tyrano_export",
            "payload": {"script_artifact_id": "script-1"},
            "lease_id": "lease-1",
            "lease_expires_at": self.expiry,
            "attempt": 1,
        }
        self.requests: list[httpx.Request] = []
        self.completions: list[bytes] = []
        self.failures: list[dict] = []
        self.registers = 0
        self.heartbeats = 0
        self.corrupt_asset = False
        self.stale_after: int | None = None
        self.completion_disconnects = 0
        self.renewed = threading.Event()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path == "/api/workers":
            self.registers += 1
            assert json.loads(request.content) == {
                "name": "test-worker",
                "capabilities": ["tyrano_export"],
            }
            return httpx.Response(200, json={"id": "worker-1"})
        if path == "/api/workers/worker-1/claim":
            return httpx.Response(200, json={"job": self.job})
        if path == "/api/jobs/job-1/heartbeat":
            self.heartbeats += 1
            assert json.loads(request.content) == {
                "worker_id": "worker-1",
                "lease_id": "lease-1",
            }
            if self.heartbeats > 1:
                self.renewed.set()
            if self.stale_after is not None and self.heartbeats > self.stale_after:
                return httpx.Response(409, json={"detail": "Stale lease"})
            return httpx.Response(200, json={"lease_expires_at": self.expiry})
        if path == "/api/artifacts/script-1/content":
            return httpx.Response(200, content=self.script.model_dump_json().encode())
        for asset in self.script.assets:
            if path == f"/api/artifacts/{asset.artifact_id}/content":
                content = b"corrupt" if self.corrupt_asset else self.assets[asset.id]
                return httpx.Response(200, content=content)
        if path == "/api/jobs/job-1/complete":
            self.completions.append(request.content)
            assert request.headers["content-type"] == "application/zip"
            assert dict(request.url.params) == {"worker_id": "worker-1", "lease_id": "lease-1"}
            if len(self.completions) <= self.completion_disconnects:
                raise httpx.ReadError("Connection lost after upload", request=request)
            return httpx.Response(200, json={"job": {"state": "succeeded"}, "artifact": {}})
        if path == "/api/jobs/job-1/fail":
            self.failures.append(json.loads(request.content))
            return httpx.Response(200, json={"state": "failed"})
        raise AssertionError(f"Unexpected request: {request.method} {path}")


def make_worker(protocol, *, heartbeat_interval=10):
    client = httpx.Client(base_url="http://coordinator", transport=httpx.MockTransport(protocol))
    return client, WorkerClient(client, name="test-worker", heartbeat_interval=heartbeat_interval)


def test_worker_registers_once_and_leaves_idle_queue_alone():
    protocol = CoordinatorProtocol()
    protocol.job = None
    client, worker = make_worker(protocol)
    with client:
        assert worker.run_once() == "idle"
        assert worker.run_once() == "idle"
    assert protocol.registers == 1
    assert protocol.heartbeats == 0
    assert protocol.completions == []


def test_worker_fetches_inputs_and_uploads_exact_deterministic_bundle():
    protocol = CoordinatorProtocol()
    client, worker = make_worker(protocol)
    with client:
        assert worker.run_once() == "completed"
    assert protocol.completions == [compile_bundle(protocol.script, protocol.assets)]
    assert protocol.failures == []
    assert protocol.heartbeats >= 1


def test_worker_rejects_corrupt_inputs_before_upload():
    protocol = CoordinatorProtocol()
    protocol.corrupt_asset = True
    client, worker = make_worker(protocol)
    with client:
        assert worker.run_once() == "failed"
    assert protocol.completions == []
    assert len(protocol.failures) == 1
    assert "hash mismatch" in protocol.failures[0]["error"]
    assert protocol.failures[0]["lease_id"] == "lease-1"


def test_worker_ignores_stale_lease_before_fetching_inputs():
    protocol = CoordinatorProtocol()
    protocol.stale_after = 0
    client, worker = make_worker(protocol)
    with client:
        assert worker.run_once() == "stale"
    assert protocol.completions == []
    assert protocol.failures == []
    assert not any("/artifacts/" in request.url.path for request in protocol.requests)


def test_worker_retries_exact_completion_after_lost_response():
    protocol = CoordinatorProtocol()
    protocol.completion_disconnects = 1
    client, worker = make_worker(protocol)
    with client:
        assert worker.run_once() == "completed"
    assert len(protocol.completions) == 2
    assert protocol.completions[0] == protocol.completions[1]
    assert protocol.failures == []


def test_worker_never_reports_failure_after_uncertain_completion():
    protocol = CoordinatorProtocol()
    protocol.completion_disconnects = 2
    client, worker = make_worker(protocol)
    with client:
        assert worker.run_once() == "deferred"
    assert len(protocol.completions) == 2
    assert protocol.failures == []


def test_worker_renews_lease_during_compilation(monkeypatch):
    protocol = CoordinatorProtocol()

    def slow_compile(script, assets):
        assert protocol.renewed.wait(2), "Heartbeat did not renew during compilation"
        return compile_bundle(script, assets)

    monkeypatch.setattr("services.worker.client.compile_bundle", slow_compile)
    client, worker = make_worker(protocol, heartbeat_interval=0.01)
    with client:
        assert worker.run_once() == "completed"
    assert protocol.heartbeats >= 2


def test_worker_discards_compiled_output_after_lease_is_replaced(monkeypatch):
    protocol = CoordinatorProtocol()
    protocol.stale_after = 1

    def slow_compile(script, assets):
        assert protocol.renewed.wait(2), "Heartbeat did not detect the stale lease"
        # The event is set inside the HTTP callback; wait for the heartbeat
        # thread to finish publishing its error before returning the bundle.
        for thread in threading.enumerate():
            if thread.name == "worker-heartbeat":
                thread.join(2)
        return compile_bundle(script, assets)

    monkeypatch.setattr("services.worker.client.compile_bundle", slow_compile)
    client, worker = make_worker(protocol, heartbeat_interval=0.01)
    with client:
        assert worker.run_once() == "stale"
    assert protocol.completions == []
    assert protocol.failures == []
