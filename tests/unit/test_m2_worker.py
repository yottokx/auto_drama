from __future__ import annotations

import hashlib
import io
import json
import threading
import wave
from datetime import UTC, datetime, timedelta

import httpx
import pytest

from services.worker.client import M2_JOB_KINDS, WorkerClient


class GenerationProtocol:
    def __init__(self):
        self.expiry = (datetime.now(UTC) + timedelta(minutes=2)).isoformat()
        self.job = {
            "id": "generation-1",
            "kind": "m2_world",
            "payload": {"seed": 17},
            "lease_id": "lease-1",
            "lease_expires_at": self.expiry,
            "attempt": 1,
        }
        self.capabilities = []
        self.completions = []
        self.failures = []
        self.completion_disconnects = 0
        self.completion_status = 200
        self.heartbeats = 0
        self.lose_lease = False
        self.renewed = threading.Event()
        self.artifacts = {}
        self.downloads = []

    def __call__(self, request):
        path = request.url.path
        if path.startswith("/api/artifacts/") and path.endswith("/content"):
            identifier = path.split("/")[3]
            self.downloads.append(identifier)
            return httpx.Response(200, content=self.artifacts[identifier])
        if path == "/api/workers":
            self.capabilities = json.loads(request.content)["capabilities"]
            return httpx.Response(201, json={"id": "worker-1"})
        if path == "/api/workers/worker-1/claim":
            return httpx.Response(200, json={"job": self.job})
        if path.endswith("/heartbeat"):
            self.heartbeats += 1
            if self.heartbeats > 1:
                self.renewed.set()
                if self.lose_lease:
                    return httpx.Response(409)
            return httpx.Response(200, json={"lease_expires_at": self.expiry})
        if path == "/api/m2/jobs/generation-1/complete":
            assert request.headers["content-type"] == "application/zip"
            assert dict(request.url.params) == {"worker_id": "worker-1", "lease_id": "lease-1"}
            self.completions.append(request.content)
            if len(self.completions) <= self.completion_disconnects:
                raise httpx.ReadError("lost completion response", request=request)
            return httpx.Response(self.completion_status, json={})
        if path.endswith("/fail"):
            self.failures.append(json.loads(request.content))
            return httpx.Response(200, json={})
        raise AssertionError(f"Unexpected request: {request.method} {path}")


@pytest.mark.parametrize(
    "disconnects,expected", [(0, "completed"), (1, "completed"), (2, "deferred")]
)
def test_generation_dispatch_reuses_exact_result_after_uncertain_upload(
    tmp_path, disconnects, expected
):
    protocol = GenerationProtocol()
    protocol.completion_disconnects = disconnects
    executions = []

    def generate(job, directory):
        executions.append((job["payload"], directory))
        assert directory.is_relative_to(tmp_path.resolve())
        assert directory.name == job["id"]
        assert directory.is_dir()
        return b"generated bundle"

    with httpx.Client(
        base_url="http://coordinator", transport=httpx.MockTransport(protocol)
    ) as client:
        worker = WorkerClient(client, generation_runner=generate, work_dir=tmp_path)
        assert worker.run_once() == expected
    assert protocol.capabilities == ["tyrano_export", *M2_JOB_KINDS]
    assert len(executions) == 1
    assert all(result == b"generated bundle" for result in protocol.completions)
    assert protocol.failures == []


def test_generation_validation_rejection_is_reported_as_failed_attempt(tmp_path):
    protocol = GenerationProtocol()
    protocol.completion_status = 422
    with httpx.Client(
        base_url="http://coordinator", transport=httpx.MockTransport(protocol)
    ) as client:
        worker = WorkerClient(client, generation_runner=lambda *_: b"invalid", work_dir=tmp_path)
        assert worker.run_once() == "failed"
    assert len(protocol.failures) == 1
    assert protocol.failures[0]["lease_id"] == "lease-1"


def test_generation_does_not_publish_after_lease_replacement(tmp_path):
    protocol = GenerationProtocol()
    protocol.lose_lease = True

    def generate(*_):
        assert protocol.renewed.wait(2)
        for thread in threading.enumerate():
            if thread.name == "worker-heartbeat":
                thread.join(2)
        return b"obsolete"

    with httpx.Client(
        base_url="http://coordinator", transport=httpx.MockTransport(protocol)
    ) as client:
        worker = WorkerClient(
            client,
            generation_runner=generate,
            work_dir=tmp_path,
            heartbeat_interval=0.01,
        )
        assert worker.run_once() == "stale"
    assert protocol.completions == []
    assert protocol.failures == []


def test_generation_job_identifier_cannot_escape_cache(tmp_path):
    protocol = GenerationProtocol()
    protocol.job["id"] = "../escaped"

    def generate(*_):
        raise AssertionError("Unsafe job identifier reached generation")

    with httpx.Client(
        base_url="http://coordinator", transport=httpx.MockTransport(protocol)
    ) as client:
        worker = WorkerClient(client, generation_runner=generate, work_dir=tmp_path)
        assert worker.run_once() == "failed"
    assert not (tmp_path.parent / "escaped").exists()
    assert protocol.completions == []


def test_same_job_reuses_cache_but_other_coordinator_is_isolated(tmp_path):
    paths = []

    def generate(_job, directory):
        paths.append(directory)
        return b"generated bundle"

    for server in ("http://one", "http://one", "http://two"):
        protocol = GenerationProtocol()
        with httpx.Client(base_url=server, transport=httpx.MockTransport(protocol)) as client:
            assert (
                WorkerClient(client, generation_runner=generate, work_dir=tmp_path).run_once()
                == "completed"
            )
    assert paths[0] == paths[1]
    assert paths[2] != paths[0]


def reference_wav():
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(b"\x10\x00" * 2400)
    return output.getvalue()


def clone_protocol():
    protocol = GenerationProtocol()
    protocol.job["kind"] = "m2_voice_clone"
    content = reference_wav()
    protocol.artifacts["reference-1"] = content
    protocol.job["payload"].update(
        {
            "reference_voice": {
                "artifact_id": "reference-1",
                "sha256": hashlib.sha256(content).hexdigest(),
                "text": "私は司書です。",
            },
            "dialogue_text": "新しい本を探しましょう。",
        }
    )
    return protocol


def test_voice_clone_downloads_validated_reference_and_reuses_it_on_retry(tmp_path):
    protocol = clone_protocol()
    original_payload = json.dumps(protocol.job["payload"], sort_keys=True)

    def generate(job, directory):
        assert (directory / "reference-voice.wav").read_bytes() == protocol.artifacts["reference-1"]
        assert json.dumps(job["payload"], sort_keys=True) == original_payload
        return b"voice trial bundle"

    with httpx.Client(
        base_url="http://coordinator", transport=httpx.MockTransport(protocol)
    ) as client:
        worker = WorkerClient(client, generation_runner=generate, work_dir=tmp_path)
        assert worker.run_once() == "completed"
        assert worker.run_once() == "completed"
    assert protocol.downloads == ["reference-1"]
    assert protocol.completions == [b"voice trial bundle"] * 2


@pytest.mark.parametrize("failure", ["hash", "identifier", "text", "format", "truncated"])
def test_voice_clone_rejects_invalid_references_before_model_execution(tmp_path, failure):
    protocol = clone_protocol()
    reference = protocol.job["payload"]["reference_voice"]
    if failure == "hash":
        reference["sha256"] = "0" * 64
    elif failure == "identifier":
        reference["artifact_id"] = "https://elsewhere/voice.wav"
    elif failure == "text":
        reference["text"] = " "
    else:
        data = b"not a WAV" if failure == "format" else reference_wav()[:-100]
        protocol.artifacts["reference-1"] = data
        reference["sha256"] = hashlib.sha256(data).hexdigest()

    def generate(*_):
        raise AssertionError("Invalid reference reached the voice model")

    with httpx.Client(
        base_url="http://coordinator", transport=httpx.MockTransport(protocol)
    ) as client:
        worker = WorkerClient(client, generation_runner=generate, work_dir=tmp_path)
        assert worker.run_once() == "failed"
    assert not protocol.completions
    assert len(protocol.failures) == 1
    if failure in {"identifier", "text"}:
        assert not protocol.downloads
