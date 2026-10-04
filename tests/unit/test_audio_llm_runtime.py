"""One-shot local LLM ownership and selected-model-only Ollama release."""

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from urllib.error import HTTPError, URLError

import pytest

from scripts.audio import llm_runtime
from scripts.audio.engine import GenerationCancelled


def test_ollama_helper_import_does_not_require_worker_or_pydantic():
    code = """
import builtins
original = builtins.__import__
def guarded(name, *args, **kwargs):
    if name == 'services' or name.startswith('services.') or name.startswith('pydantic'):
        raise ModuleNotFoundError('worker dependencies intentionally unavailable')
    return original(name, *args, **kwargs)
builtins.__import__ = guarded
from scripts.audio.llm_runtime import is_ollama_url
assert is_ollama_url('http://127.0.0.1:11434/v1')
"""
    result = subprocess.run([sys.executable, "-c", code],
                            cwd=Path(__file__).resolve().parents[2],
                            capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr


@pytest.fixture
def deployment(tmp_path):
    (tmp_path / "config").mkdir()
    (tmp_path / "services/worker/config").mkdir(parents=True)
    (tmp_path / "models").mkdir()
    (tmp_path / "bin").mkdir()
    (tmp_path / "bin/llama-server.exe").write_bytes(b"test executable, never launched")
    (tmp_path / "services/worker/config/models.local.json").write_text(json.dumps({
        "schema_version": 1, "model_dirs": ["models"],
        "server_executable": "bin/llama-server.exe",
    }), encoding="utf-8")
    for model in ("gemma-test", "another-model"):
        (tmp_path / "models" / (model + ".gguf")).write_bytes(b"GGUFtest")
    (tmp_path / "config/m2-generation.json").write_text(json.dumps({
        "llm": {"model_id": "gemma-test"},
    }), encoding="utf-8")
    return tmp_path


def test_native_choices_use_worker_registry_and_validated_default(deployment):
    assert llm_runtime.list_native_models(deployment) == ["another-model", "gemma-test"]
    assert llm_runtime.default_native_model(deployment) == "gemma-test"


def test_native_choices_skip_invalid_model_and_choose_available_default(deployment):
    (deployment / "models/gemma-test.gguf").write_bytes(b"not-GGUF")
    assert llm_runtime.list_native_models(deployment) == ["another-model"]
    assert llm_runtime.default_native_model(deployment) == "another-model"


def test_native_choices_require_a_local_server(deployment):
    (deployment / "bin/llama-server.exe").unlink()
    assert llm_runtime.list_native_models(deployment) == []
    with pytest.raises(ValueError, match="GGUF"):
        llm_runtime.default_native_model(deployment)


@pytest.fixture
def owners(monkeypatch):
    events, launched, requests = [], [], []
    process = SimpleNamespace(poll=lambda: None)

    @contextmanager
    def lease(path, timeout):
        events.append(("lease-acquired", path, timeout))
        try:
            yield
        finally:
            events.append(("lease-released",))

    @contextmanager
    def owned(command, log, *, cwd, timeout):
        launched.append((command, log, cwd, timeout))
        events.append(("process-started",))
        try:
            yield process
        finally:
            events.append(("process-stopped",))

    def health(opener, url, *, timeout, payload=None):
        requests.append((url, timeout, payload))
        return {"status": "ok"}

    monkeypatch.setattr(llm_runtime, "gpu_lock", lease)
    monkeypatch.setattr(llm_runtime, "owned_process", owned)
    monkeypatch.setattr(llm_runtime, "_json_request", health)
    return SimpleNamespace(events=events, launched=launched, requests=requests, process=process)


def test_managed_server_uses_existing_assets_and_releases_before_gpu_lease(
        deployment, owners, tmp_path):
    statuses = []
    with llm_runtime.managed_llama(deployment, "gemma-test", tmp_path / "prompt",
                                  status=statuses.append) as (base_url, alias):
        assert base_url.startswith("http://127.0.0.1:") and base_url.endswith("/v1")
        assert alias == "bgm-local"
        assert owners.events[-1] == ("process-started",)
    assert [event[0] for event in owners.events] == [
        "lease-acquired", "process-started", "process-stopped", "lease-released",
    ]
    assert owners.events[0][1] == deployment / "services/worker/cache/m2/gpu.lock"
    command, log, cwd, timeout = owners.launched[0]
    assert command[0] == str(deployment / "bin/llama-server.exe")
    assert command[command.index("--model") + 1] == str(deployment / "models/gemma-test.gguf")
    assert command[command.index("--ctx-size") + 1] == "16384"
    assert command[command.index("--reasoning-budget") + 1] == "0"
    assert command[command.index("--alias") + 1] == "bgm-local"
    assert command[command.index("--host") + 1] == "127.0.0.1"
    assert cwd == deployment / "bin" and log.name == "llama-server.log"
    assert 0 < timeout <= 600
    assert owners.requests[0][0].endswith("/health")
    assert statuses[-1]["phase"] == "releasing_llm"


def test_managed_server_stops_on_request_failure(deployment, owners, tmp_path):
    with (pytest.raises(ValueError, match="request failed"),
          llm_runtime.managed_llama(deployment, "gemma-test", tmp_path / "prompt")):
        raise ValueError("request failed")
    assert [event[0] for event in owners.events][-2:] == ["process-stopped", "lease-released"]


def test_managed_server_does_not_launch_unknown_model(deployment, owners, tmp_path):
    with (pytest.raises(ValueError, match="unknown"),
          llm_runtime.managed_llama(deployment, "unknown", tmp_path / "prompt")):
        pytest.fail("unknown model should never start")
    assert not owners.events and not owners.launched


def test_managed_startup_failure_stops_only_owned_server(deployment, owners, tmp_path):
    owners.process.poll = lambda: 4
    with (pytest.raises(RuntimeError, match="起動"),
          llm_runtime.managed_llama(deployment, "gemma-test", tmp_path / "prompt")):
        pytest.fail("failed server must not be yielded")
    assert [event[0] for event in owners.events][-2:] == ["process-stopped", "lease-released"]


def test_managed_health_timeout_closes_server_and_lease(
        deployment, owners, tmp_path, monkeypatch):
    monkeypatch.setattr(llm_runtime, "_STARTUP_TIMEOUT", 0)
    with (pytest.raises(TimeoutError, match="準備"),
          llm_runtime.managed_llama(deployment, "gemma-test", tmp_path / "prompt")):
        pytest.fail("unhealthy server must not be yielded")
    assert [event[0] for event in owners.events][-2:] == ["process-stopped", "lease-released"]


def test_managed_callback_cancel_closes_server_and_lease(
        deployment, owners, tmp_path, monkeypatch):
    cancelled = [False]

    def health(*args, **kwargs):
        cancelled[0] = True
        return {"status": "ok"}

    monkeypatch.setattr(llm_runtime, "_json_request", health)
    with (pytest.raises(GenerationCancelled),
          llm_runtime.managed_llama(deployment, "gemma-test", tmp_path / "prompt",
                                    cancelled=lambda: cancelled[0])):
        pytest.fail("cancelled server must not be yielded")
    assert [event[0] for event in owners.events][-2:] == ["process-stopped", "lease-released"]


@pytest.mark.parametrize("url, expected", [
    ("http://127.0.0.1:11434/v1", True),
    ("http://localhost:11434/v1/", True),
    ("http://[::1]:11434/v1", True),
    ("http://127.0.0.1:8080/v1", False),
    ("http://example.com:11434/v1", False),
    ("http://localhost.evil.example:11434/v1", False),
    ("http://localhost:invalid/v1", False),
    ("localhost:11434", False),
])
def test_only_known_loopback_ollama_port_is_detected(url, expected):
    assert llm_runtime.is_ollama_url(url) is expected


def test_ollama_not_loaded_does_not_send_unload(monkeypatch):
    calls = []

    def request(opener, url, **kwargs):
        calls.append((url, kwargs))
        return {"models": [{"name": "other-model:latest"}]}

    monkeypatch.setattr(llm_runtime, "_json_request", request)
    llm_runtime.release_ollama("http://127.0.0.1:11434/v1/", "selected-model")
    assert len(calls) == 1 and calls[0][0] == "http://127.0.0.1:11434/api/ps"


def test_absent_ollama_is_a_noop(monkeypatch):
    def request(*args, **kwargs):
        raise URLError(ConnectionRefusedError("not running"))

    monkeypatch.setattr(llm_runtime, "_json_request", request)
    llm_runtime.release_ollama("http://127.0.0.1:11434/v1", "selected-model")


def test_ollama_unloads_only_selected_model_and_waits_for_confirmation(monkeypatch):
    calls = []
    states = iter([
        {"models": [{"name": "selected-model:latest"}, {"name": "other-model:latest"}]},
        {"done": True},
        {"models": [{"name": "selected-model:latest"}, {"name": "other-model:latest"}]},
        {"models": [{"name": "other-model:latest"}]},
    ])

    def request(opener, url, **kwargs):
        calls.append((url, kwargs))
        return next(states)

    monkeypatch.setattr(llm_runtime, "_json_request", request)
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda seconds: None)
    llm_runtime.release_ollama("http://127.0.0.1:11434/v1", "selected-model")
    assert [url.rsplit("/", 1)[-1] for url, _ in calls] == ["ps", "generate", "ps", "ps"]
    assert calls[1][1]["payload"] == {"model": "selected-model", "keep_alive": 0, "stream": False}
    assert all(call.get("payload", {}).get("model") != "other-model" for _, call in calls)


def test_named_ollama_tag_does_not_unload_different_tag(monkeypatch):
    calls = []

    def request(opener, url, **kwargs):
        calls.append(url)
        return {"models": [{"name": "selected-model:other-tag"}]}

    monkeypatch.setattr(llm_runtime, "_json_request", request)
    llm_runtime.release_ollama("http://127.0.0.1:11434/v1", "selected-model:latest")
    assert len(calls) == 1


def test_known_loaded_ollama_release_failure_is_visible(monkeypatch):
    def request(opener, url, **kwargs):
        if url.endswith("/ps"):
            return {"models": [{"name": "selected-model:latest"}]}
        raise HTTPError(url, 500, "unload failed", {}, None)

    monkeypatch.setattr(llm_runtime, "_json_request", request)
    with pytest.raises(RuntimeError, match="GPU解放"):
        llm_runtime.release_ollama("http://127.0.0.1:11434/v1", "selected-model")


def test_known_loaded_ollama_release_has_bounded_wait(monkeypatch):
    clock = [0.0]

    def request(opener, url, **kwargs):
        clock[0] += 0.2
        return {"done": True} if url.endswith("/generate") else {
            "models": [{"name": "selected-model:latest"}],
        }

    monkeypatch.setattr(llm_runtime, "_json_request", request)
    monkeypatch.setattr(llm_runtime.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(llm_runtime.time, "sleep", lambda seconds: None)
    with pytest.raises(RuntimeError, match="制限時間"):
        llm_runtime.release_ollama("http://127.0.0.1:11434/v1", "selected-model", timeout=0.5)
    assert clock[0] <= 0.8


def test_malformed_ollama_state_prevents_unverified_release(monkeypatch):
    monkeypatch.setattr(llm_runtime, "_json_request", lambda *args, **kwargs: {"models": "invalid"})
    with pytest.raises(ValueError, match="常駐"):
        llm_runtime.release_ollama("http://127.0.0.1:11434/v1", "selected-model")


@pytest.mark.parametrize("reason", [TimeoutError("unknown state"), OSError("network unavailable")])
def test_unverified_ollama_connectivity_error_is_not_treated_as_unloaded(monkeypatch, reason):
    def request(*args, **kwargs):
        raise URLError(reason)

    monkeypatch.setattr(llm_runtime, "_json_request", request)
    with pytest.raises(RuntimeError, match="常駐状態"):
        llm_runtime.release_ollama("http://127.0.0.1:11434/v1", "selected-model")


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf")])
def test_ollama_release_timeout_must_be_positive_and_finite(timeout):
    with pytest.raises(ValueError, match="タイムアウト"):
        llm_runtime.release_ollama("http://127.0.0.1:11434/v1", "selected-model", timeout=timeout)
