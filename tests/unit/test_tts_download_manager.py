import copy
import errno
import hashlib
import json
import logging
import shutil
import ssl
import threading
import time
from pathlib import Path

import httpx
import pytest

from packages.contracts.tts_catalog import catalog, manifest_fingerprint, resolve_bundle
from services.worker.tts_downloads import DownloadManager

MODEL = "irodori-v4.1-small"
DIRECTORY = "services/worker/runtimes/irodori/models/test"


def manifest(data=b"model-weights", precision="fp32"):
    value = {
        "provider_id": "irodori", "model_id": MODEL, "precision": precision,
        "weight_id": "test-full", "prepared_only": True,
        "files": [{"repo_id": "official/test", "revision": "a" * 40,
                   "local_dir": DIRECTORY, "path": "model.safetensors",
                   "size": len(data), "sha256": hashlib.sha256(data).hexdigest()}],
        "total_bytes": len(data),
    }
    value["manifest_id"] = manifest_fingerprint(value)
    return value


def manager(root, bundle, handler, **kwargs):
    return DownloadManager(
        root, scan_existing=False, reserve_bytes=0,
        bundle_resolver=lambda model_id, precision: {**copy.deepcopy(bundle), "precision": precision},
        client_factory=lambda: httpx.Client(transport=httpx.MockTransport(handler)), **kwargs,
    )


def wait_terminal(downloads, operation_id="test"):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        state = next(state for state in downloads.snapshots() if state["operation_id"] == operation_id)
        if state["status"] in {"completed", "cancelled", "failed"}:
            return state
        time.sleep(0.01)
    pytest.fail("Background TTS download did not finish")


def start(downloads, identifier="test", precision="fp32", **extra):
    return downloads.start({"id": identifier, "model_id": MODEL, "precision": precision, **extra})


def test_catalog_pins_all_eight_selections_and_shares_full_weights():
    assert {item["model_id"] for item in catalog()} == {MODEL, "irodori-v4-large"}
    for item in catalog():
        assert item["precisions"] == ["fp32", "bf16", "int8", "int4"]
        full = resolve_bundle(item["model_id"], "fp32")
        bf16 = resolve_bundle(item["model_id"], "bf16")
        assert full["manifest_id"] == bf16["manifest_id"]
        assert full["files"] == bf16["files"]
        for precision in item["precisions"]:
            bundle = resolve_bundle(item["model_id"], precision)
            assert bundle["total_bytes"] == sum(file["size"] for file in bundle["files"])
            assert bundle["manifest_id"] == manifest_fingerprint(bundle)
            assert all(len(file["revision"]) == 40 for file in bundle["files"])
    with pytest.raises(ValueError):
        resolve_bundle(MODEL, "fp16")


def test_resume_validates_range_and_promotes_only_after_hash(tmp_path, monkeypatch):
    data = b"model-weights"
    bundle = manifest(data)
    path = tmp_path / DIRECTORY / "model.safetensors"
    path.parent.mkdir(parents=True)
    path.with_suffix(".safetensors.part").write_bytes(data[:5])
    requests = []

    def handler(request):
        requests.append(request)
        assert request.headers["Range"] == "bytes=5-"
        assert not path.exists()
        return httpx.Response(206, headers={"Content-Range": f"bytes 5-{len(data)-1}/{len(data)}"},
                              content=data[5:])

    downloads = manager(tmp_path, bundle, handler)
    try:
        start(downloads)
        state = wait_terminal(downloads)
        assert state["status"] == "completed"
        assert state["completed_bytes"] == len(data)
        assert path.read_bytes() == data
        assert len(requests) == 1
        monkeypatch.setattr("services.worker.tts_downloads.catalog", lambda: [{
            "model_id": MODEL, "precisions": ["fp32", "bf16"]}])
        assert {item["precision"] for item in downloads.inventory()} == {"fp32", "bf16"}
        monkeypatch.setattr(downloads, "_verify", lambda *args: pytest.fail("Unexpected rehash"))
        downloads.inventory()
        start(downloads, "repeat", "bf16")
        assert wait_terminal(downloads, "repeat")["status"] == "completed"
        assert len(requests) == 1
    finally:
        downloads.close(wait=True)


@pytest.mark.parametrize("response", [
    httpx.Response(206, headers={"Content-Range": "bytes 0-12/13"}, content=b"model-weights"),
    httpx.Response(200, content=b"wrong-weights"),
])
def test_bad_range_or_hash_never_creates_ready_file(tmp_path, response):
    bundle = manifest()
    path = tmp_path / DIRECTORY / "model.safetensors"
    path.parent.mkdir(parents=True)
    if response.status_code == 206:
        path.with_suffix(".safetensors.part").write_bytes(b"model")
    downloads = manager(tmp_path, bundle, lambda request: response)
    try:
        start(downloads)
        assert wait_terminal(downloads)["status"] == "failed"
        assert not path.exists()
        assert not downloads._verified(bundle["files"][0])
    finally:
        downloads.close(wait=True)


def test_range_ignored_restarts_safely_and_hash_failure_can_retry(tmp_path):
    bundle = manifest()
    path = tmp_path / DIRECTORY / "model.safetensors"
    path.parent.mkdir(parents=True)
    path.with_suffix(".safetensors.part").write_bytes(b"model")
    responses = iter([b"wrong-weights", b"model-weights"])
    downloads = manager(tmp_path, bundle, lambda request: httpx.Response(200, content=next(responses)))
    try:
        start(downloads)
        assert wait_terminal(downloads)["status"] == "failed"
        assert list(path.parent.glob("*.corrupt-*"))
        start(downloads)
        assert wait_terminal(downloads)["status"] == "completed"
        assert path.read_bytes() == b"model-weights"
    finally:
        downloads.close(wait=True)


def test_cancellation_retains_partial_and_duplicate_start_does_not_fetch_twice(tmp_path):
    data = b"m" * (2 * 1024**2)
    bundle = manifest(data)
    entered, release = threading.Event(), threading.Event()
    calls = []

    class SlowBody(httpx.SyncByteStream):
        def __iter__(self):
            yield data[:1024**2]
            entered.set()
            release.wait(3)
            yield data[1024**2:]

    def handler(request):
        calls.append(request)
        return httpx.Response(200, stream=SlowBody())

    downloads = manager(tmp_path, bundle, handler)
    try:
        start(downloads)
        assert entered.wait(3)
        start(downloads)
        downloads.cancel("test")
        release.set()
        assert wait_terminal(downloads)["status"] == "cancelled"
        path = tmp_path / DIRECTORY / "model.safetensors"
        assert not path.exists()
        assert path.with_suffix(".safetensors.part").stat().st_size == 1024**2
        assert len(calls) == 1
    finally:
        release.set()
        downloads.close(wait=True)


def test_restart_resume_uses_saved_manifest_even_after_catalog_changes(tmp_path):
    original = manifest()
    downloads = manager(tmp_path, original, lambda request: httpx.Response(200, content=b"model-weights"))
    start(downloads)
    assert wait_terminal(downloads)["status"] == "completed"
    downloads.close(wait=True)
    changed = manifest(b"future-weights")
    downloads = manager(tmp_path, changed, lambda request: pytest.fail("Unexpected network download"))
    try:
        start(downloads)
        state = wait_terminal(downloads)
        assert state["status"] == "completed"
        assert state["manifest_id"] == original["manifest_id"]
        with pytest.raises(ValueError, match="cannot change"):
            start(downloads, manifest=changed)
    finally:
        downloads.close(wait=True)


def test_disk_space_and_manifest_escape_rejected_before_network(tmp_path, monkeypatch):
    bundle = manifest()
    downloads = manager(tmp_path, bundle, lambda request: pytest.fail("Unexpected network request"))
    try:
        monkeypatch.setattr(shutil, "disk_usage", lambda path: shutil._ntuple_diskusage(10, 10, 0))
        start(downloads)
        assert "Insufficient free disk space" in wait_terminal(downloads)["error"]
        escaped = manifest()
        escaped["files"][0]["path"] = "../../outside"
        escaped["manifest_id"] = manifest_fingerprint(escaped)
        with pytest.raises(ValueError, match="destination"):
            start(downloads, "escape", manifest=escaped)
    finally:
        downloads.close(wait=True)


def test_corrupt_existing_file_is_preserved_and_explicit_retry_repairs(tmp_path):
    bundle = manifest()
    path = tmp_path / DIRECTORY / "model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"wrong-weights")
    downloads = manager(tmp_path, bundle,
                        lambda request: httpx.Response(200, content=b"model-weights"))
    try:
        start(downloads)
        assert wait_terminal(downloads)["status"] == "failed"
        assert not path.exists()
        assert next(path.parent.glob("*.corrupt-*")).read_bytes() == b"wrong-weights"
        start(downloads)
        assert wait_terminal(downloads)["status"] == "completed"
        assert path.read_bytes() == b"model-weights"
    finally:
        downloads.close(wait=True)


def test_receipts_from_two_managers_are_merged_and_git_blob_hash_checked(tmp_path):
    bundle = manifest()
    first = bundle["files"][0]
    second = {**first, "path": "tokenizer/tokenizer.json", "size": 9}
    second.pop("sha256")
    second["git_blob_sha1"] = hashlib.sha1(b"blob 9\0tokenizer").hexdigest()
    one = manager(tmp_path, bundle, lambda request: pytest.fail("No download needed"))
    two = manager(tmp_path, bundle, lambda request: pytest.fail("No download needed"))
    try:
        for file, data, downloads in [(first, b"model-weights", one),
                                     (second, b"tokenizer", two)]:
            path = tmp_path / file["local_dir"] / file["path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            downloads._verify(path, file)
            downloads._receipt(file)
        receipts = json.loads((one.directory / "verified-files.json").read_text())
        assert len(receipts) == 2
        assert not list(one.directory.glob("*.tmp"))
        assert two._verified(first) and two._verified(second)
    finally:
        one.close(wait=True)
        two.close(wait=True)


def test_existing_small_inventory_is_verified_once_in_background(tmp_path, monkeypatch):
    bundle = manifest()
    path = tmp_path / DIRECTORY / "model.safetensors"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"model-weights")
    monkeypatch.setattr("services.worker.tts_downloads.catalog", lambda: [{
        "model_id": MODEL, "precisions": ["fp32", "bf16"]}])
    downloads = DownloadManager(
        tmp_path, bundle_resolver=lambda model_id, precision: {**bundle, "precision": precision},
        client_factory=lambda: pytest.fail("Inventory must not download"),
    )
    try:
        downloads._scanner.join(timeout=3)
        assert not downloads._scanner.is_alive()
        assert len(downloads.inventory()) == 2
        path.write_bytes(b"wrong-weights")
        assert not downloads.inventory()
    finally:
        downloads.close(wait=True)


def test_two_managers_do_not_download_same_directory_concurrently(tmp_path):
    data = b"model-weights"
    bundle = manifest(data)
    entered, release = threading.Event(), threading.Event()
    calls = []

    def handler(request):
        calls.append(request)
        entered.set()
        release.wait(3)
        return httpx.Response(200, content=data)

    one = manager(tmp_path, bundle, handler)
    two = manager(tmp_path, bundle, handler)
    try:
        start(one, "first")
        assert entered.wait(3)
        start(two, "second")
        time.sleep(0.1)
        assert len(calls) == 1
        release.set()
        assert wait_terminal(one, "first")["status"] == "completed"
        assert wait_terminal(two, "second")["status"] == "completed"
        assert len(calls) == 1
    finally:
        release.set()
        one.close(wait=True)
        two.close(wait=True)


@pytest.mark.parametrize("status", [401, 403, 404])
def test_permanent_http_errors_fail_fast_with_safe_diagnostics(tmp_path, caplog, status):
    calls = []

    def handler(request):
        calls.append(request)
        private_request = httpx.Request("GET", "https://private.invalid?token=PRIVATE_TOKEN")
        raise httpx.HTTPStatusError(
            "PRIVATE_MESSAGE C:\\PRIVATE_PATH\\model", request=private_request,
            response=httpx.Response(status, request=private_request),
        )

    downloads = manager(tmp_path, manifest(), handler)
    try:
        with caplog.at_level(logging.ERROR, logger="services.worker.tts_downloads"):
            start(downloads)
            state = wait_terminal(downloads)
        assert state["status"] == "failed"
        assert state["error"] == f"HTTP {status} from model source"
        assert len(calls) == 1
        assert "operation_id=test current_file=model.safetensors" in caplog.text
        assert f"type=HTTPStatusError,http_status={status}" in caplog.text
        assert "PRIVATE" not in caplog.text
        assert "https://" not in caplog.text
        assert all(record.exc_info is None for record in caplog.records)
    finally:
        downloads.close(wait=True)


def test_blocked_network_cause_fails_fast_without_logging_secrets(tmp_path, caplog):
    calls = []

    def handler(request):
        calls.append(request)
        blocked = PermissionError(errno.EACCES, "PRIVATE_SOCKET_MESSAGE")
        blocked.winerror = 10013
        error = httpx.ConnectError("PRIVATE_TOKEN https://private.invalid")
        error.__cause__ = blocked
        raise error

    downloads = manager(tmp_path, manifest(), handler)
    try:
        with caplog.at_level(logging.ERROR, logger="services.worker.tts_downloads"):
            start(downloads)
            state = wait_terminal(downloads)
        assert state["status"] == "failed"
        assert state["error"] == "Network access blocked (WinError 10013)"
        assert len(calls) == 1
        assert "type=ConnectError" in caplog.text
        assert "type=PermissionError,errno=13,winerror=10013" in caplog.text
        assert "PRIVATE" not in caplog.text
        assert "https://" not in caplog.text
    finally:
        downloads.close(wait=True)


def test_local_storage_permission_is_distinguished_from_source_access(
    tmp_path, monkeypatch, caplog,
):
    calls = []
    original_open = Path.open

    def denied_open(path, *args, **kwargs):
        if path.name.endswith(".part"):
            raise PermissionError(errno.EACCES, "PRIVATE_MESSAGE", "C:\\PRIVATE_PATH\\model")
        return original_open(path, *args, **kwargs)

    def handler(request):
        calls.append(request)
        return httpx.Response(200, content=b"model-weights")

    downloads = manager(tmp_path, manifest(), handler)
    try:
        monkeypatch.setattr(Path, "open", denied_open)
        with caplog.at_level(logging.ERROR, logger="services.worker.tts_downloads"):
            start(downloads)
            state = wait_terminal(downloads)
        assert state["error"] == "Local storage permission denied"
        assert len(calls) == 1
        assert "type=PermissionError,errno=13" in caplog.text
        assert "PRIVATE" not in caplog.text
    finally:
        downloads.close(wait=True)


@pytest.mark.parametrize("failure,message", [
    ("timeout", "Model download timed out"),
    ("tls", "TLS verification failed"),
    ("network", "Network connection failed; retry to resume partial files"),
])
def test_retryable_communication_errors_keep_safe_final_cause(
    tmp_path, caplog, failure, message,
):
    calls = []

    def handler(request):
        calls.append(request)
        # Exercise retries without spending seconds on test-only failures.
        event = downloads._cancel["test"]
        event.wait = lambda timeout=None: event.is_set()
        if failure == "timeout":
            raise httpx.ReadTimeout("PRIVATE_TIMEOUT https://private.invalid")
        error = httpx.ConnectError("PRIVATE_NETWORK C:\\PRIVATE_PATH\\model")
        if failure == "tls":
            error.__cause__ = ssl.SSLCertVerificationError(1, "PRIVATE_CERTIFICATE")
        raise error

    downloads = manager(tmp_path, manifest(), handler)
    try:
        with caplog.at_level(logging.ERROR, logger="services.worker.tts_downloads"):
            start(downloads)
            state = wait_terminal(downloads)
        assert state["status"] == "failed"
        assert state["error"] == message
        assert len(calls) == 4
        assert len(caplog.records) == 1
        assert "PRIVATE" not in caplog.text
        assert "https://" not in caplog.text
        expected_type = "ReadTimeout" if failure == "timeout" else "ConnectError"
        assert f"type={expected_type}" in caplog.text
        if failure == "tls":
            assert "type=SSLCertVerificationError,errno=1" in caplog.text
    finally:
        downloads.close(wait=True)


def test_transient_network_retry_succeeds_without_failure_log(tmp_path, caplog):
    calls = []

    def handler(request):
        calls.append(request)
        if len(calls) == 1:
            event = downloads._cancel["test"]
            event.wait = lambda timeout=None: event.is_set()
            raise httpx.ConnectError("PRIVATE_RETRY_MESSAGE")
        return httpx.Response(200, content=b"model-weights")

    downloads = manager(tmp_path, manifest(), handler)
    try:
        with caplog.at_level(logging.ERROR, logger="services.worker.tts_downloads"):
            start(downloads)
            assert wait_terminal(downloads)["status"] == "completed"
        assert len(calls) == 2
        assert not caplog.records
    finally:
        downloads.close(wait=True)


def test_unknown_background_failure_records_only_safe_exception_type(tmp_path, caplog):
    def handler(request):
        raise RuntimeError("PRIVATE_EXCEPTION https://private.invalid?token=PRIVATE_TOKEN")

    downloads = manager(tmp_path, manifest(), handler)
    try:
        with caplog.at_level(logging.ERROR, logger="services.worker.tts_downloads"):
            start(downloads)
            state = wait_terminal(downloads)
        assert state["status"] == "failed"
        assert state["error"] == "Model download failed"
        assert "type=RuntimeError" in caplog.text
        assert "PRIVATE" not in caplog.text
        assert "https://" not in caplog.text
    finally:
        downloads.close(wait=True)
