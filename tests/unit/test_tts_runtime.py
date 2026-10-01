from __future__ import annotations

import hashlib
from copy import deepcopy

import pytest

from packages.contracts.tts_catalog import manifest_fingerprint, resolve_bundle
from services.worker.generation import tts_runtime


def profile_entry(model_id="irodori-v4-large", precision="bf16"):
    bundle = resolve_bundle(model_id, precision)
    return {"provider_id": bundle["provider_id"], "model_id": model_id,
            "precision": precision, "manifest_id": bundle["manifest_id"], "bundle": bundle,
            "num_steps": 40, "codec_precision": "fp32"}


def test_frozen_profile_does_not_re_resolve_changed_catalog(monkeypatch):
    entry = profile_entry()
    monkeypatch.setattr(tts_runtime, "resolve_bundle", lambda *args: pytest.fail("catalog re-resolved"))
    profile = tts_runtime.resolve_request_profile({"tts_profile": entry, "num_steps": 40})
    assert profile.bundle == entry["bundle"]
    assert profile.bundle is not entry["bundle"]
    assert profile.compute_precision == "bf16"


@pytest.mark.parametrize("mutation", ["profile_id", "bundle_id", "revision", "destination",
                                     "precision", "steps", "codec"])
def test_rejects_inconsistent_frozen_profile(mutation):
    entry = profile_entry()
    if mutation == "profile_id":
        entry["manifest_id"] = "other"
    elif mutation == "bundle_id":
        entry["bundle"]["manifest_id"] = "other"
    elif mutation == "revision":
        entry["bundle"]["files"][0]["revision"] = "other"
    elif mutation == "destination":
        entry["bundle"]["files"][0]["local_dir"] = "outside/worker"
        entry["bundle"]["manifest_id"] = manifest_fingerprint(entry["bundle"])
        entry["manifest_id"] = entry["bundle"]["manifest_id"]
    elif mutation == "precision":
        entry["precision"] = "fp32"
    elif mutation == "steps":
        entry["num_steps"] = 20
    elif mutation == "codec":
        entry["codec_precision"] = "bf16"
    with pytest.raises(ValueError):
        tts_runtime.resolve_request_profile({"tts_profile": entry, "num_steps": 40})


def test_identity_includes_model_precision_source_dependencies_and_device_settings():
    original = tts_runtime.TTSRuntimeProfile(resolve_bundle("irodori-v4-large", "bf16"))
    fp32 = tts_runtime.TTSRuntimeProfile(resolve_bundle("irodori-v4-large", "fp32"))
    small = tts_runtime.TTSRuntimeProfile(resolve_bundle("irodori-v4.1-small", "bf16"))
    assert len({original.identity, fp32.identity, small.identity}) == 3
    changed = deepcopy(original.bundle)
    changed["files"][-1]["revision"] = "0" * 40
    assert tts_runtime.TTSRuntimeProfile(changed).identity != original.identity
    changed = deepcopy(original.bundle)
    changed["source_commit"] = "0" * 40
    assert tts_runtime.TTSRuntimeProfile(changed).identity != original.identity


def test_legacy_request_retains_small_and_requested_precision():
    profile = tts_runtime.resolve_request_profile({"model_precision": "fp32"})
    assert profile.bundle["model_id"] == "irodori-v4.1-small"
    assert profile.compute_precision == "fp32"


def test_verifies_sha256_and_git_blob_then_detects_corruption(tmp_path):
    payload = b"test weight"
    directory = "services/worker/runtimes/irodori/models/test"
    files = [
        {"local_dir": directory, "path": "model.safetensors", "size": len(payload),
         "sha256": hashlib.sha256(payload).hexdigest()},
        {"local_dir": directory, "path": "README.md", "size": len(payload),
         "git_blob_sha1": hashlib.sha1(f"blob {len(payload)}\0".encode() + payload).hexdigest()},
    ]
    profile = tts_runtime.TTSRuntimeProfile({"files": files})
    for file in files:
        path = profile.path(file, tmp_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    tts_runtime.verify_profile_files(profile, tmp_path)
    profile.path(files[1], tmp_path).write_bytes(b"bad payload")
    with pytest.raises(ValueError, match="integrity check failed"):
        tts_runtime.verify_profile_files(profile, tmp_path)
    profile.path(files[1], tmp_path).unlink()
    with pytest.raises(FileNotFoundError, match="download is incomplete"):
        tts_runtime.verify_profile_files(profile, tmp_path)


def test_destination_cannot_escape_worker_model_storage(tmp_path):
    profile = tts_runtime.TTSRuntimeProfile({})
    with pytest.raises(ValueError, match="inside worker model storage"):
        profile.path({"local_dir": "../outside", "path": "weights"}, tmp_path)
