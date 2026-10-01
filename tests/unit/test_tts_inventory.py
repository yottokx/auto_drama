import copy
import json

import pytest

from services.worker import tts_inventory
from services.worker.tts_downloads import _identity


@pytest.mark.parametrize(("compute", "torchao", "expected"), [
    ("12.0", "0.16.0", ["fp32", "bf16", "int8", "int4"]),
    ("12.0", None, ["fp32", "bf16"]),
    ("7.5", "0.16.0", ["fp32"]),
])
def test_runtime_support_checks_the_actual_device_without_loading_torch(tmp_path, monkeypatch, compute, torchao, expected):
    python = tmp_path / "services/worker/runtimes/irodori/.venv/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.touch()
    versions = {"torch": "2.10.0+cu128", "torchao": torchao}
    commands = []

    def command(args, **_kwargs):
        commands.append(args)
        if args[0] == "git":
            return "a" * 40
        if args[0] == "nvidia-smi":
            return compute + "\n8.0"
        return json.dumps(versions)

    monkeypatch.setattr(tts_inventory, "_command", command)
    tts_inventory._runtime_cache.clear()
    assert tts_inventory.runtime_support(tmp_path)["precisions"] == expected
    assert "import torch" not in commands[1][-1]
    assert tts_inventory.runtime_support(tmp_path)["precisions"] == expected
    assert len(commands) == 3


def test_verified_receipts_are_required_and_changed_files_drop_readiness(tmp_path, monkeypatch):
    directory = "services/worker/runtimes/irodori/models/test"
    file = {"local_dir": directory, "path": "model.safetensors", "size": 7,
            "repo_id": "test/tts", "revision": "a" * 40, "sha256": "b" * 64}
    path = tmp_path / directory / file["path"]
    path.parent.mkdir(parents=True)
    path.write_bytes(b"weights")
    bundle = {"files": [file], "manifest_id": "fixed", "source_commit": "a" * 40}
    monkeypatch.setattr(tts_inventory, "catalog", lambda: [{"model_id": "model", "precisions": ["fp32", "bf16"]}])
    monkeypatch.setattr(tts_inventory, "resolve_bundle", lambda *_: copy.deepcopy(bundle))
    assert tts_inventory.downloaded_inventory(tmp_path) == []
    receipts = tmp_path / "services/worker/runtimes/irodori/models/.tts-downloads/verified-files.json"
    receipts.parent.mkdir()
    receipts.write_text(json.dumps({str(path.relative_to(tmp_path)): {
        "identity": _identity(file), "mtime_ns": path.stat().st_mtime_ns}}))
    downloaded = tts_inventory.downloaded_inventory(tmp_path)
    assert {item["precision"] for item in downloaded} == {"fp32", "bf16"}
    monkeypatch.setattr(tts_inventory, "runtime_support", lambda _: {
        "ready": True, "precisions": ["bf16"], "source_commit": "a" * 40})
    ready = tts_inventory.generation_inventory(tmp_path, downloaded)
    assert ready[0]["generation_ready"] is False
    assert ready[1]["generation_purposes"] == ["voice_design", "voice_clone"]
    assert "generation_ready" not in downloaded[0]
    bundle["source_commit"] = "c" * 40
    assert not any(item["generation_ready"] for item in tts_inventory.generation_inventory(tmp_path, downloaded))
    path.write_bytes(b"changed-weights")
    assert tts_inventory.downloaded_inventory(tmp_path) == []
