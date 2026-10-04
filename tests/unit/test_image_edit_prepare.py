"""Verify snapshot pinning, corruption detection, cancellation and atomic publish."""

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.image_edit import prepare
from scripts.image_edit.common import MODEL_ID

REVISION = "a" * 40


def complete_model(directory: Path) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    index = {"_class_name": "QwenImage21Pipeline"}
    for component in prepare.COMPONENTS:
        (directory / component).mkdir(exist_ok=True)
        index[component] = prepare.COMPONENT_CLASSES[component]
    for name in prepare.REQUIRED_FILES:
        (directory / name).write_text("{}", encoding="utf-8")
    (directory / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
    for component in ("text_encoder", "transformer", "vae"):
        (directory / component / "model.safetensors").write_bytes(b"verified fixture weights")
    entries = [{"path": path.relative_to(directory).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
               for path in prepare.model_files(directory)]
    manifest = {"schema_version": 1, "model_id": MODEL_ID, "revision": REVISION,
                "gated": False, "snapshot_required_bytes": sum(x["size_bytes"] for x in entries),
                "files": entries}
    (directory / prepare.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    return manifest


def test_existing_model_rehashed_without_download_or_mutation(tmp_path, monkeypatch):
    model = tmp_path / "model"
    complete_model(model)
    before = {p.relative_to(model): p.read_bytes() for p in model.rglob("*") if p.is_file()}
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("download existing model"))
    result = prepare.prepare_model(model, tmp_path / "status")
    assert result["reused"] and result["revision"] == REVISION
    assert before == {p.relative_to(model): p.read_bytes() for p in model.rglob("*") if p.is_file()}


@pytest.mark.parametrize("change", [
    "digest", "missing_shard", "extra_file", "wrong_model", "escape", "component_class",
])
def test_invalid_model_is_preserved_and_rejected(tmp_path, change):
    model = tmp_path / "model"
    manifest = complete_model(model)
    if change == "digest":
        weights = model / "vae/model.safetensors"
        weights.write_bytes(b"x" * weights.stat().st_size)
    elif change == "missing_shard":
        (model / "transformer/model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {"layer": "missing.safetensors"}}), encoding="utf-8")
    elif change == "extra_file":
        (model / "unrecorded.json").write_text("{}", encoding="utf-8")
    elif change == "wrong_model":
        manifest["model_id"] = "wrong/model"
    elif change == "escape":
        manifest["files"][0]["path"] = "../keep"
    else:
        index = prepare.read_json(model / "model_index.json")
        index["vae"] = ["diffusers", "OtherModel"]
        prepare.write_json(model / "model_index.json", index)
    (model / prepare.MANIFEST).write_text(json.dumps(manifest), encoding="utf-8")
    before = {p.relative_to(model): p.read_bytes() for p in model.rglob("*") if p.is_file()}
    with pytest.raises(ValueError):
        prepare.prepare_model(model, tmp_path / "status")
    assert before == {p.relative_to(model): p.read_bytes() for p in model.rglob("*") if p.is_file()}


def test_unrecorded_existing_files_are_not_overwritten(tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "important.txt").write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError, match="既存"):
        prepare.prepare_model(model, tmp_path / "status")
    assert (model / "important.txt").read_text(encoding="utf-8") == "keep"


def test_snapshot_only_published_after_worker_validation(tmp_path, monkeypatch):
    output = tmp_path / "model"

    def download(job_path, status):
        assert not output.exists()
        job = prepare.read_json(job_path)
        complete_model(Path(job["stage_dir"]))
        assert job["revision"] == REVISION

    monkeypatch.setattr(prepare, "run_child", download)
    result = prepare.prepare_model(output, tmp_path / "status", REVISION)
    assert not result["reused"] and result["revision"] == REVISION
    prepare.validate_manifest(output, tmp_path / "status")
    assert not list(tmp_path.glob(".model.prepare-*"))


def test_failed_download_removes_only_its_owned_stage(tmp_path, monkeypatch):
    keep = tmp_path / "keep"
    keep.mkdir()
    (keep / "important").write_bytes(b"keep")
    monkeypatch.setattr(prepare, "run_child", lambda *args: None)
    with pytest.raises(ValueError, match="model_index"):
        prepare.prepare_model(tmp_path / "model", tmp_path / "status")
    assert not (tmp_path / "model").exists()
    assert not list(tmp_path.glob(".model.prepare-*"))
    assert (keep / "important").read_bytes() == b"keep"


def test_cancelled_cli_completes_gui_result(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.mkdir()
    (status / "stop.request").touch()
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("download after stop"))
    assert prepare.main(["--model-dir", str(tmp_path / "model"), "--status-dir", str(status)]) == 0
    assert prepare.read_json(status / "result.json") == {
        "ok": False, "cancelled": True, "error": "Qwen Image 2.1のモデル準備を中止しました。"}
    assert prepare.read_json(status / "status.json")["phase"] == "cancelled"


def test_cancellation_reaps_only_the_created_download_child(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.mkdir()
    (status / "stop.request").touch()
    job = tmp_path / "job.json"
    prepare.write_json(job, {"stage_dir": str(tmp_path / "stage")})

    class Child:
        returncode = None
        terminated = False
        waited = False

        def poll(self):
            return 1 if self.terminated else None

        def terminate(self):
            self.terminated = True

        def wait(self, timeout):
            self.waited = True

    child = Child()
    monkeypatch.setattr(prepare.subprocess, "Popen", lambda *args, **kwargs: child)
    with pytest.raises(prepare.PreparationCancelled):
        prepare.run_child(job, status)
    assert child.terminated and child.waited


def test_worker_resolves_revision_then_pins_every_download(tmp_path, monkeypatch):
    fixture = tmp_path / "fixture"
    complete_model(fixture)
    stage = tmp_path / "stage"
    stage.mkdir()
    status = tmp_path / "status"
    job = tmp_path / "job.json"
    prepare.write_json(job, {"stage_dir": str(stage), "status_dir": str(status), "revision": "requested-tag"})
    siblings = [SimpleNamespace(rfilename=p.relative_to(fixture).as_posix(),
                               size=p.stat().st_size, lfs=None) for p in prepare.model_files(fixture)]
    calls = []

    class Api:
        def model_info(self, model_id, **kwargs):
            calls.append(("info", model_id, kwargs))
            return SimpleNamespace(sha=REVISION, siblings=siblings, gated=False)

    def snapshot(model_id, **kwargs):
        calls.append(("snapshot", model_id, kwargs))
        for source in prepare.model_files(fixture):
            destination = stage / source.relative_to(fixture)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(HfApi=Api, snapshot_download=snapshot))
    prepare.download_worker(job)
    assert calls[0][2]["revision"] == "requested-tag"
    assert calls[1][2]["revision"] == REVISION
    manifest = prepare.validate_manifest(stage, status)
    assert manifest["gated"] is False
    assert manifest["snapshot_required_bytes"] == sum(x["size_bytes"] for x in manifest["files"])


def test_access_error_is_actionable_and_credentials_are_redacted(monkeypatch):
    monkeypatch.setenv("HF_TOKEN", "secret-fixture-token")
    error = prepare.safe_error(RuntimeError("403 gated: secret-fixture-token and hf_abc123"))
    assert "hf auth login" in error and f"https://huggingface.co/{MODEL_ID}" in error
    assert "secret-fixture-token" not in error and "hf_abc123" not in error
