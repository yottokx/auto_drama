"""The face detector is pinned like the other models and is never fetched by the script that uses it."""

import hashlib
import json

import pytest

from scripts.m0 import face_detect


def manifest(tmp_path, monkeypatch, content=b"weights", *, size=None, digest=None):
    directory = tmp_path / "models"
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"models": [
        {"id": "isnet-anime", "local_dir": "elsewhere", "files": []},
        {"id": face_detect.MODEL_ID, "local_dir": "models", "files": [{
            "path": "detector/model.onnx", "size": len(content) if size is None else size,
            "sha256": digest or hashlib.sha256(content).hexdigest()}]}]}), encoding="utf-8")
    monkeypatch.setattr(face_detect, "MANIFEST", path)
    monkeypatch.setattr(face_detect, "ROOT", tmp_path)
    return directory / "detector" / "model.onnx"


def test_the_pinned_weights_are_required_and_checked(tmp_path, monkeypatch):
    weights = manifest(tmp_path, monkeypatch)
    with pytest.raises(FileNotFoundError, match="Download the face detection model"):
        face_detect.model_path()
    weights.parent.mkdir(parents=True)
    weights.write_bytes(b"weights")
    assert face_detect.model_path() == weights
    weights.write_bytes(b"changed")
    with pytest.raises(ValueError, match="pinned manifest"):
        face_detect.model_path()
    manifest(tmp_path, monkeypatch, size=3)
    with pytest.raises(FileNotFoundError):
        face_detect.model_path()


def test_the_repository_manifest_pins_the_detector():
    entry = next(row for row in json.loads(face_detect.MANIFEST.read_text(encoding="utf-8-sig"))["models"]
                 if row["id"] == face_detect.MODEL_ID)
    (item,) = entry["files"]
    assert entry["repo_id"] and len(entry["revision"]) == 40 and len(item["sha256"]) == 64
    assert entry["local_dir"].startswith("services/worker/runtimes/diffusers/models/")
