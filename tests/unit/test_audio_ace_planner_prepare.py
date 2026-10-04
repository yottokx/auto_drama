"""Public planner preparation is pinned, allowlisted and atomically published."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.audio import ace_planner_prepare as prepare


def config():
    return {"model_type": "qwen3", "hidden_size": 2048, "intermediate_size": 6144,
            "num_hidden_layers": 28, "num_attention_heads": 16, "num_key_value_heads": 8,
            "head_dim": 128, "vocab_size": 217204, "tie_word_embeddings": True}


def complete_model(path):
    path.mkdir(parents=True, exist_ok=True)
    for name in prepare.MODEL_FILES:
        (path / name).write_text(json.dumps(config()) if name == "config.json" else "fixture", encoding="utf-8")
    (path / "ace_planner_preparation.json").write_text(json.dumps({
        "model": prepare.MODEL_KEY, "repo_id": prepare.REPO_ID, "repo_revision": prepare.REPO_REVISION,
        "model_subfolder": prepare.MODEL_SUBFOLDER, "remote_code": False,
    }), encoding="utf-8")


@pytest.mark.parametrize("field,value", [
    ("model_type", "gemma"), ("hidden_size", 1024), ("num_hidden_layers", 36),
    ("vocab_size", 151669), ("auto_map", {"AutoModel": "remote.Module"}),
])
def test_wrong_or_remote_code_architectures_are_rejected(field, value):
    actual = config()
    actual[field] = value
    with pytest.raises(ValueError, match="1.7B"):
        prepare.validate_config(actual)


def test_download_is_pinned_allowlisted_public_and_validated_before_weights(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    complete_model(snapshot / prepare.MODEL_SUBFOLDER)
    calls = []

    def one(repo, name, **kwargs):
        calls.append((repo, name, kwargs))
        return snapshot / name

    def all_files(repo, **kwargs):
        calls.append((repo, kwargs))
        return snapshot

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(hf_hub_download=one, snapshot_download=all_files))
    assert prepare.download_checkpoint(tmp_path / "cache", tmp_path / "status") == snapshot / prepare.MODEL_SUBFOLDER
    assert calls[0][2]["revision"] == prepare.REPO_REVISION and calls[0][2]["token"] is False
    assert calls[1][1]["revision"] == prepare.REPO_REVISION and calls[1][1]["token"] is False
    assert calls[1][1]["allow_patterns"] == list(prepare.SOURCE_FILES)
    assert len(prepare.SOURCE_FILES) == 9 and not any(name.endswith(".py") or "*" in name for name in prepare.SOURCE_FILES)


def test_wrong_config_never_downloads_weights(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text('{"model_type":"remote"}')
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=lambda *args, **kwargs: path,
        snapshot_download=lambda *args, **kwargs: pytest.fail("invalid architecture"),
    ))
    with pytest.raises(ValueError, match="Qwen3"):
        prepare.download_checkpoint(tmp_path / "cache", tmp_path / "status")


@pytest.mark.parametrize("defect", ["missing_weight", "zero_weight", "wrong_revision", "remote_code"])
def test_incomplete_or_unverified_models_are_rejected(tmp_path, defect):
    complete_model(tmp_path)
    assert prepare.validate_model(tmp_path)["model"] == prepare.MODEL_KEY
    if defect == "missing_weight":
        (tmp_path / "model.safetensors").unlink()
    elif defect == "zero_weight":
        (tmp_path / "model.safetensors").write_bytes(b"")
    else:
        path = tmp_path / "ace_planner_preparation.json"
        manifest = json.loads(path.read_text())
        manifest["repo_revision" if defect == "wrong_revision" else "remote_code"] = "main" if defect == "wrong_revision" else True
        path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        prepare.validate_model(tmp_path)


def test_existing_verified_model_is_reused_without_download(tmp_path, monkeypatch):
    output = tmp_path / "model"
    complete_model(output)
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("reuse cannot download"))
    assert prepare.prepare_model(output, tmp_path / "cache", tmp_path / "status") == output


def test_existing_unrelated_or_incomplete_files_are_preserved(tmp_path, monkeypatch):
    output = tmp_path / "model"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("existing contents"))
    with pytest.raises(ValueError):
        prepare.prepare_model(output, tmp_path / "cache", tmp_path / "status")
    assert (output / "keep.txt").read_text() == "keep"


def test_publish_only_after_complete_validation_and_clean_owned_stage(tmp_path, monkeypatch):
    output = tmp_path / "model"

    def child(job_path, _status):
        assert not output.exists()
        complete_model(Path(json.loads(job_path.read_text())["stage_dir"]))

    monkeypatch.setattr(prepare, "run_child", child)
    assert prepare.prepare_model(output, tmp_path / "cache", tmp_path / "status") == output
    assert prepare.validate_model(output)["remote_code"] is False
    assert not list(tmp_path.glob(".model.prepare-*"))


def test_partial_child_does_not_publish_and_unrelated_models_survive(tmp_path, monkeypatch):
    unrelated = tmp_path / "other-model"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep")
    monkeypatch.setattr(prepare, "run_child", lambda *args: None)
    with pytest.raises(ValueError):
        prepare.prepare_model(tmp_path / "model", tmp_path / "cache", tmp_path / "status")
    assert not (tmp_path / "model").exists() and not list(tmp_path.glob(".model.prepare-*"))
    assert (unrelated / "keep.txt").read_text() == "keep"


def test_output_changed_during_download_is_never_overwritten(tmp_path, monkeypatch):
    output = tmp_path / "model"

    def child(job_path, _status):
        complete_model(Path(json.loads(job_path.read_text())["stage_dir"]))
        output.mkdir()
        (output / "keep.txt").write_text("keep")

    monkeypatch.setattr(prepare, "run_child", child)
    with pytest.raises(ValueError, match="変更"):
        prepare.prepare_model(output, tmp_path / "cache", tmp_path / "status")
    assert (output / "keep.txt").read_text() == "keep"


def test_stop_request_reports_cancelled_and_never_downloads(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.mkdir()
    (status / "stop.request").touch()
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("cancelled"))
    assert prepare.main(["--output-dir", str(tmp_path / "model"), "--status-dir", str(status)]) == 130
    result = json.loads((status / "result.json").read_text(encoding="utf-8"))
    assert result["cancelled"] and not result["ok"]
