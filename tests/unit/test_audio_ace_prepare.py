"""ACE checkpoints must be pinned, planner-free and published only after validation."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.audio import ace_prepare as prepare
from scripts.audio import prepare as dispatcher


def reference_config():
    return {
        "is_turbo": True, "model_version": "turbo", "hidden_size": 2048,
        "intermediate_size": 6144, "num_hidden_layers": 24,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "in_channels": 192, "audio_acoustic_hidden_dim": 64,
        "text_hidden_dim": 1024, "timbre_hidden_dim": 64,
    }


def manifest():
    return {
        "model": prepare.MODEL_KEY, "repo_id": prepare.REPO_ID,
        "repo_revision": prepare.REPO_REVISION, "dit_config": prepare.DIT_CONFIG,
        "converter_sha256": prepare.CONVERTER_SHA256,
        "diffusers_version": prepare.DIFFUSERS_VERSION, "dtype": "bfloat16",
        "planner_enabled": False, "planner_downloaded": False,
        "silence_latent_verified": True,
    }


def complete_model(directory):
    index = {"_class_name": "AceStepPipeline"}
    for component in prepare.COMPONENTS:
        folder = directory / component
        folder.mkdir(parents=True)
        index[component] = ["diffusers", "Model"]
        if component not in {"tokenizer", "scheduler"}:
            (folder / "config.json").write_text("{}", encoding="utf-8")
            (folder / "model.safetensors").write_bytes(b"fixture weights")
    (directory / "transformer/config.json").write_text(json.dumps(reference_config()), encoding="utf-8")
    (directory / "scheduler/scheduler_config.json").write_text(
        '{"num_train_timesteps":1,"shift":1.0}', encoding="utf-8")
    (directory / "tokenizer/tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer/tokenizer.json").write_text("{}", encoding="utf-8")
    (directory / "model_index.json").write_text(json.dumps(index), encoding="utf-8")
    (directory / "ace_preparation.json").write_text(json.dumps(manifest()), encoding="utf-8")


@pytest.mark.parametrize("field,value", [
    ("is_turbo", False), ("hidden_size", 2560), ("num_hidden_layers", 48),
    ("model_version", "base"), ("text_hidden_dim", 2048),
])
def test_only_original_15_turbo_architecture_is_accepted(field, value):
    config = reference_config()
    prepare.validate_reference_config(config)
    config[field] = value
    with pytest.raises(ValueError, match="2B"):
        prepare.validate_reference_config(config)


def test_downloads_only_pinned_required_files_and_never_lm(tmp_path, monkeypatch):
    snapshot = tmp_path / "snapshot"
    for name in prepare.SOURCE_FILES:
        path = snapshot / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reference_config()) if name == "acestep-v15-turbo/config.json" else "fixture")
    calls = []

    def download(repo, name, **kwargs):
        calls.append(("file", repo, name, kwargs))
        return snapshot / name

    def download_all(repo, **kwargs):
        calls.append(("snapshot", repo, kwargs))
        return snapshot

    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=download, snapshot_download=download_all,
    ))
    path, config = prepare.download_checkpoint(tmp_path / "cache", tmp_path / "status")
    assert path == snapshot and config == reference_config()
    assert calls[0][3]["revision"] == prepare.REPO_REVISION
    assert calls[0][3]["token"] is False
    assert calls[1][2]["revision"] == prepare.REPO_REVISION
    assert calls[1][2]["token"] is False
    assert calls[1][2]["allow_patterns"] == list(prepare.SOURCE_FILES)
    assert all("lm" not in name.lower() and "*" not in name for name in prepare.SOURCE_FILES)
    assert {Path(name).parts[0] for name in prepare.SOURCE_FILES} == {
        "acestep-v15-turbo", "vae", "Qwen3-Embedding-0.6B",
    }


def test_wrong_checkpoint_fails_before_weight_download(tmp_path, monkeypatch):
    config = tmp_path / "config.json"
    config.write_text('{"hidden_size":2560}', encoding="utf-8")
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(
        hf_hub_download=lambda *args, **kwargs: config,
        snapshot_download=lambda *args, **kwargs: pytest.fail("weights must not download"),
    ))
    with pytest.raises(ValueError, match="2B"):
        prepare.download_checkpoint(tmp_path / "cache", tmp_path / "status")


@pytest.mark.parametrize("defect", ["missing_codec", "zero_weights", "wrong_schedule", "missing_shard", "wrong_model"])
def test_partial_or_wrong_pipeline_is_rejected(tmp_path, defect):
    complete_model(tmp_path)
    assert prepare.is_complete_model(tmp_path)
    if defect == "missing_codec":
        (tmp_path / "audio_tokenizer/model.safetensors").unlink()
    elif defect == "zero_weights":
        (tmp_path / "transformer/model.safetensors").write_bytes(b"")
    elif defect == "wrong_schedule":
        (tmp_path / "scheduler/scheduler_config.json").write_text('{"num_train_timesteps":1000,"shift":1.0}')
    elif defect == "missing_shard":
        (tmp_path / "transformer/model.safetensors.index.json").write_text(
            '{"weight_map":{"weight":"missing.safetensors"}}')
    else:
        config = reference_config()
        config["hidden_size"] = 2560
        (tmp_path / "transformer/config.json").write_text(json.dumps(config))
    assert not prepare.is_complete_model(tmp_path)


def test_existing_verified_model_is_reused_unchanged(tmp_path, monkeypatch):
    output = tmp_path / "ace15_turbo"
    complete_model(output)
    previous = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("existing model must not convert"))
    assert prepare.prepare_model("ace15_turbo", output, tmp_path / "cache", tmp_path / "status") == output
    assert previous == {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}


@pytest.mark.parametrize("field,value", [("model", "medium"), ("repo_revision", "main"), ("planner_enabled", True)])
def test_existing_unverified_manifest_is_preserved(tmp_path, field, value):
    output = tmp_path / "ace15_turbo"
    complete_model(output)
    changed = manifest()
    changed[field] = value
    path = output / "ace_preparation.json"
    path.write_text(json.dumps(changed))
    previous = path.read_bytes()
    with pytest.raises(ValueError, match="生成記録"):
        prepare.prepare_model("ace15_turbo", output, tmp_path / "cache", tmp_path / "status")
    assert path.read_bytes() == previous


def test_existing_partial_folder_is_never_overwritten(tmp_path):
    output = tmp_path / "ace15_turbo"
    output.mkdir()
    (output / "keep.txt").write_text("keep")
    with pytest.raises(ValueError, match="既存"):
        prepare.prepare_model("ace15_turbo", output, tmp_path / "cache", tmp_path / "status")
    assert (output / "keep.txt").read_text() == "keep"


def test_atomic_publication_after_complete_conversion(tmp_path, monkeypatch):
    output = tmp_path / "ace15_turbo"

    def convert(job_path, status):
        assert not output.exists()
        job = json.loads(job_path.read_text(encoding="utf-8"))
        assert job["dtype"] == "bfloat16"
        complete_model(Path(job["stage_dir"]))

    monkeypatch.setattr(prepare, "run_child", convert)
    prepare.prepare_model("ace15_turbo", output, tmp_path / "cache", tmp_path / "status")
    assert prepare.is_complete_model(output)
    assert not list(tmp_path.glob(".ace15_turbo.prepare-*"))


def test_failure_cleans_only_owned_stage(tmp_path, monkeypatch):
    unrelated = tmp_path / "other-model"
    unrelated.mkdir()
    (unrelated / "keep.txt").write_text("keep")
    monkeypatch.setattr(prepare, "run_child", lambda *args: None)
    with pytest.raises(RuntimeError, match="検証"):
        prepare.prepare_model("ace15_turbo", tmp_path / "ace15_turbo", tmp_path / "cache", tmp_path / "status")
    assert not list(tmp_path.glob(".ace15_turbo.prepare-*"))
    assert (unrelated / "keep.txt").read_text() == "keep"


def test_output_changed_during_conversion_is_preserved(tmp_path, monkeypatch):
    output = tmp_path / "ace15_turbo"

    def convert(job_path, status):
        complete_model(Path(json.loads(job_path.read_text())["stage_dir"]))
        output.mkdir()
        (output / "keep.txt").write_text("keep")

    monkeypatch.setattr(prepare, "run_child", convert)
    with pytest.raises(ValueError, match="変更"):
        prepare.prepare_model("ace15_turbo", output, tmp_path / "cache", tmp_path / "status")
    assert (output / "keep.txt").read_text() == "keep"


def test_cancellation_never_starts_conversion(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.mkdir()
    (status / "stop.request").touch()
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("cancelled conversion"))
    assert prepare.main(["--output-dir", str(tmp_path / "ace15_turbo"), "--cache-dir", str(tmp_path / "cache"),
                         "--status-dir", str(status)]) == 0
    result = json.loads((status / "result.json").read_text(encoding="utf-8"))
    assert result["cancelled"] and not result["ok"]


def test_cancellation_reaps_child(tmp_path, monkeypatch):
    (tmp_path / "stop.request").touch()

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
        prepare.run_child(tmp_path / "job.json", tmp_path)
    assert child.terminated and child.waited


def test_direct_stable_prepare_script_reports_ace_cancellation(tmp_path):
    status = tmp_path / "status"
    status.mkdir()
    (status / "stop.request").touch()
    completed = subprocess.run([
        sys.executable, str(Path(dispatcher.__file__).resolve()), "--model", "ace15_turbo",
        "--output-dir", str(tmp_path / "ace15_turbo"), "--cache-dir", str(tmp_path / "cache"),
        "--status-dir", str(status), "--dtype", "bfloat16",
    ], capture_output=True, check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    assert completed.returncode == 0, completed.stderr.decode(errors="replace")
    result = json.loads((status / "result.json").read_text(encoding="utf-8"))
    assert result["cancelled"] and not result["ok"]


def test_unverified_converter_cannot_execute(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            return b"changed upstream source"

    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="チェックサム"):
        prepare.fetch_converter(tmp_path)
    assert not list(tmp_path.rglob("*.py"))


def test_verified_converter_is_usable_offline(tmp_path, monkeypatch):
    source = b"reviewed fixture"
    monkeypatch.setattr(prepare, "CONVERTER_SHA256", hashlib.sha256(source).hexdigest())
    directory = tmp_path / "converter"
    directory.mkdir()
    path = directory / f"convert_ace_step_{prepare.CONVERTER_REVISION}.py"
    path.write_bytes(source)
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network"))
    assert prepare.fetch_converter(tmp_path) == path


def test_converter_uses_only_local_checkpoint_and_records_no_planner(tmp_path, monkeypatch):
    converter = tmp_path / "converter.py"
    converter.write_text(
        "import os\n"
        "def convert_ace_step_weights(**kwargs):\n"
        "    assert os.environ['HF_HUB_OFFLINE'] == '1'\n"
        "    assert os.environ['TRANSFORMERS_OFFLINE'] == '1'\n"
        "    from scripts.audio.ace_prepare import write_json\n"
        "    from pathlib import Path\n"
        "    write_json(Path(kwargs['output_dir']) / 'arguments.json', kwargs)\n"
    )
    output = tmp_path / "stage"
    complete_model(output)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(__version__="0.40.0"))
    monkeypatch.setattr(prepare, "fetch_converter", lambda cache: converter)
    monkeypatch.setattr(prepare, "download_checkpoint", lambda cache, status: (checkpoint, reference_config()))
    checked = []
    monkeypatch.setattr(prepare, "validate_silence_buffer", lambda *args: checked.append(args))
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "old")
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"status_dir": str(tmp_path / "status"), "cache_dir": str(tmp_path / "cache"),
                               "stage_dir": str(output), "dtype": "bfloat16"}))
    prepare.conversion_worker(job)
    arguments = json.loads((output / "arguments.json").read_text())
    assert arguments == {"checkpoint_dir": str(checkpoint), "dit_config": "acestep-v15-turbo",
                         "output_dir": str(output), "dtype_str": "bf16"}
    saved = json.loads((output / "ace_preparation.json").read_text())
    assert saved["planner_enabled"] is False and saved["planner_downloaded"] is False
    assert saved["repo_revision"] == prepare.REPO_REVISION
    assert saved["source_files"] == list(prepare.SOURCE_FILES)
    assert checked == [(output, checkpoint, "bfloat16")]
    assert "HF_HUB_OFFLINE" not in prepare.os.environ
    assert prepare.os.environ["TRANSFORMERS_OFFLINE"] == "old"


def test_existing_prepare_cli_dispatches_ace_without_stable_conversion(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(prepare, "prepare_model", lambda *args: calls.append(args) or tmp_path / "model")
    assert dispatcher.main(["--model", "ace15_turbo", "--output-dir", str(tmp_path / "model"),
                            "--cache-dir", str(tmp_path / "cache"), "--status-dir", str(tmp_path / "status"),
                            "--dtype", "bfloat16"]) == 0
    assert calls == [("ace15_turbo", tmp_path / "model", tmp_path / "cache", tmp_path / "status", "bfloat16")]


@pytest.mark.parametrize("defect", [None, "missing", "zeros", "transpose"])
def test_silence_buffer_must_match_original_transpose(tmp_path, defect):
    torch = pytest.importorskip("torch")
    safetensors = pytest.importorskip("safetensors.torch")
    checkpoint = tmp_path / "checkpoint" / prepare.DIT_CONFIG
    checkpoint.mkdir(parents=True)
    source = torch.ones((1, 64, 15000), dtype=torch.float32) * 0.17
    torch.save(source, checkpoint / "silence_latent.pt")
    output = tmp_path / "output" / "condition_encoder"
    output.mkdir(parents=True)
    expected = source.transpose(1, 2).to(torch.bfloat16).contiguous()
    if defect == "missing":
        weights = {"other": torch.ones(1)}
    elif defect == "zeros":
        weights = {"silence_latent": torch.zeros_like(expected)}
    elif defect == "transpose":
        weights = {"silence_latent": source.to(torch.bfloat16)}
    else:
        weights = {"silence_latent": expected}
    safetensors.save_file(weights, str(output / "model.safetensors"))
    if defect is None:
        prepare.validate_silence_buffer(output.parent, checkpoint.parent, "bfloat16")
    else:
        with pytest.raises(ValueError, match="無音潜在表現"):
            prepare.validate_silence_buffer(output.parent, checkpoint.parent, "bfloat16")
