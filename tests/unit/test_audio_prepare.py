import hashlib
import json
from pathlib import Path

import pytest

from scripts.audio import prepare
from scripts.audio.same_compat import same_config_fields


def reference_config(model="small"):
    encoder = {"strides": [16], "variable_stride": True,
               "chunk_size": 32, "chunk_midpoint_shift": True}
    decoder = {**encoder, "conv_mapping": True, "mask_noise": 0.01}
    if model == "medium":
        encoder.update(sliding_window=[1, 1], chunk_midpoint_shift=False)
        decoder.update(sliding_window=[1, 1], chunk_midpoint_shift=False, conv_mapping=False)
    return {"sample_rate": 44100, "audio_channels": 2, "model": {
        "pretransform": {"config": {"encoder": {"config": encoder}, "decoder": {"config": decoder}}},
        "diffusion": {"diffusion_objective": "rf_denoiser", "config": {
            "embed_dim": 1024 if model == "small" else 1536,
            "depth": 20 if model == "small" else 24,
            "attn_kwargs": {"differential": model == "medium"},
        }},
    }}


def complete_model(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    index = {"_class_name": "StableAudio3Pipeline"}
    for component in prepare.COMPONENTS:
        folder = directory / component
        folder.mkdir()
        index[component] = ["diffusers", "Model"]
        (folder / "config.json").write_text("{}", encoding="utf-8")
        if component in ("vae", "duration_embedder", "transformer", "text_encoder"):
            (folder / "model.safetensors").write_bytes(b"test weights")
    (directory / "scheduler" / "scheduler_config.json").write_text("{}", encoding="utf-8")
    (directory / "tokenizer" / "tokenizer_config.json").write_text("{}", encoding="utf-8")
    (directory / "model_index.json").write_text(json.dumps(index), encoding="utf-8")


@pytest.mark.parametrize("model", ["small", "medium"])
def test_reference_architecture_is_validated_and_preserved(model):
    config = reference_config(model)
    prepare._validate_reference_config(config, model)
    fields = same_config_fields(config)
    assert fields["decoder_chunk_size"] == (32 if model == "small" else 0)
    assert fields["decoder_mapping_kernel"] == (3 if model == "small" else 1)
    assert fields["decoder_mask_noise"] == 0.01
    config["model"]["diffusion"]["diffusion_objective"] = "rectified_flow"
    with pytest.raises(ValueError, match="post-trained"):
        prepare._validate_reference_config(config, model)


def test_missing_original_small_architecture_is_rejected():
    config = reference_config()
    config["model"]["pretransform"]["config"]["decoder"]["config"]["conv_mapping"] = False
    with pytest.raises(ValueError, match="Small SAME"):
        prepare._validate_reference_config(config, "small")


def test_model_revision_is_taken_from_configuration_snapshot():
    revision = "1234567890abcdef" * 2 + "12345678"
    config = Path("cache") / "snapshots" / revision / "model_config.json"
    assert prepare.cached_repo_revision(config) == revision
    with pytest.raises(ValueError, match="リビジョン"):
        prepare.cached_repo_revision(Path("models") / "model_config.json")
    with pytest.raises(ValueError, match="リビジョン"):
        prepare.cached_repo_revision(Path("cache") / "snapshots" / "main" / "model_config.json")


def test_invalid_shift_stride_is_rejected_before_conversion():
    config = reference_config()
    config["model"]["pretransform"]["config"]["decoder"]["config"]["chunk_size"] = 31
    with pytest.raises(ValueError, match="divisible"):
        same_config_fields(config)


def test_existing_model_is_reused_without_download_or_modification(tmp_path, monkeypatch):
    output = tmp_path / "small"
    complete_model(output)
    contents = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("should not start conversion"))
    assert prepare.prepare_model("small", output, tmp_path / "cache", tmp_path / "status") == output
    assert contents == {str(path.relative_to(output)): path.read_bytes() for path in output.rglob("*") if path.is_file()}


def test_wrong_existing_model_is_not_relabelled(tmp_path):
    output = tmp_path / "model"
    complete_model(output)
    (output / "sa3_preparation.json").write_text('{"model": "medium"}', encoding="utf-8")
    with pytest.raises(ValueError, match="一致"):
        prepare.prepare_model("small", output, tmp_path / "cache", tmp_path / "status")


def test_partial_existing_model_is_preserved(tmp_path):
    output = tmp_path / "small"
    output.mkdir()
    (output / "model_index.json").write_text('{"_class_name": "StableAudio3Pipeline"}', encoding="utf-8")
    with pytest.raises(ValueError, match="既存"):
        prepare.prepare_model("small", output, tmp_path / "cache", tmp_path / "status")
    assert (output / "model_index.json").exists()


def test_conversion_is_published_only_after_validation(tmp_path, monkeypatch):
    output = tmp_path / "small"

    def convert(job_path, status):
        assert not output.exists()
        job = json.loads(job_path.read_text(encoding="utf-8"))
        complete_model(Path(job["stage_dir"]))

    monkeypatch.setattr(prepare, "run_child", convert)
    prepare.prepare_model("small", output, tmp_path / "cache", tmp_path / "status")
    assert prepare.is_complete_model(output)
    assert not list(tmp_path.glob(".small.prepare-*"))


def test_failed_conversion_removes_only_owned_temporary_folder(tmp_path, monkeypatch):
    unrelated = tmp_path / "keep"
    unrelated.mkdir()
    (unrelated / "important").write_text("keep", encoding="utf-8")
    output = tmp_path / "small"
    monkeypatch.setattr(prepare, "run_child", lambda *args: None)
    with pytest.raises(RuntimeError, match="検証"):
        prepare.prepare_model("small", output, tmp_path / "cache", tmp_path / "status")
    assert not output.exists()
    assert not list(tmp_path.glob(".small.prepare-*"))
    assert (unrelated / "important").read_text(encoding="utf-8") == "keep"


def test_cancelled_cli_writes_result_without_starting_child(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.mkdir()
    (status / "stop.request").touch()
    monkeypatch.setattr(prepare, "run_child", lambda *args: pytest.fail("conversion after cancellation"))
    assert prepare.main(["--model", "small", "--output-dir", str(tmp_path / "model"),
                         "--cache-dir", str(tmp_path / "cache"), "--status-dir", str(status)]) == 0
    result = json.loads((status / "result.json").read_text(encoding="utf-8"))
    assert result["cancelled"] and not result["ok"]
    assert json.loads((status / "status.json").read_text(encoding="utf-8"))["phase"] == "cancelled"


def test_unverified_converter_is_never_executed_or_cached(tmp_path, monkeypatch):
    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, count):
            return b"unexpected code"

    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="チェックサム"):
        prepare.fetch_converter(tmp_path)
    assert not list(tmp_path.rglob("*.py"))


def test_verified_converter_cache_is_used_offline(tmp_path, monkeypatch):
    source = b"reviewed test fixture"
    monkeypatch.setattr(prepare, "CONVERTER_SHA256", hashlib.sha256(source).hexdigest())
    directory = tmp_path / "converter"
    directory.mkdir()
    path = directory / f"convert_stable_audio_3_{prepare.CONVERTER_VERSION}.py"
    path.write_bytes(source)
    monkeypatch.setattr(prepare.urllib.request, "urlopen", lambda *args, **kwargs: pytest.fail("network"))
    assert prepare.fetch_converter(tmp_path) == path


def test_converter_child_is_reaped_on_cancellation(tmp_path, monkeypatch):
    status = tmp_path / "status"
    status.mkdir()
    (status / "stop.request").touch()

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
        prepare.run_child(tmp_path / "job.json", status)
    assert child.terminated and child.waited


def small_native_vae():
    pytest.importorskip("torch")
    diffusers = pytest.importorskip("diffusers")
    return diffusers.AutoencoderSAME(
        patch_size=2, encoder_channels=4, encoder_c_mults=(2,), encoder_strides=(2,),
        encoder_transformer_depths=(1,), latent_dim=4, dim_heads=4,
    )


def test_same_derived_rotary_buffers_are_validated_before_strict_loading():
    model = small_native_vae()
    weights = model.state_dict()
    for side in ("encoder", "decoder"):
        layer = getattr(model, side).blocks[0].transformers[0]
        weights[f"{side}.blocks.0.transformers.0.rope.inv_freq"] = layer.attn.rope.inv_freq.clone()
    converted = prepare._validated_same_state_dict(model, weights)
    model.load_state_dict(converted, strict=True)
    assert set(converted) == set(model.state_dict())
    assert len(weights) == len(converted) + 2


@pytest.mark.parametrize("change", ["frequencies", "shape"])
def test_same_incompatible_rotary_buffer_is_rejected(change):
    model = small_native_vae()
    key = "decoder.blocks.0.transformers.0.rope.inv_freq"
    value = model.decoder.blocks[0].transformers[0].attn.rope.inv_freq.clone()
    value = value + 0.1 if change == "frequencies" else value.unsqueeze(0)
    with pytest.raises(ValueError, match="rotary"):
        prepare._validated_same_state_dict(model, {key: value})


def test_same_filter_does_not_hide_unknown_weights_or_dit_rotary_buffer():
    model = small_native_vae()
    weights = model.state_dict()
    value = model.decoder.blocks[0].transformers[0].attn.rope.inv_freq.clone()
    weights["rotary_pos_emb.inv_freq"] = value
    weights["decoder.blocks.0.transformers.0.rope.learned_weight"] = value
    weights["decoder.blocks.0.transformers.99.rope.inv_freq"] = value
    converted = prepare._validated_same_state_dict(model, weights)
    assert set(converted) == set(weights)
    with pytest.raises(RuntimeError, match="Unexpected key"):
        model.load_state_dict(converted, strict=True)


def test_same_persistent_rotary_buffer_is_not_discarded():
    model = small_native_vae()
    rope = model.decoder.blocks[0].transformers[0].attn.rope
    rope._non_persistent_buffers_set.remove("inv_freq")
    key = "decoder.blocks.0.transformers.0.rope.inv_freq"
    weights = {key: rope.inv_freq.clone()}
    assert key in prepare._validated_same_state_dict(model, weights)


def test_child_auth_failure_remains_actionable_in_gui_result(tmp_path, monkeypatch):
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"model": "small", "status_dir": str(tmp_path)}), encoding="utf-8")

    def failed_worker(job):
        raise RuntimeError("401 gated model access")

    monkeypatch.setattr(prepare, "conversion_worker", failed_worker)
    assert prepare.main(["--internal-job", str(job)]) == 1
    details = json.loads((tmp_path / "child_error.json").read_text(encoding="utf-8"))
    assert "https://huggingface.co/stabilityai/stable-audio-3-small-music" in details["error"]
    assert "hf auth login" in details["error"]

    class Child:
        returncode = 1

        def poll(self):
            return 1

    monkeypatch.setattr(prepare.subprocess, "Popen", lambda *args, **kwargs: Child())
    with pytest.raises(RuntimeError, match="hf auth login"):
        prepare.run_child(job, tmp_path)
