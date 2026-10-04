from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

from scripts.image_edit import engine, resources, runner


def make_model(directory: Path) -> Path:
    directory.mkdir()
    names = [*engine.REQUIRED_FILES, "text_encoder/model.safetensors",
             "transformer/diffusion_pytorch_model.safetensors", "vae/diffusion_pytorch_model.safetensors"]
    files = []
    for name in names:
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = (json.dumps({"_class_name": "QwenImage21Pipeline", **engine.COMPONENT_CLASSES}).encode()
                   if name == "model_index.json" else b"locally pinned fake component")
        path.write_bytes(payload)
        files.append({"path": name, "size_bytes": len(payload),
                      "sha256": hashlib.sha256(payload).hexdigest()})
    (directory / "model-manifest.json").write_text(json.dumps({
        "schema_version": 1, "model_id": engine.MODEL_ID, "revision": "a" * 40, "files": files,
    }))
    return directory


def make_reference(path: Path, color=(255, 0, 0, 128), size=(32, 64)) -> Path:
    Image.new("RGBA", size, color).save(path)
    return path


def make_request(tmp_path: Path, **overrides) -> dict:
    reference = make_reference(tmp_path / "source.png")
    model = make_model(tmp_path / "model")
    return {"schema_version": 1, "mode": "portrait", "prompt": "Keep the same face. Smile.",
            "references": [{"path": str(reference), "name": "Alice", "character_id": "alice",
                            "source": {"artifact_id": "image-1", "project_id": "story-1"}}],
            "model_path": str(model), "width": 64, "height": 96, "steps": 2,
            "seed": 7, "dtype": "bfloat16", "cpu_offload": True,
            "use_kv_cache": True, "reference_resolution": 1024,
            "transparent": True, "context": {"scene_id": "scene-1"}, **overrides}


@pytest.fixture
def fake_runtime(tmp_path, monkeypatch):
    calls = SimpleNamespace(loads=[], generates=[], sequence=[], inference_depth=0,
                            on_generate=None, output_alpha=128)
    monkeypatch.setattr(engine, "ROOT", tmp_path)
    monkeypatch.setattr(engine, "RUNTIME", tmp_path / "runtime")

    @contextmanager
    def inference_mode():
        calls.inference_depth += 1
        try:
            yield
        finally:
            calls.inference_depth -= 1

    @contextmanager
    def lease(*_args, **_kwargs):
        calls.sequence.append("lease-acquired")
        try:
            yield
        finally:
            calls.sequence.append("lease-released")

    class Generator:
        def __init__(self, device):
            assert device == "cpu"

        def manual_seed(self, seed):
            self.seed = seed
            return self

    class Pipe:
        def enable_model_cpu_offload(self):
            assert calls.inference_depth > 0
            calls.sequence.append("offload")

        def to(self, device):
            assert device == "cuda" and calls.inference_depth > 0
            calls.sequence.append("cuda")

        def set_progress_bar_config(self, **_kwargs):
            pass

        def __call__(self, **kwargs):
            assert calls.inference_depth > 0
            calls.generates.append(kwargs)
            if calls.on_generate:
                calls.on_generate(len(calls.generates), kwargs)
            for index in range(kwargs["num_inference_steps"]):
                payload = {"unchanged": True}
                returned = kwargs["callback_on_step_end"](self, index, index, payload)
                assert returned is payload
            return SimpleNamespace(images=[Image.new("RGBA", (kwargs["width"], kwargs["height"]),
                                                      (0, 100, 255, calls.output_alpha))])

    def load(*args, **kwargs):
        assert calls.inference_depth > 0
        calls.loads.append((args, kwargs))
        calls.sequence.append("load")
        return Pipe()

    cuda = SimpleNamespace(is_available=lambda: True, get_device_name=lambda: "test GPU",
                           reset_peak_memory_stats=lambda: None, max_memory_allocated=lambda: 200,
                           max_memory_reserved=lambda: 300,
                           empty_cache=lambda: calls.sequence.append("empty-cache"),
                           synchronize=lambda: calls.sequence.append("synchronize"))
    torch = SimpleNamespace(bfloat16="bf16", float16="fp16", float32="fp32", cuda=cuda,
                            inference_mode=inference_mode, Generator=Generator)
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "diffusers", SimpleNamespace(
        QwenImage21Pipeline=SimpleNamespace(from_pretrained=load)))
    monkeypatch.setattr(engine, "gpu_lock", lease)
    return calls


@pytest.mark.parametrize("changes", [
    {"width": 80}, {"height": True}, {"seed": True}, {"seeds": [-1]}, {"seeds": []},
    {"seeds": [2**63]}, {"references": []}, {"cpu_offload": "true"}, {"prompt": " "},
    {"schema_version": True}, {"context": {"nan": float("nan")}},
])
def test_invalid_requests_are_rejected_before_loading(tmp_path, changes):
    with pytest.raises((ValueError, TypeError)):
        engine.validate_request(make_request(tmp_path, **changes))


def test_lightweight_import_never_loads_inference_modules():
    check = subprocess.run([sys.executable, "-c", (
        "import sys; from scripts.image_edit.engine import validate_request; "
        "assert not {'torch','diffusers','transformers'} & set(sys.modules)")],
        cwd=engine.ROOT, capture_output=True, text=True, timeout=30, check=False)
    assert check.returncode == 0, check.stderr


def test_reference_order_alpha_exif_and_original_bytes_are_preserved(tmp_path):
    first = make_reference(tmp_path / "first.png", (12, 33, 99, 64), (10, 20))
    second = tmp_path / "second.jpg"
    source = Image.new("RGB", (12, 25), "blue")
    exif = source.getexif()
    exif[274] = 6
    source.save(second, exif=exif)
    originals = [first.read_bytes(), second.read_bytes()]
    references = [{"path": str(first), "name": "first", "source": {"id": "one"}},
                  {"path": str(second), "name": "second", "source": {"id": "two"}}]
    output = tmp_path / "run"
    output.mkdir()
    images, records = engine.normalize_references(references, output)
    assert [record["name"] for record in records] == ["first", "second"]
    assert images[0].mode == "RGBA" and images[0].getpixel((0, 0)) == (12, 33, 99, 64)
    assert images[1].size == (25, 12) and images[1].getchannel("A").getextrema() == (255, 255)
    assert [record["original_sha256"] for record in records] == [
        hashlib.sha256(data).hexdigest() for data in originals]
    assert records[0]["source"] == {"id": "one"}
    assert first.read_bytes() == originals[0] and second.read_bytes() == originals[1]
    assert all(Path(record["normalized_path"]).is_file() for record in records)


def test_native_rgba_seed_batch_reuses_model_and_records_reproduction_metadata(tmp_path, fake_runtime):
    request = make_request(tmp_path, seeds=[0, 1, 2], mode="scene")
    second = make_reference(tmp_path / "bob.png", (0, 255, 0, 255))
    request["references"].append({"path": str(second), "name": "Bob", "source": {}})
    output = tmp_path / "run"
    output.mkdir()
    result = engine.generate(request, output)
    assert result["ok"] and not result["cancelled"]
    assert [item["seed"] for item in result["outputs"]] == [0, 1, 2]
    assert len(fake_runtime.loads) == 1 and len(fake_runtime.generates) == 3
    assert fake_runtime.loads[0][1] == {"dtype": "bf16", "local_files_only": True,
                                     "trust_remote_code": False}
    for call, seed in zip(fake_runtime.generates, [0, 1, 2], strict=True):
        assert call["generator"].seed == seed
        assert call["true_cfg_scale"] == 1.0 and "negative_prompt" not in call
        assert call["output_resolution"] == 1024 and call["use_kv_cache"] is True
        assert [image.getpixel((0, 0)) for image in call["image"]] == [
            (255, 0, 0, 128), (0, 255, 0, 255)]
    metadata = engine.read_json(output / "generation.json")
    assert metadata["actual_seeds"] == [0, 1, 2] and metadata["status"] == "complete"
    assert metadata["model"]["revision"] == "a" * 40
    assert metadata["requested_dimensions"] == {"width": 64, "height": 96}
    assert metadata["request"]["context"] == {"scene_id": "scene-1"}
    assert "diffusers" in metadata["library_versions"]
    assert all(item["alpha_range"] == [128, 128] and not item["opaque"]
               and item["gpu_peak_allocated"] == 200 for item in metadata["outputs"])
    assert engine.read_json(output / "status.json")["phase"] == "complete"
    release = fake_runtime.sequence.index("lease-released")
    assert "empty-cache" in fake_runtime.sequence[:release]
    assert "synchronize" in fake_runtime.sequence[:release]


def test_random_seed_is_resolved_once_and_persisted(tmp_path, fake_runtime, monkeypatch):
    request = make_request(tmp_path, seed=-1)
    draws = []
    monkeypatch.setattr(engine.secrets, "randbelow", lambda bound: draws.append(bound) or 1234)
    output = tmp_path / "run"
    output.mkdir()
    result = engine.generate(request, output)
    assert result["ok"] and result["outputs"][0]["seed"] == 1234
    assert draws == [2**63]
    assert engine.read_json(output / "generation.json")["actual_seeds"] == [1234]


def test_transparency_instruction_does_not_fabricate_alpha(tmp_path, fake_runtime):
    request = make_request(tmp_path)
    fake_runtime.output_alpha = 255
    output = tmp_path / "run"
    output.mkdir()
    result = engine.generate(request, output)
    image = result["outputs"][0]
    assert result["ok"] and image["opaque"] and image["alpha_range"] == [255, 255]
    assert image["warnings"] and Image.open(image["path"]).getchannel("A").getextrema() == (255, 255)


@pytest.mark.parametrize("stop", [True, False])
def test_second_seed_stop_or_failure_preserves_first_image(tmp_path, fake_runtime, stop):
    request = make_request(tmp_path, seeds=[0, 1, 2])
    output = tmp_path / "run"
    output.mkdir()

    def during_generation(batch, _kwargs):
        if batch == 2:
            if stop:
                (output / "stop.request").touch()
            else:
                raise RuntimeError("test CUDA failure")

    fake_runtime.on_generate = during_generation
    result = engine.generate(request, output)
    assert not result["ok"] and result["cancelled"] is stop and result["partial"]
    assert len(result["outputs"]) == 1 and result["outputs"][0]["seed"] == 0
    assert Path(result["outputs"][0]["path"]).is_file()
    assert len(list((output / "outputs").glob("*.png"))) == 1
    assert engine.read_json(output / "generation.json")["status"] == ("cancelled" if stop else "failed")
    assert engine.read_json(output / "result.json")["outputs"] == result["outputs"]
    release = fake_runtime.sequence.index("lease-released")
    assert "empty-cache" in fake_runtime.sequence[:release]


def test_stop_before_loading_has_no_output_or_model_load(tmp_path, fake_runtime):
    request = make_request(tmp_path)
    output = tmp_path / "run"
    output.mkdir()
    (output / "stop.request").touch()
    result = engine.generate(request, output)
    assert not result["ok"] and result["cancelled"] and not result["outputs"]
    assert not fake_runtime.loads and not (output / "inputs").exists()


def test_failed_pipeline_traceback_drops_weights_before_gpu_lease_release(
    tmp_path, fake_runtime, monkeypatch,
):
    request = make_request(tmp_path)
    pipeline_class = sys.modules["diffusers"].QwenImage21Pipeline
    original_load = pipeline_class.from_pretrained

    def load(*args, **kwargs):
        pipe = original_load(*args, **kwargs)
        monkeypatch.setattr(type(pipe), "__del__", lambda _self:
                            fake_runtime.sequence.append("weights-destroyed"), raising=False)
        return pipe

    def fail_in_pipeline(_batch, _kwargs):
        raise RuntimeError("failure retaining pipeline self in its traceback")

    monkeypatch.setattr(pipeline_class, "from_pretrained", load)
    fake_runtime.on_generate = fail_in_pipeline
    output = tmp_path / "run"
    output.mkdir()
    result = engine.generate(request, output)
    assert not result["ok"] and result["error_type"] == "RuntimeError"
    assert fake_runtime.sequence.index("weights-destroyed") < fake_runtime.sequence.index("lease-released")


def test_all_invalid_metadata_still_produces_a_failure_record(tmp_path):
    request = make_request(tmp_path, context={"nan": float("nan")})
    output = tmp_path / "run"
    output.mkdir()
    result = engine.generate(request, output)
    assert not result["ok"]
    assert engine.read_json(output / "result.json")["error_type"] == "ValueError"


def test_atomic_images_and_completed_runs_never_overwrite(tmp_path):
    target = tmp_path / "image.png"
    engine.atomic_image(Image.new("RGBA", (4, 4), "red"), target)
    before = target.read_bytes()
    with pytest.raises(FileExistsError):
        engine.atomic_image(Image.new("RGBA", (4, 4), "green"), target)
    assert target.read_bytes() == before
    output = tmp_path / "old-run"
    output.mkdir()
    (output / "result.json").write_text('{"old":true}')
    request = tmp_path / "request.json"
    request.write_text("{}")
    assert runner.run_request(request, output) == 2
    assert (output / "result.json").read_text() == '{"old":true}'
    assert not (output / "generation.json").exists()


def test_model_manifest_detects_tamper_and_missing_shards(tmp_path):
    model = make_model(tmp_path / "model")
    assert engine.verify_model(model)["verified_files"] == len(engine.REQUIRED_FILES) + 3
    (model / "vae/diffusion_pytorch_model.safetensors").write_bytes(b"changed")
    with pytest.raises(ValueError, match="固定記録"):
        engine.verify_model(model)


def test_manifest_requires_all_components_and_fixed_revision(tmp_path):
    model = make_model(tmp_path / "model")
    manifest = engine.read_json(model / "model-manifest.json")
    manifest["revision"] = "main"
    engine.write_json(model / "model-manifest.json", manifest)
    with pytest.raises(ValueError, match="固定リビジョン"):
        engine.verify_model(model)
    manifest["revision"] = "a" * 40
    manifest["files"] = [row for row in manifest["files"] if row["path"] != "processor/tokenizer.json"]
    engine.write_json(model / "model-manifest.json", manifest)
    with pytest.raises(ValueError, match="必須"):
        engine.verify_model(model)


def test_manifest_rejects_extra_files_path_traversal_and_unverified_shards(tmp_path):
    model = make_model(tmp_path / "model")
    original = engine.read_json(model / "model-manifest.json")
    extra = model / "unverified.json"
    extra.write_text("{}")
    with pytest.raises(ValueError, match="ファイル一覧"):
        engine.verify_model(model)
    extra.unlink()
    manifest = {**original, "files": [*original["files"], {
        "path": "../outside.json", "size_bytes": 1, "sha256": "a" * 64}]}
    engine.write_json(model / "model-manifest.json", manifest)
    with pytest.raises(ValueError, match="ファイルパス"):
        engine.verify_model(model)
    index_path = model / "transformer/diffusion_pytorch_model.safetensors.index.json"
    index_path.write_text(json.dumps({"weight_map": {"first": "unverified.safetensors"}}))
    manifest["files"] = [*original["files"], {
        "path": index_path.relative_to(model).as_posix(), "size_bytes": index_path.stat().st_size,
        "sha256": engine.digest(index_path)}]
    engine.write_json(model / "model-manifest.json", manifest)
    with pytest.raises(ValueError, match="未検証の分割重み"):
        engine.verify_model(model)


def test_model_verification_can_stop_without_importing_weights(tmp_path):
    model = make_model(tmp_path / "model")
    run = tmp_path / "run"
    run.mkdir()
    token = resources.StopToken(run)

    def progress(_data):
        (run / "stop.request").touch()

    with pytest.raises(resources.GenerationCancelled):
        engine.verify_model(model, token=token, progress=progress)


def test_gpu_lock_is_exclusive_cancelled_and_released(tmp_path):
    lock = tmp_path / "gpu.lock"
    run = tmp_path / "run"
    run.mkdir()
    token = resources.StopToken(run)
    with resources.gpu_lock(lock):
        with pytest.raises(TimeoutError), resources.gpu_lock(lock, timeout=0):
            pytest.fail("GPU lease was acquired twice")
        (run / "stop.request").touch()
        with pytest.raises(resources.GenerationCancelled), resources.gpu_lock(lock, token=token):
            pytest.fail("Cancelled waiting job ran")
    (run / "stop.request").unlink()
    with resources.gpu_lock(lock, timeout=0, token=token):
        pass


def test_lease_released_if_cancelled_immediately_after_acquisition(tmp_path):
    class Token:
        checks = 0

        def check(self):
            self.checks += 1
            if self.checks == 2:
                raise resources.GenerationCancelled("stopped")

    lock = tmp_path / "gpu.lock"
    with pytest.raises(resources.GenerationCancelled), resources.gpu_lock(lock, token=Token()):
        pytest.fail("Stopped job ran")
    with resources.gpu_lock(lock, timeout=0):
        pass


def test_waiting_gpu_lease_observes_new_stop_request(tmp_path):
    lock = tmp_path / "gpu.lock"
    run = tmp_path / "run"
    run.mkdir()
    token = resources.StopToken(run)

    def stop_later():
        threading.Event().wait(0.05)
        token.path.touch()

    with resources.gpu_lock(lock):
        stopping = threading.Thread(target=stop_later)
        stopping.start()
        try:
            with pytest.raises(resources.GenerationCancelled), resources.gpu_lock(lock, token=token):
                pytest.fail("Waiting job ignored its stop request")
        finally:
            stopping.join(timeout=2)
