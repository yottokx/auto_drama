"""Shared Qwen operations with fake tensors; never load a GPU model."""
from __future__ import annotations

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest
from PIL import Image

from services.worker.generation import event_cg_runner as runner


@pytest.fixture
def runtime(monkeypatch):
    state = SimpleNamespace(verifies=0, loads=0, renders=[], alpha=250)
    cuda = SimpleNamespace(is_available=lambda: True, reset_peak_memory_stats=lambda: None,
        empty_cache=lambda: None, synchronize=lambda: None,
        max_memory_allocated=lambda: 123, max_memory_reserved=lambda: 234)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda))
    monkeypatch.setattr(runner, "peak_ram_bytes", lambda: 345)
    monkeypatch.setattr(runner.engine, "_versions", lambda: {"torch": "fake"})

    def verify(path, **kwargs):
        state.verifies += 1
        return {"revision": "a" * 40}

    def load(request, tensors):
        state.loads += 1
        return object()

    def render(pipe, tensors, request, images, seed, callback):
        state.renders.append((request["prompt"], seed, len(images)))
        callback(pipe, 0, 0, {})
        return Image.new("RGBA", (64, 96), (1, 2, 3, state.alpha))

    monkeypatch.setattr(runner.engine, "verify_model", verify)
    monkeypatch.setattr(runner.backend, "load_pipeline", load)
    monkeypatch.setattr(runner.backend, "render_image", render)
    monkeypatch.setattr(runner.engine, "gpu_lock", lambda *_args, **_kwargs:
        pytest.fail("The child must not acquire the parent's GPU lock"))
    return state


def request(tmp_path, name):
    image = tmp_path / "reference.png"
    if not image.exists():
        Image.new("RGBA", (32, 64), "red").save(image)
    output = tmp_path / name
    output.mkdir()
    return {"output_dir": str(output), "expected_revision": "a" * 40,
        "generation": {"schema_version": 1, "mode": "scene", "prompt": name,
            "width": 64, "height": 96, "steps": 1, "seed": 7,
            "model_path": str(tmp_path / "model"), "references": [{"path": str(image),
                "name": "Alice", "sha256": hashlib.sha256(image.read_bytes()).hexdigest()}]}}


def test_verify_load_once_original_preserved_and_display_opaque(tmp_path, runtime):
    backend = runner.QwenRuntime()
    try:
        first = backend.generate(request(tmp_path, "base"), lambda _: None)
        second = backend.generate(request(tmp_path, "variant"), lambda _: None)
        assert runtime.verifies == runtime.loads == 1
        assert first["provenance"]["model_reused"] is False
        assert second["provenance"]["model_reused"] is True
        assert first["model_loaded_at_monotonic"] == second["model_loaded_at_monotonic"]
        with Image.open(tmp_path / "base/original.png") as image:
            assert image.getchannel("A").getextrema() == (250, 250)
        with Image.open(tmp_path / "base/image.png") as image:
            assert image.mode == "RGB"
        assert second["provenance"]["process_ram_peak_bytes"] == 345
    finally:
        backend.close()


def test_large_transparent_output_is_rejected(tmp_path, runtime):
    runtime.alpha = 0
    with pytest.raises(ValueError, match="substantial transparency"):
        runner.QwenRuntime().generate(request(tmp_path, "transparent"), lambda _: None)
    assert not (tmp_path / "transparent/image.png").exists()


def test_wrong_model_and_reference_identity_are_not_model_failures(tmp_path, runtime):
    source = request(tmp_path, "wrong-model")
    source["expected_revision"] = "b" * 40
    with pytest.raises(runner.ModelIdentityError, match="revision"):
        runner.QwenRuntime().generate(source, lambda _: None)
    source = request(tmp_path, "wrong-reference")
    source["generation"]["references"][0]["sha256"] = "b" * 64
    with pytest.raises(runner.ModelIdentityError, match="Reference changed"):
        runner.QwenRuntime().generate(source, lambda _: None)
    assert runtime.loads == 0


def test_ram_probe_returns_value_or_unknown():
    result = runner.peak_ram_bytes()
    assert result is None or result > 0


def test_child_report_keeps_seed_and_pixel_hashes(tmp_path, runtime):
    runner.QwenRuntime().generate(request(tmp_path, "report"), lambda _: None)
    report = json.loads((tmp_path / "report/result.json").read_text(encoding="utf-8"))
    assert report["provenance"]["seed"] == 7
    assert report["provenance"]["image_sha256"] == hashlib.sha256(
        (tmp_path / "report/image.png").read_bytes()).hexdigest()
