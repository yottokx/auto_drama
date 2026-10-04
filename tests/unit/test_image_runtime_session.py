from __future__ import annotations

import hashlib
import io
import json
import sys
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.m0 import image_runner


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    calls = SimpleNamespace(loads=[], components=[], generates=[], guiders=[], removals=[], removal_sessions=[], unloads=[])
    model = tmp_path / "anima"
    model.mkdir()
    (model / "modular_model_index.json").write_text("{}")
    for name in image_runner.COMPONENTS:
        (model / name).mkdir()
    removal = tmp_path / "rembg"
    removal.mkdir()
    weights = b"pinned fake background-removal model"
    (removal / "isnet-anime.onnx").write_bytes(weights)
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"runtime_commit": "pinned-runtime", "models": [
        {"id": "anima-base-v1-diffusers", "revision": "pinned-anima"},
        {"id": "isnet-anime", "local_dir": "rembg", "files": [
            {"size": len(weights), "md5": hashlib.md5(weights).hexdigest()}
        ]},
    ]}))
    monkeypatch.setattr(image_runner, "MANIFEST", manifest)
    monkeypatch.setattr(image_runner, "ROOT", tmp_path)
    monkeypatch.setattr(image_runner, "RUNTIME", tmp_path / "diffusers")
    monkeypatch.setattr(image_runner.importlib.metadata, "version", lambda name: "pinned-version")

    class Generated:
        def __init__(self, request):
            self.request = request
            self.size = (request["width"], request["height"])

        def save(self, path):
            path.write_text(json.dumps(self.request))

        def convert(self, _mode):
            return self

        def getchannel(self, _name):
            return SimpleNamespace(getextrema=lambda: [0, 255], getbbox=lambda: [10, 20, 700, 1000])

    class Pipe:
        def __init__(self):
            self.components = dict.fromkeys(image_runner.COMPONENTS)
            for name in image_runner.COMPONENTS:
                setattr(self, name, object())

        def load_components(self, **kwargs):
            calls.components.append(kwargs)

        def update_components(self, *, guider):
            calls.guiders.append(guider.guidance_scale)

        def to(self, device):
            assert device == "cuda"

        def __call__(self, **kwargs):
            kwargs["seed"] = kwargs.pop("generator").seed
            calls.generates.append(kwargs)
            return [Generated(kwargs)]

        def unload_components(self, components):
            calls.unloads.append(components)

    def load(*args, **kwargs):
        calls.loads.append((args, kwargs))
        return Pipe()

    class Generator:
        def __init__(self, device):
            assert device == "cpu"

        def manual_seed(self, seed):
            self.seed = seed
            return self

    def new_removal(*args, **kwargs):
        session = object()
        calls.removal_sessions.append((session, args, kwargs))
        return session

    def remove(image, *, session):
        calls.removals.append((image.request, session))
        return image

    modules = {
        "torch": SimpleNamespace(bfloat16="bf16", Generator=Generator, inference_mode=nullcontext,
                                 cuda=SimpleNamespace(is_available=lambda: True, memory_allocated=lambda: 100,
                                     max_memory_allocated=lambda: 200, reset_peak_memory_stats=lambda: None,
                                     get_device_name=lambda: "fake GPU", empty_cache=lambda: None, synchronize=lambda: None)),
        "diffusers": SimpleNamespace(AnimaModularPipeline=SimpleNamespace(from_pretrained=load),
                                      ClassifierFreeGuidance=lambda **kwargs: SimpleNamespace(**kwargs)),
        "rembg": SimpleNamespace(new_session=new_removal, remove=remove),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    calls.root, calls.model = tmp_path, model
    return calls


def request(runtime, name, *, mode="character", **changes):
    output = runtime.root / name
    output.mkdir()
    return {"output_dir": str(output), "model_dir": str(runtime.model), "mode": mode,
            "prompt": f"positive {name}", "negative_prompt": f"negative {name}",
            "width": 768 if mode == "character" else 1280,
            "height": 1024 if mode == "character" else 720, "steps": 30, "guidance_scale": 4,
            "seed": 1, **changes}


def test_anima_and_cpu_cutout_weights_reused_across_portraits_then_backgrounds(runtime):
    session = image_runner.ImageRuntimeSession()
    requests = [request(runtime, f"image-{n}", mode=mode, seed=n + 10, guidance_scale=n + 2,
                        steps=30 + n) for n, mode in enumerate(("character", "character", "background", "background"))]
    for value in requests:
        session.generate(value)
    assert len(runtime.loads) == len(runtime.components) == len(runtime.removal_sessions) == 1
    assert len(runtime.removals) == 2
    assert runtime.guiders == [2, 3, 4, 5]
    assert [generation["seed"] for generation in runtime.generates] == [10, 11, 12, 13]
    assert [generation["num_inference_steps"] for generation in runtime.generates] == [30, 31, 32, 33]
    assert [generation["width"] for generation in runtime.generates] == [768, 768, 1280, 1280]
    assert [generation["prompt"] for generation in runtime.generates] == [value["prompt"] for value in requests]
    reports = [json.loads((runtime.root / f"image-{n}" / "result.json").read_text()) for n in range(4)]
    assert [report["model_reused"] for report in reports] == [False, True, True, True]
    assert len({report["model_loaded_at_monotonic"] for report in reports}) == 1
    assert reports[0]["model_loaded_at_monotonic"] == session.loaded_at
    assert runtime.unloads == []
    session.close()
    session.close()
    assert len(runtime.unloads) == 1


@pytest.mark.parametrize("changes", [{"width": 10}, {"steps": 0}, {"seed": -1},
                                      {"guidance_scale": float("nan")}, {"mode": "unknown"}])
def test_invalid_request_does_not_load_weights(runtime, changes):
    with pytest.raises(ValueError):
        image_runner.ImageRuntimeSession().generate(request(runtime, "invalid", **changes))
    assert runtime.loads == []


def test_same_output_cannot_be_overwritten(runtime):
    session = image_runner.ImageRuntimeSession()
    value = request(runtime, "saved")
    session.generate(value)
    with pytest.raises(ValueError, match="fresh"):
        session.generate(value)
    assert len(runtime.generates) == 1
    session.close()


def test_server_discards_runtime_after_error_and_skips_later_requests(runtime, monkeypatch):
    requests = [request(runtime, "first"), request(runtime, "failed", width=1), request(runtime, "never")]
    monkeypatch.setattr(image_runner, "request_log", lambda output: nullcontext())
    output = io.StringIO()
    result = image_runner.serve(io.StringIO("".join(json.dumps(value) + "\n" for value in requests)), output)
    assert result == 1
    replies = [json.loads(line) for line in output.getvalue().splitlines()]
    assert [reply["ok"] for reply in replies] == [True, False]
    assert len(runtime.loads) == len(runtime.generates) == len(runtime.unloads) == 1
