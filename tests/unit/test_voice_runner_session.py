from __future__ import annotations

import hashlib
import io
import json
import os
import sys
from contextlib import contextmanager
from copy import deepcopy
from types import SimpleNamespace

import pytest

from packages.contracts.tts_profile import build_tts_profile
from services.worker.generation import voice_runner
from services.worker.generation.tts_runtime import TTSRuntimeProfile, resolve_request_profile


@pytest.fixture
def fake_irodori(tmp_path, monkeypatch):
    calls = SimpleNamespace(configure=[], verify=[], keys=[], sampling=[], writes=[], inspected=[])
    runtime_root = tmp_path / "irodori"
    monkeypatch.setattr(voice_runner, "RUNTIME_ROOT", runtime_root)
    monkeypatch.setattr(sys, "prefix", str(runtime_root / ".venv"))
    monkeypatch.setattr(voice_runner, "configure_environment", lambda: calls.configure.append(True))
    monkeypatch.setattr(voice_runner.subprocess, "check_output", lambda *args, **kwargs: "pinned\n")
    monkeypatch.setattr(voice_runner.importlib.metadata, "version",
                        lambda name: "0.16.0" if name == "torchao" else "test-version")

    def resolve(request):
        bundle = deepcopy(resolve_request_profile(request).bundle)
        bundle["source_commit"] = "pinned"
        return TTSRuntimeProfile(bundle)

    monkeypatch.setattr(voice_runner, "resolve_request_profile", resolve)

    def verify(profile, root):
        calls.verify.append((profile, root))

    def inspect(path):
        calls.inspected.append(path)
        return {"path": str(path), "decoded_and_non_silent": True, "duration_seconds": 1.0}

    monkeypatch.setattr(voice_runner, "verify_profile_files", verify)
    monkeypatch.setattr(voice_runner, "inspect_wav", inspect)

    audio = SimpleNamespace(ndim=1, size=8)
    audio.squeeze = lambda: audio
    numpy = SimpleNamespace(
        asarray=lambda value: audio,
        isfinite=lambda value: SimpleNamespace(all=lambda: True),
    )

    def write(path, samples, sample_rate, subtype):
        calls.writes.append((path, sample_rate, subtype))
        path.write_bytes(b"pcm16 audio")

    def synthesize(request, log_fn):
        calls.sampling.append(request)
        log_fn(f"synthesized {request.text}")
        return SimpleNamespace(audio=audio, sample_rate=24000, used_seed=request.seed)

    tokenizer = SimpleNamespace(encode=lambda text: SimpleNamespace(numel=lambda: len(text)))
    runtime = SimpleNamespace(
        watermarker=SimpleNamespace(ready=True),
        model_cfg=SimpleNamespace(use_speaker_condition_resolved=True),
        tokenizer=tokenizer,
        caption_tokenizer=tokenizer,
        default_text_max_len=100,
        default_caption_max_len=100,
        synthesize=synthesize,
    )

    def from_key(key):
        calls.keys.append(key)
        print("loading model")
        return runtime

    calls.quantization_type = "int8_weight_only"

    @contextmanager
    def checkpoint(*args, **kwargs):
        yield SimpleNamespace(metadata=lambda: {
            "irodori_quantization_json": json.dumps({"quantization_type": calls.quantization_type})
        })

    calls.cuda = SimpleNamespace(is_available=lambda: True, is_bf16_supported=lambda: True,
                                 get_device_capability=lambda: (12, 0))
    modules = {
        "numpy": numpy,
        "silentcipher": SimpleNamespace(get_model=lambda **kwargs: None),
        "soundfile": SimpleNamespace(write=write),
        "torch": SimpleNamespace(cuda=calls.cuda),
        "safetensors": SimpleNamespace(safe_open=checkpoint),
        "irodori_tts": SimpleNamespace(),
        "irodori_tts.inference_runtime": SimpleNamespace(
            InferenceRuntime=SimpleNamespace(from_key=from_key),
            RuntimeKey=SimpleNamespace,
            SamplingRequest=SimpleNamespace,
        ),
        "irodori_tts.text_normalization": SimpleNamespace(normalize_text=lambda text: text),
        "irodori_tts.quantization": SimpleNamespace(parse_quantization_metadata=lambda metadata:
                                                   json.loads(metadata["irodori_quantization_json"])),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    calls.runtime = runtime
    return calls


def make_request(tmp_path, name, **overrides):
    output = tmp_path / name
    output.mkdir()
    request = {
        "mode": "design",
        "text": name,
        "caption": "a calm voice",
        "model_precision": "bf16",
        "seed": 42,
        "num_steps": 30,
        **overrides,
    }
    if request["mode"] == "clone":
        reference = f"reference for {name}".encode()
        (output / "reference.wav").write_bytes(reference)
        request.setdefault("reference_text", f"exact transcript for {name}")
        request.setdefault("reference_sha256", hashlib.sha256(reference).hexdigest())
        request.setdefault("reference_artifact_id", f"artifact-{name}")
    (output / "request.json").write_text(json.dumps(request), encoding="utf-8")
    return output


def test_reuses_verified_weights_but_not_speaker_caption_or_seed(tmp_path, fake_irodori):
    outputs = [
        make_request(tmp_path, "first", mode="clone", caption="", seed=11),
        make_request(tmp_path, "second", mode="clone", caption="excited", seed=22),
        make_request(tmp_path, "third", seed=33),
    ]
    session = voice_runner.VoiceRuntimeSession()
    for output in outputs:
        session.generate(output)
    assert len(fake_irodori.configure) == len(fake_irodori.verify) == len(fake_irodori.keys) == 1
    assert [request.ref_wav for request in fake_irodori.sampling] == [
        str(outputs[0] / "reference.wav"), str(outputs[1] / "reference.wav"), None,
    ]
    assert [request.no_ref for request in fake_irodori.sampling] == [False, False, True]
    assert [request.caption for request in fake_irodori.sampling] == [None, "excited", "a calm voice"]
    assert [request.seed for request in fake_irodori.sampling] == [11, 22, 33]
    assert all(request.num_steps == 30 for request in fake_irodori.sampling)
    reports = [json.loads((output / "result.json").read_text()) for output in outputs]
    assert [report["model_reused"] for report in reports] == [False, True, True]
    assert all(report["runtime_pid"] == os.getpid() for report in reports)
    assert reports[0]["timings"]["initialization_seconds"] >= 0
    assert all(report["timings"]["initialization_seconds"] == 0 for report in reports[1:])
    assert all(
        report["timings"]["job_seconds"] >= report["timings"]["synthesis_seconds"] >= 0
        for report in reports
    )
    assert [report["reference_text"] for report in reports] == [
        "exact transcript for first", "exact transcript for second", "third",
    ]
    assert reports[-1]["reference_sha256"] is None
    assert reports[-1]["reference_artifact_id"] is None
    assert all(report["watermarked"] and report["format"] == "PCM_16" for report in reports)
    assert all(subtype == "PCM_16" for _, _, subtype in fake_irodori.writes)


def test_rechecks_reference_sha_before_every_request(tmp_path, fake_irodori):
    session = voice_runner.VoiceRuntimeSession()
    first = make_request(tmp_path, "first", mode="clone")
    second = make_request(tmp_path, "second", mode="clone")
    session.generate(first)
    (second / "reference.wav").write_bytes(b"changed since queueing")
    with pytest.raises(ValueError, match="SHA256 mismatch"):
        session.generate(second)
    assert len(fake_irodori.verify) == len(fake_irodori.sampling) == 1
    assert not (second / "result.json").exists()


def test_rechecks_token_limits_for_warm_runtime(tmp_path, fake_irodori):
    session = voice_runner.VoiceRuntimeSession()
    session.generate(make_request(tmp_path, "first"))
    with pytest.raises(ValueError, match="truncation is forbidden"):
        session.generate(make_request(tmp_path, "second", text="x" * 101))
    assert len(fake_irodori.sampling) == 1


def test_rejects_profile_change_without_loading_second_model(tmp_path, fake_irodori):
    session = voice_runner.VoiceRuntimeSession()
    session.generate(make_request(tmp_path, "first"))
    with pytest.raises(RuntimeError, match="selection changed; restart"):
        session.generate(make_request(tmp_path, "second", model_precision="fp32"))
    assert len(fake_irodori.keys) == len(fake_irodori.sampling) == 1


def test_cannot_initialize_with_unpinned_source(tmp_path, fake_irodori, monkeypatch):
    monkeypatch.setattr(voice_runner.subprocess, "check_output", lambda *args, **kwargs: "wrong")
    with pytest.raises(RuntimeError, match="source revision mismatch"):
        voice_runner.VoiceRuntimeSession().generate(make_request(tmp_path, "first"))
    assert fake_irodori.verify == fake_irodori.keys == fake_irodori.sampling == []


def test_watermark_failure_prevents_any_result(tmp_path, fake_irodori):
    fake_irodori.runtime.watermarker.ready = False
    output = make_request(tmp_path, "first")
    with pytest.raises(RuntimeError, match="watermark failed"):
        voice_runner.VoiceRuntimeSession().generate(output)
    assert fake_irodori.sampling == []
    assert not (output / "result.json").exists()


def selected_profile(model_id, precision):
    choice = {"provider_id": "irodori", "model_id": model_id, "precision": precision}
    return build_tts_profile({"voice_design": choice, "voice_clone": choice})["voice_design"]


def test_large_bf16_uses_large_checkpoint_and_records_actual_provenance(tmp_path, fake_irodori):
    profile = selected_profile("irodori-v4-large", "bf16")
    output = make_request(tmp_path, "large", tts_profile=profile, num_steps=profile["num_steps"])
    voice_runner.VoiceRuntimeSession().generate(output)
    key = fake_irodori.keys[0]
    assert "Irodori-TTS-v4-Large/model.safetensors" in key.checkpoint.replace("\\", "/")
    assert key.model_precision == "bf16"
    assert key.codec_precision == "fp32"
    report = json.loads((output / "result.json").read_text())
    assert report["model_id"] == "irodori-v4-large"
    assert report["precision"] == report["model_precision"] == "bf16"
    assert report["quantization"] is None
    assert report["model_revision"] == "2e0c55428ce97268a507f1feeb2478f8d9148e8b"
    assert report["manifest_id"] == profile["manifest_id"]
    assert report["dependency_revisions"]["sony/silentcipher"]


@pytest.mark.parametrize("model_id", ["irodori-v4.1-small", "irodori-v4-large"])
@pytest.mark.parametrize("precision", ["int8", "int4"])
def test_official_quantized_checkpoint_uses_bf16_compute(tmp_path, fake_irodori, model_id, precision):
    fake_irodori.quantization_type = f"{precision}_weight_only"
    profile = selected_profile(model_id, precision)
    output = make_request(tmp_path, "quantized", tts_profile=profile, num_steps=40)
    voice_runner.VoiceRuntimeSession().generate(output)
    key = fake_irodori.keys[0]
    assert f"-Quantized/{precision}-weight-only/model.safetensors" in key.checkpoint.replace("\\", "/")
    assert key.model_precision == "bf16"
    report = json.loads((output / "result.json").read_text())
    assert report["precision"] == precision
    assert report["quantization"] == f"{precision}-weight-only"
    assert report["versions"]["torchao"] == "0.16.0"


def test_rejects_wrong_quantization_before_loading_weights(tmp_path, fake_irodori):
    profile = selected_profile("irodori-v4-large", "int4")
    output = make_request(tmp_path, "wrongquant", tts_profile=profile, num_steps=40)
    with pytest.raises(ValueError, match="quantization differs"):
        voice_runner.VoiceRuntimeSession().generate(output)
    assert fake_irodori.keys == fake_irodori.sampling == []


def test_unsupported_bf16_device_fails_before_model_load(tmp_path, fake_irodori):
    fake_irodori.cuda.is_bf16_supported = lambda: False
    with pytest.raises(RuntimeError, match="bf16 support"):
        voice_runner.VoiceRuntimeSession().generate(make_request(tmp_path, "unsupported"))
    assert fake_irodori.keys == []


def test_server_outputs_only_json_and_separates_native_and_python_logs(tmp_path, monkeypatch):
    requests = []
    for name in ("first", "second"):
        (tmp_path / name).mkdir()
        requests.append(json.dumps({"output_dir": str(tmp_path / name)}))

    class LoggingSession:
        def generate(self, output):
            print(f"python stdout {output.name}")
            print(f"python stderr {output.name}", file=sys.stderr)
            os.write(1, f"native stdout {output.name}\n".encode())
            os.write(2, f"native stderr {output.name}\n".encode())

    monkeypatch.setattr(voice_runner, "VoiceRuntimeSession", LoggingSession)
    replies = io.StringIO()
    assert voice_runner.serve(io.StringIO("\n".join(requests) + "\n"), replies) == 0
    assert [json.loads(line) for line in replies.getvalue().splitlines()] == [
        {"ok": True}, {"ok": True},
    ]
    for name in ("first", "second"):
        log = (tmp_path / name / "runtime.log").read_text()
        assert log.splitlines() == [
            f"python stdout {name}", f"python stderr {name}",
            f"native stdout {name}", f"native stderr {name}",
        ]


def test_server_exits_after_failed_request_without_processing_later_jobs(tmp_path, fake_irodori):
    outputs = [
        make_request(tmp_path, "first"),
        make_request(tmp_path, "second", mode="clone", reference_sha256="invalid"),
        make_request(tmp_path, "third"),
    ]
    messages = "".join(json.dumps({"output_dir": str(output)}) + "\n" for output in outputs)
    replies = io.StringIO()
    assert voice_runner.serve(io.StringIO(messages), replies) == 1
    responses = [json.loads(line) for line in replies.getvalue().splitlines()]
    assert responses[0] == {"ok": True}
    assert responses[1] == {"ok": False, "error": "Reference voice SHA256 mismatch."}
    assert len(fake_irodori.sampling) == 1
    assert "Traceback" in (outputs[1] / "runtime.log").read_text()
    assert not (outputs[2] / "result.json").exists()


@pytest.mark.parametrize("message", ["{broken", "[]", '{}', '{"output_dir":"relative"}'])
def test_server_rejects_malformed_protocol_without_initializing(message, fake_irodori):
    replies = io.StringIO()
    assert voice_runner.serve(io.StringIO(message + "\n"), replies) == 1
    response = json.loads(replies.getvalue())
    assert response["ok"] is False
    assert response["error"]
    assert fake_irodori.configure == []


def test_one_shot_cli_remains_compatible(tmp_path, monkeypatch):
    outputs = []
    monkeypatch.setattr(sys, "argv", ["voice_runner.py", "--output-dir", str(tmp_path)])
    monkeypatch.setattr(
        voice_runner, "VoiceRuntimeSession", lambda: SimpleNamespace(generate=outputs.append)
    )
    assert voice_runner.main() == 0
    assert outputs == [tmp_path.resolve()]


def test_cli_modes_are_mutually_exclusive(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys, "argv", ["voice_runner.py", "--output-dir", str(tmp_path), "--serve"]
    )
    with pytest.raises(SystemExit) as error:
        voice_runner.main()
    assert error.value.code == 2
