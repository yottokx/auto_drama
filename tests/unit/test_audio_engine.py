from __future__ import annotations

import json
import sys
import wave
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.audio import engine, runner


def request_data(**overrides):
    return {"model": "small", "model_path": "models/small", "prompt": "quiet ambient piano", **overrides}


def model_directory(tmp_path: Path, model: str = "small") -> Path:
    directory = tmp_path / "model"
    directory.mkdir()
    (directory / "model_index.json").write_text(
        json.dumps({"_class_name": "StableAudio3Pipeline"}), encoding="utf-8"
    )
    for name in ("transformer", "vae", "text_encoder", "tokenizer", "scheduler", "duration_embedder"):
        (directory / name).mkdir()
    dimensions = (1024, 20, 16, False) if model == "small" else (1536, 24, 24, True)
    config = dict(zip(
        ("embed_dim", "depth", "num_heads", "use_differential_attention"), dimensions, strict=True
    ))
    (directory / "transformer" / "config.json").write_text(json.dumps(config), encoding="utf-8")
    return directory


def write_preview(path: Path, sample_rate: int = 44100) -> None:
    with wave.open(str(path), "wb") as target:
        target.setnchannels(2)
        target.setsampwidth(2)
        target.setframerate(sample_rate)
        target.writeframes(b"\x01\x00\xfe\xff" * 8)


def test_model_caps_and_request_defaults():
    request = engine.validate_request(request_data())
    assert request.duration == 30
    assert request.steps == 8
    assert request.seed == -1
    assert request.output_format == "mp3"
    assert request.keep_wav is False
    assert request.mp3_bitrate == 192
    assert engine.MODEL_SPECS["small"]["repo_id"] == "stabilityai/stable-audio-3-small-music"
    assert engine.MODEL_SPECS["medium"]["repo_id"] == "stabilityai/stable-audio-3-medium"
    assert engine.validate_request(request_data(duration=120)).duration == 120
    assert engine.validate_request(request_data(model="medium", duration=380)).duration == 380


@pytest.mark.parametrize("overrides", [
    {"duration": 121}, {"model": "medium", "duration": 381}, {"duration": 0},
    {"duration": float("nan")}, {"duration": float("inf")}, {"duration": True},
    {"steps": 1.5}, {"steps": 0}, {"steps": True}, {"seed": -2}, {"seed": 2**32},
    {"device": "mps"}, {"dtype": "int8"}, {"device": "cpu", "dtype": "float16"},
    {"device": "cpu", "cpu_offload": True}, {"cpu_offload": "false"},
    {"prompt": " "}, {"prompt": None}, {"model_path": None}, {"context": []},
    {"context": {"value": float("nan")}}, {"model": "large"},
    {"output_format": "flac"}, {"output_format": None}, {"keep_wav": "false"},
    {"mp3_bitrate": 0}, {"mp3_bitrate": 128}, {"mp3_bitrate": True}, {"mp3_bitrate": 192.5},
])
def test_invalid_requests_rejected_without_inference_dependencies(overrides):
    with pytest.raises(ValueError):
        engine.validate_request(request_data(**overrides))


@pytest.mark.parametrize("bitrate", engine.MP3_BITRATES)
def test_explicit_output_settings_round_trip_and_wav_choice(bitrate):
    request = engine.validate_request(request_data(
        output_format=" MP3 ", keep_wav=True, mp3_bitrate=str(bitrate),
    ))
    assert request.output_format == "mp3"
    assert request.keep_wav is True
    assert request.mp3_bitrate == bitrate
    assert engine.validate_request(request.to_dict()) == request
    assert engine.validate_request(request_data(output_format="wav")).output_format == "wav"


def test_context_copied_and_utf8_json_round_trip(tmp_path):
    context = {"work": "雨の駅", "scene": {"location": "駅前"}}
    request = engine.validate_request(request_data(context=context))
    context["scene"]["location"] = "変更"
    assert request.context["scene"]["location"] == "駅前"
    path = tmp_path / "nested" / "request.json"
    engine.write_json(path, request.to_dict())
    assert "雨の駅" in path.read_text(encoding="utf-8")
    assert not path.with_name(path.name + ".tmp").exists()
    assert engine.validate_request(json.loads(path.read_text(encoding="utf-8"))) == request


def test_local_model_directory_requires_converted_complete_sa3(tmp_path):
    directory = model_directory(tmp_path)
    assert engine.validate_model_directory(str(directory)) == directory.resolve()
    (directory / "vae").rmdir()
    with pytest.raises(engine.AudioGenerationError, match="vae"):
        engine.validate_model_directory(str(directory))
    (directory / "model_index.json").write_text('{"_class_name":"StableAudioPipeline"}')
    with pytest.raises(engine.AudioGenerationError, match="StableAudio3Pipeline"):
        engine.validate_model_directory(str(directory))
    (directory / "model_index.json").unlink()
    with pytest.raises(engine.AudioGenerationError, match="変換"):
        engine.validate_model_directory(str(directory))


def test_preview_wav_validation_detects_truncation_and_wrong_rate(tmp_path):
    path = tmp_path / "output.wav"
    write_preview(path)
    stats = engine.inspect_pcm16_wav(path, 44100)
    assert stats["channels"] == 2
    assert stats["sample_count"] == 8
    with pytest.raises(engine.AudioGenerationError, match="サンプルレート"):
        engine.inspect_pcm16_wav(path, 48000)
    path.write_bytes(path.read_bytes()[:-4])
    with pytest.raises(engine.AudioGenerationError, match="途中"):
        engine.inspect_pcm16_wav(path, 44100)


@pytest.mark.parametrize("shape", [(1, 1, 32), (2, 2, 32), (1, 2, 0)])
def test_bad_model_output_shape_never_writes_wav(tmp_path, shape):
    np = SimpleNamespace(asarray=lambda value: SimpleNamespace(ndim=3, shape=shape))
    with pytest.raises(engine.AudioGenerationError, match="stereo"):
        engine._save_audio(object(), 44100, tmp_path / "output.wav", np)
    assert not (tmp_path / "output.wav").exists()


def test_nonfinite_model_output_never_writes_wav(tmp_path):
    class Array:
        ndim = 3
        shape = (1, 2, 32)

        @property
        def T(self):
            return self

        def __getitem__(self, key):
            return self

        def astype(self, dtype):
            return self

    np = SimpleNamespace(
        asarray=lambda value: Array(),
        float32="float32",
        isfinite=lambda value: SimpleNamespace(all=lambda: False),
    )
    with pytest.raises(engine.AudioGenerationError, match="NaN / Inf"):
        engine._save_audio(object(), 44100, tmp_path / "output.wav", np)
    assert not (tmp_path / "output.wav").exists()


def fake_torch(cuda=False, bf16=False):
    calls = []
    return SimpleNamespace(
        __version__="test-torch",
        float32="fp32", float16="fp16", bfloat16="bf16",
        cuda=SimpleNamespace(
            is_available=lambda: cuda,
            is_bf16_supported=lambda: bf16,
            reset_peak_memory_stats=lambda: calls.append("reset_memory"),
            synchronize=lambda: calls.append("synchronize"),
            max_memory_allocated=lambda: 123456,
        ),
        Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: (kwargs, seed)),
        random=SimpleNamespace(fork_rng=lambda **kwargs: nullcontext()),
        manual_seed=lambda seed: calls.append(("global_seed", seed)),
        inference_mode=nullcontext,
        calls=calls,
    )


def test_auto_cpu_precision_and_cuda_errors():
    request = engine.validate_request(request_data())
    assert engine.resolve_device_dtype(fake_torch(), request) == ("cpu", "float32", "fp32")
    with pytest.raises(engine.AudioGenerationError, match="CUDA"):
        engine.resolve_device_dtype(fake_torch(), engine.validate_request(request_data(device="cuda")))
    with pytest.raises(engine.AudioGenerationError, match="bfloat16"):
        engine.resolve_device_dtype(
            fake_torch(cuda=True), engine.validate_request(request_data(dtype="bfloat16"))
        )
    with pytest.raises(engine.AudioGenerationError, match="オフロード"):
        engine.resolve_device_dtype(fake_torch(), engine.validate_request(request_data(cpu_offload=True)))


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = SimpleNamespace(
        loaded=[], generated=[], moved=[], offload=[], compat=[], saved=[],
        saved_options=[], decoded=[], cropped=[], freed=[],
    )
    torch = fake_torch(cuda=True)

    class DecodedAudio:
        def __getitem__(self, key):
            calls.cropped.append(key)
            return self

    def decode(latents):
        calls.decoded.append(latents)
        return SimpleNamespace(sample=DecodedAudio())

    class Pipeline:
        vae = SimpleNamespace(config=SimpleNamespace(sampling_rate=44100), decode=decode)

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.loaded.append((path, kwargs))
            return cls()

        def set_progress_bar_config(self, **kwargs):
            pass

        def to(self, device):
            calls.moved.append(device)
            return self

        def enable_model_cpu_offload(self, **kwargs):
            calls.offload.append(kwargs)

        def maybe_free_model_hooks(self):
            calls.freed.append(True)

        def __call__(self, **kwargs):
            calls.generated.append(kwargs)
            for step in range(kwargs["num_inference_steps"]):
                assert kwargs["callback_on_step_end"](self, step, None, {"latents": step}) == {"latents": step}
            return SimpleNamespace(audios=object())

    def save(audios, sample_rate, destination, np, **kwargs):
        calls.saved.append(destination)
        calls.saved_options.append(kwargs)
        write_preview(destination, sample_rate)
        return {**engine.inspect_pcm16_wav(destination, sample_rate), "peak": 0.5, "rms": 0.1,
                "silent": False, "clipping_fraction": 0.0, "normalized": False}

    modules = {
        "torch": torch,
        "diffusers": SimpleNamespace(__version__="0.40.0", StableAudio3Pipeline=Pipeline),
        "numpy": SimpleNamespace(__version__="test-numpy"),
        "scripts.audio.same_compat": SimpleNamespace(
            install_same_compat=lambda: calls.compat.append("install"),
            enable_chunked_decode=lambda vae, **kwargs: calls.compat.append("chunked"),
        ),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(engine, "_save_audio", save)
    monkeypatch.setattr(engine, "_check_mp3_tools", lambda: None)
    calls.torch = torch
    calls.diffusers = modules["diffusers"]
    return calls


def test_generation_pipeline_contract_progress_seed_and_saved_metadata(tmp_path, fake_runtime, monkeypatch):
    monkeypatch.setattr(engine.secrets, "randbelow", lambda upper: 31415)
    directory = model_directory(tmp_path)
    progress = []
    result = engine.generate_audio(
        request_data(model_path=str(directory), context={"scene": "窓辺"}),
        tmp_path / "run", progress=progress.append,
    )
    assert result["ok"]
    assert Path(result["audio_path"]).name == "output.mp3"
    assert fake_runtime.saved_options[0]["keep_wav"] is False
    assert fake_runtime.saved_options[0]["mp3_bitrate"] == 192
    assert fake_runtime.loaded[0][1] == {"torch_dtype": "fp32", "local_files_only": True, "use_safetensors": True}
    assert fake_runtime.moved == ["cuda"]
    assert fake_runtime.compat == ["install", "chunked"]
    kwargs = fake_runtime.generated[0]
    assert kwargs["duration"] == 30
    assert kwargs["num_inference_steps"] == 8
    assert kwargs["guidance_scale"] == 1.0
    assert kwargs["silence_padding_duration"] == 0.0
    assert kwargs["generator"] == ({"device": "cuda"}, 31415)
    assert kwargs["output_type"] == "latent"
    assert len(fake_runtime.decoded) == 1
    assert fake_runtime.cropped == [(slice(None), slice(None), slice(None, 30 * 44100))]
    assert fake_runtime.freed == [True]
    assert ("global_seed", 31415) in fake_runtime.torch.calls
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["settings"]["seed"] == 31415
    assert metadata["requested_seed"] == -1
    assert metadata["context"] == {"scene": "窓辺"}
    assert metadata["cuda_peak_memory_bytes"] == 123456
    assert metadata["normalized"] is False
    assert metadata["vae_decode"]["global_rng_seed"] == 31415
    assert metadata["vae_decode"]["clamp_output"] is False
    assert metadata["model_repo"] is None
    assert metadata["selected_model_repo"] == engine.MODEL_SPECS["small"]["repo_id"]
    assert metadata["model_identity"]["verified_by"] == "transformer/config.json"
    assert progress[0]["phase"] == "loading"
    assert [item["step"] for item in progress if item["phase"] == "generating"] == list(range(9))
    assert progress[-1]["phase"] == "done"


def test_cpu_offload_fixed_seed(tmp_path, fake_runtime):
    result = engine.generate_audio(
        request_data(model_path=str(model_directory(tmp_path)), seed=42, cpu_offload=True), tmp_path / "run"
    )
    assert fake_runtime.offload == [{"gpu_id": 0}]
    assert fake_runtime.moved == []
    assert result["metadata"]["settings"]["seed"] == 42


def test_explicit_wav_generation_preserves_output_choice(tmp_path, fake_runtime):
    result = engine.generate_audio(
        request_data(model_path=str(model_directory(tmp_path)), output_format="wav"),
        tmp_path / "run",
    )
    assert Path(result["audio_path"]).name == "output.wav"
    assert result["metadata"]["settings"]["output_format"] == "wav"
    assert result["metadata"]["audio_path"] == result["audio_path"]


def test_mp3_tool_failure_is_reported_before_loading_model(tmp_path, fake_runtime, monkeypatch):
    def missing_tools():
        raise engine.AudioGenerationError("FFmpeg がありません")

    monkeypatch.setattr(engine, "_check_mp3_tools", missing_tools)
    with pytest.raises(engine.AudioGenerationError, match="FFmpeg"):
        engine.generate_audio(
            request_data(model_path=str(model_directory(tmp_path))), tmp_path / "run",
        )
    assert fake_runtime.loaded == []
    assert fake_runtime.saved == []


@pytest.mark.parametrize("existing", ["output.wav", "output.mp3", "output-float.wav", "generation.json"])
def test_generation_never_overwrites_another_output_format(tmp_path, fake_runtime, existing):
    output = tmp_path / "run"
    output.mkdir()
    protected = output / existing
    protected.write_bytes(b"existing result")
    with pytest.raises(engine.AudioGenerationError, match="既存"):
        engine.generate_audio(
            request_data(model_path=str(model_directory(tmp_path))), output,
        )
    assert protected.read_bytes() == b"existing result"
    assert fake_runtime.loaded == []


def test_shared_loader_uses_one_model_and_exposes_generation_components(tmp_path, fake_runtime):
    directory = model_directory(tmp_path)
    loaded = engine.load_pipeline(request_data(model_path=str(directory), cpu_offload=True))
    assert len(fake_runtime.loaded) == 1
    assert fake_runtime.compat == ["install", "chunked"]
    assert fake_runtime.offload == [{"gpu_id": 0}]
    assert loaded.device == "cuda"
    assert loaded.dtype_name == "float32"
    assert loaded.model_path == directory.resolve()
    assert loaded.diffusers_version == "0.40.0"
    assert loaded.pipeline.vae.config.sampling_rate == 44100
    assert loaded.torch is fake_runtime.torch
    assert loaded.np.__version__ == "test-numpy"
    assert loaded.load_seconds >= 0


def test_shared_loader_cancelled_before_model_access(tmp_path, fake_runtime):
    with pytest.raises(engine.GenerationCancelled, match="中止"):
        engine.load_pipeline(request_data(model_path=str(tmp_path / "missing")), cancelled=lambda: True)
    assert fake_runtime.loaded == []


def test_cancelled_step_does_not_save_audio(tmp_path, fake_runtime):
    progress = []
    with pytest.raises(engine.GenerationCancelled):
        engine.generate_audio(
            request_data(model_path=str(model_directory(tmp_path))), tmp_path / "run",
            progress=progress.append, cancelled=lambda: len(progress) >= 4,
        )
    assert fake_runtime.saved == []
    assert not (tmp_path / "run" / "output.wav").exists()


def test_runner_reports_request_failure_without_loading_torch(tmp_path):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request_data(duration=121)))
    output = tmp_path / "run"
    assert runner.run_request(request_path, output) == 1
    result = json.loads((output / "result.json").read_text(encoding="utf-8"))
    status = json.loads((output / "status.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert result["error_type"] == "ValueError"
    assert status["phase"] == "error"
    assert "120" in status["message"]


def test_runner_records_cancellation_and_protects_existing_results(tmp_path, monkeypatch):
    request_path = tmp_path / "request.json"
    request_path.write_text(json.dumps(request_data()))
    output = tmp_path / "run"
    output.mkdir()
    (output / "stop.request").touch()

    def cancelled_generation(request, directory, *, cancelled, progress):
        assert cancelled()
        raise engine.GenerationCancelled("中止")

    monkeypatch.setattr(runner, "generate_audio", cancelled_generation)
    assert runner.run_request(request_path, output) == 130
    result_text = (output / "result.json").read_text(encoding="utf-8")
    assert json.loads(result_text)["cancelled"] is True
    assert runner.run_request(request_path, output) == 2
    assert (output / "result.json").read_text(encoding="utf-8") == result_text


def test_actionable_out_of_memory_message():
    assert "Small" in engine.explain_error(RuntimeError("CUDA out of memory"))


@pytest.mark.parametrize("model", ["small", "medium"])
def test_model_identity_from_transformer_and_preparation_manifest(tmp_path, model):
    directory = model_directory(tmp_path, model)
    identity = engine.validate_model_identity(directory, model)
    assert identity == {"model": model, "repo_id": None, "verified_by": "transformer/config.json"}
    other = "medium" if model == "small" else "small"
    with pytest.raises(engine.AudioGenerationError, match="モデルの選択とパス"):
        engine.validate_model_identity(directory, other)
    manifest = {"model": model, "repo_id": engine.MODEL_SPECS[model]["repo_id"],
                "repo_revision": "f" * 40, "converter_sha256": "a" * 64,
                "same_compat_version": 1}
    (directory / "sa3_preparation.json").write_text(json.dumps(manifest), encoding="utf-8")
    identity = engine.validate_model_identity(directory, model)
    assert identity["repo_id"] == manifest["repo_id"]
    assert identity["verified_by"] == "sa3_preparation.json"
    assert identity["repo_revision"] == manifest["repo_revision"]
    assert identity["converter_sha256"] == manifest["converter_sha256"]
    assert identity["same_compat_version"] == 1


def test_model_identity_rejects_invalid_signature_and_false_manifest(tmp_path):
    directory = model_directory(tmp_path)
    config_path = directory / "transformer" / "config.json"
    config_path.write_text('{"embed_dim":2048}')
    with pytest.raises(engine.AudioGenerationError, match="対応する"):
        engine.validate_model_identity(directory, "small")
    manifest = {"model": "small", "repo_id": engine.MODEL_SPECS["medium"]["repo_id"]}
    (directory / "sa3_preparation.json").write_text(json.dumps(manifest))
    with pytest.raises(engine.AudioGenerationError, match="一致"):
        engine.validate_model_identity(directory, "small")


def test_recent_diffusers_dev_version_accepted(tmp_path, fake_runtime):
    fake_runtime.diffusers.__version__ = "0.41.dev0"
    result = engine.generate_audio(
        request_data(model_path=str(model_directory(tmp_path))), tmp_path / "run"
    )
    assert result["ok"] is True


def test_old_diffusers_rejected_before_model_loading(tmp_path, fake_runtime):
    fake_runtime.diffusers.__version__ = "0.39.0"
    with pytest.raises(engine.AudioGenerationError, match="0.40.0"):
        engine.generate_audio(
            request_data(model_path=str(model_directory(tmp_path))), tmp_path / "run"
        )
    assert fake_runtime.loaded == []


def test_real_numpy_audio_level_and_clipping_preserved(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    samples = np.array([[[0, 0.25, -0.5, 1.25], [0, -0.25, 0.5, -1.25]]], dtype=np.float32)
    destination = tmp_path / "output.wav"
    stats = engine._save_audio(samples, 44100, destination, np)
    assert stats["peak"] == 1.25
    assert stats["clipping_fraction"] == 0.25
    assert stats["rms"] == pytest.approx(float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))))
    assert stats["normalized"] is False
    assert stats["preview_normalized"] is True
    assert stats["preview_gain"] == pytest.approx(0.99 / 1.25)
    assert stats["preview_peak"] == pytest.approx(0.99, abs=1 / 32767)
    assert "自動増幅は行わず" in stats["preview_normalization"]
    assert stats["sample_count"] == 4
    with wave.open(str(destination)) as preview:
        data = np.frombuffer(preview.readframes(4), dtype="<i2").reshape(-1, 2)
    assert data[1, 0] == pytest.approx(0.25 * stats["preview_gain"] * 32767, abs=1)
    assert data[3, 0] == pytest.approx(0.99 * 32767, abs=1)
    assert data[3, 1] == -data[3, 0]
    assert stats["preview_peak"] == float(np.max(np.abs(data))) / 32767
    original, sample_rate = sf.read(stats["float_audio_path"], dtype="float32")
    assert sample_rate == 44100
    assert np.array_equal(original.T, samples[0])
    assert original[3].tolist() == [1.25, -1.25]
    assert sf.info(stats["float_audio_path"]).subtype == "FLOAT"


@pytest.mark.parametrize("peak", [0.001, 0.5, 1.0])
def test_real_numpy_preview_does_not_boost_or_attenuate_in_range_audio(tmp_path, peak):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    samples = np.array([[[0, peak / 2, peak], [0, -peak / 2, -peak]]], dtype=np.float32)
    destination = tmp_path / "output.wav"
    stats = engine._save_audio(samples, 44100, destination, np)
    assert stats["preview_gain"] == 1.0
    assert stats["preview_normalized"] is False
    assert stats["normalized"] is False
    assert stats["clipping_fraction"] == 0
    with wave.open(str(destination)) as preview:
        data = np.frombuffer(preview.readframes(3), dtype="<i2").reshape(-1, 2)
    expected = np.rint(samples[0].T * 32767).astype("<i2")
    assert np.array_equal(data, expected)
    assert stats["preview_peak"] == float(np.max(np.abs(data))) / 32767
    original, _ = sf.read(stats["float_audio_path"], dtype="float32")
    assert np.array_equal(original.T, samples[0])


def test_real_numpy_silence_reported_and_nan_rejected(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    samples = np.zeros((1, 2, 8), dtype=np.float32)
    stats = engine._save_audio(samples, 44100, tmp_path / "silence.wav", np)
    assert stats["silent"] is True
    assert stats["rms"] == 0
    assert stats["preview_gain"] == 1.0
    assert stats["preview_peak"] == 0
    assert stats["preview_normalized"] is False
    samples[0, 0, 0] = np.nan
    with pytest.raises(engine.AudioGenerationError, match="NaN / Inf"):
        engine._save_audio(samples, 44100, tmp_path / "bad.wav", np)
    assert not (tmp_path / "bad.wav").exists()


def _tone_samples(np, duration=1.0, peak=0.8):
    timeline = np.arange(int(44100 * duration), dtype=np.float64) / 44100
    tone = (peak * np.sin(2 * np.pi * 440 * timeline)).astype(np.float32)
    return np.stack((tone, tone * 0.75))[None, :, :]


@pytest.mark.parametrize("keep_wav", [False, True])
def test_real_mp3_encode_has_small_final_file_and_original_metrics(tmp_path, keep_wav):
    np = pytest.importorskip("numpy")
    from scripts.audio import audio_files

    try:
        audio_files.find_ffmpeg()
        audio_files.find_ffprobe()
    except audio_files.AudioFileError as exc:
        pytest.skip(str(exc))
    if keep_wav:
        pytest.importorskip("soundfile")
    samples = _tone_samples(np, peak=1.25)
    destination = tmp_path / "output.mp3"
    stats = engine._save_audio(samples, 44100, destination, np, keep_wav=keep_wav)
    assert stats["preview_format"] == "MP3"
    assert stats["codec"] == "mp3"
    assert stats["bitrate_kbps"] == 192
    assert stats["peak"] == pytest.approx(float(np.max(np.abs(samples))))
    assert stats["peak"] > 1.0
    assert stats["preview_gain"] == pytest.approx(0.99 / stats["peak"])
    assert stats["rms"] == pytest.approx(float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))))
    assert stats["sample_count"] == 44100
    assert stats["duration_seconds"] == 1.0
    assert stats["source_sample_count"] == stats["sample_count"]
    assert stats["raw_audio_retained"] is keep_wav
    assert stats["audio_bytes"] == destination.stat().st_size
    assert stats["audio_bytes"] < samples.size * 2 / 3
    probe = audio_files.probe_audio(destination)
    assert probe["codec"] == "mp3"
    assert probe["sample_rate"] == 44100
    assert probe["channels"] == 2
    decoded = audio_files.decode_audio(destination)
    assert decoded.shape == (44100, 2)
    assert (tmp_path / "output.wav").exists() is keep_wav
    assert (tmp_path / "output-float.wav").exists() is keep_wav
    if keep_wav:
        sf = pytest.importorskip("soundfile")
        original, _ = sf.read(stats["float_audio_path"], dtype="float32")
        assert np.array_equal(original.T, samples[0])
    else:
        assert "float_audio_path" not in stats
        assert list(tmp_path.iterdir()) == [destination]
    assert not list(tmp_path.glob(".*.wav"))


@pytest.mark.parametrize("keep_wav", [False, True])
def test_cancelled_mp3_encode_removes_source_temporaries_and_unpublished_wavs(tmp_path, monkeypatch, keep_wav):
    np = pytest.importorskip("numpy")
    if keep_wav:
        pytest.importorskip("soundfile")

    class Cancelled(RuntimeError):
        pass

    def encode(source, destination, **kwargs):
        assert source.exists()
        assert not destination.exists()
        assert kwargs["sample_rate"] == 44100
        assert kwargs["bitrate_kbps"] == 256
        raise Cancelled("MP3 保存を中止しました")

    monkeypatch.setattr(engine, "_audio_files", lambda: SimpleNamespace(
        encode_mp3=encode, ExportCancelled=Cancelled, AudioFileError=RuntimeError,
    ))
    with pytest.raises(engine.GenerationCancelled, match="中止"):
        engine._save_audio(
            _tone_samples(np), 44100, tmp_path / "output.mp3", np,
            keep_wav=keep_wav, mp3_bitrate=256,
        )
    assert list(tmp_path.iterdir()) == []


def test_failed_mp3_encode_never_leaves_a_final_audio(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")

    class ExportError(RuntimeError):
        pass

    class Cancelled(RuntimeError):
        pass

    def encode(source, destination, **kwargs):
        assert source.exists()
        raise ExportError("encoder failed")

    monkeypatch.setattr(engine, "_audio_files", lambda: SimpleNamespace(
        encode_mp3=encode, ExportCancelled=Cancelled, AudioFileError=ExportError,
    ))
    with pytest.raises(engine.AudioGenerationError, match="encoder failed"):
        engine._save_audio(_tone_samples(np), 44100, tmp_path / "output.mp3", np)
    assert list(tmp_path.iterdir()) == []


def test_saving_audio_preserves_preexisting_mp3(tmp_path):
    np = pytest.importorskip("numpy")
    destination = tmp_path / "output.mp3"
    destination.write_bytes(b"existing mp3")
    with pytest.raises(engine.AudioGenerationError, match="既存"):
        engine._save_audio(_tone_samples(np), 44100, destination, np)
    assert destination.read_bytes() == b"existing mp3"
    assert list(tmp_path.iterdir()) == [destination]


def test_distinct_wav_candidates_keep_separate_raw_waveforms(tmp_path):
    np = pytest.importorskip("numpy")
    sf = pytest.importorskip("soundfile")
    samples = _tone_samples(np)
    first = engine._save_audio(samples, 44100, tmp_path / "candidate-a.wav", np)
    second = engine._save_audio(samples / 2, 44100, tmp_path / "candidate-b.wav", np)
    assert Path(first["float_audio_path"]).name == "candidate-a-float.wav"
    assert Path(second["float_audio_path"]).name == "candidate-b-float.wav"
    original_a, _ = sf.read(first["float_audio_path"], dtype="float32")
    original_b, _ = sf.read(second["float_audio_path"], dtype="float32")
    assert np.array_equal(original_a.T, samples[0])
    assert np.array_equal(original_b.T, samples[0] / 2)
