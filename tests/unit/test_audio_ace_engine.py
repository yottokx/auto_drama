from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.audio import ace_backend, engine


def request_data(**overrides):
    return {"model": "ace15_turbo", "model_path": "models/ace15_turbo",
            "prompt": "Bright quirky modern pop with syncopated bass and playful synths.", **overrides}


def model_directory(tmp_path: Path, *, manifest: bool = False) -> Path:
    directory = tmp_path / "ace"
    directory.mkdir()
    (directory / "model_index.json").write_text('{"_class_name":"AceStepPipeline"}')
    for name in ("transformer", "condition_encoder", "vae", "text_encoder", "tokenizer", "scheduler"):
        (directory / name).mkdir()
    config = {
        "hidden_size": 2048, "intermediate_size": 6144, "num_hidden_layers": 24,
        "num_attention_heads": 16, "num_key_value_heads": 8, "head_dim": 128,
        "in_channels": 192, "audio_acoustic_hidden_dim": 64,
        "is_turbo": True, "model_version": "turbo",
    }
    (directory / "transformer" / "config.json").write_text(json.dumps(config))
    if manifest:
        (directory / "ace_preparation.json").write_text(json.dumps({
            "model": "ace15_turbo", "repo_id": "ACE-Step/Ace-Step1.5",
            "dit_config": "acestep-v15-turbo", "repo_revision": "1" * 40,
            "converter_sha256": "2" * 64, "dtype": "bfloat16",
        }))
    return directory


def test_ace_request_defaults_bounds_and_metadata_round_trip():
    request = engine.validate_request(request_data())
    assert request.duration == 30 and request.steps == 8
    assert request.bpm is None and request.keyscale is None and request.timesignature is None
    assert engine.validate_request(request_data(duration=10)).duration == 10
    assert engine.validate_request(request_data(duration=600)).duration == 600
    specified = engine.validate_request(request_data(bpm=104, keyscale=" F# major ", timesignature=" 4 "))
    assert (specified.bpm, specified.keyscale, specified.timesignature) == (104, "F# major", "4")
    assert engine.validate_request(specified.to_dict()) == specified
    assert engine.validate_request(request_data(bpm=0, keyscale=" ", timesignature="")).bpm is None
    assert engine.MODEL_SPECS[request.model]["dit_config"] == "acestep-v15-turbo"


@pytest.mark.parametrize("overrides", [
    {"duration": 9.999}, {"duration": 600.1}, {"bpm": 29}, {"bpm": 301}, {"bpm": True},
    {"bpm": 104.5}, {"keyscale": "C dorian"}, {"keyscale": 0},
    {"timesignature": "4/4"}, {"timesignature": 4},
    {"lyrics": "[Verse]\nI sing"}, {"lyrics": ""}, {"vocal_language": "en"},
    {"task_type": "cover"}, {"audio_codes": "<|audio_code_12|>"}, {"audio_codes": ""},
    {"planner": True}, {"use_planner": True}, {"thinking": True}, {"use_lm": True},
    {"planner_model": "lm"}, {"lm_model_path": "models/lm"}, {"use_cot_caption": True},
])
def test_ace_rejects_invalid_or_contradictory_instrumental_requests(overrides):
    with pytest.raises(ValueError):
        engine.validate_request(request_data(**overrides))


def test_explicit_instrumental_no_planner_settings_are_accepted():
    request = engine.validate_request(request_data(
        lyrics="[Instrumental]", vocal_language="unknown", task_type="text2music",
        audio_codes=None, planner=False, use_lm=False, thinking=False,
    ))
    contract = ace_backend.generation_contract(request)
    assert contract["planner_enabled"] is False
    assert contract["audio_codes"] is None
    assert contract["lyrics"] == "[Instrumental]"
    assert contract["caption"].startswith("Instrumental background music. No vocals")


def test_shared_stable_caption_loses_only_known_stable_tags():
    music = "Quirky modern pop, bright synth bass, handclaps, 112 BPM in D major."
    shared = "TrackType: Music, VocalType: Instrumental. " + music
    caption = ace_backend.instrumental_caption(shared)
    assert caption.endswith(music)
    assert "TrackType:" not in caption and "VocalType:" not in caption
    assert caption.count("112 BPM") == caption.count("D major") == 1
    assert ace_backend.instrumental_caption(music) == caption


def test_continuity_condition_preserves_genre_energy_and_allows_melodic_rests():
    music = "Harsh industrial metal, sharp percussion and dissonant aggressive guitars."
    caption = ace_backend.instrumental_caption(music, continuous=True)
    assert music in caption and caption.index(music) < caption.index(ace_backend.CONTINUITY)
    assert "Keep any lead melody prominent, with supportive accompaniment" in caption
    assert "natural phrasing, brief rests and transitions" in caption
    assert "extended silent breaks" in caption and "premature endings" in caption
    assert ace_backend.CONTINUITY not in ace_backend.instrumental_caption(music)
    request = engine.validate_request(request_data(prompt=music, ace_continuous=True))
    assert request.ace_continuous is True
    contract = ace_backend.generation_contract(request)
    assert contract["caption"] == caption and contract["continuous_accompaniment"] is True
    assert engine.validate_request(request.to_dict()).ace_continuous is True


def test_continuity_keeps_named_lead_before_support_and_preserves_manual_drone():
    melodic = (
        "Genre: Industrial Electronic. Instruments: abrasive lead synth. "
        "The synth plays an angular recurring melody."
    )
    drone = "Genre: Dark Ambient. Instruments: sustained dissonant synthesizer drone. Nonmelodic texture."
    for music in (melodic, drone):
        caption = ace_backend.instrumental_caption(music, continuous=True)
        assert caption == ace_backend.instrumental_caption(music) + " " + ace_backend.CONTINUITY
    assert "any lead melody" in ace_backend.CONTINUITY


@pytest.mark.parametrize("change", [{"ace_continuous": "true"}, {"ace_continuous": 1},
                                    {"ace_continuous": True, "model": "medium"}])
def test_invalid_continuity_flags_are_rejected(change):
    with pytest.raises(ValueError):
        engine.validate_request(request_data(**change))


def test_directory_allows_unused_semantic_codecs_to_be_absent(tmp_path):
    directory = model_directory(tmp_path)
    assert engine.validate_model_directory(str(directory), "ace15_turbo") == directory.resolve()
    with pytest.raises(engine.AudioGenerationError, match="StableAudio3Pipeline"):
        engine.validate_model_directory(str(directory), "small")
    (directory / "condition_encoder").rmdir()
    with pytest.raises(engine.AudioGenerationError, match="condition_encoder"):
        engine.validate_model_directory(str(directory), "ace15_turbo")


def test_ace_selection_rejects_stable_audio_folder(tmp_path):
    directory = tmp_path / "stable"
    directory.mkdir()
    (directory / "model_index.json").write_text('{"_class_name":"StableAudio3Pipeline"}')
    with pytest.raises(engine.AudioGenerationError, match="Stable Audio 3"):
        engine.validate_model_directory(str(directory), "ace15_turbo")


def test_ace_identity_with_and_without_preparation_manifest(tmp_path):
    directory = model_directory(tmp_path, manifest=True)
    identity = engine.validate_model_identity(directory, "ace15_turbo")
    assert identity["verified_by"] == "ace_preparation.json"
    assert identity["repo_id"] == "ACE-Step/Ace-Step1.5"
    assert identity["repo_revision"] == "1" * 40
    (directory / "ace_preparation.json").unlink()
    assert engine.validate_model_identity(directory, "ace15_turbo") == {
        "model": "ace15_turbo", "repo_id": None, "verified_by": "transformer/config.json",
    }


def test_ace_identity_rejects_xl_even_with_valid_manifest(tmp_path):
    directory = model_directory(tmp_path, manifest=True)
    config_path = directory / "transformer" / "config.json"
    config = json.loads(config_path.read_text())
    config["hidden_size"] = 2560
    config_path.write_text(json.dumps(config))
    with pytest.raises(engine.AudioGenerationError, match="2B"):
        engine.validate_model_identity(directory, "ace15_turbo")


@pytest.mark.parametrize("change", [
    {"repo_id": "ACE-Step/acestep-v15-xl-turbo-diffusers"},
    {"model": "medium"}, {"dit_config": "acestep-v15-xl-turbo"},
])
def test_ace_identity_rejects_false_provenance(tmp_path, change):
    directory = model_directory(tmp_path, manifest=True)
    manifest_path = directory / "ace_preparation.json"
    manifest = json.loads(manifest_path.read_text())
    manifest.update(change)
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(engine.AudioGenerationError):
        engine.validate_model_identity(directory, "ace15_turbo")


@pytest.fixture
def fake_runtime(monkeypatch):
    calls = SimpleNamespace(
        loaded=[], generated=[], moved=[], offload=[], saved=[], cropped=[], freed=[],
        output_shape=(1, 2, 30 * 48000 + 1920), cuda=True, bf16=True,
    )
    torch = SimpleNamespace(
        __version__="test-torch", float32="fp32", float16="fp16", bfloat16="bf16",
        cuda=SimpleNamespace(
            is_available=lambda: calls.cuda, is_bf16_supported=lambda: calls.bf16,
            reset_peak_memory_stats=lambda: None, synchronize=lambda: None,
            max_memory_allocated=lambda: 987654,
        ),
        Generator=lambda **kwargs: SimpleNamespace(manual_seed=lambda seed: (kwargs, seed)),
        random=SimpleNamespace(fork_rng=lambda **kwargs: nullcontext()),
        manual_seed=lambda seed: None, inference_mode=nullcontext,
    )

    class Audio:
        @property
        def shape(self):
            return calls.output_shape

        def __getitem__(self, key):
            calls.cropped.append(key)
            return self

    class Pipeline:
        sample_rate = 48000

        def __init__(self):
            self.vae = SimpleNamespace()
            self.audio_tokenizer = object()
            self.audio_token_detokenizer = object()

        @classmethod
        def from_pretrained(cls, path, **kwargs):
            calls.loaded.append((path, kwargs))
            calls.pipeline = cls()
            return calls.pipeline

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
                values = {"latents": step}
                assert kwargs["callback_on_step_end"](self, step, None, values) == values
            return SimpleNamespace(audios=Audio())

    def save(audio, sample_rate, destination, np, **kwargs):
        calls.saved.append((audio, sample_rate, destination, kwargs))
        destination.write_bytes(b"fake audio")
        return {"sample_rate": sample_rate, "channels": 2, "sample_count": 30 * 48000,
                "duration_seconds": 30.0, "peak": 0.891, "rms": 0.1, "normalized": False}

    modules = {
        "torch": torch,
        "diffusers": SimpleNamespace(__version__="0.40.0", AceStepPipeline=Pipeline),
        "numpy": SimpleNamespace(__version__="test-numpy"),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(engine, "_save_audio", save)
    monkeypatch.setattr(engine, "_check_mp3_tools", lambda: None)
    calls.diffusers = modules["diffusers"]
    calls.torch = torch
    return calls


def test_ace_pipeline_contract_instrumental_seed_crop_and_metadata(tmp_path, fake_runtime):
    progress = []
    result = engine.generate_audio(request_data(
        model_path=str(model_directory(tmp_path, manifest=True)), seed=42,
        bpm=112, keyscale="D major", timesignature="4", context={"scene": "ギャグ"},
    ), tmp_path / "run", progress=progress.append)
    assert result["ok"]
    assert fake_runtime.loaded[0][1] == {
        "torch_dtype": "bf16", "local_files_only": True, "use_safetensors": True,
        "audio_tokenizer": None, "audio_token_detokenizer": None,
    }
    assert fake_runtime.moved == ["cuda"]
    vae = fake_runtime.pipeline.vae
    assert vae.use_tiling is True and (vae.tile_latent_min_length, vae.tile_latent_overlap) == (512, 64)
    kwargs = fake_runtime.generated[0]
    assert kwargs["lyrics"] == "[Instrumental]"
    assert kwargs["vocal_language"] == "unknown"
    assert kwargs["task_type"] == "text2music" and kwargs["audio_codes"] is None
    assert kwargs["audio_duration"] == 30 and kwargs["num_inference_steps"] == 8
    assert kwargs["guidance_scale"] == 1.0 and kwargs["shift"] == 3.0
    assert kwargs["output_type"] == "pt"
    assert kwargs["generator"] == ({"device": "cuda"}, 42)
    assert (kwargs["bpm"], kwargs["keyscale"], kwargs["timesignature"]) == (112, "D major", "4")
    assert "thinking" not in kwargs and "planner" not in kwargs
    assert fake_runtime.cropped == [(slice(None), slice(None), slice(None, 30 * 48000))]
    assert fake_runtime.saved[0][1] == 48000
    assert fake_runtime.freed == [True]
    metadata = json.loads(Path(result["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["settings"]["dtype"] == "bfloat16"
    assert metadata["context"] == {"scene": "ギャグ"}
    assert metadata["sample_rate"] == 48000
    assert metadata["ace_step"]["planner_enabled"] is False
    assert metadata["ace_step"]["semantic_codec_loaded"] is False
    assert metadata["vae_decode"]["pipeline_peak_normalization_dbfs"] == -1
    assert metadata["model_repo"] == "ACE-Step/Ace-Step1.5"
    assert [item["phase"] for item in progress][-3:] == ["decoding", "saving", "done"]


def test_ace_cpu_precision_and_offload_contract(tmp_path, fake_runtime):
    directory = model_directory(tmp_path)
    loaded = engine.load_pipeline(request_data(model_path=str(directory), cpu_offload=True))
    assert loaded.dtype_name == "bfloat16"
    assert fake_runtime.offload == [{"gpu_id": 0}] and fake_runtime.moved == []
    cpu_request = engine.validate_request(request_data(device="cpu"))
    assert engine.resolve_device_dtype(fake_runtime.torch, cpu_request) == ("cpu", "float32", "fp32")
    fake_runtime.bf16 = False
    with pytest.raises(engine.AudioGenerationError, match="bfloat16"):
        engine.resolve_device_dtype(fake_runtime.torch, engine.validate_request(request_data()))


@pytest.mark.parametrize("overrides", [
    {"ace_planner": "true"}, {"ace_planner": 1}, {"planner_model_path": 10},
    {"ace_planner": True}, {"ace_planner": True, "model": "medium", "planner_model_path": "lm"},
])
def test_planner_requires_boolean_local_path_and_ace_backend(overrides):
    with pytest.raises(ValueError):
        engine.validate_request(request_data(**overrides))


def test_planner_finishes_before_diffusers_load_and_uses_same_seed(tmp_path, fake_runtime, monkeypatch):
    from scripts.audio import ace_planner_bridge

    directory = model_directory(tmp_path)
    for component in ("audio_tokenizer", "audio_token_detokenizer"):
        (directory / component).mkdir()
        (directory / component / "config.json").write_text("{}")
    order = []
    original_load = engine.load_pipeline
    codes = "<|audio_code_12|>" * 150

    def planner(request, run, **kwargs):
        assert request.seed == 12345 and not fake_runtime.loaded
        assert not fake_runtime.generated
        order.append("planner_exit")
        return {"ok": True, "audio_codes": codes, "planner_released": True, "code_count": 150}

    def load(*args, **kwargs):
        assert order == ["planner_exit"]
        order.append("diffusers_load")
        return original_load(*args, **kwargs)

    monkeypatch.setattr(engine.secrets, "randbelow", lambda _: 12345)
    monkeypatch.setattr(ace_planner_bridge, "run_planner", planner)
    monkeypatch.setattr(engine, "load_pipeline", load)
    result = engine.generate_audio(request_data(
        model_path=str(directory), seed=-1, ace_planner=True, planner_model_path="local-lm",
    ), tmp_path / "run")
    assert order == ["planner_exit", "diffusers_load"]
    assert "audio_tokenizer" not in fake_runtime.loaded[0][1]
    assert "audio_token_detokenizer" not in fake_runtime.loaded[0][1]
    kwargs = fake_runtime.generated[0]
    assert kwargs["audio_codes"] == codes
    assert kwargs["lyrics"] == "[Instrumental]" and kwargs["vocal_language"] == "unknown"
    assert kwargs["generator"] == ({"device": "cuda"}, 12345)
    metadata = result["metadata"]
    assert metadata["requested_seed"] == -1 and metadata["settings"]["seed"] == 12345
    assert metadata["ace_step"]["semantic_codec_loaded"] is True
    assert metadata["ace_step"]["task_type"] == "cover"
    assert metadata["ace_step"]["audio_codes"] == codes


def test_failed_planner_never_loads_dit(tmp_path, fake_runtime, monkeypatch):
    from scripts.audio import ace_planner_bridge

    def planner(*args, **kwargs):
        raise engine.AudioGenerationError("planner failed")

    monkeypatch.setattr(ace_planner_bridge, "run_planner", planner)
    with pytest.raises(engine.AudioGenerationError, match="planner failed"):
        engine.generate_audio(request_data(
            model_path=str(model_directory(tmp_path)), ace_planner=True, planner_model_path="lm",
        ), tmp_path / "run")
    assert not fake_runtime.loaded and not fake_runtime.saved


def test_planner_requires_prepared_semantic_codecs(tmp_path, fake_runtime):
    with pytest.raises(engine.AudioGenerationError, match="audio_tokenizer"):
        engine.load_pipeline(request_data(
            model_path=str(model_directory(tmp_path)), ace_planner=True, planner_model_path="lm",
        ))
    assert not fake_runtime.loaded


def test_ace_cancelled_step_frees_model_and_never_saves(tmp_path, fake_runtime):
    progress = []
    with pytest.raises(engine.GenerationCancelled):
        engine.generate_audio(
            request_data(model_path=str(model_directory(tmp_path))), tmp_path / "run",
            progress=progress.append, cancelled=lambda: len(progress) >= 4,
        )
    assert fake_runtime.saved == [] and fake_runtime.freed == [True]
    assert not (tmp_path / "run" / "generation.json").exists()


@pytest.mark.parametrize("shape", [(1, 1, 30 * 48000), (2, 2, 30 * 48000), (1, 2, 30 * 48000 - 1)])
def test_ace_bad_channels_batch_or_short_output_are_not_published(tmp_path, fake_runtime, shape):
    fake_runtime.output_shape = shape
    with pytest.raises(engine.AudioGenerationError):
        engine.generate_audio(request_data(model_path=str(model_directory(tmp_path))), tmp_path / "run")
    assert fake_runtime.saved == [] and fake_runtime.freed == [True]


def test_ace_old_diffusers_rejected_before_loading(tmp_path, fake_runtime):
    fake_runtime.diffusers.__version__ = "0.39.0"
    with pytest.raises(engine.AudioGenerationError, match="0.40.0"):
        engine.load_pipeline(request_data(model_path=str(model_directory(tmp_path))))
    assert fake_runtime.loaded == []


def test_real_numpy_ace_output_is_cropped_and_saved_at_48k(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("soundfile")
    original = np.full((1, 2, 10 * 48000 + 1920), 0.05, dtype=np.float32)
    calls = []

    class Pipeline:
        sample_rate = 48000

        def __call__(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(audios=original)

        def maybe_free_model_hooks(self):
            pass

    request = engine.validate_request(request_data(duration=10, output_format="wav"))
    audio = ace_backend.infer_audio(
        SimpleNamespace(pipeline=Pipeline()), request, generator=object(),
        on_step_end=lambda *args: args[-1], check_cancelled=lambda: None,
        report=lambda *args, **kwargs: None,
    )
    assert audio.shape == (1, 2, 480000)
    stats = engine._save_audio(audio, 48000, tmp_path / "output.wav", np)
    assert stats["sample_count"] == 480000 and stats["duration_seconds"] == 10
    assert stats["sample_rate"] == 48000 and stats["preview_gain"] == 1.0
    assert np.array_equal(audio, original[:, :, :480000])
