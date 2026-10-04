"""CPU-only checks for loop timing, bounded repair, provenance and cancellation."""

from __future__ import annotations

import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.audio import engine, loop_ai, loop_runner, loops


@pytest.fixture
def np():
    return pytest.importorskip("numpy")


def settings(tmp_path, **updates):
    return {"source_audio": str(tmp_path / "source.wav"), "source_generation": {
        "settings": {"model": "medium", "seed": 17, "prompt": "Genre: Pop. Instruments: Piano, Bass. Bright music."},
        "context": {"scene_label": "再会", "story_context": {"sources": {"narrative_artifact_id": "fixed"}}},
    }, **updates}


def stereo(np, sample_rate=100, seconds=80):
    time = np.arange(sample_rate * seconds, dtype=np.float32) / sample_rate
    wave = (0.2 * np.sin(2 * np.pi * 3 * time)).astype(np.float32)
    return np.stack((wave, wave * 0.8), axis=1)


def features(np, seconds=80, fade_from=None):
    frames = seconds * 10 + 1
    rms = np.full(frames, 0.1)
    if fade_from is not None:
        rms[round(fade_from * 10):] = 0
    return {"beat_times": np.arange(0, seconds + 0.1, 0.5), "beat_source": "detected_beats",
            "tempo_bpm": 120.0, "frame_seconds": 0.1, "chroma": np.ones((12, frames)),
            "mfcc": np.zeros((12, frames)), "rms": rms, "onset": np.ones(frames), "librosa_version": "fixture"}


def test_request_metadata_isolated_and_file_result_unwrapped(tmp_path):
    data = settings(tmp_path)
    request = loops.validate_loop_request(data)
    data["source_generation"]["context"]["scene_label"] = "changed"
    assert request.source_generation["context"]["scene_label"] == "再会"
    metadata = tmp_path / "result.json"
    metadata.write_text(json.dumps({"metadata": request.source_generation}), encoding="utf-8")
    assert loops.validate_loop_request(settings(tmp_path, source_generation=str(metadata))).source_generation == request.source_generation


@pytest.mark.parametrize("updates", [
    {"source_audio": ""}, {"method": "unknown"}, {"target_duration": 0},
    {"target_duration": 381}, {"target_duration": float("nan")}, {"crossfade_seconds": 0},
    {"crossfade_seconds": 9}, {"candidate_count": 4}, {"candidate_count": 1.5},
    {"seed": -2}, {"seed": 2**32}, {"keep_wav": "false"}, {"source_generation": []},
    {"output_format": "flac"}, {"mp3_bitrate": 128}, {"steps": 0}, {"bridge_seconds": 9},
])
def test_invalid_requests_fail_before_any_audio_or_gpu(tmp_path, updates):
    with pytest.raises((ValueError, TypeError)):
        loops.validate_loop_request(settings(tmp_path, **updates))


@pytest.mark.parametrize("method", ["ai_bridge", "smart_ai_bridge"])
def test_ace_model_is_not_loaded_into_stable_audio_inpaint_pipeline(tmp_path, method):
    with pytest.raises(ValueError, match="Stable Audio"):
        loops.validate_loop_request(settings(tmp_path, model="ace15_turbo", method=method))


@pytest.mark.parametrize("method", ["whole_crossfade", "smart_region"])
def test_ace_audio_can_use_pcm_loop_methods(tmp_path, method):
    request = loops.validate_loop_request(settings(tmp_path, model="ace15_turbo", method=method))
    assert request.method == method


def test_whole_crossfade_has_exact_reported_shortening_and_continuous_rotated_boundary(np):
    audio = stereo(np, seconds=10)
    original = audio.copy()
    result, info = loops.whole_crossfade(audio, 100, 0.5, np)
    assert len(result) == len(audio) - 50
    assert info["overlap_shortening_seconds"] == 0.5
    assert info["period_preserved"] is False
    assert np.array_equal(result[0], audio[50])
    assert np.array_equal(result[-1], audio[49])
    assert np.array_equal(result[-50], audio[-50])
    assert np.array_equal(audio, original)


def test_region_crossfade_preserves_beat_period_and_untouched_audio(np):
    audio = stereo(np)
    output = loops.region_crossfade(audio, 1000, 4000, 100, np)
    assert len(output) == 3000
    assert np.array_equal(output[:-100], audio[1000:3900])
    assert np.array_equal(output[-1], audio[999])
    assert np.array_equal(output[0], audio[1000])
    with pytest.raises(engine.AudioGenerationError):
        loops.region_crossfade(audio, 10, 1000, 100, np)


@pytest.mark.parametrize("method", ["whole_crossfade", "smart_region", "ai_bridge", "smart_ai_bridge"])
def test_intro_once_layout_preserves_opening_and_jumps_to_adjacent_source_sample(np, method):
    audio = stereo(np, seconds=60)
    original = audio.copy()
    if method == "whole_crossfade":
        circular, info = loops.whole_crossfade(audio, 100, 0.5, np)
        expected_start, expected_end, untouched_end, source_end = 50, 6000, 5950, 6000
    elif method == "smart_region":
        circular = loops.region_crossfade(audio, 1000, 4000, 50, np)
        info = {"start": 1000, "end": 4000, "source_start_seconds": 10.0, "source_end_seconds": 40.0}
        expected_start, expected_end, untouched_end, source_end = 1000, 4000, 3950, 4000
    else:
        source = audio if method == "ai_bridge" else audio[1000:4000]
        reference, center, start, end = loop_ai.bridge_reference(source, 100, 15, 4, np)
        circular, info = loop_ai.apply_bridge(source, reference, reference + 0.25, center, start, end, 50, np)
        if method == "ai_bridge":
            info.update(source_start_seconds=0.0, source_end_seconds=60.0)
            expected_start, expected_end, untouched_end, source_end = 250, 6250, 5750, 6000
        else:
            info.update(start=1000, end=4000, source_start_seconds=10.0, source_end_seconds=40.0)
            expected_start, expected_end, untouched_end, source_end = 1250, 4250, 3750, 4000
    output, info = loops._intro_once_variant(audio, circular, info, method, 100, np)
    assert np.array_equal(output[:untouched_end], original[:untouched_end])
    assert np.array_equal(output[-1], original[expected_start - 1])
    assert np.array_equal(output[expected_start], original[expected_start])
    assert np.array_equal(audio, original)
    assert len(output) == expected_end
    assert info["start_sample"] == expected_start and info["end_sample"] == expected_end
    assert info["source_end_sample"] == source_end
    assert info["start_seconds"] == expected_start / 100
    assert info["end_seconds"] == expected_end / 100
    assert info["period_samples"] == expected_end - expected_start
    assert info["period_seconds"] == (expected_end - expected_start) / 100
    assert info["file_end_sample"] == len(output)
    assert info["format_version"] == 2 and info["intro_once"]
    assert info["playback_layout"] == "intro_then_loop" and info["intro_pcm_preserved"]
    seam = loops.seam_metrics(output[expected_start:expected_end], 100, np)
    expected_jump = float(np.max(np.abs(original[expected_start] - original[expected_start - 1])))
    assert seam["boundary_jump"] == pytest.approx(expected_jump)
    if method in loops.AI_METHODS:
        assert info["repair_coordinate_system"] == "tail_plus_head_reference"
        assert info["repair_file_start_sample"] == source_end - 200
        assert info["repair_file_end_sample"] == source_end + 200
        assert info["edited_file_start_sample"] == source_end - 250
        assert info["edited_file_end_sample"] == expected_end


def test_smart_regions_use_detected_beats_avoid_fade_and_keep_distinct_top_three(np):
    audio = stereo(np)
    selected = loops.select_regions(audio, 100, 30, 0.5, 3, np, analysis=features(np, fade_from=70))
    assert len(selected) == 3
    assert len({(row["start"], row["end"]) for row in selected}) == 3
    for row in selected:
        assert row["start"] % 50 == row["end"] % 50 == 0
        assert row["end"] < 7100
        assert row["end"] - row["start"] == 3000
        assert row["period_preserved"]
        assert row["beat_source"] == "detected_beats"
        assert row["confidence"] == "moderate"
        assert "heuristic_selection_unrated" in row["quality_flags"]


def test_smart_target_clamps_to_short_source_and_missing_beats_are_honest(np):
    audio = stereo(np, seconds=30)
    analysis = features(np, seconds=30)
    analysis["beat_source"] = "feature_frames_no_reliable_beat"
    selected = loops.select_regions(audio, 100, 60, 0.5, 3, np, analysis=analysis)
    for row in selected:
        assert row["effective_target_duration"] < 30
        assert {"no_reliable_beat", "target_duration_clamped_to_source"} <= set(row["quality_flags"])
        assert row["confidence"] == "low"


def test_smart_ai_regions_respect_ten_second_context_requirement(np):
    audio = stereo(np)
    selected = loops.select_regions(audio, 100, 10, 0.5, 1, np,
                                    analysis=features(np), minimum_duration=10.0)
    assert selected[0]["end"] - selected[0]["start"] >= 1000
    with pytest.raises(engine.AudioGenerationError, match="10秒"):
        loops.select_regions(stereo(np, seconds=10), 100, 10, 0.5, 1, np,
                             analysis=features(np, seconds=10), minimum_duration=10.0)


def test_silent_and_nonfinite_sources_rejected(np):
    with pytest.raises(engine.AudioGenerationError, match="無音"):
        loops.validate_pcm(np.zeros((20, 2)), np)
    bad = stereo(np, seconds=10)
    bad[1, 0] = np.nan
    with pytest.raises(engine.AudioGenerationError, match="NaN"):
        loops.validate_pcm(bad, np)


def test_ai_bridge_changes_only_bounded_hole_and_blended_shoulders(np):
    audio = stereo(np, seconds=60)
    original = audio.copy()
    reference, center, start, end = loop_ai.bridge_reference(audio, 100, 15, 4, np)
    assert len(reference) == 3000 and center == 1500 and end - start == 400
    generated = reference + 0.25
    output, info = loop_ai.apply_bridge(audio, reference, generated, center, start, end, 50, np)
    assert len(output) == len(audio)
    assert np.array_equal(output[250:-250], original[250:-250])
    assert np.array_equal(output[0], generated[center])
    assert np.array_equal(output[250], original[250])
    assert info["edited_sample_count"] == 500 and info["outside_pcm_preserved"]
    assert np.array_equal(audio, original)
    generated[0, 0] = np.inf
    with pytest.raises(engine.AudioGenerationError, match="数値"):
        loop_ai.apply_bridge(audio, reference, generated, center, start, end, 50, np)


def test_bridge_requires_ten_seconds_of_context(np):
    with pytest.raises(engine.AudioGenerationError, match="10秒"):
        loop_ai.bridge_reference(stereo(np, seconds=9), 100, 15, 4, np)


@pytest.mark.parametrize("method", ["ai_bridge", "smart_ai_bridge"])
def test_ai_pipeline_receives_saved_prompt_and_mask_and_keeps_original_external_pcm(tmp_path, monkeypatch, np, method):
    torch = pytest.importorskip("torch")
    original = stereo(np, seconds=60)
    calls = []
    freed = []
    loads = []
    config = SimpleNamespace(local_add_cond_dim=257, patch_size=1)
    vae = SimpleNamespace(config=SimpleNamespace(latent_dim=256),
                          decode=lambda latents: SimpleNamespace(sample=latents + 0.25))
    components = {"vae": vae, "transformer": SimpleNamespace(config=config)}
    base = SimpleNamespace(components=components, maybe_free_model_hooks=lambda: freed.append("base"))

    class Inpaint:
        def __init__(self, **supplied):
            assert supplied == components
            self.vae = supplied["vae"]

        def set_progress_bar_config(self, **kwargs):
            pass

        def __call__(self, **kwargs):
            calls.append(kwargs)
            kwargs["callback_on_step_end"](self, 0, None, {"latents": kwargs["audio"]})
            return SimpleNamespace(audios=kwargs["audio"])

        def maybe_free_model_hooks(self):
            freed.append("inpaint")

    def loaded(request, *, cancelled):
        loads.append(request)
        assert request.prompt == settings(tmp_path)["source_generation"]["settings"]["prompt"]
        assert request.duration == 30 and request.seed == 4
        return SimpleNamespace(pipeline=base, torch=torch, device="cpu", dtype_name="float32",
                               model_identity={"model": "medium"}, load_seconds=0)

    real_import = loop_ai.importlib.import_module
    monkeypatch.setattr(loop_ai, "load_pipeline", loaded)
    monkeypatch.setattr(loop_ai.importlib, "import_module", lambda name: (
        SimpleNamespace(StableAudio3InpaintPipeline=Inpaint) if name == "diffusers" else real_import(name)))
    request = loops.validate_loop_request(settings(tmp_path, method=method, device="cpu", model_path="prepared/model", seed=4))
    if method == "ai_bridge":
        output, information = loop_ai.repair_bridge(original, request, 100, seed=4)
        assert np.array_equal(output[250:-250], original[250:-250])
        source = original
    else:
        from scripts.audio import audio_files

        captured = []
        monkeypatch.setattr(loops, "SAMPLE_RATE", 100)
        monkeypatch.setattr(audio_files, "decode_audio", lambda *args, **kwargs: original)

        def regions(audio, sample_rate, target, fade, count, library, **kwargs):
            assert count == 1 and kwargs["minimum_duration"] == 10.0
            return [{"start": 1000, "end": 4000, "crossfade_seconds": 0.5,
                     "quality_flags": ["heuristic_selection_unrated"], "score": 0.05}]

        def save(samples, sample_rate, destination, library, **kwargs):
            captured.append(samples[0].T.copy())
            destination.write_bytes(b"test")
            return {"audio_path": str(destination), "duration_seconds": samples.shape[-1] / sample_rate}

        monkeypatch.setattr(loops, "select_regions", regions)
        monkeypatch.setattr(loops, "_save_audio", save)
        result = loops.create_loops(request, tmp_path / "smart-ai")
        assert len(result["candidates"]) == 1 and len(captured) == 1
        output, information = captured[0], result["metadata"]["loop"]
        assert np.array_equal(output[:3750], original[:3750])
        assert len(output) == 4250
        assert (information["start_sample"], information["end_sample"]) == (1250, 4250)
        assert (information["source_start_sample"], information["source_end_sample"]) == (1000, 4000)
        assert information["candidate_policy"] == "single_best_region_one_ai_repair"
        assert information["requested_candidate_count"] == 3
        assert information["output_seam"] == loops.seam_metrics(output[1250:4250], 100, np)
        source = original[1000:4000]
    assert calls[0]["mask_start_seconds"] == 13 and calls[0]["mask_end_seconds"] == 17
    assert calls[0]["output_type"] == "latent"
    assert "guidance_scale" not in calls[0]
    expected_reference, _, _, _ = loop_ai.bridge_reference(source, 100, 15, 4, np)
    assert np.array_equal(calls[0]["audio"].numpy()[0].T, expected_reference)
    assert information["outside_pcm_preserved"] and information["saved_prompt_reused"]
    assert len(loads) == len(calls) == 1
    assert freed == ["inpaint", "base"]


def test_create_loops_exports_engine_compatible_candidates_and_frozen_provenance(tmp_path, monkeypatch, np):
    from scripts.audio import audio_files

    audio = stereo(np, sample_rate=44100, seconds=12)
    monkeypatch.setattr(audio_files, "decode_audio", lambda *args, **kwargs: audio)
    saves = []

    def save(samples, sample_rate, destination, library, **kwargs):
        assert samples.shape[0:2] == (1, 2)
        saves.append(kwargs)
        destination.write_bytes(b"test audio")
        return {"audio_path": str(destination), "duration_seconds": samples.shape[-1] / sample_rate,
                "sample_count": samples.shape[-1], "channels": 2, "sample_rate": sample_rate,
                "peak": 0.2, "rms": 0.1, "silent": False}

    monkeypatch.setattr(loops, "_save_audio", save)
    data = settings(tmp_path, method="whole_crossfade", output_format="mp3", keep_wav=False)
    original_context = data["source_generation"]["context"].copy()
    output = tmp_path / "run"
    generated = loops.create_loops(data, output)
    assert generated["ok"] and len(generated["candidates"]) == 1
    candidate = generated["candidates"][0]
    assert Path(candidate["directory"]) == output / "candidate-01"
    assert Path(candidate["audio_path"]).suffix == ".mp3"
    saved = json.loads((output / "candidate-01/result.json").read_text(encoding="utf-8"))
    assert saved["metadata"]["context"]["story_context"] == original_context["story_context"]
    assert "全曲クロスフェード 1" in saved["metadata"]["context"]["scene_label"]
    assert saved["metadata"]["loop"]["warnings"]
    assert saved["metadata"]["duration_seconds"] == 12.0
    assert saved["metadata"]["loop"]["period_seconds"] == 11.5
    assert saved["metadata"]["loop"]["start_seconds"] == 0.5
    assert saved["metadata"]["loop"]["end_seconds"] == 12.0
    assert saved["metadata"]["loop"]["format_version"] == 2
    assert data["source_generation"]["context"] == original_context
    assert saves[0]["keep_wav"] is False and saves[0]["mp3_bitrate"] == 192
    with pytest.raises(engine.AudioGenerationError, match="上書き"):
        loops.create_loops(data, output)
    with pytest.raises(ValueError):
        loops.create_loops(replace(loops.validate_loop_request(data), candidate_count=7), tmp_path / "bad")


@pytest.mark.parametrize("method", ["whole_crossfade", "smart_region", "ai_bridge", "smart_ai_bridge"])
def test_runner_gpu_lease_only_for_ai_and_success_after_release(tmp_path, monkeypatch, method):
    events = []
    output = tmp_path / "run"

    @contextmanager
    def lease(root, device):
        assert method in loops.AI_METHODS
        events.append("enter")
        try:
            yield
        finally:
            assert not (output / "result.json").exists()
            events.append("exit")

    def create(*args, **kwargs):
        events.append("create")
        return {"ok": True, "audio_path": "output.mp3", "candidates": []}

    monkeypatch.setattr(loop_runner, "music_gpu_scope", lease)
    monkeypatch.setattr(loop_runner, "create_loops", create)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(settings(tmp_path, method=method)), encoding="utf-8")
    assert loop_runner.run_request(path, output) == 0
    assert events == (["enter", "create", "exit"] if method in loops.AI_METHODS else ["create"])
    assert json.loads((output / "result.json").read_text())["ok"]


def test_cancelled_runner_never_leaves_candidate_success_records(tmp_path, monkeypatch):
    output = tmp_path / "run"

    def create(*args, **kwargs):
        candidate = output / "candidate-01"
        candidate.mkdir()
        (candidate / "result.json").write_text('{"ok":true}')
        (output / "stop.request").touch()
        return {"ok": True, "candidates": []}

    monkeypatch.setattr(loop_runner, "create_loops", create)
    path = tmp_path / "request.json"
    path.write_text(json.dumps(settings(tmp_path, method="whole_crossfade")))
    assert loop_runner.run_request(path, output) == 130
    for result_path in (output / "result.json", output / "candidate-01/result.json"):
        result = json.loads(result_path.read_text(encoding="utf-8"))
        assert result["ok"] is False and result["cancelled"] is True
    assert loop_runner.run_request(path, output) == 2


@pytest.mark.parametrize("method", ["ai_bridge", "smart_ai_bridge", "whole_crossfade", "smart_region"])
def test_runner_releases_only_requested_ollama_before_ai_lease(tmp_path, monkeypatch, method):
    from scripts.audio import llm_runtime

    events = []

    @contextmanager
    def lease(*args):
        events.append("lease")
        yield

    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: events.append("release"))
    monkeypatch.setattr(loop_runner, "music_gpu_scope", lease)
    monkeypatch.setattr(loop_runner, "create_loops", lambda *args, **kwargs: (
        events.append("create") or {"ok": True}))
    path = tmp_path / "request.json"
    data = settings(tmp_path, method=method, llm_release={"base_url": "http://localhost:11434/v1", "model": "selected"})
    path.write_text(json.dumps(data), encoding="utf-8")
    assert loop_runner.run_request(path, tmp_path / "run") == 0
    assert events == (["release", "lease", "create"] if method in loops.AI_METHODS else ["create"])


def test_real_librosa_detects_pulses_with_features_and_no_gpu(np):
    pytest.importorskip("librosa")
    rate, seconds = 22050, 12
    audio = np.zeros((rate * seconds, 2), dtype=np.float32)
    for beat in np.arange(0.5, seconds - 0.5, 0.5):
        start = round(beat * rate)
        pulse = np.linspace(0.4, 0, 600).astype(np.float32)
        audio[start:start + len(pulse)] += pulse[:, None]
    analyzed = loops.analyze_music(audio, rate, np)
    assert analyzed["beat_source"] == "detected_beats"
    assert len(analyzed["beat_times"]) > 10
    assert 100 < analyzed["tempo_bpm"] < 140
    assert analyzed["chroma"].shape[0] == analyzed["mfcc"].shape[0] == 12
    assert np.isfinite(analyzed["rms"]).all()
