"""Actual PCM checks; run with the dedicated audio runtime's NumPy available."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from services.worker.generation import music_runner
from services.worker.generation.music.quality import inspect_quality, require_usable

np = pytest.importorskip("numpy")


def tone(duration=120, rate=100):
    mono = np.sin(np.arange(duration * rate) * 0.2).astype(np.float32) * 0.25
    return np.column_stack((mono, mono))


def test_internal_silence_is_detected_even_with_audible_start_and_tail():
    audio = tone()
    audio[1000:1700] = 0
    quality = inspect_quality(audio, 100, np=np, expected_duration=120)
    assert "long_internal_silence" in quality["flags"]
    assert quality["silence"]["intervals"] == [{"start_seconds": 10.0, "end_seconds": 17.0,
        "duration_seconds": 7.0, "position": "internal"}]
    with pytest.raises(ValueError, match="long_internal_silence"):
        require_usable(quality)


def test_short_musical_rest_is_recorded_without_rejecting_piece():
    audio = tone()
    audio[1000:1200] = 0
    quality = inspect_quality(audio, 100, np=np)
    assert not quality["needs_review"]
    assert quality["silence"]["count"] == 1
    require_usable(quality)


@pytest.mark.parametrize("bad", ["nan", "inf", "empty", "mono", "duration"])
def test_invalid_pcm_cannot_be_published(bad):
    audio = tone()
    if bad == "nan":
        audio[0, 0] = np.nan
    elif bad == "inf":
        audio[0, 0] = np.inf
    elif bad == "empty":
        audio = audio[:0]
    elif bad == "mono":
        audio = audio[:, :1]
    with pytest.raises(ValueError):
        inspect_quality(audio, 100, np=np, expected_duration=80 if bad == "duration" else 120)


def test_clipping_and_long_silent_tail_fail_with_independent_flags():
    quality = inspect_quality(np.ones((12000, 2), dtype=np.float32) * 1.1, 100, np=np)
    assert quality["flags"] == ["extensive_clipping"]
    audio = tone()
    audio[-2100:] = 0
    quality = inspect_quality(audio, 100, np=np)
    assert "long_silent_opening_or_tail" in quality["flags"]
    assert quality["silence"]["intervals"][0]["position"] == "end"


def row(start=500, end=2000):
    return {"start": start, "end": end, "score": 0.1, "confidence": "moderate",
        "quality_flags": ["heuristic_selection_unrated"], "crossfade_seconds": 0.5,
        "source_start_seconds": start / 100, "source_end_seconds": end / 100}


def test_loop_preserves_intro_and_explicit_bounds(tmp_path, monkeypatch):
    audio = tone()
    captured = {}
    monkeypatch.setattr(music_runner, "select_regions", lambda *args, **kwargs: [row()])

    def save(data, sample_rate, destination, np, **kwargs):
        captured[destination.name] = data[0].T.copy()
        destination.write_bytes(b"ID3test")
        assert kwargs["mp3_bitrate"] == 192
        return {"preview_gain": 1}

    monkeypatch.setattr(music_runner.stable_audio, "_save_audio", save)
    request = {"scene_id": "comic", "prompt": "Instrumental pop.", "duration_seconds": 120,
        "output_dir": str(tmp_path)}
    report = music_runner.process_pcm(request, SimpleNamespace(audio=audio, sample_rate=100, provenance={}), np)
    result = report["result"]
    assert (result["loop_start_seconds"], result["loop_end_seconds"], result["duration_seconds"]) == (5, 20, 20)
    assert result["source_duration_seconds"] == 120
    np.testing.assert_array_equal(captured["music.mp3"][:500], audio[:500])
    np.testing.assert_array_equal(captured["source.mp3"], audio)
    assert report["provenance"]["loop"]["playback_layout"] == "intro_then_loop"
    assert (tmp_path / "result.json").is_file()


def test_no_automatic_region_never_silently_loops_a_long_file(tmp_path, monkeypatch):
    audio = tone()
    monkeypatch.setattr(music_runner, "select_regions", lambda *args, **kwargs: [])
    request = {"scene_id": "comic", "prompt": "Instrumental pop.", "duration_seconds": 120,
        "output_dir": str(tmp_path)}
    with pytest.raises(ValueError, match="search returned no usable"):
        music_runner.process_pcm(request, SimpleNamespace(audio=audio, sample_rate=100, provenance={}), np)
    assert not (tmp_path / "source.mp3").exists()
    assert not (tmp_path / "result.json").exists()


def test_five_second_import_keeps_full_period(tmp_path, monkeypatch):
    audio = tone(duration=5)
    monkeypatch.setattr(music_runner, "select_regions", lambda *args, **kwargs: pytest.fail("short source cannot have interior loop"))

    def save(data, sample_rate, destination, np, **kwargs):
        destination.write_bytes(b"ID3test")
        return {"preview_gain": 1}

    monkeypatch.setattr(music_runner.stable_audio, "_save_audio", save)
    request = {"scene_id": "comic", "prompt": "Imported instrumental.", "duration_seconds": 5,
        "output_dir": str(tmp_path)}
    report = music_runner.process_pcm(request, SimpleNamespace(audio=audio, sample_rate=100, provenance={}), np)
    assert report["result"]["loop_start_seconds"] == 0
    assert report["result"]["loop_end_seconds"] == report["result"]["duration_seconds"] == 5
