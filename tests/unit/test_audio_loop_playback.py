from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import pytest

from scripts.audio.loop_playback import SAMPLE_RATE, loop_playback_plan


def metadata(**changes):
    return {
        "sample_rate": SAMPLE_RATE, "sample_count": SAMPLE_RATE * 60,
        "duration_seconds": 60.0, "preview_gain": 0.8,
        "loop": {"method": "smart_region", "source_start_seconds": 18.0},
        **changes,
    }


def option(plan, flag):
    return plan.cli_options[plan.cli_options.index(flag) + 1]


def test_plain_generation_has_no_loop_playback_plan():
    assert loop_playback_plan({}) is None
    assert loop_playback_plan({"sample_count": 30, "loop": None}) is None


def test_new_points_keep_intro_and_end_before_outro_without_source_file():
    plan = loop_playback_plan(metadata(loop={
        "method": "smart_region", "start_sample": 18 * SAMPLE_RATE,
        "end_sample": 55 * SAMPLE_RATE,
    }))
    assert plan.start_sample == 18 * SAMPLE_RATE
    assert plan.end_sample == 55 * SAMPLE_RATE
    assert plan.cli_options == [
        "--loop-start-sample", str(18 * SAMPLE_RATE),
        "--loop-end-sample", str(55 * SAMPLE_RATE),
    ]
    assert plan.description == "初回 0.00 → 55.00秒 / 繰り返し 18.00 → 55.00秒（37.00秒）"
    assert not plan.warning
    assert not plan.restored_intro


def test_new_markers_override_old_source_coordinates_and_unknown_method():
    plan = loop_playback_plan(metadata(loop={
        "method": "smart_ai_bridge", "start": 9000000, "end": 18000000,
        "source_start_seconds": 120, "start_sample": 4410, "end_sample": 44100,
    }))
    assert plan.start_sample == 4410
    assert plan.end_sample == 44100
    assert not plan.restored_intro


@pytest.mark.parametrize("points", [
    {"start_sample": -1, "end_sample": 100},
    {"start_sample": 10.5, "end_sample": 100},
    {"start_sample": True, "end_sample": 100},
    {"start_sample": 0, "end_sample": False},
    {"start_sample": 100, "end_sample": 100},
    {"start_sample": 101, "end_sample": 100},
    {"start_sample": 0, "end_sample": SAMPLE_RATE * 61},
    {"start_sample": math.nan, "end_sample": 100},
    {"start_sample": 0, "end_sample": math.inf},
    {"start_sample": 0}, {"end_sample": 100},
])
def test_invalid_or_partial_explicit_points_do_not_fall_back_to_legacy(points):
    with pytest.raises(ValueError):
        loop_playback_plan(metadata(loop={"method": "smart_region", **points}))


@pytest.mark.parametrize("changes", [
    {"sample_count": 0}, {"sample_count": -1}, {"sample_count": 1.5},
    {"sample_count": True}, {"sample_count": math.nan},
    {"sample_rate": 48000}, {"sample_rate": True}, {"loop": "smart_region"},
])
def test_invalid_metadata_is_reported_to_gui(changes):
    with pytest.raises(ValueError):
        loop_playback_plan(metadata(**changes))


def test_legacy_smart_restores_original_prefix_and_only_loops_after_intro(tmp_path):
    source = tmp_path / "output-float.wav"
    source.touch()
    plan = loop_playback_plan(metadata(
        source_audio=str(source), source_generation={"float_audio_path": str(source), "preview_gain": 0.5},
    ))
    assert plan.start_sample == 18 * SAMPLE_RATE
    assert plan.end_sample == 78 * SAMPLE_RATE
    assert int(option(plan, "--intro-samples")) == 18 * SAMPLE_RATE
    assert float(option(plan, "--intro-gain")) == 0.8
    assert option(plan, "--intro-input") == str(source.resolve())
    assert plan.description == "初回 0.00 → 78.00秒 / 繰り返し 18.00 → 78.00秒（60.00秒）"
    assert plan.restored_intro
    assert not plan.warning


def test_moved_float_source_matches_provenance_basename_and_uses_candidate_gain(tmp_path):
    source = tmp_path / "candidate-float.wav"
    source.touch()
    plan = loop_playback_plan(metadata(
        source_audio=str(source),
        source_generation={"float_audio_path": r"C:\old-folder\candidate-float.wav", "preview_gain": 0.25},
        preview_gain=0.8756956551493104,
    ))
    assert float(option(plan, "--intro-gain")) == 0.8756956551493104


def test_wrapped_source_metadata_matches_float_path(tmp_path):
    source = tmp_path / "raw.wav"
    source.touch()
    plan = loop_playback_plan(metadata(
        source_audio=str(source), source_generation={"metadata": {"float_audio_path": str(source)}},
    ))
    assert float(option(plan, "--intro-gain")) == 0.8


@pytest.mark.parametrize("name", ["output.wav", "output.mp3", "output-float.mp3"])
def test_pcm16_and_mp3_source_are_not_attenuated_twice(tmp_path, name):
    source = tmp_path / name
    source.touch()
    plan = loop_playback_plan(metadata(
        source_audio=str(source), preview_gain=0.6,
        source_generation={"float_audio_path": str(tmp_path / "output-float.wav"), "preview_gain": 0.4},
    ))
    assert float(option(plan, "--intro-gain")) == 1.0


def test_legacy_whole_restores_zero_to_fade_before_the_rotated_loop(tmp_path):
    source = tmp_path / "output.wav"
    source.touch()
    plan = loop_playback_plan(metadata(
        source_audio=str(source), sample_count=round(119.75 * SAMPLE_RATE),
        loop={"method": "whole_crossfade", "source_start_seconds": 0,
              "crossfade_seconds": 0.25, "overlap_shortening_seconds": 0.25},
    ))
    assert plan.start_sample == round(0.25 * SAMPLE_RATE)
    assert plan.end_sample == 120 * SAMPLE_RATE
    assert plan.restored_intro
    assert "0.25 → 120.00秒（119.75秒）" in plan.description


def test_legacy_ai_keeps_its_repaired_head_and_needs_no_original_source(tmp_path):
    plan = loop_playback_plan(metadata(
        source_audio=str(tmp_path / "missing.wav"),
        loop={"method": "ai_bridge", "source_start_seconds": 0, "crossfade_seconds": 0.25},
    ))
    assert plan.start_sample == 0
    assert plan.end_sample == 60 * SAMPLE_RATE
    assert "--intro-input" not in plan.cli_options
    assert not plan.restored_intro
    assert not plan.warning


@pytest.mark.parametrize("method,offset", [("smart_region", 18.0), ("whole_crossfade", 0.25)])
def test_missing_legacy_source_warns_about_original_offset_and_regeneration(tmp_path, method, offset):
    plan = loop_playback_plan(metadata(
        source_audio=str(tmp_path / "missing.wav"),
        loop={"method": method, "source_start_seconds": offset, "crossfade_seconds": offset},
    ))
    assert plan.start_sample == 0
    assert plan.end_sample == 60 * SAMPLE_RATE
    assert f"{offset:.2f}秒" in plan.warning
    assert "元音源が見つからない" in plan.warning
    assert "再作成" in plan.warning
    assert not plan.restored_intro
    assert "--intro-input" not in plan.cli_options


@pytest.mark.parametrize("offset", [-1, math.nan, math.inf, True, 1e308])
def test_invalid_legacy_source_point_is_not_silently_used(offset):
    with pytest.raises(ValueError):
        loop_playback_plan(metadata(loop={"method": "smart_region", "source_start_seconds": offset}))


@pytest.mark.parametrize("gain", [-1, 0, math.nan, math.inf, True, 1.1])
def test_invalid_float_prefix_gain_is_rejected(tmp_path, gain):
    source = tmp_path / "output-float.wav"
    source.touch()
    with pytest.raises(ValueError):
        loop_playback_plan(metadata(source_audio=str(source), preview_gain=gain))


def test_old_metadata_without_sample_count_uses_recorded_pcm_duration():
    recorded = metadata(loop={"method": "ai_bridge"})
    recorded.pop("sample_count")
    recorded["duration_seconds"] = 1.5
    assert loop_playback_plan(recorded).end_sample == round(1.5 * SAMPLE_RATE)


def test_helper_import_does_not_load_audio_or_model_dependencies():
    result = subprocess.run([
        sys.executable, "-c",
        ("import sys; from scripts.audio.loop_playback import loop_playback_plan; "
        "assert not ({'torch','numpy','diffusers','librosa','sounddevice'} & sys.modules.keys()); "
        "assert loop_playback_plan({}) is None"),
    ], capture_output=True, text=True, check=False, cwd=Path(__file__).resolve().parents[2])
    assert result.returncode == 0, result.stderr
