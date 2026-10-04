from __future__ import annotations

import json

import pytest

from scripts.audio.audio_quality import analyze_silence

np = pytest.importorskip("numpy")


def test_quiet_music_above_threshold_is_not_silence():
    # -40 dBFS is quiet music, but ten dB above the detector's silence threshold.
    audio = np.full((500, 2), 10 ** (-40 / 20), dtype=np.float32)
    result = analyze_silence(audio, 100, np=np)
    assert result["count"] == 0
    assert result["intervals"] == []
    assert result["total_seconds"] == 0
    assert result["longest_seconds"] == 0
    assert result["duration_seconds"] == 5
    assert result["threshold_dbfs"] == -50
    assert result["window_seconds"] == 0.1
    assert result["minimum_duration_seconds"] == 2
    json.dumps(result, allow_nan=False)


def test_detects_start_internal_and_end_intervals_without_mutation():
    audio = np.full((1100, 2), 0.02, dtype=np.float32)
    audio[:200] = 0
    audio[300:600] = -0.0005
    audio[900:] = 0.0005
    original = audio.copy()
    audio.flags.writeable = False
    result = analyze_silence(audio, 100, np=np)
    assert result["intervals"] == [
        {"start_seconds": 0, "end_seconds": 2, "duration_seconds": 2, "position": "start"},
        {"start_seconds": 3, "end_seconds": 6, "duration_seconds": 3, "position": "internal"},
        {"start_seconds": 9, "end_seconds": 11, "duration_seconds": 2, "position": "end"},
    ]
    assert result["count"] == 3
    assert result["total_seconds"] == 7
    assert result["longest_seconds"] == 3
    np.testing.assert_array_equal(audio, original)


@pytest.mark.parametrize("channel", [0, 1])
@pytest.mark.parametrize("level", [-0.01, 0.01])
def test_one_audible_channel_prevents_false_silence(channel, level):
    audio = np.zeros((400, 2), dtype=np.float32)
    audio[:, channel] = level
    assert analyze_silence(audio, 100, np=np)["count"] == 0


def test_single_sample_noise_splits_quiet_windows():
    audio = np.zeros((500, 2), dtype=np.float32)
    # A signed impulse invalidates the complete [2.0, 2.1) window, even though
    # RMS over that window would make the pulse seem much quieter.
    audio[205, 1] = -0.01
    assert analyze_silence(audio, 100, np=np)["intervals"] == [
        {"start_seconds": 0, "end_seconds": 2, "duration_seconds": 2, "position": "start"},
        {"start_seconds": 2.1, "end_seconds": 5, "duration_seconds": 2.9, "position": "end"},
    ]


def test_fractional_tail_uses_real_samples_without_padding():
    audio = np.full((417, 2), 0.01, dtype=np.float32)
    audio[200:] = 0
    result = analyze_silence(audio, 100, np=np)
    assert result["duration_seconds"] == 4.17
    assert result["intervals"] == [
        {"start_seconds": 2, "end_seconds": 4.17,
         "duration_seconds": 2.17, "position": "end"},
    ]
    assert result["total_seconds"] == 2.17


def test_audible_sample_in_partial_tail_is_not_ignored():
    audio = np.zeros((217, 2), dtype=np.float32)
    audio[-1, 1] = 0.01
    result = analyze_silence(audio, 100, np=np)
    assert result["intervals"] == [
        {"start_seconds": 0, "end_seconds": 2.1,
         "duration_seconds": 2.1, "position": "start"},
    ]


def test_all_zero_input_reports_whole_audio():
    result = analyze_silence(np.zeros((223, 2), dtype=np.float32), 100, np=np)
    assert result["intervals"] == [
        {"start_seconds": 0, "end_seconds": 2.23,
         "duration_seconds": 2.23, "position": "whole"},
    ]
    assert result["count"] == 1
    assert result["total_seconds"] == result["duration_seconds"] == 2.23


def test_short_whole_silence_is_valid_but_below_minimum():
    result = analyze_silence(np.zeros((199, 2), dtype=np.float32), 100, np=np)
    assert result["count"] == 0
    assert result["duration_seconds"] == 1.99


def test_threshold_is_absolute_and_strict():
    threshold = 10 ** (-50 / 20)
    audio = np.full((400, 2), threshold, dtype=np.float64)
    assert analyze_silence(audio, 100, np=np)["count"] == 0
    audio[:] = -threshold
    assert analyze_silence(audio, 100, np=np)["count"] == 0
    audio[:] = np.nextafter(threshold, 0)
    assert analyze_silence(audio, 100, np=np)["count"] == 1


def test_window_boundaries_remain_conservative():
    audio = np.full((600, 2), 0.01, dtype=np.float32)
    # Exact silence is [2.03, 4.17), but two boundary windows contain music.
    audio[203:417] = 0
    result = analyze_silence(audio, 100, np=np)
    assert result["intervals"] == [
        {"start_seconds": 2.1, "end_seconds": 4.1,
         "duration_seconds": 2, "position": "internal"},
    ]


def test_noncontiguous_waveform_and_numpy_sample_rate():
    audio = np.zeros((440, 2), dtype=np.float32)[::2]
    assert not audio.flags.c_contiguous
    assert analyze_silence(audio, np.int64(100), np=np)["total_seconds"] == 2.2


def test_minimum_signed_integer_is_not_misclassified_by_absolute_overflow():
    audio = np.full((400, 2), np.iinfo(np.int16).min, dtype=np.int16)
    assert analyze_silence(audio, 100, np=np)["count"] == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("location", [0, 217])
def test_nonfinite_samples_are_rejected(value, location):
    audio = np.zeros((218, 2), dtype=np.float32)
    audio[location, 1] = value
    with pytest.raises(ValueError, match="NaN"):
        analyze_silence(audio, 100, np=np)


@pytest.mark.parametrize("sample_rate", [0, -1, True, 1.5, float("nan"), None, "48000"])
def test_invalid_sample_rates_are_rejected(sample_rate):
    with pytest.raises(ValueError, match="サンプルレート"):
        analyze_silence(np.zeros((200, 2)), sample_rate, np=np)


@pytest.mark.parametrize("audio", [
    np.zeros((0, 2)), np.zeros(200), np.zeros((200, 1)), np.zeros((200, 3)),
    np.zeros((1, 200, 2)), [[0, 0], [0, 0]], None,
])
def test_empty_or_invalid_shapes_are_rejected(audio):
    with pytest.raises(ValueError, match="samples"):
        analyze_silence(audio, 100, np=np)


@pytest.mark.parametrize("dtype", [np.complex64, np.bool_, object, "U1"])
def test_nonreal_numeric_arrays_are_rejected(dtype):
    with pytest.raises(ValueError, match="実数"):
        analyze_silence(np.zeros((200, 2), dtype=dtype), 100, np=np)
