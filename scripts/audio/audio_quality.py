"""Conservative silence measurements for generated stereo audio.

NumPy is supplied by the audio runtime so importing this module remains cheap
and does not add inference dependencies to the GUI environment.
"""

from __future__ import annotations

import math
from numbers import Integral
from typing import Any

THRESHOLD_DBFS = -50.0
WINDOW_SECONDS = 0.1
MINIMUM_DURATION_SECONDS = 2.0


def analyze_silence(audio: Any, sample_rate: int, *, np: Any) -> dict:
    """Find stretches whose samples in both channels remain below -50 dBFS.

    Windows are approximately 100 ms, rounded to the nearest sample. A window
    qualifies only when every sample is quiet; this avoids counting a musical
    rest around even a brief audible accent. The last partial window uses its
    real sample count. Consecutive quiet windows must cover at least two seconds.
    The input is never changed and temporary arrays scale with window count,
    rather than with the complete waveform size.
    """
    if (isinstance(sample_rate, bool) or not isinstance(sample_rate, Integral)
            or sample_rate <= 0):
        raise ValueError("無音検出のサンプルレートには正の整数を指定してください。")
    if (not isinstance(audio, np.ndarray) or audio.ndim != 2 or audio.shape[1] != 2
            or audio.shape[0] == 0):
        raise ValueError("無音検出には空でない [samples, 2] の音声配列が必要です。")
    if not (np.issubdtype(audio.dtype, np.floating)
            or np.issubdtype(audio.dtype, np.integer)):
        raise ValueError("無音検出には実数の音声配列が必要です。")

    sample_rate = int(sample_rate)
    sample_count = int(audio.shape[0])
    window_samples = max(1, round(sample_rate * WINDOW_SECONDS))
    starts = np.arange(0, sample_count, window_samples, dtype=np.int64)

    # Signed extrema avoid constructing abs(audio), which would duplicate the
    # entire waveform. Their small float64 conversions also avoid signed integer
    # overflow for an input containing the minimum representable integer.
    maxima = np.maximum.reduceat(audio, starts, axis=0).astype(np.float64, copy=False)
    minima = np.minimum.reduceat(audio, starts, axis=0).astype(np.float64, copy=False)
    if not (np.isfinite(maxima).all() and np.isfinite(minima).all()):
        raise ValueError("無音検出の音声に NaN または無限大が含まれています。")
    peaks = np.maximum(np.abs(maxima), np.abs(minima)).max(axis=1)
    threshold = math.pow(10.0, THRESHOLD_DBFS / 20.0)
    quiet = peaks < threshold
    edges = np.diff(np.concatenate(([False], quiet, [False])).astype(np.int8))
    first_windows = np.flatnonzero(edges == 1)
    end_windows = np.flatnonzero(edges == -1)

    intervals = []
    lengths = []
    for first_window, end_window in zip(first_windows, end_windows, strict=True):
        first_sample = int(first_window) * window_samples
        end_sample = min(int(end_window) * window_samples, sample_count)
        length = end_sample - first_sample
        if length < MINIMUM_DURATION_SECONDS * sample_rate:
            continue
        if first_sample == 0 and end_sample == sample_count:
            position = "whole"
        elif first_sample == 0:
            position = "start"
        elif end_sample == sample_count:
            position = "end"
        else:
            position = "internal"
        intervals.append({
            "start_seconds": first_sample / sample_rate,
            "end_seconds": end_sample / sample_rate,
            "duration_seconds": length / sample_rate,
            "position": position,
        })
        lengths.append(length)

    return {
        "threshold_dbfs": THRESHOLD_DBFS,
        "window_seconds": WINDOW_SECONDS,
        "minimum_duration_seconds": MINIMUM_DURATION_SECONDS,
        "intervals": intervals,
        "total_seconds": sum(lengths) / sample_rate,
        "longest_seconds": max(lengths, default=0) / sample_rate,
        "count": len(intervals),
        "duration_seconds": sample_count / sample_rate,
    }
