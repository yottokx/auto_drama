"""Fail closed for broken or extensively silent PCM before publishing music."""
from __future__ import annotations

from .audio_quality import analyze_silence

QUALITY_VERSION = 1


def inspect_quality(audio, sample_rate: int, *, np, expected_duration=None) -> dict:
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("Music sample rate must be a positive integer.")
    if (audio.ndim != 2 or audio.shape[1] != 2 or not len(audio)
            or not np.isfinite(audio).all()):
        raise ValueError("Generated music is empty, non-stereo or contains NaN/Inf.")
    duration = len(audio) / sample_rate
    if expected_duration is not None and abs(duration - expected_duration) > 1 / sample_rate:
        raise ValueError("Generated music did not cover the requested duration.")
    peak = float(np.max(np.abs(audio)))
    rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    clipping_fraction = float(np.count_nonzero(np.abs(audio) > 1) / audio.size)
    silence = analyze_silence(audio, sample_rate, np=np)
    problems = []
    if peak < 1e-5 or rms < 1e-4:
        problems.append("silent_or_inaudible")
    if clipping_fraction > 0.01:
        problems.append("extensive_clipping")
    if silence["total_seconds"] > duration * 0.25:
        problems.append("extensive_silence")
    if any(row["position"] == "internal" and row["duration_seconds"] >= 6
           for row in silence["intervals"]):
        problems.append("long_internal_silence")
    if any(row["position"] in {"start", "end"} and row["duration_seconds"] >= 20
           for row in silence["intervals"]):
        problems.append("long_silent_opening_or_tail")
    return {"version": QUALITY_VERSION, "peak": peak, "rms": rms,
        "clipping_fraction": clipping_fraction, "silence": silence,
        "needs_review": bool(problems), "flags": problems,
        "thresholds": {"minimum_rms": 1e-4, "maximum_clipping_fraction": 0.01,
            "maximum_silent_fraction": 0.25, "internal_silence_seconds": 6,
            "opening_or_tail_silence_seconds": 20}}


def require_usable(quality: dict):
    if quality["needs_review"]:
        raise ValueError("Music quality check failed: " + ", ".join(quality["flags"]))
