"""Automatic loop-region search and conventional crossfade, without generative repair."""
from __future__ import annotations

import importlib
import math
from collections.abc import Callable

from .stable_audio import AudioGenerationError, GenerationCancelled

AI_METHODS = frozenset()


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise GenerationCancelled("ループ実験を停止しました。")


def fade_weights(samples: int, np):
    return (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, samples))).astype(np.float32)[:, None]


def seam_metrics(audio, sample_rate: int, np) -> dict:
    count = min(max(2, int(sample_rate * 0.1)), len(audio) // 2)
    first, last = audio[:count].astype(np.float64), audio[-count:].astype(np.float64)
    first_rms = float(np.sqrt(np.mean(first**2)))
    last_rms = float(np.sqrt(np.mean(last**2)))
    return {
        "boundary_jump": float(np.max(np.abs(audio[-1] - audio[0]))),
        "boundary_slope_jump": float(np.max(np.abs((audio[1] - audio[0]) - (audio[-1] - audio[-2])))),
        "head_rms": first_rms, "tail_rms": last_rms,
        "endpoint_energy_ratio": min(first_rms, last_rms) / max(first_rms, last_rms, 1e-12),
        "assessment": "数値は継ぎ目の診断用です。拍・和声・聴感の自然さは保証しません。",
    }


def region_crossfade(audio, start: int, end: int, fade: int, np):
    """Keep beat-to-beat period; blend the tail towards source immediately before start."""
    if not 0 < fade <= start or end - start <= 2 * fade or end > len(audio):
        raise AudioGenerationError("候補区間またはクロスフェードの長さが不正です。")
    result = audio[start:end].copy()
    weights = fade_weights(fade, np)
    result[-fade:] = result[-fade:] * (1 - weights) + audio[start - fade:start] * weights
    return result


def _intro_once_variant(audio, circular, information: dict, method: str, sample_rate: int, np):
    """Retain the original opening, then repeat the explicitly marked PCM range."""
    information = dict(information)
    if method == "whole_crossfade":
        start = len(audio) - len(circular)
        end = len(audio)
        candidate = np.concatenate((audio[:start], circular))
    elif method == "smart_region":
        start, end = int(information["start"]), int(information["end"])
        candidate = np.concatenate((audio[:start], circular))
    elif method == "ai_bridge":
        start = int(information["edited_sample_count"]) // 2
        end = len(audio) + start
        candidate = np.concatenate((audio[:start], circular[start:], circular[:start]))
    elif method == "smart_ai_bridge":
        edge = int(information["edited_sample_count"]) // 2
        start, end = int(information["start"]) + edge, int(information["end"]) + edge
        candidate = np.concatenate((audio[:start], circular[edge:], circular[:edge]))
    else:
        raise AudioGenerationError("ループ方式が不正です。")
    if not 0 < start < end == len(candidate):
        raise AudioGenerationError("冒頭とループ区間のPCM配置が不正です。")
    information.update({
        "format_version": 2, "playback_layout": "intro_then_loop", "intro_once": True,
        "start_sample": start, "end_sample": end,
        "start_seconds": start / sample_rate, "end_seconds": end / sample_rate,
        "period_samples": end - start, "period_seconds": (end - start) / sample_rate,
        "file_end_sample": len(candidate), "file_end_seconds": len(candidate) / sample_rate,
        "source_start_sample": round(information.get("source_start_seconds", 0.0) * sample_rate),
        "source_end_sample": round(information.get("source_end_seconds", len(audio) / sample_rate) * sample_rate),
        "intro_pcm_preserved": True,
        "playback_note": "0秒から冒頭を一度再生し、区間の終端Bから始点Aへ戻って繰り返します。サンプル終端は含みません。",
    })
    if method in AI_METHODS:
        source_end = information["source_end_sample"]
        half_hole = (int(information["repair_end_sample"]) - int(information["repair_start_sample"])) // 2
        edge = int(information["edited_sample_count"]) // 2
        information.update({
            "repair_file_start_sample": source_end - half_hole,
            "repair_file_end_sample": source_end + half_hole,
            "edited_file_start_sample": source_end - edge,
            "edited_file_end_sample": source_end + edge,
        })
    return candidate.astype(np.float32), information


def analyze_music(audio, sample_rate: int, np, *, cancelled=None) -> dict:
    """Use detected beats and acoustic features, rather than a rounded BPM grid."""
    librosa = importlib.import_module("librosa")
    check_cancelled(cancelled)
    analysis_rate, hop = 22050, 512
    mono = np.mean(audio, axis=1)
    if sample_rate != analysis_rate:
        mono = librosa.resample(mono, orig_sr=sample_rate, target_sr=analysis_rate)
    magnitude = np.abs(librosa.stft(mono, n_fft=2048, hop_length=hop))
    check_cancelled(cancelled)
    power = magnitude**2
    onset = librosa.onset.onset_strength(S=librosa.power_to_db(power, ref=np.max), sr=analysis_rate, hop_length=hop)
    tempo, beats = librosa.beat.beat_track(onset_envelope=onset, sr=analysis_rate, hop_length=hop, trim=False)
    check_cancelled(cancelled)
    beat_source = "detected_beats"
    if len(beats) < 6:
        beats = librosa.onset.onset_detect(onset_envelope=onset, sr=analysis_rate, hop_length=hop)
        beat_source = "detected_onsets_no_reliable_beat"
    if len(beats) < 2:
        beats = np.arange(0, len(onset), max(1, round(0.5 * analysis_rate / hop)))
        beat_source = "feature_frames_no_reliable_beat"
    chroma = librosa.feature.chroma_stft(S=power, sr=analysis_rate, hop_length=hop, tuning=0)
    mel = librosa.feature.melspectrogram(S=power, sr=analysis_rate)
    mfcc = librosa.feature.mfcc(S=librosa.power_to_db(mel, ref=np.max), n_mfcc=13)[1:]
    rms = librosa.feature.rms(S=magnitude, frame_length=2048)[0]
    mfcc = (mfcc - np.mean(mfcc, axis=1, keepdims=True)) / np.maximum(np.std(mfcc, axis=1, keepdims=True), 1)
    return {
        "beat_times": np.asarray(librosa.frames_to_time(beats, sr=analysis_rate, hop_length=hop)),
        "beat_source": beat_source, "tempo_bpm": float(np.asarray(tempo).reshape(-1)[0]),
        "chroma": chroma, "mfcc": mfcc, "rms": rms, "onset": onset,
        "frame_seconds": hop / analysis_rate, "librosa_version": librosa.__version__,
    }


def select_regions(audio, sample_rate: int, target_duration: float, crossfade_seconds: float,
                   candidate_count: int, np, *, analysis=None, cancelled=None,
                   minimum_duration: float = 0.0) -> list[dict]:
    analysis = analysis or analyze_music(audio, sample_rate, np, cancelled=cancelled)
    duration = len(audio) / sample_rate
    fade = min(crossfade_seconds, duration / 12)
    guard = min(8.0, max(fade + 0.1, duration * 0.06))
    maximum = duration - 2 * guard
    if maximum <= max(1.0, 2 * fade):
        raise AudioGenerationError("中盤候補を探すには音声が短すぎます。")
    if maximum < minimum_duration:
        raise AudioGenerationError("中盤AI修復には10秒以上の候補区間が必要です。元音声が短すぎます。")
    target = min(target_duration, maximum)
    tolerance = min(12.0, max(2.0, target * 0.2))
    beat_times = analysis["beat_times"]
    valid = np.flatnonzero((beat_times >= guard) & (beat_times <= duration - guard))
    if len(valid) < 2:
        raise AudioGenerationError("音声の中盤に十分な候補点がありません。")
    rms_floor = max(float(np.quantile(analysis["rms"], 0.7)) * 0.12, 1e-7)
    onset_scale = max(float(np.quantile(analysis["onset"], 0.9)), 1e-5)
    window = max(1, round(0.5 / analysis["frame_seconds"]))
    descriptors = {}
    for index in valid:
        frame = round(float(beat_times[index]) / analysis["frame_seconds"])
        lo, hi = max(0, frame - window), frame + window + 1
        descriptors[int(index)] = (
            np.mean(analysis["chroma"][:, lo:hi], axis=1),
            np.mean(analysis["mfcc"][:, lo:hi], axis=1),
            float(np.mean(analysis["rms"][lo:hi])),
            float(np.mean(analysis["onset"][lo:hi])) / onset_scale,
        )
    ranked = []
    for count, start_index in enumerate(valid):
        if count % 32 == 0:
            check_cancelled(cancelled)
        start = float(beat_times[start_index])
        possible = valid[(beat_times[valid] >= start + max(2 * fade + 0.1, target - tolerance, minimum_duration))
                         & (beat_times[valid] <= start + target + tolerance)]
        c1, t1, r1, o1 = descriptors[int(start_index)]
        for end_index in possible:
            c2, t2, r2, o2 = descriptors[int(end_index)]
            if min(r1, r2) < rms_floor:
                continue
            chroma_distance = 1 - float(np.dot(c1, c2) / max(np.linalg.norm(c1) * np.linalg.norm(c2), 1e-12))
            rms_distance = abs(math.log(max(r1, 1e-9) / max(r2, 1e-9)))
            timbre_distance = float(np.mean(np.abs(t1 - t2)))
            onset_distance = abs(o1 - o2)
            period = float(beat_times[end_index]) - start
            beat_count = int(end_index - start_index)
            bar_penalty = min(beat_count % 4, (-beat_count) % 4) / 4 if analysis["beat_source"] == "detected_beats" else 0
            score = (0.4 * chroma_distance + 0.25 * rms_distance + 0.2 * timbre_distance
                     + 0.15 * onset_distance + 0.1 * bar_penalty + 0.08 * abs(period - target) / target)
            ranked.append({
                "start": round(start * sample_rate), "end": round(float(beat_times[end_index]) * sample_rate),
                "score": score, "chroma_distance": chroma_distance, "rms_log_distance": rms_distance,
                "timbre_distance": timbre_distance, "onset_distance": onset_distance,
                "detected_beat_count": beat_count, "tempo_bpm": analysis["tempo_bpm"],
            })
    if not ranked:
        raise AudioGenerationError("無音やフェード区間を除いた候補が見つかりません。目標秒数を短くしてください。")
    ranked.sort(key=lambda item: (item["score"], item["start"], item["end"]))
    selected = []
    for separation in (max(2.0, target * 0.12), 0.2):
        for row in ranked:
            if all(max(abs(row["start"] - old["start"]), abs(row["end"] - old["end"]))
                   >= separation * sample_rate for old in selected):
                selected.append(row)
                if len(selected) == candidate_count:
                    break
        if len(selected) == candidate_count:
            break
    for row in selected:
        flags = ["heuristic_selection_unrated"]
        if analysis["beat_source"] != "detected_beats":
            flags.append("no_reliable_beat")
        if target < target_duration:
            flags.append("target_duration_clamped_to_source")
        if row["score"] > 0.6:
            flags.append("weak_endpoint_similarity")
        row.update({
            "source_start_seconds": row["start"] / sample_rate,
            "source_end_seconds": row["end"] / sample_rate,
            "crossfade_seconds": fade, "overlap_shortening_seconds": 0.0,
            "period_preserved": True, "effective_target_duration": target,
            "beat_source": analysis["beat_source"], "librosa_version": analysis.get("librosa_version"),
            "confidence": "low" if len(flags) > 1 else "moderate",
            "quality_flags": flags,
        })
    return selected
