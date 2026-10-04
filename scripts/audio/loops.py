"""Loop experiments on decoded PCM, with explicit provenance and confidence limits."""

from __future__ import annotations

import importlib
import json
import math
import secrets
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .engine import (
    MP3_BITRATES,
    AudioGenerationError,
    GenerationCancelled,
    _save_audio,
    utc_timestamp,
)
from .json_io import write_json

METHODS = {
    "whole_crossfade": "全曲クロスフェード",
    "smart_region": "中盤のループ候補",
    "ai_bridge": "AIで継ぎ目を修復",
    "smart_ai_bridge": "中盤候補をAIで継ぎ目修復",
}
AI_METHODS = {"ai_bridge", "smart_ai_bridge"}
SAMPLE_RATE = 44100
_FLAG_MESSAGES = {
    "intro_and_outro_retained": "原曲のイントロやフェードアウトを含みます。",
    "musical_loop_unrated": "音楽的な自然さは未評価です。連続再生で確認してください。",
    "heuristic_selection_unrated": "特徴量による候補選択です。和声やフレーズの自然な終止は保証しません。",
    "no_reliable_beat": "安定した拍を検出できず、音の変化や特徴フレームを候補点に使っています。",
    "target_duration_clamped_to_source": "元音声の長さに合わせて目標秒数を短くしました。",
    "weak_endpoint_similarity": "区間両端の類似度が低いため、継ぎ目が目立つ可能性があります。",
    "experimental_ai_bridge": "AIによる局所修復です。旋律・拍・音色の維持は保証しません。",
}


@dataclass(frozen=True)
class LoopRequest:
    source_audio: str
    source_generation: dict
    method: str = "smart_region"
    target_duration: float = 60.0
    crossfade_seconds: float = 0.5
    seed: int = 0
    candidate_count: int = 3
    output_format: str = "mp3"
    keep_wav: bool = False
    mp3_bitrate: int = 192
    model: str = "medium"
    model_path: str = ""
    device: str = "auto"
    dtype: str = "auto"
    steps: int = 8
    cpu_offload: bool = False
    bridge_context_seconds: float = 15.0
    bridge_seconds: float = 4.0


def _numeric(value, name, low, high, *, integer=False):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{name} は数値で指定してください。") from exc
    if isinstance(value, bool) or not math.isfinite(number) or not low <= number <= high:
        raise ValueError(f"{name} は {low:g}〜{high:g} の範囲で指定してください。")
    if integer and not number.is_integer():
        raise ValueError(f"{name} は整数で指定してください。")
    return int(number) if integer else number


def validate_loop_request(data: Mapping[str, Any]) -> LoopRequest:
    if not isinstance(data, Mapping):
        raise TypeError("ループ設定はJSONオブジェクトで指定してください。")
    source = data.get("source_audio")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("ループ化する音声ファイルを指定してください。")
    metadata = data.get("source_generation", {})
    if isinstance(metadata, str):
        try:
            metadata = json.loads(Path(metadata).read_text(encoding="utf-8-sig"))
        except (OSError, ValueError) as exc:
            raise ValueError("元音声の生成記録を読み込めません。") from exc
    if not isinstance(metadata, dict):
        raise TypeError("元音声の生成記録はJSONオブジェクトで指定してください。")
    metadata = metadata.get("metadata", metadata)
    if not isinstance(metadata, dict):
        raise TypeError("元音声の生成記録を読み込めません。")
    metadata = json.loads(json.dumps(metadata, ensure_ascii=False, allow_nan=False))
    method = data.get("method", "smart_region")
    if method not in METHODS:
        raise ValueError("全曲・中盤候補・全曲AI修復・中盤AI修復から選択してください。")
    if method in AI_METHODS and data.get("model", "medium") not in {"small", "medium"}:
        raise ValueError("AIループ補修にはStable AudioのSmallまたはMediumを選択してください。"
                         "ACE-Stepの生成音声はクロスフェード・自動探索でループ化できます。")
    output_format = str(data.get("output_format", "mp3")).casefold()
    if output_format not in {"mp3", "wav"}:
        raise ValueError("保存形式はMP3またはWAVを選択してください。")
    keep_wav, offload = data.get("keep_wav", False), data.get("cpu_offload", False)
    if not isinstance(keep_wav, bool) or not isinstance(offload, bool):
        raise TypeError("WAV保存・CPUオフロード設定は真偽値で指定してください。")
    bitrate = _numeric(data.get("mp3_bitrate", 192), "MP3ビットレート", 32, 320, integer=True)
    if bitrate not in MP3_BITRATES:
        raise ValueError("MP3ビットレートは192・256・320 kbpsを選択してください。")
    return LoopRequest(
        source_audio=str(Path(source.strip()).expanduser().resolve()), source_generation=metadata,
        method=method, output_format=output_format, keep_wav=keep_wav, cpu_offload=offload,
        target_duration=_numeric(data.get("target_duration", 60), "候補の目標秒数", 10, 380),
        crossfade_seconds=_numeric(data.get("crossfade_seconds", 0.5), "クロスフェード秒数", 0.01, 8),
        seed=_numeric(data.get("seed", 0), "シード", -1, 2**32 - 1, integer=True),
        candidate_count=_numeric(data.get("candidate_count", 3), "候補数", 1, 3, integer=True),
        mp3_bitrate=bitrate,
        model=str(data.get("model", "medium")), model_path=str(data.get("model_path", "")),
        device=str(data.get("device", "auto")), dtype=str(data.get("dtype", "auto")),
        steps=_numeric(data.get("steps", 8), "ステップ数", 1, 1000, integer=True),
        bridge_context_seconds=_numeric(data.get("bridge_context_seconds", 15), "AI参照秒数", 10, 20),
        bridge_seconds=_numeric(data.get("bridge_seconds", 4), "AI修復秒数", 2, 8),
    )


def check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise GenerationCancelled("ループ実験を停止しました。")


def validate_pcm(audio, np):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or audio.shape[1] != 2 or len(audio) < 4:
        raise AudioGenerationError("ループ元の音声は空でないstereo PCMである必要があります。")
    if not np.isfinite(audio).all():
        raise AudioGenerationError("ループ元の音声にNaN/Infが含まれています。")
    if float(np.max(np.abs(audio))) < 1e-7:
        raise AudioGenerationError("元の音声が無音のため、ループ候補を作成できません。")
    return audio


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


def whole_crossfade(audio, sample_rate: int, crossfade_seconds: float, np):
    """Overlap tail with head, rotating the start beyond the overlap."""
    fade = min(max(2, round(crossfade_seconds * sample_rate)), (len(audio) - 2) // 3)
    if fade < 2:
        raise AudioGenerationError("クロスフェードするには音声が短すぎます。")
    weights = fade_weights(fade, np)
    transition = audio[-fade:] * (1 - weights) + audio[:fade] * weights
    result = np.concatenate((audio[fade:-fade], transition)).astype(np.float32)
    return result, {
        "source_start_seconds": 0.0, "source_end_seconds": len(audio) / sample_rate,
        "crossfade_seconds": fade / sample_rate, "overlap_shortening_seconds": fade / sample_rate,
        "period_preserved": False, "crossfade_curve": "complementary_raised_cosine", "confidence": "unrated",
        "quality_flags": ["intro_and_outro_retained", "musical_loop_unrated"],
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


def create_loops(request: LoopRequest | Mapping[str, Any], output_dir: Path | str, *,
                 progress: Callable[[dict], None] | None = None, cancelled=None) -> dict:
    request = validate_loop_request(asdict(request) if isinstance(request, LoopRequest) else request)
    np = importlib.import_module("numpy")
    from .audio_files import AudioFileError, ExportCancelled, decode_audio

    started = time.perf_counter()
    directory = Path(output_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if (directory / "result.json").exists() or any(directory.glob("candidate-*")):
        raise AudioGenerationError("既存のループ結果は上書きできません。新しい出力先を指定してください。")

    def report(phase, message, **extra):
        check_cancelled(cancelled)
        if progress:
            progress({"phase": phase, "message": message, "updated_at": utc_timestamp(),
                      "elapsed_seconds": round(time.perf_counter() - started, 3), **extra})

    report("decoding", "元音声をPCMに展開しています。")
    try:
        audio = validate_pcm(decode_audio(request.source_audio, sample_rate=SAMPLE_RATE, cancelled=cancelled), np)
    except ExportCancelled as exc:
        raise GenerationCancelled(str(exc)) from exc
    except AudioFileError as exc:
        raise AudioGenerationError(str(exc)) from exc
    source_metrics = seam_metrics(audio, SAMPLE_RATE, np)
    selected_seed = secrets.randbelow(2**32) if request.seed == -1 else request.seed
    report("analyzing", "ループ区間と継ぎ目を調べています。")
    if request.method == "whole_crossfade":
        result_audio, info = whole_crossfade(audio, SAMPLE_RATE, request.crossfade_seconds, np)
        variants = [(result_audio, info)]
    elif request.method == "smart_region":
        regions = select_regions(audio, SAMPLE_RATE, request.target_duration, request.crossfade_seconds,
                                 request.candidate_count, np, cancelled=cancelled)
        variants = [(region_crossfade(audio, row["start"], row["end"],
                                     round(row["crossfade_seconds"] * SAMPLE_RATE), np), row)
                    for row in regions]
    elif request.method == "ai_bridge":
        from .loop_ai import repair_bridge

        result_audio, info = repair_bridge(audio, request, SAMPLE_RATE, seed=selected_seed,
                                          progress=progress, cancelled=cancelled)
        variants = [(result_audio, info)]
    else:
        from .loop_ai import repair_bridge

        row = select_regions(audio, SAMPLE_RATE, request.target_duration, request.crossfade_seconds,
                             1, np, cancelled=cancelled, minimum_duration=10.0)[0]
        start, end = row["start"], row["end"]
        if end - start < 10 * SAMPLE_RATE:
            raise AudioGenerationError("中盤AI修復には10秒以上の候補区間が必要です。")
        result_audio, bridge_info = repair_bridge(audio[start:end], request, SAMPLE_RATE,
                                                 seed=selected_seed, progress=progress, cancelled=cancelled)
        info = {**row, **bridge_info,
                "source_start_seconds": start / SAMPLE_RATE, "source_end_seconds": end / SAMPLE_RATE,
                "selection_crossfade_seconds": row["crossfade_seconds"],
                "requested_candidate_count": request.candidate_count,
                "candidate_policy": "single_best_region_one_ai_repair",
                "quality_flags": list(dict.fromkeys(row["quality_flags"] + bridge_info["quality_flags"]))}
        variants = [(result_audio, info)]
    variants = [_intro_once_variant(audio, candidate, information, request.method, SAMPLE_RATE, np)
                for candidate, information in variants]
    results = []
    for rank, (candidate, information) in enumerate(variants, 1):
        report("saving", f"候補 {rank}/{len(variants)} を保存しています。")
        folder = directory / f"candidate-{rank:02d}"
        folder.mkdir(exist_ok=False)
        destination = folder / ("output." + request.output_format)
        stats = _save_audio(candidate.T[None], SAMPLE_RATE, destination, np, keep_wav=request.keep_wav,
                            mp3_bitrate=request.mp3_bitrate, cancelled=cancelled)
        source_metadata = request.source_generation
        context = json.loads(json.dumps(source_metadata.get("context") or {}, ensure_ascii=False))
        scene_label = str(context.get("scene_label") or context.get("scene_id") or Path(request.source_audio).stem)
        context["scene_label"] = f"{scene_label} / {METHODS[request.method]} {rank}"
        information["warnings"] = [_FLAG_MESSAGES.get(flag, flag) for flag in information.get("quality_flags", [])]
        metadata = {
            "created_at": utc_timestamp(), "source_audio": request.source_audio,
            "input_duration_seconds": len(audio) / SAMPLE_RATE,
            "source_generation": source_metadata, "context": context,
            "settings": {**(source_metadata.get("settings") or {}),
                         "model": request.model if request.method in AI_METHODS else (source_metadata.get("settings") or {}).get("model", "PCM"),
                         "duration": len(candidate) / SAMPLE_RATE, "seed": selected_seed,
                         "output_format": request.output_format, "mp3_bitrate": request.mp3_bitrate},
            "loop": {"method": request.method, "rank": rank,
                     "target_duration": request.target_duration, **information,
                     "source_seam": source_metrics,
                     "output_seam": seam_metrics(candidate[information["start_sample"]:information["end_sample"]], SAMPLE_RATE, np)},
            "loop_settings": {key: value for key, value in asdict(request).items() if key != "source_generation"},
            **stats, "elapsed_seconds": round(time.perf_counter() - started, 3),
        }
        # Encoded file duration includes the one-time intro; loop period is explicit above.
        metadata["loop"]["source_duration_seconds"] = len(audio) / SAMPLE_RATE
        metadata_path = folder / "generation.json"
        write_json(metadata_path, metadata)
        result = {"ok": True, "rank": rank, "directory": str(folder), "audio_path": str(destination),
                  "wav_audio_path": stats.get("wav_audio_path"),
                  "float_audio_path": stats.get("float_audio_path"),
                  "metadata_path": str(metadata_path), "metadata": metadata}
        check_cancelled(cancelled)
        results.append(result)
    check_cancelled(cancelled)
    for result in results:
        write_json(Path(result["directory"]) / "result.json", result)
    report("done", f"{len(results)}件の候補を保存しました。連続再生で継ぎ目を比較してください。")
    return {**results[0], "candidates": results, "candidate_count": len(results)}
