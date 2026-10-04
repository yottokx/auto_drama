"""Dedicated music child. The worker parent owns the sole GPU lock.

JSON lines on stdout are protocol replies; all inference/library output goes to stderr.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from contextlib import redirect_stdout
from pathlib import Path

from .music import stable_audio
from .music.backends import GeneratedPCM, MusicRequest, load_backend
from .music.json_io import write_json
from .music.loops import _intro_once_variant, region_crossfade, seam_metrics, select_regions
from .music.quality import inspect_quality, require_usable

MUSIC_PROTOCOL_VERSION = 1
LOOP_PROTOCOL_VERSION = 2


def identity(settings: dict) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def generate(request: dict, backend, *, reused=False) -> dict:
    music = MusicRequest(request["prompt"], request["duration_seconds"], request["seed"], request["context"])
    pcm = backend.generate(music)
    return process_pcm(request, pcm, backend.loaded.np, loaded_at=backend.loaded_at, reused=reused)


def process_pcm(request: dict, pcm: GeneratedPCM, np, *, loaded_at=None, reused=False):
    """CPU-only validation, region search and encoding shared with uploaded tracks."""
    output = Path(request["output_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("source.mp3", "music.mp3", "result.json")):
        raise ValueError("Music output must be a fresh directory.")
    quality = inspect_quality(pcm.audio, pcm.sample_rate, np=np, expected_duration=request["duration_seconds"])
    require_usable(quality)
    crossfade = stable_audio._number(request.get("crossfade_seconds", 0.5), "crossfade_seconds")
    target = stable_audio._number(request.get("loop_target_seconds", 60), "loop_target_seconds")
    if not 0.01 <= crossfade <= 2 or not 5 <= target <= 380:
        raise ValueError("Invalid automatic-loop settings.")
    regions = (select_regions(pcm.audio, pcm.sample_rate, target, crossfade, 3, np,
        minimum_duration=5.0) if request["duration_seconds"] >= 7 else [])
    if request["duration_seconds"] >= 7 and not regions:
        raise ValueError("Automatic loop search returned no usable region.")
    chosen = None
    if not regions:
        # A five-second input has no room for both an intro and a five-second
        # loop. Keep the full period and blend the tail towards its opening.
        candidate = pcm.audio.copy()
        fade = min(round(crossfade * pcm.sample_rate), len(candidate) // 4)
        weights = (0.5 - 0.5 * np.cos(np.linspace(0, np.pi, fade)))[:, None]
        candidate[-fade:] = candidate[-fade:] * (1 - weights) + candidate[:fade] * weights
        loop = {"start_sample": 0, "end_sample": len(candidate), "start_seconds": 0.0,
            "end_seconds": len(candidate) / pcm.sample_rate, "score": None, "confidence": "unrated",
            "quality_flags": ["short_clip_whole_file"], "crossfade_seconds": fade / pcm.sample_rate,
            "format_version": 2, "playback_layout": "intro_then_loop", "intro_once": True,
            "period_samples": len(candidate), "period_seconds": len(candidate) / pcm.sample_rate}
        candidate_quality = inspect_quality(candidate, pcm.sample_rate, np=np)
        require_usable(candidate_quality)
        chosen = candidate, loop, candidate_quality
    for row in regions:
        circular = region_crossfade(pcm.audio, row["start"], row["end"],
            round(row["crossfade_seconds"] * pcm.sample_rate), np)
        candidate, loop = _intro_once_variant(pcm.audio, circular, row, "smart_region", pcm.sample_rate, np)
        candidate_quality = inspect_quality(candidate[loop["start_sample"]:loop["end_sample"]], pcm.sample_rate, np=np)
        if not candidate_quality["needs_review"]:
            chosen = candidate, loop, candidate_quality
            break
    if chosen is None:
        raise ValueError("No usable automatic loop region was found.")
    candidate, loop, loop_quality = chosen
    source_stats = stable_audio._save_audio(pcm.audio.T[None], pcm.sample_rate, output / "source.mp3", np,
        mp3_bitrate=192, analyze_silence=True)
    music_stats = stable_audio._save_audio(candidate.T[None], pcm.sample_rate, output / "music.mp3", np,
        mp3_bitrate=192, analyze_silence=True)
    quality.update(loop=loop_quality, loop_selection={
        "method": "smart_region", "score": loop["score"], "confidence": loop["confidence"],
        "flags": loop["quality_flags"], "crossfade_seconds": loop["crossfade_seconds"],
        "source_seam": seam_metrics(pcm.audio, pcm.sample_rate, np),
        "output_seam": seam_metrics(candidate[loop["start_sample"]:loop["end_sample"]], pcm.sample_rate, np),
    })
    result = {"scene_id": request["scene_id"], "prompt": request["prompt"],
        "loop_start_seconds": loop["start_seconds"], "loop_end_seconds": loop["end_seconds"],
        "duration_seconds": len(candidate) / pcm.sample_rate,
        "source_duration_seconds": len(pcm.audio) / pcm.sample_rate,
        "sample_rate": pcm.sample_rate, "quality": quality}
    provenance = {**pcm.provenance, "music_protocol_version": MUSIC_PROTOCOL_VERSION,
        "loop_protocol_version": LOOP_PROTOCOL_VERSION, "model_reused": reused,
        "loop": {key: value for key, value in loop.items() if key not in {"warnings", "playback_note"}},
        "encoding": {"format": "mp3", "bitrate_kbps": 192,
            "source_preview_gain": source_stats["preview_gain"], "music_preview_gain": music_stats["preview_gain"]}}
    report = {"ok": True, "result": result, "provenance": provenance}
    if loaded_at is not None:
        report["model_loaded_at_monotonic"] = loaded_at
        provenance["model_loaded_at_monotonic"] = loaded_at
    write_json(output / "result.json", report)
    return report


def import_audio(request: dict, output: Path):
    from .music.audio_files import decode_audio, probe_audio

    info = probe_audio(request["source_audio"])
    if not 5 <= info["duration_seconds"] <= 380:
        raise ValueError("Imported music must be 5–380 seconds.")
    np = importlib.import_module("numpy")
    audio = decode_audio(request["source_audio"], sample_rate=44100)
    duration = len(audio) / 44100
    if not 5 <= duration <= 380:
        raise ValueError("Decoded imported music must be 5–380 seconds.")
    request = {**request, "output_dir": str(output), "duration_seconds": duration,
        "prompt": request.get("prompt") or "Imported instrumental background music."}
    pcm = GeneratedPCM(audio, 44100, {"backend": "imported", "model": "user_upload",
        "source_sha256": hashlib.sha256(Path(request["source_audio"]).read_bytes()).hexdigest(),
        "duration_seconds": duration})
    return process_pcm(request, pcm, np)


def serve():
    backend = None
    key = None
    try:
        for line in sys.stdin:
            try:
                request = json.loads(line)
                requested_key = identity(request["settings"])
                reused = backend is not None and requested_key == key
                with redirect_stdout(sys.stderr):
                    if not reused:
                        if backend is not None:
                            backend.close()
                        backend = load_backend(request["settings"]["backend"], request["settings"])
                        key = requested_key
                    generate(request, backend, reused=reused)
                reply = {"ok": True}
            except Exception as exc:  # noqa: BLE001 - serialize child failures for the parent owner
                reply = {"ok": False, "error": str(exc)}
            print(json.dumps(reply, ensure_ascii=False), flush=True)
    finally:
        if backend is not None:
            backend.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--import-request", type=Path)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.import_request is not None:
        if args.serve or args.output_dir is None:
            parser.error("Import requires --output-dir and cannot be combined with --serve.")
        import_audio(json.loads(args.import_request.read_text(encoding="utf-8")), args.output_dir)
        return
    if not args.serve:
        parser.error("The music child requires --serve; its parent owns GPU exclusion.")
    serve()


if __name__ == "__main__":
    main()
