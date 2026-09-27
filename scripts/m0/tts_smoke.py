"""Offline Irodori voice design -> clone -> emotion smoke test.

Run with services/worker/runtimes/irodori/.venv/Scripts/python.exe.
Download config/m0-models-tts.json first. This script never downloads weights.
The JSON result records signal checks; voice identity, reading and acting still
require listening to the WAV files. Process exit releases the CUDA context.
"""

from __future__ import annotations

import argparse
import functools
import gc
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKER_ROOT = PROJECT_ROOT / "services/worker"
RUNTIME_ROOT = WORKER_ROOT / "runtimes/irodori"
MANIFEST_PATH = PROJECT_ROOT / "config/m0-models-tts.json"
DEFAULT_REFERENCE_TEXT = "私はこの町の図書館で働いています。窓から見える青い海と、静かな朝の時間が好きです。今日はあなたに、大切なお話があります。"
DEFAULT_DIALOGUE_TEXT = "来てくれてありがとう。あなたと一緒なら、この扉の向こうへ進める気がする。"


def write_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def progress(message: str) -> None:
    print(f"[tts-smoke] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_scene_text(path: Path | None) -> dict[str, str]:
    """Keep displayed words separate from pronunciation and emoji annotations."""
    if path is None:
        return {
            "reference_text": DEFAULT_REFERENCE_TEXT,
            "dialogue_text": DEFAULT_DIALOGUE_TEXT,
            "display_text": DEFAULT_DIALOGUE_TEXT,
        }
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict):
        raise TypeError("--scene-json must contain a JSON object.")
    if "dialogue_text" not in data:
        raise ValueError("--scene-json requires dialogue_text.")
    scene = {
        "reference_text": data.get("reference_text", DEFAULT_REFERENCE_TEXT),
        "dialogue_text": data["dialogue_text"],
        "display_text": data.get("display_text", data["dialogue_text"]),
    }
    for name, value in scene.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"--scene-json {name} must be a non-empty string.")
    return scene


def configure_environment() -> None:
    # Set before importing Hugging Face, Torch, DACVAE or SilentCipher.
    cache = RUNTIME_ROOT / "cache"
    environment = {
        "HF_HOME": str(cache / "huggingface"),
        "HF_HUB_CACHE": str(cache / "huggingface/hub"),
        "TORCH_HOME": str(cache / "torch"),
        "XDG_CACHE_HOME": str(cache),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    os.environ.update(environment)
    ffmpeg = list((RUNTIME_ROOT / "ffmpeg").glob("*/bin/ffmpeg.exe"))
    if os.name == "nt":
        if len(ffmpeg) != 1:
            raise RuntimeError("Expected one worker-local FFmpeg shared installation.")
        os.environ["PATH"] = str(ffmpeg[0].parent) + os.pathsep + os.environ.get("PATH", "")
    sys.path.insert(0, str(RUNTIME_ROOT / "source"))


def verify_models(manifest: dict[str, Any], model_root: Path) -> dict[str, Path]:
    paths = {
        model["id"]: model_root / Path(model["local_dir"]).name
        for model in manifest["models"]
    }
    errors = []
    for model in manifest["models"]:
        for file in model["files"]:
            path = paths[model["id"]] / file["path"]
            if not path.is_file():
                errors.append(f"Missing: {path}")
            elif path.stat().st_size != file["size"]:
                errors.append(f"Incomplete or wrong size: {path}")
    if errors:
        raise FileNotFoundError("Model download is incomplete:\n" + "\n".join(errors))
    for model in manifest["models"]:
        progress(f"Verifying SHA256: {model['repo_id']}")
        for file in model["files"]:
            path = paths[model["id"]] / file["path"]
            if sha256(path) != file["sha256"]:
                raise ValueError(f"SHA256 mismatch: {path}")
    return paths


def gpu_memory(torch: Any, device: str) -> dict[str, int] | None:
    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    torch.cuda.synchronize(device)
    return {
        "allocated_bytes": torch.cuda.memory_allocated(device),
        "reserved_bytes": torch.cuda.memory_reserved(device),
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
    }


def inspect_wav(path: Path) -> dict[str, Any]:
    import numpy as np
    import soundfile as sf

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=True)
    if audio.size == 0 or not np.isfinite(audio).all():
        raise ValueError(f"Empty/non-finite waveform: {path}")
    duration = len(audio) / sample_rate
    rms = float(np.sqrt(np.mean(audio.astype(np.float64) ** 2)))
    if duration < 0.2 or rms < 1e-5:
        raise ValueError(f"Empty or silent speech: duration={duration}, rms={rms}: {path}")
    return {
        "path": str(path),
        "sha256": sha256(path),
        "sample_rate": sample_rate,
        "channels": audio.shape[1],
        "duration_seconds": duration,
        "rms": rms,
        "peak": float(np.max(np.abs(audio))),
        "clipped_sample_fraction": float(np.mean(np.abs(audio) >= 0.999)),
        "decoded_and_non_silent": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-root", type=Path, default=RUNTIME_ROOT / "models")
    parser.add_argument(
        "--scene-json", type=Path,
        help="JSON object with dialogue_text and optional reference_text/display_text.",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--model-precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--num-steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=4221)
    parser.add_argument("--preflight-only", action="store_true", help="Validate files without loading weights.")
    args = parser.parse_args()
    if args.num_steps < 1:
        parser.error("--num-steps must be positive")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "tts-result.json"
    report: dict[str, Any] = {
        "schema_version": 1,
        "status": "running",
        "stage": "preflight",
        "output_dir": str(output),
        "model_root": str(args.model_root.resolve()),
        "seed": args.seed,
        "num_steps": args.num_steps,
        "device": args.device,
        "model_precision": args.model_precision,
        "codec_precision": "fp32",
        "started_at_unix": time.time(),
        "samples": [],
        "human_review": {
            "status": "pending",
            "checks": ["Japanese reading", "reference/clone voice consistency", "emotion adherence", "word endings"],
            "note": "A short generated reference checks connectivity; it does not establish long-reference cloning quality.",
        },
    }
    runtime = None
    torch = None
    original_get_model = None
    silentcipher = None
    exit_code = 1
    write_json(report_path, report)
    try:
        configure_environment()
        scene = load_scene_text(args.scene_json)
        report["scene"] = scene
        if args.scene_json is not None:
            report["scene_source"] = {
                "path": str(args.scene_json.resolve()),
                "sha256": sha256(args.scene_json),
            }
        if Path(sys.prefix).resolve() != (RUNTIME_ROOT / ".venv").resolve():
            raise RuntimeError(f"Use the worker-local Irodori Python environment, not {sys.prefix}.")
        if not args.model_root.resolve().is_relative_to(WORKER_ROOT.resolve()):
            raise ValueError("--model-root must remain under services/worker for worker independence.")
        manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        report["models"] = manifest["models"]
        commit = subprocess.check_output(
            ["git", "-C", str(RUNTIME_ROOT / "source"), "rev-parse", "HEAD"], text=True
        ).strip()
        report["source_commit"] = commit
        if commit != manifest["source_commit"]:
            raise RuntimeError(f"Irodori source revision differs from the model manifest: {commit}")
        paths = verify_models(manifest, args.model_root.resolve())
        report["model_files_verified"] = True
        if args.preflight_only:
            report.update(status="preflight_passed", stage="complete")
            exit_code = 0
            return exit_code

        report["stage"] = "load_runtime"
        write_json(report_path, report)
        import silentcipher as silentcipher_module
        import torch as torch_module
        from irodori_tts.inference_runtime import (
            InferenceRuntime,
            RuntimeKey,
            SamplingRequest,
            save_wav,
        )
        from irodori_tts.text_normalization import normalize_text

        torch = torch_module
        silentcipher = silentcipher_module
        report["versions"] = {
            package: importlib.metadata.version(package)
            for package in ["torch", "torchaudio", "transformers", "safetensors", "silentcipher"]
        }
        if args.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable in the worker Irodori environment.")
        report["gpu_before_load"] = gpu_memory(torch, args.device)
        watermark_path = paths["irodori-silentcipher"] / "44_1_khz/73999_iteration"
        original_get_model = silentcipher.get_model
        # Preserve Irodori's automatic watermarking, but pass pinned local files
        # to the official loader so it cannot request an unpinned HF snapshot.
        silentcipher.get_model = functools.partial(
            original_get_model,
            ckpt_path=str(watermark_path),
            config_path=str(watermark_path / "hparams.yaml"),
        )
        progress("Loading pinned Irodori, DACVAE and SilentCipher models")
        runtime = InferenceRuntime.from_key(RuntimeKey(
            checkpoint=str(paths["irodori-v4.1-small"] / "model.safetensors"),
            model_device=args.device,
            model_precision=args.model_precision,
            codec_repo=str(paths["irodori-dacvae"] / "weights.pth"),
            codec_device=args.device,
            codec_precision="fp32",
        ))
        report["watermark_ready"] = bool(runtime.watermarker.ready)
        if not runtime.watermarker.ready:
            raise RuntimeError("The pinned SilentCipher watermark model failed to load.")
        if not runtime.model_cfg.use_speaker_condition_resolved or not runtime.model_cfg.use_caption_condition:
            raise RuntimeError("Selected checkpoint must support both voice design and speaker cloning.")
        report["gpu_after_load"] = gpu_memory(torch, args.device)

        reference_text = scene["reference_text"]
        clone_text = scene["dialogue_text"]
        clone_display_text = scene["display_text"]
        samples = [
            ("voice_design", reference_text, reference_text, reference_text, "neutral", "落ち着いた若い成人女性の、明瞭で柔らかな声。自然な速さで話す。", True),
            ("clone_neutral", clone_display_text, clone_text, clone_text, "neutral", None, False),
            ("clone_happy", clone_display_text, clone_text, "😊" + clone_text, "happy", "同じ声の特徴を保ち、嬉しそうに、明るく話す。", False),
            ("clone_worried", clone_display_text, clone_text, "😟" + clone_text, "worried", "同じ声の特徴を保ち、心配そうに、ためらいながら話す。", False),
        ]
        # Upstream batch_encode truncates silently; reject oversize speech so
        # a partial reading cannot be reported as a successful scene.
        report["text_token_counts"] = {}
        for name, _, _, tts_text, _, _, _ in samples:
            normalized = normalize_text(tts_text).strip()
            if not normalized:
                raise ValueError(f"{name}: text became empty after Irodori normalization.")
            count = int(runtime.tokenizer.encode(normalized).numel())
            report["text_token_counts"][name] = count
            if count > runtime.default_text_max_len:
                raise ValueError(
                    f"{name}: {count} tokens exceeds checkpoint text limit "
                    f"{runtime.default_text_max_len}; provide one shorter complete utterance."
                )
        reference_path = output / "voice_design.wav"
        for name, display_text, reading_text, tts_text, emotion, caption, no_ref in samples:
            progress(f"Generating {name}")
            report["stage"] = name
            write_json(report_path, report)
            began = time.perf_counter()
            result = runtime.synthesize(SamplingRequest(
                text=tts_text,
                caption=caption,
                no_ref=no_ref,
                ref_wav=None if no_ref else str(reference_path),
                seed=args.seed,
                num_steps=args.num_steps,
            ), log_fn=progress)
            path = output / f"{name}.wav"
            save_wav(path, result.audio, result.sample_rate)
            signal = inspect_wav(path)
            report["samples"].append({
                "id": name,
                "display_text": display_text,
                "reading_text": reading_text,
                "tts_input": tts_text,
                "caption": caption,
                "voice_emotion": emotion,
                "reference_audio": None if no_ref else str(reference_path),
                "seed": result.used_seed,
                "generation_seconds": time.perf_counter() - began,
                "stage_timings": result.stage_timings,
                "messages": result.messages,
                "audio": signal,
            })
            del result
            write_json(report_path, report)
        report.update(status="passed", stage="complete")
        exit_code = 0
    except Exception as exc:  # noqa: BLE001 - CLI boundary records failures in the result JSON.
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        traceback.print_exc()
    finally:
        if silentcipher is not None and original_get_model is not None:
            silentcipher.get_model = original_get_model
        runtime = None
        gc.collect()
        if torch is not None and args.device.startswith("cuda") and torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                report["gpu_after_release"] = gpu_memory(torch, args.device)
            except Exception as exc:  # noqa: BLE001 - Keep a release failure in the final report.
                report["release_error"] = str(exc)
                report["status"] = "failed"
                exit_code = 1
        report["finished_at_unix"] = time.time()
        report["gpu_release_note"] = "Allocator readings are process-local; compare nvidia-smi after process exit for cross-runtime release."
        write_json(report_path, report)
        progress(f"{report['status']}: {report_path}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
