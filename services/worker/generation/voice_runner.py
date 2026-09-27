"""Execute under the pinned Irodori runtime, optionally reusing it for JSONL requests."""

from __future__ import annotations

import argparse
import contextlib
import functools
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import TextIO

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from scripts.m0.tts_smoke import (
    MANIFEST_PATH,
    RUNTIME_ROOT,
    configure_environment,
    inspect_wav,
    verify_models,
)


def validate_request(request: dict, output: Path) -> Path | None:
    """Validate the recorded reference before loading any model."""
    if request.get("mode", "design") not in ("design", "clone"):
        raise ValueError("Unsupported voice synthesis mode.")
    if not isinstance(request.get("text"), str) or not request["text"].strip():
        raise ValueError("Speech text is required.")
    if request.get("mode", "design") != "clone":
        return None
    if not isinstance(request.get("reference_text"), str) or not request["reference_text"].strip():
        raise ValueError("The reference voice must retain its exact transcript.")
    reference = (output / "reference.wav").resolve(strict=True)
    if not reference.is_relative_to(output.resolve()):
        raise ValueError("The voice reference must remain inside the generation directory.")
    if hashlib.sha256(reference.read_bytes()).hexdigest() != request.get("reference_sha256"):
        raise ValueError("Reference voice SHA256 mismatch.")
    return reference


class VoiceRuntimeSession:
    """Keep model weights, but never a request or its speaker conditioning, between jobs."""

    def __init__(self) -> None:
        self.runtime = None
        self.model_precision = None

    def _initialize(self, precision: str) -> None:
        if Path(sys.prefix).resolve() != (RUNTIME_ROOT / ".venv").resolve():
            raise RuntimeError("Voice design requires the isolated worker Irodori Python.")
        configure_environment()
        self.manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        self.commit = subprocess.check_output(
            ["git", "-C", str(RUNTIME_ROOT / "source"), "rev-parse", "HEAD"], text=True
        ).strip()
        if self.commit != self.manifest["source_commit"]:
            raise RuntimeError("Pinned Irodori source revision mismatch.")
        paths = verify_models(self.manifest, RUNTIME_ROOT / "models")
        import numpy as np
        import silentcipher
        import soundfile as sf
        import torch
        from irodori_tts.inference_runtime import InferenceRuntime, RuntimeKey, SamplingRequest
        from irodori_tts.text_normalization import normalize_text

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for Irodori voice design.")
        watermark = paths["irodori-silentcipher"] / "44_1_khz/73999_iteration"
        silentcipher.get_model = functools.partial(
            silentcipher.get_model,
            ckpt_path=str(watermark),
            config_path=str(watermark / "hparams.yaml"),
        )
        runtime = InferenceRuntime.from_key(
            RuntimeKey(
                checkpoint=str(paths["irodori-v4.1-small"] / "model.safetensors"),
                model_device="cuda",
                model_precision=precision,
                codec_repo=str(paths["irodori-dacvae"] / "weights.pth"),
                codec_device="cuda",
                codec_precision="fp32",
            )
        )
        if not runtime.watermarker.ready:
            raise RuntimeError("Pinned automatic voice watermark failed to initialize.")
        self.versions = {
            name: importlib.metadata.version(name)
            for name in ("torch", "transformers", "soundfile", "silentcipher")
        }
        self.np, self.sf = np, sf
        self.sampling_request = SamplingRequest
        self.normalize_text = normalize_text
        self.model_precision = precision
        self.runtime = runtime

    def generate(self, output: Path) -> None:
        started = time.perf_counter()
        request = json.loads((output / "request.json").read_text(encoding="utf-8"))
        reference = validate_request(request, output)
        model_reused = self.runtime is not None
        initialization_seconds = 0.0
        if model_reused:
            if self.model_precision != request["model_precision"]:
                raise RuntimeError("Voice model precision changed; restart the voice runtime.")
        else:
            initialization_started = time.perf_counter()
            self._initialize(request["model_precision"])
            initialization_seconds = time.perf_counter() - initialization_started
        runtime = self.runtime
        if reference is not None and not runtime.model_cfg.use_speaker_condition_resolved:
            raise RuntimeError("The pinned model does not support reference voice cloning.")
        for name, text, tokenizer, limit in (
            (
                "speech",
                self.normalize_text(request["text"]).strip(),
                runtime.tokenizer,
                runtime.default_text_max_len,
            ),
            (
                "caption",
                request.get("caption", "").strip(),
                runtime.caption_tokenizer,
                runtime.default_caption_max_len,
            ),
        ):
            if name == "caption" and not text and reference is not None:
                continue
            if not text or tokenizer is None or int(tokenizer.encode(text).numel()) > limit:
                raise ValueError(
                    f"{name} is empty or exceeds the checkpoint token limit; truncation is forbidden."
                )
        if reference is not None:
            inspect_wav(reference)
        # This pinned Irodori uses waveform speaker conditioning and has no transcript
        # parameter. A fresh SamplingRequest prevents another character's reference or
        # caption from leaking into this job. Its internal KV cache is local to synthesize.
        synthesis_started = time.perf_counter()
        generated = runtime.synthesize(
            self.sampling_request(
                text=request["text"],
                caption=request.get("caption") or None,
                no_ref=reference is None,
                ref_wav=str(reference) if reference else None,
                seed=request["seed"],
                num_steps=request["num_steps"],
            ),
            log_fn=lambda message: print(message, flush=True),
        )
        synthesis_seconds = time.perf_counter() - synthesis_started
        # PCM16 is part of the coordinator contract; never send float WAV or silent audio.
        audio = generated.audio
        if hasattr(audio, "detach"):
            audio = audio.detach().cpu().numpy()
        audio = self.np.asarray(audio).squeeze()
        if audio.ndim != 1 or audio.size == 0 or not self.np.isfinite(audio).all():
            raise ValueError("Voice design returned an invalid waveform.")
        self.sf.write(output / "voice.wav", audio, generated.sample_rate, subtype="PCM_16")
        signal = inspect_wav(output / "voice.wav")
        signal.pop("path", None)
        report = {
            **signal,
            "model": "Irodori-TTS-v4.1-Small",
            "source_commit": self.commit,
            "model_revision": self.manifest["models"][0]["revision"],
            "seed": generated.used_seed,
            "mode": request.get("mode", "design"),
            "text": request["text"],
            "caption": request.get("caption", ""),
            "reference_text": request["reference_text"] if reference else request["text"],
            "reference_sha256": request.get("reference_sha256") if reference else None,
            "reference_artifact_id": request.get("reference_artifact_id") if reference else None,
            "num_steps": request["num_steps"],
            "watermarked": True,
            "format": "PCM_16",
            "versions": self.versions,
            "runtime_pid": os.getpid(),
            "model_reused": model_reused,
            "timings": {
                "initialization_seconds": initialization_seconds,
                "synthesis_seconds": synthesis_seconds,
                "job_seconds": time.perf_counter() - started,
            },
        }
        (output / "result.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )


@contextlib.contextmanager
def request_log(output: Path):
    """Keep Python and native library output away from the JSONL protocol stream."""
    with (output / "runtime.log").open("a", encoding="utf-8", buffering=1) as log:
        sys.stdout.flush()
        sys.stderr.flush()
        stdout_fd, stderr_fd = os.dup(1), os.dup(2)
        try:
            os.dup2(log.fileno(), 1)
            os.dup2(log.fileno(), 2)
            with contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
                try:
                    yield
                except Exception:
                    traceback.print_exc()
                    raise
                finally:
                    log.flush()
        finally:
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def serve(input_stream: TextIO, output_stream: TextIO) -> int:
    session = VoiceRuntimeSession()
    for line in input_stream:
        try:
            message = json.loads(line)
            if not isinstance(message, dict) or not isinstance(message.get("output_dir"), str):
                raise TypeError("Voice runtime request requires an absolute output_dir.")
            output = Path(message["output_dir"])
            if not output.is_absolute() or not output.is_dir():
                raise ValueError("Voice runtime output_dir must be an existing absolute directory.")
            with request_log(output.resolve()):
                session.generate(output.resolve())
        except Exception as error:  # noqa: BLE001 - protocol boundary must report and exit on failure
            print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False),
                  file=output_stream, flush=True)
            # Discard the whole process after any failure, including a partial CUDA operation.
            return 1
        print(json.dumps({"ok": True}), file=output_stream, flush=True)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--output-dir", type=Path)
    mode.add_argument("--serve", action="store_true")
    args = parser.parse_args()
    if args.serve:
        return serve(sys.stdin, sys.stdout)
    VoiceRuntimeSession().generate(args.output_dir.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
