"""Backend-neutral PCM boundary; model-specific options stay in the implementation."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

from . import stable_audio


@dataclass(frozen=True)
class MusicRequest:
    prompt: str
    duration_seconds: float = 120.0
    seed: int = 0
    context: dict = field(default_factory=dict)


@dataclass(frozen=True)
class GeneratedPCM:
    audio: Any  # [samples, stereo], finite float32; NumPy belongs to the child runtime.
    sample_rate: int
    provenance: dict


class MusicBackend(Protocol):
    def generate(self, request: MusicRequest, *, cancelled=None) -> GeneratedPCM: ...
    def close(self) -> None: ...


class StableAudioBackend:
    """One prepared SA3 model, shared across requests under the parent's GPU lease."""

    def __init__(self, settings: dict):
        self.settings = dict(settings)
        request = self._request(MusicRequest("Instrumental background music."))
        self.loaded = stable_audio.load_pipeline(request)
        self.loaded_at = time.monotonic()

    def _request(self, request: MusicRequest):
        return stable_audio.validate_request({
            "model": self.settings.get("model", "medium"),
            "model_path": self.settings["model_path"], "prompt": request.prompt,
            "duration": request.duration_seconds, "seed": request.seed,
            "steps": 8, "device": self.settings.get("device", "cuda"), "dtype": "float32",
            "cpu_offload": self.settings.get("cpu_offload", False), "context": request.context,
        })

    def generate(self, request: MusicRequest, *, cancelled=None) -> GeneratedPCM:
        settings = self._request(request)
        audio, sample_rate, timing = stable_audio.infer_pcm(self.loaded, settings, cancelled=cancelled)
        return GeneratedPCM(audio, sample_rate, {
            "backend": "stable_audio3", "model": settings.model,
            "model_identity": self.loaded.model_identity, "effective_seed": settings.seed,
            "duration_seconds": settings.duration, "steps": 8, "guidance_scale": 1.0,
            "precision": self.loaded.dtype_name, "device": self.loaded.device,
            "cpu_offload": settings.cpu_offload,
            "vae_decode": {"chunk_size": 128, "overlap": 32, "clamp_output": False},
            "versions": {"torch": str(self.loaded.torch.__version__),
                "diffusers": self.loaded.diffusers_version, "numpy": str(self.loaded.np.__version__)},
            **timing,
        })

    def close(self):
        loaded, self.loaded = self.loaded, None
        if loaded is not None:
            loaded.pipeline.maybe_free_model_hooks()


def load_backend(backend: str, settings: dict) -> MusicBackend:
    if backend != "stable_audio3":
        raise ValueError("Unsupported music backend.")
    return StableAudioBackend(settings)
