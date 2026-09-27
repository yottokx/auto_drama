"""Model-free verification of the worker's independent Irodori environment."""

from __future__ import annotations

import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile


def main() -> None:
    runtime = Path(__file__).resolve().parent
    source = runtime / "source"
    ffmpeg = list((runtime / "ffmpeg").glob("*/bin/ffmpeg.exe"))
    if len(ffmpeg) != 1:
        raise RuntimeError("Expected one local FFmpeg shared installation.")
    if not Path(sys.executable).resolve().is_relative_to(runtime):
        raise RuntimeError("Verification must use the Irodori runtime's Python.")
    if not Path(sys.base_prefix).resolve().is_relative_to(runtime / ".python"):
        raise RuntimeError("Irodori must use its own managed Python installation.")
    os.environ.update(
        HF_HOME=str(runtime / "cache/huggingface"),
        HF_HUB_CACHE=str(runtime / "cache/huggingface/hub"),
        TORCH_HOME=str(runtime / "cache/torch"),
        XDG_CACHE_HOME=str(runtime / "cache"),
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        PYTHONUTF8="1",
        PATH=str(ffmpeg[0].parent) + os.pathsep + os.environ.get("PATH", ""),
    )
    sys.path.insert(0, str(source))
    import dacvae
    import numpy as np
    import silentcipher
    import soundfile as sf
    import torch
    import torchaudio
    import torchcodec
    from irodori_tts.inference_runtime import InferenceRuntime

    for module in (dacvae, silentcipher, torch, torchaudio, torchcodec):
        if not Path(module.__file__).resolve().is_relative_to(runtime / ".venv"):
            raise RuntimeError(f"Dependency was loaded from outside the runtime: {module.__name__}")
    if not torch.cuda.is_available():
        raise RuntimeError("The configured cu128 environment requires a working NVIDIA GPU.")
    tensor = torch.ones((32, 32), device="cuda")
    result = (tensor @ tensor).sum().item()
    if result != 32768:
        raise RuntimeError(f"CUDA operation produced an unexpected result: {result}")
    del tensor
    torch.cuda.empty_cache()

    with tempfile.TemporaryDirectory(prefix="audio-check-", dir=runtime) as temporary:
        sample = Path(temporary) / "sample.wav"
        sf.write(sample, np.zeros(480, dtype=np.float32), 48000)
        decoded, sample_rate = torchaudio.load(str(sample))
        if sample_rate != 48000 or tuple(decoded.shape) != (1, 480):
            raise RuntimeError("The reference-audio decoder returned unexpected metadata.")

    cli = subprocess.run(
        [sys.executable, "-E", "-s", "-X", "utf8", str(source / "infer.py"), "--help"],
        cwd=source,
        capture_output=True,
        text=True,
        encoding="utf-8",
        check=True,
    )
    if "--caption" not in cli.stdout or "--checkpoint" not in cli.stdout:
        raise RuntimeError("Irodori inference CLI did not expose the expected arguments.")
    commit = subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip()
    report = {
        "status": "ok",
        "source_commit": commit,
        "source_is_git_clone": (source / ".git").is_dir(),
        "python": sys.version.split()[0],
        "python_executable": sys.executable,
        "python_base_prefix": sys.base_prefix,
        "dependencies": {
            name: importlib.metadata.version(name)
            for name in ("torch", "torchaudio", "torchcodec", "transformers", "dacvae", "silentcipher")
        },
        "cuda_version": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "cuda_operation": "passed",
        "audio_decode": "passed: mono WAV, 48000 Hz, 480 samples",
        "inference_cli": "passed: infer.py --help",
        "ffmpeg": subprocess.check_output([str(ffmpeg[0]), "-version"], text=True).splitlines()[0],
        "model_inference": "not tested; model weights have not been installed",
    }
    (runtime / ".runtime-state.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
