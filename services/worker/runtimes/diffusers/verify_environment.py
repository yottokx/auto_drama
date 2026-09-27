"""Verify the worker image runtime without downloading or loading model weights."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-gpu", action="store_true")
    args = parser.parse_args()
    runtime = Path(__file__).resolve().parent
    if not Path(sys.executable).resolve().is_relative_to(runtime / ".venv"):
        raise RuntimeError("Run this diagnostic with the Diffusers runtime's .venv Python.")

    os.environ["HF_HOME"] = str(runtime / "cache" / "huggingface")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["U2NET_HOME"] = str(runtime / "models" / "rembg")
    os.environ["NUMBA_CACHE_DIR"] = str(runtime / "cache" / "numba")
    os.environ["PYTHONNOUSERSITE"] = "1"
    os.environ.pop("PYTHONPATH", None)

    import onnxruntime
    import rembg
    import torch
    import torchvision
    from diffusers import AnimaAutoBlocks, AnimaModularPipeline, AnimaTextConditioner
    from diffusers import AutoencoderKLQwenImage, CosmosTransformer3DModel, ModularPipeline
    from transformers import Qwen3Model, T5TokenizerFast

    expected_commit = "7263f3317f6b392d62f41e9d75ed9d7e21fc5a5c"
    source_commit = subprocess.check_output(
        ["git", "-C", str(runtime / "source"), "rev-parse", "HEAD"], text=True
    ).strip()
    if source_commit != expected_commit:
        raise RuntimeError(f"Unexpected Diffusers checkout: {source_commit}")
    direct_url = json.loads(importlib.metadata.distribution("diffusers").read_text("direct_url.json"))
    if direct_url.get("vcs_info", {}).get("commit_id") != expected_commit:
        raise RuntimeError("Installed Diffusers package does not match the pinned source commit.")

    converter = runtime / "source" / "scripts" / "convert_anima_to_diffusers.py"
    converter_result = subprocess.run(
        [sys.executable, str(converter), "--help"],
        cwd=runtime,
        capture_output=True,
        text=True,
        check=True,
    )
    if "--transformer_ckpt_path" not in converter_result.stdout:
        raise RuntimeError("The Anima converter did not expose its expected CLI.")
    if "CPUExecutionProvider" not in onnxruntime.get_available_providers():
        raise RuntimeError("Background removal CPU execution provider is missing.")

    gpu: dict[str, object] = {"checked": False}
    if not args.skip_gpu:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Check the NVIDIA GPU driver.")
        with torch.inference_mode():
            matrix = torch.ones((32, 32), device="cuda", dtype=torch.bfloat16)
            product = matrix @ matrix
            torch.cuda.synchronize()
            if not bool(torch.all(product == 32).item()):
                raise RuntimeError("CUDA bfloat16 matrix multiplication returned an invalid result.")
        gpu = {
            "checked": True,
            "name": torch.cuda.get_device_name(0),
            "capability": list(torch.cuda.get_device_capability(0)),
            "compiled_architectures": torch.cuda.get_arch_list(),
            "bfloat16_matrix_multiply": "passed",
        }
        del matrix, product
        torch.cuda.empty_cache()

    report = {
        "status": "passed",
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "versions": {
            package: importlib.metadata.version(package)
            for package in ("torch", "torchvision", "diffusers", "transformers", "accelerate", "rembg", "onnxruntime")
        },
        "diffusers_commit": source_commit,
        "cuda_runtime": torch.version.cuda,
        "gpu": gpu,
        "anima_api": [AnimaModularPipeline.__name__, AnimaAutoBlocks.__name__, AnimaTextConditioner.__name__],
        "converter_cli": "passed",
        "background_removal_providers": onnxruntime.get_available_providers(),
        "model_weights_loaded": False,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
