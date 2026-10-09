"""Qwen child protocol. The parent exclusively owns the GPU lock."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
import traceback
from contextlib import redirect_stdout
from pathlib import Path

from scripts.image_edit import backend, engine
from scripts.image_edit.common import write_json


class ModelIdentityError(ValueError):
    """A frozen model/reference identity must never become an omission."""


def peak_ram_bytes():
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD),
                *[(name, ctypes.c_size_t) for name in ("PeakWorkingSetSize", "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage", "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]]

        info = Counters()
        info.cb = ctypes.sizeof(info)
        kernel = ctypes.windll.kernel32
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        read_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        read_memory.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
        read_memory.restype = wintypes.BOOL
        if read_memory(kernel.GetCurrentProcess(), ctypes.byref(info), info.cb):
            return info.PeakWorkingSetSize
        return None
    import resource

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (1 if sys.platform == "darwin" else 1024)


class QwenRuntime:
    def __init__(self):
        self.pipe = self.torch = self.model = self.loaded_at = None

    def close(self):
        self.pipe = None
        gc.collect()
        if self.torch is not None:
            self.torch.cuda.empty_cache()
            self.torch.cuda.synchronize()

    def generate(self, request, progress):
        output = Path(request["output_dir"])
        validated = engine.validate_request(request["generation"])
        if request.get("expected_revision") is None:
            raise ModelIdentityError("Missing frozen model revision.")
        images, records = engine.normalize_references(validated["references"], output)
        # Even the parent-fetched copies are rechecked at the child boundary.
        for source, record in zip(validated["references"], records, strict=True):
            if source["sha256"] != record["original_sha256"]:
                raise ModelIdentityError("Reference changed after the worker validated its hash.")
        reused = self.pipe is not None
        verify_seconds = load_seconds = 0
        if not reused:
            started = time.perf_counter()
            try:
                self.model = engine.verify_model(Path(validated["model_path"]), progress=progress)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ModelIdentityError(str(exc)) from exc
            if self.model["revision"] != request["expected_revision"]:
                raise ModelIdentityError("Qwen model does not match the frozen revision.")
            verify_seconds = time.perf_counter() - started
            import torch

            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError("Qwen Image 2.1 requires a CUDA GPU.")
            progress({"phase": "loading"})
            started = time.perf_counter()
            self.pipe = backend.load_pipeline(validated, torch)
            load_seconds = time.perf_counter() - started
            self.loaded_at = time.monotonic()
        self.torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        def callback(_pipe, step, _timestep, values):
            progress({"phase": "generating", "step": step + 1, "steps": validated["steps"]})
            return values

        generated = backend.render_image(self.pipe, self.torch, validated, images,
            validated["seed"], callback)
        if generated.size != (validated["width"], validated["height"]):
            raise ValueError("CG output dimensions do not match the requested dimensions.")
        if "A" in generated.getbands():
            histogram = generated.getchannel("A").histogram()
            if sum(histogram[:245]) / (generated.width * generated.height) > 0.001:
                raise ValueError("CG output contains substantial transparency.")
        engine.atomic_image(generated, output / "original.png")
        engine.atomic_image(generated.convert("RGB"), output / "image.png")
        report = {"ok": True, "model_loaded_at_monotonic": self.loaded_at,
            "provenance": {"model": self.model, "references": records,
                "model_reused": reused, "library_versions": engine._versions(),
                "verification_seconds": verify_seconds, "initialization_seconds": load_seconds,
                "inference_seconds": time.perf_counter() - started,
                "gpu_peak_allocated": self.torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved": self.torch.cuda.max_memory_reserved(),
                "process_ram_peak_bytes": peak_ram_bytes(), "seed": validated["seed"],
                "width": generated.width, "height": generated.height,
                "original_sha256": engine.digest(output / "original.png"),
                "image_sha256": engine.digest(output / "image.png"),
                "quantization": "fp8" if validated["transformer_storage"] == "fp8" else "none",
                "reference_resolution": validated["reference_resolution"],
                "text_encoder_offload": validated["text_encoder_offload"],
                "vae_tiling": validated["vae_tiling"], "dtype": validated["dtype"]}}
        write_json(output / "result.json", report)
        return report


def serve():
    protocol = sys.stdout
    runtime = QwenRuntime()
    key = None

    def send(value):
        print(json.dumps(value, ensure_ascii=False), file=protocol, flush=True)

    try:
        for line in sys.stdin:
            request = None
            try:
                request = json.loads(line)
                requested_key = request["runtime_identity"]
                with redirect_stdout(sys.stderr):
                    if key != requested_key:
                        runtime.close()
                        runtime = QwenRuntime()
                        key = requested_key
                    runtime.generate(request, lambda value: send({"progress": value}))
                send({"ok": True})
            except Exception as exc:  # noqa: BLE001 - boundary between owned child and worker
                traceback.print_exc(file=sys.stderr)
                failure = {"ok": False, "error": str(exc),
                    "error_kind": "identity" if isinstance(exc, ModelIdentityError) else "generation"}
                if isinstance(request, dict) and isinstance(request.get("output_dir"), str):
                    write_json(Path(request["output_dir"]) / "result.json", failure)
                send(failure)
                # Errors can retain weights through traceback frames; exit before
                # the parent releases its lease, never reuse a failed pipeline.
                break
    finally:
        runtime.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    if not parser.parse_args().serve:
        parser.error("The Qwen child requires --serve and a parent GPU owner.")
    serve()


if __name__ == "__main__":
    main()
