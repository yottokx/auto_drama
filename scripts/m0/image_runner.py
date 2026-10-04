"""Offline resident Anima runtime. Individual requests retain no prompt or seed state."""
from __future__ import annotations

import contextlib
import gc
import importlib.metadata
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

from scripts.m0.image_smoke import COMPONENTS, MANIFEST, ROOT, RUNTIME, digest, resolve_prompts


class ImageRuntimeSession:
    def __init__(self):
        self.pipe = self.model_dir = self.torch = self.removal = None
        self.loaded_at = None

    def close(self):
        try:
            if self.pipe is not None:
                self.pipe.unload_components(list(self.pipe.components))
        finally:
            self.pipe = self.model_dir = self.removal = None
            self.loaded_at = None
            if self.torch is not None:
                gc.collect()
                self.torch.cuda.empty_cache()
                self.torch.cuda.synchronize()

    def _initialize(self, model_dir):
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["HF_HOME"] = str(RUNTIME / "cache/huggingface")
        import torch
        from diffusers import AnimaModularPipeline

        if not torch.cuda.is_available():
            raise RuntimeError("Image generation requires an available CUDA GPU.")
        self.torch = torch
        print("Loading locally pinned Anima components...", flush=True)
        self.pipe = AnimaModularPipeline.from_pretrained(
            str(model_dir), local_files_only=True, trust_remote_code=False)
        self.pipe.load_components(pretrained_model_name_or_path=str(model_dir), dtype=torch.bfloat16,
                                  local_files_only=True, trust_remote_code=False)
        missing = [name for name in COMPONENTS if getattr(self.pipe, name, None) is None]
        if missing:
            raise RuntimeError(f"Failed to load required Anima components: {missing}")
        self.pipe.to("cuda")
        self.model_dir = model_dir

    def _remove_background(self, generated, models):
        entry = models["isnet-anime"]
        directory = ROOT / entry["local_dir"]
        model = directory / "isnet-anime.onnx"
        if (not model.is_file() or model.stat().st_size != entry["files"][0]["size"]
                or digest(model, "md5") != entry["files"][0]["md5"]):
            raise ValueError("isnet-anime weights do not match the pinned manifest.")
        os.environ["U2NET_HOME"] = str(directory)
        os.environ["REMBG_HOME"] = str(directory)
        from rembg import new_session, remove

        print("Removing background with local isnet-anime on CPU...", flush=True)
        if self.removal is None:
            self.removal = new_session("isnet-anime", providers=["CPUExecutionProvider"])
        character = remove(generated, session=self.removal).convert("RGBA")
        alpha = character.getchannel("A")
        extrema, bbox = alpha.getextrema(), alpha.getbbox()
        if character.size != generated.size or bbox is None or extrema[0] >= 16 or extrema[1] <= 239:
            raise RuntimeError(f"Invalid background removal output: size={character.size}, alpha={extrema}, bbox={bbox}")
        return character, {"alpha_range": extrema, "foreground_bbox": bbox,
                           "anchor_bottom_center": [(bbox[0] + bbox[2]) / 2, bbox[3]],
                           "background_model_sha256": digest(model)}

    def generate(self, request):
        mode = request["mode"]
        prompt, negative = resolve_prompts(mode, request["prompt"], request["negative_prompt"])
        width, height, steps, seed = (request[name] for name in ("width", "height", "steps", "seed"))
        guidance = request["guidance_scale"]
        if (any(type(value) is not int for value in (width, height, steps, seed))
                or min(width, height, steps) <= 0 or width % 16 or height % 16
                or not 0 <= seed < 2**63 or type(guidance) not in (int, float)
                or not math.isfinite(guidance)):
            raise ValueError("Invalid image dimensions, steps, seed or guidance.")
        output, model_dir = Path(request["output_dir"]), Path(request["model_dir"]).resolve()
        if not output.is_absolute() or not output.is_dir() or (output / "result.json").exists():
            raise ValueError("Image output must be a fresh existing absolute directory.")
        if not (model_dir / "modular_model_index.json").is_file():
            raise FileNotFoundError("Download or convert Anima first.")
        if any(not (model_dir / name).is_dir() for name in COMPONENTS):
            raise FileNotFoundError("A required Anima component is missing.")
        manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
        models = {entry["id"]: entry for entry in manifest["models"]}
        reused = self.pipe is not None
        if reused and model_dir != self.model_dir:
            raise ValueError("Image model changed; restart the resident runtime.")
        started, initialization = time.monotonic(), 0.0
        if not reused:
            self._initialize(model_dir)
            self.loaded_at = time.monotonic()
            initialization = time.monotonic() - started
        from diffusers import ClassifierFreeGuidance

        baseline = self.torch.cuda.memory_allocated()
        self.torch.cuda.reset_peak_memory_stats()
        # The guider and random generator belong to this request, not the session.
        self.pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=guidance))
        print(f"Generating {width}x{height}, {steps} steps, seed={seed}...", flush=True)
        with self.torch.inference_mode():
            generated = self.pipe(
                prompt=prompt, negative_prompt=negative, width=width, height=height,
                num_inference_steps=steps, generator=self.torch.Generator("cpu").manual_seed(seed),
                output_type="pil", output="images")[0]
        generated.save(output / "image.png")
        artifacts, versions, removal = ["image.png"], ["diffusers", "torch", "transformers"], {}
        if mode == "character":
            character, removal = self._remove_background(generated, models)
            character.save(output / "character.png")
            artifacts.append("character.png")
            versions.append("rembg")
        report = {
            "schema_version": 1, "status": "generated_requires_visual_review", "mode": mode,
            "model_dir": str(model_dir), "official_diffusers_revision": models["anima-base-v1-diffusers"]["revision"],
            "runtime_commit": manifest["runtime_commit"],
            "conversion_record": str(model_dir / "conversion.json") if (model_dir / "conversion.json").exists() else None,
            "versions": {name: importlib.metadata.version(name) for name in versions},
            "prompt": prompt, "negative_prompt": negative, "seed": seed,
            "width": width, "height": height, "steps": steps, "guidance_scale": guidance,
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "initialization_seconds": initialization, "model_reused": reused,
            "model_loaded_at_monotonic": self.loaded_at,
            "gpu": self.torch.cuda.get_device_name(), "gpu_allocated_before": baseline,
            "gpu_peak_allocated": self.torch.cuda.max_memory_allocated(),
            "gpu_allocated_after_generation": self.torch.cuda.memory_allocated(),
            "gpu_process_exit_required_for_context_release": True,
            **removal, "artifacts": {name: digest(output / name) for name in artifacts},
            "review_required": "Inspect generated imagery, silhouette and composition.",
        }
        (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


@contextlib.contextmanager
def request_log(output):
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
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


def serve(input_stream, output_stream):
    session = ImageRuntimeSession()
    try:
        for line in input_stream:
            try:
                request = json.loads(line)
                output = Path(request["output_dir"])
                if not output.is_absolute() or not output.is_dir():
                    raise ValueError("Image output_dir must be an existing absolute directory.")
                with request_log(output):
                    session.generate(request)
            except Exception as error:  # noqa: BLE001 - discard CUDA state after any failed request
                print(json.dumps({"ok": False, "error": str(error)}, ensure_ascii=False), file=output_stream, flush=True)
                return 1
            print(json.dumps({"ok": True}), file=output_stream, flush=True)
        return 0
    finally:
        session.close()
