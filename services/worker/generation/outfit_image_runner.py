"""Qwen child for an alternate look of one portrait: a transparent, single-character edit."""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from scripts.image_edit import backend, engine
from scripts.image_edit.common import write_json

from . import event_cg_runner
from .event_cg_runner import ModelIdentityError, QwenRuntime, peak_ram_bytes


class OutfitRuntime(QwenRuntime):
    def generate(self, request, progress):
        output = Path(request["output_dir"])
        validated = engine.validate_request(request["generation"])
        if validated["mode"] != "portrait" or not validated["transparent"]:
            raise ModelIdentityError("An outfit image is a transparent edit of one portrait.")
        images, records = engine.normalize_references(validated["references"], output)
        if validated["references"][0]["sha256"] != records[0]["original_sha256"]:
            raise ModelIdentityError("Reference changed after the worker validated its hash.")
        reused = self.pipe is not None
        if not reused:
            try:
                self.model = engine.verify_model(Path(validated["model_path"]), progress=progress)
            except (OSError, ValueError, TypeError, KeyError) as exc:
                raise ModelIdentityError(str(exc)) from exc
            if self.model["revision"] != request["expected_revision"]:
                raise ModelIdentityError("Qwen model does not match the frozen revision.")
            import torch

            self.torch = torch
            if not torch.cuda.is_available():
                raise RuntimeError("Qwen Image 2.1 requires a CUDA GPU.")
            progress({"phase": "loading"})
            self.pipe = backend.load_pipeline(validated, torch)
            self.loaded_at = time.monotonic()
        started = time.perf_counter()

        def callback(_pipe, step, _timestep, values):
            progress({"phase": "generating", "step": step + 1, "steps": validated["steps"]})
            return values

        generated = backend.render_image(self.pipe, self.torch, validated, images, validated["seed"], callback)
        if generated.size != (validated["width"], validated["height"]) or "A" not in generated.getbands():
            raise ValueError("The outfit image has no transparency or the wrong size.")
        histogram = generated.getchannel("A").histogram()
        pixels = generated.width * generated.height
        if sum(histogram[:16]) / pixels < 0.05 or sum(histogram[240:]) / pixels < 0.05:
            raise ValueError("The outfit image needs both a transparent background and a visible character.")
        # The session protocol publishes two names; a portrait has a single rendition.
        for name in ("original.png", "image.png"):
            engine.atomic_image(generated, output / name)
        report = {"ok": True, "model_loaded_at_monotonic": self.loaded_at,
                  "provenance": {"model": self.model, "references": records, "model_reused": reused,
                                 "library_versions": engine._versions(),
                                 "inference_seconds": time.perf_counter() - started,
                                 "process_ram_peak_bytes": peak_ram_bytes(), "seed": validated["seed"],
                                 "width": generated.width, "height": generated.height,
                                 "image_sha256": engine.digest(output / "image.png")}}
        write_json(output / "result.json", report)
        return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--serve", action="store_true")
    if not parser.parse_args().serve:
        parser.error("The Qwen child requires --serve and a parent GPU owner.")
    event_cg_runner.QwenRuntime = OutfitRuntime
    event_cg_runner.serve()


if __name__ == "__main__":
    main()
