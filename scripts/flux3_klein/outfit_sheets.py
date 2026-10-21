"""Experiment: every outfit case dressed from an outfit picture, the way a shared material is.

The four cases first drawn from words get a sheet drawn from those words by the portrait model,
as services/worker/generation/material_image.py draws one; 034b89ca already had one. Then each
case is dressed from its sheet with the material prompt of outfit_image.py, by Qwen standard and
Turbo here, and by Klein with ``turbo_cases.py --evaluation <out>``. The folder is laid out like
tmp/expression-exp/turbo-eval so that script reads it unchanged.

    .venv/Scripts/python.exe -X utf8 scripts/flux3_klein/outfit_sheets.py sheets
    services/worker/runtimes/qwen_image_edit/.venv/Scripts/python.exe -X utf8 scripts/flux3_klein/outfit_sheets.py qwen
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

ROOT = Path(__file__).resolve().parents[2]
EVALUATION = ROOT / "tmp/expression-exp/turbo-eval"
JOBS = ROOT / "services/worker/cache/m2/jobs"
OUT = ROOT / "outputs/image-edit-tests/outfit-sheets-20261016"
CASES = ("601c03da", "8bf180fb", "67dd0b22", "bcd0dc85", "034b89ca")
WITH_SHEET = "034b89ca"


def outfit_words(case: str) -> str:
    (job,) = JOBS.glob(f"*/{case}*")
    design = json.loads((job / "outfit-design.json").read_text(encoding="utf-8"))
    if design.get("outfit_en"):
        return design["outfit_en"]
    # An edit names only what changes ("Change the outfit to ..."): the garments are what follows.
    return design["instruction_en"].removeprefix("Change the outfit to ").rstrip(".")


def sheets() -> None:
    from services.worker.generation import pipeline
    from services.worker.generation.material_image import HEIGHT, SHEET, TEXT_WIDTH
    from services.worker.generation.outfit_image import build_material_prompt

    OUT.mkdir(parents=True, exist_ok=False)
    settings = pipeline.load_config()["image"]
    source = {job["id"]: job for job in json.loads((EVALUATION / "jobs.json").read_text(encoding="utf-8"))}
    jobs = []
    for case in CASES:
        job = source[case]
        shutil.copy(EVALUATION / f"{case}-ref1.png", OUT / f"{case}-ref1.png")
        words = outfit_words(case)
        if case == WITH_SHEET:
            shutil.copy(EVALUATION / f"{case}-ref2.png", OUT / f"{case}-ref2.png")
            sheet = {"done": "already made"}
        else:
            # As material_image._draw_from_words, with the case's own seed.
            prompt = ", ".join(filter(None, (
                settings.get("positive_prompt_prefix", "").strip().rstrip(", "), "no humans, clothes only",
                f"{words} {SHEET['outfit']} Japanese anime color illustration, clear details, bright frontal "
                "lighting, solid white background, no text or logos.")))
            negative = (settings.get("negative_prompt", "") + ", 1girl, 1boy, person, face, skin, hands, "
                        "multiple views, variations, reference sheet, turnaround, two outfits, multiple outfits, "
                        "side by side, outfit comparison, separate garments")
            work = OUT / "sheets" / case
            subprocess.run([str(ROOT / settings["python"]), str(ROOT / "scripts/m0/image_smoke.py"),
                            "--mode", "background", "--prompt", prompt, "--negative-prompt", negative,
                            "--model-dir", str(ROOT / settings["model_dir"]), "--output-dir", str(work),
                            "--width", str(TEXT_WIDTH), "--height", str(HEIGHT), "--steps", str(settings["steps"]),
                            "--guidance-scale", str(settings["guidance_scale"]), "--seed", str(job["seed"])],
                           cwd=ROOT, check=True)
            shutil.copy(work / "image.png", OUT / f"{case}-ref2.png")
            sheet = {"prompt": prompt}
        prompt = (job["prompt"] if case == WITH_SHEET
                  else build_material_prompt({"category": "outfit", "outfit_en": words}))
        jobs.append({**{key: job[key] for key in ("id", "group", "width", "height", "seed", "transparent")},
                     "prompt": prompt, "references": [f"{case}-ref1.png", f"{case}-ref2.png"],
                     "names": ["base portrait", "shared material"], "baseline": f"{case}-base40.png",
                     "outfit_en": words, "sheet": sheet})
        print(f"sheet {case}: {words}", flush=True)
    (OUT / "jobs.json").write_text(json.dumps(jobs, ensure_ascii=False, indent=1), encoding="utf-8")


def qwen() -> None:
    """Each case dressed by the standard model, then all by Turbo: one model stays loaded."""
    import torch

    from scripts.image_edit import backend, engine
    from scripts.image_edit.common import MODELS

    profile = json.loads((ROOT / "tmp/expression-exp/profiles.json").read_text(encoding="utf-8"))["vram32"]
    jobs = json.loads((OUT / "jobs.json").read_text(encoding="utf-8"))
    log = OUT / "qwen.jsonl"
    for model in ("standard", "turbo"):
        model_path = Path(MODELS[model]["directory"])
        sigmas = backend.sample_sigmas(model_path)
        pipe = None
        for job in jobs:
            target = (OUT / job["baseline"] if model == "standard" else OUT / "out" / job["id"] / "image.png")
            if job["id"] == WITH_SHEET:
                # Already made from this sheet and prompt.
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(EVALUATION / (job["baseline"] if model == "standard" else f"out/{job['id']}/image.png"),
                            target)
                continue
            work = OUT / "qwen" / model / job["id"]
            work.mkdir(parents=True, exist_ok=True)
            references = [{"path": str(OUT / path), "name": name,
                           "sha256": hashlib.sha256((OUT / path).read_bytes()).hexdigest()}
                          for path, name in zip(job["references"], job["names"])]
            request = engine.validate_request({
                "schema_version": 1, "mode": "scene", "prompt": job["prompt"], "transparent": True,
                "model_path": str(model_path), "width": job["width"], "height": job["height"], "seed": job["seed"],
                "references": references, "steps": len(sigmas) if sigmas else profile["steps"],
                **{key: profile[key] for key in ("dtype", "cpu_offload", "use_kv_cache", "reference_resolution",
                                                  "text_encoder_offload", "transformer_storage", "vae_tiling")}})
            images, _records = engine.normalize_references(request["references"], work)
            if pipe is None:
                pipe = backend.load_pipeline(request, torch)
            torch.cuda.reset_peak_memory_stats()
            began = time.perf_counter()
            image = backend.render_image(pipe, torch, request, images, job["seed"], None)
            target.parent.mkdir(parents=True, exist_ok=True)
            image.save(target)
            row = {"id": job["id"], "model": model, "inference": round(time.perf_counter() - began, 1),
                   "mode": image.mode, "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2)}
            with log.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        del pipe
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser(description="Outfit cases dressed from outfit pictures")
    parser.add_argument("step", choices=("sheets", "qwen"))
    {"sheets": sheets, "qwen": qwen}[parser.parse_args().step]()
    return 0


if __name__ == "__main__":
    sys.exit(main())
