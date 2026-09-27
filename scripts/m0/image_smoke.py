"""Offline Anima generation and anime background-removal smoke test.

Run with the uv-managed Diffusers runtime's Python. Downloads are a separate,
visible preparation step. This program never trusts model repository Python.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "services/worker/runtimes/diffusers"
MANIFEST = ROOT / "config/m0-models-image.json"
COMPONENTS = ("scheduler", "tokenizer", "t5_tokenizer", "text_encoder", "text_conditioner", "transformer", "vae")
CHARACTER_PROMPT = (
    "masterpiece, best quality, score_7, safe, anime illustration, detailed clean line art, full color, "
    "1girl, solo, adult woman, full body, standing, natural pale skin, detailed expressive face, "
    "gentle smile, short brown hair, green eyes, blue travel coat with buttons and seams, cream shirt, "
    "brown leather boots, visible hands. A fully rendered Japanese anime character design with soft "
    "shading, visible skin on her face and hands, and detailed folds in the coat. Front lighting "
    "illuminates her entire face and clothing. Her entire body fits in the frame with a white background."
)
BACKGROUND_PROMPT = (
    "masterpiece, best quality, score_7, safe, scenery, indoors, lighthouse interior, night, "
    "no humans, no people, empty room, visual novel background. The quiet upper room of an old "
    "coastal lighthouse is lit by a warm oil lamp beside a wooden desk with nautical charts. "
    "A large window shows the dark sea and moonlit waves, with stone walls and a spiral staircase."
)
DEFAULT_NEGATIVE_PROMPT = (
    "worst quality, low quality, score_1, score_2, score_3, artist name, blurry, "
    "jpeg artifacts, chromatic aberration, cropped, text"
)


def resolve_prompts(
    mode: str, prompt: str | None = None, negative_prompt: str | None = None
) -> tuple[str, str]:
    """Resolve defaults before loading models; explicit values are complete overrides."""
    if mode not in {"character", "background"}:
        raise ValueError(f"Unsupported image mode: {mode}")
    if prompt is not None and not prompt.strip():
        raise ValueError("--prompt must contain text.")
    effective_prompt = prompt if prompt is not None else (
        CHARACTER_PROMPT if mode == "character" else BACKGROUND_PROMPT
    )
    if negative_prompt is None:
        negative_prompt = DEFAULT_NEGATIVE_PROMPT
        if mode == "background":
            negative_prompt += ", person, human, girl, boy, face, silhouette"
    return effective_prompt, negative_prompt


def digest(path: Path, algorithm: str = "sha256") -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("character", "background"), default="character")
    parser.add_argument("--prompt", help="Override the mode's default generation prompt.")
    parser.add_argument(
        "--negative-prompt",
        help="Replace the complete negative prompt; an empty string disables it.",
    )
    parser.add_argument("--model-dir", type=Path, default=RUNTIME / "models/anima-base-v1-diffusers")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "private/m0/image")
    parser.add_argument("--width", type=int, help="Default: character 768; background 1280.")
    parser.add_argument("--height", type=int, help="Default: character 1024; background 720.")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=20260922)
    parser.add_argument("--check-only", action="store_true", help="Check local files without importing inference libraries.")
    args = parser.parse_args()
    args.width = args.width if args.width is not None else (768 if args.mode == "character" else 1280)
    args.height = args.height if args.height is not None else (1024 if args.mode == "character" else 720)
    try:
        prompt, negative = resolve_prompts(args.mode, args.prompt, args.negative_prompt)
    except ValueError as exc:
        parser.error(str(exc))
    if min(args.width, args.height, args.steps) <= 0 or args.width % 16 or args.height % 16:
        parser.error("Dimensions must be positive multiples of 16; steps must be positive.")
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8-sig"))
    models = {entry["id"]: entry for entry in manifest["models"]}
    args.model_dir = args.model_dir.resolve()
    index = args.model_dir / "modular_model_index.json"
    if not index.is_file():
        raise FileNotFoundError(f"Download or convert Anima first: {index}")
    for name in COMPONENTS:
        if not (args.model_dir / name).is_dir():
            raise FileNotFoundError(f"Missing model component: {name}")
    if args.mode == "character":
        rembg_entry = models["isnet-anime"]
        rembg_dir = ROOT / rembg_entry["local_dir"]
        rembg_model = rembg_dir / "isnet-anime.onnx"
        if not rembg_model.is_file() or rembg_model.stat().st_size != rembg_entry["files"][0]["size"]:
            raise FileNotFoundError(f"Download complete isnet-anime weights first: {rembg_model}")
    if args.check_only:
        print(json.dumps({"status": "local_files_present", "inference_run": False,
                          "mode": args.mode, "model_dir": str(args.model_dir),
                          "prompt": prompt, "negative_prompt": negative}))
        return

    # Set before imports: no implicit downloads or home-directory cache sharing.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(RUNTIME / "cache/huggingface")
    if args.mode == "character":
        os.environ["U2NET_HOME"] = str(rembg_dir)
        os.environ["REMBG_HOME"] = str(rembg_dir)
        if digest(rembg_model, "md5") != rembg_entry["files"][0]["md5"]:
            raise ValueError("isnet-anime checksum does not match official rembg 2.0.85.")

    import torch
    from diffusers import AnimaModularPipeline, ClassifierFreeGuidance

    if not torch.cuda.is_available():
        raise RuntimeError("M0 image smoke test requires an available CUDA GPU.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "result.json").exists():
        raise FileExistsError("Choose a new --output-dir to preserve the earlier smoke result.")
    started = time.monotonic()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    print("Loading locally pinned Anima components...", flush=True)
    pipe = AnimaModularPipeline.from_pretrained(str(args.model_dir), local_files_only=True, trust_remote_code=False)
    try:
        # The official modular index embeds remote repo IDs; explicitly replace
        # every component's source with our downloaded local directory.
        pipe.load_components(pretrained_model_name_or_path=str(args.model_dir), dtype=torch.bfloat16,
                             local_files_only=True, trust_remote_code=False)
        missing = [name for name in COMPONENTS if getattr(pipe, name, None) is None]
        if missing:
            raise RuntimeError(f"Failed to load required Anima components: {missing}")
        pipe.update_components(guider=ClassifierFreeGuidance(guidance_scale=args.guidance_scale))
        pipe.to("cuda")
        print(f"Generating {args.width}x{args.height}, {args.steps} steps, seed={args.seed}...", flush=True)
        with torch.inference_mode():
            generated = pipe(prompt=prompt, negative_prompt=negative, width=args.width, height=args.height,
                             num_inference_steps=args.steps, generator=torch.Generator("cpu").manual_seed(args.seed),
                             output_type="pil", output="images")[0]
        generated.save(args.output_dir / "image.png")
        peak = torch.cuda.max_memory_allocated()
    finally:
        pipe.unload_components(list(pipe.components))
        del pipe
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    released = torch.cuda.memory_allocated()
    removal_report = {}
    artifacts = ["image.png"]
    versions = ["diffusers", "torch", "transformers"]
    if args.mode == "character":
        from rembg import new_session, remove

        print("Removing background with local isnet-anime on CPU...", flush=True)
        session = new_session("isnet-anime", providers=["CPUExecutionProvider"])
        character = remove(generated, session=session).convert("RGBA")
        character.save(args.output_dir / "character.png")
        alpha = character.getchannel("A")
        extrema = alpha.getextrema()
        bbox = alpha.getbbox()
        if character.size != generated.size or bbox is None or extrema[0] >= 16 or extrema[1] <= 239:
            raise RuntimeError(f"Invalid background removal output: size={character.size}, alpha={extrema}, bbox={bbox}")
        removal_report = {"alpha_range": extrema, "foreground_bbox": bbox,
                          "anchor_bottom_center": [(bbox[0] + bbox[2]) / 2, bbox[3]],
                          "background_model_sha256": digest(rembg_model)}
        artifacts.append("character.png")
        versions.append("rembg")
    report = {
        "schema_version": 1, "status": "generated_requires_visual_review", "mode": args.mode,
        "model_dir": str(args.model_dir),
        "official_diffusers_revision": models["anima-base-v1-diffusers"]["revision"],
        "conversion_record": str(args.model_dir / "conversion.json") if (args.model_dir / "conversion.json").exists() else None,
        "runtime_commit": manifest["runtime_commit"],
        "versions": {name: importlib.metadata.version(name) for name in versions},
        "prompt": prompt, "negative_prompt": negative, "seed": args.seed,
        "width": args.width, "height": args.height, "steps": args.steps, "guidance_scale": args.guidance_scale,
        "elapsed_seconds": round(time.monotonic() - started, 3), "gpu": torch.cuda.get_device_name(),
        "gpu_allocated_before": baseline, "gpu_peak_allocated": peak, "gpu_allocated_after_unload": released,
        "gpu_process_exit_required_for_context_release": True,
        **removal_report,
        "artifacts": {name: digest(args.output_dir / name) for name in artifacts},
        "review_required": (
            "Inspect face, hair, small objects, silhouette and transparency; quality is not automatically certified."
            if args.mode == "character" else
            "Inspect lighthouse setting, composition and absence of people; quality is not automatically certified."
        ),
    }
    (args.output_dir / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "result": str(args.output_dir / "result.json")}), flush=True)


if __name__ == "__main__":
    main()
