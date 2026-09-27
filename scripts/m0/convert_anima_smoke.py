"""Convert pinned official Anima single files using the pinned Diffusers converter.

No downloads or GPU inference. Conversion, checksums, and CPU reload can take
several minutes; run with a visible log and sufficient RAM/disk space.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "services/worker/runtimes/diffusers"


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=RUNTIME / "models/anima-base-v1-converted")
    parser.add_argument("--check-only", action="store_true", help="Check local paths/sizes; do not load weights.")
    args = parser.parse_args()
    manifest = json.loads((ROOT / "config/m0-models-image.json").read_text(encoding="utf-8-sig"))
    models = {entry["id"]: entry for entry in manifest["models"]}
    source = models["anima-base-v1-single"]
    source_dir = ROOT / source["local_dir"]
    tokenizer_dir = ROOT / models["anima-base-v1-diffusers"]["local_dir"]
    converter = RUNTIME / "source/scripts/convert_anima_to_diffusers.py"
    actual_commit = subprocess.check_output(["git", "-C", str(RUNTIME / "source"), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != manifest["runtime_commit"]:
        raise RuntimeError(f"Converter source is not the pinned commit: {actual_commit}")
    if not converter.is_file():
        raise FileNotFoundError(converter)
    for entry in source["files"]:
        path = source_dir / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size"]:
            raise FileNotFoundError(f"Missing or incomplete conversion input: {path}")
    for tokenizer in ("tokenizer", "t5_tokenizer"):
        if not (tokenizer_dir / tokenizer / "tokenizer.json").is_file():
            raise FileNotFoundError(f"Download official tokenizer first: {tokenizer_dir / tokenizer}")
    if args.check_only:
        print(json.dumps({"status": "conversion_inputs_present", "conversion_run": False}))
        return
    args.output_dir = args.output_dir.resolve()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Choose an empty --output-dir; an earlier conversion will not be overwritten.")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_HOME"] = str(RUNTIME / "cache/huggingface")
    started = time.monotonic()
    input_hashes = {}
    for entry in source["files"]:
        print(f"Verifying {entry['path']}...", flush=True)
        value = sha256(source_dir / entry["path"])
        if entry.get("sha256") and value != entry["sha256"]:
            raise ValueError(f"Input SHA256 mismatch: {entry['path']}")
        input_hashes[entry["path"]] = value
    command = [sys.executable, "-I", str(converter),
               "--transformer_ckpt_path", str(source_dir / "split_files/diffusion_models/anima-base-v1.0.safetensors"),
               "--text_encoder_ckpt_path", str(source_dir / "split_files/text_encoders/qwen_3_06b_base.safetensors"),
               "--vae_ckpt_path", str(source_dir / "split_files/vae/qwen_image_vae.safetensors"),
               "--qwen_tokenizer_path", str(tokenizer_dir / "tokenizer"),
               "--t5_tokenizer_path", str(tokenizer_dir / "t5_tokenizer"),
               "--output_path", str(args.output_dir), "--save_pipeline", "--dtype", "bf16"]
    # The upstream converter imports its adjacent helper. Isolated Python does
    # not put the script directory on sys.path, so invoke it through a small
    # bootstrap that adds only the pinned converter directory.
    bootstrap = "import runpy,sys; from pathlib import Path; p=Path(sys.argv[1]); sys.path.insert(0,str(p.parent)); sys.argv=sys.argv[1:]; runpy.run_path(str(p),run_name='__main__')"
    command = command[:2] + ["-c", bootstrap] + command[2:]
    print("Running official converter (CPU); its key/shape checks must pass...", flush=True)
    subprocess.run(command, check=True, cwd=ROOT)
    import torch
    from diffusers import AnimaModularPipeline

    print("Reloading converted components on CPU...", flush=True)
    pipe = AnimaModularPipeline.from_pretrained(str(args.output_dir), local_files_only=True, trust_remote_code=False)
    names = ("scheduler", "tokenizer", "t5_tokenizer", "text_encoder", "text_conditioner", "transformer", "vae")
    try:
        pipe.load_components(pretrained_model_name_or_path=str(args.output_dir), dtype=torch.bfloat16,
                             local_files_only=True, trust_remote_code=False)
        missing = [name for name in names if getattr(pipe, name, None) is None]
        if missing:
            raise RuntimeError(f"Converted components failed to reload: {missing}")
    finally:
        pipe.unload_components(list(pipe.components))
        del pipe
        gc.collect()
    print("Hashing converted artifacts...", flush=True)
    outputs = {path.relative_to(args.output_dir).as_posix(): sha256(path)
               for path in sorted(args.output_dir.rglob("*")) if path.is_file()}
    report = {"schema_version": 1, "status": "converted_and_cpu_reloaded_image_smoke_pending",
              "source_repo": source["repo_id"], "source_revision": source["revision"],
              "converter_commit": actual_commit, "converter_sha256": sha256(converter),
              "input_sha256": input_hashes, "output_sha256": outputs,
              "elapsed_seconds": round(time.monotonic() - started, 3),
              "next_step": "Run scripts/m0/image_smoke.py --model-dir <this directory> --output-dir <new directory>."}
    (args.output_dir / "conversion.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "result": str(args.output_dir / "conversion.json")}), flush=True)


if __name__ == "__main__":
    main()
