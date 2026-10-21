"""Fetch the pinned official FLUX 3 Image Klein (distilled, BF16) files or its prompt upsampler and verify them.

Experiment only: nothing in the main pipeline reads this folder. Only the Diffusers layout is
fetched (modular_model_index.json, transformer/, vae/, text_encoder/); BFL's single-file
checkpoints, the FP8/NVFP4 variants and the gallery images are left on the Hub.
The repository is gated: accept the FLUX Kommunity License on its model page with the account
of the existing Hub login first. No token is written to the manifest or status output.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import sys
import uuid
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.image_edit.common import ROOT, hub_environment, read_json, write_json

RUNTIME = ROOT / "services/worker/runtimes/flux3_klein"
MANIFEST = "model-manifest.json"
COMPONENTS = ("transformer", "vae", "text_encoder")
# Revisions are main on 2026-10-16. The upsampler is the 9B LLM the model card says to run
# before the image model; it is public (Apache-2.0) and is fetched whole except .gitattributes.
MODELS = {
    "klein": {
        "model_id": "black-forest-labs/flux-3-image-klein",
        "revision": "337337621b8526e120f2275d250242f4a6bedd75",
        "directory": RUNTIME / "models/flux-3-image-klein",
        "wanted": lambda name: (name in {"modular_model_index.json", "LICENSE.md", "NOTICE.md", "README.md"}
                                or name.split("/")[0] in COMPONENTS),
        "required": ("modular_model_index.json", *(f"{name}/config.json" for name in COMPONENTS)),
    },
    "pup": {
        "model_id": "black-forest-labs/flux-3-image-klein-pup",
        "revision": "3f3edb52b2c55de0d49feab28598b68f7f8e4830",
        "directory": RUNTIME / "models/flux-3-image-klein-pup",
        "wanted": lambda name: name != ".gitattributes",
        "required": ("config.json", "model.safetensors", "system_prompts.json", "tokenizer.json"),
    },
}


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def verify(directory: Path, entries: list[dict]) -> list[dict]:
    """Check every file's size and, where the Hub publishes one, its SHA256."""
    verified = []
    for entry in entries:
        path = directory / entry["path"]
        if not path.is_file() or path.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"ファイルのサイズが一致しません: {entry['path']}")
        print(f"SHA256を検証しています: {entry['path']}", flush=True)
        digest = file_digest(path)
        if entry.get("sha256") and digest != entry["sha256"]:
            raise ValueError(f"公式のSHA256と一致しません: {entry['path']}")
        verified.append({"path": entry["path"], "size_bytes": entry["size_bytes"], "sha256": digest})
    return verified


def check_existing(output: Path, model: dict) -> dict:
    manifest = read_json(output / MANIFEST)
    if manifest.get("model_id") != model["model_id"] or manifest.get("revision") != model["revision"]:
        raise ValueError("出力先に別のモデル・リビジョンがあります。既存ファイルは変更しません。")
    verify(output, manifest["files"])
    return manifest


def prepare(output: Path, model: dict) -> dict:
    from huggingface_hub import HfApi, snapshot_download

    output = output.expanduser().resolve()
    if output.exists() and any(output.iterdir()):
        if not (output / MANIFEST).is_file():
            raise ValueError("出力先に検証記録のないファイルがあります。空のフォルダーを指定してください。")
        return {**check_existing(output, model), "reused": True}
    model_id, revision = model["model_id"], model["revision"]
    info = HfApi().model_info(model_id, revision=revision, files_metadata=True)
    entries = []
    for item in info.siblings:
        if not model["wanted"](item.rfilename):
            continue
        if type(item.size) is not int:
            raise ValueError(f"ファイルサイズを確定できません: {item.rfilename}")
        entry = {"path": item.rfilename, "size_bytes": item.size}
        if getattr(item.lfs, "sha256", None):
            entry["sha256"] = item.lfs.sha256
        entries.append(entry)
    entries.sort(key=lambda entry: entry["path"])
    names = {entry["path"] for entry in entries}
    if any(name not in names for name in model["required"]):
        raise ValueError("公式リポジトリの構成が想定と異なります。")
    total = sum(entry["size_bytes"] for entry in entries)
    print(f"{model_id}@{revision[:12]} を取得します: {len(entries)}ファイル, {total / 1e9:.2f} GB", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = output.parent / f".{output.name}.prepare-{uuid.uuid4().hex}"
    stage.mkdir()
    try:
        snapshot_download(model_id, revision=revision, local_dir=stage,
                          allow_patterns=sorted(names), max_workers=3)
        verified = verify(stage, entries)
        manifest = {"schema_version": 1, "model_id": model_id, "revision": revision,
                    "gated": info.gated, "snapshot_required_bytes": total, "files": verified}
        write_json(stage / MANIFEST, manifest)
        shutil.rmtree(stage / ".cache", ignore_errors=True)
        if output.exists():
            output.rmdir()
        stage.rename(output)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    return {**manifest, "reused": False}


def safe_error(exc: Exception, model_id: str) -> str:
    message = re.sub(r"hf_[A-Za-z0-9]+", "[credential]", str(exc).strip() or type(exc).__name__)
    if type(exc).__name__ == "GatedRepoError" or "403" in message:
        return (f"アクセスが拒否されました。https://huggingface.co/{model_id} で"
                "FLUX Kommunity Licenseに同意してから再実行してください。\n" + message)
    return message


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="FLUX 3 Image Klein (BF16) の取得・検証")
    parser.add_argument("--model", choices=sorted(MODELS), default="klein",
                        help="klein: 画像モデル / pup: プロンプトアップサンプラー")
    parser.add_argument("--model-dir", type=Path)
    args = parser.parse_args(argv)
    model = MODELS[args.model]
    output = args.model_dir or model["directory"]
    os.environ.update(hub_environment(RUNTIME / "cache/huggingface", os.environ))
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        result = prepare(output, model)
    except Exception as exc:  # noqa: BLE001 - CLI reports failures without a traceback.
        print(safe_error(exc, model["model_id"]), file=sys.stderr, flush=True)
        return 1
    state = "検証済みのモデルを再利用しました" if result["reused"] else "取得と検証が完了しました"
    print(f"{state}: {output} ({result['snapshot_required_bytes'] / 1e9:.2f} GB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
