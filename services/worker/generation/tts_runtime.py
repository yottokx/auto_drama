"""Resolve pinned TTS selections without importing GPU libraries or fetching files."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from packages.contracts.tts_catalog import resolve_bundle
from services.worker.tts_downloads import _validated_manifest

ROOT = Path(__file__).resolve().parents[3]


@dataclass(frozen=True)
class TTSRuntimeProfile:
    bundle: dict

    @property
    def compute_precision(self) -> str:
        return "bf16" if self.bundle["quantization"] is not None else self.bundle["precision"]

    @property
    def identity(self) -> str:
        # The entire pinned bundle includes every tokenizer, codec and watermark
        # revision/hash. Execution precision is significant even when files are shared.
        value = {
            "bundle": self.bundle, "model_device": "cuda", "codec_device": "cuda",
            "compute_precision": self.compute_precision, "codec_precision": "fp32",
            "codec_deterministic_encode": True, "codec_deterministic_decode": True,
            "compile_model": False,
        }
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()

    def path(self, file: dict, root: Path = ROOT) -> Path:
        storage = (root / "services/worker/runtimes/irodori/models").resolve()
        path = (root / file["local_dir"] / file["path"]).resolve()
        if not path.is_relative_to(storage):
            raise ValueError("TTS model files must remain inside worker model storage.")
        return path

    def file(self, filename: str, *, repo_id: str | None = None) -> dict:
        matches = [file for file in self.bundle["files"]
                   if Path(file["path"]).name == filename
                   and (repo_id is None or file["repo_id"] == repo_id)]
        if len(matches) != 1:
            raise ValueError(f"Pinned TTS bundle requires exactly one {filename}.")
        return matches[0]

    def provenance(self) -> dict:
        checkpoint = self.file("model.safetensors")
        dependencies = {
            file["repo_id"]: file["revision"] for file in self.bundle["files"]
            if file["repo_id"] != checkpoint["repo_id"]
        }
        return {
            "provider_id": self.bundle["provider_id"], "model_id": self.bundle["model_id"],
            "model": self.bundle["label"], "precision": self.bundle["precision"],
            "model_precision": self.compute_precision,
            "manifest_id": self.bundle["manifest_id"],
            "weight_id": self.bundle["weight_id"], "model_repo_id": checkpoint["repo_id"],
            "model_revision": checkpoint["revision"],
            "source_commit": self.bundle["source_commit"],
            "quantization": self.bundle["quantization"], "dependency_revisions": dependencies,
            "model_device": "cuda", "codec_device": "cuda", "codec_precision": "fp32",
            "runtime_identity": self.identity,
        }


def resolve_request_profile(request: dict) -> TTSRuntimeProfile:
    """Old jobs retain their Small settings; new jobs carry a frozen profile."""
    selection = request.get("tts_profile")
    if selection is None:
        model_id, precision = "irodori-v4.1-small", request.get("model_precision", "fp32")
    else:
        if not isinstance(selection, dict) or selection.get("provider_id") != "irodori":
            raise ValueError("Unsupported TTS provider.")
        model_id, precision = selection.get("model_id"), selection.get("precision")
        if selection.get("codec_precision", "fp32") != "fp32":
            raise ValueError("Pinned TTS codec requires fp32 precision.")
        if "num_steps" in selection and selection["num_steps"] != request.get("num_steps"):
            raise ValueError("TTS sampling steps differ from the frozen job profile.")
    if selection is not None and "bundle" in selection:
        bundle = _validated_manifest(selection["bundle"], model_id, precision)
        if bundle["provider_id"] != selection["provider_id"]:
            raise ValueError("TTS manifest provider differs from its profile.")
    else:
        bundle = resolve_bundle(model_id, precision)
    if selection is not None and selection.get("manifest_id") != bundle["manifest_id"]:
        raise ValueError("Pinned TTS profile differs from the installed model catalog.")
    return TTSRuntimeProfile(bundle)


def runtime_identity(request: dict) -> str:
    return resolve_request_profile(request).identity


def verify_profile_files(profile: TTSRuntimeProfile, root: Path = ROOT) -> None:
    """Verify one selected bundle completely before loading a CUDA checkpoint."""
    paths = [(file, profile.path(file, root)) for file in profile.bundle["files"]]
    for file, path in paths:
        if not path.is_file() or path.stat().st_size != file["size"]:
            raise FileNotFoundError("Selected TTS model download is incomplete; download it in settings.")
    for file, path in paths:
        digest = hashlib.sha256() if "sha256" in file else hashlib.sha1()  # Git blob identity.
        if "sha256" not in file:
            digest.update(f"blob {file['size']}\0".encode())
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != file.get("sha256", file.get("git_blob_sha1")):
            raise ValueError("Selected TTS model file integrity check failed; download it again.")
