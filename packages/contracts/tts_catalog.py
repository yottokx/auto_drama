"""Pinned, provider-neutral download catalog; acquisition does not enable synthesis."""

from __future__ import annotations

import copy
import hashlib
import json
from functools import lru_cache
from pathlib import Path

PRECISIONS = ("fp32", "bf16", "int8", "int4")
CATALOG_PATH = Path(__file__).resolve().parents[2] / "config/tts-models.json"


def manifest_fingerprint(bundle: dict) -> str:
    """Execution precision does not create a second copy of full precision files."""
    pinned = {key: value for key, value in bundle.items() if key not in {"manifest_id", "precision"}}
    canonical = json.dumps(pinned, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


@lru_cache(maxsize=1)
def _bundles() -> tuple[dict, ...]:
    document = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    result = []
    for item in document["bundles"]:
        bundle = copy.deepcopy(item)
        bundle["source_commit"] = document["source_commit"]
        bundle["total_bytes"] = sum(file["size"] for file in bundle["files"])
        bundle["manifest_id"] = manifest_fingerprint(bundle)
        result.append(bundle)
    return tuple(result)


def resolve_bundle(model_id: str, precision: str) -> dict:
    """Return a detached manifest whose ID fixes every file and dependency."""
    for bundle in _bundles():
        if (bundle["model_id"], bundle["precision"]) == (model_id, precision):
            return copy.deepcopy(bundle)
    raise ValueError("Unknown TTS model or precision")


def catalog() -> list[dict]:
    result = []
    for model_id in dict.fromkeys(bundle["model_id"] for bundle in _bundles()):
        bundles = [bundle for bundle in _bundles() if bundle["model_id"] == model_id]
        first = bundles[0]
        result.append({
            "provider_id": first["provider_id"], "model_id": model_id,
            "label": first["label"], "license": first["license"],
            "source_url": first["source_url"], "precisions": list(PRECISIONS),
            "purposes": ["voice_design", "voice_clone"], "prepared_only": True,
            "variants": [{
                "precision": bundle["precision"], "weight_id": bundle["weight_id"],
                "manifest_id": bundle["manifest_id"], "download_bytes": bundle["total_bytes"],
                "weight_bytes": next(file["size"] for file in bundle["files"]
                                     if file["path"].endswith("model.safetensors")),
                "download_note": ("通常ウェイトを共用／実行精度" if bundle["precision"]
                                  in {"fp32", "bf16"} else "公式量子化ウェイト"),
            } for bundle in bundles],
        })
    return result


public_catalog = catalog
resolve_download = resolve_bundle
