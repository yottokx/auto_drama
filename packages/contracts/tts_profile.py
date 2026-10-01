"""Freeze each TTS purpose's weights and inference settings into generation jobs."""

from __future__ import annotations

from copy import deepcopy

from .tts_catalog import manifest_fingerprint, resolve_bundle

PURPOSES = ("voice_design", "voice_clone")


def build_tts_profile(settings: dict) -> dict:
    result = {"schema_version": 1}
    for purpose in PURPOSES:
        choice = settings[purpose]
        bundle = resolve_bundle(choice["model_id"], choice["precision"])
        if choice["provider_id"] != bundle["provider_id"]:
            raise ValueError("TTS provider does not match its model")
        result[purpose] = {
            "provider_id": choice["provider_id"], "model_id": choice["model_id"],
            "precision": choice["precision"], "manifest_id": bundle["manifest_id"],
            "bundle": bundle, "num_steps": 40, "codec_precision": "fp32",
        }
    return result


def select_tts_profile(profile: dict | None, purpose: str) -> dict | None:
    """Select a frozen entry; absent profiles retain legacy Small behavior."""
    if purpose not in PURPOSES:
        raise ValueError("Unknown TTS generation purpose")
    if profile is None:
        return None
    if profile.get("schema_version") != 1:
        raise ValueError("Unsupported TTS profile schema version")
    entry = deepcopy(profile[purpose])
    bundle = entry["bundle"]
    if (entry["manifest_id"] != manifest_fingerprint(bundle)
            or bundle.get("manifest_id") != entry["manifest_id"]
            or any(entry[key] != bundle[key] for key in ("provider_id", "model_id", "precision"))):
        raise ValueError("TTS profile does not match its frozen manifest")
    if (not isinstance(entry["num_steps"], int) or isinstance(entry["num_steps"], bool)
            or entry["num_steps"] < 1 or entry["codec_precision"] != "fp32"):
        raise ValueError("Unsupported TTS inference parameters")
    return entry
