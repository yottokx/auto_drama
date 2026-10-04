"""Bounded, closed worker result format for first-chapter production."""

from __future__ import annotations

import io
import json
import zipfile

from packages.contracts.m3 import GenerationEnvelope
from packages.contracts.portrait_recovery import PortraitOmission

from .m2_bundle import MAX_BUNDLE_BYTES, validate_png, validate_wav


def validate_background(data: bytes) -> None:
    validate_png(data, require_transparency=False)


def validate_bundle(data: bytes, kind: str) -> tuple[dict, dict[str, bytes]]:
    if not data or len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("invalid result bundle size")
    expected = {"result.json"}
    if kind in ("m3_background", "m3_image"):
        expected.add("image.png")
    elif kind in ("m3_voice", "m3_voice_clone"):
        expected.add("voice.wav")
    elif kind == "m3_music":
        expected.update({"music.mp3", "source.mp3"})
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        names = {entry.filename for entry in entries}
        omission_bundle = kind == "m3_image" and names == {"result.json"}
        if omission_bundle:
            expected = {"result.json"}
        if len(entries) != len(expected) or names != expected:
            raise ValueError("bundle must contain exactly the expected files")
        if sum(entry.file_size for entry in entries) > MAX_BUNDLE_BYTES:
            raise ValueError("decompressed bundle exceeds limit")
        if any(entry.flag_bits & 1 or entry.is_dir() for entry in entries):
            raise ValueError("encrypted files and directories are unsupported")
        files = {entry.filename: archive.read(entry) for entry in entries}
    if len(files["result.json"]) > 2 * 1024 * 1024:
        raise ValueError("result JSON exceeds limit")
    envelope = GenerationEnvelope.model_validate_json(files["result.json"]).model_dump()
    if omission_bundle:
        if set(envelope["result"]) != {"portrait"}:
            raise ValueError("Missing explicit portrait omission")
        PortraitOmission.model_validate(envelope["result"]["portrait"])
    if envelope["kind"] != kind or (kind not in {"m3_narrative", "m3_plan", "m3_music_plan", "m3_music"} and not omission_bundle and envelope["result"]):
        raise ValueError("result differs from job contract")
    if kind == "m3_music_plan":
        from packages.contracts.music import MusicPlan

        MusicPlan.model_validate(envelope["result"])
    elif kind == "m3_music":
        from packages.contracts.music import MusicResult

        MusicResult.model_validate(envelope["result"])
    if kind == "m3_background":
        validate_background(files["image.png"])
    elif kind == "m3_image" and not omission_bundle:
        validate_png(files["image.png"])
    elif "voice.wav" in files:
        validate_wav(files["voice.wav"])
    json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return envelope, files
