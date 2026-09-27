"""Validate bounded M2 result bundles without extracting worker-controlled paths."""

from __future__ import annotations

import binascii
import io
import json
import struct
import wave
import zipfile
import zlib

from packages.contracts.m2 import (
    CharacterResult,
    GenerationEnvelope,
    LegacyCharacterResult,
    RelationshipsResult,
    WorldResult,
)

MAX_BUNDLE_BYTES = 32 * 1024 * 1024


def validate_png(data: bytes, *, require_transparency: bool = True) -> None:
    """Require a decodable, nonempty transparent RGBA PNG (the worker output contract)."""
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("image is not a PNG")
    offset, chunks, pixels = 8, [], bytearray()
    width = height = 0
    while offset < len(data):
        if offset + 12 > len(data):
            raise ValueError("truncated PNG")
        size, kind = struct.unpack_from(">I4s", data, offset)
        end = offset + 12 + size
        if end > len(data):
            raise ValueError("truncated PNG chunk")
        body = data[offset + 8 : end - 4]
        if binascii.crc32(kind + body) & 0xFFFFFFFF != struct.unpack_from(">I", data, end - 4)[0]:
            raise ValueError("PNG checksum mismatch")
        if not chunks and kind != b"IHDR":
            raise ValueError("missing PNG header")
        if kind == b"IHDR":
            if chunks or len(body) != 13:
                raise ValueError("invalid PNG header")
            width, height, depth, color, compression, filtering, interlace = struct.unpack(
                ">IIBBBBB", body
            )
            if not (1 <= width <= 4096 and 1 <= height <= 4096):
                raise ValueError("PNG dimensions exceed limit")
            colors = (6,) if require_transparency else (2, 6)
            if color not in colors or (depth, compression, filtering, interlace) != (8, 0, 0, 0):
                raise ValueError("PNG must be non-interlaced 8-bit RGBA")
        if kind == b"IDAT":
            pixels.extend(body)
        if kind == b"IEND" and (body or end != len(data)):
            raise ValueError("invalid PNG end")
        chunks.append(kind)
        offset = end
    if not chunks or chunks[-1] != b"IEND" or not pixels:
        raise ValueError("incomplete PNG")
    channels = 4 if color == 6 else 3
    stride = width * channels
    expected = (stride + 1) * height
    decoder = zlib.decompressobj()
    raw = decoder.decompress(pixels, expected + 1)
    if len(raw) != expected or not decoder.eof or decoder.unused_data:
        raise ValueError("PNG decoded size does not match dimensions")
    previous = bytearray(stride)
    transparent = visible = False
    for y in range(height):
        start = y * (stride + 1)
        filtering = raw[start]
        row = bytearray(raw[start + 1 : start + 1 + stride])
        if filtering > 4:
            raise ValueError("invalid PNG filter")
        for x in range(stride):
            a = row[x - channels] if x >= channels else 0
            b = previous[x]
            c = previous[x - channels] if x >= channels else 0
            predictor = 0
            if filtering == 1:
                predictor = a
            elif filtering == 2:
                predictor = b
            elif filtering == 3:
                predictor = (a + b) // 2
            elif filtering == 4:
                p = a + b - c
                pa, pb, pc = abs(p - a), abs(p - b), abs(p - c)
                predictor = a if pa <= pb and pa <= pc else b if pb <= pc else c
            row[x] = (row[x] + predictor) & 255
        alphas = row[3::4] if channels == 4 else [255]
        transparent |= min(alphas) < 255
        visible |= max(alphas) > 0
        previous = row
    if not visible or (require_transparency and not transparent):
        raise ValueError("PNG must include a visible character and transparent background")


def validate_wav(data: bytes) -> None:
    with wave.open(io.BytesIO(data), "rb") as audio:
        if audio.getcomptype() != "NONE" or audio.getnchannels() not in (1, 2):
            raise ValueError("voice must be mono/stereo PCM")
        if audio.getsampwidth() not in (1, 2, 3, 4) or not 8_000 <= audio.getframerate() <= 192_000:
            raise ValueError("invalid voice sample format")
        frames = audio.getnframes()
        if not 1 <= frames <= audio.getframerate() * 300:
            raise ValueError("empty or oversized voice")
        if len(audio.readframes(frames)) != frames * audio.getsampwidth() * audio.getnchannels():
            raise ValueError("truncated voice")


def validate_bundle(
    data: bytes, kind: str, character_id: str | None, character_contract_version: int = 1
) -> tuple[dict, dict[str, bytes]]:
    if not data or len(data) > MAX_BUNDLE_BYTES:
        raise ValueError("invalid result bundle size")
    expected = {"result.json"}
    if kind == "m2_image":
        expected.add("image.png")
    elif kind in ("m2_voice", "m2_voice_clone"):
        expected.add("voice.wav")
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        entries = archive.infolist()
        if len(entries) != len(expected) or {entry.filename for entry in entries} != expected:
            raise ValueError("bundle must contain exactly the expected files")
        if sum(entry.file_size for entry in entries) > MAX_BUNDLE_BYTES:
            raise ValueError("decompressed bundle exceeds limit")
        if any(entry.flag_bits & 1 or entry.is_dir() for entry in entries):
            raise ValueError("encrypted files and directories are unsupported")
        files = {entry.filename: archive.read(entry) for entry in entries}
    if len(files["result.json"]) > 2 * 1024 * 1024:
        raise ValueError("result JSON exceeds limit")
    envelope = GenerationEnvelope.model_validate_json(files["result.json"]).model_dump()
    if envelope["kind"] != kind:
        raise ValueError("result kind differs from job")
    if kind == "m2_world":
        envelope["result"] = WorldResult.model_validate(envelope["result"]).model_dump()
    elif kind == "m2_character":
        contract = (
            CharacterResult
            if character_contract_version >= 2
            or "selfIntroduction" in envelope["result"]
            or "sampleLines" in envelope["result"]
            else LegacyCharacterResult
        )
        result = contract.model_validate(envelope["result"]).model_dump()
        if result["id"] != character_id:
            raise ValueError("character ID differs from job")
        envelope["result"] = result
    elif kind == "m2_relationships":
        envelope["result"] = RelationshipsResult.model_validate(envelope["result"]).model_dump(
            mode="json"
        )
    elif envelope["result"]:
        raise ValueError("media jobs cannot change character text")
    if "image.png" in files:
        validate_png(files["image.png"])
    if "voice.wav" in files:
        validate_wav(files["voice.wav"])
    # Ensure metadata can be serialized without nonfinite numbers/unpaired surrogates.
    json.dumps(envelope, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return envelope, files
