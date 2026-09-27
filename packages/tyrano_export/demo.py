"""Small CPU-only fixture: original geometric PNGs and a fixed Japanese scene."""

from __future__ import annotations

import hashlib
import struct
import zlib

from packages.contracts import Script


def _png(width: int, height: int, pixel) -> bytes:
    def chunk(kind: bytes, content: bytes) -> bytes:
        return (
            struct.pack(">I", len(content))
            + kind
            + content
            + struct.pack(">I", zlib.crc32(kind + content) & 0xFFFFFFFF)
        )

    pixels = b"".join(
        b"\0" + b"".join(bytes(pixel(x, y)) for x in range(width)) for y in range(height)
    )
    # An uncompressed DEFLATE stream makes the fixture independent of zlib versions.
    blocks = [pixels[i : i + 65535] for i in range(0, len(pixels), 65535)]
    deflate = (
        b"\x78\x01"
        + b"".join(
            bytes([int(i == len(blocks) - 1)])
            + struct.pack("<HH", len(block), len(block) ^ 65535)
            + block
            for i, block in enumerate(blocks)
        )
        + struct.pack(">I", zlib.adler32(pixels) & 0xFFFFFFFF)
    )
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", deflate)
        + chunk(b"IEND", b"")
    )


def demo_content() -> tuple[Script, dict[str, bytes]]:
    """Return a fresh fixture; coordinator replaces demo artifact IDs when storing it."""
    assets = {
        "station": _png(96, 64, lambda x, y: (30, 52, 76, 255) if y < 45 else (70, 78, 90, 255)),
        "aki_sprite": _png(
            28, 42, lambda x, y: (232, 165, 96, 255) if 5 < x < 23 and y > 3 else (0, 0, 0, 0)
        ),
        "ren_sprite": _png(
            28, 42, lambda x, y: (115, 185, 202, 255) if 5 < x < 23 and y > 3 else (0, 0, 0, 0)
        ),
    }
    script = Script.model_validate(
        {
            "schema_version": 1,
            "id": "demo_chapter_01",
            "title": "雨の駅で、もう一度",
            "characters": [
                {"id": "aki", "name": "アキ", "image_asset_id": "aki_sprite"},
                {"id": "ren", "name": "レン", "image_asset_id": "ren_sprite"},
            ],
            "utterances": [
                {
                    "id": "line_001",
                    "speaker_id": None,
                    "display_text": "雨の駅。アキは、濡れた封筒をレンの前に差し出した。",
                    "spoken_text": "雨の駅。アキは、濡れた封筒をレンの前に差し出した。",
                },
                {
                    "id": "line_002",
                    "speaker_id": "ren",
                    "voice_emotion": "worried",
                    "display_text": "今さら戻れないよ。僕が壊したんだから。",
                    "spoken_text": "今さら戻れないよ。僕が壊したんだから。",
                },
                {
                    "id": "line_003",
                    "speaker_id": "aki",
                    "voice_emotion": "determined",
                    "display_text": "だから、一緒に直したいの。ほら、まだ捨ててない。",
                    "spoken_text": "だから、一緒に直したいの。ほら、まだ捨ててない。",
                },
                {
                    "id": "line_004",
                    "speaker_id": "ren",
                    "voice_emotion": "hopeful",
                    "delivery": "少しためらってから、静かに",
                    "display_text": "……その封筒、僕にも持たせて。今度は逃げない。",
                    "spoken_text": "その封筒、僕にも持たせて。今度は逃げない。",
                },
            ],
            "directions": [
                {
                    "id": "bg_001",
                    "kind": "background",
                    "utterance_id": "line_001",
                    "asset_id": "station",
                },
                {
                    "id": "enter_aki",
                    "kind": "enter",
                    "utterance_id": "line_001",
                    "character_id": "aki",
                    "position": "left",
                },
                {
                    "id": "enter_ren",
                    "kind": "enter",
                    "utterance_id": "line_001",
                    "character_id": "ren",
                    "position": "right",
                },
                {
                    "id": "pause_ren",
                    "kind": "pause",
                    "utterance_id": "line_004",
                    "duration_ms": 450,
                },
            ],
            "assets": [
                {
                    "id": asset_id,
                    "kind": "background" if asset_id == "station" else "character",
                    "artifact_id": f"demo-{asset_id}",
                    "filename": f"{asset_id}.png",
                    "sha256": hashlib.sha256(content).hexdigest(),
                }
                for asset_id, content in assets.items()
            ],
        }
    )
    return script, assets
