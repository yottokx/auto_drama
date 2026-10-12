"""Where a portrait's face is, and putting an edited face back onto the portrait it came from.

The editing model redraws the whole picture: outlines shift by a pixel, and a prop behind the
figure can vanish. Only the face of its answer is kept, so an expression differs from its
portrait in the face alone and the two can be swapped line by line without the figure moving.
"""
from __future__ import annotations

import hashlib
import io
import json
import tempfile
from pathlib import Path

from . import pipeline
from .processes import run_process

# Room around the detected face, as fractions of its size: it hugs the face, and brows,
# a blush or tears need a little more. Tears run below the chin.
MARGIN = {"side": 0.12, "top": 0.05, "bottom": 0.08}
BOTTOM_MARGIN = {"cry": 0.3}
DETECTOR_VERSION = 1


class FaceNotFound(ValueError):
    """No face could be located in the portrait; the user can mark it by hand."""


def detect(reference: Path, config: dict) -> list[int]:
    """The portrait's face as a pixel box, found once per picture on the CPU."""
    digest = hashlib.sha256(reference.read_bytes()).hexdigest()
    cache = pipeline.ROOT / "services/worker/cache/m2/faces" / f"{digest}-v{DETECTOR_VERSION}.json"
    if not cache.is_file():
        cache.parent.mkdir(parents=True, exist_ok=True)
        output = Path(tempfile.mkdtemp(prefix="face-", dir=cache.parent))
        settings = config["image"]
        run_process([str(pipeline.ROOT / settings["python"]), str(pipeline.ROOT / "scripts/m0/face_detect.py"),
                     "--image", str(reference), "--output", str(output / "faces.json")],
                    output / "runtime.log", cwd=pipeline.ROOT, timeout=120)
        (output / "faces.json").replace(cache)
    (result,) = json.loads(cache.read_text(encoding="utf-8"))["results"]
    if not result["faces"]:
        raise FaceNotFound("立ち絵から顔を見つけられませんでした。顔の範囲を指定してください。")
    # The detector lists the most certain face first; a portrait shows one person.
    return result["faces"][0]["box"]


def area(face: list[float], size: tuple[int, int], expression: str) -> tuple[int, int, int, int]:
    """The part of the picture an expression may change, within the picture."""
    left, top, right, bottom = face
    width, height = right - left, bottom - top
    if width <= 0 or height <= 0:
        raise ValueError("The face box is empty.")
    below = BOTTOM_MARGIN.get(expression, MARGIN["bottom"])
    box = (left - MARGIN["side"] * width, top - MARGIN["top"] * height,
           right + MARGIN["side"] * width, bottom + below * height)
    return (max(0, round(box[0])), max(0, round(box[1])), min(size[0], round(box[2])), min(size[1], round(box[3])))


def composite(base: bytes, edited: bytes, box: tuple[int, int, int, int]) -> bytes:
    """The base portrait with only ``box`` taken from the edit, its border blended."""
    from PIL import Image, ImageDraw, ImageFilter

    source = Image.open(io.BytesIO(base)).convert("RGBA")
    edit = Image.open(io.BytesIO(edited)).convert("RGBA")
    if edit.size != source.size:
        edit = edit.resize(source.size, Image.Resampling.LANCZOS)
    size = box[2] - box[0]
    mask = Image.new("L", source.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=round(size * 0.2), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(max(1, round(size * 0.06))))
    # Blend with premultiplied colour so a transparent edge never bleeds into the face.
    mixed = Image.composite(edit.convert("RGBa"), source.convert("RGBa"), mask).convert("RGBA")
    result = source.copy()
    changed = mask.getbbox()
    if changed is None:
        raise ValueError("The face box lies outside the portrait.")
    # Every pixel the blend does not reach is the base portrait's own, bit for bit.
    result.paste(mixed.crop(changed), changed)
    stream = io.BytesIO()
    result.save(stream, "PNG")
    return stream.getvalue()
