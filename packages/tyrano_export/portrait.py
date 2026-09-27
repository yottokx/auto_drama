"""Aspect-preserving stage coordinates derived from the visible portrait silhouette."""

from __future__ import annotations

import io
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from PIL import Image, ImageFilter, UnidentifiedImageError

from packages.contracts.script import PortraitBounds

Framing = Literal["auto", "upper_body", "full_body"]
SLOT_CENTERS = {"left": 200, "center": 480, "right": 760}
DEFAULT_HUMANOID_HEIGHT_CM = 170.0


@dataclass(frozen=True)
class PortraitSource:
    image: bytes
    framing: Framing = "auto"
    height_cm: float | None = None
    body_bounds: Mapping[str, float] | None = None


@dataclass(frozen=True)
class PortraitLayout:
    width: int
    height: int
    top: int
    center_offset: float
    mode: str

    def left(self, position: str) -> int:
        return round(SLOT_CENTERS[position] - self.center_offset)


@dataclass(frozen=True)
class _Silhouette:
    width: int
    height: int
    left: float
    top: float
    right: float
    bottom: float
    mode: str

    @property
    def visible_width(self) -> float:
        return self.right - self.left

    @property
    def visible_height(self) -> float:
        return self.bottom - self.top

    def layout(self, scale: float, image_top: float) -> PortraitLayout:
        return PortraitLayout(
            width=max(1, round(self.width * scale)),
            height=max(1, round(self.height * scale)),
            top=round(image_top),
            center_offset=(self.left + self.right) * scale / 2,
            mode=self.mode,
        )


def _silhouette(
    image: bytes, framing: Framing, body_bounds: Mapping[str, float] | None = None,
) -> _Silhouette:
    if framing not in ("auto", "upper_body", "full_body"):
        raise ValueError("unsupported portrait framing")
    try:
        with Image.open(io.BytesIO(image)) as source:
            width, height = source.size
            if not (1 <= width <= 4096 and 1 <= height <= 4096):
                raise ValueError("portrait dimensions exceed limit")
            alpha = source.convert("RGBA").getchannel("A")
            mask = alpha.point(lambda value: 255 if value >= 32 else 0)
            # Background removal can leave faint haze or isolated solid pixels.
            # Use the unfiltered mask for very small/thin artwork it would erase.
            bounds = mask.filter(ImageFilter.MedianFilter(3)).getbbox() or mask.getbbox()
            if bounds is None:
                bounds = alpha.getbbox()
            if bounds is None:
                raise ValueError("portrait has no visible pixels")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("portrait must be a decodable image") from exc

    left, top, right, bottom = bounds
    if body_bounds is not None:
        reference = PortraitBounds.model_validate(dict(body_bounds))
        left, top = reference.left * width, reference.top * height
        right, bottom = reference.right * width, reference.bottom * height
        # The reference controls geometry only. Keep the original image intact,
        # including hats and props outside the frame; never fit them back down.
    visible_width, visible_height = right - left, bottom - top
    mode = framing
    if mode == "auto":
        mode = "upper_body" if visible_width / visible_height <= 0.60 else "full_body"
    return _Silhouette(width, height, left, top, right, bottom, mode)


def _legacy_layout(subject: _Silhouette) -> PortraitLayout:
    if subject.mode == "upper_body":
        # Headroom starts below the toolbar. Roughly the upper half of a tall
        # portrait occupies y56..440; the lower body continues behind the window.
        scale = min(
            (440 - 56) / (0.5 * subject.visible_height), 380 / subject.visible_width
        )
        image_top = 56 - subject.top * scale
    else:
        scale = min(400 / subject.visible_height, 300 / subject.visible_width)
        # Slight overlap with the message window avoids feet floating above it.
        image_top = 456 - subject.bottom * scale
    return subject.layout(scale, image_top)


def portrait_layout(
    image: bytes, framing: Framing = "auto", body_bounds: Mapping[str, float] | None = None,
) -> PortraitLayout:
    """Retain legacy single-portrait framing when no physical height is supplied."""
    return _legacy_layout(_silhouette(image, framing, body_bounds))


def portrait_layouts(sources: Mapping[str, PortraitSource]) -> dict[str, PortraitLayout]:
    """Place a chapter's humanoids on one ground line at one pixels/cm scale.

    Only explicitly upper-body subjects join the physical-height group. Auto,
    full-body subjects and chapters without height metadata retain legacy
    framing. An unspecified humanoid height in a mixed chapter uses 170 cm.
    A supplied body rectangle replaces silhouette bounds for height, width,
    horizontal center and ground anchoring. Shoulders are still estimated at
    30% below the head. Pose and anatomy are not recognized automatically.
    """
    subjects = {
        key: _silhouette(source.image, source.framing, source.body_bounds)
        for key, source in sources.items()
    }
    for source in sources.values():
        if source.height_cm is not None and (
            isinstance(source.height_cm, bool)
            or not math.isfinite(source.height_cm)
            or source.height_cm <= 0
        ):
            raise ValueError("portrait height must be a finite positive number")
    layouts = {key: _legacy_layout(subject) for key, subject in subjects.items()}
    humanoids = {
        key: subject for key, subject in subjects.items() if sources[key].framing == "upper_body"
    }
    if not any(sources[key].height_cm is not None for key in humanoids):
        return layouts
    heights = {
        key: sources[key].height_cm or DEFAULT_HUMANOID_HEIGHT_CM for key in humanoids
    }
    tallest, shortest = max(heights.values()), min(heights.values())
    # One group-wide scale preserves physical height ratios even when a broad
    # pose or the shortest character's shoulders constrain the composition.
    pixels_per_cm = min(
        768 / tallest,
        (420 - 56) / (tallest - 0.7 * shortest),
        *(380 * subject.visible_height / (subject.visible_width * heights[key])
          for key, subject in humanoids.items()),
    )
    baseline = 56 + tallest * pixels_per_cm
    for key, subject in humanoids.items():
        scale = heights[key] * pixels_per_cm / subject.visible_height
        layouts[key] = subject.layout(scale, baseline - subject.bottom * scale)
    return layouts
