"""One immutable portrait baseline for preview, chapters and Tyrano publication."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Mapping
from dataclasses import asdict

from packages.contracts import Script
from packages.contracts.script import PortraitBaseline, PortraitBounds, PortraitTransform

from .portrait import (
    DEFAULT_HUMANOID_HEIGHT_CM,
    SLOT_CENTERS,
    PortraitLayout,
    PortraitSource,
    _legacy_layout,
    _silhouette,
    height_group_geometry,
)

STAGE = {"width": 960, "height": 640, "max_portraits": 3}
BACKGROUND = {"fit": "cover", "position": "center center", "z_index": 0}
MESSAGE_WINDOW = {
    "left": 20, "top": 440, "width": 920, "height": 180,
    "padding": {"left": 24, "top": 18, "right": 24, "bottom": 18},
    "font_size": 24, "line_spacing": 6, "color": "#f3f4f6",
    "background": "#101923", "opacity": 230 / 255, "z_index": 100,
}
PORTRAIT_Z_INDEX = 10


def portrait_source_hash(source: PortraitSource) -> str:
    value = {"image_sha256": hashlib.sha256(source.image).hexdigest(), "framing": source.framing,
             "height_cm": float(source.height_cm) if source.height_cm is not None else None,
             "body_bounds": PortraitBounds.model_validate(source.body_bounds).model_dump(mode="json")
             if source.body_bounds is not None else None}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def make_portrait_baseline(sources: Mapping[str, PortraitSource], previous: dict | None = None) -> dict:
    """Compute the all-cast baseline once; later changes update only their own layouts.

    Keep records for characters outside a chapter/preview subset. New images or
    body rectangles invalidate only that source's layout; original group metrics
    stay fixed, including when a taller/wider character is added afterwards.
    """
    baseline = PortraitBaseline.model_validate(previous).model_dump(mode="json") if previous is not None else None
    for source in sources.values():
        if source.height_cm is not None and (isinstance(source.height_cm, bool)
                or not math.isfinite(source.height_cm) or source.height_cm <= 0):
            raise ValueError("portrait height must be a finite positive number")
    subjects = ({key: _silhouette(source.image, source.framing, source.body_bounds)
                 for key, source in sources.items()} if baseline is None else {})
    if baseline is None:
        humanoids = {key: subject for key, subject in subjects.items()
                     if sources[key].framing == "upper_body"}
        metrics = (height_group_geometry(humanoids, {
            key: sources[key].height_cm or DEFAULT_HUMANOID_HEIGHT_CM for key in humanoids
        }) if any(sources[key].height_cm is not None for key in humanoids) else (None, None))
        baseline = {"schema_version": 1, "pixels_per_cm": metrics[0], "ground_y": metrics[1], "layouts": {}}
    for key, source in sources.items():
        fingerprint = portrait_source_hash(source)
        saved = baseline["layouts"].get(key)
        if saved is not None and saved["source_sha256"] == fingerprint:
            continue
        subject = subjects.get(key) or _silhouette(source.image, source.framing, source.body_bounds)
        if source.framing == "upper_body" and baseline["pixels_per_cm"] is not None:
            scale = ((source.height_cm or DEFAULT_HUMANOID_HEIGHT_CM) * baseline["pixels_per_cm"]
                     / subject.visible_height)
            layout = subject.layout(scale, baseline["ground_y"] - subject.bottom * scale)
        else:
            layout = _legacy_layout(subject)
        anchor = (baseline["ground_y"] if source.framing == "upper_body" and baseline["pixels_per_cm"] is not None
                  else layout.top + subject.bottom * layout.height / subject.height)
        baseline["layouts"][key] = {**asdict(layout), "anchor_y": anchor, "source_sha256": fingerprint}
    return PortraitBaseline.model_validate(baseline).model_dump(mode="json")


def resolved_portrait_layout(base: dict, adjustment: dict | None = None) -> PortraitLayout:
    transform = PortraitTransform.model_validate(adjustment or {})
    return PortraitLayout(
        width=max(1, round(base["width"] * transform.scale)),
        height=max(1, round(base["height"] * transform.scale)),
        top=round(base["anchor_y"] + (base["top"] - base["anchor_y"]) * transform.scale + transform.offset_y),
        center_offset=base["center_offset"] * transform.scale, mode=base["mode"],
    )


def stage_geometry(sources: Mapping[str, PortraitSource], baseline: dict | None = None,
                   adjustments: dict | None = None) -> dict:
    baseline = make_portrait_baseline(sources, baseline)
    adjustments = adjustments or {}
    characters = {}
    for key in sources:
        base = baseline["layouts"][key]
        transform = PortraitTransform.model_validate(adjustments.get(key, {})).model_dump(mode="json")
        layout = resolved_portrait_layout(base, transform)
        characters[key] = {
            "baseline": base, "adjustment": transform, "layout": asdict(layout), "z_index": PORTRAIT_Z_INDEX,
            "positions": {position: {"left": layout.left(position), "top": layout.top,
                                     "width": layout.width, "height": layout.height}
                          for position in SLOT_CENTERS},
        }
    return {"stage": dict(STAGE), "background": dict(BACKGROUND), "message_window": copy.deepcopy(MESSAGE_WINDOW),
            "slots": dict(SLOT_CENTERS), "baseline": baseline, "characters": characters}


def script_portrait_sources(script: Script, images: dict[str, bytes]) -> dict[str, PortraitSource]:
    return {character.id: PortraitSource(images[character.image_asset_id], character.framing, character.height_cm,
                                         character.body_bounds.model_dump() if character.body_bounds else None)
            for character in script.characters if character.image_asset_id}


def script_portrait_layouts(script: Script, images: dict[str, bytes]) -> dict[str, PortraitLayout]:
    sources = script_portrait_sources(script, images)
    if script.portrait_baseline is None:
        # Persisted pre-edition scripts keep their original chapter-wide framing.
        defaults = make_portrait_baseline(sources)
    else:
        defaults = script.portrait_baseline.model_dump(mode="json")
        for key, source in sources.items():
            if defaults["layouts"][key]["source_sha256"] != portrait_source_hash(source):
                raise ValueError("portrait baseline differs from the selected image or body bounds")
    adjustments = {character.id: {"offset_y": character.offset_y, "scale": character.scale}
                   for character in script.characters}
    return {key: resolved_portrait_layout(defaults["layouts"][key], adjustments[key]) for key in sources}


def apply_portrait_presentation(script: Script, baseline: dict, adjustments: dict | None = None) -> Script:
    values = script.model_dump(mode="json")
    values["portrait_baseline"] = baseline
    adjustments = adjustments or {}
    for character in values["characters"]:
        character.update(PortraitTransform.model_validate(adjustments.get(character["id"], {})).model_dump(mode="json"))
    return Script.model_validate(values)
