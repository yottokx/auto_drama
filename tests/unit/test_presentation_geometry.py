from __future__ import annotations

import hashlib
import io
import re
from copy import deepcopy

import pytest
from PIL import Image, ImageDraw

from packages.contracts import Script
from packages.tyrano_export import compile_scenario, demo_content
from packages.tyrano_export.player import player_config
from packages.tyrano_export.portrait import PortraitSource
from packages.tyrano_export.presentation import (
    apply_portrait_presentation,
    make_portrait_baseline,
    script_portrait_sources,
    stage_geometry,
)


def portrait(size=(400, 900), bounds=(100, 20, 300, 880)):
    image = Image.new("RGBA", size)
    ImageDraw.Draw(image).rectangle(bounds, fill="navy")
    output = io.BytesIO()
    image.save(output, "PNG")
    return output.getvalue()


def sources():
    return {"adult": PortraitSource(portrait(), "upper_body", 175),
            "child": PortraitSource(portrait(), "upper_body", 110)}


def test_chapter_and_checked_cast_subsets_keep_union_baseline_dimensions():
    cast = sources()
    baseline = make_portrait_baseline(cast)
    full = stage_geometry(cast, baseline)
    for cid in cast:
        single = stage_geometry({cid: cast[cid]}, baseline)
        assert single["characters"][cid] == full["characters"][cid]
        assert single["baseline"] == baseline


def test_later_tall_or_wide_cast_additions_never_resize_existing_portraits():
    cast = sources()
    baseline = make_portrait_baseline(cast)
    new = PortraitSource(portrait((1000, 1000), (10, 10, 990, 990)), "upper_body", 230)
    expanded = make_portrait_baseline({**cast, "tall_wide": new}, baseline)
    assert {cid: expanded["layouts"][cid] for cid in cast} == baseline["layouts"]
    assert expanded["pixels_per_cm"] == baseline["pixels_per_cm"]
    assert expanded["ground_y"] == baseline["ground_y"]
    assert "tall_wide" in expanded["layouts"]
    assert baseline["layouts"].keys() == cast.keys()  # Caller snapshot is immutable.


def test_replacement_image_or_body_bounds_recalculates_only_affected_character():
    cast = sources()
    baseline = make_portrait_baseline(cast)
    replacement = PortraitSource(portrait((500, 1000), (150, 100, 350, 900)), "upper_body", 175)
    updated = make_portrait_baseline({**cast, "adult": replacement}, baseline)
    assert updated["layouts"]["child"] == baseline["layouts"]["child"]
    assert updated["layouts"]["adult"] != baseline["layouts"]["adult"]
    assert updated["ground_y"] == baseline["ground_y"]
    bounded = PortraitSource(replacement.image, "upper_body", 175,
                             {"left": .35, "right": .65, "top": .2, "bottom": .8})
    corrected = make_portrait_baseline({"adult": bounded}, updated)
    assert corrected["layouts"]["child"] == baseline["layouts"]["child"]
    assert corrected["layouts"]["adult"] != updated["layouts"]["adult"]


def test_adjustment_scales_around_fixed_ground_anchor_then_offsets_vertically():
    cast = sources()
    baseline = make_portrait_baseline(cast)
    adjusted = stage_geometry(cast, baseline, {"adult": {"scale": 1.5, "offset_y": -32}})
    base, result = baseline["layouts"]["adult"], adjusted["characters"]["adult"]
    assert result["layout"]["width"] == round(base["width"] * 1.5)
    assert result["layout"]["height"] == round(base["height"] * 1.5)
    assert result["layout"]["top"] == round(base["anchor_y"] + (base["top"] - base["anchor_y"]) * 1.5 - 32)
    assert result["positions"]["right"]["left"] - result["positions"]["center"]["left"] == 280
    assert adjusted["characters"]["child"] == stage_geometry(cast, baseline)["characters"]["child"]
    assert adjusted["baseline"] == baseline


def test_preview_and_output_use_same_coordinates_for_entry_move_and_message_window():
    script, assets = demo_content()
    values = script.model_dump(mode="json")
    for character, cm in zip(values["characters"], (175, 110), strict=True):
        character.update(framing="upper_body", height_cm=cm)
        raw = portrait()
        assets[character["image_asset_id"]] = raw
        for asset in values["assets"]:
            if asset["id"] == character["image_asset_id"]:
                asset["sha256"] = hashlib.sha256(raw).hexdigest()
    values["directions"].append({"id": "move_adjusted", "kind": "position", "utterance_id": "line_004",
                                  "character_id": "aki", "position": "right"})
    script = Script.model_validate(values)
    cast = script_portrait_sources(script, assets)
    baseline = make_portrait_baseline(cast)
    changes = {"aki": {"offset_y": -20, "scale": 1.25}}
    adjusted = apply_portrait_presentation(script, baseline, changes)
    preview = stage_geometry(cast, baseline, changes)
    scenario = compile_scenario(adjusted, assets)
    show = next(line for line in scenario.splitlines() if line.startswith('[chara_show name="ad_aki"'))
    tags = dict(re.findall(r'(\w+)="([^"]*)"', show))
    for key in ("left", "top", "width", "height"):
        assert int(tags[key]) == preview["characters"]["aki"]["positions"]["left"][key]
    move = next(line for line in scenario.splitlines() if line.startswith('[chara_move name="ad_aki"'))
    assert f'left="{preview["characters"]["aki"]["positions"]["right"]["left"]}"' in move
    assert 'zindex="10"' in show
    assert 'position="center center"' in next(line for line in scenario.splitlines() if line.startswith("[bg "))
    window = preview["message_window"]
    # The viewing screen draws its own window; the engine's message layer stays hidden.
    assert "[position " not in scenario and '[layopt layer="message0" visible="false"]' in scenario
    config = player_config(adjusted.model_dump_json().encode()).decode()
    for setting, key in (("ml", "left"), ("mt", "top"), ("mw", "width"), ("mh", "height")):
        assert f";{setting} = {window[key]};" in config
    assert script.utterances == adjusted.utterances
    assert script.directions == adjusted.directions
    assert script.assets == adjusted.assets


def test_stale_baseline_cannot_render_a_new_image_with_old_body_geometry():
    script, assets = demo_content()
    baseline = make_portrait_baseline(script_portrait_sources(script, assets))
    adjusted = apply_portrait_presentation(script, baseline)
    changed = {**assets, script.characters[0].image_asset_id: portrait()}
    with pytest.raises(ValueError, match="differs from the selected"):
        compile_scenario(adjusted, changed)


def test_height_and_body_numeric_normalization_survives_json_script_roundtrip():
    raw = portrait()
    source = PortraitSource(raw, "upper_body", 175, {"left": 0, "right": 1, "top": 0, "bottom": 1})
    before = make_portrait_baseline({"adult": source})
    after = make_portrait_baseline({"adult": PortraitSource(raw, "upper_body", 175.0,
                        {"left": 0.0, "right": 1.0, "top": 0.0, "bottom": 1.0})}, before)
    assert before == after


@pytest.mark.parametrize("changes", [{"offset_y": 641}, {"scale": 0}, {"scale": float("nan")}, {"scale": True}])
def test_invalid_transforms_rejected_without_mutating_baseline(changes):
    cast = sources()
    baseline = make_portrait_baseline(cast)
    saved = deepcopy(baseline)
    with pytest.raises(ValueError):
        stage_geometry(cast, baseline, {"adult": changes})
    assert baseline == saved
