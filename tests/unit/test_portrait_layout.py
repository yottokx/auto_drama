from __future__ import annotations

import hashlib
import io
import re

import pytest
from PIL import Image, ImageDraw

from packages.contracts import Script
from packages.tyrano_export import compile_scenario, demo_content
from packages.tyrano_export.player import player_config
from packages.tyrano_export.portrait import PortraitSource, portrait_layout, portrait_layouts


def png(image: Image.Image) -> bytes:
    output = io.BytesIO()
    image.save(output, format="PNG")
    return output.getvalue()


def silhouette(
    size: tuple[int, int] = (400, 800),
    bounds: tuple[int, int, int, int] = (130, 40, 270, 760),
) -> Image.Image:
    image = Image.new("RGBA", size)
    left, top, right, bottom = bounds
    ImageDraw.Draw(image).rectangle((left, top, right - 1, bottom - 1), fill="white")
    return image


def visible_rect(layout, size, bounds, position="center"):
    scale_x, scale_y = layout.width / size[0], layout.height / size[1]
    left, top, right, bottom = bounds
    return (
        layout.left(position) + left * scale_x,
        layout.top + top * scale_y,
        layout.left(position) + right * scale_x,
        layout.top + bottom * scale_y,
    )


def test_tall_silhouette_places_upper_body_above_message_and_preserves_aspect():
    image = silhouette()
    layout = portrait_layout(png(image))
    left, top, right, bottom = visible_rect(layout, image.size, (130, 40, 270, 760))

    assert layout.mode == "upper_body"
    assert 40 <= top <= 90  # A little headroom, with the head still on the stage.
    assert 0.42 <= (440 - top) / (bottom - top) <= 0.58
    assert bottom > 640  # The lower body continues behind the window/offscreen.
    assert (left + right) / 2 == pytest.approx(480, abs=1)
    assert layout.width / layout.height == pytest.approx(image.width / image.height, abs=0.002)


def test_transparent_padding_does_not_change_apparent_character_size_or_location():
    tight_size, tight_bounds = (220, 640), (40, 20, 180, 620)
    padded_size, padded_bounds = (920, 1140), (420, 370, 560, 970)
    tight = portrait_layout(png(silhouette(tight_size, tight_bounds)))
    padded = portrait_layout(png(silhouette(padded_size, padded_bounds)))

    assert tight.mode == padded.mode == "upper_body"
    assert visible_rect(tight, tight_size, tight_bounds) == pytest.approx(
        visible_rect(padded, padded_size, padded_bounds), abs=2
    )


def test_alpha_noise_and_isolated_opaque_speckles_do_not_determine_framing():
    image = silhouette()
    clean = portrait_layout(png(image))
    noisy = Image.new("RGBA", image.size, (255, 255, 255, 20))
    noisy.alpha_composite(image)
    noisy.putpixel((1, 1), (255, 255, 255, 255))
    noisy.putpixel((398, 798), (255, 255, 255, 255))

    assert portrait_layout(png(noisy)) == clean


def test_round_subject_remains_entirely_visible_at_message_boundary():
    image = Image.new("RGBA", (800, 800))
    ImageDraw.Draw(image).ellipse((250, 250, 549, 549), fill="white")
    layout = portrait_layout(png(image))
    left, top, right, bottom = visible_rect(layout, image.size, (250, 250, 550, 550))

    assert layout.mode == "full_body"
    assert top >= 50
    assert 435 <= bottom <= 460
    assert right - left <= 302
    assert right - left == pytest.approx(bottom - top, abs=1)


def test_wide_subject_is_fitted_without_distortion_or_cropping():
    size, bounds = (1000, 400), (100, 100, 900, 300)
    layout = portrait_layout(png(silhouette(size, bounds)))
    left, top, right, bottom = visible_rect(layout, size, bounds)

    assert layout.mode == "full_body"
    assert 0 < left < right < 960
    assert 0 < top < bottom <= 460
    assert (right - left) / (bottom - top) == pytest.approx(4, abs=0.02)
    assert right - left <= 302


@pytest.mark.parametrize("position,center", [("left", 200), ("center", 480), ("right", 760)])
def test_position_anchors_visible_subject_instead_of_transparent_image_canvas(position, center):
    size, bounds = (800, 900), (500, 60, 690, 840)
    layout = portrait_layout(png(silhouette(size, bounds)))
    left, top, right, bottom = visible_rect(layout, size, bounds, position)

    assert (left + right) / 2 == pytest.approx(center, abs=1)
    assert top >= 40
    assert bottom > 640
    assert layout.left("right") - layout.left("left") == 560


def test_full_body_override_keeps_a_tall_nonhumanoid_whole():
    image = silhouette()
    automatic = portrait_layout(png(image))
    full = portrait_layout(png(image), framing="full_body")
    _, top, _, bottom = visible_rect(full, image.size, (130, 40, 270, 760))

    assert automatic.mode == "upper_body"
    assert full.mode == "full_body"
    assert full.height < automatic.height
    assert top >= 50
    assert 435 <= bottom <= 460


def test_upper_body_override_is_available_for_a_broad_silhouette():
    size, bounds = (600, 800), (40, 40, 560, 760)
    image = silhouette(size, bounds)
    automatic = portrait_layout(png(image))
    upper = portrait_layout(png(image), framing="upper_body")
    left, top, right, bottom = visible_rect(upper, size, bounds)

    assert automatic.mode == "full_body"
    assert upper.mode == "upper_body"
    assert upper.height > automatic.height
    assert right - left <= 382
    assert 40 <= top <= 90
    assert bottom > 440


def test_opaque_image_retains_aspect_and_has_deterministic_layout():
    image = Image.new("RGB", (320, 640), "white")
    data = png(image)
    first = portrait_layout(data)

    assert first == portrait_layout(data)
    assert first.width / first.height == pytest.approx(0.5, abs=0.002)


@pytest.mark.parametrize("data", [b"", b"not an image", png(Image.new("RGBA", (20, 20)))])
def test_invalid_or_invisible_image_cannot_produce_a_layout(data):
    with pytest.raises(ValueError):
        portrait_layout(data)


def test_declared_heights_share_ground_and_preserve_ratio_with_child_shoulders_visible():
    # Generated adult and child images both fill their canvas. Their apparent
    # stage heights must follow the character settings rather than pixel height.
    size, bounds = (768, 1024), (240, 20, 530, 1000)
    image = png(silhouette(size, bounds))
    layouts = portrait_layouts({
        "adult": PortraitSource(image, "upper_body", 175),
        "child": PortraitSource(image, "upper_body", 105),
    })
    adult = visible_rect(layouts["adult"], size, bounds)
    child = visible_rect(layouts["child"], size, bounds)
    adult_height, child_height = adult[3] - adult[1], child[3] - child[1]

    assert adult[1] == pytest.approx(56, abs=1)
    assert adult[3] == pytest.approx(child[3], abs=1)
    assert adult_height / child_height == pytest.approx(175 / 105, abs=0.005)
    assert child[1] > adult[1] + 100
    assert child[1] + 0.3 * child_height <= 421
    for layout in layouts.values():
        assert layout.width / layout.height == pytest.approx(768 / 1024, abs=0.002)


def test_height_group_ignores_canvas_resolution_padding_and_member_order():
    tight_size, tight_bounds = (220, 640), (40, 20, 180, 620)
    padded_size, padded_bounds = (920, 1140), (420, 370, 560, 970)
    larger_size, larger_bounds = (440, 1280), (80, 40, 360, 1240)
    sources = {
        "tight": PortraitSource(png(silhouette(tight_size, tight_bounds)), "upper_body", 160),
        "padded": PortraitSource(png(silhouette(padded_size, padded_bounds)), "upper_body", 160),
        "larger": PortraitSource(png(silhouette(larger_size, larger_bounds)), "upper_body", 160),
    }
    layouts = portrait_layouts(sources)
    reference = visible_rect(layouts["tight"], tight_size, tight_bounds)

    assert layouts == portrait_layouts(dict(reversed(list(sources.items()))))
    assert visible_rect(layouts["padded"], padded_size, padded_bounds) == pytest.approx(
        reference, abs=2
    )
    assert visible_rect(layouts["larger"], larger_size, larger_bounds) == pytest.approx(
        reference, abs=2
    )


def test_unknown_explicit_humanoid_uses_170_cm_on_same_ground():
    size, bounds = (400, 800), (130, 40, 270, 760)
    image = png(silhouette(size, bounds))
    layouts = portrait_layouts({
        "child": PortraitSource(image, "upper_body", 100),
        "unknown": PortraitSource(image, "upper_body"),
    })
    child = visible_rect(layouts["child"], size, bounds)
    unknown = visible_rect(layouts["unknown"], size, bounds)

    assert unknown[3] == pytest.approx(child[3], abs=1)
    assert (unknown[3] - unknown[1]) / (child[3] - child[1]) == pytest.approx(1.7, abs=0.005)


def test_legacy_auto_and_full_body_are_not_inferred_as_humanoids_in_mixed_chapter():
    image = png(silhouette())
    group_only = portrait_layouts({"person": PortraitSource(image, "upper_body", 175)})
    sources = {
        "person": PortraitSource(image, "upper_body", 175),
        "auto": PortraitSource(image),
        "auto_height": PortraitSource(image, "auto", 300),
        "animal": PortraitSource(image, "full_body", 500),
    }
    layouts = portrait_layouts(sources)

    assert layouts["person"] == group_only["person"]
    assert layouts["auto"] == layouts["auto_height"] == portrait_layout(image)
    assert layouts["animal"] == portrait_layout(image, "full_body")


def test_chapter_without_height_metadata_preserves_each_legacy_layout():
    image = png(silhouette())
    sources = {
        "auto": PortraitSource(image),
        "upper": PortraitSource(image, "upper_body"),
        "full": PortraitSource(image, "full_body"),
    }
    assert portrait_layouts(sources) == {
        key: portrait_layout(value.image, value.framing) for key, value in sources.items()
    }
    assert portrait_layouts({}) == {}


def test_broad_pose_constrains_whole_group_without_losing_height_ratio():
    narrow_size, narrow_bounds = (400, 800), (130, 40, 270, 760)
    broad_size, broad_bounds = (1000, 800), (20, 40, 980, 760)
    layouts = portrait_layouts({
        "narrow": PortraitSource(png(silhouette(narrow_size, narrow_bounds)), "upper_body", 180),
        "broad": PortraitSource(png(silhouette(broad_size, broad_bounds)), "upper_body", 120),
    })
    narrow = visible_rect(layouts["narrow"], narrow_size, narrow_bounds)
    broad = visible_rect(layouts["broad"], broad_size, broad_bounds)

    assert broad[2] - broad[0] <= 381
    assert narrow[3] == pytest.approx(broad[3], abs=1)
    assert (narrow[3] - narrow[1]) / (broad[3] - broad[1]) == pytest.approx(1.5, abs=0.005)
    assert broad[1] + 0.3 * (broad[3] - broad[1]) <= 421


def normalized_bounds(size, bounds):
    width, height = size
    left, top, right, bottom = bounds
    return {"left": left / width, "top": top / height,
            "right": right / width, "bottom": bottom / height}


def test_body_reference_keeps_hats_and_umbrellas_from_shrinking_or_recentering_the_cast():
    size, body = (1000, 1200), (560, 280, 740, 1140)
    person = silhouette(size, body)
    accessorized = person.copy()
    # A large off-center prop extends well above and beside the body.
    ImageDraw.Draw(accessorized).rectangle((30, 20, 940, 330), fill="brown")
    reference = normalized_bounds(size, body)
    other = PortraitSource(png(silhouette()), "upper_body", 172)
    clean = portrait_layouts({
        "person": PortraitSource(png(person), "upper_body", 158), "other": other,
    })
    corrected = portrait_layouts({
        "person": PortraitSource(png(accessorized), "upper_body", 158, reference), "other": other,
    })
    uncorrected = portrait_layouts({
        "person": PortraitSource(png(accessorized), "upper_body", 158), "other": other,
    })
    assert corrected == clean
    assert corrected["person"].height > uncorrected["person"].height
    assert corrected["other"].height > uncorrected["other"].height
    for slot, center in (("left", 200), ("center", 480), ("right", 760)):
        rect = visible_rect(corrected["person"], size, body, slot)
        assert (rect[0] + rect[2]) / 2 == pytest.approx(center, abs=1)
    rect = visible_rect(corrected["person"], size, body)
    other_rect = visible_rect(corrected["other"], (400, 800), (130, 40, 270, 760))
    assert rect[3] == pytest.approx(other_rect[3], abs=1)
    assert (rect[3] - rect[1]) / (other_rect[3] - other_rect[1]) == pytest.approx(158 / 172, abs=.003)
    assert corrected["person"].width / corrected["person"].height == pytest.approx(1000 / 1200, abs=.002)
    # The original canvas, including props above the viewport, remains present.
    assert corrected["person"].top < 0


def test_reference_coordinates_are_independent_of_image_resolution_and_legacy_framing():
    size, body = (500, 800), (160, 180, 340, 760)
    image = silhouette(size, body)
    ImageDraw.Draw(image).rectangle((0, 10, 499, 240), fill="brown")
    bounds = normalized_bounds(size, body)
    bigger = image.resize((1000, 1600), Image.Resampling.NEAREST)
    for framing in ("auto", "upper_body", "full_body"):
        small = portrait_layout(png(image), framing, bounds)
        large = portrait_layout(png(bigger), framing, bounds)
        assert visible_rect(small, size, body) == pytest.approx(
            visible_rect(large, bigger.size, tuple(value * 2 for value in body)), abs=2,
        )


@pytest.mark.parametrize("bounds", [
    {"left": .4, "top": .2, "right": .3, "bottom": .8},
    {"left": .2, "top": .2, "right": 1.1, "bottom": .8},
    {"left": .2, "top": .2, "right": .8, "bottom": .21},
    {"left": True, "top": .2, "right": .8, "bottom": .8},
    {"left": .2, "top": float("nan"), "right": .8, "bottom": .8},
])
def test_invalid_body_reference_is_rejected_even_for_direct_layout_calls(bounds):
    with pytest.raises(ValueError):
        portrait_layouts({"person": PortraitSource(png(silhouette()), "upper_body", 170, bounds)})


def test_compiler_uses_body_reference_on_entry_and_move_without_changing_image_bytes():
    script, assets = demo_content()
    data = script.model_dump(mode="json")
    size, body = (1000, 1200), (560, 280, 740, 1140)
    image = silhouette(size, body)
    ImageDraw.Draw(image).rectangle((30, 20, 940, 330), fill="brown")
    raw = png(image)
    reference = normalized_bounds(size, body)
    character = data["characters"][0]
    character.update(framing="upper_body", height_cm=158, body_bounds=reference)
    assets[character["image_asset_id"]] = raw
    expected = portrait_layouts({character["id"]: PortraitSource(raw, "upper_body", 158, reference)})[character["id"]]
    data["directions"].append({"id": "move_ref", "kind": "position", "utterance_id": "line_004",
                               "character_id": character["id"], "position": "right"})
    scenario = compile_scenario(Script.model_validate(data), assets)
    entrance = next(line for line in scenario.splitlines()
                    if line.startswith('[chara_show name="ad_' + character["id"] + '"'))
    assert f'width="{expected.width}" height="{expected.height}"' in entrance
    assert f'top="{expected.top}"' in entrance
    assert f'left="{expected.left("right")}"' in next(
        line for line in scenario.splitlines() if line.startswith('[chara_move name="ad_' + character["id"] + '"'))
    assert assets[character["image_asset_id"]] == raw


@pytest.mark.parametrize("height", [0, -1, float("nan"), float("inf"), True])
def test_invalid_physical_height_is_rejected(height):
    with pytest.raises(ValueError, match="finite positive"):
        portrait_layouts({"character": PortraitSource(png(silhouette()), "upper_body", height)})


def test_compiler_uses_chapter_height_group_for_every_entrance_and_move():
    script, assets = demo_content()
    data = script.model_dump(mode="json")
    image = png(silhouette())
    for character, height in zip(data["characters"], (175, 105), strict=True):
        character.update(framing="upper_body", height_cm=height)
        assets[character["image_asset_id"]] = image
    data["directions"].extend([
        {"id": "exit_aki", "kind": "exit", "utterance_id": "line_002", "character_id": "aki"},
        {"id": "return_aki", "kind": "enter", "utterance_id": "line_003",
         "character_id": "aki", "position": "center"},
        {"id": "move_aki", "kind": "position", "utterance_id": "line_004",
         "character_id": "aki", "position": "left"},
    ])
    scenario = compile_scenario(Script.model_validate(data), assets)
    shows = [dict(re.findall(r'(\w+)="([^"]*)"', line))
             for line in scenario.splitlines() if line.startswith("[chara_show ")]
    adult, child, adult_return = shows

    assert int(adult["height"]) / int(child["height"]) == pytest.approx(175 / 105, abs=0.005)
    assert int(child["top"]) > int(adult["top"])
    assert {key: adult[key] for key in ("height", "width", "top")} == {
        key: adult_return[key] for key in ("height", "width", "top")
    }
    assert int(adult_return["left"]) - int(adult["left"]) == 280
    assert f'[chara_move name="ad_aki" left="{adult["left"]}"' in scenario


def test_player_storage_namespace_changes_for_physical_height_layout_revision():
    script_bytes = b'{"id":"unchanged-chapter"}'
    config = player_config(script_bytes).decode()
    old = hashlib.sha256(b"portrait-layout-v2\0" + script_bytes).hexdigest()
    current = hashlib.sha256(b"portrait-layout-v3-player-m5\0" + script_bytes).hexdigest()

    assert f"auto_drama_{current}" in config
    assert old not in config
