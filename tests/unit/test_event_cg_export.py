"""Adopted CG intervals preserve dialogue waits and the underlying normal stage."""

import copy
import hashlib
import io
import json
import zipfile

import pytest

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, compile_scenario, demo_content
from packages.tyrano_export.compiler import event_cg_frames, scene_stage_data
from packages.tyrano_export.demo import _png
from packages.tyrano_export.player import player_config


def cg_content():
    original, assets = demo_content()
    value = original.model_dump(mode="json")
    for name, kind, color in (("cg_base", "event_cg", (180, 70, 90, 255)),
                              ("cg_variant", "event_cg", (90, 70, 180, 255)),
                              ("new_background", "background", (40, 90, 60, 255))):
        assets[name] = _png(96, 54, lambda x, y, color=color: color)
        value["assets"].append({"id": name, "kind": kind, "artifact_id": name,
                                "filename": name + ".png", "sha256": hashlib.sha256(assets[name]).hexdigest()})
    value["event_cg_segments"] = [{"id": "event_one", "start_utterance_id": "line_002",
                                   "end_utterance_id": "line_004", "base_asset_id": "cg_base",
                                   "variants": [{"id": "variant_one", "utterance_id": "line_003",
                                                 "asset_id": "cg_variant"}]}]
    value["directions"].extend([
        {"id": "inside_bg", "kind": "background", "utterance_id": "line_002", "asset_id": "new_background"},
        {"id": "inside_move", "kind": "position", "utterance_id": "line_003", "character_id": "ren", "position": "center"},
        {"id": "inside_exit", "kind": "exit", "utterance_id": "line_003", "timing": "after", "character_id": "aki"},
    ])
    return Script.model_validate(value), assets


def test_frames_switch_before_lines_and_end_at_exclusive_boundary():
    script, _ = cg_content()
    frames = event_cg_frames(script)
    assert set(frames) == {"line_002", "line_003"}
    assert frames["line_002"] == {"segment_id": "event_one", "variant_id": None, "asset_id": "cg_base"}
    assert frames["line_003"]["asset_id"] == "cg_variant"


def test_cg_end_restores_current_background_position_and_cast_not_entry_snapshot():
    script, assets = cg_content()
    first, changed, after, restored = scene_stage_data(script, assets)["scenes"]
    assert first["event_cg"]["storage"] == "cg_base.png"
    assert changed["event_cg"]["storage"] == "cg_variant.png"
    assert after["timing"] == "after" and [character["id"] for character in after["characters"]] == ["ren"]
    assert restored["utterance_id"] == "line_004" and restored["event_cg"] is None
    assert restored["background"]["storage"] == "new_background.png"
    assert [character["id"] for character in restored["characters"]] == ["ren"]
    assert restored["characters"][0]["left"] != first["characters"][1]["left"]


def test_hidden_stage_commands_do_not_leak_and_last_cg_line_keeps_click_wait():
    script, assets = cg_content()
    scenario = compile_scenario(script, assets)
    assert '[bg storage="new_background.png"' not in scenario
    assert "[chara_move" not in scenario and "[chara_hide" not in scenario
    assert scenario.index('[ad_transition cue="cg:line_002"]') < scenario.index('[ad_say id="line_002"]')
    last = scenario.index('[ad_say id="line_003"]')
    assert last < scenario.index("[ad_wait]", last) < scenario.index('[ad_transition cue="cg:line_004"]')


def test_scene_entry_cg_uses_one_transition_and_one_music_barrier():
    script, assets = cg_content()
    value = script.model_dump(mode="json")
    value["scene_transitions"] = [{"id": "entry", "utterance_id": "line_002", "visual": "fade"}]
    value["music_cues"] = [{"id": "continue", "utterance_id": "line_002", "action": "continue"}]
    script = Script.model_validate(value)
    scenario = compile_scenario(script, assets)
    assert scenario.count('[ad_transition cue="entry"]') == 1
    assert '[ad_transition cue="cg:line_002"]' not in scenario
    assert '[ad_music cue="continue"]' not in scenario
    assert scene_stage_data(script, assets)["scenes"][0]["music_cue_id"] == "continue"


def test_adjacent_cgs_switch_directly_and_final_null_end_retains_picture():
    script, assets = cg_content()
    value = script.model_dump(mode="json")
    value["event_cg_segments"].append({"id": "event_two", "start_utterance_id": "line_004",
                                       "end_utterance_id": None, "base_asset_id": "cg_base"})
    script = Script.model_validate(value)
    stages = scene_stage_data(script, assets)["scenes"]
    assert len(stages) == 4 and stages[-1]["event_cg"]["segment_id"] == "event_two"
    assert compile_scenario(script, assets).count('[ad_transition cue="cg:line_004"]') == 1


@pytest.mark.parametrize("patch", [
    {"start_utterance_id": "missing"}, {"end_utterance_id": "missing"},
    {"end_utterance_id": "line_001"}, {"end_utterance_id": "line_002"},
    {"base_asset_id": "station"}, {"base_asset_id": "missing"},
    {"variants": [{"id": "v", "utterance_id": "line_002", "asset_id": "cg_variant"}]},
    {"variants": [{"id": "v", "utterance_id": "line_004", "asset_id": "cg_variant"}]},
    {"variants": [{"id": "v", "utterance_id": "missing", "asset_id": "cg_variant"}]},
    {"variants": [{"id": "v", "utterance_id": "line_003", "asset_id": "station"}]},
    {"variants": [{"id": "v", "utterance_id": "line_003", "asset_id": "cg_base"},
                  {"id": "other", "utterance_id": "line_003", "asset_id": "cg_variant"}]},
])
def test_invalid_cg_references_and_intervals_are_rejected(patch):
    script, _ = cg_content()
    value = script.model_dump(mode="json")
    value["event_cg_segments"][0].update(patch)
    with pytest.raises(ValueError):
        Script.model_validate(value)


def test_overlapping_intervals_are_rejected_even_when_unordered():
    script, _ = cg_content()
    value = script.model_dump(mode="json")
    other = {**copy.deepcopy(value["event_cg_segments"][0]), "id": "other", "end_utterance_id": None}
    value["event_cg_segments"].insert(0, other)
    with pytest.raises(ValueError, match="nonoverlapping"):
        Script.model_validate(value)


def test_zip_contains_cg_hash_manifest_and_deterministic_stage_data():
    script, assets = cg_content()
    bundle = compile_bundle(script, assets)
    assert bundle == compile_bundle(script, dict(reversed(list(assets.items()))))
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        assert archive.read("data/cgimage/cg_base.png") == assets["cg_base"]
        manifest = json.loads(archive.read("manifest.json"))
        assert manifest["files"]["data/cgimage/cg_variant.png"]["sha256"] == hashlib.sha256(assets["cg_variant"]).hexdigest()
        assert json.loads(archive.read("data/others/auto_drama_stages.json")) == scene_stage_data(script, assets)


def test_legacy_empty_cg_has_identical_serialization_save_identity_and_scenario():
    original, assets = demo_content()
    empty = Script.model_validate({**original.model_dump(mode="json"), "event_cg_segments": []})
    assert empty.model_dump_json() == original.model_dump_json()
    assert player_config(empty.model_dump_json().encode()) == player_config(original.model_dump_json().encode())
    assert compile_scenario(empty, assets) == compile_scenario(original, assets)
    with zipfile.ZipFile(io.BytesIO(compile_bundle(empty, assets))) as archive:
        assert "data/others/auto_drama_stages.json" not in archive.namelist()
