"""Scene entry is one trusted stage transaction; legacy source stays unchanged."""

import io
import json
import zipfile

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, compile_scenario, demo_content
from packages.tyrano_export.compiler import scene_stage_data
from packages.tyrano_export.player import player_config


def transition_script():
    original, assets = demo_content()
    value = original.model_dump(mode="json")
    value["scene_transitions"] = [
        {"id": "entry_one", "utterance_id": "line_001", "visual": "dissolve"},
        {"id": "entry_two", "utterance_id": "line_003", "visual": "fade"},
    ]
    value["directions"].extend([
        {"id": "reset_aki", "kind": "exit", "utterance_id": "line_003", "character_id": "aki"},
        {"id": "reset_ren", "kind": "exit", "utterance_id": "line_003", "character_id": "ren"},
        {"id": "return_aki", "kind": "enter", "utterance_id": "line_003", "character_id": "aki", "position": "left"},
        {"id": "return_ren", "kind": "enter", "utterance_id": "line_003", "timing": "start", "character_id": "ren", "position": "right"},
        {"id": "entry_black", "kind": "blackout", "utterance_id": "line_003", "timing": "start", "duration_ms": 500},
        {"id": "mid_black", "kind": "blackout", "utterance_id": "line_004", "timing": "after", "duration_ms": 300},
        {"id": "mid_position", "kind": "position", "utterance_id": "line_002", "character_id": "ren", "position": "center", "duration_ms": 250},
    ])
    return Script.model_validate(value), assets


def test_entry_reset_and_reenter_fold_into_the_final_stage_without_flicker():
    script, assets = transition_script()
    stages = scene_stage_data(script, assets)
    first, second = stages["scenes"]
    assert stages["schema_version"] == 1
    assert first["background"] == second["background"] == {"storage": "station.png", "position": ""}
    assert first["characters"] == second["characters"]
    assert [value["id"] for value in first["characters"]] == ["aki", "ren"]
    assert all(value["width"] > 0 and value["height"] > 0 for value in first["characters"])


def test_stage_collection_includes_mid_scene_state_and_after_directions():
    script, assets = transition_script()
    value = script.model_dump(mode="json")
    value["scene_transitions"][1]["utterance_id"] = "line_004"
    value["directions"] = [row for row in value["directions"] if not row["id"].startswith(("reset_", "return_"))]
    value["directions"].append({"id": "leave_aki", "kind": "exit", "utterance_id": "line_003", "timing": "after", "character_id": "aki"})
    folded = scene_stage_data(Script.model_validate(value), assets)["scenes"]
    assert [row["id"] for row in folded[1]["characters"]] == ["ren"]
    assert folded[0]["characters"][1]["left"] != folded[1]["characters"][0]["left"]


def test_compiler_waits_on_one_scene_tag_before_speaker_voice_and_body():
    script, assets = transition_script()
    scenario = compile_scenario(script, assets)
    for cue in script.scene_transitions:
        start = scenario.index("*utterance_" + cue.utterance_id)
        boundary = scenario.index(f'[ad_transition cue="{cue.id}"]', start)
        assert start < boundary < scenario.index("[chara_ptext", boundary)
    assert "[bg " not in scenario and "[chara_show" not in scenario and "[chara_hide" not in scenario
    assert '[chara_move name="ad_ren"' in scenario
    assert '[wait time="450"]' in scenario
    assert scenario.count("[mask ") == 1 and '[mask color="0x000000" time="300"]' in scenario


def test_entry_music_is_coordinated_by_scene_tag_instead_of_separate_immediate_tag():
    script, assets = transition_script()
    value = script.model_dump(mode="json")
    value["music_cues"] = [{"id": "keep_music", "utterance_id": "line_003", "action": "continue"}]
    script = Script.model_validate(value)
    scenario = compile_scenario(script, assets)
    assert '[ad_music cue="keep_music"]' not in scenario
    assert scenario.endswith('[s]\n')
    assert '[ad_music_stop' not in scenario
    assert scene_stage_data(script, assets)["scenes"][1]["music_cue_id"] == "keep_music"


def test_immutable_zip_contains_typed_stage_json_and_owned_runtime():
    script, assets = transition_script()
    first = compile_bundle(script, assets)
    assert first == compile_bundle(script, dict(reversed(list(assets.items()))))
    with zipfile.ZipFile(io.BytesIO(first)) as archive:
        assert json.loads(archive.read("data/others/auto_drama_stages.json")) == scene_stage_data(script, assets)
        assert "data/others/auto_drama_transitions.js" in archive.namelist()
        assert b"auto_drama_transitions.js" in archive.read("index.html")
        for asset in script.assets:
            folder = "bgimage" if asset.kind == "background" else "fgimage"
            assert archive.read(f"data/{folder}/{asset.filename}") == assets[asset.id]


def test_legacy_scripts_use_native_tags_and_unchanged_save_identity():
    original, assets = demo_content()
    value = original.model_dump(mode="json")
    explicit = Script.model_validate({**value, "scene_transitions": []})
    assert original.model_dump_json() == explicit.model_dump_json()
    assert player_config(original.model_dump_json().encode()) == player_config(explicit.model_dump_json().encode())
    scenario = compile_scenario(explicit, assets)
    assert "[ad_transition" not in scenario and '[bg storage="station.png" time="0"]' in scenario
    with zipfile.ZipFile(io.BytesIO(compile_bundle(explicit, assets))) as archive:
        assert "data/others/auto_drama_stages.json" not in archive.namelist()


def test_chapter_end_keeps_track_without_reusing_last_scene_fade_duration():
    script, assets = transition_script()
    value = script.model_dump(mode="json")
    value["music_cues"] = [{"id": "stop_music", "utterance_id": "line_003", "action": "stop"}]
    value["scene_transitions"][0]["music_fade_out_ms"] = 100
    value["scene_transitions"][1]["music_fade_out_ms"] = 700
    value["scene_transitions"].reverse()
    scenario = compile_scenario(Script.model_validate(value), assets)
    assert scenario.endswith('[s]\n')
    assert '[ad_music_stop' not in scenario
    assert scene_stage_data(Script.model_validate(value), assets)["scenes"][1]["music_cue_id"] == "stop_music"
