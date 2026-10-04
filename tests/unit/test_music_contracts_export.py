import hashlib
import io
import json
import zipfile

import pytest
from pydantic import ValidationError

from packages.contracts import MusicCue, Script
from packages.tyrano_export import compile_bundle, compile_scenario, demo_content, validate_bundle
from packages.tyrano_export.compiler import canonical_json
from packages.tyrano_export.player import player_config


def music_script():
    script, assets = demo_content()
    value = script.model_dump(mode="json")
    assets["scene_music"] = b"ID3adopted music fixture"
    value["assets"].append({"id": "scene_music", "kind": "music", "artifact_id": "adopted-music",
                            "filename": "scene_music.mp3",
                            "sha256": hashlib.sha256(assets["scene_music"]).hexdigest()})
    value["music_cues"] = [{"id": "scene_music_start", "utterance_id": script.utterances[0].id,
                            "action": "play", "asset_id": "scene_music",
                            "loop_start_seconds": 12.5, "loop_end_seconds": 90.0}]
    return Script.model_validate(value), assets


def test_old_scripts_keep_their_serialized_contract_when_music_is_absent():
    script, _assets = demo_content()
    assert script.music_cues == []
    assert "music_cues" not in script.model_dump(mode="json")
    assert "music_cues" not in json.loads(script.model_dump_json())
    assert Script.model_validate_json(script.model_dump_json()) == script


def test_music_revisions_get_separate_saves_and_empty_music_preserves_old_namespace():
    old, _assets = demo_content()
    old_data = old.model_dump(mode="json")
    explicit_empty = Script.model_validate({**old_data, "music_cues": []})
    old_config = player_config(canonical_json(old_data))
    assert player_config(canonical_json(explicit_empty.model_dump(mode="json"))) == old_config
    adopted, _assets = music_script()
    adopted_data = adopted.model_dump(mode="json")
    adopted_config = player_config(canonical_json(adopted_data))
    assert adopted_config != old_config
    adjusted = json.loads(json.dumps(adopted_data))
    adjusted["music_cues"][0]["volume"] = 0.25
    assert player_config(canonical_json(Script.model_validate(adjusted).model_dump(mode="json"))) != adopted_config


@pytest.mark.parametrize("action", ["stop", "continue"])
def test_stop_and_continue_are_backend_neutral_without_assets(action):
    cue = MusicCue(id="scene_boundary", utterance_id="line_001", action=action)
    assert cue.volume == 0.35
    assert cue.model_dump(mode="json") == {"id": "scene_boundary", "utterance_id": "line_001",
                                           "action": action, "volume": 0.35}


@pytest.mark.parametrize("mutation", [
    {"asset_id": None}, {"loop_start_seconds": -1}, {"loop_end_seconds": 12.5},
    {"loop_end_seconds": None}, {"loop_start_seconds": None}, {"loop_end_seconds": float("nan")},
    {"volume": -0.1}, {"volume": 1.1}, {"volume": True}, {"loop_start_seconds": "12.5"},
    {"action": "javascript"}, {"url": "https://unadopted.invalid/music.mp3"},
])
def test_music_cue_rejects_invalid_or_executable_parameters(mutation):
    value = music_script()[0].music_cues[0].model_dump(mode="json")
    value.update(mutation)
    with pytest.raises(ValidationError):
        MusicCue.model_validate(value)


@pytest.mark.parametrize("mutation", [
    {"action": "stop"}, {"action": "continue"}, {"utterance_id": "unknown_line"},
    {"asset_id": "station"}, {"asset_id": "missing_music"},
])
def test_script_music_cues_require_correct_action_anchor_and_adopted_asset(mutation):
    value = music_script()[0].model_dump(mode="json")
    value["music_cues"][0].update(mutation)
    with pytest.raises(ValidationError):
        Script.model_validate(value)


def test_one_music_action_per_stable_utterance():
    value = music_script()[0].model_dump(mode="json")
    value["music_cues"].append({"id": "conflicting_music", "utterance_id": value["utterances"][0]["id"],
                                "action": "stop"})
    with pytest.raises(ValidationError, match="one music cue"):
        Script.model_validate(value)


@pytest.mark.parametrize("filename", ["music.wav", "../music.mp3", "music.mp3?token=private", "CON.mp3"])
def test_music_assets_are_safe_mp3_files(filename):
    value = music_script()[0].model_dump(mode="json")
    value["assets"][-1]["filename"] = filename
    with pytest.raises(ValidationError):
        Script.model_validate(value)


def test_music_bundle_contains_only_adopted_mp3_cues_and_owned_player_code():
    script, assets = music_script()
    content = compile_bundle(script, assets)
    validate_bundle(content, script, assets)
    assert content == compile_bundle(script, dict(reversed(list(assets.items()))))
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        assert archive.read("data/bgm/scene_music.mp3") == assets["scene_music"]
        adopted = json.loads(archive.read("script.json"))
        assert adopted["music_cues"][0]["loop_start_seconds"] == 12.5
        assert adopted["music_cues"][0]["volume"] == 0.35
        assert "data/others/auto_drama_music.js" in archive.namelist()
        assert b'id="ad-bgm-volume"' in archive.read("index.html")
        assert not any("output-float" in path or "request.json" in path for path in archive.namelist())
        assert archive.read("data/scenario/first.ks").count(b'[ad_music cue="scene_music_start"]') == 1


def test_music_compiler_uses_cue_ids_and_retains_the_last_track_at_chapter_end():
    script, assets = music_script()
    scenario = compile_scenario(script, assets)
    anchor = scenario.index("*utterance_" + script.music_cues[0].utterance_id)
    cue = scenario.index('[ad_music cue="scene_music_start"]')
    assert anchor < cue < scenario.index("[p]", cue)
    assert scenario.endswith("[s]\n")
    assert "[ad_music_stop" not in scenario
    assert "scene_music.mp3" not in scenario
    original, images = demo_content()
    assert "[ad_music" not in compile_scenario(original, images)
