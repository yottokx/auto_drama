"""Media-free experiments still produce the existing playable Tyrano source format."""

import copy
import hashlib
import io
import json
import zipfile

import pytest
from PIL import Image

from packages.contracts import Script
from packages.contracts.m3 import NarrativeResult
from packages.tyrano_export import validate_bundle
from services.worker.generation.script_export import _safe_path, export_debug_chapter
from tests.unit.test_script_conversion import conversion_inputs


def experiment():
    narrative, snapshot, _, _ = conversion_inputs(audio=False)
    value = narrative.model_dump(mode="json")
    value.update(workflow_version=2, workflow_policy="script_continuation_v1")
    for scene in value["scenes"]:
        scene["review"] = {"policy": "not_evaluated", "passed": False, "issues": [], "events": []}
    return NarrativeResult.model_validate(value), snapshot


def test_export_preserves_script_staging_sources_and_a_valid_tyrano_bundle(tmp_path):
    narrative, snapshot = experiment()
    result = export_debug_chapter(narrative, snapshot, tmp_path)
    script = Script.model_validate_json((tmp_path / result["script_path"]).read_bytes())
    source = (tmp_path / "sources/s1.txt").read_bytes()
    assert source == narrative.scenes[0].raw_text.encode("utf-8")
    assert result["quality_acceptance"] == "not_evaluated" and not result["engine_included"]
    assert all(utterance.audio_asset_id is None for utterance in script.utterances)
    assert script.utterances[0].voice_emotion == "calm"
    assert script.utterances[0].delivery == "小声で"
    with zipfile.ZipFile(io.BytesIO((tmp_path / result["bundle_path"]).read_bytes())) as archive:
        assert "index.html" in archive.namelist()
        assert not any(name.startswith("tyrano/") for name in archive.namelist())
        assert not any(name.startswith("data/sound/") for name in archive.namelist())
        scenario = archive.read("data/scenario/first.ks").decode("utf-8")
        assert all(tag in scenario for tag in (
            "[chara_show", "[chara_hide", "[chara_move", "[chara_ptext", "[bg ", "[wait ", "[mask "))
        assert "[playse " not in scenario and "[wse]" not in scenario
        assert script.utterances[0].display_text in scenario
        assert archive.read("sources/s1.txt") == source
        assets = {asset.id: archive.read(
            f"data/{'fgimage' if asset.kind == 'character' else 'bgimage'}/{asset.filename}")
                  for asset in script.assets}
        documents = {name: archive.read(name) for name in archive.namelist()
                     if name in {"approval.json", "narrative.json"} or name.startswith("sources/")}
    manifest = validate_bundle((tmp_path / result["bundle_path"]).read_bytes(), script,
                               assets, documents=documents)
    assert result["bundle_manifest"] == manifest
    assert (tmp_path / result["player_entrypoint"]).is_file()
    for relative, metadata in result["files"].items():
        data = (tmp_path / relative).read_bytes()
        assert hashlib.sha256(data).hexdigest() == metadata["sha256"]
        assert len(data) == metadata["bytes"]
    for asset in script.assets:
        with Image.open(io.BytesIO(assets[asset.id])) as image:
            assert image.format == "PNG"
            if asset.kind == "character":
                assert image.getchannel("A").getextrema() == (0, 255)


def test_original_media_requests_remain_distinct_from_test_images(tmp_path):
    narrative, snapshot = experiment()
    export_debug_chapter(narrative, snapshot, tmp_path)
    requirements = json.loads((tmp_path / "asset-requirements.json").read_text("utf-8"))
    assert not requirements["placeholder_assets_are_production_assets"]
    rows = requirements["requirements"]
    backgrounds = [row for row in rows if row["kind"] == "m3_background"]
    assert backgrounds[0]["descriptor"]["location"] == narrative.locations[0].model_dump(mode="json")
    assert backgrounds[0]["resolution"] == "fixed_test_image"
    voices = [row for row in rows if row["kind"] == "m3_voice_clone"]
    assert voices[0]["descriptor"]["dialogue_text"] == narrative.scenes[0].utterances[0].spoken_text
    assert voices[0]["descriptor"]["delivery"] == "小声で"
    assert all(row["resolution"] == "not_generated" for row in voices)
    assert all("debug_asset_id" not in row for row in voices)


def test_multiple_scenes_keep_background_changes_and_source_order(tmp_path):
    narrative, snapshot = experiment()
    value = narrative.model_dump(mode="json")
    second = copy.deepcopy(value["scenes"][0])
    second["id"] = second["plan"]["id"] = "s2"
    second["plan"]["location_id"] = "gate_night"
    for utterance in second["utterances"]:
        utterance["id"] = utterance["id"].replace("s1-", "s2-")
    for direction in second["directions"]:
        direction["id"] = direction["id"].replace("s1-", "s2-")
        direction["utterance_id"] = direction["utterance_id"].replace("s1-", "s2-")
    value["scenes"].append(second)
    value["locations"].append({**value["locations"][0], "id": "gate_night", "time_of_day": "夜"})
    result = export_debug_chapter(NarrativeResult.model_validate(value), snapshot, tmp_path)
    script = Script.model_validate_json((tmp_path / result["script_path"]).read_bytes())
    backgrounds = [row for row in script.directions if row.kind == "background"]
    assert len(backgrounds) == 2 and backgrounds[0].asset_id != backgrounds[1].asset_id
    assert [row.id for row in script.utterances] == [
        row["id"] for scene in value["scenes"] for row in scene["utterances"]]
    assert (tmp_path / "sources/s2.txt").read_bytes() == second["raw_text"].encode("utf-8")


def test_repeated_export_is_identical_and_keeps_engine_outside_the_result(tmp_path):
    narrative, snapshot = experiment()
    first = export_debug_chapter(narrative, snapshot, tmp_path / "first")
    second = export_debug_chapter(narrative, snapshot, tmp_path / "second")
    replay = export_debug_chapter(narrative, snapshot, tmp_path / "first")
    assert first == second == replay
    assert not any(path.name == "tyrano" for path in tmp_path.rglob("*"))
    assert not (tmp_path / "first/story.html").exists()


@pytest.mark.parametrize("relative", ["../outside.txt", "player/../../../outside.txt"])
def test_export_paths_cannot_escape_the_selected_directory(tmp_path, relative):
    with pytest.raises(ValueError, match="leaves its output"):
        _safe_path(tmp_path.resolve(), relative)
