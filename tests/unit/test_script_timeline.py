"""The stage each line shows while its text is up, as the adjustment preview reads it."""
from __future__ import annotations

import hashlib

from packages.contracts.script import Script
from packages.tyrano_export.compiler import entry_states, line_stages
from tests.unit.test_expressions import face, with_faces


def storage(stages, line, character="aki"):
    return next((row["storage"] for row in stages["lines"][line]["characters"] if row["id"] == character), None)


def with_rows(script, assets, *, directions=(), cg=None, transitions=()):
    data = script.model_dump(mode="json")
    for index, row in enumerate(directions, 1):
        data["directions"].append({"id": f"extra_{index}", "timing": "before", **row})
    if cg:
        content = assets["station"]
        data["assets"].append({"id": "cg", "kind": "event_cg", "artifact_id": "fixture", "filename": "cg.png",
                               "sha256": hashlib.sha256(content).hexdigest()})
        assets["cg"] = content
        data["event_cg_segments"] = [{"id": "segment", "base_asset_id": "cg", **cg}]
    data["scene_transitions"] = [{"id": f"cue_{index}", "utterance_id": line} for index, line in enumerate(transitions, 1)]
    return Script.model_validate(data), assets


def test_a_line_shows_its_own_directions_while_jumps_still_start_before_them():
    script, assets = with_faces([face("line_002", "smile")])
    stages = line_stages(script, assets)
    assert storage(stages, "line_002") == "aki_base_smile.png"
    assert next(row["storage"] for row in entry_states(script, assets)["entries"]["line_002"]["characters"]
                if row["id"] == "aki") == "aki_sprite.png"
    assert stages["lines"]["line_001"]["background"]["storage"] == "station.png"
    # The speaker is the one left bright; a line of narration dims everyone.
    assert stages["lines"]["line_002"]["lit"] == "ren" and stages["lines"]["line_001"]["lit"] is None


def test_a_face_that_never_reaches_the_screen_is_marked_unseen():
    script, assets = with_faces([face("line_002", "smile"), face("line_003", "angry"), face("line_004", "neutral")])
    # Anger has no picture, yet it takes the smile away; the ordinary face after it changes nothing.
    assert line_stages(script, assets)["unseen"] == ["face_3"]


def test_a_face_under_an_event_cg_counts_only_if_it_outlasts_the_cg():
    script, assets = with_faces([face("line_002", "smile"), face("line_003", "neutral")])
    script, assets = with_rows(script, assets, cg={"start_utterance_id": "line_002", "end_utterance_id": "line_004"})
    stages = line_stages(script, assets)
    assert stages["lines"]["line_002"]["event_cg"]["storage"] == "cg.png"
    assert stages["lines"]["line_004"]["event_cg"] is None
    # The smile is gone before the CG ends; the ordinary face that replaced it is what line 4 shows.
    assert stages["unseen"] == ["face_1"]


def test_a_face_set_while_away_is_reset_by_the_entrance():
    stages = line_stages(*with_faces([
        {"kind": "exit", "utterance_id": "line_002", "character_id": "aki"},
        face("line_003", "smile"),
        {"kind": "enter", "utterance_id": "line_004", "character_id": "aki", "position": "left"}]))
    assert stages["unseen"] == ["face_2"] and storage(stages, "line_003") is None


def test_a_blackout_that_the_scene_transition_replaces_is_unseen():
    script, assets = with_faces([])
    script, assets = with_rows(script, assets, transitions=["line_003"], directions=[
        {"kind": "blackout", "utterance_id": "line_003", "timing": "start", "duration_ms": 300},
        {"kind": "blackout", "utterance_id": "line_004", "timing": "after", "duration_ms": 300}])
    assert line_stages(script, assets)["unseen"] == ["extra_1"]


def test_a_focus_at_the_start_of_a_line_lights_someone_else():
    script, assets = with_faces([])
    script, assets = with_rows(script, assets, directions=[
        {"kind": "focus", "utterance_id": "line_002", "timing": "start", "character_id": "aki"},
        {"kind": "focus", "utterance_id": "line_003", "timing": "before", "character_id": "ren"}])
    lines = line_stages(script, assets)["lines"]
    # One given before the line is replaced by the speaker's own light.
    assert lines["line_002"]["lit"] == "aki" and lines["line_003"]["lit"] == "aki"
