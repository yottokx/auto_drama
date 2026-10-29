"""The stage each line shows while its text is up, as the adjustment preview reads it."""
from __future__ import annotations

import hashlib

from packages.contracts.script import Script
from packages.tyrano_export.compiler import entry_states, line_stages
from services.coordinator.script_adjustments import apply_script_edits, pauses_overrun
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


def direction_edit(direction_id, **fields):
    return {"production_id": "chapter", "direction_id": direction_id, "disabled": False,
            "expression": None, "duration_ms": None, "original": None, **fields}


def test_edits_change_words_faces_and_waits_and_leave_disabled_directions_out():
    script, assets = with_faces([face("line_002", "smile")])
    pause = next(row for row in script.directions if row.id == "pause_ren")
    edits = {"lines": [{"production_id": "chapter", "utterance_id": "line_003", "display_text": "やっぱり行こう。",
                        "spoken_text": "やっぱりいこう。", "voice_emotion": "happy", "delivery": ""}],
             "directions": [direction_edit("face_1", expression="angry"),
                            direction_edit("pause_ren", disabled=True, original=pause.model_dump(mode="json"))]}
    edited = apply_script_edits(script, edits, "chapter")
    line = next(row for row in edited.utterances if row.id == "line_003")
    assert (line.display_text, line.spoken_text, line.voice_emotion) == ("やっぱり行こう。", "やっぱりいこう。", "happy")
    assert next(row for row in edited.directions if row.id == "face_1").expression == "angry"
    assert "pause_ren" not in {row.id for row in edited.directions}
    # Another chapter's edits leave this one as it is.
    assert apply_script_edits(script, edits, "elsewhere") is script
    # The edited script stages the edited face.
    assert storage(line_stages(edited, assets), "line_002") == "aki_sprite.png"


def test_a_direction_left_out_by_an_earlier_edition_comes_back_when_turned_on():
    script, _assets = with_faces([])
    pause = next(row for row in script.directions if row.id == "pause_ren").model_dump(mode="json")
    published = apply_script_edits(script, {"directions": [direction_edit("pause_ren", disabled=True, original=pause)]}, "chapter")
    assert "pause_ren" not in {row.id for row in published.directions}
    restored = apply_script_edits(published, {"directions": [direction_edit("pause_ren", duration_ms=900, original=pause)]}, "chapter")
    assert next(row for row in restored.directions if row.id == "pause_ren").duration_ms == 900


def test_a_line_may_not_wait_longer_than_the_staging_step_would():
    script, _assets = with_faces([])
    script, _assets = with_rows(script, _assets, directions=[
        {"kind": "pause", "utterance_id": "line_004", "timing": "after", "duration_ms": 500}])
    assert not pauses_overrun(script, {"directions": [direction_edit("pause_ren", duration_ms=1000)]}, "chapter")
    assert pauses_overrun(script, {"directions": [direction_edit("pause_ren", duration_ms=1100)]}, "chapter")
    # A pause turned off no longer counts.
    assert not pauses_overrun(script, {"directions": [direction_edit("pause_ren", disabled=True),
                                                      direction_edit("extra_1", duration_ms=1500)]}, "chapter")


def test_a_focus_at_the_start_of_a_line_lights_someone_else():
    script, assets = with_faces([])
    script, assets = with_rows(script, assets, directions=[
        {"kind": "focus", "utterance_id": "line_002", "timing": "start", "character_id": "aki"},
        {"kind": "focus", "utterance_id": "line_003", "timing": "before", "character_id": "ren"}])
    lines = line_stages(script, assets)["lines"]
    # One given before the line is replaced by the speaker's own light.
    assert lines["line_002"]["lit"] == "aki" and lines["line_003"]["lit"] == "aki"
