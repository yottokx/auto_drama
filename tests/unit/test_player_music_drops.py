"""A drop holds the playing track silent at its place; a resume, continue or new track ends it."""

from __future__ import annotations

import pytest

from tests.unit.test_player_music import run


@pytest.fixture
def script() -> dict:
    return {
        "utterances": [{"id": "first"}, {"id": "second"}, {"id": "third"}],
        "assets": [{"id": "a", "kind": "music", "filename": "theme-a.mp3"},
                   {"id": "b", "kind": "music", "filename": "theme-b.mp3"}],
        "music_cues": [
            {"id": "open", "utterance_id": "first", "action": "play", "asset_id": "a",
             "loop_start_seconds": 5, "loop_end_seconds": 15, "volume": 0.4},
            {"id": "hush", "utterance_id": "second", "action": "drop", "volume": 0.35},
            {"id": "back", "utterance_id": "third", "action": "resume", "volume": 0.35},
            {"id": "keep", "utterance_id": "third", "action": "continue", "volume": 0.35},
            {"id": "next", "utterance_id": "third", "action": "play", "asset_id": "b", "volume": 0.35},
        ],
    }


def dropping(*ops: dict) -> list[dict]:
    return [{"op": "unlock"}, {"op": "cue", "id": "open"}, {"op": "advance", "seconds": 3},
            {"op": "beginCue", "id": "hush", "key": "drop"}, {"op": "advance", "seconds": 0.6},
            {"op": "wait", "key": "drop"}, *ops]


def fade(result) -> dict:
    return result["contexts"][0]["gains"][1]


def test_drop_fades_out_then_holds_the_place_until_resume_fades_back_in(script):
    result = run(script, dropping(
        {"op": "snapshot", "key": "held"}, {"op": "advance", "seconds": 20}, {"op": "snapshot", "key": "later"},
        {"op": "cue", "id": "back"}, {"op": "advance", "seconds": 1}, {"op": "snapshot", "key": "back"},
    ))
    first, second = result["sources"]
    assert fade(result)["ramps"][0]["value"] == 0 and fade(result)["ramps"][0]["end"] == pytest.approx(3.6)
    assert first["stops"] == 1
    assert result["snapshots"]["held"]["dropped"] is True
    assert result["snapshots"]["held"]["position_seconds"] == pytest.approx(3.6)
    assert result["snapshots"]["later"]["position_seconds"] == pytest.approx(3.6)
    # The same track comes back from where it stopped, fading in, without another decode.
    assert second["starts"] == [[0, pytest.approx(3.6)]]
    assert fade(result)["ramps"][-1]["value"] == 1 and fade(result)["ramps"][-1]["end"] - fade(result)["ramps"][-1]["start"] == pytest.approx(1.5)
    assert "dropped" not in result["snapshots"]["back"]
    assert result["decodes"] == ["a"]


def test_next_scene_continue_brings_back_a_track_held_to_the_scene_end(script):
    result = run(script, dropping({"op": "cue", "id": "keep"}, {"op": "snapshot", "key": "kept"}))
    assert len(result["sources"]) == 2
    assert result["snapshots"]["kept"]["cue_id"] == "open"
    assert "dropped" not in result["snapshots"]["kept"]


@pytest.mark.parametrize("op", [{"op": "transition", "id": "keep"}])
def test_scene_transition_continue_also_brings_it_back(script, op):
    result = run(script, dropping(op, {"op": "snapshot", "key": "kept"}))
    assert len(result["sources"]) == 2 and "dropped" not in result["snapshots"]["kept"]


def test_a_new_track_or_a_stop_ends_the_drop(script):
    played = run(script, dropping({"op": "cue", "id": "next"}, {"op": "snapshot", "key": "next"}))
    assert played["snapshots"]["next"]["cue_id"] == "next" and "dropped" not in played["snapshots"]["next"]
    assert played["sources"][-1]["starts"] == [[0, 0]]
    stopped = run(script, dropping({"op": "stop"}, {"op": "cue", "id": "back"}, {"op": "snapshot", "key": "after"}))
    assert stopped["snapshots"]["after"] is None and len(stopped["sources"]) == 1


def test_resume_during_the_fade_out_turns_it_around_without_restarting(script):
    result = run(script, [
        {"op": "unlock"}, {"op": "cue", "id": "open"}, {"op": "advance", "seconds": 3},
        {"op": "beginCue", "id": "hush", "key": "drop"}, {"op": "advance", "seconds": 0.3},
        {"op": "cue", "id": "back"}, {"op": "advance", "seconds": 1}, {"op": "wait", "key": "drop"},
        {"op": "snapshot", "key": "back"},
    ])
    assert len(result["sources"]) == 1 and result["sources"][0]["stops"] == 0
    assert "dropped" not in result["snapshots"]["back"]
    assert result["snapshots"]["back"]["position_seconds"] == pytest.approx(4.3)


def test_skipping_drops_at_once(script):
    result = run(script, [{"op": "unlock"}, {"op": "cue", "id": "open"}, {"op": "advance", "seconds": 2},
                          {"op": "cue", "id": "hush", "options": {"fast": True}}, {"op": "snapshot", "key": "held"}])
    assert result["sources"][0]["stops"] == 1
    assert result["snapshots"]["held"]["dropped"] is True
    assert result["snapshots"]["held"]["position_seconds"] == 2


def test_a_jump_into_a_drop_loads_the_track_silent(script):
    result = run(script, [{"op": "unlock"}, {"op": "cue", "id": "open", "options": {"dropped": True}},
                          {"op": "advance", "seconds": 5}, {"op": "snapshot", "key": "held"},
                          {"op": "cue", "id": "back"}, {"op": "snapshot", "key": "back"}])
    assert result["sources"][0]["starts"] == [[0, 0]]
    assert result["snapshots"]["held"]["dropped"] is True and result["snapshots"]["held"]["position_seconds"] == 0
    assert len(result["sources"]) == 1


def test_a_saved_drop_restores_silent(script):
    result = run(script, dropping({"op": "snapshot", "key": "saved"}, {"op": "stop"},
                                  {"op": "restore", "value": "saved"}, {"op": "snapshot", "key": "restored"},
                                  {"op": "cue", "id": "back"}))
    assert result["snapshots"]["restored"] == result["snapshots"]["saved"]
    assert [len(source["starts"]) for source in result["sources"]] == [1, 1]
    assert result["sources"][1]["starts"] == [[0, pytest.approx(3.6)]]


def test_a_drop_with_nothing_playing_does_nothing(script):
    result = run(script, [{"op": "unlock"}, {"op": "cue", "id": "hush"}, {"op": "snapshot", "key": "none"},
                          {"op": "cue", "id": "back"}])
    assert result["snapshots"]["none"] is None and result["sources"] == []
