"""Scene presentation metadata cannot rewrite dialogue or contain engine commands."""

import copy

import pytest
from pydantic import ValidationError

from packages.contracts import SceneTransitionCue, SceneTransitionSpec, Script
from packages.tyrano_export import demo_content


def test_transition_metadata_preserves_every_dialogue_and_direction():
    original, _ = demo_content()
    value = original.model_dump(mode="json")
    preserved = copy.deepcopy(value)
    value["scene_transitions"] = [{"id": "scene_1", "utterance_id": original.utterances[0].id,
                                   "visual": "fade", "duration_ms": 500}]
    revised = Script.model_validate(value)
    assert revised.utterances == original.utterances
    assert revised.directions == original.directions
    assert revised.characters == original.characters
    assert revised.assets == original.assets
    serialized = revised.model_dump(mode="json")
    serialized.pop("scene_transitions")
    assert serialized == preserved
    assert Script.model_validate_json(revised.model_dump_json()) == revised


def test_legacy_empty_transition_metadata_does_not_change_script_bytes():
    original, _ = demo_content()
    old_bytes = original.model_dump_json()
    assert "scene_transitions" not in original.model_dump(mode="json")
    explicit = Script.model_validate({**original.model_dump(mode="json"), "scene_transitions": []})
    assert explicit.model_dump_json() == old_bytes


@pytest.mark.parametrize("patch", [
    {"visual": "javascript"}, {"duration_ms": -1}, {"duration_ms": 3001},
    {"duration_ms": True}, {"duration_ms": "500"}, {"duration_ms": 0.5},
    {"music_fade_out_ms": -1}, {"music_fade_in_ms": 5001},
    {"music_fade_in_ms": float("nan")}, {"music_fade_out_ms": True},
    {"url": "https://unadopted.invalid/stage.js"}, {"command": "[jump target=elsewhere]"},
])
def test_only_bounded_declarative_transitions_are_accepted(patch):
    with pytest.raises(ValidationError):
        SceneTransitionSpec.model_validate(patch)


@pytest.mark.parametrize("visual", ["none", "dissolve", "fade", "cut"])
def test_transition_presets_allow_intentional_immediate_switch(visual):
    cue = SceneTransitionCue(id="boundary", utterance_id="line", visual=visual,
                             duration_ms=0, music_fade_out_ms=0, music_fade_in_ms=0)
    assert cue.duration_ms == cue.music_fade_out_ms == cue.music_fade_in_ms == 0


@pytest.mark.parametrize("problem", ["unknown_anchor", "duplicate_anchor", "duplicate_id"])
def test_transition_identity_and_anchor_are_unambiguous(problem):
    script, _ = demo_content()
    value = script.model_dump(mode="json")
    first, second = script.utterances[:2]
    cues = [{"id": "boundary_one", "utterance_id": first.id}]
    if problem == "unknown_anchor":
        cues[0]["utterance_id"] = "unknown_line"
    elif problem == "duplicate_anchor":
        cues.append({"id": "boundary_two", "utterance_id": first.id})
    else:
        cues.append({"id": "boundary_one", "utterance_id": second.id})
    with pytest.raises(ValidationError):
        Script.model_validate({**value, "scene_transitions": cues})
