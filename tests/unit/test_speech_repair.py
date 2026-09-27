"""Scene metadata and reusable audio stay attached to their exact source after a split."""

from dataclasses import replace

import pytest

from packages.contracts.m3 import NarrativeScene
from packages.narrative import parse_scene_text
from packages.narrative.repair import remap_scene, voice_reuse_map
from packages.narrative.speech import separate_stage_directions


def scene_and_separation():
    raw = (
        "Hero: （眼鏡を直す）待って。（小声で）お願い。（うなずく）\r\n"
        "NARRATOR: 扉が開いた。\r\n"
        "Dog: （しっぽを振る）\r\n"
        "Hero: 同じ台詞。\r\n"
        "Hero: （ワン）は合図だよ。\r\n"
    )
    utterances = parse_scene_text(raw, "s1", {"Hero", "Dog"})
    utterances[0] = utterances[0].model_copy(update={
        "inner_emotion": "期待", "voice_emotion": "angry", "delivery": "ゆっくり",
    })
    utterances[1] = utterances[1].model_copy(update={
        "inner_emotion": "不安", "voice_emotion": "afraid", "delivery": "小声",
    })
    utterances[2] = utterances[2].model_copy(update={"voice_emotion": "happy", "delivery": "元気に"})
    directions = [
        {"id": "before", "utterance_id": "s1-u1", "kind": "enter", "timing": "before",
         "character_id": "Hero", "position": "left", "duration_ms": 0},
        {"id": "start", "utterance_id": "s1-u1", "kind": "pause", "timing": "start",
         "character_id": None, "position": None, "duration_ms": 20},
        {"id": "after", "utterance_id": "s1-u1", "kind": "pause", "timing": "after",
         "character_id": None, "position": None, "duration_ms": 30},
        {"id": "focus", "utterance_id": "s1-u1", "kind": "focus", "timing": "start",
         "character_id": "Hero", "position": None, "duration_ms": 0},
        {"id": "dog-enter", "utterance_id": "s1-u3", "kind": "enter", "timing": "before",
         "character_id": "Dog", "position": "right", "duration_ms": 0},
        {"id": "dog-focus", "utterance_id": "s1-u3", "kind": "focus", "timing": "start",
         "character_id": "Dog", "position": None, "duration_ms": 0},
    ]
    scene = NarrativeScene(
        id="s1", raw_text=raw, utterances=utterances, directions=directions,
        plan={"id": "s1", "location_id": "door", "character_ids": ["Hero", "Dog"],
              "objectives": "扉を開けてもらう", "start_state": "閉じている",
              "required_events": [{"id": "open", "description": "頼みを聞いて扉が開く"}],
              "end_state": "開く", "atmosphere": "期待"},
        review={"passed": True, "issues": [], "events": [{"event_id": "open",
            "dramatized": True, "evidence_utterance_ids": ["s1-u1", "s1-u2", "s1-u3"],
            "reason": "動作を交えた頼みを受けて扉が開いた"}]},
    )
    separation = separate_stage_directions(
        raw, "s1", {"Hero", "Dog"}, ["s1-u1-p1", "s1-u1-p3", "s1-u3-p1"],
        {"Hero": "陽介", "Dog": "ムギ"},
        direction_narrations={"s1-u1-p1": "陽介が眼鏡を直した。",
                              "s1-u1-p3": "陽介がうなずいた。",
                              "s1-u3-p1": "ムギがしっぽを振った。"},
        delivery_candidate_ids=["s1-u1-p2"],
    )
    return scene, separation


def test_remap_preserves_dialogue_annotations_and_resets_narration_without_mutation():
    scene, separation = scene_and_separation()
    original = scene.model_dump()
    repaired = remap_scene(scene, separation)

    assert scene.model_dump() == original
    assert repaired.raw_text == separation.raw_text
    assert [u.spoken_text for u in repaired.utterances] == [
        "陽介が眼鏡を直した。", "待って。お願い。", "陽介がうなずいた。", "扉が開いた。",
        "ムギがしっぽを振った。", "同じ台詞。", "（ワン）は合図だよ。",
    ]
    dialogue = repaired.utterances[1]
    assert (dialogue.inner_emotion, dialogue.voice_emotion, dialogue.delivery) == (
        "期待", "angry", "ゆっくり",
    )
    for utterance in repaired.utterances:
        assert repaired.raw_text[utterance.source_start:utterance.source_end] == utterance.display_text
        if utterance.speaker_id is None:
            assert utterance.inner_emotion == "未指定"
            assert utterance.voice_emotion == "neutral"
            assert utterance.delivery is None


def test_directions_follow_split_boundaries_and_only_spoken_focus_survives():
    scene, separation = scene_and_separation()
    repaired = remap_scene(scene, separation)
    assert {d.id: d.utterance_id for d in repaired.directions} == {
        "before": "s1-u1", "start": "s1-u1", "after": "s1-u3",
        "focus": "s1-u2", "dog-enter": "s1-u5",
    }
    assert [d.duration_ms for d in repaired.directions] == [0, 20, 30, 0, 0]


def test_review_evidence_expands_in_original_order_and_deduplicates():
    scene, separation = scene_and_separation()
    value = scene.model_dump()
    value["review"].update(passed=False, issues=["追加の内容確認が必要"])
    value["review"]["events"][0]["evidence_utterance_ids"] = ["s1-u3", "s1-u1", "s1-u3", "s1-u2"]
    scene = NarrativeScene.model_validate(value)
    repaired = remap_scene(scene, separation)
    assert repaired.review.events[0].evidence_utterance_ids == [
        "s1-u5", "s1-u1", "s1-u2", "s1-u3", "s1-u4",
    ]
    assert repaired.review.events[0].reason == scene.review.events[0].reason
    assert repaired.review.passed is False
    assert repaired.review.issues == scene.review.issues


def test_delivery_hint_overrides_previous_annotation_for_only_selected_dialogue():
    scene, separation = scene_and_separation()
    repaired = remap_scene(scene, separation, {"s1-u2": "小声で"})
    assert repaired.utterances[1].delivery == "小声で"
    assert repaired.utterances[5].delivery is None
    assert scene.utterances[0].delivery == "ゆっくり"


@pytest.mark.parametrize("hints", [
    {"s1-u2": "長" * 201}, {"s1-u2": ""}, {"s1-u1": "小声"}, {"missing": "小声"},
])
def test_delivery_hint_must_fit_contract_and_target_dialogue(hints):
    scene, separation = scene_and_separation()
    with pytest.raises(ValueError):
        remap_scene(scene, separation, hints)


def test_exact_unchanged_dialogue_audio_is_reusable_even_when_ids_shift():
    scene, separation = scene_and_separation()
    repaired = remap_scene(scene, separation)
    assert voice_reuse_map(scene, repaired, separation) == {
        "s1-u6": "s1-u4", "s1-u7": "s1-u5",
    }


@pytest.mark.parametrize(("field", "value"), [
    ("speaker_id", "Dog"), ("spoken_text", "同じ台詞。 "),
    ("voice_emotion", "happy"), ("delivery", "小声で"),
])
def test_audio_is_not_reused_when_any_spoken_input_changes(field, value):
    scene, separation = scene_and_separation()
    repaired = remap_scene(scene, separation)
    repaired.utterances[5] = repaired.utterances[5].model_copy(update={field: value})
    assert voice_reuse_map(scene, repaired, separation) == {"s1-u7": "s1-u5"}


def test_inner_emotion_is_not_a_spoken_input():
    scene, separation = scene_and_separation()
    repaired = remap_scene(scene, separation)
    repaired.utterances[5] = repaired.utterances[5].model_copy(
        update={"inner_emotion": "胸の内に喜びを秘める"},
    )
    assert voice_reuse_map(scene, repaired, separation)["s1-u6"] == "s1-u4"


def test_incomplete_mapping_cannot_silently_drop_scene_metadata():
    scene, separation = scene_and_separation()
    incomplete = replace(separation, utterance_map={
        key: value for key, value in separation.utterance_map.items() if key != "s1-u3"
    })
    with pytest.raises(ValueError, match="every original"):
        remap_scene(scene, incomplete)
