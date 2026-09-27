"""Unevaluated script continuation preserves technical source and lineage checks."""

import copy

import pytest

from packages.contracts.m3 import NarrativeResult, SceneReview
from packages.narrative import parse_scene_text, validate_narrative
from packages.narrative.continuity import narrative_hash
from tests.unit.test_causal_validation import bind_reviews, causal_fixture
from tests.unit.test_m3_narrative import narrative_fixture
from tests.unit.test_m4_narrative import state


def script_fixture(previous=None):
    result, snapshot = narrative_fixture()
    result.update(workflow_version=2, workflow_policy="script_continuation_v1", storyline_id="story-1")
    for scene in result["scenes"]:
        scene["review"] = {"policy": "not_evaluated", "passed": False, "issues": [], "events": []}
    if previous is not None:
        result.update(chapter_number=previous["chapter_number"] + 1,
                      previous_narrative_artifact_id=f"narrative-{previous['chapter_number']}",
                      previous_narrative_hash=narrative_hash(previous))
    return result, snapshot


def test_script_chapters_continue_without_state_extraction_or_meaning_verdicts():
    first, snapshot = script_fixture()
    second, _ = script_fixture(first)
    third, _ = script_fixture(second)
    for result, previous, artifact in ((first, None, None), (second, first, "narrative-1"),
                                       (third, second, "narrative-2")):
        checked = validate_narrative(result, snapshot, previous, require_state=True,
                                     expected_previous_artifact_id=artifact)
        assert checked.start_state is None and checked.end_state is None
        assert checked.story_memory is None and checked.workflow_reviews == []
        assert checked.scenes[0].review.policy == "not_evaluated"
        assert checked.scenes[0].review.passed is False


def test_semantic_inconsistency_does_not_replace_the_script_source_or_block_the_chapter():
    result, snapshot = script_fixture()
    raw = "Hero: 時刻は十九時四十分だ。\nNARRATOR: 時計の針が十九時半を指した。\n"
    result["scenes"][0].update(raw_text=raw, utterances=[item.model_dump() for item in
        parse_scene_text(raw, "s1", {"Hero", "keeper"})])
    checked = validate_narrative(result, snapshot)
    assert checked.scenes[0].raw_text == raw
    assert [line.speaker_id for line in checked.scenes[0].utterances] == ["Hero", None]


@pytest.mark.parametrize("mutation", [
    lambda value: value["outline"]["chapters"].pop(),
    lambda value: value.update(chapter_number=4),
    lambda value: value["outline"]["character_arcs"].pop(),
    lambda value: value["outline"]["foreshadowing"][0].update(payoff_chapter=4),
    lambda value: value["scenes"][0]["utterances"].pop(),
    lambda value: value["scenes"][0]["utterances"][0].update(display_text="別の台詞"),
    lambda value: value["scenes"][0]["utterances"][0].update(spoken_text="別の読み上げ"),
    lambda value: value["scenes"][0]["utterances"][0].update(speaker_id="keeper"),
    lambda value: value["scenes"][0]["utterances"][0].update(source_start=0),
    lambda value: value["scenes"][0]["utterances"].reverse(),
    lambda value: value["scenes"][0]["plan"].update(location_id="unknown"),
    lambda value: value["scenes"][0]["plan"].update(character_ids=["Hero", "unknown"]),
    lambda value: value["scenes"][0]["plan"].update(character_ids=["Hero", "Hero"]),
    lambda value: value["scenes"][0]["plan"].update(id="different"),
    lambda value: value["scenes"][0]["directions"][0].update(utterance_id="missing"),
    lambda value: value["scenes"][0]["directions"][0].update(character_id="intruder"),
    lambda value: value["scenes"][0]["directions"][0].update(position=None),
    lambda value: value["scenes"][0]["directions"][0].update(kind="javascript"),
    lambda value: value["scenes"][0]["directions"].pop(),
    lambda value: value["scenes"].append(copy.deepcopy(value["scenes"][0])),
    lambda value: value["locations"].append({**value["locations"][0], "id": "unused"}),
    lambda value: value.update(title="長" * 201),
])
def test_unevaluated_policy_retains_technical_validation(mutation):
    result, snapshot = script_fixture()
    mutation(result)
    with pytest.raises(ValueError):
        validate_narrative(result, snapshot)


@pytest.mark.parametrize("review", [
    {"policy": "not_evaluated", "passed": True, "issues": [], "events": []},
    {"policy": "not_evaluated", "passed": False, "issues": ["unperformed judgement"], "events": []},
    {"policy": "not_evaluated", "passed": False, "issues": [], "events": [
        {"event_id": "event", "dramatized": True, "evidence_utterance_ids": ["s1-u1"], "reason": "claim"}]},
])
def test_unperformed_reviews_cannot_claim_a_verdict(review):
    with pytest.raises(ValueError, match="cannot claim"):
        SceneReview.model_validate(review)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(workflow_version=1),
    lambda value: value.update(chapter_review_mode="partitioned"),
    lambda value: value.update(start_state=state(0)),
    lambda value: value.update(end_state=state(1)),
    lambda value: value.update(previous_state_hash="0" * 64),
    lambda value: value.update(predecessor_state_inferred=True),
    lambda value: value.update(continuity_review={"passed": True, "issues": [],
        "checked_character_ids": ["Hero", "keeper"], "checked_foreshadowing_indices": [0]}),
    lambda value: value["scenes"][0]["review"].update(policy="deferred_to_chapter"),
])
def test_script_policy_does_not_fabricate_state_or_use_another_review_policy(mutation):
    result, _ = script_fixture()
    mutation(result)
    with pytest.raises(ValueError):
        NarrativeResult.model_validate(result)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(storyline_id=None),
    lambda value: value.update(previous_narrative_artifact_id="narrative-0"),
    lambda value: value.update(previous_narrative_hash="0" * 64),
])
def test_first_script_chapter_requires_its_own_identity_and_no_predecessor(mutation):
    first, snapshot = script_fixture()
    mutation(first)
    with pytest.raises(ValueError):
        validate_narrative(first, snapshot)


@pytest.mark.parametrize("mutation", [
    lambda value: value.update(previous_narrative_hash="0" * 64),
    lambda value: value.update(previous_narrative_hash=None),
    lambda value: value.update(previous_narrative_artifact_id=None),
    lambda value: value.update(previous_narrative_artifact_id="another-artifact"),
    lambda value: value.update(chapter_number=3),
    lambda value: value.update(storyline_id="another-story"),
])
def test_later_script_chapter_must_match_exact_predecessor(mutation):
    first, snapshot = script_fixture()
    second, _ = script_fixture(first)
    mutation(second)
    with pytest.raises(ValueError):
        validate_narrative(second, snapshot, first, expected_previous_artifact_id="narrative-1")


def test_later_chapter_cannot_skip_predecessor_comparison():
    first, snapshot = script_fixture()
    second, _ = script_fixture(first)
    with pytest.raises(ValueError, match="expected predecessor artifact"):
        validate_narrative(second, snapshot, first)
    with pytest.raises(ValueError, match="preceding script"):
        validate_narrative(second, snapshot, expected_previous_artifact_id="narrative-1")
    with pytest.raises(ValueError, match="first script chapter"):
        validate_narrative(first, snapshot, first, expected_previous_artifact_id="narrative-1")


def test_source_hash_tracks_actual_script_and_is_stable_across_serialization():
    result, _ = script_fixture()
    checked = NarrativeResult.model_validate(result)
    assert narrative_hash(result) == narrative_hash(checked)
    assert narrative_hash(result) == narrative_hash(checked.model_dump(mode="json"))
    changed = copy.deepcopy(result)
    changed["scenes"][0]["raw_text"] += "\n"
    assert narrative_hash(result) != narrative_hash(changed)
    changed = copy.deepcopy(result)
    changed["scenes"][0]["directions"][0]["position"] = "center"
    assert narrative_hash(result) != narrative_hash(changed)


def test_existing_legacy_and_causal_reviews_are_not_relaxed():
    legacy, snapshot = narrative_fixture()
    legacy["scenes"][0]["review"] = {
        "policy": "not_evaluated", "passed": False, "issues": [], "events": []}
    with pytest.raises(ValueError, match="content review"):
        validate_narrative(legacy, snapshot)
    causal, snapshot, _ = causal_fixture()
    editor = bind_reviews(causal.model_copy(update={"workflow_policy": "chapter_editor_v1"}))
    for value in (causal, editor):
        data = value.model_dump(mode="json")
        data["scenes"][0]["review"] = {
            "policy": "not_evaluated", "passed": False, "issues": [], "events": []}
        with pytest.raises(ValueError):
            validate_narrative(data, snapshot)
        data = value.model_dump(mode="json")
        data["workflow_reviews"] = []
        with pytest.raises(ValueError, match="requires plans"):
            NarrativeResult.model_validate(data)


def test_source_hash_field_does_not_change_old_serialized_narratives():
    legacy, _ = narrative_fixture()
    assert "previous_narrative_hash" not in NarrativeResult.model_validate(legacy).model_dump(mode="json")
    legacy["previous_narrative_hash"] = "0" * 64
    with pytest.raises(ValueError, match="source hashes"):
        NarrativeResult.model_validate(legacy)
