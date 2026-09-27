"""Chapter lineage and knowledge boundaries must survive replay and continuation."""

import copy
import json

import pytest

from packages.contracts.m3 import NarrativeResult
from packages.narrative import story_state_hash, validate_narrative
from services.worker.generation.narrative import (
    StructuredGenerationError,
    _previous_chapter_context,
    generate_narrative,
)
from tests.unit.test_m3_narrative import FakeLLM, narrative_fixture, narrative_responses


def state(chapter):
    return {"schema_version": 1, "chapter_number": chapter, "summary": "門前で相談する。",
        "facts": [{"id": "secret", "detail": "薬袋には秘密の刻印がある。"}],
        "characters": [{"character_id": cid, "location": "城門", "relationships": [],
            "possessions": ["薬袋"] if cid == "Hero" else [], "injuries": [], "promises": [],
            "inner_emotion": "緊張", "knowledge": [{"fact_id": "secret", "acquired_chapter": 0,
                "evidence_utterance_ids": []}] if cid == "Hero" else []}
            for cid in ("Hero", "keeper")],
        "foreshadowing": [{"outline_index": 0,
            "status": "pending" if chapter == 0 else "resolved" if chapter == 3 else "planted",
            "changed_chapter": 0 if chapter == 0 else 3 if chapter == 3 else 1,
            "evidence_utterance_ids": [] if chapter == 0 else ["s1-u2"]}]}


def continuity_review():
    return {"passed": True, "issues": [], "checked_character_ids": ["Hero", "keeper"],
            "checked_foreshadowing_indices": [0]}


def chapter_fixture(chapter=1, *, previous=None):
    value, snapshot = narrative_fixture()
    value.update(chapter_number=chapter, storyline_id="story-1", start_state=state(chapter - 1),
                 end_state=state(chapter), continuity_review=continuity_review())
    if chapter > 1:
        value.update(previous_narrative_artifact_id=f"narrative-{chapter - 1}",
                     previous_state_hash=story_state_hash(value["start_state"]))
        if previous and previous.get("end_state") is None:
            value["predecessor_state_inferred"] = True
    return value, snapshot


def response(value):
    return {"content": json.dumps(value, ensure_ascii=False)}


def test_three_chapters_keep_private_knowledge_and_resolve_foreshadowing():
    first, snapshot = chapter_fixture()
    second, _ = chapter_fixture(2)
    third, _ = chapter_fixture(3)
    validate_narrative(first, snapshot, require_state=True)
    validate_narrative(second, snapshot, first, require_state=True)
    actual = validate_narrative(third, snapshot, second, require_state=True)
    assert actual.end_state.foreshadowing[0].status == "resolved"
    assert actual.end_state.characters[1].knowledge == []


@pytest.mark.parametrize("mutate", [
    lambda value: value.update(previous_state_hash="0" * 64),
    lambda value: value.update(previous_narrative_artifact_id=None),
    lambda value: value.update(chapter_number=3),
    lambda value: value.update(storyline_id="another-story"),
    lambda value: value["outline"].update(ending="異なる結末"),
    lambda value: value["start_state"].update(summary="前章の状態を書き換えた"),
    lambda value: value["end_state"]["characters"][0].update(knowledge=[]),
    lambda value: value["end_state"]["characters"][1].update(knowledge=[{
        "fact_id": "secret", "acquired_chapter": 0, "evidence_utterance_ids": []}]),
    lambda value: value["end_state"]["characters"][1].update(knowledge=[{
        "fact_id": "secret", "acquired_chapter": 2, "evidence_utterance_ids": ["missing"]}]),
    lambda value: value["end_state"]["foreshadowing"][0].update(status="resolved"),
    lambda value: value["continuity_review"].update(checked_character_ids=["Hero"]),
    lambda value: value["continuity_review"].update(checked_foreshadowing_indices=[]),
    lambda value: value["continuity_review"].update(passed=False, issues=["秘密の漏洩"]),
])
def test_inconsistent_chapter_continuity_is_rejected(mutate):
    first, snapshot = chapter_fixture()
    second, _ = chapter_fixture(2)
    mutate(second)
    with pytest.raises(ValueError):
        validate_narrative(second, snapshot, first, require_state=True)


def test_new_knowledge_requires_evidence_without_sharing_it_with_other_characters():
    first, snapshot = chapter_fixture()
    second, _ = chapter_fixture(2)
    second["end_state"]["characters"][1]["knowledge"] = [{"fact_id": "secret", "acquired_chapter": 2,
        "evidence_utterance_ids": ["s1-u1", "s1-u2"]}]
    result = validate_narrative(second, snapshot, first, require_state=True)
    assert result.end_state.characters[1].knowledge[0].acquired_chapter == 2
    assert result.start_state.characters[1].knowledge == []


def test_legacy_first_chapter_is_readable_but_new_first_chapter_requires_states():
    first, snapshot = narrative_fixture()
    assert validate_narrative(first, snapshot).end_state is None
    with pytest.raises(ValueError, match="start/end"):
        validate_narrative(first, snapshot, require_state=True)


def test_state_hash_is_canonical_and_changes_with_private_knowledge():
    original = state(1)
    reordered = dict(reversed(list(original.items())))
    assert story_state_hash(original) == story_state_hash(reordered)
    reordered = copy.deepcopy(reordered)
    reordered["characters"][1]["knowledge"] = copy.deepcopy(original["characters"][0]["knowledge"])
    assert story_state_hash(original) != story_state_hash(reordered)


def test_first_chapter_m4_adds_state_generation_without_changing_legacy_path():
    first, snapshot = chapter_fixture()
    responses = narrative_responses(first)
    responses.insert(2, response(first["start_state"]))
    responses.extend([response(first["end_state"]), response(continuity_review())])
    llm = FakeLLM(responses)
    result = generate_narrative({"approval_snapshot": snapshot, "m4": True,
                                 "chapter_number": 1, "storyline_id": "story-1"}, llm)
    assert result["start_state"] == first["start_state"]
    assert result["end_state"] == first["end_state"]
    assert result["previous_state_hash"] is None
    assert [call[0][0] for call in llm.calls] == ["story_outline", "supporting_character", "story_state",
        "scene_plan", "scene-text-s1", "staging", "quality_review", "story_state", "continuity_review"]


@pytest.mark.parametrize("legacy", [False, True])
def test_later_chapter_reuses_outline_cast_and_transfers_state_without_rewriting_predecessor(legacy):
    previous, snapshot = narrative_fixture() if legacy else chapter_fixture()
    original = copy.deepcopy(previous)
    second, _ = chapter_fixture(2, previous=previous)
    responses = narrative_responses(second)[2:]
    if legacy:
        responses.insert(0, response(second["start_state"]))
        responses.append(response(continuity_review()))
    responses.extend([response(second["end_state"]), response(continuity_review())])
    llm = FakeLLM(responses)
    result = generate_narrative({"approval_snapshot": snapshot, "m4": True,
        "chapter_number": 2, "storyline_id": "story-1", "previous_narrative": previous,
        "previous_narrative_artifact_id": "narrative-1",
        "previous_state_hash": None if legacy else story_state_hash(previous["end_state"])}, llm)
    assert result["chapter_number"] == 2
    assert result["outline"] == previous["outline"]
    assert result["supporting_characters"] == previous["supporting_characters"]
    assert result["previous_state_hash"] == story_state_hash(result["start_state"])
    assert result["predecessor_state_inferred"] is legacy
    assert previous == original
    purposes = [call[0][0] for call in llm.calls]
    assert "story_outline" not in purposes and "supporting_character" not in purposes
    scene_prompt = next(call[0][1][-1]["content"] for call in llm.calls if call[0][0] == "scene_plan")
    assert "第2章だけ" in scene_prompt and "人物別の知識" in scene_prompt
    assert previous["scenes"][-1]["utterances"][-1]["display_text"] in scene_prompt
    assert previous["locations"][0]["image_prompt"] in scene_prompt
    assert "一字も変えず再利用" in scene_prompt
    validate_narrative(result, snapshot, previous, require_state=True)


@pytest.mark.parametrize("length", [200, 985])
def test_preceding_chapter_quote_is_bounded_and_keeps_original_source_lines(length):
    value, _ = chapter_fixture()
    lines = ["Hero: " + str(number) + "語" * length + "\r\n" for number in range(12)]
    value["scenes"][0]["raw_text"] = "".join(lines)
    context = _previous_chapter_context(NarrativeResult.model_validate(value))
    encoded = context.split("（時系列順の原文引用）: ", 1)[1].split("\n直前章で", 1)[0]
    excerpts = json.loads(encoded)
    assert len(excerpts) <= 8
    assert sum(len(item["source_line"]) for item in excerpts) <= 5000
    assert [item["source_line"] for item in excerpts] == lines[-len(excerpts):]


def test_wrong_predecessor_hash_is_rejected_before_any_model_call():
    previous, snapshot = chapter_fixture()
    llm = FakeLLM([])
    with pytest.raises(ValueError, match="hash"):
        generate_narrative({"approval_snapshot": snapshot, "m4": True, "chapter_number": 2,
            "previous_narrative": previous, "previous_narrative_artifact_id": "narrative-1",
            "previous_state_hash": "0" * 64}, llm)
    assert llm.calls == []


def test_semantic_knowledge_leak_cannot_pass_and_retry_includes_source():
    first, snapshot = chapter_fixture()
    responses = narrative_responses(first)
    responses.insert(2, response(first["start_state"]))
    failed = {**continuity_review(), "passed": False, "issues": ["門番が未取得の秘密を話している。"]}
    responses.extend([response(first["end_state"]), response(failed), response(failed)])
    llm = FakeLLM(responses)
    with pytest.raises(StructuredGenerationError, match="秘密"):
        generate_narrative({"approval_snapshot": snapshot, "m4": True, "storyline_id": "story-1"}, llm)
    assert llm.failure_request == 5  # Keep design/start state, regenerate rejected source.


def test_rejected_inferred_predecessor_retries_state_extraction_not_only_new_prose():
    previous, snapshot = narrative_fixture()
    original = copy.deepcopy(previous)
    second, _ = chapter_fixture(2, previous=previous)
    failed = {**continuity_review(), "passed": False,
              "issues": ["前章の開始状態で門番に未取得の秘密がある。"]}
    replies = [response(second["start_state"]), *narrative_responses(second)[2:],
               response(failed), response(failed)]
    llm = FakeLLM(replies)
    with pytest.raises(StructuredGenerationError, match="未取得の秘密"):
        generate_narrative({"approval_snapshot": snapshot, "m4": True,
            "chapter_number": 2, "storyline_id": "story-1", "previous_narrative": previous,
            "previous_narrative_artifact_id": "narrative-1", "previous_state_hash": None}, llm)
    assert llm.failure_request == 1
    assert previous == original


def test_bad_end_state_metadata_retries_extraction_without_rewriting_valid_prose():
    first, snapshot = chapter_fixture()
    replies = narrative_responses(first)
    replies.insert(2, response(first["start_state"]))
    bad_state = {**first["end_state"], "chapter_number": 0}
    replies.extend([response(bad_state), response(bad_state)])
    llm = FakeLLM(replies)
    with pytest.raises(StructuredGenerationError, match="chapter number"):
        generate_narrative({"approval_snapshot": snapshot, "m4": True,
                            "chapter_number": 1, "storyline_id": "story-1"}, llm)
    assert llm.failure_request == 8  # Retain design, initial state, prose and scene review.
