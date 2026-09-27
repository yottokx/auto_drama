"""Compact continuity prompts retain source evidence and fail closed on gaps."""

import copy
import json

import pytest

from packages.contracts.m3 import NarrativeResult, StoryOutline
from packages.contracts.m4 import StoryState
from packages.narrative import parse_scene_text, story_state_hash
from services.worker.generation.continuity import (
    InferredStateReviewError,
    _expected_foreshadowing,
    _source,
    _source_json,
    _state_schema,
    finish_narrative,
    start_state,
)
from services.worker.generation.llm import ContextBudgetError
from services.worker.generation.narrative import StructuredGenerationError, _data
from tests.unit.test_m3_narrative import FakeLLM, narrative_fixture
from tests.unit.test_m4_narrative import chapter_fixture, continuity_review, response, state


def test_compact_source_preserves_every_word_speaker_and_evidence_id_in_order():
    value, _ = narrative_fixture()
    scene = value["scenes"][0]
    scene["raw_text"] = "\r\n".join(
        f"Hero:  {number}番目。秘密の合図は『月』です。　" for number in range(180))
    scene["utterances"] = [u.model_dump() for u in
        parse_scene_text(scene["raw_text"], "s1", {"Hero", "keeper"})]
    # Distinct speech is uncommon but must not disappear from the source view.
    scene["utterances"][5]["spoken_text"] = "ごばんめ"
    second = copy.deepcopy(scene)
    second["id"] = second["plan"]["id"] = "s2"
    second["utterances"] = [u.model_dump() for u in
        parse_scene_text(scene["raw_text"], "s2", {"Hero", "keeper"})]
    value["scenes"].append(second)
    original = copy.deepcopy(value)
    result = NarrativeResult.model_validate(value)

    compact = _source(result.scenes)
    assert json.loads(_source_json(result.scenes)) == compact
    for scene, projected in zip(result.scenes, compact, strict=True):
        assert projected["plan"] == scene.plan.model_dump()
        assert len(projected["utterances"]) == len(scene.utterances)
        for utterance, item in zip(scene.utterances, projected["utterances"], strict=True):
            assert item["id"] == utterance.id
            assert item["speaker_id"] == utterance.speaker_id
            assert item["display_text"] == utterance.display_text
            assert item.get("spoken_text", item["display_text"]) == utterance.spoken_text
            assert ("spoken_text" in item) == (utterance.spoken_text != utterance.display_text)
            assert not {"source_start", "source_end", "inner_emotion", "voice_emotion", "delivery"} & item.keys()
    assert value == original
    full = [{"plan": scene.plan.model_dump(),
             "utterances": [u.model_dump() for u in scene.utterances]} for scene in result.scenes]
    assert len(_source_json(result.scenes)) < len(_data(full)) * 0.6


def test_start_state_avoids_duplicate_outline_and_keeps_old_prompt_for_cache_only():
    value, _ = narrative_fixture()
    previous = NarrativeResult.model_validate(value)
    outline_text = _data(previous.outline)
    context = "承認設定\n全体プロット: " + outline_text
    llm = FakeLLM([response(state(1))])
    cache_calls = []

    def cached_chat(*args, **kwargs):
        cache_calls.append((args, kwargs))

    llm.cached_chat = cached_chat
    initial = start_state(llm, context, previous.outline, {"Hero", "keeper"}, previous)

    assert initial == StoryState.model_validate(state(1))
    assert len(llm.calls) == len(cache_calls) == 1
    prompt = llm.calls[0][0][1][-1]["content"]
    legacy_prompt = cache_calls[0][0][1][-1]["content"]
    assert prompt.count(outline_text) == 1
    assert legacy_prompt.count(outline_text) == 2
    legacy_source = [{"plan": scene.plan.model_dump(),
        "utterances": [u.model_dump() for u in scene.utterances]} for scene in previous.scenes]
    assert "\n採用済みの前章本文（空なら本編開始前）: " + _data(legacy_source) in legacy_prompt
    assert _source_json(previous.scenes) in prompt
    assert "source_start" not in prompt and "source_start" in legacy_prompt
    assert previous == NarrativeResult.model_validate(value)


def test_start_state_replays_validated_legacy_response_without_new_inference():
    value, _ = narrative_fixture()
    previous = NarrativeResult.model_validate(value)
    llm = FakeLLM([])
    cached = []

    def cached_chat(*args, **kwargs):
        cached.append((args, kwargs))
        return response(state(1))

    llm.cached_chat = cached_chat
    initial = start_state(llm, "承認設定", previous.outline, {"Hero", "keeper"}, previous)
    assert len(cached) == 1
    assert llm.calls == []
    assert initial == StoryState.model_validate(state(1))


def _chapters():
    previous, _ = narrative_fixture()
    current, _ = chapter_fixture(2, previous=previous)
    for value, text in ((previous, "前章だけの台詞。"), (current, "当章だけの台詞。")):
        scene = value["scenes"][0]
        scene["raw_text"] = scene["raw_text"].replace("扉を開けてくれ。", text)
        scene["utterances"] = [u.model_dump() for u in
            parse_scene_text(scene["raw_text"], "s1", {"Hero", "keeper"})]
    return NarrativeResult.model_validate(previous), NarrativeResult.model_validate(current)


def _finish(llm, previous, current):
    return finish_narrative(llm, {"storyline_id": "story-1",
        "previous_narrative_artifact_id": "chapter-one-artifact"}, "承認設定と全体設計",
        current, current.start_state, previous)


def test_legacy_review_and_current_review_never_combine_both_full_chapters():
    previous, current = _chapters()
    originals = previous.model_copy(deep=True), current.model_copy(deep=True)
    llm = FakeLLM([response(continuity_review()), response(state(2)), response(continuity_review())])
    completed = _finish(llm, previous, current)
    assert [call[0][0] for call in llm.calls] == ["continuity_review", "story_state", "continuity_review"]
    prompts = [call[0][1][-1]["content"] for call in llm.calls]
    assert _source_json(previous.scenes) in prompts[0]
    assert "前章だけの台詞。" in prompts[0] and "当章だけの台詞。" not in prompts[0]
    for prompt in prompts[1:]:
        assert _source_json(current.scenes) in prompt
        assert "前章だけの台詞。" not in prompt and "当章だけの台詞。" in prompt
        assert prompt.count(_data(current.start_state)) == 1
    assert (previous, current) == originals
    assert completed.start_state.characters[1].knowledge == []
    assert completed.previous_state_hash == story_state_hash(current.start_state)
    assert completed.predecessor_state_inferred is True
    verified = next(item for item in llm.trace if item["type"] == "legacy_state_review")
    assert verified["previous_narrative_artifact_id"] == "chapter-one-artifact"
    assert verified["state_hash"] == completed.previous_state_hash
    assert verified["review"] == continuity_review()


@pytest.mark.parametrize("damage", [
    {"passed": False, "issues": ["門番に知らない秘密の知識が追加されています。"]},
    {"passed": True, "issues": ["根拠の台詞が一致しません。"]},
    {"checked_character_ids": ["Hero"]},
    {"checked_character_ids": ["Hero", "Hero"]},
    {"checked_foreshadowing_indices": []},
    {"checked_foreshadowing_indices": [0, 0]},
])
def test_failed_or_incomplete_legacy_review_cannot_adopt_or_advance(damage):
    previous, current = _chapters()
    originals = previous.model_copy(deep=True), current.model_copy(deep=True)
    failed = {**continuity_review(), **damage}
    llm = FakeLLM([response(failed), response(failed)])
    with pytest.raises(InferredStateReviewError, match="Predecessor state review") as raised:
        _finish(llm, previous, current)
    assert isinstance(raised.value.__cause__, StructuredGenerationError)
    assert [call[0][0] for call in llm.calls] == ["continuity_review", "continuity_review"]
    assert not any(item["type"] == "legacy_state_review" for item in llm.trace)
    assert (previous, current) == originals


def test_invalid_legacy_state_evidence_is_rejected_before_model_review():
    previous, current = _chapters()
    current.start_state.characters[1].knowledge.append(
        current.start_state.characters[0].knowledge[0].model_copy(update={
            "acquired_chapter": 1, "evidence_utterance_ids": ["missing-utterance"]}))
    llm = FakeLLM([])
    with pytest.raises(ValueError, match="evidence"):
        _finish(llm, previous, current)
    assert llm.calls == []


@pytest.mark.parametrize("error", [ContextBudgetError("入力枠を超過"), RuntimeError("HTTP unavailable")])
def test_legacy_transport_and_budget_errors_do_not_request_state_repair(error):
    previous, current = _chapters()
    llm = FakeLLM([])

    def unavailable(*args, **kwargs):
        raise error

    llm.chat = unavailable
    with pytest.raises(type(error)) as raised:
        _finish(llm, previous, current)
    assert raised.value is error
    assert not any(item["type"] == "legacy_state_review" for item in llm.trace)


def test_ending_schema_pins_chapter_and_foreshadowing_schedule_without_changing_start_schema():
    expected_start = StoryState.model_json_schema()
    expected_start["$defs"]["CharacterStoryState"]["properties"]["character_id"]["enum"] = ["Hero", "keeper"]
    assert _state_schema({"Hero", "keeper"}) == expected_start
    value, _ = narrative_fixture()
    value["outline"]["foreshadowing"].extend([
        {"setup_chapter": 2, "payoff_chapter": 2, "detail": "当章で設置・回収"},
        {"setup_chapter": 3, "payoff_chapter": 3, "detail": "未設置"},
    ])
    outline = StoryOutline.model_validate(value["outline"])
    expected = [
        {"outline_index": 0, "status": "planted", "changed_chapter": 1},
        {"outline_index": 1, "status": "resolved", "changed_chapter": 2},
        {"outline_index": 2, "status": "pending", "changed_chapter": 0},
    ]
    assert _expected_foreshadowing(outline, 2) == expected
    schema = _state_schema({"Hero", "keeper"}, chapter_number=2, outline=outline)
    assert schema["properties"]["chapter_number"]["const"] == 2
    clues = schema["properties"]["foreshadowing"]
    assert clues["minItems"] == clues["maxItems"] == 3
    for variant, item in zip(clues["items"]["anyOf"], expected, strict=True):
        assert {key: variant["properties"][key]["const"] for key in item} == item
        evidence = variant["properties"]["evidence_utterance_ids"]
        if item["status"] == "pending":
            assert evidence["maxItems"] == 0
        else:
            assert evidence["minItems"] == 1
    assert _state_schema({"Hero", "keeper"}) == expected_start


def test_ending_schema_handles_no_foreshadowing_without_empty_any_of():
    value, _ = narrative_fixture()
    value["outline"]["foreshadowing"] = []
    outline = StoryOutline.model_validate(value["outline"])
    schema = _state_schema({"Hero", "keeper"}, chapter_number=2, outline=outline)
    clues = schema["properties"]["foreshadowing"]
    assert clues["minItems"] == clues["maxItems"] == 0
    assert "anyOf" not in clues["items"]


@pytest.mark.parametrize("invalid", ["chapter", "foreshadowing"])
def test_end_state_retries_wrong_chapter_or_foreshadowing_without_altering_text(invalid):
    previous = NarrativeResult.model_validate(chapter_fixture()[0])
    current = NarrativeResult.model_validate(chapter_fixture(2)[0])
    bad = state(2)
    if invalid == "chapter":
        bad["chapter_number"] = 1
    else:
        bad["foreshadowing"][0].update(status="resolved", changed_chapter=3)
    llm = FakeLLM([response(bad), response(state(2)), response(continuity_review())])
    completed = _finish(llm, previous, current)
    assert completed.end_state.chapter_number == 2
    assert completed.end_state.foreshadowing[0].status == "planted"
    assert [call[0][0] for call in llm.calls] == ["story_state", "story_state", "continuity_review"]
    for args, kwargs in llm.calls[:2]:
        prompt = args[1][-1]["content"]
        assert "chapter_number=2" in prompt
        assert _data(_expected_foreshadowing(current.outline, 2)) in prompt
        assert _source_json(current.scenes) in prompt
        assert kwargs["response_format"]["json_schema"]["schema"]["properties"]["chapter_number"]["const"] == 2


def test_scheduled_foreshadowing_without_real_evidence_still_fails_final_review():
    previous = NarrativeResult.model_validate(chapter_fixture()[0])
    current = NarrativeResult.model_validate(chapter_fixture(2)[0])
    failed = {**continuity_review(), "passed": False, "issues": ["予定された伏線の根拠が本文にありません。"]}
    llm = FakeLLM([response(state(2)), response(failed), response(failed)])
    with pytest.raises(StructuredGenerationError, match="根拠が本文にありません"):
        _finish(llm, previous, current)
    assert [call[0][0] for call in llm.calls] == ["story_state", "continuity_review", "continuity_review"]
