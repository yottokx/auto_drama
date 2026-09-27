"""Bad planned references are repaired at the planning boundary, before prose."""

import copy

import pytest

from tests.unit.test_causal_narrative import Harness, causal
from tests.unit.test_m3_narrative import FakeLLM


def repair_candidate(stage, instruction, model, validate, invalid, valid):
    llm = FakeLLM([{"content": model.model_validate(value).model_dump_json()}
                   for value in (invalid, valid)])
    result = causal.legacy._structured(llm, stage, instruction, model, validate=validate)
    rejected = [item for item in llm.trace if item["type"] == "structured_rejected"]
    assert llm.requests == 2 and len(rejected) == 1
    return result, rejected[0]["reason"]


def test_initial_sequence_repairs_a_planned_event_reference_before_any_prose(monkeypatch):
    harness = Harness(monkeypatch)
    original = harness.structured
    rejected = []

    def structured(run, stage, inputs, instruction, model, validate=None):
        value = original(run, stage, inputs, instruction, model, validate)
        if model is causal.SceneSequence:
            valid = value.model_dump()
            invalid = copy.deepcopy(valid)
            invalid["scenes"][0]["intent"]["inherited_event_ids"] = ["planned-not-observed"]
            assert harness.writes == []
            value, reason = repair_candidate(stage, instruction, model, validate, invalid, valid)
            assert harness.writes == []
            rejected.append(reason)
        return value

    monkeypatch.setattr(causal, "_structured", structured)
    chapter = harness.chapter()
    assert "planned/nonexistent event" in rejected[0]
    assert chapter.scene_intents[0].inherited_event_ids == []
    assert [sid for _, sid, _ in harness.writes] == ["s1", "s2"]


@pytest.mark.parametrize("change", ["event", "scene", "location", "cast"])
def test_adaptation_repairs_identity_and_accepts_actual_same_chapter_events(monkeypatch, change):
    harness = Harness(monkeypatch)
    original = harness.structured
    rejected = []

    def structured(run, stage, inputs, instruction, model, validate=None):
        value = original(run, stage, inputs, instruction, model, validate)
        if model is causal.SceneProposal:
            valid = value.model_dump()
            valid["intent"]["inherited_event_ids"] = ["c1-s1-change"]
            invalid = copy.deepcopy(valid)
            if change == "event":
                invalid["intent"]["inherited_event_ids"] = ["planned-not-observed"]
            elif change == "scene":
                invalid["plan"]["id"] = invalid["intent"]["scene_id"] = "s3"
            elif change == "location":
                invalid["plan"]["location_id"] = invalid["intent"]["location_id"] = "other-room"
            else:
                invalid["plan"]["character_ids"] = invalid["intent"]["character_ids"] = ["keeper", "Hero"]
            assert [sid for _, sid, _ in harness.writes] == ["s1"]
            value, reason = repair_candidate(stage, instruction, model, validate, invalid, valid)
            assert [sid for _, sid, _ in harness.writes] == ["s1"]
            rejected.append(reason)
        return value

    monkeypatch.setattr(causal, "_structured", structured)
    chapter = harness.chapter()
    assert ("planned/nonexistent" if change == "event" else "registered identity") in rejected[0]
    assert chapter.scene_intents[1].inherited_event_ids == ["c1-s1-change"]
    assert chapter.scene_intents[1].scene_id == "s2"
    assert [sid for _, sid, _ in harness.writes] == ["s1", "s2"]


def test_closed_thread_is_repaired_in_chapter_intent_before_writing(monkeypatch):
    harness = Harness(monkeypatch)
    original = harness.structured
    rejected = []

    def structured(run, stage, inputs, instruction, model, validate=None):
        value = original(run, stage, inputs, instruction, model, validate)
        if model is causal.SceneFacts and run.payload["chapter_number"] == 1:
            event = value.events[0]
            update = causal.ExtractedThread(id="c1-help", question="安全に作業を再開する",
                status="open" if inputs["source"][0]["scene_id"] == "s1" else "resolved",
                character_ids=["Hero", "keeper"], location_ids=["gate"],
                event_ids=[event.id], evidence=event.evidence)
            value = value.model_copy(update={"thread_updates": [update]})
            validate(value)
        elif model is causal.IntentProposal and run.payload["chapter_number"] == 2:
            valid = value.model_dump()
            invalid = copy.deepcopy(valid)
            invalid["intent"]["inherited_thread_ids"] = ["c1-help"]
            assert all(number == 1 for number, _, _ in harness.writes)
            value, reason = repair_candidate(stage, instruction, model, validate, invalid, valid)
            assert all(number == 1 for number, _, _ in harness.writes)
            rejected.append(reason)
        return value

    monkeypatch.setattr(causal, "_structured", structured)
    first = harness.chapter()
    assert first.story_memory.threads[0].status == "resolved"
    second = harness.chapter(first)
    assert "already closed thread" in rejected[0]
    assert second.chapter_intent.inherited_thread_ids == []
    assert [(number, sid) for number, sid, _ in harness.writes] == [(1, "s1"), (1, "s2"), (2, "s1")]
