"""Causal orchestration uses observed results and repairs the responsible stage.

Planning/writing/review model boundaries are stubbed, not a script of HTTP calls.
The real chapter loop, extraction mapping, evidence ledger and repair budgets run.
"""

import copy
import json
from collections import Counter
from types import SimpleNamespace

import pytest

from packages.contracts.m3 import NarrativeResult, NarrativeScene, ScenePlan
from packages.contracts.story_workflow import (
    ReviewIssue,
    ReviewReport,
    SceneIntent,
    StateEntry,
    StoryBlueprint,
)
from packages.narrative.story_ledger import (
    apply_chapter_memory,
    empty_memory,
    memory_hash,
    source_catalog,
)
from packages.narrative.validation import parse_scene_text, story_state_hash
from services.worker.generation import causal_narrative as causal
from services.worker.generation.causal_runtime import CausalRun, RepairLimitError, digest
from tests.unit.test_m3_narrative import FakeLLM, narrative_fixture


class NoServerLLM:
    def __init__(self):
        self.config = {"llm": {"model_id": "fake"}}
        self.payload = {}
        self.trace = []
        self.requests = 0

    def chat(self, *args, **kwargs):
        pytest.fail("This orchestration test must not call a model server")


def physical(memory):
    return next(entry.value for entry in memory.state
                if entry.entity_id == "Hero" and entry.key == "physical_condition")


def planning(chapter, scene_id):
    plan = ScenePlan(id=scene_id, location_id="gate", character_ids=["Hero", "keeper"],
        objectives="けがに対処して共同作業を続ける", start_state="直前の作業の結果を引き継ぐ",
        required_events=[{"id": "change", "description": "作業の方法を変える"}],
        end_state="次の作業に進める", atmosphere="落ち着いて相談する")
    intent = SceneIntent(scene_id=scene_id, chapter_number=chapter,
        character_ids=plan.character_ids, location_id=plan.location_id, story_time=f"作業第{chapter}段階",
        entry_bridge="同じ場所で直前の作業から続く", motive="安全に完成させたい",
        obstacle="片手を痛めている", planned_action="分担して対処する", expected_change="作業方法が変わる")
    return causal.SceneProposal(intent=intent, plan=plan)


class Harness:
    def __init__(self, monkeypatch):
        self.calls = []
        self.writes = []
        self.failures = {}
        self.runs = []
        self.blueprint = StoryBlueprint(central_question="助けを借りて作業を終えられるか",
            character_changes=["一人で抱えず相手の助けを使って行動する"],
            ending_conditions=["共同作業を終える"], chapters=[{
                "number": n, "question": f"作業第{n}段階をどう進めるか",
                "inherits": [] if n == 1 else ["前章の応急処置で作業を続けられるようになった"],
                "unique_progress": ["対処する" if n == 1 else "方法を選び直して作業を終える"],
                "ending_conditions": ["次の作業に移れる" if n == 1 else "作業終了"],
                "next_consequences": ["安全な作業方法を実際に試す必要が生じる"] if n == 1 else [],
                "required_character_ids": ["Hero", "keeper"]}
                for n in (1, 2)])
        self.initial = empty_memory("story", state=[StateEntry(scope="character", entity_id="Hero",
            key="physical_condition", value="手を切って作業中断")])
        self.cast = [{"id": "Hero", "name": "主人公"}, {"id": "keeper", "name": "相棒"}]
        self.setting = {"world": {"chapterCount": 2}, "relationships": {}, "main_characters": self.cast}
        monkeypatch.setattr(causal, "_structured", self.structured)
        monkeypatch.setattr(causal, "_gate", self.gate)
        monkeypatch.setattr(causal, "_write_scene", self.write)
        monkeypatch.setattr(causal, "compare_scene_facts", lambda run, memory, scene: {
            "verdict": "pass", "facts": [], "comparisons": [], "resolved_comparisons": [], "checked_knowledge_ids": [],
            "memory_hash": memory_hash(memory), "missing_information": []})
        monkeypatch.setattr(causal, "review_state_deltas", self.state_review)
        monkeypatch.setattr(causal, "annotate_information",
                            lambda run, memory, scene, raw, **kwargs: {
                                **raw.model_dump(mode="json"), "knowledge_updates": [], "introductions": []})

    def state_review(self, run, scene, extraction):
        subject = {"scene": scene.model_dump(mode="json"), "extraction": extraction.model_dump(mode="json")}
        return SimpleNamespace(report=self.gate(run, "state-comparison-" + scene.id, subject,
            "状態差分の根拠を原文と比較する", ["state_support", "action_stage", "object_identity"],
            [scene], subject=subject))

    def structured(self, run, stage, inputs, instruction, model, validate=None):
        self.calls.append((stage, copy.deepcopy(inputs)))
        number = run.payload["chapter_number"]
        if model is causal.IntentProposal:
            value = model(intent={"chapter_number": number, "question": "共同作業をどう進めるか",
                "entry_bridge": "直前の作業から続く", "unique_progress": ["新しい方法を決める"],
                "desired_end": ["作業を進める"], "character_ids": ["Hero", "keeper"],
                "location_ids": ["gate"],
                "inherited_event_ids": [event["id"] for event in inputs["actual_memory"]["events"]]})
        elif model is causal.SceneSequence:
            fixture, _ = narrative_fixture()
            known = {location["id"] for location in inputs["known_location_registry"]}
            value = model(locations=[{**location,
                "structural_description": location["description"] if location["id"] not in known else None}
                for location in fixture["locations"]],
                scenes=[planning(number, f"s{i}")
                for i in range(1, 3 if number == 1 else 2)])
        elif model is causal.SceneProposal:
            value = model.model_validate(inputs["planned"])
            start = next(entry["value"] for entry in inputs["actual_memory"]["state"]
                         if entry["key"] == "physical_condition")
            value = value.model_copy(update={"plan": value.plan.model_copy(update={"start_state": start})})
        elif model is causal.SceneFacts:
            scene = inputs["source"][0]
            event_id = f"c{number}-{scene['scene_id']}-change"
            previous = inputs["prior_observed_memory"]
            after = self.result_state(number, scene["scene_id"])
            evidence = [utterance["id"] for utterance in scene["utterances"]]
            value = model(summary=after, events=[{"id": event_id, "description": after,
                "character_ids": ["Hero", "keeper"], "location_ids": ["gate"],
                "causes": [previous["event_index"][-1]["id"]] if previous["event_index"] else [],
                "story_time": inputs["story_time"], "assertion": "observed", "evidence": evidence}], state_deltas=[{
                "id": f"c{number}-{scene['scene_id']}-condition", "scope": "character",
                "entity_id": "Hero", "key": "physical_condition", "historical_before": None, "after": after,
                "event_id": event_id, "evidence": evidence}], thread_updates=[])
        else:
            raise AssertionError(f"Unexpected planning boundary: {stage}")
        if validate:
            validate(value)
        return value

    def gate(self, run, stage, inputs, instruction, categories, scenes=(), **kwargs):
        self.calls.append((stage, copy.deepcopy(inputs)))
        scopes = self.failures.get(stage, [])
        subject_hash = digest(kwargs.get("subject", inputs))
        if scopes:
            scope = scopes.pop(0)
            report = ReviewReport(scope=stage, subject_hash=subject_hash, verdict="fail",
                checked_categories=categories, checked_scene_ids=[scene.id for scene in scenes],
                rationale="原文と必要な変化が一致しない", issues=[ReviewIssue(code="state-mismatch",
                    severity="error", description="原文と必要な変化が一致しない", repair_scope=scope)])
            raise causal.GateError(report)
        return ReviewReport(scope=stage, subject_hash=subject_hash, verdict="pass",
            checked_categories=categories, checked_scene_ids=[scene.id for scene in scenes],
            rationale="対象の原文と実績を照合した")

    @staticmethod
    def result_state(number, scene_id):
        return {(1, "s1"): "応急処置済み", (1, "s2"): "保護手袋を装着", (2, "s1"): "無理せず作業終了"}[
            number, scene_id]

    def write(self, run, setting, memory, proposal, names, cast):
        number, scene_id = run.payload["chapter_number"], proposal.plan.id
        self.writes.append((number, scene_id, memory.model_copy(deep=True)))
        after = self.result_state(number, scene_id)
        raw = f"Hero: {after}。この方法なら続けられる。\nNARRATOR: 相棒は{after}の状態を確かめて頷いた。\n"
        utterances = parse_scene_text(raw, scene_id, {"Hero", "keeper"})
        return NarrativeScene(id=scene_id, plan=proposal.plan, raw_text=raw, utterances=utterances,
            directions=[{"id": f"{scene_id}-d{i}", "utterance_id": f"{scene_id}-u1", "kind": "enter",
                         "timing": "before", "character_id": cid, "position": position, "duration_ms": 0}
                        for i, (cid, position) in enumerate((("Hero", "left"), ("keeper", "right")), 1)],
            review={"passed": True, "issues": [], "events": [{"event_id": "change", "dramatized": True,
                     "evidence_utterance_ids": [u.id for u in utterances], "reason": "対処の結果が描かれる"}]})

    def chapter(self, previous=None):
        memory = previous.story_memory if previous else self.initial
        payload = {"chapter_number": previous.chapter_number + 1 if previous else 1,
                   "storyline_id": "story"}
        if previous:
            payload.update(previous_narrative_artifact_id="first-commit",
                           previous_state_hash=story_state_hash(previous.end_state))
        with CausalRun(NoServerLLM(), payload) as run:
            self.runs.append(run)
            return causal._chapter(run, self.setting, self.blueprint, memory, previous, self.cast, "")


def test_observed_scene_state_and_events_reach_next_scene_then_next_chapter(monkeypatch):
    harness = Harness(monkeypatch)
    initial_bytes = harness.initial.model_dump()
    first = harness.chapter()
    second = harness.chapter(first)
    assert [(number, sid, physical(memory)) for number, sid, memory in harness.writes] == [
        (1, "s1", "手を切って作業中断"), (1, "s2", "応急処置済み"), (2, "s1", "保護手袋を装着")]
    adapted = next(inputs for stage, inputs in harness.calls if stage == "adapt-plan-s2")
    assert adapted["actual_memory"]["events"][0]["id"] == "c1-s1-change"
    assert first.scenes[1].plan.start_state == "応急処置済み"
    assert second.start_memory == first.story_memory
    assert second.start_state == first.end_state
    assert second.chapter_intent.predecessor_memory_hash == memory_hash(first.story_memory)
    assert second.chapter_intent.inherited_event_ids == ["c1-s1-change", "c1-s2-change"]
    assert [event.id for event in second.story_memory.events] == [
        "c1-s1-change", "c1-s2-change", "c2-s1-change"]
    assert second.story_memory.events[-1].causes == ["c1-s2-change"]
    assert second.story_memory.previous_memory_hash == memory_hash(first.story_memory)
    assert [delta.before for delta in second.story_memory.state_history] == [
        "手を切って作業中断", "応急処置済み", "保護手袋を装着"]
    assert harness.initial.model_dump() == initial_bytes


def extract_condition(monkeypatch, memory, delta, text, *, historical=False):
    """Run the real extraction expansion/replay around an immutable model response."""
    harness = Harness(monkeypatch)
    proposal = planning(1, "s1")
    raw = f"NARRATOR: {text}\n"
    scene = harness.write(type("Run", (), {"payload": {"chapter_number": 1}})(), {}, memory,
                          proposal, {}, [])
    scene = scene.model_copy(update={"raw_text": raw,
        "utterances": parse_scene_text(raw, "s1", {"Hero", "keeper"})})
    response = {"summary": text, "events": [{
        "id": "c1-s1-change", "description": text, "character_ids": ["Hero", "keeper"],
        "location_ids": ["gate"], "causes": [], "story_time": "十年前" if historical else "今",
        "presentation": "flashback" if historical else "current", "assertion": "observed",
        "evidence": ["s1-u1"]}], "state_deltas": [{
            "id": "c1-s1-condition", "scope": "character", "entity_id": "Hero",
            "key": "physical_condition", "event_id": "c1-s1-change",
            "evidence": ["s1-u1"], "historical_before": None, **delta}],
        "thread_updates": []}
    before = copy.deepcopy(response)

    def structured(run, stage, inputs, instruction, model, validate=None):
        value = model.model_validate(response)
        validate(value)
        return value

    monkeypatch.setattr(causal, "_structured", structured)
    with CausalRun(NoServerLLM(), {"storyline_id": "story", "chapter_number": 1}) as run:
        extraction, _ = causal._extract(run, memory, scene, proposal.intent,
                                        {"Hero", "keeper"}, {"gate"})
    assert response == before
    result = apply_chapter_memory(memory, extraction, scenes=[scene],
                                  storyline_id="story", chapter_number=1)
    return extraction, result


def test_current_delta_uses_exact_state_key_and_does_not_infer_before_from_canon(monkeypatch):
    memory = empty_memory("story", state=[
        {"scope": "character", "entity_id": "keeper", "key": "physical_condition", "value": "健康"},
        {"scope": "character", "entity_id": "Hero", "key": "goal", "value": "作業を続ける"},
        {"scope": "world", "entity_id": "Hero", "key": "physical_condition", "value": "設定上の分類"},
    ], author_facts=[{"id": "initial_condition", "content": "主人公は作業前に手を切っていた。",
        "origin": "initial_canon", "authority_id": "setup", "authority_hash": "a" * 64,
        "established_at": "作業前", "known_by_character_ids": ["Hero", "keeper"],
        "disclosure_condition": "already_known"}])
    before = memory.model_dump()
    extraction, result = extract_condition(monkeypatch, memory, {"after": "応急処置済み"},
        "相棒が主人公の手を洗い、包帯を巻いた。")
    assert "before" not in causal.ExtractedDelta.model_json_schema()["properties"]
    assert extraction.state_deltas[0].before is None
    assert extraction.state_deltas[0].after == "応急処置済み"
    assert physical(result) == "応急処置済み"
    assert memory.model_dump() == before


def test_historical_delta_preserves_prose_before_without_rewinding_current_state(monkeypatch):
    memory = empty_memory("story", state=[{"scope": "character", "entity_id": "Hero",
        "key": "physical_condition", "value": "完治している"}])
    extraction, result = extract_condition(monkeypatch, memory, {
        "time_scope": "historical", "historical_before": "負傷していない", "after": "手を切った",
        "effective_time": "十年前"}, "十年前、主人公は健康な手を割れたガラスで切った。", historical=True)
    assert extraction.state_deltas[0].before == "負傷していない"
    assert extraction.state_deltas[0].after == "手を切った"
    assert result.state_history[-1].effective_time == "十年前"
    assert physical(result) == "完治している"
    assert result.state == memory.state


@pytest.mark.parametrize("delta,existing", [
    ({"after": "応急処置済み"}, "応急処置済み"),
    ({"after": None}, None),
    ({"time_scope": "historical", "historical_before": "健康", "after": "健康",
      "effective_time": "十年前"}, "完治している"),
])
def test_expanded_delta_rejects_noop_without_changing_after(monkeypatch, delta, existing):
    memory = empty_memory("story", state=[] if existing is None else [{
        "scope": "character", "entity_id": "Hero", "key": "physical_condition", "value": existing}])
    before = memory.model_dump()
    with pytest.raises(ValueError, match="must change an explicit value"):
        extract_condition(monkeypatch, memory, delta, "主人公の状態は変わらなかった。",
                          historical=delta.get("time_scope") == "historical")
    assert memory.model_dump() == before


def test_current_delta_rejects_llm_inferred_before_and_historical_before():
    delta = {"id": "c1-s1-condition", "scope": "character", "entity_id": "Hero",
        "key": "physical_condition", "event_id": "c1-s1-change", "evidence": ["s1-u1"],
        "historical_before": None, "after": "応急処置済み"}
    with pytest.raises(ValueError, match="Extra inputs"):
        causal.ExtractedDelta.model_validate({**delta, "before": "手を切っていた"})
    with pytest.raises(ValueError, match="cannot supply a historical before"):
        causal.ExtractedDelta.model_validate({**delta, "historical_before": "手を切っていた"})
    with pytest.raises(ValueError, match="effective story time"):
        causal.ExtractedDelta.model_validate({**delta, "time_scope": "historical"})


def test_thread_precondition_is_derived_from_exact_memory_not_generated(monkeypatch):
    harness = Harness(monkeypatch)
    original = harness.structured

    def structured(run, stage, inputs, instruction, model, validate=None):
        value = original(run, stage, inputs, instruction, model, validate)
        if model is causal.SceneFacts:
            event = value.events[0]
            update = causal.ExtractedThread(id="c1-help", question="共同作業を終える",
                status="open" if inputs["source"][0]["scene_id"] == "s1" else "resolved",
                character_ids=["Hero", "keeper"], location_ids=["gate"], event_ids=[event.id],
                evidence=event.evidence)
            value = value.model_copy(update={"thread_updates": [update]})
            validate(value)
        return value

    monkeypatch.setattr(causal, "_structured", structured)
    result = harness.chapter()
    assert "expected_status" not in causal.ExtractedThread.model_json_schema()["properties"]
    assert result.scene_extractions[0].thread_updates[0].expected_status is None
    assert result.scene_extractions[1].thread_updates[0].expected_status == "open"
    assert result.story_memory.threads[0].status == "resolved"


def test_two_chapter_entrypoint_passes_real_adoption_and_review_binding_checks(monkeypatch):
    harness = Harness(monkeypatch)
    monkeypatch.setattr(causal, "_initial", lambda *args: harness.initial)
    monkeypatch.setattr(causal, "_blueprint", lambda *args: (harness.blueprint, ReviewReport(
        scope="blueprint-review", subject_hash=digest(harness.blueprint), verdict="pass",
        rationale="章ごとの進展と結末準備を確認した",
        checked_categories=["causality", "progression", "ending_preparation", "fixed_conditions"])))
    _, snapshot = narrative_fixture()
    snapshot["world"]["result"]["chapterCount"] = 2
    payload = {"approval_snapshot": snapshot, "storyline_id": "story", "chapter_number": 1}
    first = causal.generate_causal_narrative(payload, NoServerLLM())
    second = causal.generate_causal_narrative({**payload, "chapter_number": 2,
        "previous_narrative": first, "previous_narrative_artifact_id": "first-commit",
        "previous_state_hash": story_state_hash(first["end_state"])}, NoServerLLM())
    adopted = NarrativeResult.model_validate(second)
    assert adopted.workflow_version == 2 and adopted.chapter_number == 2
    assert adopted.start_memory.model_dump(mode="json") == first["story_memory"]
    assert physical(adopted.story_memory) == "無理せず作業終了"
    assert adopted.location_registry[0].introduced_chapter == 1
    assert [item.model_dump(mode="json") for item in adopted.location_registry] == first["location_registry"]
    sequences = [inputs for stage, inputs in harness.calls if stage == "scene-sequence"]
    assert sequences[0]["known_location_registry"] == []
    assert sequences[1]["known_location_registry"] == first["location_registry"]


@pytest.mark.parametrize("target", ["extraction-review-s1", "state-comparison-s1"])
def test_extraction_repair_keeps_source_and_does_not_apply_rejected_delta(monkeypatch, target):
    harness = Harness(monkeypatch)
    harness.failures[target] = ["extraction"]
    result = harness.chapter()
    assert Counter(sid for _, sid, _ in harness.writes) == {"s1": 1, "s2": 1}
    attempts = [inputs for stage, inputs in harness.calls if stage == "extract-s1"]
    assert len(attempts) == 2
    assert attempts[0]["prior_observed_memory"] == attempts[1]["prior_observed_memory"]
    assert attempts[0]["source"] == attempts[1]["source"]
    assert len(result.story_memory.state_history) == 2
    assert [issue["scope"] for issue in harness.runs[-1].state["issues"]] == ["extraction"]


@pytest.mark.parametrize("target", ["extraction-review-s1", "state-comparison-s1"])
def test_exhausted_extraction_repairs_do_not_rewrite_sound_prose_or_replan_chapter(monkeypatch, target):
    harness = Harness(monkeypatch)
    harness.failures[target] = ["extraction", "extraction"]
    monkeypatch.setattr(causal, "_initial", lambda *args: harness.initial)
    monkeypatch.setattr(causal, "_blueprint", lambda *args: (harness.blueprint, ReviewReport(
        scope="blueprint-review", subject_hash=digest(harness.blueprint), verdict="pass",
        checked_categories=["causality", "progression", "ending_preparation", "fixed_conditions"])))
    _, snapshot = narrative_fixture()
    snapshot["world"]["result"]["chapterCount"] = 2
    llm = NoServerLLM()
    with pytest.raises(causal.GateError):
        causal.generate_causal_narrative({"approval_snapshot": snapshot,
            "storyline_id": "story", "chapter_number": 1}, llm)
    assert [(chapter, scene) for chapter, scene, _ in harness.writes] == [(1, "s1")]
    assert [stage for stage, _ in harness.calls].count("chapter-intent") == 1
    assert [stage for stage, _ in harness.calls].count("extract-s1") == 2
    assert [event["scope"] for event in llm.trace if event["type"] == "causal_repair"] == ["extraction"]


def test_extraction_retry_preserves_retrieved_original_evidence(monkeypatch):
    harness = Harness(monkeypatch)
    original = harness.gate
    failed = False
    supplement = {"status": "ready", "source_quotes": [{"id": "q1", "text": "原文の確定した約束"}]}

    def gate(run, stage, inputs, instruction, categories, scenes=(), **kwargs):
        nonlocal failed
        report = original(run, stage, inputs, instruction, categories, scenes, **kwargs)
        if stage == "extraction-review-s1" and not failed:
            failed = True
            raise causal.GateError(report.model_copy(update={"verdict": "fail", "issues": [
                ReviewIssue(code="incorrect-promise", severity="error", description="約束の対象が違う",
                            repair_scope="extraction")]}), [supplement])
        return report

    monkeypatch.setattr(causal, "_gate", gate)
    harness.chapter()
    attempts = [inputs for stage, inputs in harness.calls if stage == "extract-s1"]
    feedback = json.loads(attempts[1]["feedback"])
    assert feedback["review"]["issues"][0]["description"] == "約束の対象が違う"
    assert feedback["retrieved_evidence"] == [supplement]


def test_scene_failure_repairs_only_failed_scene_before_using_its_state(monkeypatch):
    harness = Harness(monkeypatch)
    harness.failures["scene-continuity-s2"] = ["scene"]
    result = harness.chapter()
    assert Counter(sid for _, sid, _ in harness.writes) == {"s1": 1, "s2": 2}
    retries = [memory for _, sid, memory in harness.writes if sid == "s2"]
    assert retries[0] == retries[1]
    assert physical(retries[0]) == "応急処置済み"
    assert len(result.story_memory.state_history) == 2
    assert [issue["scope"] for issue in harness.runs[-1].state["issues"]] == ["scene"]


def test_unresolved_scene_failure_returns_no_chapter_and_preserves_predecessor(monkeypatch):
    harness = Harness(monkeypatch)
    first = harness.chapter()
    before = first.model_dump()
    harness.failures["scene-continuity-s1"] = ["scene"] * 3
    with pytest.raises(causal.GateError):
        harness.chapter(first)
    assert first.model_dump() == before
    run = harness.runs[-1]
    assert run.state["repairs"] == 2
    assert not any(stage == "chapter-progress-review" for stage, _ in harness.calls[
        next(i for i, (stage, inputs) in enumerate(harness.calls)
             if stage == "chapter-intent" and inputs["chapter_role"]["number"] == 2):])


def test_initial_canon_prompt_separates_future_plans_from_established_state(monkeypatch):
    captured = []

    def respond(llm, stage, prompt, model, validate=None, system=None):
        captured.append(prompt)
        value = model(state=[{"scope": "character", "entity_id": "Hero", "key": "goal",
                              "value": "作業を完成させたい", "changed_chapter": 0}], secrets=[])
        validate(value)
        return value

    monkeypatch.setattr(causal.legacy, "_structured", respond)
    with CausalRun(NoServerLLM(), {"storyline_id": "story", "chapter_number": 1}) as run:
        memory = causal._initial(run, {"approved_goal": "作業を完成させたい"}, [{"id": "Hero"}])
    assert memory.chapter_number == 0 and memory.events == [] and memory.author_facts == []
    assert memory.state[0].value == "作業を完成させたい"
    assert "既に起きた状態にしません" in captured[0]
    assert "将来の事件はsecretsに入れません" in captured[0]


@pytest.mark.parametrize("damage", ["future_state", "unknown_character", "secret_known_by_unknown"])
def test_initial_canon_rejects_invalid_evidence_or_characters(monkeypatch, damage):
    def respond(llm, stage, prompt, model, validate=None, system=None):
        state = {"scope": "character", "entity_id": "Hero", "key": "goal", "value": "作業を続ける"}
        secrets = []
        if damage == "future_state":
            state["changed_chapter"] = 1
        elif damage == "unknown_character":
            state["entity_id"] = "not-approved"
        else:
            secrets = [{"id": "secret", "content": "作業前からの秘密", "established_at": "前日",
                        "known_by_character_ids": ["not-approved"], "disclosure_condition": "打ち明ける時"}]
        value = model(state=[state], secrets=secrets)
        validate(value)
        return value

    monkeypatch.setattr(causal.legacy, "_structured", respond)
    with (CausalRun(NoServerLLM(), {"storyline_id": "story", "chapter_number": 1}) as run,
          pytest.raises(ValueError)):
        causal._initial(run, {}, [{"id": "Hero"}])


def evidence_memory():
    fixture, _ = narrative_fixture()
    scenes = fixture["scenes"]
    refs = source_catalog(scenes, storyline_id="story", chapter_number=1)
    return apply_chapter_memory(empty_memory("story"), {
        "chapter_number": 1, "summary": "協力を決めた。", "events": [{
            "id": "c1-cooperation", "description": "薬を届けるため協力を決めた。",
            "character_ids": ["Hero", "keeper"], "location_ids": ["gate"],
            "story_time": "初日の夕方", "evidence": [ref.model_dump() for ref in refs]}]},
        scenes=scenes, storyline_id="story", chapter_number=1)


def verdict(*, event_ids=(), source_ids=(), passed=False):
    return {"content": json.dumps({
        "verdict": "pass" if passed else "insufficient_evidence",
        "checked_categories": ["knowledge"],
        "rationale": "過去の原文を照合した" if passed else "過去の原文が必要",
        "issues": [], "missing_information": [] if passed else ["過去の発言の原文"],
        "requested_event_ids": list(event_ids), "requested_source_ids": list(source_ids)},
        ensure_ascii=False)}


@pytest.mark.parametrize("request_type", ["event", "source"])
def test_real_gate_retrieves_saved_original_before_passing_in_shared_budget(request_type):
    memory = evidence_memory()
    before = memory.model_dump()
    packet = causal._memory_context(memory, {"Hero", "keeper"}, scope="extraction")
    assert packet["source_quotes"] == [] and packet["events"] == []
    alias = packet["event_index"][0]["evidence"][0]
    request = {"event_ids": ["c1-cooperation"]} if request_type == "event" else {"source_ids": [alias]}
    llm = FakeLLM([verdict(**request), verdict(passed=True)])
    subject = {"chapter": 2}
    inputs = {"observed": packet}
    with CausalRun(llm, {"storyline_id": "story", "chapter_number": 2}) as run:
        report = causal._gate(run, "knowledge-review", inputs, "知識の根拠を照合する。",
                               ["knowledge"], subject=subject, memory=memory)
        assert run.state["calls"] == 2 and run.state["repairs"] == 0
    supplement = causal.retrieve_context_evidence(memory, memory_hash=memory_hash(memory),
        record_ids=request.get("event_ids", ()), source_ids=request.get("source_ids", ()))
    assert report.verdict == "pass"
    assert report.subject_hash == digest(subject)
    assert report.input_hash == digest({**inputs, "retrieved_evidence_1": supplement})
    assert len([item for item in llm.trace if item["type"] == "context_retrieval"]) == 1
    followup_prompt = json.dumps(llm.calls[1][0][1], ensure_ascii=False)
    assert supplement["source_quotes"][0]["text"] in followup_prompt
    assert "retrieved_evidence_1" in followup_prompt
    assert memory.model_dump() == before


def test_real_gate_keeps_unknown_source_alias_insufficient_without_resampling():
    memory = evidence_memory()
    llm = FakeLLM([verdict(source_ids=["q999999"])])
    with CausalRun(llm, {"storyline_id": "story", "chapter_number": 2}) as run:
        with pytest.raises(causal.GateError) as rejected:
            causal._gate(run, "knowledge-review", {}, "根拠を照合する。", ["knowledge"],
                         subject={}, memory=memory)
        assert rejected.value.report.verdict == "insufficient_evidence"
        assert run.state["calls"] == 1 and run.state["repairs"] == 0
    assert not any(item["type"] == "context_retrieval" for item in llm.trace)


def test_real_gate_stops_after_two_retrievals_without_resetting_call_budget():
    memory = evidence_memory()
    llm = FakeLLM([verdict(event_ids=["c1-cooperation"])] * 3)
    with CausalRun(llm, {"storyline_id": "story", "chapter_number": 2}) as run:
        with pytest.raises(causal.GateError) as rejected:
            causal._gate(run, "knowledge-review", {}, "根拠を照合する。", ["knowledge"],
                         subject={}, memory=memory)
        assert rejected.value.report.verdict == "insufficient_evidence"
        assert run.state["calls"] == 3 and run.state["repairs"] == 0
    assert len([item for item in llm.trace if item["type"] == "context_retrieval"]) == 2


def test_retrieval_followup_cannot_bypass_existing_llm_call_limit():
    memory = evidence_memory()
    llm = FakeLLM([verdict(event_ids=["c1-cooperation"]), verdict(passed=True)])
    with CausalRun(llm, {"storyline_id": "story", "chapter_number": 2,
                        "workflow_limits": {"max_calls": 1}}) as run:
        with pytest.raises(RepairLimitError, match="call budget"):
            causal._gate(run, "knowledge-review", {}, "根拠を照合する。", ["knowledge"],
                         subject={}, memory=memory)
        assert run.state["calls"] == 1
        assert llm.requests == 1


@pytest.mark.parametrize("target", ["scene-continuity-s1", "extraction-review-s1"])
def test_missing_evidence_does_not_trigger_source_or_extraction_repairs(monkeypatch, target):
    real_gate = causal._gate
    harness = Harness(monkeypatch)
    requests = []

    def structured(run, stage, inputs, instruction, model, validate=None):
        if model is not causal.GateVerdict:
            return harness.structured(run, stage, inputs, instruction, model, validate)
        requests.append(stage)
        categories = (["causality", "progression", "knowledge", "state", "repetition"]
                      if target.startswith("scene-") else
                      ["support", "completeness", "knowledge", "state", "promises", "introductions"])
        value = model(verdict="insufficient_evidence", checked_categories=categories,
            rationale="参照すべき原文を取得できない", missing_information=["過去の発言"],
            requested_source_ids=["q999999"])
        validate(value)
        return value

    def gate(run, stage, *args, **kwargs):
        selected = real_gate if stage == target else harness.gate
        return selected(run, stage, *args, **kwargs)

    monkeypatch.setattr(causal, "_structured", structured)
    monkeypatch.setattr(causal, "_gate", gate)
    initial = harness.initial.model_dump()
    with pytest.raises(causal.GateError) as rejected:
        harness.chapter()
    assert rejected.value.report.verdict == "insufficient_evidence"
    assert requests == [target]
    assert [(chapter, sid) for chapter, sid, _ in harness.writes] == [(1, "s1")]
    assert harness.runs[-1].state["repairs"] == 0
    assert harness.initial.model_dump() == initial


def revised_future(blueprint):
    value = blueprint.model_dump(mode="json")
    value.update(revision=blueprint.revision + 1,
                 revision_reason="前章で協力が成立したため、その成果を使う次の作業へ進める。")
    value["chapters"][1].update(question="分担した方法で作業を終えられるか",
                                 unique_progress=["得意分野を分担して共同作業を完了する"])
    return value


@pytest.mark.parametrize("change", ["future_only", "prefix", "immutable", "ending", "count", "revision", "reason"])
def test_blueprint_revision_only_changes_unadopted_future_with_versioned_reason(monkeypatch, change):
    # Stub only the model boundary; _revise_blueprint owns and runs the validation callback.
    harness = Harness(monkeypatch)
    blueprint = harness.blueprint.model_copy(update={"immutable_conditions": ["超常現象は起きない"]})
    original = blueprint.model_dump(mode="json")
    memory = evidence_memory()
    memory_before = memory.model_dump(mode="json")
    proposed = revised_future(blueprint)
    if change == "prefix":
        proposed["chapters"][0]["question"] = "採用済みの過去を違う計画にした"
    elif change == "immutable":
        proposed["immutable_conditions"] = ["超常現象で解決する"]
    elif change == "ending":
        proposed["ending_conditions"] = ["共同作業を中止する"]
    elif change == "count":
        proposed["chapters"].pop()
    elif change == "revision":
        proposed["revision"] = blueprint.revision
    elif change == "reason":
        proposed["revision_reason"] = None
    calls = []

    def propose(run, stage, inputs, instruction, model, validate=None):
        calls.append(stage)
        assert inputs["first_editable_chapter"] == 2
        value = model.model_validate(proposed)
        validate(value)
        return value

    monkeypatch.setattr(causal, "_structured", propose)
    with CausalRun(NoServerLLM(), {"storyline_id": "story", "chapter_number": 2}) as run:
        if change == "future_only":
            result, review = causal._revise_blueprint(run, {}, blueprint, memory, 2, "導入が重複している")
            assert result.chapters[0] == blueprint.chapters[0]
            assert result.chapters[1] != blueprint.chapters[1]
            assert result.revision == blueprint.revision + 1 and result.revision_reason
            assert result.immutable_conditions == blueprint.immutable_conditions
            assert result.ending_conditions == blueprint.ending_conditions
            assert review.subject_hash == digest(result) and review.verdict == "pass"
        else:
            with pytest.raises(ValueError, match="Blueprint revision"):
                causal._revise_blueprint(run, {}, blueprint, memory, 2, "導入が重複している")
            assert not any(stage == "blueprint-review" for stage, _ in harness.calls)
    assert calls == ["revise-blueprint"]
    assert blueprint.model_dump(mode="json") == original
    assert memory.model_dump(mode="json") == memory_before


def test_blueprint_gate_failure_revises_plan_and_adopts_new_review_before_writing(monkeypatch):
    harness = Harness(monkeypatch)
    harness.blueprint = harness.blueprint.model_copy(update={"immutable_conditions": ["超常現象は起きない"]})
    monkeypatch.setattr(causal, "_initial", lambda *args: harness.initial)
    monkeypatch.setattr(causal, "_blueprint", lambda *args: (harness.blueprint, ReviewReport(
        scope="blueprint-review", subject_hash=digest(harness.blueprint), verdict="pass",
        rationale="章ごとの進展と結末準備を確認した",
        checked_categories=["causality", "progression", "ending_preparation", "fixed_conditions"])))
    _, snapshot = narrative_fixture()
    snapshot["world"]["result"]["chapterCount"] = 2
    payload = {"approval_snapshot": snapshot, "storyline_id": "story", "chapter_number": 1}
    first = causal.generate_causal_narrative(payload, NoServerLLM())
    original = copy.deepcopy(first)
    harness.calls.clear()
    harness.writes.clear()
    harness.failures["chapter-intent-review"] = ["blueprint"]

    def structured(run, stage, inputs, instruction, model, validate=None):
        if stage != "revise-blueprint":
            return harness.structured(run, stage, inputs, instruction, model, validate)
        harness.calls.append((stage, copy.deepcopy(inputs)))
        assert inputs["first_editable_chapter"] == 2
        assert "blueprint" in inputs["feedback"]
        proposed = model.model_validate(revised_future(StoryBlueprint.model_validate(inputs["blueprint"])))
        validate(proposed)
        return proposed

    monkeypatch.setattr(causal, "_structured", structured)
    llm = NoServerLLM()
    second = causal.generate_causal_narrative({**payload, "chapter_number": 2,
        "previous_narrative": first, "previous_narrative_artifact_id": "first-commit",
        "previous_state_hash": story_state_hash(first["end_state"])}, llm)
    adopted = NarrativeResult.model_validate(second)
    stages = [stage for stage, _ in harness.calls]
    assert stages[:5] == ["chapter-intent", "chapter-intent-review", "revise-blueprint",
                          "blueprint-review", "chapter-intent"]
    assert stages.count("revise-blueprint") == 1
    assert [(n, sid) for n, sid, _ in harness.writes] == [(2, "s1")]
    assert [entry["scope"] for entry in llm.trace if entry["type"] == "causal_repair"] == ["blueprint"]
    assert adopted.blueprint.revision == 2 and adopted.chapter_intent.blueprint_revision == 2
    reviews = [review for review in adopted.workflow_reviews if review.scope == "blueprint-review"]
    assert len(reviews) == 1 and reviews[0].subject_hash == digest(adopted.blueprint)
    assert reviews[0].subject_hash != digest(first["blueprint"])
    assert first == original
