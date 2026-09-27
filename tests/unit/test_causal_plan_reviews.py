"""Exercise the real planning review prompts, retrieval and repair routing."""

import pytest

from services.worker.generation import causal_narrative as causal
from services.worker.generation.causal_runtime import CausalRun
from tests.unit.test_causal_narrative import Harness, evidence_memory, verdict
from tests.unit.test_m3_narrative import FakeLLM

CATEGORIES = ["causality", "progression", "ending_preparation", "fixed_conditions"]
PAYLOAD = {"storyline_id": "plan-review", "chapter_number": 1,
           "workflow_policy": "chapter_editor_v1"}


def blueprint():
    return causal.CausalBlueprint(
        central_question="二人で展示を完成できるか", ending_conditions=["展示を完成する"],
        character_changes=["相談して作業を進める"], chapters=[{
            "number": 1, "question": "未確認の写真をどう扱うか", "inherits": [],
            "unique_progress": ["未確認の札を付けると合意する"],
            "ending_conditions": ["展示を完成する"], "next_consequences": [],
            "required_character_ids": ["aoi", "ren"],
        }])


def response(value):
    return {"content": value.model_dump_json()}


def review(status="pass", *, scope="blueprint", categories=CATEGORIES):
    return causal.GateVerdict(
        verdict=status, checked_categories=categories,
        rationale="計画内の因果を照合した" if status == "pass" else "第2章と第3章の制約が一致しない",
        missing_information=["必要な承認設定がない"] if status == "insufficient_evidence" else [],
        issues=[causal.GateIssue(code="constraint-changed", repair_scope=scope,
            description="第2章では移動不能、第3章では原因なく移動する")]
            if status == "fail" else [])


def assert_planning_prompt(call):
    messages = call[0][1]
    assert messages[0]["content"] == causal.PLAN_REVIEW_SYSTEM
    prompt = messages[1]["content"]
    assert "未執筆の章・場面の本文や台詞・発話IDは要求しません" in prompt
    assert "計画として成立すればpass" in prompt
    assert "その原因・行動・橋渡しが計画にあるか" in prompt
    assert "後章の説明を前章へ遡及適用しません" in prompt
    assert "issues.evidence_idsは[]" in prompt
    assert "issues.evidence_idsは今回本文の発話ID" not in prompt
    assert causal.RULES not in prompt


def test_initial_blueprint_is_reviewed_as_a_plan_without_future_prose():
    llm = FakeLLM([response(blueprint()), response(review())])
    with CausalRun(llm, PAYLOAD) as run:
        plan, report = causal._blueprint(run, {}, 1)
        assert run.state["repairs"] == 0
    assert plan == blueprint() and report.verdict == "pass"
    assert len(llm.calls) == 2
    assert_planning_prompt(llm.calls[1])


def test_revised_blueprint_uses_the_same_planning_role():
    plan = blueprint()
    revised = plan.model_copy(update={"revision": 2, "revision_reason": "前章の結果を反映"})
    llm = FakeLLM([response(revised), response(review())])
    with CausalRun(llm, PAYLOAD) as run:
        result, report = causal._revise_blueprint(
            run, {}, plan, causal.empty_memory("plan-review"), 1, "制約を保持する")
    assert result.revision == 2 and report.verdict == "pass"
    assert_planning_prompt(llm.calls[1])


def test_chapter_and_scene_planning_call_sites_select_real_planning_prompts(monkeypatch):
    real_gate, real_structured = causal._gate, causal._structured
    harness = Harness(monkeypatch)
    calls = {}

    def gate(run, stage, inputs, instruction, categories, scenes=(), **kwargs):
        if stage not in {"chapter-intent-review", "scene-sequence-review"}:
            return harness.gate(run, stage, inputs, instruction, categories, scenes, **kwargs)
        llm = FakeLLM([response(review(categories=categories))])
        with monkeypatch.context() as local, CausalRun(llm, run.payload) as actual:
            local.setattr(causal, "_structured", real_structured)
            report = real_gate(actual, stage, inputs, instruction, categories, scenes, **kwargs)
        calls[stage] = llm.calls[0]
        return report

    monkeypatch.setattr(causal, "_gate", gate)
    harness.chapter()
    assert set(calls) == {"chapter-intent-review", "scene-sequence-review"}
    for call in calls.values():
        assert_planning_prompt(call)


@pytest.mark.parametrize("policy", [None, "chapter_editor_v1"])
@pytest.mark.parametrize("status", ["insufficient_evidence", "fail"])
def test_missing_context_never_rewrites_blueprint_or_consumes_repair_budget(policy, status):
    llm = FakeLLM([response(blueprint()), response(review(status, scope="context"))])
    payload = {**PAYLOAD, "workflow_policy": policy}
    with CausalRun(llm, payload) as run, pytest.raises(causal.GateError) as rejected:
        try:
            causal._blueprint(run, {}, 1)
        finally:
            assert run.state["repairs"] == 0
            assert run.state["calls"] == 2
    assert rejected.value.report.verdict == status


def test_concrete_plan_defect_still_rewrites_blueprint_with_feedback():
    revised = blueprint().model_copy(update={"character_changes": ["互いの判断を確認して展示を仕上げる"]})
    llm = FakeLLM([response(blueprint()), response(review("fail")),
                   response(revised), response(review())])
    with CausalRun(llm, PAYLOAD) as run:
        _, report = causal._blueprint(run, {}, 1)
        assert run.state["repairs"] == 1 and run.state["calls"] == 4
    assert report.verdict == "pass"
    assert "第2章では移動不能、第3章では原因なく移動する" in llm.calls[2][0][1][1]["content"]


def test_plan_defect_with_unresolved_context_does_not_trigger_rewrite():
    failure = review("fail").model_copy(update={"missing_information": ["必要な承認設定がない"]})
    llm = FakeLLM([response(blueprint()), response(failure)])
    with CausalRun(llm, PAYLOAD) as run, pytest.raises(causal.GateError):
        try:
            causal._blueprint(run, {}, 1)
        finally:
            assert run.state["repairs"] == 0 and run.state["calls"] == 2


def test_repeated_plan_failure_stops_at_existing_editor_repair_limit():
    revised = blueprint().model_copy(update={"character_changes": ["互いの判断を確認して展示を仕上げる"]})
    llm = FakeLLM([response(blueprint()), response(review("fail")),
                   response(revised), response(review("fail"))])
    with CausalRun(llm, PAYLOAD) as run, pytest.raises(causal.GateError):
        try:
            causal._blueprint(run, {}, 1)
        finally:
            assert run.state["repairs"] == 1 and run.state["calls"] == 4


@pytest.mark.parametrize("scope", ["blueprint", "chapter_intent"])
def test_planning_review_can_retrieve_adopted_past_sources(scope):
    memory = evidence_memory()
    llm = FakeLLM([verdict(event_ids=["c1-cooperation"]), verdict(passed=True)])
    with CausalRun(llm, {"storyline_id": "story", "chapter_number": 2}) as run:
        report = causal._gate(run, "chapter-intent-review", {}, "計画の継承を検査。",
            ["knowledge"], subject={}, memory=memory, plan_scope=scope)
        assert run.state["repairs"] == 0 and run.state["calls"] == 2
    assert report.verdict == "pass"
    assert "retrieved_evidence_1" in llm.calls[1][0][1][1]["content"]
    assert memory.sources[0].text in llm.calls[1][0][1][1]["content"]
    for call in llm.calls:
        assert_planning_prompt(call)


def test_chapter_planning_can_escalate_an_actual_blueprint_defect():
    llm = FakeLLM([response(review("fail"))])
    with CausalRun(llm, PAYLOAD) as run, pytest.raises(causal.GateError) as rejected:
        causal._gate(run, "chapter-intent-review", {}, "残り章が成立するか。",
            CATEGORIES, subject={}, plan_scope="chapter_intent")
    assert rejected.value.report.issues[0].repair_scope == "blueprint"
    assert len(llm.calls) == 1


def test_planning_review_repairs_a_response_that_targets_unwritten_prose():
    llm = FakeLLM([response(review("fail", scope="scene")), response(review("fail"))])
    with CausalRun(llm, PAYLOAD) as run, pytest.raises(causal.GateError) as rejected:
        causal._gate(run, "blueprint-review", {}, "計画を検査。", CATEGORIES,
                     subject={}, plan_scope="blueprint")
    assert rejected.value.report.issues[0].repair_scope == "blueprint"
    assert len(llm.calls) == 2
    for call in llm.calls:
        assert_planning_prompt(call)


def test_source_review_still_requires_prose_evidence():
    llm = FakeLLM([verdict()])
    with CausalRun(llm, PAYLOAD) as run, pytest.raises(causal.GateError) as rejected:
        causal._gate(run, "chapter-editor-review", {}, "本文を検査。", ["knowledge"], subject={})
    assert rejected.value.report.verdict == "insufficient_evidence"
    prompt = llm.calls[0][0][1][1]["content"]
    assert causal.RULES in prompt
    assert "資料不足を合格にしません" in prompt
    assert "issues.evidence_idsは今回本文の発話ID" in prompt
    assert "計画として成立すればpass" not in prompt


@pytest.mark.parametrize("changed", ["rules", "system"])
def test_stage_cache_includes_review_rules_and_system(tmp_path, changed):
    def execute(options, responses):
        llm = FakeLLM(responses)
        llm.output = tmp_path / "llm"
        causal.legacy._purpose(llm, "blueprint-review")
        with CausalRun(llm, PAYLOAD) as run:
            causal._structured(run, "blueprint-review", {}, "検査。", causal.GateVerdict, **options)
        return llm

    original = {"rules": "計画の因果を検査", "system": causal.PLAN_REVIEW_SYSTEM}
    updated = {**original, changed: original[changed] + "。承認設定も照合"}
    execute(original, [response(review())])
    assert len(execute(updated, [response(review())]).calls) == 1
    assert execute(updated, []).calls == []
