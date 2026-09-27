import copy

import pytest

from packages.narrative.causal_validation import validate_causal_continuity
from packages.narrative.story_ledger import memory_hash, source_catalog
from services.worker.generation import causal_narrative as causal
from services.worker.generation.causal_runtime import CausalRun
from tests.unit.test_causal_narrative import Harness, NoServerLLM, planning
from tests.unit.test_causal_validation import causal_fixture


def rejected_audit(memory, scene):
    result = {"verdict": "fail", "memory_hash": memory_hash(memory), "checked_knowledge_ids": ["k1"],
        "facts": [{"fact_id": "f1", "knowledge_record_id": "k1", "source_ids": ["src001"]}],
        "missing_information": [], "comparisons": [{"fact_id": "f1", "outcome": "unexplained_change",
            "comparison": "前の場面で確定した内容を根拠なく変えた。",
            "earlier_evidence": [{"source_id": "src001", "quote": "確定した以前の発言"}],
            "later_evidence": [{"source_id": "src002",
                                "quote": scene.utterances[0].display_text}],
            "explanation_of_change_evidence": []}]}
    result["resolved_comparisons"] = copy.deepcopy(result["comparisons"])
    result["resolved_comparisons"][0]["later_evidence"][0]["ref"] = source_catalog(
        [scene], storyline_id=memory.storyline_id, chapter_number=1)[0].model_dump(mode="json")
    return result


def test_fact_rejection_feedback_preserves_exact_pairs_and_current_source_ref(monkeypatch):
    harness = Harness(monkeypatch)
    with CausalRun(NoServerLLM(), {"storyline_id": "story", "chapter_number": 1}) as run:
        scene = harness.write(run, {}, harness.initial, planning(1, "s1"), {}, [])
        monkeypatch.setattr(causal, "compare_scene_facts", lambda *_: rejected_audit(harness.initial, scene))
        with pytest.raises(causal.GateError) as rejected:
            causal._fact_review(run, harness.initial, scene)
    feedback = rejected.value.feedback()
    comparison = feedback["retrieved_evidence"][0]["fact_comparison"]["comparisons"][0]
    assert comparison["earlier_evidence"][0]["quote"] == "確定した以前の発言"
    assert comparison["later_evidence"][0]["quote"] == scene.utterances[0].display_text
    issue = rejected.value.report.issues[0]
    assert issue.repair_scope == "scene" and issue.evidence[0].utterance_id == scene.utterances[0].id


def test_fact_failure_repairs_only_the_responsible_scene(monkeypatch):
    harness = Harness(monkeypatch)
    failed = False

    def compare(run, memory, scene):
        nonlocal failed
        if scene.id == "s2" and not failed:
            failed = True
            return rejected_audit(memory, scene)
        return {"verdict": "pass", "facts": [], "comparisons": [], "resolved_comparisons": [], "checked_knowledge_ids": [],
                "memory_hash": memory_hash(memory), "missing_information": []}

    monkeypatch.setattr(causal, "compare_scene_facts", compare)
    result = harness.chapter()
    assert [scene for _, scene, _ in harness.writes] == ["s1", "s2", "s2"]
    assert len(result.scene_extractions) == 2
    assert [issue["scope"] for issue in harness.runs[-1].state["issues"]] == ["scene"]


@pytest.mark.parametrize("change", ["missing", "stale"])
@pytest.mark.parametrize("scope", ["fact-comparison-s1", "state-comparison-s1"])
def test_adoption_requires_source_comparison_bound_to_its_actual_subject(change, scope):
    result, _, cast = causal_fixture()
    reports = copy.deepcopy(result.workflow_reviews)
    if change == "missing":
        reports = [report for report in reports if report.scope != scope]
    else:
        reports = [report.model_copy(update={"subject_hash": "f" * 64})
                   if report.scope == scope else report for report in reports]
    with pytest.raises(ValueError, match="every required stage|stale subject"):
        validate_causal_continuity(result.model_copy(update={"workflow_reviews": reports}), cast)
