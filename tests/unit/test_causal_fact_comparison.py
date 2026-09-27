"""Fact comparisons quote original sources, cover all knowledge, and fail closed."""

import copy
import json

import pytest

from packages.contracts.m3 import NarrativeScene
from packages.narrative.story_ledger import (
    apply_scene_memory,
    empty_memory,
    source_catalog,
    source_excerpts,
)
from packages.narrative.validation import parse_scene_text
from services.worker.generation.causal_fact_comparison import (
    SYSTEM,
    compare_scene_facts,
    fact_comparison_request,
    prepare_fact_comparison,
    validate_fact_assessment,
)
from services.worker.generation.causal_runtime import CausalRun, RepairLimitError
from services.worker.generation.llm import ContextBudgetError
from services.worker.generation.narrative import StructuredGenerationError
from tests.unit.test_m3_narrative import narrative_fixture


def scene(text, scene_id="s1"):
    document, _ = narrative_fixture()
    value = document["scenes"][0]
    value.update(id=scene_id, raw_text=text, utterances=[u.model_dump() for u in
        parse_scene_text(text, scene_id, {"Hero", "keeper"})])
    value["plan"]["id"] = scene_id
    return NarrativeScene.model_validate(value)


def sources(count=1):
    prior = scene("Hero: 記憶と名簿の日付が合わない。\nkeeper: 不一致を確認した。")
    refs = source_catalog([prior], storyline_id="story", chapter_number=1)
    memory = apply_scene_memory(empty_memory("story"), {
        "chapter_number": 1, "summary": "日付の食い違いを共有した。", "events": [{
            "id": "date-mismatch", "description": "記憶と名簿の日付が合わないと話した。",
            "story_time": "開始時", "character_ids": ["Hero", "keeper"], "location_ids": ["gate"],
            "evidence": refs}], "knowledge_updates": [{
                "id": f"knowledge-{index}", "character_id": "Hero" if index % 2 else "keeper",
                "fact_id": f"fact-{index}", "content": "記憶と名簿の日付が合わない。",
                "acquired_at": "開始時", "event_ids": ["date-mismatch"], "evidence": [refs[0]]}
                for index in range(1, count + 1)]}, scenes=[prior], storyline_id="story", chapter_number=1)
    current = scene("keeper: 記憶と名簿の日付が合わない。", "s2")
    return memory, current


class InspectorLLM:
    def __init__(self, *, output=None, damage=None, overflow_above=None):
        self.config = {"llm": {"model_id": "fake", "temperature": 0.1, "max_tokens": 4096}}
        self.payload = {"storyline_id": "story", "chapter_number": 2, "story_workflow_version": 2}
        self.trace, self.calls, self.requests, self.schemas = [], [], 0, []
        self.damage, self.overflow_above = damage, overflow_above
        if output is not None:
            self.output = output

    def runtime_identity(self):
        return {"model": "fake"}

    def chat(self, stage, messages, **kwargs):
        self.requests += 1
        assert messages[0] == {"role": "system", "content": SYSTEM}
        prompt = messages[-1]["content"]
        inputs, _ = json.JSONDecoder().raw_decode(prompt.split("資料: ", 1)[1])
        self.calls.append(inputs)
        self.schemas.append(kwargs["response_format"]["json_schema"]["schema"])
        facts = inputs["earlier_facts"]
        if self.overflow_above is not None and len(facts) > self.overflow_above:
            self.trace.append({"type": "context_budget", "fits": False, "prompt_tokens": 20000})
            raise ContextBudgetError("fake oversized comparison")
        self.trace.append({"type": "llm_generation", "cache_hit": False,
                           "usage": {"prompt_tokens": 100, "completion_tokens": 20}})
        later = inputs["later_source"][0]
        earlier = {item["source_id"]: item for item in inputs["earlier_source"]}
        rows = [{"fact_id": fact["fact_id"], "earlier_evidence": [{
                    "source_id": fact["source_ids"][0], "quote": earlier[fact["source_ids"][0]]["text"]}],
                 "later_evidence": [{"source_id": later["source_id"], "quote": later["text"]}],
                 "comparison": "前後で同じ情報を述べている。", "explanation_of_change_evidence": [],
                 "outcome": "consistent"} for fact in facts]
        if self.damage:
            self.damage(rows, inputs)
        return {"content": json.dumps({"comparisons": rows}, ensure_ascii=False)}


def compare(llm, memory, current, **kwargs):
    with CausalRun(llm, llm.payload) as run:
        return compare_scene_facts(run, memory, current, **kwargs)


def test_consistent_and_unexplained_change_return_exact_sources_without_changing_memory():
    memory, current = sources(2)
    before = memory.model_dump()
    good = compare(InspectorLLM(), memory, current)
    assert good["verdict"] == "pass"
    assert good["checked_knowledge_ids"] == ["knowledge-1", "knowledge-2"]
    assert [row["fact_id"] for row in good["comparisons"]] == ["f1", "f2"]
    assert good["earlier_source"][0]["text"] == "記憶と名簿の日付が合わない。"

    def unexplained(rows, inputs):
        rows[0].update(outcome="unexplained_change", comparison="日付の不一致を記録不存在へ変更している。")

    negative = scene("keeper: 名簿に載っていない。", "s2")
    bad = compare(InspectorLLM(damage=unexplained), memory, negative)
    assert bad["verdict"] == "fail"
    assert bad["comparisons"][0]["later_evidence"][0]["quote"] == "名簿に載っていない。"
    assert memory.model_dump() == before


@pytest.mark.parametrize("damage", [
    lambda rows, inputs: rows.pop(),
    lambda rows, inputs: rows[0].update(fact_id="invented"),
    lambda rows, inputs: rows[0]["earlier_evidence"][0].update(quote="原文にない過去"),
    lambda rows, inputs: rows[0]["earlier_evidence"][0].update(source_id="s2-u1"),
    lambda rows, inputs: rows[0]["later_evidence"][0].update(quote="原文にない現在"),
    lambda rows, inputs: rows[0].update(later_evidence=[]),
    lambda rows, inputs: rows[0].update(outcome="not_referred_to"),
    lambda rows, inputs: rows[0].update(outcome="unexplained_change",
        explanation_of_change_evidence=copy.deepcopy(rows[0]["later_evidence"])),
])
def test_missing_coverage_wrong_quotes_and_unsupported_outcomes_are_not_accepted(damage):
    memory, current = sources(2)
    llm = InspectorLLM(damage=damage)
    with pytest.raises(StructuredGenerationError):
        compare(llm, memory, current)
    assert len(llm.calls) == 2


@pytest.mark.parametrize("outcome,verdict", [("not_referred_to", "pass"), ("insufficient_evidence", "insufficient_evidence")])
def test_unmentioned_facts_and_insufficient_evidence_are_distinct(outcome, verdict):
    memory, current = sources()

    def damage(rows, inputs):
        rows[0].update(outcome=outcome, later_evidence=[], comparison="後の場面に対応する情報がない。")

    result = compare(InspectorLLM(damage=damage), memory, current)
    assert result["verdict"] == verdict
    assert bool(result["missing_information"]) == (verdict == "insufficient_evidence")


@pytest.mark.parametrize("damage", ["missing", "changed"])
def test_missing_or_changed_adopted_original_stops_before_any_model_request(damage):
    memory, current = sources()
    damaged = memory.model_copy(update={"sources": [] if damage == "missing" else [
        source.model_copy(update={"text": "異なる原文"}) for source in memory.sources]})
    llm = InspectorLLM()
    result = compare(llm, damaged, current)
    assert result["verdict"] == "insufficient_evidence" and result["missing_information"]
    assert llm.calls == []


def test_only_latest_knowledge_of_current_characters_is_compared_without_merging_beliefs():
    memory, _ = sources(3)
    corrected = memory.knowledge[0].model_copy(update={
        "id": "knowledge-revised", "kind": "uncertain", "supersedes_id": "knowledge-1"})
    memory = memory.model_copy(update={"knowledge": [*memory.knowledge, corrected]})
    current = scene("Hero: 記憶と名簿の日付が合わない。", "s2")
    current = current.model_copy(update={"plan": current.plan.model_copy(update={"character_ids": ["Hero"]})})
    result = compare(InspectorLLM(), memory, current)
    assert result["checked_knowledge_ids"] == ["knowledge-revised", "knowledge-3"]
    assert result["facts"][0]["knowledge_kind"] == "uncertain"
    assert result["facts"][0]["supersedes_id"] == "knowledge-1"
    assert all(fact["character_id"] == "Hero" for fact in result["facts"])


def test_empty_knowledge_needs_no_model_request():
    _, current = sources()
    llm = InspectorLLM()
    result = compare(llm, empty_memory("story"), current)
    assert result["verdict"] == "pass" and result["checked_knowledge_ids"] == [] and llm.calls == []


def test_budget_split_preserves_every_fact_and_complete_current_source(tmp_path):
    memory, current = sources(9)
    llm = InspectorLLM(output=tmp_path / "llm-v7", overflow_above=2)
    result = compare(llm, memory, current)
    assert [row["fact_id"] for row in result["comparisons"]] == [f"f{i}" for i in range(1, 10)]
    assert all(call["later_source"] == result["later_source"] for call in llm.calls)
    assert len(result["batches"]) == 5
    assert len([entry for entry in llm.trace if entry["type"] == "fact_comparison_split"]) == 3
    assert all(len(batch["fact_ids"]) <= 2 for batch in result["batches"])


def test_an_indivisible_context_error_is_not_silently_omitted():
    memory, current = sources()
    llm = InspectorLLM(overflow_above=0)
    with pytest.raises(ContextBudgetError):
        compare(llm, memory, current)
    assert len(llm.calls) == 1


def test_split_attempts_share_the_chapter_call_budget():
    memory, current = sources(2)
    llm = InspectorLLM(overflow_above=1)
    llm.payload["workflow_limits"] = {"max_calls": 1}
    with pytest.raises(RepairLimitError, match="call budget"):
        compare(llm, memory, current)
    assert len(llm.calls) == 1


def test_completed_comparisons_resume_from_validated_stage_cache(tmp_path):
    memory, current = sources(2)
    first = compare(InspectorLLM(output=tmp_path / "llm-v7"), memory, current)
    resumed = InspectorLLM(output=tmp_path / "llm-v7")
    second = compare(resumed, memory, current)
    assert second == first and resumed.calls == []


def test_cached_comparisons_still_require_exact_quoted_source(tmp_path):
    memory, current = sources()
    compare(InspectorLLM(output=tmp_path / "llm-v7"), memory, current)
    path = next((tmp_path / "causal-stages").glob("fact-comparison-*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    saved["value"]["assessment"]["comparisons"][0]["earlier_evidence"][0]["quote"] = "書き換えた引用"
    path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="exact original source"):
        compare(InspectorLLM(output=tmp_path / "llm-v7"), memory, current)


def separate_facts():
    memory, current = sources(2)
    records = [memory.knowledge[0], memory.knowledge[1].model_copy(update={
        "evidence": [memory.sources[1].ref], "content": memory.sources[1].text})]
    return memory.model_copy(update={"knowledge": records}), current


def test_request_has_single_source_namespace_with_fact_specific_schema_and_temporal_meaning():
    memory, current = separate_facts()
    first = memory.knowledge[0].model_copy(update={"acquired_at": "s1-u1の説明を聞いた時"})
    memory = memory.model_copy(update={"knowledge": [first, memory.knowledge[1]],
        "events": [memory.events[0].model_copy(update={"story_time": "十年前", "presentation": "flashback"})]})
    snapshot = prepare_fact_comparison(memory, current, chapter_number=2)
    request = fact_comparison_request(snapshot)
    encoded_request = json.dumps(request, ensure_ascii=False)
    for old in ("s1-u1", "s1-u2", "s2-u1", "utterance_id", '"q1"', "knowledge-1", "date-mismatch"):
        assert old not in encoded_request
    f1 = request["inputs"]["earlier_facts"][0]
    assert f1["acquisition"] == {"description": "src001の説明を聞いた時",
        "source_ids": ["src001"], "reference_status": "resolved"}
    assert f1["disclosed_chapter"] == 1
    assert f1["event_times"][0]["story_time"]["description"] == "十年前"
    assert f1["event_times"][0]["presentation"] == "flashback"
    assert request["inputs"]["earlier_source"][0]["speaker"] == "Hero"
    branches = request["schema"]["properties"]["comparisons"]["items"]["anyOf"]
    for branch, fid, allowed in zip(branches, ["f1", "f2"], [["src001"], ["src002"]], strict=True):
        assert branch["properties"]["fact_id"]["enum"] == [fid]
        assert branch["properties"]["earlier_evidence"]["items"]["properties"]["source_id"]["enum"] == allowed
        assert branch["properties"]["later_evidence"]["items"]["properties"]["source_id"]["enum"] == ["src003"]
    subset = fact_comparison_request(snapshot, fact_ids=["f2"])
    assert subset["inputs"]["earlier_source"][0]["source_id"] == "src002"
    assert subset["snapshot_hash"] == request["snapshot_hash"]


def test_another_facts_valid_source_is_rejected_even_if_quote_matches():
    memory, current = separate_facts()

    def damage(rows, inputs):
        source = inputs["earlier_source"][1]
        rows[0]["earlier_evidence"] = [{"source_id": source["source_id"], "quote": source["text"]}]

    with pytest.raises(StructuredGenerationError, match="Invalid source_id"):
        compare(InspectorLLM(damage=damage), memory, current)


@pytest.mark.parametrize("role", ["earlier_evidence", "later_evidence"])
def test_valid_id_from_wrong_role_is_rejected(role):
    memory, current = sources()

    def damage(rows, inputs):
        source = inputs["later_source" if role == "earlier_evidence" else "earlier_source"][0]
        rows[0][role] = [{"source_id": source["source_id"], "quote": source["text"]}]

    with pytest.raises(StructuredGenerationError, match="Invalid source_id"):
        compare(InspectorLLM(damage=damage), memory, current)


def test_insufficient_evidence_can_return_no_quote_but_other_decisions_cannot():
    memory, current = sources()

    def missing(rows, inputs):
        rows[0].update(earlier_evidence=[], later_evidence=[], outcome="insufficient_evidence",
                       comparison="前の発言がどの写真の話か曖昧で対応を決められない。")

    result = compare(InspectorLLM(damage=missing), memory, current)
    assert result["verdict"] == "insufficient_evidence"
    with pytest.raises(StructuredGenerationError, match="earlier source evidence"):
        compare(InspectorLLM(damage=lambda rows, _: rows[0].update(earlier_evidence=[])), memory, current)


def test_duplicate_quotes_are_not_counted_as_extra_evidence():
    memory, current = sources()

    def repeat(rows, inputs):
        rows[0]["earlier_evidence"] *= 2

    with pytest.raises(StructuredGenerationError, match="Duplicate evidence"):
        compare(InspectorLLM(damage=repeat), memory, current)


def test_complete_refs_remain_in_snapshot_and_resolved_output_not_model_prompt():
    memory, current = sources()
    result = compare(InspectorLLM(), memory, current)
    resolved = result["resolved_comparisons"][0]
    assert resolved["earlier_evidence"][0]["ref"] == memory.knowledge[0].evidence[0].model_dump(mode="json")
    assert resolved["later_evidence"][0]["ref"] == source_catalog(
        [current], storyline_id="story", chapter_number=2)[0].model_dump(mode="json")
    assert "ref" not in result["comparisons"][0]["earlier_evidence"][0]


@pytest.mark.parametrize("part", ["fact", "mapping", "memory_version"])
def test_request_rejects_mutated_fixed_snapshot(part):
    memory, current = separate_facts()
    snapshot = prepare_fact_comparison(memory, current, chapter_number=2).model_dump(mode="json")
    if part == "fact":
        snapshot["facts"][0]["source_ids"] = ["src002"]
    elif part == "mapping":
        snapshot["catalog"]["sources"][0]["ref"]["scene_revision"] = "f" * 64
    else:
        snapshot["memory_hash"] = "f" * 64
    with pytest.raises(ValueError):
        fact_comparison_request(snapshot)


def test_stale_revision_evidence_does_not_resolve_by_matching_text_or_utterance_id():
    memory, current = sources()
    changed_ref = memory.knowledge[0].evidence[0].model_copy(update={"scene_revision": "f" * 64})
    memory = memory.model_copy(update={"knowledge": [memory.knowledge[0].model_copy(update={"evidence": [changed_ref]})]})
    llm = InspectorLLM()
    result = compare(llm, memory, current)
    assert result["verdict"] == "insufficient_evidence" and not llm.calls


def test_ambiguous_legacy_acquisition_pointer_is_explicit_and_never_picks_first_chapter():
    memory, current = sources()
    earlier_again = scene("Hero: 別の章でも情報を確認した。")
    extra = source_excerpts([earlier_again], storyline_id="story", chapter_number=2)[0]
    knowledge = memory.knowledge[0].model_copy(update={
        "evidence": [memory.knowledge[0].evidence[0], extra.ref], "acquired_at": "s1-u1"})
    memory = memory.model_copy(update={"chapter_number": 2, "sources": [*memory.sources, extra], "knowledge": [knowledge]})
    request = fact_comparison_request(prepare_fact_comparison(memory, current, chapter_number=3))
    acquisition = request["inputs"]["earlier_facts"][0]["acquisition"]
    assert acquisition["reference_status"] == "unresolved" and acquisition["source_ids"] == []
    assert "s1-u1" not in acquisition["description"]
    assert len(request["inputs"]["earlier_source"]) == 2


def test_saved_stage_contains_exact_catalog_and_revalidates_it_on_cache_hit(tmp_path):
    memory, current = sources()
    compare(InspectorLLM(output=tmp_path / "llm-v7"), memory, current)
    path = next((tmp_path / "causal-stages").glob("fact-comparison-*.json"))
    saved = json.loads(path.read_text(encoding="utf-8"))
    snapshot = saved["value"]["snapshot"]
    assert snapshot["catalog"]["sources"][0]["ref"] == memory.knowledge[0].evidence[0].model_dump(mode="json")
    snapshot["facts"][0]["source_ids"] = ["src002"]
    path.write_text(json.dumps(saved, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="snapshot hash"):
        compare(InspectorLLM(output=tmp_path / "llm-v7"), memory, current)


def test_public_assessment_validator_checks_order_and_returns_source_bound_quotes():
    memory, current = sources(2)
    snapshot = prepare_fact_comparison(memory, current, chapter_number=2)
    result = compare(InspectorLLM(), memory, current)
    assert validate_fact_assessment({"comparisons": result["comparisons"]}, snapshot) == result["resolved_comparisons"]
    with pytest.raises(ValueError, match="every supplied fact"):
        validate_fact_assessment({"comparisons": list(reversed(result["comparisons"]))}, snapshot)
