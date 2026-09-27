"""Fixed source/after pairs test provenance, phase promotion and local failures.

Model responses below are controlled observations and comparisons, not claims of
real-model quality. Live source-first discrimination requires the separate probe.
"""

import copy
import json

import pytest
from pydantic import ValidationError

from packages.contracts.m3 import NarrativeScene
from packages.contracts.story_workflow import ChapterExtraction
from packages.narrative.story_ledger import source_catalog, text_hash
from packages.narrative.validation import parse_scene_text
from services.worker.generation import causal_state_observation as state_audit
from services.worker.generation.causal_runtime import digest
from services.worker.generation.causal_state_observation import (
    INSPECTOR_SYSTEM,
    REMOVED_STATE,
    ComparisonDraft,
    ObservationDraft,
    compare_state_deltas,
    observe_scene_state,
    review_state_deltas,
)
from services.worker.generation.narrative import StructuredGenerationError
from tests.unit.test_m3_narrative import FakeLLM, narrative_fixture

REQUEST_SOURCE = (
    "NARRATOR: 係は写真の横にテープを一枚貼った。\n"
    "keeper: ここを保留にする。\n"
    "Hero: 分かった。二枚から配置し直して。\n"
    "keeper: 決まりだな。\n"
    "NARRATOR: 係はパネルへ手を伸ばした。主人公は白紙の説明札を机に戻し、ノートに「不詳」と書いた。"
)


def scene(raw=REQUEST_SOURCE):
    value = copy.deepcopy(narrative_fixture()[0]["scenes"][0])
    value["raw_text"] = raw
    value["utterances"] = [u.model_dump() for u in parse_scene_text(raw, "s1", {"Hero", "keeper"})]
    return NarrativeScene.model_validate(value)


def observation(utterance, quote, *, target="写真二枚", action="再配置", quantity="二枚",
                kind="physical", stage="requested", actor_ids=None, **changes):
    value = {"actor_ids": ["keeper"] if actor_ids is None else actor_ids,
        "target": target, "action": action, "quantity": quantity, "kind": kind, "stage": stage,
        "assertion": "observed", "polarity": "affirmed", "time_scope": "current",
        "evidence": [{"utterance_id": utterance, "quote": quote}],
        "reason": "原文でこの対象と実行段階が示されている。"}
    value.update(changes)
    return value


def annotation(current, rows=None, **changes):
    rows = rows if rows is not None else [
        observation("s1-u3", "二枚から配置し直して。"),
        observation("s1-u4", "決まりだな。", target="二枚の再配置方針", kind="agreement",
                    action="再配置に合意する", stage="established", actor_ids=["Hero", "keeper"]),
        observation("s1-u1", "写真の横にテープを一枚貼った。", target="写真の横", action="テープを貼る",
                    quantity="一枚", stage="completed"),
        observation("s1-u5", "パネルへ手を伸ばした。", target="パネル", action="手を伸ばす",
                    quantity=None, stage="completed"),
        observation("s1-u5", "ノートに「不詳」と書いた。", target="ノート", action="不詳と書く",
                    quantity=None, stage="completed", actor_ids=["Hero"]),
        observation("s1-u5", "白紙の説明札を机に戻し", target="説明札", action="机へ戻す",
                    quantity=None, stage="completed", actor_ids=["Hero"]),
    ]
    value = {"inspected_utterance_ids": [u.id for u in current.utterances],
             "observations": rows, "no_observations_reason": None, "missing_information": []}
    value.update(changes)
    return value


def extraction(after="写真二枚を再配置済み。", *, assertion="observed", time_scope="current"):
    return {"summary": "作業方針を話し合う。",
        "events": [{"id": "e1", "description": "二枚を再配置することで合意した。",
                    "assertion": assertion, "presentation": "current", "story_time": "今",
                    "evidence": ["s1-u3", "s1-u4"]}],
        "state_deltas": [{"id": "d1", "scope": "location", "entity_id": "gate", "key": "panel_status",
            "before": None, "after": after, "event_id": "e1", "time_scope": time_scope,
            "evidence": ["s1-u3", "s1-u4"]}], "thread_updates": []}


def match(oid="o1", relation="supports", **changes):
    value = {"observation_id": oid, "actor_matches": True, "target_matches": True,
             "action_matches": True, "quantity_matches": True, "relation": relation,
             "reason": "この原文の動作段階と対象を候補の主張に照合した。"}
    value.update(changes)
    return value


def claim(after, *, kind="physical", stage="completed", matches=None, **changes):
    value = {"after_quote": after, "kind": kind, "stage": stage, "polarity": "affirmed",
        "time_scope": "current", "matches": [match()] if matches is None else matches,
        "missing_information": None}
    value.update(changes)
    return value


def comparison(after="写真二枚を再配置済み。", *, claims=None):
    return {"comparisons": [{"delta_id": "d1", "claims": claims or [claim(after)]}]}


class Run:
    def __init__(self, *responses):
        self.payload = {"chapter_number": 1, "storyline_id": "fixed-audit-source"}
        self.llm = FakeLLM([{"content": json.dumps(value, ensure_ascii=False), "_finish_reason": "stop"}
                            for value in responses])
        self.cache, self.nodes = {}, []

    def node(self, stage, inputs, model, produce):
        self.nodes.append((stage, inputs))
        key = (stage, digest(inputs))
        if key not in self.cache:
            result = produce()
            self.cache[key] = result.model_dump(mode="json")
        return model.model_validate(self.cache[key])


@pytest.mark.parametrize("assertion", ["believed", "observed"])
def test_request_is_not_promoted_to_completed_even_after_event_label_is_fixed(assertion):
    current, candidate = scene(), extraction(assertion=assertion)
    before_source, before_extraction = current.model_dump(), copy.deepcopy(candidate)
    run = Run(annotation(current), comparison())
    result = review_state_deltas(run, current, candidate)
    assert result.report.verdict == "fail"
    issue = result.report.issues[0]
    assert "d1" in issue.description and "requested" in issue.description
    assert issue.repair_scope == "extraction"
    assert issue.evidence[0].utterance_id == "s1-u3"
    assert result.report.subject_hash == digest({"scene": current.model_dump(mode="json"),
                                               "extraction": candidate})
    assert result.report.scope == "state-comparison-s1"
    assert result.report.checked_categories == ["state_support", "action_stage", "object_identity"]
    assert result.report.checked_scene_ids == ["s1"]
    assert candidate == before_extraction and current.model_dump() == before_source


def test_minimal_agreement_delta_and_marker_result_pass_against_the_same_body():
    current = scene()
    after = "二枚の再配置に合意した。写真の横へテープ一枚を貼った。"
    response = comparison(claims=[
        claim("二枚の再配置に合意した。", kind="agreement", stage="established", matches=[match("o2")]),
        claim("写真の横へテープ一枚を貼った。", matches=[match("o3")]),
    ])
    result = review_state_deltas(Run(annotation(current), response), current, extraction(after))
    assert result.report.verdict == "pass"
    assert result.report.checked_event_ids == ["e1"]


def test_expanded_ledger_extraction_has_exact_binding_and_short_prompt_references():
    current, raw = scene(), extraction()
    run = Run(annotation(current), comparison())
    refs = {ref.utterance_id: ref.model_dump() for ref in source_catalog(
        [current], storyline_id=run.payload["storyline_id"], chapter_number=1)}
    raw["chapter_number"] = 1
    for group in ("events", "state_deltas"):
        for record in raw[group]:
            record["evidence"] = [refs[uid] for uid in record["evidence"]]
    expanded = ChapterExtraction.model_validate(raw)
    result = review_state_deltas(run, current, expanded)
    assert result.report.subject_hash == digest({"scene": current.model_dump(mode="json"),
                                               "extraction": expanded.model_dump(mode="json")})
    assert "source_start" not in encoded_json(run.nodes[1][1]["inputs"])
    assert run.nodes[1][1]["inputs"]["candidate_deltas"][0]["evidence_ids"] == ["s1-u3", "s1-u4"]


@pytest.mark.parametrize("field,description", [
    ("target_matches", "説明札へ不詳と記入した。"),
    ("quantity_matches", "写真の横へテープ二枚を貼った。"),
    ("actor_matches", "係がノートへ不詳と記入した。"),
    ("action_matches", "ノートから不詳の文字を消した。"),
])
def test_wrong_target_quantity_actor_or_action_cannot_support_a_result(field, description):
    current = scene()
    row_id = "o3" if field == "quantity_matches" else "o5"
    response = comparison(description, claims=[claim(description, matches=[match(row_id, **{field: False})])])
    result = review_state_deltas(Run(annotation(current), response), current, extraction(description))
    assert result.report.verdict == "fail"
    assert "actor/target/action/quantity" in result.report.issues[0].description


def test_notebook_writing_is_supported_and_is_not_moved_to_caption():
    current, after = scene(), "ノートへ不詳と記入した。"
    result = review_state_deltas(Run(annotation(current), comparison(claims=[
        claim(after, matches=[match("o5")])])), current, extraction(after))
    assert result.report.verdict == "pass"
    quote = result.observations.observations[4].verified_evidence[0]
    assert current.raw_text[quote.quote_start:quote.quote_end] == "ノートに「不詳」と書いた。"
    assert quote.quote_hash == text_hash(quote.quote)


@pytest.mark.parametrize("first_relation", ["does_not_establish", "supports"])
def test_same_action_request_then_actual_completion_is_supported(first_relation):
    current = scene("Hero: 二枚を入れ替えて。\nNARRATOR: 係は二枚の写真の位置を入れ替えた。\n"
                    "NARRATOR: 主人公は説明札を取ろうと手を伸ばした。")
    rows = [observation("s1-u1", "二枚を入れ替えて。"),
        observation("s1-u2", "二枚の写真の位置を入れ替えた。", stage="completed"),
        observation("s1-u3", "説明札を取ろうと手を伸ばした。", target="説明札", action="取る",
                    quantity=None, stage="attempted", actor_ids=["Hero"])]
    after = "写真二枚を再配置済み。"
    response = comparison(claims=[claim(after, matches=[match("o1", first_relation), match("o2"),
        match("o3", "does_not_establish", target_matches=False, actor_matches=False)])])
    result = review_state_deltas(Run(annotation(current, rows), response), current, extraction(after))
    assert result.report.verdict == "pass"
    assert "完了" not in current.raw_text


def test_actual_contradiction_is_not_erased_by_another_supporting_match():
    current = scene("NARRATOR: 係は二枚の写真の位置を入れ替えた。\n"
                    "NARRATOR: その後、二枚を元の位置へ戻した。")
    rows = [observation("s1-u1", "二枚の写真の位置を入れ替えた。", stage="completed"),
            observation("s1-u2", "二枚を元の位置へ戻した。", stage="completed")]
    response = comparison(claims=[claim("写真二枚を再配置済み。", matches=[
        match("o1"), match("o2", "contradicts", reason="後の復元で再配置済みの現在状態は失効した。")])])
    result = review_state_deltas(Run(annotation(current, rows), response), current, extraction())
    assert result.report.verdict == "fail" and "contradicts" in result.report.issues[0].description


@pytest.mark.parametrize("stage", ["requested", "intended", "agreed", "attempted", "in_progress", "unknown"])
def test_incomplete_stages_cannot_become_completed(stage):
    current = scene()
    rows = annotation(current)
    rows["observations"][0]["stage"] = stage
    result = review_state_deltas(Run(rows, comparison()), current, extraction())
    assert result.report.verdict == "fail"


@pytest.mark.parametrize("assertion", ["reported", "believed", "hypothetical"])
def test_report_or_belief_does_not_become_current_physical_result(assertion):
    current = scene()
    rows = annotation(current)
    rows["observations"][0].update(stage="completed", assertion=assertion)
    result = review_state_deltas(Run(rows, comparison()), current, extraction())
    assert result.report.verdict == "fail"
    assert "report, belief or hypothesis" in result.report.issues[0].description


def test_historical_observation_does_not_mutate_present():
    current = scene()
    rows = annotation(current)
    rows["observations"][0].update(stage="completed", time_scope="historical")
    result = review_state_deltas(Run(rows, comparison()), current, extraction())
    assert result.report.verdict == "fail"
    assert "historical" in result.report.issues[0].description


def test_partial_after_coverage_cannot_hide_the_false_completion_clause():
    current = scene()
    after = "写真の横へテープ一枚を貼った。写真二枚を再配置済み。"
    response = comparison(claims=[claim("写真の横へテープ一枚を貼った。", matches=[match("o3")])])
    run = Run(annotation(current), response, response)
    with pytest.raises(StructuredGenerationError, match="entire after"):
        review_state_deltas(run, current, extraction(after))
    assert len(run.cache) == 1  # The rejected comparison is never a successful node.
    assert run.llm.requests == 3  # One A and the bounded two B attempts.


@pytest.mark.parametrize("change,error", [
    (lambda value: value["comparisons"].clear(), "every delta"),
    (lambda value: value["comparisons"][0].update(delta_id="absent"), "every delta"),
    (lambda value: value["comparisons"][0]["claims"][0]["matches"][0].update(observation_id="absent"),
     "observation IDs"),
    (lambda value: value["comparisons"][0]["claims"][0].update(time_scope="historical"), "scope"),
])
def test_missing_delta_and_invented_or_retimed_references_are_rejected(change, error):
    current, response = scene(), comparison()
    change(response)
    with pytest.raises(StructuredGenerationError, match=error):
        review_state_deltas(Run(annotation(current), response, response), current, extraction())


@pytest.mark.parametrize("change,error", [
    (lambda value: value["inspected_utterance_ids"].pop(), "all source utterances"),
    (lambda value: value["observations"][0]["evidence"][0].update(quote="二枚を配置し終えた。"), "exact"),
    (lambda value: value["observations"][0]["evidence"][0].update(utterance_id="s2-u1"), "current utterance"),
    (lambda value: value["observations"][0].update(actor_ids=["ghost"]), "current character"),
])
def test_source_quotes_and_coverage_are_validated_before_observation_cache(change, error):
    current = scene()
    response = annotation(current)
    change(response)
    run = Run(response, response)
    with pytest.raises(StructuredGenerationError, match=error):
        observe_scene_state(run, current)
    assert run.cache == {} and run.llm.requests == 2


def test_ambiguous_quote_requires_wider_source_span():
    current = scene("Hero: 待って、待って。")
    response = annotation(current, [observation("s1-u1", "待って")])
    with pytest.raises(StructuredGenerationError, match="unambiguous"):
        observe_scene_state(Run(response, response), current)


def test_missing_source_information_is_not_a_pass_and_skips_comparison():
    current = scene()
    response = annotation(current, missing_information=["写真の指示対象を原文から区別できない。"])
    run = Run(response)
    result = review_state_deltas(run, current, extraction())
    assert result.report.verdict == "insufficient_evidence"
    assert result.report.missing_information == response["missing_information"]
    assert run.llm.requests == 1


def test_unmatched_claim_preserves_specific_missing_information():
    current, after = scene(), "写真の設置数が変化した。"
    response = comparison(claims=[claim(after, matches=[], missing_information="配置された枚数が未観測。")])
    result = review_state_deltas(Run(annotation(current), response), current, extraction(after))
    assert result.report.verdict == "insufficient_evidence"
    assert result.report.missing_information == ["d1: 配置された枚数が未観測。"]


def test_null_after_is_explicitly_compared_as_removing_the_prior_state():
    current = scene("NARRATOR: 係はパネルを覆う布を外した。")
    rows = [observation("s1-u1", "パネルを覆う布を外した。", target="布", action="外す",
                        quantity=None, stage="completed")]
    candidate = extraction(None)
    candidate["state_deltas"][0]["before"] = "パネルを布が覆っている。"
    result = review_state_deltas(Run(annotation(current, rows), comparison(REMOVED_STATE)), current, candidate)
    assert result.report.verdict == "pass"


def test_empty_deltas_are_explicitly_outside_this_false_completion_audit():
    current, candidate = scene(), extraction()
    candidate["state_deltas"] = []
    run = Run(annotation(current))
    result = review_state_deltas(run, current, candidate)
    assert result.report.verdict == "pass" and result.comparisons == []
    assert "Omitted deltas" in result.report.rationale
    assert run.llm.requests == 1


def test_inspector_prompts_are_independent_and_have_only_compact_reference_metadata():
    current, candidate = scene(), extraction()
    run = Run(annotation(current), comparison())
    review_state_deltas(run, current, candidate)
    a_inputs = run.nodes[0][1]
    assert set(a_inputs["source"]) == {"scene_id", "scene_revision", "character_ids", "location_id", "utterances"}
    assert "after" not in encoded_json(a_inputs) and "candidate_events" not in encoded_json(a_inputs)
    assert "objectives" not in encoded_json(a_inputs) and "author_facts" not in encoded_json(a_inputs)
    assert a_inputs["schema"]["$defs"]["RawStateObservation"]["properties"]["actor_ids"]["items"]["enum"] == [
        "Hero", "keeper"]
    assert "id" not in a_inputs["schema"]["$defs"]["RawStateObservation"]["properties"]
    for args, _kwargs in run.llm.calls:
        assert args[1][0] == {"role": "system", "content": INSPECTOR_SYSTEM}
    b_inputs = run.nodes[1][1]["inputs"]
    assert b_inputs["extraction_hash"] == digest(candidate)
    assert "verified_evidence" not in encoded_json(b_inputs)
    assert "source_start" not in encoded_json(b_inputs)
    assert "quote_hash" not in encoded_json(b_inputs)
    assert b_inputs["candidate_deltas"][0]["evidence_ids"] == ["s1-u3", "s1-u4"]


def encoded_json(value):
    return json.dumps(value, ensure_ascii=False)


def test_changing_only_extraction_reuses_source_observation_node():
    current = scene()
    correct = "二枚の再配置に合意した。"
    good_comparison = comparison(claims=[claim(correct, kind="agreement", stage="established",
                                             matches=[match("o2")])])
    run = Run(annotation(current), comparison(), good_comparison)
    failed = review_state_deltas(run, current, extraction())
    passed = review_state_deltas(run, current, extraction(correct))
    assert failed.report.verdict == "fail" and passed.report.verdict == "pass"
    assert failed.observations == passed.observations
    assert run.llm.requests == 3
    assert run.nodes[0] == run.nodes[2]


@pytest.mark.parametrize("field,value", [("quote_start", 0), ("quote_hash", "0" * 64), ("id", "o99")])
def test_observation_binding_is_revalidated_before_comparison(field, value):
    current = scene()
    run = Run(annotation(current))
    observed = observe_scene_state(run, current).model_dump(mode="json")
    if field == "id":
        observed["observations"][0]["id"] = value
    else:
        observed["observations"][0]["verified_evidence"][0][field] = value
    with pytest.raises(ValueError, match="stale or altered"):
        compare_state_deltas(run, current, extraction(), observed)
    assert run.llm.requests == 1


def test_cached_comparison_is_revalidated_and_cannot_hide_a_deleted_row():
    current, candidate = scene(), extraction()
    run = Run(annotation(current), comparison())
    result = review_state_deltas(run, current, candidate)
    key = next(key for key in run.cache if key[0] == "state-comparison-s1")
    run.cache[key]["comparisons"] = []
    with pytest.raises(ValueError, match="every delta"):
        compare_state_deltas(run, current, candidate, result.observations)
    assert run.llm.requests == 2


def test_cached_observation_is_revalidated_and_cannot_acquire_fabricated_source():
    current = scene()
    run = Run(annotation(current))
    observe_scene_state(run, current)
    key = next(iter(run.cache))
    run.cache[key]["observations"][0]["evidence"][0]["quote"] = "本文にない完了描写"
    with pytest.raises(ValueError, match="exact"):
        observe_scene_state(run, current)
    assert run.llm.requests == 1


def test_report_validation_finishes_before_a_comparison_is_cached(monkeypatch):
    current, candidate = scene(), extraction()
    original_report, report_calls = state_audit._report, []

    def validate_report(*args, **kwargs):
        report_calls.append(True)
        if len(report_calls) == 1:
            raise ValueError("Final audit report exceeds its contract bound.")
        return original_report(*args, **kwargs)

    monkeypatch.setattr(state_audit, "_report", validate_report)
    run = Run(annotation(current), comparison(), comparison())
    result = review_state_deltas(run, current, candidate)
    assert result.report.verdict == "fail"
    assert run.llm.requests == 3
    assert len(run.cache) == 2


def test_every_model_array_is_required_and_generated_record_ids_are_forbidden():
    current = scene()
    draft = annotation(current)
    draft.pop("missing_information")
    with pytest.raises(ValidationError):
        ObservationDraft.model_validate(draft)
    draft = annotation(current)
    draft["observations"][0]["id"] = "invented"
    with pytest.raises(ValidationError):
        ObservationDraft.model_validate(draft)
    candidate = comparison()
    candidate["comparisons"][0]["claims"][0].pop("matches")
    with pytest.raises(ValidationError):
        ComparisonDraft.model_validate(candidate)
