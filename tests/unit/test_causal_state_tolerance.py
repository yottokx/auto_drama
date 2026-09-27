"""Positive controls for r7 false positives, with physical safety boundaries."""

import pytest

from services.worker.generation.causal_state_observation import review_state_deltas
from tests.unit.test_causal_state_observation import (
    Run,
    annotation,
    claim,
    comparison,
    extraction,
    match,
    observation,
    scene,
)


def blank_case(*, stage="attempted", polarity="denied", match_changes=None, later_written=False):
    raw = "NARRATOR: 主人公は白紙の説明札を見つめたが、ペン先は動かなかった。"
    if later_written:
        raw += "\nNARRATOR: その後、主人公は説明札に日付を書いた。"
    current, after = scene(raw), "説明札は未記入のまま。"
    rows = [observation("s1-u1", "ペン先は動かなかった。", target="説明札", action="記入",
                        actor_ids=["Hero"], quantity=None, stage=stage, polarity=polarity)]
    matches = [match(**(match_changes or {}))]
    if later_written:
        rows.append(observation("s1-u2", "説明札に日付を書いた。", target="説明札", action="記入",
                                actor_ids=["Hero"], quantity=None, stage="completed"))
        matches.append(match("o2", "contradicts"))
    response = comparison(claims=[claim(after, polarity="denied", matches=matches)])
    return review_state_deltas(Run(annotation(current, rows), response), current, extraction(after))


def test_blank_result_is_not_mistaken_for_affirmative_completion():
    result = blank_case()
    assert result.report.verdict == "pass"
    warning = result.report.issues[0]
    assert warning.code == "state_interpretation" and warning.severity == "warning"
    assert warning.evidence[0].utterance_id == "s1-u1"


@pytest.mark.parametrize("stage", ["requested", "intended", "in_progress", "unknown"])
def test_absence_is_not_inferred_from_request_or_partial_progress(stage):
    assert blank_case(stage=stage).report.verdict == "fail"


@pytest.mark.parametrize("changes", [{"target_matches": False}, {"actor_matches": False},
                                     {"quantity_matches": False}, {"action_matches": False}])
def test_negation_keeps_object_actor_action_and_quantity_checks(changes):
    assert blank_case(match_changes=changes).report.verdict == "fail"


def test_subsequent_actual_writing_still_defeats_blank_claim():
    assert blank_case(later_written=True).report.verdict == "fail"


def mental_case(*, kind="physical", polarity="affirmed", assertion="observed", time_scope="current",
                stage="completed", match_changes=None):
    current = scene("Hero: 勝手に決めないで。\nNARRATOR: 主人公は腕を組んで作業台から離れた。")
    after = "主人公は相手の独断に反発している。"
    utterance, quote = ("s1-u1", "勝手に決めないで。") if kind != "physical" else (
        "s1-u2", "腕を組んで作業台から離れた。")
    rows = [observation(utterance, quote, actor_ids=["Hero"], target="相手または腕と作業台",
        action="抗議または距離を置く", quantity=None, kind=kind, stage=stage,
        assertion=assertion, time_scope=time_scope, polarity=polarity)]
    changes = {"target_matches": False, "action_matches": False, **(match_changes or {})}
    response = comparison(claims=[claim(after, kind="mental", stage="established", matches=[match(**changes)])])
    return review_state_deltas(Run(annotation(current, rows), response), current, extraction(after))


@pytest.mark.parametrize("kind,polarity", [("physical", "affirmed"), ("speech", "affirmed"),
                                           ("agreement", "denied")])
def test_expressed_reaction_can_support_mental_interpretation_with_warning(kind, polarity):
    result = mental_case(kind=kind, polarity=polarity)
    assert result.report.verdict == "pass"
    assert result.report.issues and all(issue.severity == "warning" for issue in result.report.issues)


@pytest.mark.parametrize("changes", [{"assertion": "reported"}, {"assertion": "believed"},
    {"assertion": "hypothetical"}, {"time_scope": "historical"}, {"stage": "requested"},
    {"stage": "unknown"}, {"polarity": "uncertain"}, {"match_changes": {"actor_matches": False}},
    {"match_changes": {"relation": "does_not_establish"}}])
def test_mental_tolerance_requires_current_same_actor_observed_support(changes):
    assert mental_case(**changes).report.verdict == "fail"


@pytest.mark.parametrize("entity,verdict", [("Hero", "pass"), ("keeper", "fail")])
def test_character_state_requires_that_characters_own_observed_reaction(entity, verdict):
    current = scene("NARRATOR: 主人公は腕を組んで作業台から離れた。")
    after = "相手の独断に反発している。"
    rows = [observation("s1-u1", "腕を組んで作業台から離れた。", actor_ids=["Hero"],
                        target="腕と作業台", quantity=None, stage="completed")]
    candidate = extraction(after)
    candidate["state_deltas"][0].update(scope="character", entity_id=entity)
    response = comparison(claims=[claim(after, kind="mental", stage="established", matches=[
        match(target_matches=False, action_matches=False)])])
    result = review_state_deltas(Run(annotation(current, rows), response), current, candidate)
    assert result.report.verdict == verdict
