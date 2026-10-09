"""New CG staging rules preserve long holds without changing legacy contracts."""
import copy

import pytest

from packages.contracts.event_cg import EventCgPolicy, image_spans, validate_plan


def staging_request():
    return {"policy": EventCgPolicy(planning_version=2, max_cgs=2, max_variants_per_cg=3).model_dump(),
            "chapter_budget": 2, "references": [{"character_id": "alice"}],
            "context": {"source_sha256": "a" * 64, "narrative": {"scenes": [
                {"id": "s1", "plan": {"character_ids": ["alice"]},
                 "utterances": [{"id": f"u{i}", "display_text": f"発話{i}"} for i in range(1, 16)]},
                {"id": "s2", "plan": {"character_ids": ["alice"]},
                 "utterances": [{"id": f"u{i}", "display_text": "……"} for i in range(16, 21)]}]}}}


def stage(evidence, end, *, change=""):
    return {"visual_state": "相手を見つめている。", "evidence_utterance_ids": [evidence],
            "safe_end_utterance_id": end, "reason": "約束のやり取りを見せる。", "change": change}


def staging_plan():
    return {"planning_version": 2, "source_sha256": "a" * 64, "cgs": [{
        "id": "cg_one", "scene_id": "s1", "start_utterance_id": "u1", "end_utterance_id": "u11",
        "character_ids": ["alice"], "interpretation": "緊張から受容へ", "prompt": "Two people talking.",
        "composition": "向かい合う二人", "end_reason": "次の話題に移る。", "end_evidence_utterance_id": "u11",
        "staging": stage("u1", "u6"), "variants": [{"id": "cg_one_v1", "start_utterance_id": "u6",
            "interpretation": "微笑みながら受け入れる。", "prompt": "A sustained warm smile.",
            "staging": stage("u6", "u11", change="緊張が解け、相手を受け入れる微笑みになる。")}]}]}


def test_each_image_gets_its_own_exact_metrics_and_source_is_immutable():
    source, value = staging_request(), staging_plan()
    original = copy.deepcopy((source, value))
    plan = validate_plan(value, source)
    spans = image_spans(plan.cgs[0], source["context"]["narrative"])
    assert [row["utterance_count"] for row in spans] == [5, 5]
    assert [row["character_count"] for row in spans] == [15, 16]
    assert [row["variant_id"] for row in spans] == [None, "cg_one_v1"]
    assert [row["end_utterance_id"] for row in spans] == ["u6", "u11"]
    assert (source, value) == original


@pytest.mark.parametrize("start,end,expected", [("u3", "u11", "2 utterances"),
                                               ("u6", "u8", "2 utterances"),
                                               ("u2", "u4", "1 utterances")])
def test_short_base_or_final_variant_is_rejected_with_counts(start, end, expected):
    value = staging_plan()
    cg = value["cgs"][0]
    cg["variants"][0]["start_utterance_id"] = start
    cg["end_utterance_id"] = cg["end_evidence_utterance_id"] = end
    with pytest.raises(ValueError, match=expected):
        validate_plan(value, staging_request())


def test_a_single_five_utterance_cg_is_allowed_without_filling_the_variant_allowance():
    value = staging_plan()
    cg = value["cgs"][0]
    cg.update(variants=[], end_utterance_id="u6", end_evidence_utterance_id="u6")
    assert not validate_plan(value, staging_request()).cgs[0].variants
    cg.update(end_utterance_id="u5", end_evidence_utterance_id="u5")
    with pytest.raises(ValueError, match="4 utterances"):
        validate_plan(value, staging_request())


@pytest.mark.parametrize("evidence", [["u7"], ["u16"], ["unknown"], ["u6", "u6"]])
def test_later_or_other_scene_evidence_cannot_justify_a_variant(evidence):
    value = staging_plan()
    value["cgs"][0]["variants"][0]["staging"]["evidence_utterance_ids"] = evidence
    with pytest.raises(ValueError, match="evidence"):
        validate_plan(value, staging_request())


@pytest.mark.parametrize("safe_end", ["u5", "u12", "unknown", None])
def test_safe_end_must_cover_the_image_and_cannot_extend_beyond_the_cg(safe_end):
    value = staging_plan()
    value["cgs"][0]["staging"]["safe_end_utterance_id"] = safe_end
    with pytest.raises(ValueError, match="safe end"):
        validate_plan(value, staging_request())


def test_base_can_remain_safe_past_an_optional_change():
    value = staging_plan()
    value["cgs"][0]["staging"]["safe_end_utterance_id"] = "u11"
    assert validate_plan(value, staging_request()).cgs[0].staging.safe_end_utterance_id == "u11"


def test_ending_before_a_later_visual_change_preserves_its_real_evidence():
    value = staging_plan()
    value["cgs"][0]["end_evidence_utterance_id"] = "u14"
    assert validate_plan(value, staging_request()).cgs[0].end_evidence_utterance_id == "u14"
    value["cgs"][0]["end_evidence_utterance_id"] = "u16"
    with pytest.raises(ValueError, match="ending evidence"):
        validate_plan(value, staging_request())


def test_scene_end_and_chapter_end_keep_exclusive_boundary_semantics():
    source, value = staging_request(), staging_plan()
    cg = value["cgs"][0]
    cg.update(variants=[], end_utterance_id="u16", end_evidence_utterance_id="u15")
    cg["staging"]["safe_end_utterance_id"] = "u16"
    assert image_spans(validate_plan(value, source).cgs[0], source["context"]["narrative"])[0]["utterance_count"] == 15
    cg.update(scene_id="s2", start_utterance_id="u16", end_utterance_id=None,
              end_evidence_utterance_id="u20", staging=stage("u16", None))
    span = image_spans(validate_plan(value, source).cgs[0], source["context"]["narrative"])[0]
    assert span["utterance_count"] == 5 and span["end_utterance_id"] is None


@pytest.mark.parametrize("field,patch", [("composition", " "), ("end_reason", " "),
                                         ("end_evidence_utterance_id", "u4"), ("staging", None)])
def test_new_plans_need_meaningful_staging_metadata(field, patch):
    value = staging_plan()
    value["cgs"][0][field] = patch
    with pytest.raises(ValueError):
        validate_plan(value, staging_request())


def test_empty_change_is_not_a_valid_variant():
    value = staging_plan()
    value["cgs"][0]["variants"][0]["staging"]["change"] = " "
    with pytest.raises(ValueError, match="sustained visual change"):
        validate_plan(value, staging_request())


def test_frozen_version_cannot_be_downgraded_even_for_an_empty_plan():
    source = staging_request()
    value = {"source_sha256": "a" * 64, "cgs": []}
    with pytest.raises(ValueError, match="version"):
        validate_plan(value, source)
    assert not validate_plan({**value, "planning_version": 2}, source).cgs


def test_legacy_short_intervals_and_missing_staging_remain_valid():
    source, value = staging_request(), staging_plan()
    source["policy"].pop("planning_version")
    value.pop("planning_version")
    cg = value["cgs"][0]
    for name in ("staging", "composition", "end_reason", "end_evidence_utterance_id"):
        cg.pop(name)
    cg["variants"][0].pop("staging")
    cg["variants"][0]["start_utterance_id"] = "u2"
    cg["end_utterance_id"] = "u3"
    result = validate_plan(value, source)
    assert result.planning_version == 1
    assert [span["utterance_count"] for span in image_spans(result.cgs[0], source["context"]["narrative"])] == [1, 1]
