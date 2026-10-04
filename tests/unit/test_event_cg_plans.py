"""CG selection preserves the adopted text, identities and finite allowances."""
import copy

import pytest
from pydantic import ValidationError

from packages.contracts.event_cg import (
    BUDGET_ALLOCATION_VERSION,
    EventCgPolicy,
    EventCgResult,
    image_input_sha256,
    validate_budget,
    validate_plan,
)


def request():
    return {"policy": EventCgPolicy(max_cgs=2, max_variants_per_cg=1).model_dump(),
            "chapter_budget": 1, "references": [{"character_id": "alice"}],
            "context": {"source_sha256": "a" * 64,
                        "overall_plot": {"chapters": [{"number": 1}, {"number": 2}]},
                        "narrative": {"scenes": [
                            {"id": "scene-a", "plan": {"character_ids": ["alice"]},
                             "utterances": [{"id": "u1"}, {"id": "u2"}, {"id": "u3"}]},
                            {"id": "scene-b", "plan": {"character_ids": ["alice"]},
                             "utterances": [{"id": "u4"}]}]}}}


def plan():
    return {"source_sha256": "a" * 64, "cgs": [{"id": "cg-one", "scene_id": "scene-a",
            "start_utterance_id": "u1", "end_utterance_id": "u4", "character_ids": ["alice"],
            "interpretation": "二人の約束", "prompt": "Image 1 holds an old book in a quiet library.",
            "variants": [{"id": "smile", "start_utterance_id": "u3", "interpretation": "微笑む",
                          "prompt": "Keep the base composition and make the character smile."}]}]}


def test_scene_end_uses_next_scene_exclusive_anchor_without_changing_source():
    payload = request()
    before = copy.deepcopy(payload)
    result = validate_plan(plan(), payload)
    assert result.cgs[0].end_utterance_id == "u4"
    assert payload == before


@pytest.mark.parametrize("patch", [
    {"start_utterance_id": "unknown"}, {"end_utterance_id": None},
    {"end_utterance_id": "u1"}, {"scene_id": "scene-b"},
    {"character_ids": ["bob"]}, {"character_ids": ["alice", "alice"]},
    {"variants": [{"id": "bad", "start_utterance_id": "u4", "interpretation": "外",
                   "prompt": "smile"}]},
    {"variants": [{"id": "bad", "start_utterance_id": "u1", "interpretation": "同時",
                   "prompt": "smile"}]},
    {"variants": [{"id": "cg-one", "start_utterance_id": "u2", "interpretation": "同一ID",
                   "prompt": "smile"}]},
])
def test_rejects_unbound_cast_ids_and_invalid_intervals(patch):
    value = plan()
    value["cgs"][0].update(patch)
    with pytest.raises(ValueError):
        validate_plan(value, request())


def test_plan_rejects_new_version_over_budget_or_extra_variations():
    value, payload = plan(), request()
    payload["chapter_budget"] = 0
    with pytest.raises(ValueError):
        validate_plan(value, payload)
    payload = request()
    payload["policy"]["max_variants_per_cg"] = 0
    with pytest.raises(ValueError):
        validate_plan(value, payload)
    value["source_sha256"] = "b" * 64
    with pytest.raises(ValueError):
        validate_plan(value, request())


def test_overlap_is_rejected_and_chapter_end_is_permitted_only_in_last_scene():
    value, payload = plan(), request()
    payload["chapter_budget"] = 2
    second = {**value["cgs"][0], "id": "cg-two", "variants": []}
    value["cgs"].append(second)
    with pytest.raises(ValueError, match="overlap"):
        validate_plan(value, payload)
    second.update(scene_id="scene-b", start_utterance_id="u4", end_utterance_id=None)
    assert len(validate_plan(value, payload).cgs) == 2


def test_budget_cannot_drop_duplicate_or_add_chapters_or_exceed_total():
    value = {"source_sha256": "a" * 64,
             "chapters": [{"chapter_number": 1, "limit": 1, "reason": "出会い"},
                          {"chapter_number": 2, "limit": 1, "reason": "結末"}]}
    assert len(validate_budget(value, request()).chapters) == 2
    for chapters in [value["chapters"][:1], value["chapters"][::-1], value["chapters"] * 2,
                     [{**row, "limit": 2} for row in value["chapters"]]]:
        with pytest.raises(ValueError):
            validate_budget({**value, "chapters": chapters}, request())


@pytest.mark.parametrize("limits", [[0, 0], [1, 0]])
def test_current_budget_requires_all_slots_but_legacy_results_remain_compatible(limits):
    value = {"source_sha256": "a" * 64, "chapters": [
        {"chapter_number": number, "limit": limit, "reason": "配分"}
        for number, limit in enumerate(limits, 1)]}
    assert [row.limit for row in validate_budget(value, request()).chapters] == limits
    payload = {**request(), "budget_allocation_version": BUDGET_ALLOCATION_VERSION}
    with pytest.raises(ValueError, match=f"total {sum(limits)}; allocate all 2"):
        validate_budget(value, payload)
    value["chapters"][1]["limit"] = 2 - limits[0]
    assert sum(row.limit for row in validate_budget(value, payload).chapters) == 2


def test_current_budget_can_explicitly_omit_allocation_or_disable_generation():
    payload = {**request(), "budget_allocation_version": BUDGET_ALLOCATION_VERSION}
    value = {"source_sha256": "a" * 64, "chapters": [
        {"chapter_number": number, "limit": 0, "reason": "配分失敗"} for number in (1, 2)],
        "omission_reason": "CG配分を確定できませんでした。"}
    assert validate_budget(value, payload).omission_reason
    value["chapters"][1]["limit"] = 1
    with pytest.raises(ValueError, match="Omitted CG budget"):
        validate_budget(value, payload)
    value["chapters"][1]["limit"] = 0
    value["omission_reason"] = ""
    payload["policy"]["max_cgs"] = 0
    assert all(row.limit == 0 for row in validate_budget(value, payload).chapters)


def test_empty_plan_is_valid_but_omission_must_not_contain_generated_images():
    assert validate_plan({"source_sha256": "a" * 64}, request()).cgs == []
    with pytest.raises(ValueError):
        validate_plan({**plan(), "omission_reason": "失敗"}, request())
    with pytest.raises(ValidationError):
        EventCgResult(input_sha256="a" * 64, cg_id="cg-one", status="omitted")


def test_fingerprint_ignores_local_paths_but_pins_order_profile_and_seed():
    payload = {"references": [{"artifact_id": "a"}, {"artifact_id": "b"}], "seed": 0,
               "cg_profile": {"steps": 40}}
    digest = image_input_sha256(payload)
    assert image_input_sha256({**payload, "input_sha256": digest, "reference_paths": ["local"]}) == digest
    assert image_input_sha256({**payload, "references": list(reversed(payload["references"]))}) != digest
    assert image_input_sha256({**payload, "seed": 1}) != digest
    assert image_input_sha256({**payload, "cg_profile": {"steps": 20}}) != digest


def test_prompt_omissions_reserve_the_selected_budget_and_have_unique_ids():
    value, payload = plan(), request()
    payload["chapter_budget"] = 2
    value["prompt_omissions"] = [{"cg_id": "cg-two", "reason": "指示作成に失敗"}]
    assert len(validate_plan(value, payload).prompt_omissions) == 1
    payload["chapter_budget"] = 1
    with pytest.raises(ValueError, match="allowance"):
        validate_plan(value, payload)
    payload["chapter_budget"] = 2
    value["prompt_omissions"][0]["cg_id"] = "cg-one"
    with pytest.raises(ValueError, match="unique"):
        validate_plan(value, payload)
    value["cgs"] = []
    value["prompt_omissions"] *= 2
    with pytest.raises(ValueError, match="unique"):
        validate_plan(value, payload)


def test_all_prompt_failures_are_valid_as_explicit_omissions():
    value = {"source_sha256": "a" * 64, "cgs": [],
             "omission_reason": "有効な画像指示が得られませんでした。",
             "prompt_omissions": [{"cg_id": "cg-one", "reason": "参照人物の不一致"}]}
    result = validate_plan(value, request())
    assert not result.cgs
    assert result.prompt_omissions[0].cg_id == "cg-one"


@pytest.mark.parametrize("patch", [{"max_cgs": True}, {"max_cgs": -1},
                                  {"max_cgs": 101}, {"max_variants_per_cg": 11}])
def test_invalid_generation_limits_are_not_coerced(patch):
    with pytest.raises(ValidationError):
        EventCgPolicy(**patch)
