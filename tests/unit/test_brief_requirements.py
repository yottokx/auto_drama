from __future__ import annotations

import copy
import json

import pytest

from services.worker.generation.brief_requirements import (
    applied_instructions,
    ensure_brief_requirements,
)
from services.worker.generation.schemas import object_schema, validate_schema

SCHEMA = object_schema({"setting": {"type": "string"}})
ISSUE = {
    "requirement": "魔法を登場させない",
    "source_quote": "魔法は禁止",
    "problem": "灯台守が魔法で海を照らしている",
}


class ReviewLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = []
        self.trace = []

    def structured(self, stage, messages, schema):
        self.calls.append({"stage": stage, "messages": copy.deepcopy(messages), "schema": schema})
        return copy.deepcopy(next(self.responses))


def review(llm, result=None, *, payload=None, normalize=None, schema=SCHEMA, kind="m2_world"):
    if payload is None:
        payload = {"world_input": {"prompt": "港町の物語。魔法は禁止"}}
    if result is None:
        result = {"setting": "灯台守が魔法で海を照らす港町。"}

    def validated(value):
        validate_schema(value, schema)
        return value

    return ensure_brief_requirements(
        kind, payload, result, llm, context=json.dumps(payload, ensure_ascii=False),
        system="設定を生成してください。", schema=schema, normalize=normalize or validated,
    )


def test_compatible_creative_expansion_is_accepted_without_repair():
    result = {"setting": "港町では灯台守が古いランプを守り、漁師が朝市に集う。"}
    llm = ReviewLLM([{"issues": []}])

    accepted = review(llm, result)

    assert accepted == result
    assert [call["stage"] for call in llm.calls] == ["m2_world-requirements-review-1"]
    assert llm.trace[0]["issues"] == []
    assert "原文との単語一致や逐語的再現は要求しません" in llm.calls[0]["messages"][1]["content"]


def test_repair_is_validated_then_reviewed_against_same_original_sources():
    invalid = {"setting": "灯台守が魔法で海を照らす港町。"}
    fixed = {"setting": "灯台守がランプで海を照らす港町。"}
    payload = {"world_input": {"prompt": "港町の物語。魔法は禁止"}}
    before = copy.deepcopy(payload)
    normalized = []
    llm = ReviewLLM([{"issues": [ISSUE]}, fixed, {"issues": []}])

    def normalize(value):
        validate_schema(value, SCHEMA)
        normalized.append(copy.deepcopy(value))
        return value

    assert review(llm, invalid, payload=payload, normalize=normalize) == fixed
    assert normalized == [fixed]
    assert invalid["setting"].startswith("灯台守が魔法")
    assert payload == before
    assert [call["stage"] for call in llm.calls] == [
        "m2_world-requirements-review-1", "m2_world-requirements-repair-1",
        "m2_world-requirements-review-2",
    ]
    assert json.dumps(fixed, ensure_ascii=False) in llm.calls[-1]["messages"][1]["content"]
    assert llm.trace[-1]["issues"] == []


def test_persistent_violation_fails_after_two_repairs_and_records_each_review():
    unchanged = {"setting": "灯台守が魔法で海を照らす港町。"}
    llm = ReviewLLM([
        {"issues": [ISSUE]}, unchanged, {"issues": [ISSUE]}, unchanged, {"issues": [ISSUE]},
    ])

    with pytest.raises(ValueError, match="STEP1.*魔法を登場させない"):
        review(llm)

    assert len(llm.calls) == 5
    assert [record["attempt"] for record in llm.trace] == [1, 2, 3]
    assert all(record["issues"] == [ISSUE] for record in llm.trace)


def test_patch_repairs_are_normalized_before_review_and_protected_fields_stay_intact():
    payload = {
        "character_id": "one", "character_input": {"id": "one", "settings": "優しい人物"},
        "scope": "settings", "locked": {"appearance": True},
        "instruction": "優しさが伝わる設定に変更",
    }
    initial = {"id": "one", "settings": "他人を傷つけて楽しむ。", "appearance": "青い服"}
    patch_schema = object_schema({
        "id": {"type": "string"},
        "changes": {**object_schema({
            "settings": {"type": "string"}, "appearance": {"type": "string"},
        }), "required": []},
    })
    llm = ReviewLLM([
        {"issues": [{"requirement": "優しい人物", "source_quote": "優しい人物",
                     "problem": "他人を傷つけることを楽しむ設定がある"}]},
        {"id": "one", "changes": {"settings": "困った旅人を助ける。", "appearance": "赤い服"}},
        {"issues": []},
    ])

    def normalize(patch):
        return {**initial, **patch["changes"], "appearance": initial["appearance"]}

    result = review(llm, initial, payload=payload, normalize=normalize,
                    schema=patch_schema, kind="m2_character")

    assert result == {**initial, "settings": "困った旅人を助ける。"}
    final_review = llm.calls[-1]["messages"][1]["content"]
    assert json.dumps(result, ensure_ascii=False) in final_review
    assert "赤い服" not in final_review


@pytest.mark.parametrize("malformed", [
    {}, {"issues": "none"}, {"issues": [{"requirement": "禁止"}]},
    {"issues": [{**ISSUE, "problem": "   "}]},
    {"issues": [{**ISSUE, "source_quote": "生成結果にしかない指示"}]},
    {"issues": [{**ISSUE, "unexpected": "extra"}]},
])
def test_malformed_or_ungrounded_reviews_do_not_silently_pass(malformed):
    llm = ReviewLLM([malformed])

    with pytest.raises(ValueError, match="STEP1.*整合性確認"):
        review(llm)

    assert len(llm.calls) == 1


def test_repair_schema_errors_stop_before_re_review():
    llm = ReviewLLM([{"issues": [ISSUE]}, {"setting": 123}])

    with pytest.raises(ValueError, match="must be string"):
        review(llm)

    assert len(llm.calls) == 2


def test_generated_context_cannot_supply_requirement_evidence():
    payload = {
        "world_input": {"prompt": "自由な港町の物語"},
        "world_result": {"setting": "魔法は禁止"},
    }
    llm = ReviewLLM([{"issues": [ISSUE]}])

    with pytest.raises(ValueError, match="根拠が元の指示にありません"):
        review(llm, payload=payload)


def test_character_review_keeps_other_people_full_instructions_out_of_its_source_block():
    payload = {
        "character_id": "one", "character_input": {"id": "one", "freeform": "優しい人物"},
        "cast_inputs": [
            {"id": "one", "freeform": "優しい人物"},
            {"id": "two", "freeform": "船長", "settings": "PRIVATE_SETTINGS",
             "appearance": "PRIVATE_APPEARANCE", "voice": "PRIVATE_VOICE"},
        ],
    }
    llm = ReviewLLM([{"issues": []}])

    ensure_brief_requirements(
        "m2_character", payload, {"setting": "困った旅人を助ける。"}, llm,
        context='{"character_id": "one"}', system="設定を生成", schema=SCHEMA,
        normalize=lambda value: value,
    )

    content = llm.calls[0]["messages"][1]["content"]
    assert "PRIVATE_" not in content
    assert "船長" in content


@pytest.mark.parametrize("entry", [
    {"kind": "m2_world", "scope": "world", "instruction": "魔法は禁止"},
    {"kind": "m2_world", "scope": "world", "changes": {"notes": "魔法は禁止"}},
])
def test_previously_applied_instructions_and_direct_edits_can_ground_requirements(entry):
    payload = {
        "world_input": {"prompt": "港町の物語"},
        "applied_instructions": [entry],
        "instruction": "市場もある港町にしてください",
    }
    fixed = {"setting": "灯台守がランプを守り、市場で漁師が魚を売る。"}
    llm = ReviewLLM([{"issues": [ISSUE]}, fixed, {"issues": []}])

    assert review(llm, payload=payload) == fixed
    source_json = llm.calls[0]["messages"][1]["content"].split("\noriginal_user_inputs: ")[1]
    sources = json.JSONDecoder().raw_decode(source_json)[0]
    assert sources["applied_instructions"] == [entry]
    assert sources["instruction"] == payload["instruction"]


@pytest.mark.parametrize("metadata", ["m2_world", "world", "character-1"])
def test_history_metadata_is_not_requirement_evidence(metadata):
    payload = {"applied_instructions": [{
        "kind": "m2_world", "scope": "world", "character_id": "character-1",
        "instruction": "港町を舞台にする",
    }]}
    llm = ReviewLLM([{"issues": [{**ISSUE, "source_quote": metadata}]}])

    with pytest.raises(ValueError, match="根拠が元の指示にありません"):
        review(llm, payload=payload)


def test_applied_history_keeps_chronology_target_identity_and_other_character_isolation():
    history = [
        {"kind": "m2_world", "scope": "world", "instruction": "砂漠の町に変更"},
        {"kind": "m2_character", "character_id": "one", "scope": "settings",
         "changes": {"name": "ミナ", "settings": "主人公の詳しい経歴"}},
        {"kind": "m2_character", "character_id": "two", "scope": "all",
         "instruction": "名前をリオンに変更",
         "changes": {"name": "リオン", "role": "船長", "settings": "PRIVATE_SETTINGS",
                     "appearance": "PRIVATE_APPEARANCE", "voice": "PRIVATE_VOICE"}},
        {"kind": "m2_world", "scope": "world", "changes": {"setting": "海辺の町に変更"}},
    ]
    payload = {"character_id": "one", "applied_instructions": history}
    before = copy.deepcopy(payload)

    copied = applied_instructions("m2_character", payload)

    assert copied[:2] == history[:2]
    assert copied[2] == {**history[2], "changes": {"name": "リオン", "role": "船長"}}
    assert copied[3] == history[3]
    assert "PRIVATE_" not in json.dumps(copied)
    copied[1]["changes"]["name"] = "別名"
    assert payload == before
    assert applied_instructions("m2_world", payload) == history


def test_repair_receives_original_output_rules_without_adding_them_to_review_requirements():
    fixed = {"setting": "灯台守がランプで海を照らす港町。"}
    llm = ReviewLLM([{"issues": [ISSUE]}, fixed, {"issues": []}])
    rules = "VOICE_OUTPUT_RULES: 台詞に地の文や役名を含めない。"

    accepted = ensure_brief_requirements(
        "m2_world", {"world_input": {"prompt": "魔法は禁止"}},
        {"setting": "魔法で海を照らす。"}, llm,
        context="{}", system="設定を生成", schema=SCHEMA,
        normalize=lambda value: value, repair_instructions=rules,
    )

    assert accepted == fixed
    assert rules not in llm.calls[0]["messages"][1]["content"]
    assert rules in llm.calls[1]["messages"][1]["content"]
    assert rules not in llm.calls[2]["messages"][1]["content"]
