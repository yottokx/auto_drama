from __future__ import annotations

import copy
import json

import pytest

from services.worker.generation import pipeline
from services.worker.generation.schemas import preserve_character, validate_schema


def _character(character_id="character-2", name="ノア"):
    return {
        "id": character_id,
        "name": name,
        "age": "24歳",
        "gender": "女性",
        "role": "星図の修復師",
        "freeform": "古い道具を大切にする人物",
        "settings": "修復工房で育ち、失われた星図を探している。慎重で観察眼が鋭い。",
        "appearance": "銀髪を短く結び、緑の作業着を着ている。",
        "voice": "落ち着いた中音域で、言葉を選びながらゆっくり話す。",
        "selfIntroduction": f"私は{name}。この町で星図を修復しています。古い道具の声を聞くのが好きです。",
        "sampleLines": ["小さな傷にも理由がある。", "この印はまだ読める。", "急がず、確かめましょう。"],
        "height_cm": 164,
        "body_type": "humanoid",
        "locked": {"settings": False, "appearance": False, "voice": False},
    }


PRIVATE_FIELDS = ("settings", "appearance", "voice", "selfIntroduction", "sampleLines")


def _payload(*, revision=True):
    target = _character()
    other = _character("character-1", "リオン")
    for field in PRIVATE_FIELDS:
        marker = f"PRIVATE_PROTAGONIST_{field}"
        other[field] = [marker, marker + "_2", marker + "_3"] if field == "sampleLines" else marker
    target_input = {**target, "name": "未定", "freeform": "星図を修復する職人"}
    return {
        "schema_version": 1,
        "seed": 71,
        "character_contract_version": 2,
        "world_input": {"prompt": "星図を修復する町の物語"},
        "world_result": {"title": "星図の町", "setting": "誰もが星を道標に暮らす町。"},
        "character_id": target["id"],
        "character_input": target_input,
        "character_result": target if revision else None,
        "cast_inputs": [copy.deepcopy(other), copy.deepcopy(target_input)],
        "cast_results": [copy.deepcopy(other), copy.deepcopy(target)] if revision else [copy.deepcopy(other)],
        "relationship_inputs": [{"characterIds": [other["id"], target["id"]], "instruction": "仕事仲間"}],
        "relationships_result": None,
        "scope": "all",
        "instruction": "名前だけをミナに変更し、自己紹介の名前も揃えてください。" if revision else "",
        "locked": copy.deepcopy(target["locked"]),
    }


class CharacterLLM:
    """Replay structured replies while checking the same schema as LocalLLM."""

    def __init__(self, *, patch=None, response_id="character-2", final=None):
        self.patch = {"name": "ミナ"} if patch is None else patch
        self.response_id = response_id
        self.final = _character() if final is None else final
        self.trace = []
        self.calls = []

    def structured(self, stage, messages, schema):
        self.calls.append({"stage": stage, "messages": copy.deepcopy(messages), "schema": copy.deepcopy(schema)})
        if stage == "character-revision":
            result = {"id": self.response_id, "changes": copy.deepcopy(self.patch)}
        elif stage.endswith("-candidates"):
            result = {"candidates": [
                {"id": letter, "concept": "星図を修復する職人", "tension": "失われた星図の謎"}
                for letter in "ABC"
            ]}
        elif stage.endswith("-selected"):
            result = {"selected_id": "A", "reason": "本人の指示と確定世界に合う。"}
        elif stage == "final":
            result = copy.deepcopy(self.final)
        elif "-requirements-review-" in stage:
            result = {"issues": []}
        else:
            raise AssertionError(f"Unexpected structured stage: {stage}")
        validate_schema(result, schema)
        return result

    def random_context(self, stage, messages):
        self.calls.append({"stage": stage, "messages": copy.deepcopy(messages)})
        return [*messages, {"role": "assistant", "content": "本人の指示に合う候補Aを採用する。"}]


def test_name_revision_merges_only_changed_fields_into_current_second_character():
    payload = _payload()
    original = copy.deepcopy(payload)
    introduction = payload["character_result"]["selfIntroduction"].replace("ノア", "ミナ")
    patch = {"name": "ミナ", "selfIntroduction": introduction}
    llm = CharacterLLM(patch=patch)

    result = pipeline.generate_text("m2_character", payload, llm)

    assert result == {**original["character_result"], **patch}
    assert payload == original
    assert [call["stage"] for call in llm.calls] == [
        "character-revision", "m2_character-requirements-review-1",
    ]
    assert llm.calls[0]["schema"]["properties"]["id"]["enum"] == ["character-2"]


@pytest.mark.parametrize("revision", [False, True], ids=["initial-generation", "revision"])
def test_every_character_stage_excludes_other_characters_full_settings(revision):
    payload = _payload(revision=revision)
    payload["cast_inputs"][0].update(name="指定したリオン", freeform="右腕のない記録係")
    payload["applied_instructions"] = [
        {"kind": "m2_character", "character_id": "character-1", "scope": "settings",
         "changes": {"name": "リオン", "settings": "PRIVATE_PROTAGONIST_accepted_settings"}},
        {"kind": "m2_character", "character_id": "character-2", "scope": "appearance",
         "changes": {"appearance": "銀髪を短く結び、緑の作業着を着ている。"}},
    ]
    history = [
        {**payload["applied_instructions"][0], "changes": {"name": "リオン"}},
        payload["applied_instructions"][1],
    ]
    llm = CharacterLLM()

    pipeline.generate_text("m2_character", payload, llm)

    expected = ["character-revision"] if revision else ["final"]
    expected.append("m2_character-requirements-review-1")
    assert [call["stage"] for call in llm.calls] == expected
    for call in llm.calls:
        text = json.dumps(call["messages"], ensure_ascii=False)
        assert "PRIVATE_PROTAGONIST_" not in text, call["stage"]
        context = json.JSONDecoder().raw_decode(call["messages"][1]["content"])[0]
        assert [item["id"] for item in context["cast_inputs"]] == ["character-2"]
        assert context["character_id"] == "character-2"
        assert context["character_input"] == payload["character_input"]
        assert context["character_result"] == payload["character_result"]
        assert context["applied_instructions"] == history
        assert "cast_results" not in context
        references = context["other_characters"]
        assert [item["id"] for item in references] == ["character-1"]
        assert references[0]["name"] == "リオン"
        assert not set(PRIVATE_FIELDS) & references[0].keys()
        assert set(references[0]) <= {"id", "name", "age", "gender", "role", "freeform"}
        assert context["other_character_inputs"] == [{
            key: payload["cast_inputs"][0][key]
            for key in ("id", "name", "age", "gender", "role", "freeform")
        }]
        assert context["other_character_inputs"][0]["freeform"] == "右腕のない記録係"
        assert references[0]["freeform"] != context["other_character_inputs"][0]["freeform"]
        if "-requirements-review-" in call["stage"]:
            sources, _ = json.JSONDecoder().raw_decode(
                call["messages"][1]["content"].split("\noriginal_user_inputs: ", 1)[1]
            )
            assert sources["applied_instructions"] == history


def test_revision_review_repairs_contradiction_without_discarding_requested_edit():
    payload = _payload()
    payload["applied_instructions"] = [{
        "kind": "m2_character", "character_id": "character-2", "scope": "settings",
        "instruction": "名前をノアに変更し、自己紹介もその名前に揃える。",
    }]
    requirement = "生まれつき目が見えず、視覚能力もない星図の修復師"
    payload["character_input"]["freeform"] = requirement
    payload["cast_inputs"][1]["freeform"] = requirement
    payload["character_result"]["settings"] = "生まれつき目が見えず、紙の凹凸と触覚で星図を修復する。"
    original = copy.deepcopy(payload)
    bad_settings = "星図に触れたことで視力を取り戻し、目で細かな傷を発見する。"
    introduction = payload["character_result"]["selfIntroduction"].replace("ノア", "ミナ")
    patch = {"name": "ミナ", "selfIntroduction": introduction, "settings": bad_settings}
    reviews = []

    class RepairingRevisionLLM(CharacterLLM):
        def structured(self, stage, messages, schema):
            assert not any(item["type"] == "complete_result" for item in self.trace)
            context, _ = json.JSONDecoder().raw_decode(messages[-1]["content"])
            assert context["applied_instructions"] == original["applied_instructions"]
            assert context["instruction"] == original["instruction"]
            if "-requirements-repair-" in stage:
                assert pipeline.VOICE_DESIGN_RULES in messages[-1]["content"]
                assert "changesには明示された要望を満たすために変更が必要な項目だけ" in (
                    messages[-1]["content"]
                )
                self.calls.append({"stage": stage, "messages": copy.deepcopy(messages),
                                   "schema": copy.deepcopy(schema)})
                result = {"id": payload["character_id"], "changes": {
                    "settings": original["character_result"]["settings"],
                }}
            else:
                result = super().structured(stage, messages, schema)
                if "-requirements-review-" in stage:
                    content = messages[-1]["content"]
                    context, _ = json.JSONDecoder().raw_decode(content)
                    assert context["character_input"]["freeform"] == requirement
                    assert context["instruction"] == payload["instruction"]
                    sources, _ = json.JSONDecoder().raw_decode(
                        content.split("\noriginal_user_inputs: ", 1)[1]
                    )
                    assert sources["character_input"] == original["character_input"]
                    assert sources["applied_instructions"] == original["applied_instructions"]
                    assert sources["instruction"] == original["instruction"]
                    reviewed_result = json.loads(content.split("\nresult: ", 1)[1])
                    assert reviewed_result["selfIntroduction"] == introduction
                    assert reviewed_result["name"] == "ミナ"
                    expected_settings = bad_settings if not reviews else (
                        original["character_result"]["settings"]
                    )
                    assert reviewed_result["settings"] == expected_settings
                    reviews.append(stage)
                    result = {"issues": [{"requirement": requirement,
                                          "source_quote": requirement,
                                          "problem": "視力を取り戻す設定は指定と矛盾している。"}]
                              if len(reviews) == 1 else []}
            validate_schema(result, schema)
            return result

    llm = RepairingRevisionLLM(patch=patch)
    result = pipeline.generate_text("m2_character", payload, llm)

    assert result == {**original["character_result"], "name": "ミナ",
                      "selfIntroduction": introduction}
    assert payload == original
    assert [call["stage"] for call in llm.calls] == [
        "character-revision", "m2_character-requirements-review-1",
        "m2_character-requirements-repair-1", "m2_character-requirements-review-2",
    ]
    assert llm.trace[-1]["type"] == "complete_result"
    assert set(llm.trace[-1]["changed_fields"]) == {"name", "selfIntroduction"}


@pytest.mark.parametrize("revision", [False, True], ids=["initial-generation", "revision"])
def test_model_cannot_return_the_protagonist_id_for_a_second_character(revision):
    payload = _payload(revision=revision)
    llm = CharacterLLM(response_id="character-1", final=_character("character-1", "リオン"))

    with pytest.raises(ValueError):
        pipeline.generate_text("m2_character", payload, llm)

    assert llm.calls[-1]["schema"]["properties"]["id"]["enum"] == ["character-2"]


def test_preserve_character_rejects_wrong_identity_instead_of_relabelling_it():
    payload = _payload()
    protagonist = _character("character-1", "リオン")
    original = copy.deepcopy(protagonist)

    with pytest.raises(ValueError):
        preserve_character(protagonist, payload)

    assert protagonist == original


@pytest.mark.parametrize("scope,locked,patch", [
    ("settings", {}, {"appearance": "主人公と同じ外見"}),
    ("settings", {}, {"height_cm": 180}),
    ("settings", {}, {"body_type": "nonhumanoid"}),
    ("appearance", {}, {"name": "リオン"}),
    ("voice", {}, {"selfIntroduction": "私はリオン。"}),
    ("all", {"settings": True}, {"name": "リオン"}),
    ("all", {"settings": True}, {"sampleLines": ["一", "二", "三"]}),
    ("all", {"appearance": True}, {"appearance": "新しい服"}),
    ("all", {"appearance": True}, {"height_cm": 180}),
    ("all", {"voice": True}, {"voice": "新しい声"}),
    ("settings", {"voice": True}, {"selfIntroduction": "私はミナ。"}),
    ("all", {}, {"id": "character-1"}),
    ("all", {}, {"locked": {"settings": False, "appearance": False, "voice": False}}),
])
def test_revision_schema_rejects_fields_outside_scope_or_locks(scope, locked, patch):
    payload = _payload()
    payload["scope"] = scope
    payload["locked"].update(locked)
    original = copy.deepcopy(payload)
    llm = CharacterLLM(patch=patch)

    with pytest.raises(ValueError):
        pipeline.generate_text("m2_character", payload, llm)

    assert [call["stage"] for call in llm.calls] == ["character-revision"]
    allowed = llm.calls[0]["schema"]["properties"]["changes"]["properties"]
    assert not set(patch) & allowed.keys()
    assert payload == original


@pytest.mark.parametrize("scope,patch", [
    ("appearance", {"height_cm": 168}),
    ("voice", {"voice": "少し明るい中音域の声で、ゆっくりと話す。"}),
    ("settings", {"name": "ミナ"}),
])
def test_allowed_partial_revision_preserves_every_omitted_field(scope, patch):
    payload = _payload()
    payload["scope"] = scope
    original = copy.deepcopy(payload["character_result"])

    result = pipeline.generate_text("m2_character", payload, CharacterLLM(patch=patch))

    assert result == {**original, **patch}


def test_renaming_with_voice_lock_retains_the_existing_recorded_introduction():
    payload = _payload()
    payload["scope"] = "settings"
    payload["locked"]["voice"] = True
    original = copy.deepcopy(payload["character_result"])
    llm = CharacterLLM(patch={"name": "ミナ"})

    result = pipeline.generate_text("m2_character", payload, llm)

    assert result == {**original, "name": "ミナ", "locked": payload["locked"]}
    changes_schema = llm.calls[0]["schema"]["properties"]["changes"]["properties"]
    assert "selfIntroduction" not in changes_schema
