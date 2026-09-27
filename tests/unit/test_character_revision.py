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
    assert [call["stage"] for call in llm.calls] == ["character-revision"]
    assert llm.calls[0]["schema"]["properties"]["id"]["enum"] == ["character-2"]


@pytest.mark.parametrize("revision", [False, True], ids=["initial-generation", "revision"])
def test_every_character_stage_excludes_other_characters_full_settings(revision):
    payload = _payload(revision=revision)
    llm = CharacterLLM()

    pipeline.generate_text("m2_character", payload, llm)

    expected = ["character-revision"] if revision else [
        "direction-candidates", "direction-consider", "direction-selected",
        "details-candidates", "details-consider", "details-selected", "final",
    ]
    assert [call["stage"] for call in llm.calls] == expected
    for call in llm.calls:
        text = json.dumps(call["messages"], ensure_ascii=False)
        assert "PRIVATE_PROTAGONIST_" not in text, call["stage"]
        context = json.JSONDecoder().raw_decode(call["messages"][1]["content"])[0]
        assert [item["id"] for item in context["cast_inputs"]] == ["character-2"]
        assert context["character_id"] == "character-2"
        assert context["character_input"] == payload["character_input"]
        assert context["character_result"] == payload["character_result"]
        assert "cast_results" not in context
        references = context["other_characters"]
        assert [item["id"] for item in references] == ["character-1"]
        assert references[0]["name"] == "リオン"
        assert not set(PRIVATE_FIELDS) & references[0].keys()
        assert set(references[0]) <= {"id", "name", "age", "gender", "role", "freeform"}


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
