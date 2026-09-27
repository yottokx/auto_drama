"""Presentation dimensions are explicit, optional for old results and never stretch sprites."""

import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_m2_generation import FakeLLM as M2FakeLLM
from test_m2_generation import character, job
from test_m3_narrative import FakeLLM, narrative_fixture, narrative_responses

from packages.contracts.m2 import CharacterResult, EditCharacter, LegacyCharacterResult
from packages.contracts.script import Character
from services.worker.generation.narrative import generate_narrative
from services.worker.generation.pipeline import generate_text
from services.worker.generation.schemas import (
    CHARACTER_SCHEMA,
    preserve_character,
    validate_result,
    validate_schema,
)


def test_old_character_results_and_scripts_keep_optional_presentation_fields():
    value = character()
    result = CharacterResult.model_validate(value)
    assert result.height_cm is None and result.body_type == "unknown"
    assert result.model_dump(exclude_unset=True) == value
    legacy = {k: v for k, v in value.items() if k not in ("sampleLines", "selfIntroduction")}
    assert LegacyCharacterResult.model_validate(legacy).model_dump(exclude_unset=True) == legacy
    assert Character(id="child", name="子供").height_cm is None
    assert validate_result("m2_character", value) == value


def test_new_generation_schema_requires_dimensions_but_saved_old_results_still_load():
    value = {**character(), "height_cm": 115, "body_type": "humanoid"}
    validate_schema(value, CHARACTER_SCHEMA)
    assert CharacterResult.model_validate(value).height_cm == 115
    validate_schema({**value, "height_cm": None, "body_type": "nonhumanoid"}, CHARACTER_SCHEMA)
    with pytest.raises(ValueError, match="missing"):
        validate_schema(character(), CHARACTER_SCHEMA)
    schema = json.loads(Path("packages/contracts/m2-character.schema.json").read_text("utf-8"))
    assert schema == CharacterResult.model_json_schema()
    assert "height_cm" not in schema["required"]


@pytest.mark.parametrize("height", [0, -1, 10001, float("inf"), float("nan"), True, "120"])
def test_height_rejects_invalid_measurements_in_input_script_and_generated_result(height):
    value = {**character(), "height_cm": height, "body_type": "humanoid"}
    with pytest.raises(ValidationError):
        CharacterResult.model_validate(value)
    with pytest.raises(ValidationError):
        Character(id="child", name="子供", height_cm=height)
    with pytest.raises(ValueError):
        validate_result("m2_character", value)


@pytest.mark.parametrize("patch", [
    {"height_cm": "115"}, {"height_cm": True}, {"height_cm": 0},
    {"height_cm": ["115"]}, {"body_type": "dog"}, {"body_type": None},
    {"body_type": ["humanoid"]}, {"appearance": 120}, {"appearance": None},
])
def test_edit_patch_does_not_coerce_text_or_invalid_height(patch):
    with pytest.raises(ValidationError):
        EditCharacter(action="edit-character", expected_revision=1,
                      character_id="character-1", patch=patch)


def test_edit_patch_accepts_height_body_type_and_explicit_unknown():
    for patch in ({"height_cm": 115.5, "body_type": "humanoid"},
                  {"height_cm": None, "body_type": "unknown"}):
        edit = EditCharacter(action="edit-character", expected_revision=1,
                             character_id="character-1", patch=patch)
        assert edit.patch == patch


@pytest.mark.parametrize("scope,locked", [("voice", False), ("settings", False), ("all", True)])
def test_partial_or_locked_appearance_preserves_dimensions(scope, locked):
    payload = job("m2_character")["payload"]
    payload["character_result"].update(height_cm=115, body_type="humanoid")
    payload["scope"] = scope
    payload["locked"]["appearance"] = locked
    generated = {**character(), "height_cm": 175, "body_type": "unknown"}
    preserved = preserve_character(generated, payload)
    assert preserved["height_cm"] == 115 and preserved["body_type"] == "humanoid"
    del payload["character_result"]["height_cm"], payload["character_result"]["body_type"]
    preserved = preserve_character(generated, payload)
    assert "height_cm" not in preserved and "body_type" not in preserved


def test_main_character_generation_requests_user_height_without_changing_proportions():
    payload = job("m2_character")["payload"]
    payload["character_input"]["height_cm"] = 115

    class CaptureLLM(M2FakeLLM):
        def structured(self, stage, messages, schema):
            if stage == "final":
                prompt = messages[-1]["content"]
                assert '"height_cm": 115' in prompt
                assert "ユーザーが明示した身長を最優先" in prompt
                assert "元の頭身・画風を変えたり" in prompt
                assert {"height_cm", "body_type"} <= set(schema["required"])
                return {**character(), "height_cm": 115, "body_type": "humanoid"}
            return super().structured(stage, messages, schema)

    result = generate_text("m2_character", payload, CaptureLLM(None, None, payload, None))
    assert result["height_cm"] == 115


def test_supporting_character_generation_requests_height_in_strict_schema():
    result, snapshot = narrative_fixture()
    before = copy.deepcopy(snapshot)
    llm = FakeLLM(narrative_responses(result))
    generate_narrative({"approval_snapshot": snapshot}, llm)
    args, kwargs = llm.calls[1]
    assert args[0] == "supporting_character"
    schema = kwargs["response_format"]["json_schema"]["schema"]
    assert {"height_cm", "body_type"} <= set(schema["$defs"]["CharacterResult"]["required"])
    assert "頭身や画風を変えません" in str(args)
    assert snapshot == before
