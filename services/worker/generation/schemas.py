"""Strict generated-result validation, independent of coordinator adoption."""

from __future__ import annotations

import copy
import math
from itertools import combinations

WORLD_TEXT = ("title", "prompt", "genre", "mood", "notes", "setting")
CHARACTER_TEXT = (
    "id",
    "name",
    "age",
    "gender",
    "role",
    "freeform",
    "settings",
    "appearance",
    "voice",
    "selfIntroduction",
)
SCOPE_FIELDS = {
    "settings": (
        "name",
        "age",
        "gender",
        "role",
        "freeform",
        "settings",
        "selfIntroduction",
        "sampleLines",
    ),
    "appearance": ("appearance", "height_cm", "body_type"),
    "voice": ("voice",),
}


def object_schema(properties: dict) -> dict:
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": properties,
        "required": list(properties),
    }


WORLD_SCHEMA = object_schema(
    {
        **{key: {"type": "string"} for key in WORLD_TEXT},
        "chapterCount": {"type": "integer", "minimum": 1, "maximum": 100},
    }
)
LEGACY_CHARACTER_SCHEMA = object_schema(
    {
        **{key: {"type": "string"} for key in CHARACTER_TEXT if key != "selfIntroduction"},
        "locked": object_schema({key: {"type": "boolean"} for key in SCOPE_FIELDS}),
        "height_cm": {"type": ["number", "null"], "minimum": 1, "maximum": 10_000},
        "body_type": {"type": "string", "enum": ["humanoid", "nonhumanoid", "unknown"]},
    }
)
CHARACTER_SCHEMA = object_schema(
    {
        **LEGACY_CHARACTER_SCHEMA["properties"],
        "selfIntroduction": {"type": "string"},
        "sampleLines": {"type": "array", "minItems": 3, "maxItems": 3, "items": {"type": "string"}},
    }
)
RELATIONSHIPS_SCHEMA = object_schema(
    {
        "pairs": {
            "type": "array",
            "minItems": 1,
            "maxItems": 3,
            "items": object_schema(
                {
                    "characterIds": {
                        "type": "array",
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "string"},
                    },
                    "summary": {"type": "string"},
                    "firstToSecond": {"type": "string"},
                    "secondToFirst": {"type": "string"},
                }
            ),
        }
    }
)
CANDIDATES_SCHEMA = object_schema(
    {
        "candidates": {
            "type": "array",
            "minItems": 3,
            "maxItems": 3,
            "items": object_schema(
                {key: {"type": "string"} for key in ("id", "concept", "tension")}
            ),
        }
    }
)
SELECTION_SCHEMA = object_schema({"selected_id": {"type": "string"}, "reason": {"type": "string"}})
IMAGE_VISUAL_FIELDS = (
    "subject", "body", "skin", "hair", "eyes", "clothing", "accessories", "other_features",
)
# Empty values mean unspecified/not applicable, including nonhuman anatomy.
# Requiring the keys keeps the local JSON grammar simple without inventing traits.
IMAGE_PROMPT_SCHEMA = object_schema(
    {key: {"type": "string"} for key in (*IMAGE_VISUAL_FIELDS, "rendering")}
)


def validate_schema(value: object, schema: dict, location: str = "result") -> None:
    """Validate every schema feature emitted above; reject bool as integer."""
    kind = schema["type"]
    if isinstance(kind, list):
        for candidate in kind:
            try:
                validate_schema(value, {**schema, "type": candidate}, location)
                return
            except ValueError:
                pass
        raise ValueError(f"{location} must be one of {kind}.")
    types = {"object": (dict,), "array": (list,), "string": (str,), "integer": (int,),
             "boolean": (bool,), "number": (int, float), "null": (type(None),)}
    if type(value) not in types[kind]:
        raise ValueError(f"{location} must be {kind}.")
    if "enum" in schema and value not in schema["enum"]:
        raise ValueError(f"{location} must be one of the permitted values.")
    if kind == "object":
        if not set(schema.get("required", schema["properties"])) <= set(value) or (
            set(value) - set(schema["properties"])
        ):
            raise ValueError(f"{location} has missing or unexpected fields.")
        for key, child in schema["properties"].items():
            if key in value:
                validate_schema(value[key], child, f"{location}.{key}")
    elif kind == "array":
        if not schema.get("minItems", 0) <= len(value) <= schema.get("maxItems", 10_000):
            raise ValueError(f"{location} has an invalid length.")
        for item in value:
            validate_schema(item, schema["items"], location + "[]")
    elif kind in ("integer", "number"):
        if not math.isfinite(value) or not (
            schema.get("minimum", -(2**63)) <= value <= schema.get("maximum", 2**63)
        ):
            raise ValueError(f"{location} is out of range.")
    elif kind == "string" and len(value) > 50_000:
        raise ValueError(f"{location} is too long.")


def validate_result(kind: str, value: dict, contract_version: int = 2) -> dict:
    schema = (
        WORLD_SCHEMA
        if kind == "m2_world"
        else (CHARACTER_SCHEMA if contract_version >= 2 else LEGACY_CHARACTER_SCHEMA)
    )
    if kind == "m2_character":
        # New model requests require presentation fields; saved old results remain valid.
        schema = {**schema, "required": [key for key in schema["required"]
                                        if key not in ("height_cm", "body_type")]}
    validate_schema(value, schema)
    mandatory = (
        ("title", "genre", "mood", "setting")
        if kind == "m2_world"
        else ("id", "name", "age", "gender", "role", "settings", "appearance", "voice")
    )
    if kind == "m2_character" and contract_version >= 2:
        mandatory += ("selfIntroduction",)
    if any(not value[key].strip() for key in mandatory):
        raise ValueError("Generated result has incomplete required content.")
    if (
        kind == "m2_character"
        and contract_version >= 2
        and any(not line.strip() for line in value["sampleLines"])
    ):
        raise ValueError("Generated sample dialogue is empty.")
    return value


def validate_relationships(value: dict, characters: list[dict]) -> dict:
    validate_schema(value, RELATIONSHIPS_SCHEMA)
    ids = [character["id"] for character in characters]
    if not 2 <= len(ids) <= 3 or len(set(ids)) != len(ids):
        raise ValueError("Relationships require two or three distinct completed characters.")
    # The direction text belongs to the supplied ID order. Canonicalizing only
    # the IDs would silently assign each character the other one's feelings.
    result = copy.deepcopy(value)
    for pair in result["pairs"]:
        first, second = pair["characterIds"]
        if first > second:
            pair["characterIds"] = [second, first]
            pair["firstToSecond"], pair["secondToFirst"] = (
                pair["secondToFirst"], pair["firstToSecond"]
            )
    expected = set(combinations(sorted(ids), 2))
    actual = [tuple(pair["characterIds"]) for pair in result["pairs"]]
    if len(set(actual)) != len(actual) or set(actual) != expected:
        raise ValueError(
            "Relationships must cover every distinct character pair exactly once in ID order."
        )
    for pair in result["pairs"]:
        if any(not pair[field].strip() for field in ("summary", "firstToSecond", "secondToFirst")):
            raise ValueError(
                "Relationship descriptions must include both characters' perspectives."
            )
    result["pairs"].sort(key=lambda pair: pair["characterIds"])
    return result


def preserve_character(generated: dict, payload: dict) -> dict:
    if not payload.get("character_id") or generated.get("id") != payload["character_id"]:
        raise ValueError("Generated character ID does not match the requested target.")
    result = copy.deepcopy(generated)
    previous = payload.get("character_result")
    locked = payload.get("locked") or {scope: False for scope in SCOPE_FIELDS}
    scope = payload.get("scope", "all")
    if previous:
        for field_scope, fields in SCOPE_FIELDS.items():
            if locked.get(field_scope) or scope not in ("all", field_scope):
                for field in fields:
                    if field in previous:
                        result[field] = previous[field]
                    elif field in ("height_cm", "body_type"):
                        result.pop(field, None)
        if locked.get("voice") and "selfIntroduction" in result:
            if not previous.get("selfIntroduction", "").strip():
                raise ValueError(
                    "Unlock the legacy reference voice before generating a self-introduction."
                )
            result["selfIntroduction"] = previous["selfIntroduction"]
    result["locked"] = {scope: bool(locked.get(scope)) for scope in SCOPE_FIELDS}
    return result
