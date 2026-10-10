"""Alternate looks of an approved main character, chosen per scene by the script plan.

A character without alternates has no outfit data anywhere: every field that
carries it is omitted when empty, so such productions keep their exact bytes.
"""

from __future__ import annotations

from typing import Annotated

from pydantic import Field, StringConstraints

from .script import Contract, Identifier, portrait_key

DEFAULT_OUTFIT = "default"
CharacterId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]
OutfitText = Annotated[str, StringConstraints(max_length=400)]


class OutfitDefinition(Contract):
    """``default`` may be listed to name and describe the approved base portrait."""

    id: Identifier
    name: Annotated[str, StringConstraints(min_length=1, max_length=40)]
    description: OutfitText = ""
    occasion: OutfitText = ""


class OutfitAppearance(Contract):
    """A planned chapter in which an alternate look is actually shown."""

    character_id: CharacterId
    outfit_id: Identifier
    occasion: str = Field(min_length=1, max_length=160,
                          description="当章のどの出来事・状況でその姿になるかを1文で記す。")


class SceneOutfit(Contract):
    """The look a character has when the scene opens, and one change inside it."""

    character_id: CharacterId
    outfit_id: Identifier = Field(description="場面の開始時の姿。")
    change_to: Identifier | None = Field(
        default=None, exclude_if=lambda value: value is None,
        description="場面の途中で変わる先の姿。変わらなければnull。")


def image_target(character_id: str, outfit_id: str | None) -> str:
    """Asset requirement target; the base portrait keeps the character's own ID."""
    return portrait_key(character_id, outfit_id)


def snapshot_outfits(snapshot: dict) -> dict[str, list[OutfitDefinition]]:
    """Main characters that have at least one alternate look, in approval order."""
    result = {}
    for row in snapshot["characters"]:
        outfits = [OutfitDefinition.model_validate(
            {key: value for key, value in item.items() if key in OutfitDefinition.model_fields})
            for item in row.get("outfits", [])]
        ids = [item.id for item in outfits]
        if len(ids) != len(set(ids)):
            raise ValueError("A character's outfit IDs must be unique.")
        if any(item.id != DEFAULT_OUTFIT for item in outfits):
            if DEFAULT_OUTFIT not in ids:
                outfits.insert(0, OutfitDefinition(id=DEFAULT_OUTFIT, name="通常の姿"))
            result[row.get("id", row["result"]["id"])] = outfits
    return result


def outfits_after(scenes: list, state: dict[str, str] | None = None) -> dict[str, str]:
    """Each character's look once the given scene plans have played, in order."""
    state = dict(state or {})
    for plan in scenes:
        for cid in plan.character_ids:
            state[cid] = DEFAULT_OUTFIT
        for row in plan.outfits:
            state[row.character_id] = row.change_to or row.outfit_id
    return state


def look_timeline(scenes: list[dict]) -> dict[str, dict[str, str]]:
    """The look each character shows at every utterance of a saved narrative.

    Only characters that ever leave their base look appear; an empty result
    means the chapter has no alternate looks at all.
    """
    timeline, involved = {}, set()
    for scene in scenes:
        looks = {row["character_id"]: row["outfit_id"] for row in scene["plan"].get("outfits", [])}
        changes = {}
        for row in scene.get("directions", []):
            if row["kind"] == "outfit":
                changes.setdefault(row["utterance_id"], {})[row["character_id"]] = row["outfit_id"]
        involved.update(looks, *(change for change in changes.values()))
        for utterance in scene["utterances"]:
            looks = {**looks, **changes.get(utterance["id"], {})}
            timeline[utterance["id"]] = looks
    if not involved:
        return {}
    return {uid: {cid: looks.get(cid, DEFAULT_OUTFIT) for cid in sorted(involved)}
            for uid, looks in timeline.items()}
