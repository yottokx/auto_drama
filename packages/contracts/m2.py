"""M2 inputs, complete generation results and wizard action contracts."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .expressions import Expression
from .script import Contract, HeightCm, Identifier

Text = Annotated[str, StringConstraints(max_length=30_000)]
ResultText = Annotated[str, StringConstraints(min_length=1, max_length=30_000)]
CharacterId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")]
Step = Literal["world-input", "world-review", "character-input", "character-review", "planning-review", "production"]
Scope = Literal["settings", "appearance", "voice"]
Kind = Literal[
    "m2_world", "m2_character", "m2_image", "m2_voice", "m2_relationships", "m2_voice_clone",
    "m2_outfit_image", "m2_material_image",
]
SpokenText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=1000)]
BodyType = Literal["humanoid", "nonhumanoid", "unknown"]


class Locks(Contract):
    settings: bool = False
    appearance: bool = False
    voice: bool = False


class WorldBrief(Contract):
    title: Text = ""
    prompt: Text = ""
    genre: Text = ""
    mood: Text = ""
    notes: Text = ""
    chapterCount: int = Field(default=3, ge=1, le=100, strict=True)
    setting: Text = ""


class CharacterCore(Contract):
    id: CharacterId
    name: Text = ""
    age: Text = ""
    gender: Text = ""
    role: Text = ""
    freeform: Text = ""
    settings: Text = ""
    appearance: Text = ""
    height_cm: HeightCm | None = None
    body_type: BodyType = "unknown"
    voice: Text = ""
    locked: Locks = Field(default_factory=Locks)


class CharacterBrief(CharacterCore):
    selfIntroduction: Text = ""
    sampleLines: list[SpokenText] = Field(default_factory=list, max_length=10)


class WorldResult(WorldBrief):
    title: ResultText = Field(...)
    prompt: Text = Field(...)
    genre: ResultText = Field(...)
    mood: ResultText = Field(...)
    notes: Text = Field(...)
    chapterCount: int = Field(..., ge=1, le=100, strict=True)
    setting: ResultText = Field(...)

    @model_validator(mode="after")
    def complete_result(self):
        if set(type(self).model_fields) != self.model_fields_set:
            raise ValueError("world result must contain every field")
        if not all(getattr(self, name).strip() for name in ("title", "genre", "mood", "setting")):
            raise ValueError("world result is incomplete")
        return self


class LegacyCharacterResult(CharacterCore):
    name: ResultText = Field(...)
    age: ResultText = Field(...)
    gender: ResultText = Field(...)
    role: ResultText = Field(...)
    freeform: Text = Field(...)
    settings: ResultText = Field(...)
    appearance: ResultText = Field(...)
    voice: ResultText = Field(...)
    locked: Locks = Field(...)

    @model_validator(mode="after")
    def complete_result(self):
        required_fields = {
            name for name, field in type(self).model_fields.items() if field.is_required()
        }
        if not required_fields <= self.model_fields_set:
            raise ValueError("character result must contain every field")
        if not all(
            getattr(self, name).strip()
            for name in ("name", "age", "gender", "role", "settings", "appearance", "voice")
        ):
            raise ValueError("character result is incomplete")
        return self


class CharacterResult(LegacyCharacterResult):
    selfIntroduction: SpokenText = Field(...)
    sampleLines: list[SpokenText] = Field(..., min_length=3, max_length=3)


class RelationshipInput(Contract):
    characterIds: tuple[CharacterId, CharacterId]
    instruction: Text = ""

    @model_validator(mode="after")
    def distinct_characters(self):
        if self.characterIds[0] == self.characterIds[1]:
            raise ValueError("a relationship requires two distinct characters")
        return self


class RelationshipPair(Contract):
    characterIds: tuple[CharacterId, CharacterId]
    summary: ResultText
    firstToSecond: ResultText
    secondToFirst: ResultText

    @model_validator(mode="after")
    def complete_pair(self):
        if self.characterIds[0] >= self.characterIds[1]:
            raise ValueError("relationship result IDs must be distinct and sorted")
        if not all(
            getattr(self, key).strip() for key in ("summary", "firstToSecond", "secondToFirst")
        ):
            raise ValueError("relationship fields must not be blank")
        return self


class RelationshipsResult(Contract):
    pairs: list[RelationshipPair] = Field(max_length=3)


class M2ProjectInput(Contract):
    world: WorldBrief
    characters: list[CharacterBrief] = Field(default_factory=list, max_length=3)
    relationshipInputs: list[RelationshipInput] = Field(default_factory=list, max_length=3)

    @model_validator(mode="after")
    def unique_characters(self):
        if len({character.id for character in self.characters}) != len(self.characters):
            raise ValueError("character IDs must be unique")
        return self


class Action(Contract):
    expected_revision: int = Field(ge=1, strict=True)


class SaveWorld(Action):
    action: Literal["save-world", "edit-world"]
    world: WorldBrief


class GenerateWorld(Action):
    action: Literal["generate-world"]
    instruction: Text = ""


class SimpleAction(Action):
    action: Literal["confirm-world", "generate-characters", "approve"]
    reconfirm: bool = Field(default=False, strict=True)


class OutfitInput(Contract):
    """An alternate look of one main character. ``default`` names the approved base portrait."""

    id: Identifier | None = None
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=400)] = ""
    occasion: Annotated[str, StringConstraints(strip_whitespace=True, max_length=400)] = ""
    # What the look shows, as stated by a direct edit; ``description`` stays the user's wish.
    appearance: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=400)] | None = None
    # A shared material declared in the brief that this look wears; ``default`` carries the base look's.
    material: Identifier | None = None


class MaterialWish(Contract):
    """An outfit or item several characters share, declared with the world."""

    id: Identifier
    name: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=40)]
    description: Annotated[str, StringConstraints(strip_whitespace=True, max_length=400)] = ""


class SaveCharacters(Action):
    action: Literal["save-characters"]
    characters: list[CharacterBrief] = Field(min_length=1, max_length=3)
    relationshipInputs: list[RelationshipInput] | None = Field(default=None, max_length=3)
    # The alternate looks wished for each character; a character left out keeps its looks.
    outfits: dict[CharacterId, list[OutfitInput]] | None = None
    # The shared materials of the work; left out, they stay as they are.
    materials: list[MaterialWish] | None = Field(default=None, max_length=8)

    @model_validator(mode="after")
    def bounded_outfits(self):
        if any(len([row for row in rows if row.id != "default"]) > 4 for rows in (self.outfits or {}).values()):
            raise ValueError("a character has at most four alternate looks")
        if self.materials is not None and len({row.id for row in self.materials}) != len(self.materials):
            raise ValueError("shared materials need distinct identifiers")
        return self

    @model_validator(mode="after")
    def unique_characters(self):
        if len({character.id for character in self.characters}) != len(self.characters):
            raise ValueError("character IDs must be unique")
        return self


class SaveBrief(SaveCharacters):
    action: Literal["save-brief"]
    world: WorldBrief
    relationshipInputs: list[RelationshipInput] = Field(default_factory=list, max_length=3)


class ReviseCharacter(Action):
    action: Literal["revise-character"]
    character_id: CharacterId
    scope: Literal["all", "settings", "appearance", "voice"]
    instruction: Annotated[
        str, StringConstraints(strip_whitespace=True, min_length=1, max_length=30_000)
    ]


class EditCharacter(Action):
    action: Literal["edit-character"]
    character_id: CharacterId
    patch: dict[
        Literal[
            "name",
            "age",
            "gender",
            "role",
            "freeform",
            "settings",
            "appearance",
            "height_cm",
            "body_type",
            "voice",
            "selfIntroduction",
            "sampleLines",
        ],
        Text | list[SpokenText] | HeightCm | None,
    ]

    @model_validator(mode="after")
    def correct_patch_types(self):
        for key, value in self.patch.items():
            if key == "height_cm":
                if value is not None and type(value) not in (int, float):
                    raise ValueError("height_cm accepts only a number or null")
            elif key == "body_type":
                if value not in ("humanoid", "nonhumanoid", "unknown"):
                    raise ValueError("body_type must be humanoid, nonhumanoid or unknown")
            elif key == "sampleLines":
                if not isinstance(value, list):
                    raise ValueError("sampleLines accepts only an array")
            elif not isinstance(value, str):
                raise ValueError("character text fields accept only strings")
        return self


class ToggleLock(Action):
    action: Literal["toggle-lock"]
    character_id: CharacterId
    scope: Scope


class Retake(Action):
    action: Literal["retake"]
    character_id: CharacterId
    scope: Literal["image-retake", "voice-retake"]
    instruction: Text = ""


class SaveOutfit(Action):
    action: Literal["save-outfit"]
    character_id: CharacterId
    outfit: OutfitInput


class OutfitAction(Action):
    action: Literal["delete-outfit", "generate-outfit"]
    character_id: CharacterId
    outfit_id: Identifier
    # A shared design to put on this look; the base look is redrawn only this way.
    material_image_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9]{1,64}$")] | None = None
    clear_material: bool = False


ArtifactRef = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9]{1,64}$")]


class EditImage(Action):
    """Redraw a look from a chosen picture of the same character, changing only what is wished."""

    action: Literal["edit-image"]
    character_id: CharacterId
    outfit_id: Identifier
    instruction: Annotated[str, StringConstraints(strip_whitespace=True, max_length=1000)] = ""
    # Any earlier picture of this character, rejected ones included; the current one when left out.
    base_artifact_id: ArtifactRef | None = None
    material_image_id: ArtifactRef | None = None

    @model_validator(mode="after")
    def something_to_do(self):
        if not self.instruction and not self.material_image_id:
            raise ValueError("an edit needs a wish or a shared material")
        return self


class AdoptImage(Action):
    """Go back to an earlier picture of a look without drawing anything."""

    action: Literal["adopt-image"]
    character_id: CharacterId
    outfit_id: Identifier
    artifact_id: ArtifactRef


FaceFraction = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]


class GenerateExpressions(Action):
    """Draw expression pictures of a character's looks; none of them is ever required."""

    action: Literal["generate-expressions"]
    character_id: CharacterId
    # The looks to draw them for; every look that has a current portrait when left out.
    outfit_ids: list[Identifier] | None = Field(default=None, min_length=1, max_length=5)
    expressions: list[Expression] = Field(min_length=1, max_length=11)
    # Draw again even what exists and still matches its portrait.
    redraw: bool = False
    # The face marked by hand (left, top, right, bottom as fractions of the picture), for a
    # portrait whose face could not be found. It applies to a single look.
    face_box: tuple[FaceFraction, FaceFraction, FaceFraction, FaceFraction] | None = None

    @model_validator(mode="after")
    def drawable(self):
        if "neutral" in self.expressions or len(set(self.expressions)) != len(self.expressions):
            raise ValueError("name each expression once; the ordinary face is the portrait itself")
        if self.face_box is not None and (
                self.outfit_ids is None or len(self.outfit_ids) != 1
                or self.face_box[2] - self.face_box[0] < 0.02 or self.face_box[3] - self.face_box[1] < 0.02):
            raise ValueError("a marked face belongs to one look and has a size")
        return self


class DeleteExpression(Action):
    action: Literal["delete-expression"]
    character_id: CharacterId
    outfit_id: Identifier
    expression: Expression


class SetImageEditing(Action):
    """Whether this work may use the image editing model at all."""

    action: Literal["set-image-editing"]
    enabled: bool = Field(strict=True)


class GoTo(Action):
    action: Literal["go-to"]
    step: Step


class SaveRelationships(Action):
    action: Literal["save-relationships"]
    relationshipInputs: list[RelationshipInput] = Field(max_length=3)


class GenerateRelationships(Action):
    action: Literal["generate-relationships"]
    instruction: Text = ""


class CloneVoice(Action):
    action: Literal["clone-voice"]
    character_id: CharacterId
    text: SpokenText


M2Action = Annotated[
    SaveWorld
    | SaveBrief
    | GenerateWorld
    | SimpleAction
    | SaveCharacters
    | ReviseCharacter
    | EditCharacter
    | ToggleLock
    | Retake
    | SaveOutfit
    | OutfitAction
    | EditImage
    | AdoptImage
    | GenerateExpressions
    | DeleteExpression
    | SetImageEditing
    | GoTo
    | SaveRelationships
    | GenerateRelationships
    | CloneVoice,
    Field(discriminator="action"),
]


class GenerationEnvelope(Contract):
    schema_version: Literal[1]
    kind: Kind
    result: dict
    provenance: dict
    trace: list = Field(max_length=10_000)
