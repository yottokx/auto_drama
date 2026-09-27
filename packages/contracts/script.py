"""M1 intermediate script. Arbitrary engine tags and JavaScript are not part of this format."""

from __future__ import annotations

import re
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)

Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_-]{0,63}$")]
Text = Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
Position = Literal["left", "center", "right"]
Duration = Annotated[int, Field(strict=True, ge=0, le=10_000)]
HeightCm = Annotated[float, Field(strict=True, ge=1, le=10_000, allow_inf_nan=False)]
NormalizedCoordinate = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @field_validator("*", mode="after")
    @classmethod
    def no_control_characters(cls, value):
        if isinstance(value, str) and any(
            ord(char) < 32 and char not in "\t\n\r" or 0xD800 <= ord(char) <= 0xDFFF
            for char in value
        ):
            raise ValueError("control characters and unpaired surrogates are not supported")
        return value


class AssetReference(Contract):
    id: Identifier
    kind: Literal["background", "character", "audio"]
    artifact_id: Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")]
    filename: Annotated[str, StringConstraints(min_length=1, max_length=128)]
    sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]

    @model_validator(mode="after")
    def safe_filename(self) -> Self:
        # Restrict both URL syntax and Windows filename syntax, independently of host OS.
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.[A-Za-z0-9]+", self.filename):
            raise ValueError("asset filename must be a safe ASCII basename with an extension")
        stem = self.filename.split(".", 1)[0].upper()
        if stem in {"CON", "PRN", "AUX", "NUL", "CONIN$", "CONOUT$"} or re.fullmatch(
            r"(?:COM|LPT)[1-9]", stem
        ):
            raise ValueError("Windows device filenames are not supported")
        extension = self.filename.rsplit(".", 1)[1].lower()
        supported = (
            {"wav", "mp3", "ogg", "m4a"} if self.kind == "audio" else {"png", "jpg", "jpeg", "webp"}
        )
        if extension not in supported:
            raise ValueError("asset extension does not match its supported media kind")
        return self


class PortraitBounds(Contract):
    left: NormalizedCoordinate
    top: NormalizedCoordinate
    right: NormalizedCoordinate
    bottom: NormalizedCoordinate

    @model_validator(mode="after")
    def ordered_with_minimum_size(self) -> Self:
        # Decimal browser coordinates can differ by a few binary floating-point ulps.
        if self.right - self.left < 0.05 - 1e-9 or self.bottom - self.top < 0.05 - 1e-9:
            raise ValueError("portrait bounds must be ordered and at least 0.05 wide and high")
        return self


class Character(Contract):
    id: Identifier
    name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    image_asset_id: Identifier | None = None
    framing: Literal["auto", "upper_body", "full_body"] = "auto"
    height_cm: HeightCm | None = None
    body_bounds: PortraitBounds | None = None


class Utterance(Contract):
    id: Identifier
    speaker_id: Identifier | None = None
    display_text: Text
    spoken_text: Text
    voice_emotion: Annotated[str, StringConstraints(min_length=1, max_length=80)] = "neutral"
    delivery: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None
    audio_asset_id: Identifier | None = None


class DirectionBase(Contract):
    id: Identifier
    utterance_id: Identifier
    timing: Literal["before", "start", "after"] = "before"


class BackgroundDirection(DirectionBase):
    kind: Literal["background"]
    asset_id: Identifier
    duration_ms: Duration = 0


class EnterDirection(DirectionBase):
    kind: Literal["enter"]
    character_id: Identifier
    position: Position = "center"
    duration_ms: Duration = 0


class ExitDirection(DirectionBase):
    kind: Literal["exit"]
    character_id: Identifier
    duration_ms: Duration = 0


class PositionDirection(DirectionBase):
    kind: Literal["position"]
    character_id: Identifier
    position: Position
    duration_ms: Duration = 0


class FocusDirection(DirectionBase):
    kind: Literal["focus"]
    character_id: Identifier


class BlackoutDirection(DirectionBase):
    """Fade to black, then fade back in; duration is per fade, in milliseconds."""

    kind: Literal["blackout"]
    duration_ms: Annotated[int, Field(strict=True, ge=1, le=10_000)] = 300


class PauseDirection(DirectionBase):
    kind: Literal["pause"]
    duration_ms: Annotated[int, Field(strict=True, ge=1, le=10_000)]


Direction = Annotated[
    BackgroundDirection
    | EnterDirection
    | ExitDirection
    | PositionDirection
    | FocusDirection
    | BlackoutDirection
    | PauseDirection,
    Field(discriminator="kind"),
]


class Script(Contract):
    schema_version: Literal[1] = 1
    id: Identifier
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    characters: list[Character] = Field(default_factory=list, max_length=100)
    utterances: list[Utterance] = Field(min_length=1, max_length=10_000)
    directions: list[Direction] = Field(default_factory=list, max_length=50_000)
    assets: list[AssetReference] = Field(default_factory=list, max_length=20_000)

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_schema_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        for values in (self.characters, self.utterances, self.directions, self.assets):
            ids = [value.id for value in values]
            if len(ids) != len(set(ids)):
                raise ValueError("duplicate stable IDs in a script collection")
        characters = {value.id: value for value in self.characters}
        utterances = {value.id for value in self.utterances}
        assets = {value.id: value for value in self.assets}
        paths = [(value.kind, value.filename.casefold()) for value in self.assets]
        if len(paths) != len(set(paths)):
            raise ValueError("asset output filenames collide")

        def asset(reference: str | None, kind: str) -> None:
            if reference is not None and (
                reference not in assets or assets[reference].kind != kind
            ):
                raise ValueError(f"unknown or incorrectly typed {kind} asset: {reference}")

        for character in self.characters:
            asset(character.image_asset_id, "character")
        for utterance in self.utterances:
            if utterance.speaker_id is not None and utterance.speaker_id not in characters:
                raise ValueError(f"unknown speaker: {utterance.speaker_id}")
            asset(utterance.audio_asset_id, "audio")
        for direction in self.directions:
            if direction.utterance_id not in utterances:
                raise ValueError(f"unknown direction utterance: {direction.utterance_id}")
            if direction.kind == "background":
                asset(direction.asset_id, "background")
            if direction.kind in {"enter", "exit", "position", "focus"}:
                character = characters.get(direction.character_id)
                if character is None or character.image_asset_id is None:
                    raise ValueError("character direction requires a character with an image")
        return self
