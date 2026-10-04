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
DisplayOffset = Annotated[float, Field(strict=True, ge=-640, le=640, allow_inf_nan=False)]
DisplayScale = Annotated[float, Field(strict=True, ge=0.1, le=3, allow_inf_nan=False)]
StageCoordinate = Annotated[float, Field(strict=True, ge=-1_000_000, le=1_000_000, allow_inf_nan=False)]
MusicSeconds = Annotated[float, Field(strict=True, ge=0, le=3600, allow_inf_nan=False)]
MusicVolume = Annotated[float, Field(strict=True, ge=0, le=1, allow_inf_nan=False)]


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
    kind: Literal["background", "character", "audio", "music"]
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
        supported = {"png", "jpg", "jpeg", "webp"}
        if self.kind == "audio":
            supported = {"wav", "mp3", "ogg", "m4a"}
        elif self.kind == "music":
            supported = {"mp3"}
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
    # Default presentation adds no fields to the established Script bytes.
    offset_y: DisplayOffset = Field(default=0, exclude_if=lambda value: value == 0)
    scale: DisplayScale = Field(default=1, exclude_if=lambda value: value == 1)


class PortraitTransform(Contract):
    offset_y: DisplayOffset = 0
    scale: DisplayScale = 1


class PortraitGeometry(Contract):
    width: Annotated[int, Field(strict=True, ge=1, le=1_000_000)]
    height: Annotated[int, Field(strict=True, ge=1, le=1_000_000)]
    top: Annotated[int, Field(strict=True, ge=-1_000_000, le=1_000_000)]
    center_offset: StageCoordinate
    anchor_y: StageCoordinate
    mode: Literal["upper_body", "full_body"]
    source_sha256: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class PortraitBaseline(Contract):
    schema_version: Literal[1] = 1
    pixels_per_cm: Annotated[float, Field(strict=True, gt=0, le=1_000, allow_inf_nan=False)] | None = None
    ground_y: StageCoordinate | None = None
    layouts: dict[Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")], PortraitGeometry] = Field(default_factory=dict, max_length=100)

    @model_validator(mode="after")
    def paired_height_geometry(self) -> Self:
        if (self.pixels_per_cm is None) != (self.ground_y is None):
            raise ValueError("physical portrait baseline requires both scale and ground line")
        return self


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


class SceneTransitionSpec(Contract):
    """Bounded presentation choices decided after the chapter text is frozen."""

    visual: Literal["none", "dissolve", "fade", "cut"] = "fade"
    duration_ms: Annotated[int, Field(strict=True, ge=0, le=3000)] = 500
    music_fade_out_ms: Annotated[int, Field(strict=True, ge=0, le=5000)] = 1000
    music_fade_in_ms: Annotated[int, Field(strict=True, ge=0, le=5000)] = 1000


class SceneTransitionCue(SceneTransitionSpec):
    id: Identifier
    utterance_id: Identifier


class MusicCue(Contract):
    """Backend-neutral scene music; media and engine code are adopted separately."""

    id: Identifier
    utterance_id: Identifier
    action: Literal["play", "stop", "continue"]
    asset_id: Identifier | None = Field(default=None, exclude_if=lambda value: value is None)
    loop_start_seconds: MusicSeconds | None = Field(default=None, exclude_if=lambda value: value is None)
    loop_end_seconds: MusicSeconds | None = Field(default=None, exclude_if=lambda value: value is None)
    volume: MusicVolume = 0.35

    @model_validator(mode="after")
    def playback_parameters_match_action(self) -> Self:
        if self.action == "play":
            if self.asset_id is None:
                raise ValueError("play music cue requires an adopted music asset")
            if (self.loop_start_seconds is None) != (self.loop_end_seconds is None):
                raise ValueError("music loop points must be provided together")
            if (self.loop_start_seconds is not None
                    and self.loop_end_seconds <= self.loop_start_seconds):
                raise ValueError("music loop points must satisfy 0 <= A < B")
        elif (self.asset_id is not None or self.loop_start_seconds is not None
                or self.loop_end_seconds is not None):
            raise ValueError("stop and continue music cues cannot select media or loop points")
        return self


class Script(Contract):
    schema_version: Literal[1] = 1
    id: Identifier
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    characters: list[Character] = Field(default_factory=list, max_length=100)
    utterances: list[Utterance] = Field(min_length=1, max_length=10_000)
    directions: list[Direction] = Field(default_factory=list, max_length=50_000)
    assets: list[AssetReference] = Field(default_factory=list, max_length=20_000)
    music_cues: list[MusicCue] = Field(default_factory=list, max_length=10_000,
                                      exclude_if=lambda value: not value)
    scene_transitions: list[SceneTransitionCue] = Field(default_factory=list, max_length=10_000,
                                                      exclude_if=lambda value: not value)
    portrait_baseline: PortraitBaseline | None = Field(default=None, exclude_if=lambda value: value is None)

    @field_validator("schema_version", mode="before")
    @classmethod
    def strict_schema_version(cls, value):
        if type(value) is not int:
            raise ValueError("schema_version must be an integer")
        return value

    @model_validator(mode="after")
    def validate_references(self) -> Self:
        for values in (self.characters, self.utterances, self.directions, self.assets,
                       self.music_cues, self.scene_transitions):
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
            if (self.portrait_baseline is not None and character.image_asset_id is not None
                    and character.id not in self.portrait_baseline.layouts):
                raise ValueError("fixed portrait baseline is missing a displayed character")
        for utterance in self.utterances:
            if utterance.speaker_id is not None and utterance.speaker_id not in characters:
                raise ValueError(f"unknown speaker: {utterance.speaker_id}")
            asset(utterance.audio_asset_id, "audio")
        music_anchors = [value.utterance_id for value in self.music_cues]
        if len(music_anchors) != len(set(music_anchors)):
            raise ValueError("one music cue is allowed per utterance")
        for cue in self.music_cues:
            if cue.utterance_id not in utterances:
                raise ValueError(f"unknown music cue utterance: {cue.utterance_id}")
            asset(cue.asset_id, "music")
        transition_anchors = [value.utterance_id for value in self.scene_transitions]
        if len(transition_anchors) != len(set(transition_anchors)):
            raise ValueError("one scene transition is allowed per utterance")
        for transition in self.scene_transitions:
            if transition.utterance_id not in utterances:
                raise ValueError(f"unknown scene transition utterance: {transition.utterance_id}")
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
