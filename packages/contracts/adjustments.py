"""Presentation-only drafts over a complete immutable story publication."""
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .expressions import Expression
from .m2 import CharacterId
from .m3 import Emotion
from .music import SceneMusicSetting
from .script import Contract, Identifier, PortraitBounds

Id = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Revision = Annotated[int, Field(strict=True, ge=1)]


class AdjustmentCharacter(Contract):
    character_id: CharacterId
    image_candidate_id: Id | None = None
    # An alternate look of a main character is adjusted as an image-only row.
    voice_candidate_id: Id | None = None
    framing: Literal["auto", "upper_body", "full_body"] = "auto"
    height_cm: float | None = Field(default=None, strict=True, ge=1, le=10000, allow_inf_nan=False)
    body_bounds: PortraitBounds | None = None
    offset_y: float = Field(default=0, ge=-640, le=640, allow_inf_nan=False)
    scale: float = Field(default=1, ge=0.1, le=3, allow_inf_nan=False)


class ScriptLineEdit(Contract):
    """What a published line should say. Its speaker and its place in the script stay as they are.

    ``audio`` and ``voice_job_id`` settle the voice of a line whose reading, emotion or delivery
    changed: keep the old recording, or use the candidate that job made for exactly these words.
    """

    production_id: Id
    utterance_id: Identifier
    display_text: Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
    spoken_text: Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
    voice_emotion: Emotion
    delivery: str = Field(default="", max_length=200)
    audio: Literal["old"] | None = None
    voice_job_id: Id | None = None

    @model_validator(mode="after")
    def words_present(self):
        if not self.display_text.strip() or not self.spoken_text.strip():
            raise ValueError("台詞・地の文を空にすることはできません。")
        if self.audio is not None and self.voice_job_id is not None:
            raise ValueError("旧音声と新しい候補の一方を選んでください。")
        return self


class ScriptDirectionEdit(Contract):
    """A changed or disabled direction; a disabled one is left out of the published script."""

    production_id: Id
    direction_id: Identifier
    disabled: bool = Field(default=False, strict=True)
    expression: Expression | None = None
    duration_ms: int | None = Field(default=None, strict=True, ge=1, le=10_000)


class ScriptEdits(Contract):
    lines: list[ScriptLineEdit] = Field(default_factory=list, max_length=20_000)
    directions: list[ScriptDirectionEdit] = Field(default_factory=list, max_length=50_000)

    @model_validator(mode="after")
    def one_edit_each(self):
        lines = [(row.production_id, row.utterance_id) for row in self.lines]
        directions = [(row.production_id, row.direction_id) for row in self.directions]
        if len(set(lines)) != len(lines) or len(set(directions)) != len(directions):
            raise ValueError("Each line and direction needs one edit.")
        return self


class AdjustmentSave(Contract):
    expected_revision: Revision
    characters: list[AdjustmentCharacter] = Field(min_length=1, max_length=100)
    scene_music: list[SceneMusicSetting] | None = Field(default=None, max_length=800)
    script: ScriptEdits | None = None

    @model_validator(mode="after")
    def unique_people(self):
        ids = [row.character_id for row in self.characters]
        if len(set(ids)) != len(ids):
            raise ValueError("Each character needs one presentation setting.")
        if self.scene_music is not None:
            scopes = [(row.production_id, row.scene_id) for row in self.scene_music]
            if len(set(scopes)) != len(scopes):
                raise ValueError("Each scene needs one music setting.")
        return self


class AdjustmentRevision(Contract):
    expected_revision: Revision


class AdjustmentStart(Contract):
    expected_edition_id: Id | None = None


class AdjustmentGenerate(AdjustmentRevision):
    character_id: CharacterId
    kind: Literal["image", "voice"]
    instruction: str = Field(default="", max_length=10000)
    source_prompt: str | None = Field(default=None, min_length=1, max_length=10000)
    reference_text: str | None = Field(default=None, min_length=1, max_length=2000)

    @model_validator(mode="after")
    def generation_prompt(self):
        if self.source_prompt is not None and not self.source_prompt.strip():
            raise ValueError("編集したプロンプトを空にすることはできません。")
        if not self.instruction.strip() and self.source_prompt is None:
            raise ValueError("生成指示または編集したプロンプトを入力してください。")
        return self


class AdjustmentSample(AdjustmentRevision):
    candidate_id: Id


class AdjustmentMusicGenerate(AdjustmentRevision):
    production_id: Id
    scene_id: Id
    instruction: str = Field(default="", max_length=10000)
    source_prompt: str | None = Field(default=None, min_length=1, max_length=10000)

    @model_validator(mode="after")
    def prompt_not_blank(self):
        if self.source_prompt is not None and not self.source_prompt.strip():
            raise ValueError("音楽プロンプトを空にすることはできません。")
        return self


class ScriptVoiceRequest(AdjustmentRevision):
    """Record one line again in the speaker's chosen voice, with these words and this feeling."""

    production_id: Id
    utterance_id: Identifier
    spoken_text: Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
    voice_emotion: Emotion
    delivery: str = Field(default="", max_length=200)


class ScriptPreview(Contract):
    lines: list[ScriptLineEdit] = Field(default_factory=list, max_length=20_000)
    directions: list[ScriptDirectionEdit] = Field(default_factory=list, max_length=50_000)
