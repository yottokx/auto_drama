"""Presentation-only drafts over a complete immutable story publication."""
from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .m2 import CharacterId
from .script import Contract, PortraitBounds

Id = Annotated[str, StringConstraints(min_length=1, max_length=128)]
Revision = Annotated[int, Field(strict=True, ge=1)]


class AdjustmentCharacter(Contract):
    character_id: CharacterId
    image_candidate_id: Id | None = None
    voice_candidate_id: Id
    framing: Literal["auto", "upper_body", "full_body"] = "auto"
    height_cm: float | None = Field(default=None, strict=True, ge=1, le=10000, allow_inf_nan=False)
    body_bounds: PortraitBounds | None = None
    offset_y: float = Field(default=0, ge=-640, le=640, allow_inf_nan=False)
    scale: float = Field(default=1, ge=0.1, le=3, allow_inf_nan=False)


class AdjustmentSave(Contract):
    expected_revision: Revision
    characters: list[AdjustmentCharacter] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def unique_people(self):
        ids = [row.character_id for row in self.characters]
        if len(set(ids)) != len(ids):
            raise ValueError("Each character needs one presentation setting.")
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
