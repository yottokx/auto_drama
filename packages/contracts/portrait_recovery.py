"""Evidence-bearing result for failure-only supporting portrait recovery."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class PortraitDiagnosis(BaseModel):
    model_config = ConfigDict(extra="forbid")
    action: Literal["repair_prompt", "omit_portrait", "stop"]
    reason: str = Field(min_length=1, max_length=1000)
    source_field: Literal["appearance", "freeform", "settings", "none"]
    source_quote: str = Field(max_length=1500)
    revised_prompt: str = Field(max_length=5000)


class PortraitOmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status: Literal["omitted"]
    character_id: str = Field(min_length=1)
    failed_stage: Literal["background_removal"]
    failure: str = Field(min_length=1, max_length=4000)
    diagnosis: PortraitDiagnosis

    def validate_source(self, character: dict) -> None:
        diagnosis = self.diagnosis
        if (self.character_id != character.get("id")
                or diagnosis.action != "omit_portrait" or diagnosis.revised_prompt
                or diagnosis.source_field == "none" or not diagnosis.source_quote.strip()
                or diagnosis.source_quote not in character.get(diagnosis.source_field, "")):
            raise ValueError("Portrait omission requires an exact quotation from the original character.")
