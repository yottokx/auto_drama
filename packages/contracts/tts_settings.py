"""TTS settings and independently managed model downloads."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

TTSPrecision = Literal["fp32", "bf16", "int8", "int4"]
TTSModelId = Literal["irodori-v4.1-small", "irodori-v4-large"]


class TTSModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TTSSelection(TTSModel):
    provider_id: Literal["irodori"] = "irodori"
    model_id: TTSModelId = "irodori-v4.1-small"
    precision: TTSPrecision = "fp32"


class TTSSettings(TTSModel):
    schema_version: Literal[1] = 1
    generation_active: Literal[True] = True
    voice_design: TTSSelection = Field(default_factory=TTSSelection)
    voice_clone: TTSSelection = Field(default_factory=TTSSelection)


class DownloadInput(TTSModel):
    model_id: TTSModelId
    precision: TTSPrecision


class DownloadResume(TTSModel):
    worker_id: str | None = Field(default=None, min_length=1, max_length=120)


class DownloadProgress(TTSModel):
    worker_id: str = Field(min_length=1, max_length=120)
    lease_id: str = Field(min_length=1, max_length=120)
    status: Literal["downloading", "verifying", "completed", "cancelled", "failed"]
    done_bytes: int = Field(default=0, ge=0, strict=True)
    total_bytes: int = Field(default=0, ge=0, strict=True)
    phase: str | None = Field(default=None, max_length=120)
    current_file: str | None = Field(default=None, max_length=512)
    error: str | None = Field(default=None, max_length=2000)


class TTSFileInventory(TTSModel):
    model_id: TTSModelId
    precision: TTSPrecision
    manifest_id: str = Field(min_length=1, max_length=256)
    file_download_ready: bool = Field(strict=True)
    generation_ready: bool = Field(default=False, strict=True)
    generation_purposes: list[Literal["voice_design", "voice_clone"]] = Field(default_factory=list, max_length=2)


class TTSInventory(TTSModel):
    models: list[TTSFileInventory] = Field(default_factory=list, max_length=256)
