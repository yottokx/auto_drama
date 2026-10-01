"""Public LLM settings and worker capabilities; never contain filesystem paths."""
from pydantic import BaseModel, ConfigDict, Field

class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=120)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    reasoning_effort: str
    ctx_size: int = Field(ge=16384, strict=True)

    def profile(self) -> dict:
        return {"provider": "local", "model_id": self.model,
                "temperature": self.temperature, "top_p": self.top_p,
                "reasoning_level": self.reasoning_effort, "context_size": self.ctx_size,
                "common_settings_version": 1}


class LLMCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=120)
    # Accepted from older workers; common settings no longer impose this cap.
    max_context_size: int | None = Field(default=None, ge=16384, strict=True)
    reasoning_efforts: list[str] = Field(min_length=1)

    def supports(self, profile: dict) -> bool:
        return (profile.get("model_id") == self.model
                and 16384 <= profile.get("context_size", 16384)
                and (isinstance(profile.get("reasoning_level", "none"), str)
                     if profile.get("common_settings_version")
                     else profile.get("reasoning_level", "none") in self.reasoning_efforts))
