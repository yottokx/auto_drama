"""Public LLM settings and worker capabilities; never contain filesystem paths."""
from pydantic import BaseModel, ConfigDict, Field, model_validator

# The contexts a work can be made with. Every limit on a work's size (chapters, characters,
# alternate looks) is set per tier (docs/long-form-plan.md). Reasoning spends the answer slot,
# which 16k has no room for.
CONTEXT_TIERS = (16384, 32768, 65536)


def context_tier(size: int) -> int:
    """The largest tier a context size reaches; values in between use the tier below."""
    return max([tier for tier in CONTEXT_TIERS if tier <= size] or [CONTEXT_TIERS[0]])


# External API models are named provider/model; a local GGUF file name cannot contain '/'.
API_PROVIDERS = ("openai", "anthropic", "openrouter")


def model_provider(model_id: str) -> str:
    provider, separator, _ = model_id.partition("/")
    return provider if separator and provider in API_PROVIDERS else "local"


class LLMSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    model: str = Field(min_length=1, max_length=120)
    temperature: float = Field(ge=0, le=2)
    top_p: float = Field(gt=0, le=1)
    reasoning_effort: str
    ctx_size: int = Field(strict=True)

    @model_validator(mode="after")
    def one_of_the_tiers(self):
        if self.ctx_size not in CONTEXT_TIERS:
            raise ValueError("ctx_size must be one of " + ", ".join(map(str, CONTEXT_TIERS)))
        if self.ctx_size == CONTEXT_TIERS[0] and self.reasoning_effort != "none":
            raise ValueError("reasoning needs a context of 32768 or more")
        return self

    def profile(self) -> dict:
        return {"provider": model_provider(self.model), "model_id": self.model,
                "temperature": self.temperature, "top_p": self.top_p,
                "reasoning_level": self.reasoning_effort, "context_size": self.ctx_size,
                "common_settings_version": 1}


class LLMCapability(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = Field(min_length=1, max_length=120)
    # Accepted from older workers; common settings no longer impose this cap.
    max_context_size: int | None = Field(default=None, ge=16384, strict=True)
    reasoning_efforts: list[str] = Field(min_length=1)
    # The reasoning efforts with which temperature and top_p take effect; None means all.
    # Settings hide both where a model would refuse them.
    sampling_efforts: list[str] | None = None

    def supports(self, profile: dict) -> bool:
        return (profile.get("model_id") == self.model
                and 16384 <= profile.get("context_size", 16384)
                and (isinstance(profile.get("reasoning_level", "none"), str)
                     if profile.get("common_settings_version")
                     else profile.get("reasoning_level", "none") in self.reasoning_efforts))
