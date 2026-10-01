"""Reproducible scene output estimates, independent of the available input space."""

import math

from pydantic import Field, model_validator

from packages.contracts.script import Contract


class SceneSize(Contract):
    length_weight: int = Field(ge=1, le=10, strict=True)
    body_characters: int = Field(ge=0, strict=True)
    dialogue_characters: int = Field(ge=0, strict=True)

    @model_validator(mode="after")
    def dialogue_within_body(self):
        if self.dialogue_characters > self.body_characters:
            raise ValueError("Scene dialogue target cannot exceed its body target.")
        return self


class SceneTokenPolicy(Contract):
    minimum_tokens: int = Field(default=2048, ge=256, strict=True)
    body_tokens_per_character: float = Field(default=1.25, gt=0, allow_inf_nan=False)
    label_tokens_per_character: float = Field(default=0.5, gt=0, allow_inf_nan=False)
    characters_per_line: int = Field(default=40, ge=1, strict=True)
    safety_factor: float = Field(default=1.2, ge=1, allow_inf_nan=False)
    fixed_tokens: int = Field(default=128, ge=0, strict=True)
    rounding_tokens: int = Field(default=256, ge=1, strict=True)


def scene_output_budget(size: SceneSize, character_ids: list[str], policy: SceneTokenPolicy,
                        output_limit: int, samples: list[dict] = (), *, extension_tokens: int = 0) -> dict:
    """Cap only at the explicit model/profile allowance, never at leftover context."""
    if type(output_limit) is not int or output_limit < 1:
        raise ValueError("Scene output allowance must be a positive integer.")
    if type(extension_tokens) is not int or extension_tokens < 0:
        raise ValueError("Scene output extension must be a nonnegative integer.")
    spoken_lines = math.ceil(size.dialogue_characters / policy.characters_per_line)
    narrator_lines = math.ceil((size.body_characters - size.dialogue_characters)
                              / policy.characters_per_line)
    label_length = max((len(cid) for cid in character_ids), default=0)
    labels = (spoken_lines * (label_length + 3) + narrator_lines * len("NARRATOR: \n"))
    estimated = (size.body_characters * policy.body_tokens_per_character
                 + labels * policy.label_tokens_per_character + policy.fixed_tokens)
    observed = max((row["completion_tokens"] / row["body_characters"] for row in samples
                    if row["body_characters"] > 0), default=0)
    estimated = max(estimated, size.body_characters * observed + policy.fixed_tokens)
    recommended = max(policy.minimum_tokens, math.ceil(
        estimated * policy.safety_factor / policy.rounding_tokens) * policy.rounding_tokens)
    result = {"max_tokens": min(recommended, output_limit), "recommended_tokens": recommended,
            "profile_output_limit": output_limit, "limited_by_profile": recommended > output_limit,
            "scene_size": size.model_dump(), "policy": policy.model_dump(),
            "samples": list(samples), "observed_tokens_per_body_character": observed,
            "estimated_label_characters": labels}
    if extension_tokens:
        result.update(baseline_max_tokens=result["max_tokens"],
                      context_extension_tokens=extension_tokens)
        for key in ("max_tokens", "recommended_tokens", "profile_output_limit"):
            result[key] += extension_tokens
    return result
