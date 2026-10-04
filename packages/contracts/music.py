"""Backend-neutral scene music, independent of speech and model-specific knobs."""
from typing import Literal

from pydantic import Field, model_validator

from .script import Contract, Identifier, SceneTransitionSpec


class SceneMusicPrompt(Contract):
    scene_id: Identifier
    prompt: str = Field(default="", max_length=10000, exclude_if=lambda value: value == "")
    interpretation: str = Field(default="", max_length=10000)
    action: Literal["play", "continue", "stop"] = Field(default="play", exclude_if=lambda value: value == "play")
    transition: SceneTransitionSpec | None = Field(default=None, exclude_if=lambda value: value is None)
    reason: str = Field(default="", max_length=2000, exclude_if=lambda value: value == "")

    @model_validator(mode="after")
    def prompt_for_new_track(self):
        if self.action == "play" and not self.prompt.strip():
            raise ValueError("A new music group requires its instrumental prompt.")
        if self.action != "play" and self.prompt:
            raise ValueError("Continue and stop decisions cannot request a new track.")
        return self


class MusicPlan(Contract):
    planning_version: Literal[1, 2] = Field(default=1, exclude_if=lambda value: value == 1)
    scenes: list[SceneMusicPrompt] = Field(min_length=1, max_length=8)

    @model_validator(mode="after")
    def unique_scenes(self):
        if len({row.scene_id for row in self.scenes}) != len(self.scenes):
            raise ValueError("Music prompts must name each scene once.")
        active = False
        for row in self.scenes:
            if row.action == "continue" and not active:
                raise ValueError("Music cannot continue before play or after stop.")
            if row.action != "continue":
                active = row.action == "play"
            if self.planning_version == 2 and (row.transition is None or not row.reason.strip()):
                raise ValueError("Chapter music decisions require a transition and a reason.")
        return self


class MusicResult(Contract):
    scene_id: Identifier
    prompt: str = Field(min_length=1, max_length=10000)
    loop_start_seconds: float = Field(ge=0, allow_inf_nan=False)
    loop_end_seconds: float = Field(gt=0, le=380, allow_inf_nan=False)
    duration_seconds: float = Field(gt=0, le=381, allow_inf_nan=False)
    source_duration_seconds: float = Field(gt=0, le=381, allow_inf_nan=False)
    sample_rate: int = Field(ge=8000, le=192000, strict=True)
    quality: dict = Field(default_factory=dict)

    @model_validator(mode="after")
    def usable_loop(self):
        if not self.loop_start_seconds < self.loop_end_seconds <= self.duration_seconds + 0.002:
            raise ValueError("Music loop must be inside the playback file.")
        if self.loop_end_seconds - self.loop_start_seconds < 5:
            raise ValueError("Music loop is too short.")
        if self.quality.get("needs_review"):
            raise ValueError("Music requiring review cannot be adopted automatically.")
        return self


class SceneMusicSetting(Contract):
    production_id: str = Field(min_length=1, max_length=128)
    scene_id: Identifier
    candidate_id: str | None = Field(default=None, min_length=1, max_length=128)
    action: Literal["play", "continue", "stop"] = "play"
    volume: float = Field(default=0.35, ge=0, le=1, allow_inf_nan=False)
    transition: SceneTransitionSpec | None = Field(default=None, exclude_if=lambda value: value is None)
    reason: str = Field(default="", max_length=2000, exclude_if=lambda value: value == "")

    @model_validator(mode="after")
    def selected_track(self):
        if self.action == "play" and self.candidate_id is None:
            raise ValueError("再生するBGM候補を選択してください。")
        if self.action != "play" and self.candidate_id is not None:
            raise ValueError("BGMなし・継続では候補を指定できません。")
        return self


class MusicReplanRequest(Contract):
    expected_revision: int = Field(ge=1, strict=True)
    production_id: str = Field(min_length=1, max_length=128)
