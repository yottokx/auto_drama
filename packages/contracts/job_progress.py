"""Leased, factual generation progress; no timing-derived percentages."""
from __future__ import annotations

from typing import Annotated, Literal, Self

from pydantic import Field, StringConstraints, model_validator

from .script import Contract

StepId = Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_-]{1,160}$")]
Stage = Literal["cast_plan", "plot_plan", "chapter_scene_plan",
                "supporting_characters", "relationships", "story_core", "plot",
                "chapter_plan", "scene_plan", "script", "speech_extraction", "staging",
                "validation", "memory", "revision", "event_cg_budget", "event_cg_plan", "event_cg_generate"]
Count = Annotated[int, Field(strict=True, ge=0, le=10000)]


class ProgressStep(Contract):
    id: StepId
    stage: Stage
    status: Literal["running", "completed", "failed"]
    chapter_number: Annotated[int, Field(strict=True, ge=1, le=100)] | None = None
    scene_number: Annotated[int, Field(strict=True, ge=1, le=100)] | None = None
    completed: Count | None = None
    total: Count | None = None

    @model_validator(mode="after")
    def valid_counts(self) -> Self:
        if self.completed is not None and self.total is not None and self.completed > self.total:
            raise ValueError("Completed progress cannot exceed its total.")
        return self


class ChapterPlanProgress(Contract):
    chapter_number: Annotated[int, Field(strict=True, ge=1, le=100)]
    scene_count: Annotated[int, Field(strict=True, ge=1, le=100)]
    supporting_character_count: Annotated[int, Field(strict=True, ge=0, le=97)]


class JobProgress(Contract):
    schema_version: Literal[1] = 1
    sequence: Annotated[int, Field(strict=True, ge=1, le=1000000)]
    phase: Literal["planning", "chapter"]
    current_step: StepId | None = None
    steps: list[ProgressStep] = Field(min_length=1, max_length=512)
    chapter_plan: ChapterPlanProgress | None = None

    @model_validator(mode="after")
    def valid_steps(self) -> Self:
        identifiers = [step.id for step in self.steps]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Progress step IDs must be unique.")
        if self.current_step is not None and self.current_step not in identifiers:
            raise ValueError("Current progress step must be present.")
        return self
