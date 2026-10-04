"""M3 narrative output. Text is mapped by exact source spans, never retyped by an LLM."""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .m2 import CharacterId, CharacterResult
from .m4 import ContinuityReview, StateHash, StoryState
from .script import Contract, Identifier, PortraitBounds
from .story_workflow import (
    ChapterExtraction,
    ChapterIntent,
    ChapterMemory,
    LocationIdentity,
    ReviewReport,
    SceneIntent,
    StoryBlueprint,
)

Text = Annotated[str, StringConstraints(min_length=1, max_length=30_000)]
Emotion = Literal["neutral", "happy", "sad", "angry", "surprised", "afraid", "calm"]
M3_KINDS = ("m3_plan", "m3_narrative", "m3_background", "m3_image", "m3_voice", "m3_voice_clone",
            "m3_music_plan", "m3_music")
EMOTION_TAGS = {
    "neutral": "", "happy": "😊", "sad": "😢", "angry": "😠",
    "surprised": "😮", "afraid": "😟", "calm": "",
}


class PortraitSetting(Contract):
    character_id: CharacterId
    framing: Literal["auto", "upper_body", "full_body"]
    height_cm: float | None = Field(default=None, ge=1, le=10000, allow_inf_nan=False, strict=True)
    body_bounds: PortraitBounds | None = None
    image_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)

    @model_validator(mode="after")
    def height_requires_humanoid(self):
        if self.height_cm is not None and self.framing != "upper_body":
            raise ValueError("height_cm applies to humanoid framing only")
        if self.body_bounds is not None and self.image_artifact_id is None:
            raise ValueError("body_bounds requires the source image_artifact_id")
        return self


class PortraitSettingsUpdate(Contract):
    expected_build_id: str = Field(min_length=1, max_length=128)
    characters: list[PortraitSetting] = Field(max_length=10)

    @model_validator(mode="after")
    def unique_characters(self):
        ids = [value.character_id for value in self.characters]
        if len(ids) != len(set(ids)):
            raise ValueError("portrait settings must name each character at most once")
        return self


class PublishedPortrait(PortraitSetting):
    name: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    image_artifact_id: str = Field(min_length=1, max_length=128)
    image_url: str


class ProductionPortraits(Contract):
    project_id: str
    production_id: str | None
    build_id: str | None
    characters: list[PublishedPortrait] = Field(max_length=10)


class OutlineChapter(Contract):
    number: int = Field(ge=1, le=100, strict=True)
    title: Text
    role: Text
    summary: Text


class CharacterArc(Contract):
    character_id: CharacterId
    change: Text


class Foreshadowing(Contract):
    setup_chapter: int = Field(ge=1, le=100, strict=True)
    payoff_chapter: int = Field(ge=1, le=100, strict=True)
    detail: Text


class StoryOutline(Contract):
    ending: Text
    character_arcs: list[CharacterArc] = Field(min_length=1, max_length=10)
    chapters: list[OutlineChapter] = Field(min_length=1, max_length=100)
    foreshadowing: list[Foreshadowing] = Field(max_length=30)


class PlotCharacter(Contract):
    id: CharacterId
    name: Annotated[str, StringConstraints(min_length=1, max_length=200)]


class PlotReview(Contract):
    outline: StoryOutline
    characters: list[PlotCharacter] = Field(max_length=10)


class ProductionPlot(Contract):
    project_id: str
    production_id: str | None
    narrative_artifact_id: str | None
    plot: PlotReview | None


class Location(Contract):
    id: Identifier
    name: Text
    description: Text
    time_of_day: Text
    atmosphere: Text
    image_prompt: Text


class RequiredEvent(Contract):
    id: Identifier
    description: Text


class ScenePlan(Contract):
    id: Identifier
    location_id: Identifier
    character_ids: list[CharacterId] = Field(min_length=1, max_length=3)
    objectives: Text
    start_state: Text
    required_events: list[RequiredEvent] = Field(min_length=1, max_length=8)
    end_state: Text
    atmosphere: Text


class MappedUtterance(Contract):
    id: Identifier
    speaker_id: CharacterId | None
    display_text: Text
    spoken_text: Text
    source_start: int = Field(ge=0, strict=True)
    source_end: int = Field(ge=1, strict=True)
    inner_emotion: Text = "未指定"
    voice_emotion: Emotion = "neutral"
    delivery: Annotated[str, StringConstraints(min_length=1, max_length=200)] | None = None


class NarrativeDirection(Contract):
    id: Identifier
    utterance_id: Identifier
    kind: Literal["enter", "exit", "position", "focus", "pause", "blackout"]
    timing: Literal["before", "start", "after"]
    character_id: CharacterId | None
    position: Literal["left", "center", "right"] | None
    duration_ms: int = Field(ge=0, le=10_000, strict=True)


class EventReview(Contract):
    event_id: Identifier
    dramatized: bool = Field(strict=True)
    evidence_utterance_ids: list[Identifier] = Field(min_length=1, max_length=100)
    reason: Text


class SceneReview(Contract):
    policy: Literal["scene", "deferred_to_chapter", "not_evaluated"] = Field(
        default="scene", exclude_if=lambda value: value == "scene")
    passed: bool = Field(strict=True)
    issues: list[Text] = Field(max_length=20)
    events: list[EventReview] = Field(max_length=8)

    @model_validator(mode="after")
    def review_policy_contract(self):
        if self.policy == "scene" and not self.events:
            raise ValueError("A scene review must check its planned events.")
        if self.policy in {"deferred_to_chapter", "not_evaluated"} and (
                self.passed or self.issues or self.events):
            raise ValueError("A deferred or unevaluated review cannot claim a scene verdict.")
        return self


class NarrativeScene(Contract):
    id: Identifier
    plan: ScenePlan
    raw_text: Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
    utterances: list[MappedUtterance] = Field(min_length=1, max_length=1000)
    directions: list[NarrativeDirection] = Field(max_length=3000)
    review: SceneReview


class NarrativeResult(Contract):
    schema_version: Literal[1]
    chapter_number: int = Field(ge=1, le=100, strict=True)
    title: Annotated[str, StringConstraints(min_length=1, max_length=200)]
    outline: StoryOutline
    supporting_characters: list[CharacterResult] = Field(max_length=97)
    locations: list[Location] = Field(min_length=1, max_length=8)
    scenes: list[NarrativeScene] = Field(min_length=1, max_length=8)
    # Defaults keep completed M3 chapters readable without altering their bytes.
    storyline_id: str | None = Field(default=None, min_length=1, max_length=128)
    previous_narrative_artifact_id: str | None = Field(default=None, min_length=1, max_length=128)
    previous_narrative_hash: StateHash | None = Field(
        default=None, exclude_if=lambda value: value is None)
    previous_state_hash: StateHash | None = None
    predecessor_state_inferred: bool = Field(default=False, strict=True)
    start_state: StoryState | None = None
    end_state: StoryState | None = None
    continuity_review: ContinuityReview | None = None
    # Causal workflow is opt-in; legacy chapters retain their original checks.
    workflow_version: Literal[1, 2] = 1
    workflow_policy: Literal["scene_gate_v1", "chapter_editor_v1", "script_continuation_v1"] = Field(
        default="scene_gate_v1", exclude_if=lambda value: value == "scene_gate_v1")
    chapter_review_mode: Literal["full", "partitioned"] = Field(
        default="full", exclude_if=lambda value: value == "full")
    location_registry: list[LocationIdentity] = Field(default_factory=list, max_length=800)
    blueprint: StoryBlueprint | None = None
    chapter_intent: ChapterIntent | None = None
    scene_intents: list[SceneIntent] = Field(default_factory=list, max_length=100)
    start_memory: ChapterMemory | None = None
    story_memory: ChapterMemory | None = None
    scene_extractions: list[ChapterExtraction] = Field(default_factory=list, max_length=100)
    workflow_reviews: list[ReviewReport] = Field(default_factory=list, max_length=1000)

    @model_validator(mode="after")
    def workflow_contract(self):
        if self.workflow_policy == "script_continuation_v1":
            if self.workflow_version != 2 or self.chapter_review_mode != "full":
                raise ValueError("Script continuation requires workflow version 2 and full chapter mode.")
            if (self.start_state is not None or self.end_state is not None
                    or self.continuity_review is not None or self.previous_state_hash is not None
                    or self.predecessor_state_inferred or self.location_registry
                    or self.blueprint or self.chapter_intent or self.scene_intents
                    or self.start_memory or self.story_memory or self.scene_extractions
                    or self.workflow_reviews):
                raise ValueError("Unevaluated script continuation cannot claim state or semantic review artifacts.")
            if any(scene.review.policy != "not_evaluated" for scene in self.scenes):
                raise ValueError("Script continuation scenes must declare content as not evaluated.")
            return self
        if self.previous_narrative_hash is not None:
            raise ValueError("Narrative source hashes require the script continuation policy.")
        if self.workflow_version == 1:
            if self.workflow_policy != "scene_gate_v1" or self.chapter_review_mode != "full":
                raise ValueError("Legacy narratives cannot use a causal editor policy.")
            if len(self.supporting_characters) > 3:
                raise ValueError("Legacy narrative supports up to three supporting characters.")
            if (self.location_registry or self.blueprint or self.chapter_intent or self.scene_intents
                    or self.start_memory or self.story_memory or self.scene_extractions
                    or self.workflow_reviews):
                raise ValueError("Causal artifacts require workflow_version=2.")
        elif not (self.location_registry and self.blueprint and self.chapter_intent and self.scene_intents
                  and self.start_memory and self.story_memory and self.scene_extractions
                  and self.workflow_reviews):
            raise ValueError("Causal narrative requires plans, memory and evidence reviews.")
        if self.workflow_policy != "chapter_editor_v1" and self.chapter_review_mode != "full":
            raise ValueError("Only the chapter editor supports partitioned source review.")
        return self


class GenerationEnvelope(Contract):
    schema_version: Literal[1]
    kind: Literal["m3_plan", "m3_narrative", "m3_background", "m3_image", "m3_voice", "m3_voice_clone",
                  "m3_music_plan", "m3_music"]
    result: dict
    provenance: dict
    trace: list = Field(max_length=10_000)
