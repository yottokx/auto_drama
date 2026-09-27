"""Versioned plans and evidence-backed observations for the story workflow.

Plans describe intentions. Only verified source references can enter the observed
ledger; author-only canon has its own provenance and never impersonates prose.
"""

from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints, model_validator

from .m2 import CharacterId
from .script import Contract, Identifier

StoryText = Annotated[str, StringConstraints(min_length=1, max_length=6000)]
Digest = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]
ChapterNumber = Annotated[int, Field(ge=1, le=100, strict=True)]


class LocationIdentity(Contract):
    """Stable place identity, separate from a chapter's light, time and condition."""

    id: Identifier
    name: StoryText
    structural_description: StoryText
    introduced_chapter: ChapterNumber


class SourceRef(Contract):
    storyline_id: str = Field(min_length=1, max_length=128)
    chapter_number: ChapterNumber
    scene_id: Identifier
    scene_revision: Digest
    utterance_id: Identifier | None = None
    source_start: int = Field(ge=0, strict=True)
    source_end: int = Field(ge=1, strict=True)
    text_hash: Digest

    @model_validator(mode="after")
    def nonempty_span(self):
        if self.source_end <= self.source_start:
            raise ValueError("Source evidence must name a nonempty source span.")
        return self


class SourceExcerpt(Contract):
    ref: SourceRef
    text: Annotated[str, StringConstraints(min_length=1, max_length=100_000)]
    speaker_id: CharacterId | None = None


class BlueprintChapter(Contract):
    number: ChapterNumber
    question: StoryText
    inherits: list[StoryText] = Field(default_factory=list, max_length=30)
    unique_progress: list[StoryText] = Field(min_length=1, max_length=30)
    ending_conditions: list[StoryText] = Field(min_length=1, max_length=30)
    next_consequences: list[StoryText] = Field(default_factory=list, max_length=30)
    required_character_ids: list[CharacterId] = Field(default_factory=list, max_length=100)
    location_needs: list[StoryText] = Field(default_factory=list, max_length=30)


class StoryBlueprint(Contract):
    schema_version: Literal[1] = 1
    revision: int = Field(default=1, ge=1, strict=True)
    central_question: StoryText
    ending_conditions: list[StoryText] = Field(min_length=1, max_length=30)
    immutable_conditions: list[StoryText] = Field(default_factory=list, max_length=100)
    character_changes: list[StoryText] = Field(default_factory=list, max_length=100)
    chapters: list[BlueprintChapter] = Field(min_length=1, max_length=100)
    revision_reason: StoryText | None = None

    @model_validator(mode="after")
    def sequential_chapters(self):
        if [chapter.number for chapter in self.chapters] != list(range(1, len(self.chapters) + 1)):
            raise ValueError("Blueprint chapter numbers must be consecutive from one.")
        return self


class ChapterIntent(Contract):
    schema_version: Literal[1] = 1
    chapter_number: ChapterNumber
    blueprint_revision: int = Field(default=1, ge=1, strict=True)
    predecessor_memory_hash: Digest | None = None
    question: StoryText
    entry_bridge: StoryText
    inherited_event_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    inherited_thread_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    unique_progress: list[StoryText] = Field(min_length=1, max_length=30)
    not_to_repeat: list[StoryText] = Field(default_factory=list, max_length=100)
    desired_end: list[StoryText] = Field(min_length=1, max_length=30)
    next_consequences: list[StoryText] = Field(default_factory=list, max_length=30)
    character_ids: list[CharacterId] = Field(default_factory=list, max_length=100)
    location_ids: list[Identifier] = Field(default_factory=list, max_length=100)


class SceneIntent(Contract):
    schema_version: Literal[1] = 1
    scene_id: Identifier
    chapter_number: ChapterNumber
    character_ids: list[CharacterId] = Field(min_length=1, max_length=100)
    location_id: Identifier
    story_time: StoryText
    presentation: Literal["current", "flashback", "parallel", "report"] = "current"
    viewpoint_character_id: CharacterId | None = None
    entry_bridge: StoryText
    motive: StoryText
    obstacle: StoryText
    planned_action: StoryText
    expected_change: StoryText
    inherited_event_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    forbidden_knowledge: list[StoryText] = Field(default_factory=list, max_length=100)


class StoryEvent(Contract):
    id: Identifier
    description: StoryText
    character_ids: list[CharacterId] = Field(default_factory=list, max_length=100)
    location_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    causes: list[Identifier] = Field(default_factory=list, max_length=100)
    story_time: StoryText
    presentation: Literal["current", "flashback", "parallel", "report"] = "current"
    assertion: Literal["observed", "reported", "believed"] = "observed"
    evidence: list[SourceRef] = Field(min_length=1, max_length=100)


class StateDelta(Contract):
    id: Identifier
    scope: Literal["character", "location", "world"]
    entity_id: str = Field(min_length=1, max_length=128)
    key: Identifier
    before: StoryText | None = None
    after: StoryText | None = None
    event_id: Identifier
    time_scope: Literal["current", "historical"] = "current"
    effective_time: StoryText | None = None
    evidence: list[SourceRef] = Field(min_length=1, max_length=100)

    @model_validator(mode="after")
    def changes_value(self):
        if self.before == self.after:
            raise ValueError("State deltas must change an explicit value.")
        if self.time_scope == "historical" and self.effective_time is None:
            raise ValueError("Historical state changes need an effective story time.")
        return self


class StateEntry(Contract):
    scope: Literal["character", "location", "world"]
    entity_id: str = Field(min_length=1, max_length=128)
    key: Identifier
    value: StoryText
    changed_chapter: int = Field(default=0, ge=0, le=100, strict=True)
    evidence: list[SourceRef] = Field(default_factory=list, max_length=100)


class PendingFinding(Contract):
    """A system-side uncertainty about the present, separate from character belief."""

    id: Identifier
    scope: Literal["character", "location", "world"]
    entity_id: str = Field(min_length=1, max_length=128)
    key: Identifier
    claim: StoryText
    reason: StoryText
    last_known_value: StoryText | None = None
    evidence: list[SourceRef] = Field(min_length=1, max_length=100)
    resolved_by: list[SourceRef] = Field(default_factory=list, max_length=100)


class KnowledgeUpdate(Contract):
    id: Identifier
    character_id: CharacterId
    fact_id: Identifier
    content: StoryText
    kind: Literal["knowledge", "belief", "uncertain", "retracted"] = "knowledge"
    acquired_at: StoryText
    event_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    supersedes_id: Identifier | None = None
    evidence: list[SourceRef] = Field(min_length=1, max_length=100)


class KnowledgeRecord(KnowledgeUpdate):
    disclosed_chapter: ChapterNumber


class ThreadUpdate(Contract):
    id: Identifier
    question: StoryText
    status: Literal["open", "resolved", "dropped"] = "open"
    expected_status: Literal["open", "resolved", "dropped"] | None = None
    character_ids: list[CharacterId] = Field(default_factory=list, max_length=100)
    location_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    event_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    evidence: list[SourceRef] = Field(min_length=1, max_length=100)


class OpenThread(ThreadUpdate):
    introduced_chapter: ChapterNumber
    changed_chapter: ChapterNumber


class IntroductionRecord(Contract):
    id: Identifier
    kind: Literal["character", "location", "fact", "relationship"]
    entity_ids: list[str] = Field(min_length=1, max_length=100)
    description: StoryText
    # An introduction can concern the reader or only particular characters.
    audience_character_ids: list[CharacterId] = Field(default_factory=list, max_length=100)
    reader_visible: bool = Field(default=True, strict=True)
    evidence: list[SourceRef] = Field(min_length=1, max_length=100)


class AuthorFact(Contract):
    id: Identifier
    content: StoryText
    origin: Literal["initial_canon", "author_event"]
    authority_id: str = Field(min_length=1, max_length=128)
    authority_hash: Digest
    established_at: StoryText
    known_by_character_ids: list[CharacterId] = Field(default_factory=list, max_length=100)
    disclosure_condition: StoryText
    cause_event_ids: list[Identifier] = Field(default_factory=list, max_length=100)
    # Author facts are deliberately not given fake SourceRef evidence.


class ChapterExtraction(Contract):
    schema_version: Literal[1] = 1
    chapter_number: ChapterNumber
    summary: StoryText
    events: list[StoryEvent] = Field(default_factory=list, max_length=1000)
    state_deltas: list[StateDelta] = Field(default_factory=list, max_length=1000)
    knowledge_updates: list[KnowledgeUpdate] = Field(default_factory=list, max_length=1000)
    thread_updates: list[ThreadUpdate] = Field(default_factory=list, max_length=1000)
    introductions: list[IntroductionRecord] = Field(default_factory=list, max_length=1000)
    pending_findings: list[PendingFinding] = Field(
        default_factory=list, max_length=1000, exclude_if=lambda value: not value)


class ChapterMemory(Contract):
    schema_version: Literal[1] = 1
    storyline_id: str = Field(min_length=1, max_length=128)
    chapter_number: int = Field(ge=0, le=100, strict=True)
    revision: int = Field(default=0, ge=0, strict=True)
    previous_memory_hash: Digest | None = None
    summary: StoryText
    events: list[StoryEvent] = Field(default_factory=list)
    state: list[StateEntry] = Field(default_factory=list)
    state_history: list[StateDelta] = Field(default_factory=list)
    knowledge: list[KnowledgeRecord] = Field(default_factory=list)
    threads: list[OpenThread] = Field(default_factory=list)
    thread_history: list[OpenThread] = Field(default_factory=list)
    introductions: list[IntroductionRecord] = Field(default_factory=list)
    sources: list[SourceExcerpt] = Field(default_factory=list)
    author_facts: list[AuthorFact] = Field(default_factory=list)
    pending_findings: list[PendingFinding] = Field(
        default_factory=list, exclude_if=lambda value: not value)


class ReviewIssue(Contract):
    code: Identifier
    severity: Literal["warning", "error"]
    description: StoryText
    repair_scope: Literal["blueprint", "chapter_intent", "scene", "extraction", "context", "media"]
    evidence: list[SourceRef] = Field(default_factory=list, max_length=100)
    missing_information: list[StoryText] = Field(default_factory=list, max_length=100)


class ReviewReport(Contract):
    schema_version: Literal[1] = 1
    scope: StoryText
    subject_hash: Digest
    input_hash: Digest | None = None
    verdict: Literal["pass", "fail", "insufficient_evidence"]
    checked_scene_ids: list[Identifier] = Field(default_factory=list, max_length=1000)
    checked_event_ids: list[Identifier] = Field(default_factory=list, max_length=1000)
    checked_categories: list[Identifier] = Field(default_factory=list, max_length=100)
    rationale: StoryText | None = None
    issues: list[ReviewIssue] = Field(default_factory=list, max_length=100)
    missing_information: list[StoryText] = Field(default_factory=list, max_length=100)

    @model_validator(mode="after")
    def consistent_verdict(self):
        missing = self.missing_information or any(i.missing_information for i in self.issues)
        if self.verdict == "pass" and (missing or any(i.severity == "error" for i in self.issues)):
            raise ValueError("A review with errors or missing evidence cannot pass.")
        if self.verdict == "fail" and not any(i.severity == "error" for i in self.issues):
            raise ValueError("A failed review must identify its error.")
        if self.verdict == "insufficient_evidence" and not missing:
            raise ValueError("An incomplete review must identify missing evidence.")
        return self


class ContextPacket(Contract):
    schema_version: Literal[1] = 1
    memory_hash: Digest
    chapter_number: int = Field(ge=0, le=100, strict=True)
    character_ids: list[CharacterId]
    location_ids: list[Identifier]
    events: list[StoryEvent]
    state: list[StateEntry]
    knowledge: list[KnowledgeRecord]
    threads: list[OpenThread]
    introductions: list[IntroductionRecord]
    sources: list[SourceExcerpt]
    author_facts: list[AuthorFact] = Field(default_factory=list)
    required_event_ids: list[Identifier]
    missing_information: list[StoryText]
    status: Literal["ready", "insufficient_evidence"]


class RepeatedPassage(Contract):
    previous_scene_id: Identifier
    current_scene_id: Identifier
    previous_utterance_ids: list[Identifier]
    current_utterance_ids: list[Identifier]
    character_count: int = Field(ge=1, strict=True)
    # This is a review candidate; flashbacks/refrains require semantic judgment.
