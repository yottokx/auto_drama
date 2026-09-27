"""Persistent chapter continuity, separate from approved character settings."""

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from .m2 import CharacterId
from .script import Contract, Identifier

StateText = Annotated[str, StringConstraints(min_length=1, max_length=4000)]
StateHash = Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{64}$")]


class StoryFact(Contract):
    id: Identifier
    detail: StateText


class CharacterKnowledge(Contract):
    fact_id: Identifier
    acquired_chapter: int = Field(ge=0, le=100, strict=True)
    evidence_utterance_ids: list[Identifier] = Field(max_length=30)


class CharacterStoryState(Contract):
    character_id: CharacterId
    location: StateText
    relationships: list[StateText] = Field(max_length=20)
    possessions: list[StateText] = Field(max_length=30)
    injuries: list[StateText] = Field(max_length=20)
    promises: list[StateText] = Field(max_length=30)
    knowledge: list[CharacterKnowledge] = Field(max_length=100)
    inner_emotion: StateText


class ForeshadowingState(Contract):
    # An index into the immutable StoryOutline.foreshadowing list.
    outline_index: int = Field(ge=0, le=29, strict=True)
    status: Literal["pending", "planted", "resolved"]
    changed_chapter: int = Field(ge=0, le=100, strict=True)
    evidence_utterance_ids: list[Identifier] = Field(max_length=30)


class StoryState(Contract):
    schema_version: Literal[1]
    chapter_number: int = Field(ge=0, le=100, strict=True)
    summary: StateText
    facts: list[StoryFact] = Field(max_length=100)
    characters: list[CharacterStoryState] = Field(min_length=1, max_length=100)
    foreshadowing: list[ForeshadowingState] = Field(max_length=30)


class ContinuityReview(Contract):
    passed: bool = Field(strict=True)
    issues: list[StateText] = Field(max_length=30)
    checked_character_ids: list[CharacterId] = Field(min_length=1, max_length=100)
    checked_foreshadowing_indices: list[Annotated[int, Field(ge=0, le=29, strict=True)]] = Field(max_length=30)
