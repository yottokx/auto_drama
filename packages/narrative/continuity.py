"""Deterministic state and lineage checks around semantic continuity review."""

from __future__ import annotations

import hashlib
import json

from packages.contracts.m3 import NarrativeResult, StoryOutline
from packages.contracts.m4 import StoryState


def story_state_hash(value: dict | StoryState) -> str:
    state = StoryState.model_validate(value)
    encoded = json.dumps(state.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def narrative_hash(value: dict | NarrativeResult) -> str:
    """Identify the exact predecessor script without inventing an extracted state."""
    narrative = NarrativeResult.model_validate(value)
    encoded = json.dumps(narrative.model_dump(mode="json"), ensure_ascii=False,
                         sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_script_continuity(result: NarrativeResult, previous_narrative,
                                expected_previous_artifact_id: str | None) -> None:
    if not result.storyline_id:
        raise ValueError("Script continuation requires a storyline ID.")
    if result.chapter_number == 1:
        if (result.previous_narrative_artifact_id is not None
                or result.previous_narrative_hash is not None or previous_narrative is not None
                or expected_previous_artifact_id is not None):
            raise ValueError("The first script chapter cannot have predecessor metadata.")
        return
    if not result.previous_narrative_artifact_id or not result.previous_narrative_hash:
        raise ValueError("A later script chapter requires its predecessor artifact and source hash.")
    if expected_previous_artifact_id is None:
        raise ValueError("The expected predecessor artifact is required to verify script lineage.")
    if result.previous_narrative_artifact_id != expected_previous_artifact_id:
        raise ValueError("Script predecessor artifact differs from the requested artifact.")
    if previous_narrative is None:
        raise ValueError("The preceding script is required to verify chapter lineage.")
    previous = NarrativeResult.model_validate(previous_narrative)
    if previous.workflow_policy != "script_continuation_v1":
        raise ValueError("Script continuation cannot silently change its predecessor policy.")
    if previous.chapter_number + 1 != result.chapter_number:
        raise ValueError("Script chapters must continue sequentially.")
    if previous.storyline_id != result.storyline_id:
        raise ValueError("Script continuation cannot change its storyline.")
    if narrative_hash(previous) != result.previous_narrative_hash:
        raise ValueError("Script predecessor source hash does not match the saved narrative.")


def _unique(values, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} in story state.")


def validate_story_state(value: dict | StoryState, *, chapter_number: int,
                         character_ids: set[str], outline: StoryOutline,
                         utterance_ids: set[str], previous: StoryState | None = None) -> StoryState:
    state = StoryState.model_validate(value)
    if state.chapter_number != chapter_number:
        raise ValueError("Story state chapter number does not match its source.")
    ids = [character.character_id for character in state.characters]
    _unique(ids, "characters")
    if set(ids) != character_ids:
        raise ValueError("Story state must contain every cast member exactly once.")
    facts = {fact.id: fact.detail for fact in state.facts}
    _unique([fact.id for fact in state.facts], "facts")
    old_characters = {c.character_id: c for c in previous.characters} if previous else {}
    if previous:
        if previous.chapter_number != chapter_number - 1:
            raise ValueError("Story state must continue the preceding chapter.")
        if any(facts.get(f.id) != f.detail for f in previous.facts):
            raise ValueError("Existing story facts must retain their identifiers and meaning.")
    for character in state.characters:
        _unique([k.fact_id for k in character.knowledge], "character knowledge")
        old = {k.fact_id: k for k in old_characters[character.character_id].knowledge} \
            if character.character_id in old_characters else {}
        knowledge = {k.fact_id: k for k in character.knowledge}
        if any(knowledge.get(fid) != item for fid, item in old.items()):
            raise ValueError("Previously acquired character knowledge must be preserved.")
        for item in character.knowledge:
            if item.fact_id not in facts or item.acquired_chapter > chapter_number:
                raise ValueError("Character knowledge references an unknown or future fact.")
            evidence = item.evidence_utterance_ids
            _unique(evidence, "knowledge evidence")
            if item.acquired_chapter == 0 and evidence:
                raise ValueError("Initial knowledge cannot cite future utterances.")
            if previous and item.fact_id not in old and item.acquired_chapter != chapter_number:
                raise ValueError("New character knowledge must be acquired in the current chapter.")
            if (item.acquired_chapter == chapter_number and chapter_number > 0
                    and (not evidence or not set(evidence).issubset(utterance_ids))):
                raise ValueError("New knowledge needs evidence from the adopted chapter text.")
    indices = [clue.outline_index for clue in state.foreshadowing]
    _unique(indices, "foreshadowing")
    if set(indices) != set(range(len(outline.foreshadowing))):
        raise ValueError("Story state must track every outline foreshadowing item.")
    old_clues = {c.outline_index: c for c in previous.foreshadowing} if previous else {}
    for clue in state.foreshadowing:
        planned = outline.foreshadowing[clue.outline_index]
        expected = ("resolved" if chapter_number >= planned.payoff_chapter else
                    "planted" if chapter_number >= planned.setup_chapter else "pending")
        if clue.status != expected or clue.changed_chapter > chapter_number:
            raise ValueError("Foreshadowing must follow the approved setup/payoff chapters.")
        old = old_clues.get(clue.outline_index)
        if old and old.status == clue.status and old != clue:
            raise ValueError("Unchanged foreshadowing must retain its evidence.")
        if clue.status == "pending":
            if clue.changed_chapter != 0 or clue.evidence_utterance_ids:
                raise ValueError("Pending foreshadowing cannot claim on-screen evidence.")
        else:
            changed = planned.payoff_chapter if clue.status == "resolved" else planned.setup_chapter
            if clue.changed_chapter != changed or not clue.evidence_utterance_ids:
                raise ValueError("Foreshadowing transitions need their source chapter and evidence.")
            _unique(clue.evidence_utterance_ids, "foreshadowing evidence")
            if changed == chapter_number and not set(clue.evidence_utterance_ids).issubset(utterance_ids):
                raise ValueError("Foreshadowing evidence must reference the adopted chapter text.")
    return state


def validate_continuity(result: NarrativeResult, character_ids: set[str], *,
                        previous_narrative: dict | NarrativeResult | None = None,
                        require_state: bool = False,
                        expected_previous_artifact_id: str | None = None) -> None:
    if result.workflow_policy == "script_continuation_v1":
        _validate_script_continuity(result, previous_narrative, expected_previous_artifact_id)
        return
    if result.workflow_version == 2:
        from .causal_validation import validate_causal_continuity
        validate_causal_continuity(result, character_ids, previous_narrative=previous_narrative)
        return
    has_state = result.start_state is not None or result.end_state is not None
    if not has_state and not require_state and result.chapter_number == 1:
        if (result.previous_state_hash or result.previous_narrative_artifact_id
                or result.predecessor_state_inferred or result.continuity_review):
            raise ValueError("Legacy chapter cannot contain incomplete continuity metadata.")
        return
    if result.start_state is None or result.end_state is None or result.continuity_review is None:
        raise ValueError("Chapter requires start/end story states and a continuity review.")
    if result.start_state.chapter_number != result.chapter_number - 1:
        raise ValueError("Chapter must start at the preceding chapter's ending state.")
    if result.chapter_number == 1:
        if (result.previous_narrative_artifact_id or result.previous_state_hash
                or result.predecessor_state_inferred):
            raise ValueError("The first chapter cannot have predecessor metadata.")
        validate_story_state(result.start_state, chapter_number=0, character_ids=character_ids,
                             outline=result.outline, utterance_ids=set())
    else:
        if not result.previous_narrative_artifact_id:
            raise ValueError("A later chapter must name its adopted predecessor artifact.")
        if result.previous_state_hash != story_state_hash(result.start_state):
            raise ValueError("The predecessor state hash does not match the chapter input.")
    if previous_narrative is not None:
        previous = NarrativeResult.model_validate(previous_narrative)
        if previous.chapter_number + 1 != result.chapter_number:
            raise ValueError("Narrative chapters must continue sequentially.")
        if (previous.outline != result.outline
                or previous.supporting_characters != result.supporting_characters):
            raise ValueError("Continuation cannot replace its outline or supporting cast.")
        if previous.storyline_id and previous.storyline_id != result.storyline_id:
            raise ValueError("Continuation cannot change its storyline.")
        if previous.end_state is not None:
            if previous.end_state != result.start_state or result.predecessor_state_inferred:
                raise ValueError("Continuation must use the exact adopted predecessor state.")
        else:
            if not result.predecessor_state_inferred:
                raise ValueError("Legacy predecessor state must be explicitly marked as inferred.")
            validate_story_state(result.start_state, chapter_number=previous.chapter_number,
                character_ids=character_ids, outline=result.outline,
                utterance_ids={u.id for scene in previous.scenes for u in scene.utterances})
    utterances = {u.id for scene in result.scenes for u in scene.utterances}
    validate_story_state(result.end_state, chapter_number=result.chapter_number,
        character_ids=character_ids, outline=result.outline, utterance_ids=utterances,
        previous=result.start_state)
    review = result.continuity_review
    if not review.passed or review.issues:
        raise ValueError("Chapter did not pass continuity review: " + "; ".join(review.issues))
    if (len(review.checked_character_ids) != len(character_ids)
            or set(review.checked_character_ids) != character_ids
            or sorted(review.checked_foreshadowing_indices) != list(range(len(result.outline.foreshadowing)))):
        raise ValueError("Continuity review must check every character and foreshadowing item.")
