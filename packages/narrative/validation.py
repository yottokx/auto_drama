"""Do not allow structured generation to omit, duplicate, or rewrite source text."""

from __future__ import annotations

import hashlib
import re

from packages.contracts.m3 import MappedUtterance, NarrativeResult

from .continuity import story_state_hash, validate_continuity  # noqa: F401


def script_character_id(character_id: str) -> str:
    return "c-" + hashlib.sha256(character_id.encode()).hexdigest()[:24]


def approved_characters(snapshot: dict) -> list[dict]:
    return [item.get("result", item) for item in snapshot["characters"]]


def parse_scene_text(raw: str, scene_id: str, character_ids: set[str]) -> list[MappedUtterance]:
    """Only speaker prefixes and blank separators are syntax; preserve every body byte."""
    utterances = []
    offset = 0
    for line in raw.splitlines(keepends=True):
        content = line.rstrip("\r\n")
        if content.strip():
            match = re.fullmatch(r"([A-Za-z0-9][A-Za-z0-9_-]{0,127}|NARRATOR): (.+)", content)
            if match is None or not match[2].strip():
                raise ValueError("Scene text must contain only explicit 'speaker_id: text' lines.")
            speaker = None if match[1] == "NARRATOR" else match[1]
            if speaker is not None and speaker not in character_ids:
                raise ValueError(f"Unknown scene speaker: {speaker}")
            if len(match[2]) > 1000:
                raise ValueError("One speech line exceeds the TTS text limit.")
            utterances.append(MappedUtterance(
                id=f"{scene_id}-u{len(utterances) + 1}", speaker_id=speaker,
                display_text=match[2], spoken_text=match[2],
                source_start=offset + match.start(2), source_end=offset + match.end(2),
            ))
        offset += len(line)
    if not utterances:
        raise ValueError("Scene body must not be empty.")
    return utterances


def _unique(values: list[str], name: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {name} identifiers.")


def validate_narrative(value: dict | NarrativeResult, approval_snapshot: dict,
                       previous_narrative: dict | NarrativeResult | None = None, *,
                       require_state: bool = False,
                       expected_previous_artifact_id: str | None = None) -> NarrativeResult:
    result = NarrativeResult.model_validate(value)
    approved = approved_characters(approval_snapshot)
    approved_ids = {character["id"] for character in approved}
    for character in [*approved, *(c.model_dump() for c in result.supporting_characters)]:
        if "name" in character and not 1 <= len(character["name"]) <= 200:
            raise ValueError("Cast name exceeds the intermediate script's 200-character limit.")
    cast_ids = approved_ids | {character.id for character in result.supporting_characters}
    _unique([*approved_ids, *(c.id for c in result.supporting_characters)], "cast")
    if "NARRATOR" in cast_ids:
        raise ValueError("NARRATOR is reserved for narrative source syntax.")
    world = approval_snapshot["world"]
    world = world.get("result", world)
    count = world["chapterCount"]
    if result.chapter_number > count:
        raise ValueError("Chapter number exceeds the approved chapter count.")
    if [c.number for c in result.outline.chapters] != list(range(1, count + 1)):
        raise ValueError("Outline must respect the approved chapter count and order.")
    arcs = [arc.character_id for arc in result.outline.character_arcs]
    _unique(arcs, "character arc")
    if not approved_ids.issubset(arcs) or not set(arcs).issubset(cast_ids):
        raise ValueError("Outline character arcs must include every approved main character.")
    for clue in result.outline.foreshadowing:
        if not 1 <= clue.setup_chapter <= clue.payoff_chapter <= count:
            raise ValueError("Foreshadowing references an invalid chapter.")
    _unique([location.id for location in result.locations], "location")
    _unique([scene.id for scene in result.scenes], "scene")
    all_utterances, all_directions = [], []
    used_locations = set()
    for scene in result.scenes:
        if scene.id != scene.plan.id or len(scene.id) > 50:
            raise ValueError("Scene and plan IDs must match and allow stable utterance IDs.")
        if scene.plan.location_id not in {location.id for location in result.locations}:
            raise ValueError("Scene references an unknown location.")
        used_locations.add(scene.plan.location_id)
        _unique(scene.plan.character_ids, "scene cast")
        if not set(scene.plan.character_ids).issubset(cast_ids):
            raise ValueError("Scene references an unknown character.")
        parsed = parse_scene_text(scene.raw_text, scene.id, set(scene.plan.character_ids))
        if len(parsed) != len(scene.utterances):
            raise ValueError("Source mapping omitted or duplicated an utterance.")
        fields = ("id", "speaker_id", "display_text", "spoken_text", "source_start", "source_end")
        for expected, actual in zip(parsed, scene.utterances, strict=True):
            if any(getattr(expected, field) != getattr(actual, field) for field in fields):
                raise ValueError("Source mapping changed text, speaker, order, or source span.")
        utterances = {item.id: item for item in scene.utterances}
        if not any(item.speaker_id for item in scene.utterances):
            raise ValueError("Scene requires dramatized dialogue.")
        _unique([event.id for event in scene.plan.required_events], "required event")
        _unique([event.event_id for event in scene.review.events], "review event")
        if result.workflow_policy == "script_continuation_v1":
            if (scene.review.policy != "not_evaluated" or scene.review.passed
                    or scene.review.events or scene.review.issues):
                raise ValueError("Script continuation must explicitly leave content unevaluated.")
            continue_review = False
        elif result.workflow_policy == "chapter_editor_v1":
            if (scene.review.policy != "deferred_to_chapter" or scene.review.passed
                    or scene.review.events or scene.review.issues):
                raise ValueError("Editor policy must explicitly defer scene review to the chapter.")
            continue_review = False
        else:
            if scene.review.policy != "scene" or not scene.review.passed or scene.review.issues:
                raise ValueError("Scene did not pass content review: " + "; ".join(scene.review.issues))
            continue_review = True
        if continue_review and {e.id for e in scene.plan.required_events} != {e.event_id for e in scene.review.events}:
            raise ValueError("Review must cover every required event exactly once.")
        for event in scene.review.events if continue_review else []:
            evidence = event.evidence_utterance_ids
            _unique(evidence, "review evidence")
            if not event.dramatized or not set(evidence).issubset(utterances) or len(evidence) < 2:
                raise ValueError("Important event lacks concrete on-screen evidence.")
            if not any(utterances[uid].speaker_id for uid in evidence):
                raise ValueError("Important event evidence must include dialogue.")
            if not any(utterances[uid].speaker_id is None for uid in evidence):
                raise ValueError("Important event evidence must include a visible action/reaction.")
        visible = set()
        for direction in scene.directions:
            if direction.utterance_id not in utterances:
                raise ValueError("Direction references an unknown utterance.")
            if direction.kind in {"enter", "exit", "position", "focus"}:
                if direction.character_id not in scene.plan.character_ids:
                    raise ValueError("Direction references an absent character.")
            elif direction.character_id is not None or direction.position is not None:
                raise ValueError("A pause/blackout must not reference a character or position.")
            if direction.kind in {"enter", "position"} and direction.position is None:
                raise ValueError("Enter/position requires a screen position.")
            if direction.kind in {"pause", "blackout"} and direction.duration_ms < 1:
                raise ValueError("Pause/blackout requires a positive duration.")
            if direction.kind == "enter":
                visible.add(direction.character_id)
        if not set(scene.plan.character_ids).issubset(visible):
            raise ValueError("Each scene character requires an entrance direction.")
        all_utterances.extend(utterances)
        all_directions.extend(d.id for d in scene.directions)
    _unique(all_utterances, "utterance")
    _unique(all_directions, "direction")
    if used_locations != {location.id for location in result.locations}:
        raise ValueError("Only locations required by this chapter belong in its asset plan.")
    validate_continuity(result, cast_ids, previous_narrative=previous_narrative,
                        require_state=require_state,
                        expected_previous_artifact_id=expected_previous_artifact_id)
    return result
