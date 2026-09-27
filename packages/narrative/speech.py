"""Separate classified speech annotations without letting an LLM retype dialogue.

Parentheses alone never establish that text is a stage direction. Callers classify
the candidates, then explicitly select narration or delivery metadata. Unmatched
or mismatched enclosing parentheses remain untouched and are not candidates;
callers can review such source separately instead of silently trimming it.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Literal

from .validation import parse_scene_text


@dataclass(frozen=True)
class ParentheticalCandidate:
    id: str
    utterance_id: str
    speaker_id: str
    source_start: int
    source_end: int
    text: str


@dataclass(frozen=True)
class SourceSegment:
    original_utterance_id: str
    utterance_id: str | None
    original_speaker_id: str | None
    speaker_id: str | None
    kind: Literal["dialogue", "narration", "delivery"]
    source_start: int
    source_end: int
    text: str
    display_text: str
    normalized_source_start: int | None
    normalized_source_end: int | None
    candidate_id: str | None = None


@dataclass(frozen=True)
class SpeechSeparation:
    raw_text: str
    utterance_map: dict[str, list[str]]
    segments: list[SourceSegment]


def parenthetical_candidates(
    raw: str, scene_id: str, character_ids: set[str],
) -> list[ParentheticalCandidate]:
    """Find balanced outer ASCII/fullwidth pairs in character lines, including literals."""
    candidates = []
    opening = {"(": ")", "（": "）"}
    for utterance in parse_scene_text(raw, scene_id, character_ids):
        if utterance.speaker_id is None:
            continue
        stack, valid, start, count = [], True, 0, 0
        for index, char in enumerate(utterance.display_text):
            if char in opening:
                if not stack:
                    start, valid = index, True
                stack.append(opening[char])
            elif char in opening.values() and stack:
                valid = (stack.pop() == char) and valid
                if not stack and valid:
                    count += 1
                    source_start = utterance.source_start + start
                    source_end = utterance.source_start + index + 1
                    candidates.append(ParentheticalCandidate(
                        id=f"{utterance.id}-p{count}", utterance_id=utterance.id,
                        speaker_id=utterance.speaker_id, source_start=source_start,
                        source_end=source_end, text=raw[source_start:source_end],
                    ))
    return candidates


def _selection(values: Iterable[str], known: set[str]) -> set[str]:
    if isinstance(values, str):
        raise TypeError("Candidate selections must be an iterable of candidate IDs.")
    values = list(values)
    if any(not isinstance(value, str) for value in values):
        raise ValueError("Candidate IDs must be strings.")
    if len(values) != len(set(values)):
        raise ValueError("Duplicate candidate selection.")
    if not set(values).issubset(known):
        raise ValueError("Unknown candidate selection.")
    return set(values)


def _narration(value: str) -> str:
    if (not isinstance(value, str) or not value.strip() or len(value) > 1000
            or any(unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} for char in value)
            or re.match(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}:\s", value.lstrip())):
        raise ValueError("Direction narration must be nonempty, single-line text without a speaker prefix.")
    return value


def _actor_name(value: str) -> str:
    return "".join(
        " " if unicodedata.category(char) in {"Cc", "Cf", "Zl", "Zp"} else char
        for char in value
    ).strip()


def separate_stage_directions(
    raw: str,
    scene_id: str,
    character_ids: set[str],
    direction_candidate_ids: Iterable[str],
    character_names: Mapping[str, str],
    *,
    direction_narrations: Mapping[str, str] | None = None,
    delivery_candidate_ids: Iterable[str] = (),
) -> SpeechSeparation:
    """Normalize selected annotations, retaining exact original spans for auditing.

    Supply natural narration for production use. Omitted narration values fall back
    to an actor-prefixed, verbatim parenthesis fragment for diagnostic use. Delivery
    candidates disappear from visible source; surrounding speech stays in one line.
    Multiple source segments may therefore share a normalized utterance ID. Their
    normalized spans cover their individual rendered fragments, not the whole line.
    Blank separators and untouched lines retain their exact source representation.
    """
    original = parse_scene_text(raw, scene_id, character_ids)
    candidates = parenthetical_candidates(raw, scene_id, character_ids)
    known = {candidate.id for candidate in candidates}
    directions = _selection(direction_candidate_ids, known)
    deliveries = _selection(delivery_candidate_ids, known)
    if directions & deliveries:
        raise ValueError("A candidate cannot be both narration and delivery.")
    narration_items = list((direction_narrations or {}).items())
    narration_keys = [key for key, _ in narration_items]
    if len(narration_keys) != len(set(narration_keys)):
        raise ValueError("Duplicate direction narration mapping.")
    if not set(narration_keys).issubset(directions):
        raise ValueError("Direction narration references an unselected candidate.")
    narrations = {key: _narration(value) for key, value in narration_items}
    selected = directions | deliveries
    selected_by_utterance: dict[str, list[ParentheticalCandidate]] = {}
    for candidate in candidates:
        if candidate.id in selected:
            selected_by_utterance.setdefault(candidate.utterance_id, []).append(candidate)

    output, tokens, unit_records, old_units = [], [], [], {}
    cursor = 0
    for utterance in original:
        line_tokens = []

        def token(start, end, kind, candidate=None, display=None,
                  *, _line_tokens=line_tokens, _utterance=utterance):
            if start == end:
                return
            _line_tokens.append({
                "original": _utterance, "start": start, "end": end, "kind": kind,
                "candidate": candidate, "display": raw[start:end] if display is None else display,
                "unit": None, "within_start": None, "within_end": None,
            })

        position = utterance.source_start
        ordinary_kind = "dialogue" if utterance.speaker_id else "narration"
        for candidate in selected_by_utterance.get(utterance.id, []):
            token(position, candidate.source_start, ordinary_kind)
            if candidate.id in deliveries:
                token(candidate.source_start, candidate.source_end, "delivery", candidate.id, "")
            else:
                name = _actor_name(character_names.get(candidate.speaker_id, candidate.speaker_id))
                name = name or candidate.speaker_id
                display = narrations.get(candidate.id, name + "：" + candidate.text)
                token(candidate.source_start, candidate.source_end, "narration", candidate.id, display)
            position = candidate.source_end
        token(position, utterance.source_end, ordinary_kind)
        tokens.extend(line_tokens)

        units = []
        for part in line_tokens:
            if part["kind"] == "delivery":
                continue
            if (part["kind"] == "dialogue" and units
                    and units[-1]["kind"] == "dialogue"):
                units[-1]["tokens"].append(part)
            else:
                units.append({"kind": part["kind"], "tokens": [part]})
        if not any("".join(part["display"] for part in unit["tokens"]).strip() for unit in units):
            raise ValueError("Delivery selection cannot remove an entire speech line; use narration.")
        # Whitespace between annotations belongs to an adjacent visible unit. It
        # must not become an empty speech line or vanish from the original mapping.
        for index in range(len(units) - 1, -1, -1):
            unit = units[index]
            if "".join(part["display"] for part in unit["tokens"]).strip():
                continue
            if index:
                units[index - 1]["tokens"].extend(unit["tokens"])
            else:
                units[index + 1]["tokens"][:0] = unit["tokens"]
            units.pop(index)

        prefix_start = utterance.source_start - len(utterance.speaker_id or "NARRATOR") - 2
        output.append(raw[cursor:prefix_start])
        ending = re.match(r"\r\n|\r|\n", raw[utterance.source_end:])
        newline = ending[0] if ending else "\n"
        normalized_lines = []
        old_units[utterance.id] = []
        for unit in units:
            unit_index = len(unit_records)
            unit_records.append(unit)
            old_units[utterance.id].append(unit_index)
            body, offset = [], 0
            for part in unit["tokens"]:
                part["kind"] = unit["kind"]
                part["unit"] = unit_index
                part["within_start"] = offset
                body.append(part["display"])
                offset += len(part["display"])
                part["within_end"] = offset
            speaker = utterance.speaker_id if unit["kind"] == "dialogue" else "NARRATOR"
            normalized_lines.append(speaker + ": " + "".join(body))
        output.append(newline.join(normalized_lines))
        cursor = utterance.source_end
    output.append(raw[cursor:])
    normalized_raw = "".join(output)
    normalized = parse_scene_text(normalized_raw, scene_id, character_ids)
    if len(normalized) != len(unit_records):
        raise ValueError("Speech separation changed source line structure unexpectedly.")
    segments = []
    for part in tokens:
        source = part["original"]
        target = normalized[part["unit"]] if part["unit"] is not None else None
        segments.append(SourceSegment(
            original_utterance_id=source.id, utterance_id=target.id if target else None,
            original_speaker_id=source.speaker_id, speaker_id=target.speaker_id if target else None,
            kind=part["kind"], source_start=part["start"], source_end=part["end"],
            text=raw[part["start"]:part["end"]], display_text=part["display"],
            normalized_source_start=(target.source_start + part["within_start"]) if target else None,
            normalized_source_end=(target.source_start + part["within_end"]) if target else None,
            candidate_id=part["candidate"],
        ))
    return SpeechSeparation(
        raw_text=normalized_raw,
        utterance_map={key: [normalized[index].id for index in indices]
                       for key, indices in old_units.items()},
        segments=segments,
    )
