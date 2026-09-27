"""Remove only long, exact script repetitions at a completed chapter boundary."""

from __future__ import annotations

import re
from dataclasses import dataclass

from .causal_runtime import digest

MIN_OVERLAP_UTTERANCES = 6
MIN_OVERLAP_BODY_CHARACTERS = 300
_PHYSICAL_LINE = re.compile(r"[^\n]*(?:\n|$)")
_UTTERANCE = re.compile(r"(?P<speaker>[^\s:]+): (?P<body>[^\r\n]+)")


@dataclass(frozen=True)
class _Line:
    text: str
    start: int
    speaker: str
    body: str


def _script_lines(text: str) -> tuple[list[_Line], bool]:
    """Keep exact utterances and original offsets; ignore only blank lines/EOLs."""
    lines = []
    valid = True
    for match in _PHYSICAL_LINE.finditer(text):
        line = match.group()
        if line.endswith("\n"):
            line = line[:-1].removesuffix("\r")
        if "\r" in line:
            valid = False
            continue
        if not line.strip():
            continue
        utterance = _UTTERANCE.fullmatch(line)
        if utterance is None or not utterance["body"].strip():
            valid = False
            continue
        lines.append(_Line(line, match.start(), utterance["speaker"], utterance["body"]))
    return lines, valid


def _counts(lines: list[_Line]) -> dict[str, int]:
    return {
        "utterances": len(lines),
        "body_characters": sum(len(line.body) for line in lines),
        "dialogue_characters": sum(len(line.body) for line in lines if line.speaker != "NARRATOR"),
    }


def trim_chapter_overlap(previous_text: str, generated_text: str) -> tuple[str, dict]:
    """Trim an exact previous suffix/current prefix, preserving everything else.

    A match needs six utterances and 300 body characters. Speaker, punctuation,
    text and spacing within each utterance must match exactly. Only empty lines
    and LF/CRLF separators may differ. Malformed script is returned unchanged.
    Removed offsets refer to the original, unnormalized strings. A full copy
    returns an empty result so the caller can save the evidence and stop.
    """
    previous, previous_valid = _script_lines(previous_text)
    generated, generated_valid = _script_lines(generated_text)
    count = 0
    reason = "no_boundary_overlap"
    if not previous_valid or not generated_valid:
        reason = "invalid_script"
    elif not previous or not generated:
        reason = "empty_input"
    else:
        previous_lines = [line.text for line in previous]
        generated_lines = [line.text for line in generated]
        for candidate in range(min(len(previous), len(generated)), 0, -1):
            if previous_lines[-candidate:] == generated_lines[:candidate]:
                if (candidate >= MIN_OVERLAP_UTTERANCES
                        and _counts(generated[:candidate])["body_characters"]
                        >= MIN_OVERLAP_BODY_CHARACTERS):
                    count = candidate
                    reason = "exact_boundary_overlap"
                else:
                    reason = "below_threshold"
                break

    end = 0
    if count:
        end = generated[count].start if count < len(generated) else len(generated_text)
    result = generated_text[end:]
    metadata = {
        "policy": "chapter_boundary_exact_overlap_v1",
        "changed": bool(count),
        "reason": reason,
        "original_sha256": digest(generated_text),
        "previous_sha256": digest(previous_text),
        "result_sha256": digest(result),
        "range_unit": "characters_zero_based_end_exclusive",
        "removed_range": [0, end] if count else None,
        "previous_range": [previous[-count].start, len(previous_text)] if count else None,
        "thresholds": {
            "min_utterances": MIN_OVERLAP_UTTERANCES,
            "min_body_characters": MIN_OVERLAP_BODY_CHARACTERS,
        },
        **{"removed_" + key: value for key, value in _counts(generated[:count]).items()},
        **{"remaining_" + key: value for key, value in _counts(generated[count:]).items()},
    }
    return result, metadata
