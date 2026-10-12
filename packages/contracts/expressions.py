"""Facial expressions a staged line may ask of a character's portrait.

The kinds are fixed, so every script names them the same way and a picture made for one
serves any chapter. A script states the expression it wants whether or not a picture of it
exists: where none does the character's ordinary portrait is shown, and a picture made
later is used without annotating the script again.

A production staged before expressions existed has no expression data anywhere: every field
that carries it is omitted when empty, so such productions keep their exact bytes.
"""

from __future__ import annotations

from typing import Literal

Expression = Literal["neutral", "smile", "laugh", "angry", "sad", "cry", "surprised", "confused",
                     "anxious", "serious", "fear", "shy"]
EXPRESSIONS: tuple[str, ...] = ("neutral", "smile", "laugh", "angry", "sad", "cry", "surprised", "confused",
                                "anxious", "serious", "fear", "shy")
DEFAULT_EXPRESSION = "neutral"
EXPRESSION_NAMES = {
    "neutral": "通常", "smile": "微笑み", "laugh": "笑顔", "angry": "怒り", "sad": "悲しみ", "cry": "泣き",
    "surprised": "驚き", "confused": "困惑", "anxious": "不安", "serious": "真剣", "fear": "恐怖", "shy": "照れ",
}
# How long one face gives way to the next. Short enough not to hold back the line it accompanies.
EXPRESSION_FADE_MS = 150


def expression_changes(lines: list[tuple[str, str | None, str]],
                       listed: dict[str, dict[str, str]]) -> list[tuple[str, str, str]]:
    """Where each person's face changes within one scene, in script order.

    ``lines`` are (utterance ID, speaker or None, the speaker's expression); ``listed`` names,
    per utterance, the people other than the speaker whose face changes there. A face stays as
    it is until it changes, and everyone is in their ordinary face when the scene opens, so
    only real changes are returned.
    """
    state: dict[str, str] = {}
    changes = []
    for utterance_id, speaker, expression in lines:
        wanted = dict(listed.get(utterance_id, {}))
        if speaker is not None:
            # The line's own annotation is about the person speaking it.
            wanted[speaker] = expression
        for character_id, value in wanted.items():
            if state.get(character_id, DEFAULT_EXPRESSION) != value:
                state[character_id] = value
                changes.append((utterance_id, character_id, value))
    return changes
