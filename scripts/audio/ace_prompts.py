"""Small, dependency-free validation for ACE-Step caption metadata."""

from __future__ import annotations

import re
from collections.abc import Mapping

MUSIC_BACKENDS = frozenset({"stable_audio3", "ace_step15"})
_METADATA_FIELDS = frozenset({"bpm", "keyscale", "timesignature"})
_KEY = re.compile(r"([A-Ga-g])([#b♯♭]?)\s+(major|minor)", re.IGNORECASE)


def validate_music_backend(value: object) -> str:
    if not isinstance(value, str) or value not in MUSIC_BACKENDS:
        raise ValueError("音楽モデルには stable_audio3 または ace_step15 を指定してください。")
    return value


def normalize_ace_metadata(value: object, *, allow_unspecified: bool = False) -> dict:
    """Validate the three text metadata fields; never accept planner/code fields.

    LLM output requires a JSON integer BPM and all fields. GUI controls can pass
    integer strings; unspecified manual values normalize to 0 / empty strings.
    """
    if not isinstance(value, Mapping) or set(value) - _METADATA_FIELDS:
        raise ValueError("ACE-Stepの音楽設定には BPM・調・拍子のみ指定できます。")
    if not allow_unspecified and set(value) != _METADATA_FIELDS:
        raise ValueError("ACE-Stepの BPM・調・拍子をすべて指定してください。")

    bpm = value.get("bpm")
    if allow_unspecified and (bpm is None or bpm == ""):
        bpm = 0
    if allow_unspecified and isinstance(bpm, str) and re.fullmatch(r"\d+", bpm.strip()):
        bpm = int(bpm.strip())
    if (isinstance(bpm, bool) or not isinstance(bpm, int)
            or not (30 <= bpm <= 300 or (allow_unspecified and bpm == 0))):
        raise ValueError("ACE-Stepの BPM は30〜300の整数で指定してください（手動の未指定は0）。")

    keyscale = value.get("keyscale")
    if allow_unspecified and keyscale is None:
        keyscale = ""
    if not isinstance(keyscale, str):
        raise ValueError("ACE-Stepの調を C major や F# minor の形式で指定してください。")  # noqa: TRY004 - validate response data
    keyscale = keyscale.strip()
    if not (allow_unspecified and not keyscale):
        match = _KEY.fullmatch(keyscale)
        if not match:
            raise ValueError("ACE-Stepの調を C major や F# minor の形式で指定してください。")
        note, accidental, mode = match.groups()
        accidental = accidental.replace("♯", "#").replace("♭", "b")
        keyscale = f"{note.upper()}{accidental} {mode.lower()}"

    meter = value.get("timesignature")
    if isinstance(meter, bool):
        raise ValueError("ACE-Stepの拍子は文字列の 2・3・4・6 で指定してください。")  # noqa: TRY004 - validate response data
    if allow_unspecified and (meter is None or meter == 0):
        meter = ""
    if allow_unspecified and isinstance(meter, int) and not isinstance(meter, bool):
        meter = str(meter)
    if not isinstance(meter, str):
        raise ValueError("ACE-Stepの拍子は文字列の 2・3・4・6 で指定してください。")  # noqa: TRY004 - validate response data
    meter = meter.strip()
    if meter not in {"2", "3", "4", "6"} and not (allow_unspecified and not meter):
        raise ValueError("ACE-Stepの拍子は 2・3・4・6 で指定してください。")
    return {"bpm": bpm, "keyscale": keyscale, "timesignature": meter}
