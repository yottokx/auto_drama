"""Atomic status/result writes that tolerate brief Windows reader locks."""

from __future__ import annotations

import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

_REPLACE_TIMEOUT = 1.5
_WINDOWS_LOCK_ERRORS = {5, 32, 33}


def _replace_when_unlocked(source: Path, destination: Path) -> None:
    """Retry only Windows access/sharing/lock violations, with a fixed deadline."""
    deadline = time.monotonic() + _REPLACE_TIMEOUT
    delay = 0.025
    while True:
        try:
            os.replace(source, destination)
            return
        except OSError as exc:
            remaining = deadline - time.monotonic()
            if getattr(exc, "winerror", None) not in _WINDOWS_LOCK_ERRORS or remaining <= 0:
                raise
            time.sleep(min(delay, remaining))
            delay = min(delay * 2, 0.2)


def write_json(path: Path, data: Mapping[str, Any]) -> None:
    """Publish complete JSON while keeping prior contents on any failed write.

    Each writer owns a unique temporary file beside the destination. Close it
    before replacing, then give a polling GUI time to release its read handle.
    Invalid JSON and persistent filesystem errors remain visible to callers.
    """
    payload = json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(payload)
        _replace_when_unlocked(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
