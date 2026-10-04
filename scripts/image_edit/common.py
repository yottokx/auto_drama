"""Lightweight paths and durable experiment records shared by the GUI and runners."""

from __future__ import annotations

import json
import sys
import uuid
from datetime import datetime
from pathlib import Path

from scripts.audio.hub_auth import hub_environment
from scripts.audio.json_io import write_json

__all__ = [
    "MODEL_DIR",
    "MODEL_ID",
    "OUTPUT_DIR",
    "ROOT",
    "RUNTIME",
    "SETTINGS",
    "console_python",
    "hub_environment",
    "new_run",
    "read_json",
    "write_json",
]

ROOT = Path(__file__).resolve().parents[2]
RUNTIME = ROOT / "services/worker/runtimes/qwen_image_edit"
MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_DIR = RUNTIME / "models/qwen-image-2.1"
SETTINGS = ROOT / "private/image-edit/gui-settings.json"
OUTPUT_DIR = ROOT / "outputs/image-edit-tests"


def read_json(path: Path) -> dict:
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    except (OSError, ValueError, UnicodeError):
        return {}
    return value if isinstance(value, dict) else {}


def new_run(parent: Path, kind: str = "edit") -> Path:
    stamp = datetime.now().astimezone().strftime("%Y%m%d-%H%M%S")
    return Path(parent).expanduser().resolve() / f"{kind}-{stamp}-{uuid.uuid4().hex[:8]}"


def console_python() -> str:
    path = Path(sys.executable)
    if path.name.casefold() == "pythonw.exe":
        path = path.with_name("python.exe")
    return str(path)
