"""Immutable content-addressed objects; only committed DB rows make objects visible."""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4


@dataclass(frozen=True)
class StoredObject:
    sha256: str
    size_bytes: int
    storage_key: str


class ArtifactStore:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.staging = self.root / "staging"
        self.staging.mkdir(parents=True, exist_ok=True)

    def put(self, data: bytes) -> StoredObject:
        digest = hashlib.sha256(data).hexdigest()
        key = f"objects/{digest[:2]}/{digest}"
        target = self.root / key
        target.parent.mkdir(parents=True, exist_ok=True)
        staged = self.staging / f"{uuid4().hex}.part"
        try:
            with staged.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            # Verify before publishing. A crash before the DB commit leaves an
            # unreferenced object, never a completed artifact or completed job.
            if hashlib.sha256(staged.read_bytes()).hexdigest() != digest:
                raise OSError("Artifact write verification failed")
            if target.exists():
                if target.read_bytes() != data:
                    raise OSError("Existing artifact failed integrity verification")
            else:
                os.replace(staged, target)
                self._sync_directory(target.parent)
        finally:
            staged.unlink(missing_ok=True)
        return StoredObject(digest, len(data), key)

    def read(self, record: dict) -> bytes:
        digest = record["sha256"]
        if not re.fullmatch(r"[a-f0-9]{64}", digest):
            raise OSError("Invalid artifact digest")
        key = f"objects/{digest[:2]}/{digest}"
        if record["storage_key"] != key:
            raise OSError("Invalid artifact storage key")
        target = (self.root / key).resolve()
        if not target.is_relative_to(self.root):
            raise OSError("Invalid artifact path")
        data = target.read_bytes()
        if len(data) != record["size_bytes"] or hashlib.sha256(data).hexdigest() != digest:
            raise OSError("Artifact failed integrity verification")
        return data

    @staticmethod
    def _sync_directory(directory: Path) -> None:
        # Windows flushes the file above; opening a directory is POSIX-specific.
        if os.name != "nt":
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
