"""How long a resident media process keeps its model: five minutes after it was last used."""
from __future__ import annotations

import math
import time

MODEL_RETENTION_SECONDS = 300.0


class RuntimeLifetime:
    def __init__(self):
        self.loaded_at: float | None = None
        self.used_at: float | None = None
        self.active_scopes = 0

    def record_load(self, report: dict, *, fallback: float):
        """Record a finished request: when its model was loaded, and that it was used just now.

        Work that follows on the same model keeps it loaded however long the run is; a model
        nobody uses is released five minutes after its last request.
        """
        if self.loaded_at is None:
            loaded_at = report.get("model_loaded_at_monotonic", fallback)
            if type(loaded_at) not in (int, float) or not math.isfinite(loaded_at):
                raise ValueError("Resident model returned an invalid load timestamp.")
            self.loaded_at = float(loaded_at)
        self.used_at = max(time.monotonic(), self.loaded_at)

    def expired(self) -> bool:
        return (self.used_at is not None
                and time.monotonic() - self.used_at >= MODEL_RETENTION_SECONDS)

    def clear(self):
        self.loaded_at = self.used_at = None
