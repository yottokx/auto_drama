"""Hard model residency deadline shared by the resident media processes."""
from __future__ import annotations

import math
import time

MODEL_RETENTION_SECONDS = 300.0


class RuntimeLifetime:
    def __init__(self):
        self.loaded_at: float | None = None
        self.active_scopes = 0

    def record_load(self, report: dict, *, fallback: float):
        """Record the original load, never extending it for reused requests."""
        if self.loaded_at is not None:
            return
        loaded_at = report.get("model_loaded_at_monotonic", fallback)
        if type(loaded_at) not in (int, float) or not math.isfinite(loaded_at):
            raise ValueError("Resident model returned an invalid load timestamp.")
        self.loaded_at = float(loaded_at)

    def expired(self) -> bool:
        return (self.loaded_at is not None
                and time.monotonic() - self.loaded_at >= MODEL_RETENTION_SECONDS)

    def clear(self):
        self.loaded_at = None
