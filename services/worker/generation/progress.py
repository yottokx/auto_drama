"""Scoped observer for real generation boundaries, independent of inference identity."""
from __future__ import annotations

import copy
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar

from packages.contracts.job_progress import JobProgress

_observer: ContextVar[Callable[[dict], None] | None] = ContextVar("generation_progress", default=None)


@contextmanager
def progress_scope(callback: Callable[[dict], None]) -> Iterator[None]:
    sequence = 0

    def send(value: dict) -> None:
        nonlocal sequence
        sequence += 1
        progress = JobProgress.model_validate({**value, "sequence": sequence})
        callback(progress.model_dump(mode="json", exclude_none=True))

    token = _observer.set(send)
    try:
        yield
    finally:
        _observer.reset(token)


def report_progress(value: dict) -> None:
    observer = _observer.get()
    if observer is not None:
        observer(copy.deepcopy(value))
