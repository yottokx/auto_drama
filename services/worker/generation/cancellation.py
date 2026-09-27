"""Cooperative cancellation scoped to one worker attempt and its owned resources."""
from __future__ import annotations

import logging
import threading
from collections.abc import Callable
from contextlib import contextmanager
from contextvars import ContextVar

logger = logging.getLogger(__name__)


class GenerationCancelled(Exception):
    """The coordinator has definitively revoked this attempt's lease."""


class CancellationToken:
    def __init__(self):
        self.cancelled = threading.Event()
        self._lock = threading.RLock()
        self._callbacks: set[Callable[[], None]] = set()

    def check(self):
        if self.cancelled.is_set():
            raise GenerationCancelled("Generation attempt no longer owns its lease.")

    @staticmethod
    def _invoke(callback):
        try:
            callback()
        except Exception:
            logger.exception("Could not interrupt an owned generation resource")

    def cancel(self):
        # Removal is synchronized with invocation. A late callback can never
        # kill a resident process after it has been handed to the next job.
        with self._lock:
            if self.cancelled.is_set():
                return
            self.cancelled.set()
            for callback in tuple(self._callbacks):
                self._invoke(callback)

    def register(self, callback: Callable[[], None]) -> Callable[[], None]:
        with self._lock:
            if self.cancelled.is_set():
                self._invoke(callback)
                self.check()
            self._callbacks.add(callback)

        def remove():
            with self._lock:
                self._callbacks.discard(callback)

        return remove


_active: ContextVar[CancellationToken | None] = ContextVar("generation_cancellation", default=None)


@contextmanager
def cancellation_scope(token: CancellationToken | None = None):
    cancellation = token or CancellationToken()
    previous = _active.set(cancellation)
    try:
        cancellation.check()
        yield cancellation
    finally:
        _active.reset(previous)


def check_cancelled():
    token = _active.get()
    if token is not None:
        token.check()


@contextmanager
def register_cancel_callback(callback: Callable[[], None]):
    token = _active.get()
    remove = token.register(callback) if token is not None else None
    try:
        yield
    finally:
        if remove is not None:
            remove()
