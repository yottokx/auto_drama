"""Share serial LLM jobs; M2 expires from load, M3 after completed use."""
from __future__ import annotations

import time
from contextlib import contextmanager
from contextvars import ContextVar

from .cancellation import check_cancelled

LLM_KINDS = frozenset({"m2_world", "m2_character", "m2_relationships",
                       "m3_plan", "m3_narrative", "m3_music_plan"})
COMPLETION_RETENTION_KINDS = frozenset({"m3_plan", "m3_narrative", "m3_music_plan"})
CONVERSION_KINDS = frozenset({"m2_image"})
MODEL_RETENTION_SECONDS = 300.0
_active: ContextVar[LLMSession | None] = ContextVar("llm_session", default=None)


class LLMSession:
    def __init__(self):
        self._lease = self._backend = self._key = self._loaded_at = None
        self._active_client = None
        self._idle_since = None
        self._retention_from_completion = False
        self.enabled = True

    def _expired(self) -> bool:
        origin = self._idle_since if self._retention_from_completion else self._loaded_at
        return origin is not None and time.monotonic() - origin >= MODEL_RETENTION_SECONDS

    def expire_if_needed(self) -> bool:
        """Idle polling expires weights; active inference finishes before expiry."""
        if self._active_client is None and self._expired():
            self.close()
            return True
        return False

    @contextmanager
    def gpu_scope(self, lease):
        self.expire_if_needed()
        if self._lease is None:
            lease.__enter__()
            self._lease = lease
        try:
            check_cancelled()
            yield
        except BaseException:
            self.close()
            raise

    def release_model(self, *args):
        """Stop the server while retaining GPU ownership for a replacement."""
        backend, self._backend = self._backend, None
        self._key = self._loaded_at = self._idle_since = None
        if backend is not None:
            backend._stop_runtime(*(args or (None, None, None)))

    def close(self):
        try:
            self.release_model()
        finally:
            self._active_client = None
            lease, self._lease = self._lease, None
            if lease is not None:
                lease.__exit__(None, None, None)

    def attach(self, client):
        if self._lease is None:
            raise RuntimeError("A resident LLM requires an exclusive GPU lease.")
        if self._active_client is not None:
            raise RuntimeError("The resident LLM is already serving another generation context.")
        key = client._runtime_key()
        backend = self._backend
        if (backend is None or key != self._key or self._expired()
                or backend.process.poll() is not None):
            self.release_model()
            client._start_runtime(persistent=True)
            self._backend, self._key = client, key
            self._loaded_at = time.monotonic()
        else:
            for name in ("process", "base_url", "server_context_size",
                         "_launch_context_size", "context_source"):
                setattr(client, name, getattr(backend, name))
            client.trace.append({"type": "model_reuse", "status": "ready",
                                 "loaded_seconds_ago": time.monotonic() - self._loaded_at})
            client.trace.append({"type": "llm_runtime", **client.runtime_identity(),
                                 "context_source": client.context_source})
        self._active_client = client
        if self._retention_from_completion:
            self._idle_since = None

    def detach(self, client, args):
        if self._active_client is client:
            self._active_client = None
        if args[0] is not None:
            self.close()
        else:
            # Chapter writing may take longer than five minutes. Its successor
            # can reuse the completed review; the original M2 load deadline is
            # unchanged, and idle polling still releases M3 after five minutes.
            if self._retention_from_completion and self._backend is not None:
                self._idle_since = time.monotonic()
            self.expire_if_needed()


@contextmanager
def reuse_llm_runtime(*, enabled: bool = True):
    session = LLMSession() if enabled else None
    token = _active.set(session)
    try:
        yield session
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            _active.reset(token)


def current_session() -> LLMSession | None:
    return _active.get()


def prepare_job(kind: str):
    session = current_session()
    if session is not None:
        session.expire_if_needed()
        completion_retention = kind in COMPLETION_RETENTION_KINDS
        if completion_retention != session._retention_from_completion:
            # Changing job kinds alone does not grant a fresh retention window.
            if completion_retention and session._backend is not None:
                session._idle_since = session._loaded_at
            session._retention_from_completion = completion_retention
        session.enabled = kind in LLM_KINDS | CONVERSION_KINDS
        if not session.enabled:
            session.close()


def gpu_scope(kind: str, lease):
    session = current_session()
    if session is not None and session.enabled and kind in LLM_KINDS:
        return session.gpu_scope(lease)
    return lease
