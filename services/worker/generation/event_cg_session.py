"""Parent-owned Qwen child, retained only across jobs for the same event CG."""
from __future__ import annotations

import json
import queue
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .cancellation import check_cancelled
from .music_session import MusicSession
from .progress import report_progress

KINDS = frozenset({"m3_event_cg"})
_active: ContextVar[EventCgSession | None] = ContextVar("event_cg_session", default=None)


class EventCgGenerationError(RuntimeError):
    """A definitive image-model failure may resolve to a visible omission."""


class EventCgSession(MusicSession):
    """Reuse the established owned-process/lease lifetime, with a CG protocol."""

    def run(self, request, *, python, cwd, timeout, identity, command=None):
        return super().run(request, python=python, cwd=cwd, timeout=timeout,
            identity=identity + ":" + request["cg_id"], command=command or
            [python, "-m", "services.worker.generation.event_cg_runner", "--serve"])

    def _request(self, request, timeout, started):
        replies = queue.Queue()
        process = self._process

        def receive():
            try:
                while True:
                    line = process.stdout.readline(65536)
                    replies.put(line)
                    if not line or "progress" not in json.loads(line):
                        break
            except (OSError, ValueError) as exc:
                replies.put(exc)

        threading.Thread(target=receive, name="event-cg-response", daemon=True).start()
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        while True:
            try:
                line = replies.get(timeout=max(0.001, timeout - (time.monotonic() - started)))
            except queue.Empty as exc:
                raise EventCgGenerationError(f"CG generation exceeded {timeout}s.") from exc
            check_cancelled()
            if isinstance(line, Exception):
                raise RuntimeError("CG response pipe closed.") from line  # noqa: TRY004
            if not line:
                raise RuntimeError("CG child exited without its result.")
            response = json.loads(line)
            if "progress" in response:
                state = response["progress"]
                step = {"id": "event-cg-image", "stage": "event_cg_generate", "status": "running"}
                if state.get("phase") == "generating":
                    step.update(completed=state["step"], total=state["steps"])
                report_progress({"phase": "chapter", "current_step": step["id"], "steps": [step]})
                continue
            if response.get("ok") is not True:
                error = response.get("error", "Invalid CG child response.")
                if response.get("error_kind") == "generation":
                    raise EventCgGenerationError(error)
                raise ValueError(error)
            output = Path(request["output_dir"])
            if not all((output / name).is_file() for name in ("result.json", "image.png", "original.png")):
                raise ValueError("CG child did not write a complete result.")
            report = json.loads((output / "result.json").read_text(encoding="utf-8"))
            self._lifetime.record_load(report, fallback=started)
            return report


@contextmanager
def reuse_event_cg_runtime(*, enabled=True):
    session = EventCgSession() if enabled else None
    token = _active.set(session)
    try:
        yield session
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            _active.reset(token)


def current_session():
    return _active.get()


def prepare_job(kind):
    session = current_session()
    if session is not None and kind not in KINDS:
        session.close()
