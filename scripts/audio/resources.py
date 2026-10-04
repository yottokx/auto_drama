"""Cooperative cancellation for GUI-owned audio and prompt subprocesses."""

from __future__ import annotations

import os
import threading
import time
from contextlib import contextmanager, nullcontext
from contextvars import ContextVar
from pathlib import Path

from .engine import GenerationCancelled

_active: ContextVar[object | None] = ContextVar("audio_cancellation", default=None)


class AudioCancellationToken:
    def __init__(self):
        self.cancelled = threading.Event()

    def cancel(self):
        self.cancelled.set()

    def check(self):
        if self.cancelled.is_set():
            raise GenerationCancelled("生成を停止しました。")


@contextmanager
def cancellation_watcher(directory: Path, *, worker_helpers: bool = False):
    # Only the root-Python prompt runner needs the worker's owned-server callbacks.
    # Music remains independent of the worker package and its Python dependencies.
    if worker_helpers:
        from services.worker.generation.cancellation import CancellationToken, cancellation_scope
    else:
        CancellationToken = AudioCancellationToken

    class FileCancellationToken(CancellationToken):
        def check(self):
            if (directory / "stop.request").exists():
                self.cancel()
            super().check()

    token = FileCancellationToken()
    finished = threading.Event()

    def watch():
        while not finished.is_set():
            if (directory / "stop.request").exists():
                token.cancel()
                return
            finished.wait(0.1)

    watcher = threading.Thread(target=watch, daemon=True)
    previous = _active.set(token)
    try:
        with cancellation_scope(token) if worker_helpers else nullcontext():
            watcher.start()
            try:
                yield token
            finally:
                finished.set()
                watcher.join(timeout=0.5)
    finally:
        _active.reset(previous)


@contextmanager
def gpu_lock(path: Path, timeout: float):
    """Lock the worker's existing lease byte using only the standard library."""
    began = time.monotonic()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        while True:
            token = _active.get()
            if token is not None:
                token.check()
            stream.seek(0)
            try:
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() - began >= timeout:
                    raise TimeoutError("他のGPU処理の終了待ちが制限時間を超えました。") from None
                time.sleep(0.1)
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@contextmanager
def music_gpu_scope(root: Path, device: str):
    """Use the same GPU lease as local worker jobs, including retained LLMs."""
    if device == "cpu":
        yield
        return
    with gpu_lock(root / "services/worker/cache/m2/gpu.lock", timeout=3600):
        yield
