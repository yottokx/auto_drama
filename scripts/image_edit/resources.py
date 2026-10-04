"""Cancellation and the shared GPU lease without inference or worker imports."""

from __future__ import annotations

import os
import time
from contextlib import contextmanager
from pathlib import Path


class GenerationCancelled(Exception):
    """The experiment was stopped before the current image completed."""


class StopToken:
    def __init__(self, directory: Path):
        self.path = Path(directory) / "stop.request"

    def check(self) -> None:
        if self.path.exists():
            raise GenerationCancelled("生成を停止しました。完了済みの画像は保持しています。")


@contextmanager
def gpu_lock(path: Path, timeout: float = 3600, *, token: StopToken | None = None):
    """Own the same lock byte as local image, voice, LLM and audio jobs."""
    started = time.monotonic()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a+b") as stream:
        if os.fstat(stream.fileno()).st_size == 0:
            stream.write(b"0")
            stream.flush()
        while True:
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
                if time.monotonic() - started >= timeout:
                    raise TimeoutError("他のGPU処理の終了待ちが制限時間を超えました。") from None
                time.sleep(0.1)
        try:
            if token is not None:
                token.check()
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
