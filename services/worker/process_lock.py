"""Prevent simultaneous worker processes from sharing a persistent work directory."""

from __future__ import annotations

import errno
import os
from contextlib import contextmanager
from pathlib import Path


class WorkerAlreadyRunning(RuntimeError):
    """Another process currently owns this worker's work directory."""


@contextmanager
def worker_process_lock(work_dir: Path):
    """Keep an OS-owned lock until exit; stale files do not prevent restarting."""
    directory = work_dir.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    # Never delete this file: another process may already hold its open handle.
    with (directory / ".worker.lock").open("a+b") as stream:
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt

                # Windows byte-range locks can cover a byte past the end of a file.
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl

                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as error:
            if error.errno not in {errno.EACCES, errno.EAGAIN, errno.EDEADLK}:
                raise
            raise WorkerAlreadyRunning(
                f"同じ作業フォルダーのワーカーが既に起動しています: {directory}。"
                "起動済みの端末で Ctrl+C を押して終了してから再実行してください。"
                "別のワーカーを起動する場合は -WorkDir（--work-dir）で別のフォルダーを指定してください。"
            ) from error
        # Closing the handle releases the lock, including after an exception or kill.
        yield directory
