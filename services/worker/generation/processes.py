"""Own only spawned processes; serialize GPU use across local worker processes."""
from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path

from scripts.m0.llm_smoke import WindowsChildJob

from .cancellation import check_cancelled, register_cancel_callback

_thread_lock = threading.Lock()


@contextmanager
def gpu_lock(path: Path, timeout: float):
    began = time.monotonic()
    while True:
        check_cancelled()
        remaining = timeout - (time.monotonic() - began)
        if remaining <= 0:
            raise TimeoutError("Timed out waiting for another local GPU job.")
        if _thread_lock.acquire(timeout=min(0.1, remaining)):
            break
    try:
        check_cancelled()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a+b") as stream:
            stream.seek(0)
            if not stream.read(1):
                stream.write(b"0")
                stream.flush()
            acquired = False
            while not acquired:
                check_cancelled()
                stream.seek(0)
                try:
                    if os.name == "nt":
                        import msvcrt
                        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    else:
                        import fcntl
                        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                except (BlockingIOError, OSError):
                    if time.monotonic() - began >= timeout:
                        raise TimeoutError("Timed out waiting for another local GPU worker.")
                    time.sleep(0.25)
            try:
                check_cancelled()
                yield
            finally:
                stream.seek(0)
                if os.name == "nt":
                    import msvcrt
                    msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
    finally:
        _thread_lock.release()


def kill_owned_process(process, child_job):
    """Interrupt only the process/session started by this resource owner."""
    child_job.close()
    if process.poll() is None:
        try:
            if os.name == "nt":
                process.kill()
            else:
                # Every caller spawned this child with start_new_session=True.
                os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


@contextmanager
def owned_process(command: list[str], log: Path, *, cwd: Path, timeout: float):
    """Kill-on-close Windows job object protects against supervisor termination."""
    check_cancelled()
    child_job = WindowsChildJob()
    process = None
    timer = None
    timed_out = threading.Event()
    environment = {key: value for key, value in os.environ.items()
                   if not key.startswith("LLAMA_")}
    environment.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                       HF_HUB_DISABLE_TELEMETRY="1")
    try:
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(command, cwd=cwd, stdin=subprocess.DEVNULL,
                stdout=stream, stderr=subprocess.STDOUT, env=environment,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                start_new_session=os.name != "nt")
            child_job.assign(process)

            def terminate_after_limit():
                if process.poll() is None:
                    timed_out.set()
                    process.kill()

            timer = threading.Timer(timeout, terminate_after_limit)
            timer.daemon = True
            timer.start()
            with register_cancel_callback(lambda: kill_owned_process(process, child_job)):
                check_cancelled()
                yield process
                check_cancelled()
                if timed_out.is_set():
                    raise TimeoutError(f"Generation subprocess exceeded {timeout}s; see {log.name}.")
    finally:
        if timer:
            timer.cancel()
        if process and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        # Closing the owned Windows job also stops any children spawned by the model.
        child_job.close()


def run_process(command: list[str], log: Path, *, cwd: Path, timeout: float) -> None:
    with owned_process(command, log, cwd=cwd, timeout=timeout) as process:
        try:
            code = process.wait(timeout=timeout + 5)
        except subprocess.TimeoutExpired as exc:
            raise TimeoutError(f"Generation subprocess timed out; see {log.name}.") from exc
        check_cancelled()
        if code:
            tail = log.read_text(encoding="utf-8", errors="replace")[-3000:]
            raise RuntimeError(f"Generation subprocess failed ({code}): {tail}")
