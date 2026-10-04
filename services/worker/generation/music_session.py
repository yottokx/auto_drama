"""Own one resident music child and exclusive GPU lease across music jobs."""
from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

from .cancellation import check_cancelled, register_cancel_callback
from .music.process_tree import WindowsChildJob
from .processes import kill_owned_process
from .runtime_lifetime import RuntimeLifetime

MUSIC_KINDS = frozenset({"m3_music"})
_active: ContextVar[MusicSession | None] = ContextVar("music_session", default=None)


class MusicSession:
    def __init__(self):
        self._lease = self._process = self._child_job = self._stderr = self._key = None
        self._lifetime = RuntimeLifetime()

    @contextmanager
    def gpu_scope(self, lease):
        self.expire_if_needed()
        if self._lease is None:
            lease.__enter__()
            self._lease = lease
        self._lifetime.active_scopes += 1
        try:
            check_cancelled()
            yield
        except BaseException:
            self.close()
            raise
        finally:
            self._lifetime.active_scopes -= 1
            self.expire_if_needed()

    def expire_if_needed(self):
        if self._lifetime.active_scopes or not self._lifetime.expired():
            return False
        self.close()
        return True

    def release_model(self):
        process = self._process
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=10)
        finally:
            if self._child_job is not None:
                self._child_job.close()
            if process is not None:
                for stream in (process.stdin, process.stdout):
                    if stream is not None:
                        stream.close()
            if self._stderr is not None:
                self._stderr.close()
            self._process = self._child_job = self._stderr = self._key = None
            self._lifetime.clear()

    def close(self):
        try:
            self.release_model()
        finally:
            lease, self._lease = self._lease, None
            if lease is not None:
                lease.__exit__(None, None, None)

    def run(self, request: dict, *, python: str, cwd: Path, timeout: float, identity: str,
            command: list[str] | None = None) -> dict:
        if self._lease is None:
            raise RuntimeError("A music child requires the parent's exclusive GPU lease.")
        started = time.monotonic()
        output = Path(request["output_dir"])
        key = (python, identity)
        try:
            check_cancelled()
            if (key != self._key or self._process is None or self._process.poll() is not None
                    or self._lifetime.expired()):
                self.release_model()
                environment = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_")}
                environment.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                    HF_HUB_DISABLE_TELEMETRY="1")
                self._stderr = (output / "runtime.log").open("w", encoding="utf-8")
                self._child_job = WindowsChildJob()
                self._process = subprocess.Popen(command or [python, "-m",
                    "services.worker.generation.music_runner", "--serve"], cwd=cwd, env=environment,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                    text=True, encoding="utf-8", bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    start_new_session=os.name != "nt")
                self._child_job.assign(self._process)
                self._key = key
            process, child_job = self._process, self._child_job
            with register_cancel_callback(lambda: kill_owned_process(process, child_job)):
                report = self._request(request, timeout, started)
                check_cancelled()
                return report
        except BaseException:
            self.close()
            raise

    def _request(self, request: dict, timeout: float, started: float):
        replies = queue.Queue(maxsize=1)
        process = self._process

        def receive():
            try:
                replies.put(process.stdout.readline(65536))
            except (OSError, ValueError) as exc:
                replies.put(exc)

        threading.Thread(target=receive, name="music-response", daemon=True).start()
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        try:
            line = replies.get(timeout=max(0.001, timeout - (time.monotonic() - started)))
        except queue.Empty as exc:
            raise TimeoutError(f"Music generation exceeded {timeout}s.") from exc
        check_cancelled()
        if isinstance(line, Exception):
            raise RuntimeError("Music response pipe closed.") from line  # noqa: TRY004
        if not line or json.loads(line).get("ok") is not True:
            raise RuntimeError("Music generation failed: " + (json.loads(line).get("error", "unknown") if line else "child exited"))
        output = Path(request["output_dir"])
        if not all((output / name).is_file() for name in ("result.json", "source.mp3", "music.mp3")):
            raise RuntimeError("Music child did not write a complete result.")
        report = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self._lifetime.record_load(report, fallback=started)
        return report


@contextmanager
def reuse_music_runtime(*, enabled=True):
    session = MusicSession() if enabled else None
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
    if session is not None and kind not in MUSIC_KINDS:
        session.close()
