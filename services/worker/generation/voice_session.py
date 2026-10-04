"""Reuse one voice child during consecutive jobs while retaining the local GPU lease.

Opted into by the worker loop, not by standalone generation calls. The loop expires
the session five minutes after load; other GPU work closes it before taking the lock.
"""
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

from scripts.m0.llm_smoke import WindowsChildJob

from .cancellation import check_cancelled, register_cancel_callback
from .processes import kill_owned_process
from .runtime_lifetime import RuntimeLifetime
from .tts_runtime import runtime_identity

VOICE_KINDS = frozenset({"m2_voice", "m2_voice_clone", "m3_voice", "m3_voice_clone"})
_active: ContextVar[VoiceSession | None] = ContextVar("voice_session", default=None)


class VoiceSession:
    def __init__(self):
        self._lease = None
        self._process = None
        self._child_job = None
        self._stderr = None
        self._key = None
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

    def expire_if_needed(self) -> bool:
        """Release an expired idle model; inference keeps its lease until completion."""
        if self._lifetime.active_scopes or not self._lifetime.expired():
            return False
        self.close()
        return True

    def _stop_process(self):
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
            # Includes descendants and abrupt worker termination on Windows.
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
            self._stop_process()
        finally:
            lease, self._lease = self._lease, None
            if lease is not None:
                lease.__exit__(None, None, None)

    def run(self, command: list[str], log: Path, *, cwd: Path, timeout: float):
        if self._lease is None:
            raise RuntimeError("A resident voice process requires an exclusive GPU lease.")
        started = time.monotonic()
        output = log.parent.resolve()
        request = json.loads((output / "request.json").read_text(encoding="utf-8"))
        key = (tuple(command[:2]), runtime_identity(request))
        try:
            check_cancelled()
            if (key != self._key or self._process is None or self._process.poll() is not None
                    or self._lifetime.expired()):
                self._stop_process()
                environment = {k: v for k, v in os.environ.items() if not k.startswith("LLAMA_")}
                environment.update(PYTHONUTF8="1", HF_HUB_OFFLINE="1",
                    TRANSFORMERS_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
                self._stderr = (output / "session-stderr.log").open("w", encoding="utf-8")
                self._child_job = WindowsChildJob()
                self._process = subprocess.Popen(
                    [*command[:2], "--serve"], cwd=cwd, env=environment,
                    stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                    text=True, encoding="utf-8", bufsize=1,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
                    start_new_session=os.name != "nt",
                )
                self._child_job.assign(self._process)
                self._key = key
            process = self._process
            child_job = self._child_job
            with register_cancel_callback(lambda: kill_owned_process(process, child_job)):
                self._request(process, output, log, timeout, started)
                check_cancelled()
        except BaseException:
            self.close()
            raise

    def _request(self, process, output: Path, log: Path, timeout: float, started: float):
        check_cancelled()
        replies = queue.Queue(maxsize=1)

        def receive():
            try:
                replies.put(process.stdout.readline(65536))
            except (OSError, ValueError) as exc:
                replies.put(exc)

        reader = threading.Thread(target=receive, name="voice-response", daemon=True)
        reader.start()
        process.stdin.write(json.dumps({"output_dir": str(output)}) + "\n")
        process.stdin.flush()
        try:
            line = replies.get(timeout=max(0.001, timeout - (time.monotonic() - started)))
        except queue.Empty as exc:
            raise TimeoutError(f"Voice generation exceeded {timeout}s; see {log}.") from exc
        check_cancelled()
        if isinstance(line, Exception):
            raise RuntimeError("Voice process closed its response pipe.") from line  # noqa: TRY004
        if not line:
            raise RuntimeError(f"Voice process exited without a result; see {log}.")
        response = json.loads(line)
        if response.get("ok") is not True:
            raise RuntimeError(f"Voice generation failed: {response.get('error', 'unknown error')}")
        if not (output / "result.json").is_file() or not (output / "voice.wav").is_file():
            raise RuntimeError("Voice process did not write its complete result.")
        report = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self._lifetime.record_load(report, fallback=started)


@contextmanager
def reuse_voice_runtime(*, enabled: bool = True):
    session = VoiceSession() if enabled else None
    token = _active.set(session)
    try:
        yield session
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            _active.reset(token)


def current_session() -> VoiceSession | None:
    return _active.get()


def prepare_job(kind: str):
    session = current_session()
    if session is not None and kind not in VOICE_KINDS:
        session.close()


def gpu_scope(kind: str, lease):
    session = current_session()
    if session is None:
        return lease
    if kind in VOICE_KINDS:
        return session.gpu_scope(lease)
    session.close()
    return lease
