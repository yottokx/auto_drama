"""Retain one image child and GPU lease across consecutive local image jobs."""
from __future__ import annotations

import hashlib
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
from .processes import kill_owned_process, run_process
from .runtime_lifetime import RuntimeLifetime

IMAGE_KINDS = frozenset({"m2_image", "m3_image", "m3_background"})
_active: ContextVar[ImageSession | None] = ContextVar("image_session", default=None)


def request_from_command(command: list[str]) -> dict:
    values = dict(zip(command[2::2], command[3::2], strict=True))
    fields = {"--mode", "--prompt", "--negative-prompt", "--model-dir", "--output-dir",
              "--width", "--height", "--steps", "--guidance-scale", "--seed"}
    if set(values) != fields:
        raise ValueError("Unsupported resident image command.")
    request = {key[2:].replace("-", "_"): value for key, value in values.items()}
    for name in ("width", "height", "steps", "seed"):
        request[name] = int(request[name])
    request["guidance_scale"] = float(request["guidance_scale"])
    return request


def runtime_identity(request: dict, cwd: Path) -> str:
    model = Path(request["model_dir"]).resolve()
    records = {}
    for path in (cwd / "config/m0-models-image.json", model / "conversion.json",
                 model / "modular_model_index.json"):
        records[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return hashlib.sha256(json.dumps({"model_dir": str(model), "records": records,
                                     "precision": "bf16"}, sort_keys=True).encode()).hexdigest()


class ImageSession:
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

    def expire_if_needed(self) -> bool:
        """Release an expired idle model; inference keeps its lease until completion."""
        if self._lifetime.active_scopes or not self._lifetime.expired():
            return False
        self.close()
        return True

    def release_model(self):
        """Release weights before an LLM recovery, retaining exclusive GPU ownership."""
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

    def run(self, command: list[str], log: Path, *, cwd: Path, timeout: float):
        if self._lease is None:
            raise RuntimeError("A resident image process requires an exclusive GPU lease.")
        started = time.monotonic()
        request = request_from_command(command)
        output = Path(request["output_dir"]).resolve()
        key = (tuple(command[:2]), runtime_identity(request, cwd))
        try:
            check_cancelled()
            if (key != self._key or self._process is None or self._process.poll() is not None
                    or self._lifetime.expired()):
                self.release_model()
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
            process, child_job = self._process, self._child_job
            with register_cancel_callback(lambda: kill_owned_process(process, child_job)):
                self._request(request, output, log, timeout, started)
                check_cancelled()
        except BaseException:
            self.close()
            raise

    def _request(self, request, output, log, timeout, started):
        replies = queue.Queue(maxsize=1)
        process = self._process

        def receive():
            try:
                replies.put(process.stdout.readline(65536))
            except (OSError, ValueError) as exc:
                replies.put(exc)

        threading.Thread(target=receive, name="image-response", daemon=True).start()
        process.stdin.write(json.dumps(request, ensure_ascii=False) + "\n")
        process.stdin.flush()
        try:
            line = replies.get(timeout=max(0.001, timeout - (time.monotonic() - started)))
        except queue.Empty as exc:
            raise TimeoutError(f"Image generation exceeded {timeout}s; see {log}.") from exc
        check_cancelled()
        if isinstance(line, Exception):
            raise RuntimeError("Image process closed its response pipe.") from line  # noqa: TRY004
        if not line:
            raise RuntimeError(f"Image process exited without a result; see {log}.")
        response = json.loads(line)
        if response.get("ok") is not True:
            raise RuntimeError(f"Image generation failed: {response.get('error', 'unknown error')}")
        files = ["result.json", "image.png"] + (["character.png"] if request["mode"] == "character" else [])
        if not all((output / name).is_file() for name in files):
            raise RuntimeError("Image process did not write its complete result.")
        report = json.loads((output / "result.json").read_text(encoding="utf-8"))
        self._lifetime.record_load(report, fallback=started)


@contextmanager
def reuse_image_runtime(*, enabled: bool = True):
    session = ImageSession() if enabled else None
    token = _active.set(session)
    try:
        yield session
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            _active.reset(token)


def current_session() -> ImageSession | None:
    return _active.get()


def prepare_job(kind: str):
    session = current_session()
    if session is not None and kind not in IMAGE_KINDS:
        session.close()


def gpu_scope(kind: str, lease):
    session = current_session()
    if session is None:
        return lease
    if kind in IMAGE_KINDS:
        return session.gpu_scope(lease)
    session.close()
    return lease


def run_image_process(command, log, *, cwd, timeout):
    session = current_session()
    if session is None:
        return run_process(command, log, cwd=cwd, timeout=timeout)
    return session.run(command, log, cwd=cwd, timeout=timeout)
