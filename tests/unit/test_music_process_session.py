"""Resident music ownership tested with real lightweight protocol children."""
from __future__ import annotations

import subprocess
import sys
import threading
from contextlib import contextmanager

import pytest

from services.worker.__main__ import prepare_media_job
from services.worker.generation import music_session
from services.worker.generation.cancellation import GenerationCancelled, cancellation_scope

FAKE_SERVER = '''
import json, os, sys, time
from pathlib import Path
root = Path(__file__).parent
with (root / "starts.jsonl").open("a") as stream:
    stream.write(str(os.getpid()) + "\\n")
for line in sys.stdin:
    request = json.loads(line)
    output = Path(request["output_dir"])
    behavior = request["prompt"]
    if behavior == "timeout":
        while True: time.sleep(0.02)
    if behavior == "exit": raise SystemExit(2)
    if behavior == "malformed":
        print("{invalid", flush=True)
        continue
    if behavior == "error":
        print(json.dumps({"ok": False, "error": "intentional music failure"}), flush=True)
        continue
    report = {"pid": os.getpid(), "request": request}
    if "loaded_at" in request: report["model_loaded_at_monotonic"] = request["loaded_at"]
    (output / "result.json").write_text(json.dumps(report))
    (output / "source.mp3").write_bytes(b"ID3source")
    if behavior != "incomplete": (output / "music.mp3").write_bytes(b"ID3loop")
    print(json.dumps({"ok": True}), flush=True)
'''


class Probe:
    def __init__(self, root):
        self.root, self.events, self.processes = root, [], []
        self.held = False
        self.script = root / "fake_music.py"
        self.script.write_text(FAKE_SERVER, encoding="utf-8")

    @contextmanager
    def lease(self):
        assert not self.held
        self.held = True
        self.events.append("acquire")
        try:
            yield
        finally:
            assert all(process.poll() is not None for process in self.processes)
            self.held = False
            self.events.append("release")

    def run(self, session, name, *, identity="medium-v1", prompt="instrumental", timeout=10, loaded_at=None):
        output = self.root / name
        output.mkdir()
        request = {"settings": {"model": "medium"}, "prompt": prompt, "output_dir": str(output)}
        if loaded_at is not None:
            request["loaded_at"] = loaded_at
        with session.gpu_scope(self.lease()):
            report = session.run(request, python=sys.executable, cwd=self.root, timeout=timeout,
                identity=identity, command=[sys.executable, str(self.script)])
            if session._process not in self.processes:
                self.processes.append(session._process)
        return report


@pytest.fixture
def runtime(tmp_path):
    return Probe(tmp_path)


def test_consecutive_music_reuses_one_child_and_lease_then_switches(runtime):
    with music_session.reuse_music_runtime() as session:
        first = runtime.run(session, "first")
        second = runtime.run(session, "second")
        assert first["pid"] == second["pid"]
        assert runtime.events == ["acquire"]
        prepare_media_job("m3_music_plan")
        assert session._process is session._lease is None
        assert runtime.events == ["acquire", "release"]


def test_checkpoint_change_reloads_under_same_exclusive_lease(runtime):
    with music_session.reuse_music_runtime() as session:
        first = runtime.run(session, "first")
        second = runtime.run(session, "new", identity="medium-v2")
        assert first["pid"] != second["pid"]
        assert runtime.events == ["acquire"]
        assert runtime.processes[0].poll() is not None


def test_reuse_does_not_extend_original_model_deadline(runtime, monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(music_session.time, "monotonic", lambda: elapsed[0])
    with music_session.reuse_music_runtime() as session:
        first = runtime.run(session, "first", loaded_at=0)
        elapsed[0] = 250
        assert runtime.run(session, "second", loaded_at=250)["pid"] == first["pid"]
        elapsed[0] = 300
        assert session.expire_if_needed()
        assert not runtime.held
        assert session._process is None


@pytest.mark.parametrize("behavior", ["error", "exit", "timeout", "malformed", "incomplete"])
def test_music_failure_releases_owned_child_and_allows_new_attempt(runtime, behavior):
    with music_session.reuse_music_runtime() as session:
        runtime.run(session, "warmup")
        previous = session._process
        with pytest.raises((RuntimeError, ValueError, TimeoutError)):
            runtime.run(session, "failure", prompt=behavior, timeout=0.2 if behavior == "timeout" else 10)
        assert previous.poll() is not None
        assert session._process is session._lease is None
        assert not runtime.held
        runtime.run(session, "retry")


def test_cancellation_kills_only_the_music_child(runtime):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        with music_session.reuse_music_runtime() as session, cancellation_scope() as token:
            runtime.run(session, "warmup")
            previous = session._process
            timer = threading.Timer(0.05, token.cancel)
            timer.start()
            try:
                with pytest.raises(GenerationCancelled):
                    runtime.run(session, "cancel", prompt="timeout")
            finally:
                timer.cancel()
            assert previous.poll() is not None
            assert unrelated.poll() is None
            assert not runtime.held
    finally:
        unrelated.kill()
        unrelated.wait(timeout=10)


def test_dedicated_child_import_has_no_supervisor_or_gpu_dependencies():
    code = ("import sys; import services.worker.generation.music_runner; "
        "assert not any(name in sys.modules for name in ('torch', 'numpy', 'pydantic', 'httpx', 'services.worker.client'))")
    process = subprocess.run([sys.executable, "-S", "-c", code],
        capture_output=True, text=True, timeout=10, check=False)
    assert process.returncode == 0, process.stderr
