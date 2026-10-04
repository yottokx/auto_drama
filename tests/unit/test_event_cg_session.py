"""Qwen lifecycle uses real lightweight children without a GPU runtime."""
from __future__ import annotations

import subprocess
import sys
import threading
from contextlib import contextmanager

import pytest

from services.worker.__main__ import prepare_media_job
from services.worker.generation import event_cg_session
from services.worker.generation.cancellation import GenerationCancelled, cancellation_scope
from services.worker.generation.progress import progress_scope

CHILD = '''
import json, os, sys, time
from pathlib import Path
for line in sys.stdin:
    request = json.loads(line)
    output = Path(request["output_dir"])
    mode = request.get("mode", "ok")
    if mode == "wait":
        while True: time.sleep(0.01)
    if mode in ("generation", "identity"):
        print(json.dumps({"ok": False, "error_kind": mode, "error": "injected failure"}), flush=True)
        continue
    print(json.dumps({"progress": {"phase": "generating", "step": 1, "steps": 2}}), flush=True)
    (output / "image.png").write_bytes(b"image")
    (output / "original.png").write_bytes(b"original")
    (output / "result.json").write_text(json.dumps({"ok": True, "pid": os.getpid()}))
    print(json.dumps({"ok": True}), flush=True)
'''


class Probe:
    def __init__(self, root):
        self.root, self.events, self.processes = root, [], []
        self.script = root / "child.py"
        self.script.write_text(CHILD, encoding="utf-8")

    @contextmanager
    def lease(self):
        self.events.append("acquire")
        try:
            yield
        finally:
            assert all(process.poll() is not None for process in self.processes)
            self.events.append("release")

    def run(self, session, name, cg_id="one", mode="ok", timeout=10):
        output = self.root / name
        output.mkdir()
        request = {"output_dir": str(output), "cg_id": cg_id, "mode": mode}
        with session.gpu_scope(self.lease()):
            result = session.run(request, python=sys.executable, cwd=self.root,
                timeout=timeout, identity="model-v1", command=[sys.executable, str(self.script)])
            if session._process not in self.processes:
                self.processes.append(session._process)
        return result


def test_same_cg_reuses_owned_child_and_other_cg_reloads(tmp_path):
    probe = Probe(tmp_path)
    progress = []
    with event_cg_session.reuse_event_cg_runtime() as session, progress_scope(progress.append):
        first = probe.run(session, "base")
        variant = probe.run(session, "variant")
        assert first["pid"] == variant["pid"]
        changed = probe.run(session, "next-cg", cg_id="two")
        assert first["pid"] != changed["pid"]
        assert probe.events == ["acquire"]
        prepare_media_job("m3_voice_clone")
        assert session._process is session._lease is None
        assert probe.events == ["acquire", "release"]
    assert len(progress) == 3
    assert progress[-1]["steps"][0]["completed"] == 1


@pytest.mark.parametrize("mode,error", [("generation", event_cg_session.EventCgGenerationError),
    ("identity", ValueError), ("wait", event_cg_session.EventCgGenerationError)])
def test_child_error_categories_release_gpu_before_return(tmp_path, mode, error):
    probe = Probe(tmp_path)
    with event_cg_session.reuse_event_cg_runtime() as session:
        probe.run(session, "warm")
        process = session._process
        with pytest.raises(error):
            probe.run(session, "fail", mode=mode, timeout=0.2 if mode == "wait" else 10)
        assert process.poll() is not None
        assert session._lease is session._process is None
        assert probe.events == ["acquire", "release"]


def test_cancel_kills_qwen_and_releases_lease(tmp_path):
    probe = Probe(tmp_path)
    with event_cg_session.reuse_event_cg_runtime() as session, cancellation_scope() as token:
        probe.run(session, "warm")
        process = session._process
        timer = threading.Timer(0.1, token.cancel)
        timer.start()
        try:
            with pytest.raises(GenerationCancelled):
                probe.run(session, "cancel", mode="wait")
        finally:
            timer.cancel()
        assert process.poll() is not None
        assert session._lease is None


def test_qwen_child_import_needs_no_worker_dependencies():
    result = subprocess.run([sys.executable, "-S", "-c",
        ("import sys; import services.worker.generation.event_cg_runner; "
        "assert not any(x in sys.modules for x in ('torch','pydantic','httpx','transformers','diffusers'))")],
        capture_output=True, text=True, timeout=10, check=False)
    assert result.returncode == 0, result.stderr
