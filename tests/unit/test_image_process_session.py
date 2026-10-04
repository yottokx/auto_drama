"""Resident image ownership exercised with lightweight real child processes."""
from __future__ import annotations

import json
import subprocess
import sys
import threading
from contextlib import contextmanager
from pathlib import Path

import pytest

from services.worker.__main__ import prepare_media_job
from services.worker.generation import image_session, voice_session
from services.worker.generation.cancellation import GenerationCancelled, cancellation_scope

FAKE_SERVER = r'''
import json
import os
import sys
import time
from pathlib import Path
root = Path(__file__).parent
with (root / "starts.jsonl").open("a") as stream:
    stream.write(json.dumps({"pid": os.getpid()}) + "\n")
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
        print(json.dumps({"ok": False, "error": "intentional render failure"}), flush=True)
        continue
    report = {"pid": os.getpid(), "request": request}
    if behavior.startswith("loaded-at:"):
        report["model_loaded_at_monotonic"] = float(behavior.split(":", 1)[1])
    (output / "result.json").write_text(json.dumps(report))
    if behavior != "incomplete":
        (output / "image.png").write_bytes(json.dumps(request).encode())
        if request["mode"] == "character": (output / "character.png").write_bytes(b"cutout")
    print(json.dumps({"ok": True}), flush=True)
'''


class RuntimeProbe:
    def __init__(self, root):
        self.root = root
        self.script = root / "fake_image.py"
        self.script.write_text(FAKE_SERVER, encoding="utf-8")
        self.held = False
        self.events, self.processes = [], []

    @contextmanager
    def lease(self, name="image"):
        assert not self.held, "GPU leases must not overlap"
        self.held = True
        self.events.append(("acquire", name))
        try:
            yield
        finally:
            assert all(process.poll() is not None for process in self.processes)
            self.held = False
            self.events.append(("release", name))

    def command(self, name, *, mode="character", prompt=None, model="model-a", seed=1):
        output = self.root / name
        output.mkdir()
        return [sys.executable, str(self.script), "--mode", mode, "--prompt", prompt or name,
                "--negative-prompt", "", "--model-dir", str(self.root / model), "--output-dir", str(output),
                "--width", "768" if mode == "character" else "1280",
                "--height", "1024" if mode == "character" else "720", "--steps", "30",
                "--guidance-scale", "4.0", "--seed", str(seed)]

    def run(self, session, command, *, timeout=10):
        output = Path(image_session.request_from_command(command)["output_dir"])
        with session.gpu_scope(self.lease()):
            session.run(command, output / "runtime.log", cwd=self.root, timeout=timeout)
            if session._process not in self.processes:
                self.processes.append(session._process)
        return json.loads((output / "result.json").read_text())

    def starts(self):
        return (self.root / "starts.jsonl").read_text().splitlines()


@pytest.fixture
def runtime(tmp_path):
    return RuntimeProbe(tmp_path)


def test_portraits_then_backgrounds_share_one_process_and_lease(runtime):
    with image_session.reuse_image_runtime() as session:
        reports = [runtime.run(session, runtime.command(f"request-{n}", mode=mode, seed=11 + n))
                   for n, mode in enumerate(("character", "character", "background", "background"))]
        assert len({report["pid"] for report in reports}) == len(runtime.starts()) == 1
        assert [report["request"]["seed"] for report in reports] == [11, 12, 13, 14]
        assert [report["request"]["width"] for report in reports] == [768, 768, 1280, 1280]
        assert [report["request"]["prompt"] for report in reports] == [f"request-{n}" for n in range(4)]
        assert runtime.events == [("acquire", "image")]
    assert runtime.events == [("acquire", "image"), ("release", "image")]
    assert image_session.current_session() is None


def test_model_identity_change_stops_old_child_under_same_lease(runtime):
    with image_session.reuse_image_runtime() as session:
        runtime.run(session, runtime.command("first"))
        previous = session._process
        runtime.run(session, runtime.command("second", model="model-b"))
        assert previous.poll() is not None
        assert len(runtime.starts()) == 2
        assert runtime.events == [("acquire", "image")]


def test_idle_image_reuse_does_not_extend_five_minutes_from_load(runtime, monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(image_session.time, "monotonic", lambda: elapsed[0])
    with image_session.reuse_image_runtime() as session:
        first = runtime.run(session, runtime.command("first"))
        elapsed[0] = 200
        assert not session.expire_if_needed()
        second = runtime.run(session, runtime.command("retake"))
        assert first["pid"] == second["pid"]
        elapsed[0] = 299.9
        assert not session.expire_if_needed()
        assert runtime.held
        elapsed[0] = 300
        assert session.expire_if_needed()
        assert not session.expire_if_needed()
        assert not runtime.held
        assert runtime.processes[0].poll() is not None
        runtime.run(session, runtime.command("after-timeout"))
        assert len(runtime.starts()) == 2


def test_image_deadline_uses_completed_model_load_time(runtime, monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(image_session.time, "monotonic", lambda: elapsed[0])
    with image_session.reuse_image_runtime() as session:
        request = session._request

        def loaded_then_generated(*args):
            request(*args)
            elapsed[0] = 350

        with monkeypatch.context() as initialization:
            initialization.setattr(session, "_request", loaded_then_generated)
            runtime.run(session, runtime.command("slow-load", prompt="loaded-at:120"))
        assert not session.expire_if_needed()
        elapsed[0] = 400
        runtime.run(session, runtime.command("retake", prompt="loaded-at:300"))
        elapsed[0] = 420
        assert session.expire_if_needed()
        assert len(runtime.starts()) == 1
        assert not runtime.held


def test_image_expiring_during_inference_releases_at_scope_exit(runtime, monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(image_session.time, "monotonic", lambda: elapsed[0])
    with image_session.reuse_image_runtime() as session:
        runtime.run(session, runtime.command("first"))
        previous = session._process
        request = session._request

        def crossing_deadline(*args):
            elapsed[0] = 301
            assert not session.expire_if_needed()
            assert previous.poll() is None
            request(*args)
            assert previous.poll() is None

        monkeypatch.setattr(session, "_request", crossing_deadline)
        elapsed[0] = 299
        runtime.run(session, runtime.command("long-retake"))
        assert previous.poll() is not None
        assert not runtime.held
        assert session._process is session._lease is None


def test_new_image_model_gets_its_own_load_deadline(runtime, monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(image_session.time, "monotonic", lambda: elapsed[0])
    with image_session.reuse_image_runtime() as session:
        runtime.run(session, runtime.command("first"))
        elapsed[0] = 250
        runtime.run(session, runtime.command("new-model", model="model-b"))
        elapsed[0] = 300
        assert not session.expire_if_needed()
        elapsed[0] = 550
        assert session.expire_if_needed()
        assert len(runtime.starts()) == 2


@pytest.mark.parametrize("kind", ["m3_narrative", "m3_voice", "m3_voice_clone", "tyrano_export"])
def test_nonimage_releases_before_other_gpu_work(runtime, kind):
    with image_session.reuse_image_runtime() as session:
        runtime.run(session, runtime.command("image"))
        prepare_media_job(kind)
        assert not runtime.held
        with image_session.gpu_scope(kind, runtime.lease("other")):
            assert runtime.held


def test_release_model_before_llm_preserves_exclusive_lease(runtime):
    with image_session.reuse_image_runtime() as session:
        runtime.run(session, runtime.command("first"))
        previous = session._process
        session.release_model()
        assert previous.poll() is not None
        assert runtime.held
        runtime.run(session, runtime.command("after-recovery"))
        assert len(runtime.starts()) == 2
        assert runtime.events == [("acquire", "image")]


@pytest.mark.parametrize("behavior", ["error", "exit", "timeout", "malformed", "incomplete"])
def test_failed_request_closes_child_and_lease_and_allows_retry(runtime, behavior):
    with image_session.reuse_image_runtime() as session:
        runtime.run(session, runtime.command("warmup"))
        previous = session._process
        with pytest.raises((RuntimeError, ValueError, TimeoutError)):
            runtime.run(session, runtime.command("failed", prompt=behavior), timeout=0.2 if behavior == "timeout" else 10)
        assert previous.poll() is not None
        assert not runtime.held
        assert session._process is session._lease is None
        runtime.run(session, runtime.command("retry"))
        assert len(runtime.starts()) == 2


def test_cancelled_request_kills_only_owned_child_and_releases_gpu(runtime):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        with image_session.reuse_image_runtime() as session, cancellation_scope() as token:
            runtime.run(session, runtime.command("warmup"))
            previous = session._process
            timer = threading.Timer(0.05, token.cancel)
            timer.start()
            try:
                with pytest.raises(GenerationCancelled):
                    runtime.run(session, runtime.command("cancelled", prompt="timeout"))
            finally:
                timer.cancel()
                timer.join()
            assert previous.poll() is not None
            assert unrelated.poll() is None
            assert not runtime.held
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_voice_and_image_preparation_prevents_overlapping_gpu_leases(runtime):
    with voice_session.reuse_voice_runtime() as voices, image_session.reuse_image_runtime() as images:
        with voices.gpu_scope(runtime.lease("voice")):
            pass
        prepare_media_job("m3_image")
        assert not runtime.held
        runtime.run(images, runtime.command("portrait"))
        prepare_media_job("m3_voice")
        assert not runtime.held
        with voices.gpu_scope(runtime.lease("voice")):
            pass
    assert runtime.events == [("acquire", "voice"), ("release", "voice"),
                              ("acquire", "image"), ("release", "image"),
                              ("acquire", "voice"), ("release", "voice")]


def test_disabled_session_keeps_standalone_subprocess_path(runtime, monkeypatch):
    calls = []
    monkeypatch.setattr(image_session, "run_process", lambda *args, **kwargs: calls.append((args, kwargs)))
    with image_session.reuse_image_runtime(enabled=False) as session:
        assert session is None
        lease = runtime.lease()
        assert image_session.gpu_scope("m3_background", lease) is lease
        image_session.run_image_process([], runtime.root / "runtime.log", cwd=runtime.root, timeout=10)
    assert len(calls) == 1


def test_portrait_failure_reacquires_lease_and_releases_weights_before_llm(runtime, monkeypatch):
    from services.worker.generation import portrait_recovery

    generated, diagnoses = [], []
    monkeypatch.setattr(portrait_recovery, "gpu_lock", lambda *args: runtime.lease("diagnosis"))
    payload = {"character_id": "traveller", "character_result": {"id": "traveller", "appearance": "A visible traveller."}}

    @contextmanager
    def llm(*args):
        assert runtime.held
        assert session._process is None
        diagnoses.append(True)
        yield type("LLM", (), {"trace": [], "structured": lambda *args: {
            "action": "repair_prompt", "reason": "Fix the composition.", "source_field": "appearance",
            "source_quote": payload["character_result"]["appearance"], "revised_prompt": "repaired",
        }})()

    monkeypatch.setattr(portrait_recovery, "LocalLLM", llm)
    with image_session.reuse_image_runtime() as session:
        runtime.run(session, runtime.command("warmup"))

        def generate(payload, prompt, work, config):
            generated.append(prompt)
            command = runtime.command(f"render-{len(generated)}", prompt="error" if len(generated) == 1 else prompt)
            try:
                runtime.run(session, command)
            except RuntimeError as error:
                raise portrait_recovery.PortraitRenderError("image_generation", str(error)) from error
            return (runtime.root / f"render-{len(generated)}" / "character.png").read_bytes(), {}

        options = {"root": runtime.root, "generate": generate}
        result = portrait_recovery.generate_with_recovery(payload, "original", runtime.root,
                                                          {"gpu_lock_timeout_seconds": 2}, **options)
        assert result[0] == b"cutout"
        assert generated == ["original", "repaired"]
        assert diagnoses == [True]
        assert runtime.events == [("acquire", "image"), ("release", "image"), ("acquire", "diagnosis")]


def test_late_worker_revocation_after_image_result_releases_resident_gpu(runtime):
    from tests.unit.test_worker_cancellation import InterruptedProtocol, run_worker

    protocol = InterruptedProtocol()
    protocol.job["kind"] = "m2_image"
    with image_session.reuse_image_runtime() as session:
        def generate(_job, _directory):
            runtime.run(session, runtime.command("finished"))
            protocol.interrupt.set()
            assert protocol.observed.wait(2)
            for thread in threading.enumerate():
                if thread.name == "worker-heartbeat":
                    thread.join(2)
            return b"completed too late"

        assert run_worker(protocol, generate, runtime.root) == "stale"
        assert session._process is session._lease is None
        assert not runtime.held
        assert protocol.completions == protocol.failures == []


def test_old_cancellation_token_cannot_kill_reused_child_for_next_image(runtime):
    with image_session.reuse_image_runtime() as session:
        with cancellation_scope() as old:
            runtime.run(session, runtime.command("first"))
        previous = session._process
        with cancellation_scope():
            old.cancel()
            assert previous.poll() is None
            runtime.run(session, runtime.command("second"))
        assert session._process is previous
        assert len(runtime.starts()) == 1
