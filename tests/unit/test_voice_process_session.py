"""Exercise resident voice ownership with real, lightweight child processes."""
from __future__ import annotations

import json
import subprocess
import sys
from contextlib import contextmanager
from pathlib import Path

import httpx
import pytest

from packages.contracts.tts_profile import build_tts_profile
from packages.tyrano_export import demo_content
from services.worker.client import WorkerClient
from services.worker.generation import voice_session

FAKE_SERVER = r'''
import json
import os
import sys
import time
from pathlib import Path

root = Path(__file__).parent
with (root / "starts.jsonl").open("a", encoding="utf-8") as stream:
    stream.write(json.dumps({"pid": os.getpid()}) + "\n")
for line in sys.stdin:
    output = Path(json.loads(line)["output_dir"])
    request = json.loads((output / "request.json").read_text(encoding="utf-8"))
    behavior = request.get("behavior", "success")
    if behavior == "malformed":
        print("{invalid json", flush=True)
        continue
    if behavior == "error":
        print(json.dumps({"ok": False, "error": "intentional synthesis failure"}), flush=True)
        continue
    if behavior == "exit":
        raise SystemExit(2)
    if behavior == "timeout":
        while True:
            time.sleep(0.02)
    report = {"pid": os.getpid(), "request": request}
    (output / "result.json").write_text(json.dumps(report), encoding="utf-8")
    if behavior != "incomplete":
        (output / "voice.wav").write_bytes(request.get("text", "音声").encode("utf-8"))
    print(json.dumps({"ok": True}), flush=True)
'''


class RuntimeProbe:
    def __init__(self, root: Path):
        self.root = root
        self.script = root / "fake_voice.py"
        self.script.write_text(FAKE_SERVER, encoding="utf-8")
        self.events = []
        self.processes = []
        self.held = False

    @contextmanager
    def lease(self, name="voice"):
        assert not self.held, "GPU leases must not overlap"
        self.held = True
        self.events.append(("acquire", name))
        try:
            yield
        finally:
            # Releasing the GPU before the child exits could overlap with an LLM.
            assert all(process.poll() is not None for process in self.processes)
            self.events.append(("release", name))
            self.held = False

    def output(self, name, **request):
        output = self.root / name
        output.mkdir()
        (output / "request.json").write_text(
            json.dumps({"model_precision": "fp32", "text": name, **request}),
            encoding="utf-8",
        )
        return output

    def run(self, session, output, *, timeout=10):
        with session.gpu_scope(self.lease()):
            session.run(
                [sys.executable, str(self.script), "--output-dir", str(output)],
                output / "runtime.log", cwd=self.root, timeout=timeout,
            )
            if session._process not in self.processes:
                self.processes.append(session._process)
        return json.loads((output / "result.json").read_text(encoding="utf-8"))

    def starts(self):
        return [json.loads(line) for line in (self.root / "starts.jsonl").read_text().splitlines()]


@pytest.fixture
def runtime(tmp_path):
    return RuntimeProbe(tmp_path)


def test_consecutive_requests_reuse_process_and_keep_outputs_separate(runtime):
    with voice_session.reuse_voice_runtime() as session:
        outputs = [
            runtime.output("design", mode="design", caption="落ち着いた女性", text="自己紹介"),
            runtime.output("clone-a", mode="clone", reference_sha256="a" * 64, text="台詞A"),
            runtime.output("clone-b", mode="clone", reference_sha256="b" * 64, text="台詞B"),
        ]
        reports = [runtime.run(session, output) for output in outputs]
        assert len({report["pid"] for report in reports}) == 1
        assert len(runtime.starts()) == 1
        assert reports[0]["request"]["caption"] == "落ち着いた女性"
        assert "caption" not in reports[1]["request"]
        assert reports[1]["request"]["reference_sha256"] == "a" * 64
        assert reports[2]["request"]["reference_sha256"] == "b" * 64
        assert [output.joinpath("voice.wav").read_bytes().decode() for output in outputs] == [
            "自己紹介", "台詞A", "台詞B",
        ]
        assert runtime.events == [("acquire", "voice")]
        assert runtime.processes[0].poll() is None
    assert runtime.events == [("acquire", "voice"), ("release", "voice")]
    assert voice_session.current_session() is None


def test_precision_change_restarts_child_under_same_lease(runtime):
    with voice_session.reuse_voice_runtime() as session:
        runtime.run(session, runtime.output("first"))
        previous = runtime.processes[0]
        runtime.run(session, runtime.output("second", model_precision="bf16"))
        assert previous.poll() is not None
        assert len(runtime.processes) == len(runtime.starts()) == 2
        assert runtime.events == [("acquire", "voice")]


def test_design_small_to_clone_large_restarts_and_exits_previous_child(runtime):
    choices = {
        "voice_design": {"provider_id": "irodori", "model_id": "irodori-v4.1-small", "precision": "bf16"},
        "voice_clone": {"provider_id": "irodori", "model_id": "irodori-v4-large", "precision": "bf16"},
    }
    profile = build_tts_profile(choices)
    with voice_session.reuse_voice_runtime() as session:
        first = runtime.run(session, runtime.output("design", mode="design", num_steps=40,
                                                   tts_profile=profile["voice_design"]))
        previous = runtime.processes[0]
        second = runtime.run(session, runtime.output("clone", mode="clone", num_steps=40,
                                                    tts_profile=profile["voice_clone"]))
        assert previous.poll() is not None
        assert first["pid"] != second["pid"]
        assert len(runtime.starts()) == 2
        assert runtime.events == [("acquire", "voice")]
        third = runtime.run(session, runtime.output("clone2", mode="clone", num_steps=40,
                                                   tts_profile=profile["voice_clone"]))
        assert third["pid"] == second["pid"]
        assert len(runtime.starts()) == 2


@pytest.mark.parametrize("kind", ["m2_image", "m3_narrative", "tyrano_export"])
def test_nonvoice_preparation_stops_child_before_other_gpu_lease(runtime, kind):
    with voice_session.reuse_voice_runtime() as session:
        runtime.run(session, runtime.output("first"))
        voice_session.prepare_job(kind)
        assert session._process is None
        assert not runtime.held
        with voice_session.gpu_scope(kind, runtime.lease("other")):
            assert runtime.held
    assert runtime.events == [
        ("acquire", "voice"), ("release", "voice"),
        ("acquire", "other"), ("release", "other"),
    ]


def test_explicit_idle_close_is_idempotent_and_next_voice_reacquires(runtime):
    with voice_session.reuse_voice_runtime() as session:
        runtime.run(session, runtime.output("first"))
        session.close()
        session.close()
        assert not runtime.held
        runtime.run(session, runtime.output("after-idle"))
        assert len(runtime.starts()) == 2
    assert runtime.events == [
        ("acquire", "voice"), ("release", "voice"),
        ("acquire", "voice"), ("release", "voice"),
    ]


def test_keyboard_interrupt_exits_child_and_resets_active_session(runtime):
    with pytest.raises(KeyboardInterrupt), voice_session.reuse_voice_runtime() as session:
        runtime.run(session, runtime.output("first"))
        raise KeyboardInterrupt
    assert not runtime.held
    assert voice_session.current_session() is None
    assert runtime.processes[0].poll() is not None


@pytest.mark.parametrize("behavior", ["malformed", "error", "exit", "timeout", "incomplete"])
def test_failed_request_releases_child_and_next_request_recovers(runtime, behavior):
    with voice_session.reuse_voice_runtime() as session:
        runtime.run(session, runtime.output("warmup"))
        failed_child = runtime.processes[0]
        output = runtime.output("failure", behavior=behavior)
        with pytest.raises((RuntimeError, ValueError, TimeoutError)):
            runtime.run(session, output, timeout=0.2 if behavior == "timeout" else 10)
        assert session._process is None
        assert session._lease is None
        assert not runtime.held
        assert failed_child.poll() is not None
        runtime.run(session, runtime.output("recovered"))
        assert len(runtime.starts()) == 2
        assert runtime.events[:3] == [
            ("acquire", "voice"), ("release", "voice"), ("acquire", "voice"),
        ]


def test_timeout_does_not_kill_unrelated_process(runtime):
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(20)"])
    try:
        with voice_session.reuse_voice_runtime() as session:
            runtime.run(session, runtime.output("warmup"))
            with pytest.raises(TimeoutError):
                runtime.run(session, runtime.output("timeout", behavior="timeout"), timeout=0.2)
            assert unrelated.poll() is None
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=10)


def test_long_large_voice_sequence_keeps_child_and_gpu_lease(runtime, monkeypatch):
    elapsed = [0.0]
    monkeypatch.setattr(voice_session.time, "monotonic", lambda: elapsed[0])
    choice = {"provider_id": "irodori", "model_id": "irodori-v4-large", "precision": "bf16"}
    profile = build_tts_profile({"voice_design": choice, "voice_clone": choice})["voice_clone"]
    with voice_session.reuse_voice_runtime() as session:
        for index, seconds in enumerate((0, 45, 65, 125, 600)):
            elapsed[0] = seconds
            runtime.run(session, runtime.output(f"line-{index}", mode="clone", num_steps=40,
                                               tts_profile=profile))
        assert len(runtime.starts()) == 1
        assert runtime.processes[0].poll() is None
        assert runtime.events == [("acquire", "voice")]
    assert runtime.processes[0].poll() is not None
    assert runtime.events == [("acquire", "voice"), ("release", "voice")]


def test_disabled_or_absent_session_preserves_original_lease(runtime):
    assert voice_session.current_session() is None
    lease = runtime.lease()
    assert voice_session.gpu_scope("m2_voice", lease) is lease
    with voice_session.reuse_voice_runtime(enabled=False) as session:
        assert session is None
        with voice_session.gpu_scope("m3_voice_clone", lease):
            assert runtime.held
        voice_session.prepare_job("m3_image")
    assert runtime.events == [("acquire", "voice"), ("release", "voice")]


@pytest.mark.parametrize("kind", ["m2_world", "tyrano_export"])
def test_worker_before_job_hook_releases_voice_before_generation_or_export(runtime, kind):
    script, assets = demo_content()
    contents = {"script-1": script.model_dump_json().encode()}
    contents.update({asset.artifact_id: assets[asset.id] for asset in script.assets})
    executed = []

    def transport(request):
        assert not runtime.held
        executed.append("artifact")
        return httpx.Response(200, content=contents[request.url.path.split("/")[-2]])

    def generate(job, directory):
        assert not runtime.held
        executed.append("generation")
        return b"generation-result"

    class Heartbeat:
        def check(self):
            assert not runtime.held

    with voice_session.reuse_voice_runtime() as session:
        runtime.run(session, runtime.output("previous-voice"))
        with httpx.Client(base_url="http://coordinator", transport=httpx.MockTransport(transport)) as c:
            worker = WorkerClient(
                c, generation_runner=generate, generation_kinds=["m2_world"],
                before_job=voice_session.prepare_job, work_dir=runtime.root / "jobs",
            )
            result = worker._execute({
                "id": "next-job", "kind": kind,
                "payload": {"script_artifact_id": "script-1"},
            }, Heartbeat())
        assert result
        assert executed
        assert not runtime.held
        assert runtime.processes[0].poll() is not None
