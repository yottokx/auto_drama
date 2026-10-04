"""CPU-only checks for prompt/music GPU ownership and release boundaries."""

import json
import subprocess
import sys
import textwrap
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from threading import Event

import pytest

from scripts.audio import engine, llm_runner, llm_runtime, resources, runner

PROMPT = "Gentle cinematic piano and restrained strings support the dialogue."
INTERPRETATION = "友人との再会を、安堵と親密さのある場面として捉えます。温かなピアノと弦で会話を支えます。"
DETAILS = {"prompt": PROMPT, "scene_interpretation": INTERPRETATION}


@pytest.fixture(autouse=True)
def isolate_prompt_gpu_lease(monkeypatch):
    """Unit tests must never wait on a real user's running GPU generation."""
    @contextmanager
    def lease(*args, **kwargs):
        yield

    monkeypatch.setattr(llm_runner, "music_gpu_scope", lease)


def prompt_request(tmp_path, **updates):
    path = tmp_path / "prompt-request.json"
    data = {
        "backend": "llama_cpp", "base_url": "http://api.example/v1", "model": "external-model",
        "local_model": "gemma-local", "options": {"style": "acoustic", "tempo": 72},
        "scene": {"id": "scene-1", "label": "再会", "context": {"raw_text": "友人と再会する。"}},
    }
    data.update(updates)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
    return path


def music_request(tmp_path, **updates):
    path = tmp_path / "music-request.json"
    data = {
        "model": "small", "model_path": str(tmp_path / "model"), "prompt": PROMPT,
        "device": "cuda", "duration": 30, "steps": 8,
    }
    data.update(updates)
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def result(output):
    return json.loads((output / "result.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_native_prompt_success_is_published_only_after_server_closes(tmp_path, monkeypatch, cleanup_fails):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def native(root, model, directory, *, status, cancelled):
        assert model == "gemma-local" and directory == output
        events.append("native-open")
        try:
            yield "http://127.0.0.1:32123/v1", "bgm-local"
        finally:
            assert not (output / "result.json").exists()
            events.append("native-closed")
            if cleanup_fails:
                raise RuntimeError("native cleanup failed")

    def generate(scene, url, model, **options):
        assert scene.id == "scene-1"
        assert url == "http://127.0.0.1:32123/v1" and model == "bgm-local"
        assert options == {"style": "acoustic", "tempo": 72,
                           "request_options": {"chat_template_kwargs": {"enable_thinking": False}},
                           "return_details": True}
        events.append("prompt")
        return DETAILS

    write = llm_runner.write_json

    def publish(path, value):
        if path.name == "result.json" and value.get("ok"):
            assert events[-1] == "native-closed"
        write(path, value)

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runner, "request_llm_prompt", generate)
    monkeypatch.setattr(llm_runner, "write_json", publish)
    assert llm_runner.run_request(prompt_request(tmp_path), output) == (1 if cleanup_fails else 0)
    assert events == ["native-open", "prompt", "native-closed"]
    saved = result(output)
    assert saved["ok"] is not cleanup_fails
    if cleanup_fails:
        assert "prompt" not in saved and "native cleanup failed" in saved["error"]
    else:
        assert saved["prompt"] == PROMPT and saved["llm_released"] is True
        assert saved["scene_interpretation"] == INTERPRETATION


@pytest.mark.parametrize("prompt_fails,release_fails", [(False, False), (True, False), (False, True), (True, True)])
def test_ollama_always_releases_and_failed_release_prevents_success(
    tmp_path, monkeypatch, prompt_fails, release_fails,
):
    output = tmp_path / "run"
    events = []

    def generate(scene, url, model, **options):
        assert options["request_options"] == {"reasoning_effort": "none"}
        assert options["return_details"] is True
        events.append("prompt")
        if prompt_fails:
            raise ValueError("no English final answer")
        return DETAILS

    def release(url, model):
        assert url == "http://127.0.0.1:11434/v1" and model == "gemma4-31b-16k"
        assert not (output / "result.json").exists()
        events.append("released")
        if release_fails:
            raise RuntimeError("release verification failed")

    monkeypatch.setattr(llm_runner, "request_llm_prompt", generate)
    monkeypatch.setattr(llm_runtime, "release_ollama", release)
    code = llm_runner.run_request(prompt_request(
        tmp_path, backend="ollama", base_url="http://127.0.0.1:11434/v1", model="gemma4-31b-16k",
    ), output)
    assert events == ["prompt", "released"]
    assert code == (1 if prompt_fails or release_fails else 0)
    saved = result(output)
    if code:
        assert saved["ok"] is False and "prompt" not in saved
    else:
        assert saved["ok"] is True and saved["llm_released"] is True
        assert saved["prompt"] == PROMPT
        assert saved["scene_interpretation"] == INTERPRETATION


def test_ollama_release_runs_even_if_releasing_status_cannot_be_written(tmp_path, monkeypatch):
    output = tmp_path / "run"
    released = []
    write = llm_runner.write_json

    def fail_releasing_status(path, value):
        if path.name == "status.json" and value.get("phase") == "releasing":
            raise PermissionError("persistent status access error")
        write(path, value)

    monkeypatch.setattr(llm_runner, "request_llm_prompt", lambda *args, **kwargs: PROMPT)
    monkeypatch.setattr(llm_runner, "write_json", fail_releasing_status)
    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: released.append(args))
    assert llm_runner.run_request(prompt_request(tmp_path, backend="ollama"), output) == 1
    assert len(released) == 1
    assert result(output)["ok"] is False
    assert "persistent status access error" in result(output)["error"]


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama", "openai"])
@pytest.mark.parametrize("legacy", [False, True])
def test_all_prompt_backends_publish_scene_interpretation_with_legacy_string_compatibility(
    tmp_path, monkeypatch, backend, legacy,
):
    output = tmp_path / "run"

    @contextmanager
    def native(*args, **kwargs):
        yield "http://local/v1", "bgm-local"

    def generate(*args, **kwargs):
        assert kwargs["return_details"] is True
        return "  " + PROMPT + "  " if legacy else {
            "prompt": "  " + PROMPT + "  ", "scene_interpretation": "  " + INTERPRETATION + "  ",
        }

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: None)
    monkeypatch.setattr(llm_runner, "request_llm_prompt", generate)
    assert llm_runner.run_request(prompt_request(tmp_path, backend=backend), output) == 0
    saved = result(output)
    assert saved["prompt"] == PROMPT
    assert saved["scene_interpretation"] == ("" if legacy else INTERPRETATION)
    assert saved["backend"] == backend
    assert saved["llm_released"] is (backend != "openai")


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama", "openai"])
@pytest.mark.parametrize("reply", [
    None, [], {}, {"prompt": PROMPT},
    {"prompt": " ", "scene_interpretation": INTERPRETATION},
    {"prompt": "穏やかな音楽を生成してください。", "scene_interpretation": INTERPRETATION},
    {"prompt": True, "scene_interpretation": INTERPRETATION},
    {"prompt": PROMPT, "scene_interpretation": None},
    {"prompt": PROMPT, "scene_interpretation": " "},
    {"prompt": PROMPT, "scene_interpretation": "A warm, reassuring reunion scene."},
    {"prompt": PROMPT, "scene_interpretation": "あ" * 601},
])
def test_invalid_prompt_details_fail_after_model_release_without_publishing_success(
    tmp_path, monkeypatch, backend, reply,
):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def native(*args, **kwargs):
        try:
            yield "http://local/v1", "bgm-local"
        finally:
            assert not (output / "result.json").exists()
            events.append("released")

    def release(*args):
        assert not (output / "result.json").exists()
        events.append("released")

    def generate(*args, **kwargs):
        assert kwargs["return_details"] is True
        events.append("prompt")
        return reply

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runtime, "release_ollama", release)
    monkeypatch.setattr(llm_runner, "request_llm_prompt", generate)
    assert llm_runner.run_request(prompt_request(tmp_path, backend=backend), output) == 1
    saved = result(output)
    assert saved["ok"] is False
    assert saved["error_type"] == "ValueError"
    assert "prompt" not in saved and "scene_interpretation" not in saved
    assert events == (["prompt"] if backend == "openai" else ["prompt", "released"])


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama", "openai"])
def test_existing_stop_request_prevents_prompt_or_model_start(tmp_path, monkeypatch, backend):
    output = tmp_path / "run"
    output.mkdir()
    (output / "stop.request").touch()
    monkeypatch.setattr(llm_runner, "request_llm_prompt", lambda *args, **kwargs: pytest.fail("prompt after stop"))
    monkeypatch.setattr(llm_runtime, "managed_llama", lambda *args, **kwargs: pytest.fail("native after stop"))
    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: pytest.fail("release before stop check"))
    assert llm_runner.run_request(prompt_request(tmp_path, backend=backend), output) == 130
    assert result(output)["cancelled"] is True


@pytest.mark.parametrize("backend", ["llama_cpp", "ollama"])
def test_cancel_after_prompt_releases_model_without_publishing_prompt(tmp_path, monkeypatch, backend):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def native(*args, **kwargs):
        try:
            yield "http://local/v1", "bgm-local"
        finally:
            events.append("released")

    def generate(*args, **kwargs):
        (output / "stop.request").touch()
        events.append("prompt")
        return PROMPT

    monkeypatch.setattr(llm_runtime, "managed_llama", native)
    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: events.append("released"))
    monkeypatch.setattr(llm_runner, "request_llm_prompt", generate)
    assert llm_runner.run_request(prompt_request(tmp_path, backend=backend), output) == 130
    assert events == ["prompt", "released"]
    saved = result(output)
    assert saved["ok"] is False and saved["cancelled"] is True and "prompt" not in saved


@pytest.mark.parametrize("entry", [llm_runner.run_request, runner.run_request])
def test_existing_results_are_never_overwritten_or_restarted(tmp_path, monkeypatch, entry):
    output = tmp_path / "run"
    output.mkdir()
    original = '{"ok":true,"preserve":"previous experiment"}'
    (output / "result.json").write_text(original, encoding="utf-8")
    monkeypatch.setattr(llm_runner, "request_llm_prompt", lambda *args, **kwargs: pytest.fail("prompt"))
    monkeypatch.setattr(runner, "generate_audio", lambda *args, **kwargs: pytest.fail("music"))
    assert entry(tmp_path / "nonexistent-request.json", output) == 2
    assert (output / "result.json").read_text(encoding="utf-8") == original
    assert not (output / "status.json").exists()


def test_music_releases_ollama_then_takes_shared_gpu_lock_before_generating(tmp_path, monkeypatch):
    output = tmp_path / "run"
    events = []

    def release(url, model):
        assert url == "http://127.0.0.1:11434/v1" and model == "gemma"
        events.append("release")

    @contextmanager
    def lock(root, device):
        assert device == "cuda"
        events.append("lock-enter")
        try:
            yield
        finally:
            assert not (output / "result.json").exists()
            events.append("lock-exit")

    def generate(request, directory, *, progress, cancelled):
        assert events == ["release", "lock-enter"] and not cancelled()
        events.append("generate")
        return {"ok": True, "audio_path": str(directory / "output.wav")}

    monkeypatch.setattr(llm_runtime, "release_ollama", release)
    monkeypatch.setattr(runner, "music_gpu_scope", lock)
    monkeypatch.setattr(runner, "generate_audio", generate)
    request = music_request(tmp_path, llm_release={"base_url": "http://127.0.0.1:11434/v1", "model": "gemma"})
    assert runner.run_request(request, output) == 0
    assert events == ["release", "lock-enter", "generate", "lock-exit"]
    assert result(output)["ok"] is True


def test_music_release_failure_blocks_lock_and_generation(tmp_path, monkeypatch):
    def release(*args):
        raise RuntimeError("Ollama remains in VRAM")

    monkeypatch.setattr(llm_runtime, "release_ollama", release)
    monkeypatch.setattr(runner, "music_gpu_scope", lambda *args: pytest.fail("lock acquired before release"))
    monkeypatch.setattr(runner, "generate_audio", lambda *args, **kwargs: pytest.fail("music"))
    output = tmp_path / "run"
    request = music_request(tmp_path, llm_release={"base_url": "http://localhost:11434/v1", "model": "gemma"})
    assert runner.run_request(request, output) == 1
    assert result(output)["ok"] is False and "Ollama remains" in result(output)["error"]


@pytest.mark.parametrize("when", ["before-start", "waiting-lock", "generation-return"])
def test_music_cancellation_never_publishes_success(tmp_path, monkeypatch, when):
    output = tmp_path / "run"
    output.mkdir()
    events = []
    if when == "before-start":
        (output / "stop.request").touch()

    @contextmanager
    def lock(*args):
        events.append("lock-enter")
        if when == "waiting-lock":
            (output / "stop.request").touch()
        try:
            yield
        finally:
            events.append("lock-exit")

    def generate(*args, **kwargs):
        assert when == "generation-return"
        events.append("generate")
        (output / "stop.request").touch()
        return {"ok": True}

    monkeypatch.setattr(runner, "music_gpu_scope", lock)
    monkeypatch.setattr(runner, "generate_audio", generate)
    assert runner.run_request(music_request(tmp_path), output) == 130
    assert result(output)["cancelled"] is True and result(output)["ok"] is False
    if when == "before-start":
        assert events == []
    else:
        assert events[-1] == "lock-exit"


def test_cancellation_check_reads_stop_marker_without_waiting_for_watcher(tmp_path, monkeypatch):
    class UnscheduledWatcher:
        def __init__(self, *args, **kwargs):
            pass

        def start(self):
            pass

        def join(self, timeout):
            pass

    monkeypatch.setattr(resources.threading, "Thread", UnscheduledWatcher)
    with resources.cancellation_watcher(tmp_path) as token:
        (tmp_path / "stop.request").touch()
        with pytest.raises(engine.GenerationCancelled):
            token.check()


@pytest.mark.parametrize("device", ["cpu", "auto", "cuda"])
def test_music_scope_uses_worker_gpu_lease_and_cpu_bypasses_it(tmp_path, monkeypatch, device):
    events = []

    @contextmanager
    def lock(path, timeout):
        assert path == tmp_path / "services/worker/cache/m2/gpu.lock" and timeout == 3600
        events.append("lock-enter")
        try:
            yield
        finally:
            events.append("lock-exit")

    monkeypatch.setattr(resources, "gpu_lock", lock)
    with resources.music_gpu_scope(tmp_path, device):
        events.append("work")
    assert events == (["work"] if device == "cpu" else ["lock-enter", "work", "lock-exit"])


@pytest.mark.parametrize("prompt_fails", [False, True])
def test_ollama_shared_gpu_lease_covers_request_and_release(tmp_path, monkeypatch, prompt_fails):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def lease(root, device):
        assert device == "cuda"
        events.append("lease-enter")
        try:
            yield
        finally:
            assert not (output / "result.json").exists()
            events.append("lease-exit")

    def generate(*args, **kwargs):
        assert events == ["lease-enter"]
        events.append("prompt")
        if prompt_fails:
            raise ValueError("invalid final answer")
        return PROMPT

    def release(*args):
        assert events == ["lease-enter", "prompt"]
        events.append("release")

    monkeypatch.setattr(llm_runner, "music_gpu_scope", lease)
    monkeypatch.setattr(llm_runner, "request_llm_prompt", generate)
    monkeypatch.setattr(llm_runtime, "release_ollama", release)
    assert llm_runner.run_request(prompt_request(tmp_path, backend="ollama"), output) == (1 if prompt_fails else 0)
    assert events == ["lease-enter", "prompt", "release", "lease-exit"]
    assert result(output)["ok"] is not prompt_fails


def test_ollama_cancel_while_waiting_for_gpu_prevents_request_and_releases(tmp_path, monkeypatch):
    output = tmp_path / "run"
    events = []

    @contextmanager
    def lease(*args):
        events.append("lease-enter")
        (output / "stop.request").touch()
        try:
            yield
        finally:
            events.append("lease-exit")

    monkeypatch.setattr(llm_runner, "music_gpu_scope", lease)
    monkeypatch.setattr(llm_runner, "request_llm_prompt", lambda *args, **kwargs: pytest.fail("prompt after stop"))
    monkeypatch.setattr(llm_runtime, "release_ollama", lambda *args: events.append("release"))
    assert llm_runner.run_request(prompt_request(tmp_path, backend="ollama"), output) == 130
    assert events == ["lease-enter", "release", "lease-exit"]
    assert result(output)["cancelled"] is True


def test_fresh_music_process_does_not_import_worker_or_pydantic(tmp_path):
    root = Path(__file__).resolve().parents[2]
    source = textwrap.dedent("""
        import importlib.abc
        import io
        import json
        import sys
        from pathlib import Path

        blocked = []
        class ForbidWorkerImports(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == 'services' or fullname.startswith('services.') or fullname.startswith('pydantic'):
                    blocked.append(fullname)
                    raise ImportError('Music runtime must not require ' + fullname)

        sys.meta_path.insert(0, ForbidWorkerImports())
        sys.path.insert(0, sys.argv[1])
        from scripts.audio import engine, llm_runtime, resources, runner
        directory = Path(sys.argv[2])
        with resources.cancellation_watcher(directory) as token:
            token.check()
            with resources.music_gpu_scope(directory, 'cuda'):
                assert (directory / 'services/worker/cache/m2/gpu.lock').is_file()
            assert (directory / 'services/worker/cache/m2/gpu.lock').read_bytes() == b'0'
            (directory / 'stop.request').touch()
            try:
                token.check()
            except engine.GenerationCancelled:
                pass
            else:
                raise AssertionError('stop marker was not detected')

        pending = [{'models': [{'name': 'gemma:latest'}]}, {'done': True}, {'models': []}]
        requests = []
        class FakeOpener:
            def open(self, request, timeout):
                requests.append((request.full_url, request.get_method(), request.data))
                return io.BytesIO(json.dumps(pending.pop(0)).encode())
        llm_runtime.build_opener = lambda *args: FakeOpener()
        llm_runtime.release_ollama('http://127.0.0.1:11434/v1', 'gemma')
        assert [url for url, _, _ in requests] == [
            'http://127.0.0.1:11434/api/ps', 'http://127.0.0.1:11434/api/generate',
            'http://127.0.0.1:11434/api/ps']
        assert json.loads(requests[1][2]) == {'model': 'gemma', 'keep_alive': 0, 'stream': False}
        assert not pending and not blocked, blocked
        print('stdlib music lifecycle passed')
    """)
    completed = subprocess.run(
        [sys.executable, "-I", "-c", source, str(root), str(tmp_path)],
        capture_output=True, text=True, timeout=10, check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "stdlib music lifecycle passed" in completed.stdout


def test_stdlib_gpu_lock_waits_for_existing_holder_then_acquires(tmp_path):
    path = tmp_path / "gpu.lock"
    started, acquired = Event(), Event()

    def second_holder():
        started.set()
        with resources.gpu_lock(path, timeout=2):
            acquired.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with resources.gpu_lock(path, timeout=2):
            future = executor.submit(second_holder)
            assert started.wait(timeout=1)
            assert not acquired.wait(timeout=0.05), "Second holder bypassed the existing lease"
        future.result(timeout=2)
    assert acquired.is_set()
    assert path.read_bytes() == b"0"


def test_stdlib_gpu_lock_timeout_preserves_existing_holder_and_next_acquisition(tmp_path):
    path = tmp_path / "gpu.lock"
    with (
        resources.gpu_lock(path, timeout=1),
        pytest.raises(TimeoutError, match="制限時間"),
        resources.gpu_lock(path, timeout=0.05),
    ):
        pytest.fail("Acquired an already-held lease")
    with resources.gpu_lock(path, timeout=0.05):
        pass
    assert path.read_bytes() == b"0"


def test_stdlib_gpu_wait_is_cancellable_without_releasing_other_holder(tmp_path):
    path = tmp_path / "gpu.lock"
    cancellation_dir = tmp_path / "waiting-job"
    cancellation_dir.mkdir()
    started = Event()

    def waiting_job():
        with resources.cancellation_watcher(cancellation_dir):
            started.set()
            with resources.gpu_lock(path, timeout=2):
                pytest.fail("Cancelled waiter acquired the owner's GPU lease")

    with ThreadPoolExecutor(max_workers=1) as executor:
        with resources.gpu_lock(path, timeout=1):
            future = executor.submit(waiting_job)
            assert started.wait(timeout=1)
            (cancellation_dir / "stop.request").touch()
            with pytest.raises(engine.GenerationCancelled):
                future.result(timeout=1)
        with resources.gpu_lock(path, timeout=0.05):
            pass
    assert path.read_bytes() == b"0"
