"""A worker owns its directory before registering, without stale PID markers."""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from services.worker import __main__ as worker_main
from services.worker.process_lock import WorkerAlreadyRunning, worker_process_lock

ROOT = Path(__file__).resolve().parents[2]
LOCK_ATTEMPT = """
import sys
from pathlib import Path
from services.worker.process_lock import WorkerAlreadyRunning, worker_process_lock
try:
    with worker_process_lock(Path(sys.argv[1])):
        pass
except WorkerAlreadyRunning:
    sys.exit(3)
"""
LOCK_HOLDER = """
import sys
from pathlib import Path
from services.worker.process_lock import worker_process_lock
with worker_process_lock(Path(sys.argv[1])):
    Path(sys.argv[2]).write_text('ready', encoding='utf-8')
    sys.stdin.read()
"""


def attempt(work_dir: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-c", LOCK_ATTEMPT, str(work_dir)], cwd=ROOT,
        capture_output=True, text=True, timeout=10, check=False,
    )


def test_same_directory_rejected_but_separate_directory_allowed(tmp_path):
    with worker_process_lock(tmp_path / "first"):
        duplicate = attempt(tmp_path / "first" / ".." / "first")
        assert duplicate.returncode == 3, duplicate.stderr
        separate = attempt(tmp_path / "second")
        assert separate.returncode == 0, separate.stderr
    assert attempt(tmp_path / "first").returncode == 0


@pytest.mark.parametrize("force_exit", [False, True], ids=["normal-exit", "killed-owner"])
def test_child_exit_releases_lock_even_when_file_remains(tmp_path, force_exit):
    work_dir, ready = tmp_path / "work", tmp_path / "ready"
    child = subprocess.Popen(
        [sys.executable, "-c", LOCK_HOLDER, str(work_dir), str(ready)], cwd=ROOT,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        deadline = time.monotonic() + 10
        while not ready.exists() and child.poll() is None and time.monotonic() < deadline:
            time.sleep(0.02)
        assert ready.exists(), "The test child did not acquire its directory lock."
        assert attempt(work_dir).returncode == 3
        if force_exit:
            child.kill()
        child.communicate(input="exit", timeout=10)
        if not force_exit:
            assert child.returncode == 0
        assert (work_dir / ".worker.lock").exists()
        assert attempt(work_dir).returncode == 0
    finally:
        if child.poll() is None:
            child.kill()
        child.communicate(timeout=10)


def test_duplicate_main_exits_before_readiness_or_registration(monkeypatch, tmp_path, caplog):
    from services.worker import generation

    def unexpected(*args, **kwargs):
        pytest.fail("Duplicate worker must exit before readiness checks or registration.")

    monkeypatch.setattr(generation, "check_readiness", unexpected)
    monkeypatch.setattr(worker_main, "WorkerClient", unexpected)
    monkeypatch.setattr(sys, "argv", ["worker", "--work-dir", str(tmp_path)])
    with worker_process_lock(tmp_path):
        assert worker_main.main() == 1
    assert "ワーカーが既に起動しています" in caplog.text
    assert "Ctrl+C" in caplog.text


@pytest.mark.parametrize("interrupt", [False, True])
def test_main_holds_lock_until_worker_exits(monkeypatch, tmp_path, interrupt):
    def run(args, root):
        assert args.work_dir == tmp_path.resolve()
        with pytest.raises(WorkerAlreadyRunning), worker_process_lock(args.work_dir):
            pytest.fail("A worker must keep its process lock until shutdown.")
        if interrupt:
            raise KeyboardInterrupt
        return 0

    monkeypatch.setattr(worker_main, "run_worker", run)
    monkeypatch.setattr(sys, "argv", ["worker", "--work-dir", str(tmp_path)])
    assert worker_main.main() == 0
    assert attempt(tmp_path).returncode == 0


def test_default_work_directory_matches_generation_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(worker_main, "__file__", str(tmp_path / "services/worker/__main__.py"))
    monkeypatch.setattr(sys, "argv", ["worker", "--export-only"])

    def run(args, root):
        assert root == tmp_path
        assert args.work_dir == tmp_path / "services/worker/cache/m2/jobs"
        with pytest.raises(WorkerAlreadyRunning), worker_process_lock(args.work_dir):
            pytest.fail("The default work directory must also be protected.")
        return 0

    monkeypatch.setattr(worker_main, "run_worker", run)
    assert worker_main.main() == 0
