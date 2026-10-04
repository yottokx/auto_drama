"""Idle polling preserves resident models and still services their deadlines."""
from contextlib import contextmanager

import pytest

from services.worker import __main__ as worker_main


class Runtime:
    def __init__(self):
        self.loaded = True
        self.expirations = 0
        self.closes = 0

    def expire_if_needed(self):
        self.expirations += 1

    def close(self):
        self.loaded = False
        self.closes += 1


@pytest.mark.parametrize("waiting_result", ["idle", "deferred"])
def test_worker_reuses_models_after_waiting_and_closes_on_exit(monkeypatch, tmp_path, waiting_result):
    runtimes = [Runtime() for _ in range(3)]

    @contextmanager
    def scope(runtime, **kwargs):
        try:
            yield runtime
        finally:
            runtime.close()

    monkeypatch.setattr(worker_main, "reuse_voice_runtime", lambda **kw: scope(runtimes[0], **kw))
    monkeypatch.setattr(worker_main.image_session, "reuse_image_runtime", lambda **kw: scope(runtimes[1], **kw))
    monkeypatch.setattr(worker_main.llm_session, "reuse_llm_runtime", lambda **kw: scope(runtimes[2], **kw))
    monkeypatch.setattr(worker_main.sys, "argv", [
        "worker", "--export-only", "--poll-interval", "2.5", "--work-dir", str(tmp_path)
    ])
    sleeps = []
    monkeypatch.setattr(worker_main.time, "sleep", sleeps.append)
    outcomes = iter(["completed", waiting_result, "completed"])

    class Worker:
        def __init__(self, *args, **kwargs):
            pass

        def run_once(self):
            assert all(runtime.loaded for runtime in runtimes)
            try:
                return next(outcomes)
            except StopIteration:
                raise KeyboardInterrupt from None

    monkeypatch.setattr(worker_main, "WorkerClient", Worker)
    assert worker_main.main() == 0
    assert sleeps == [1.0, 1.0, 0.5]
    assert all(runtime.expirations >= 3 and runtime.closes == 1 for runtime in runtimes)


@pytest.mark.parametrize("result", ["failed", "stale"])
def test_failed_or_revoked_attempt_releases_all_models(result):
    runtimes = [Runtime() for _ in range(3)]
    worker_main.maintain_runtimes(result, *runtimes, None)
    assert all(not runtime.loaded and runtime.closes == 1 for runtime in runtimes)
