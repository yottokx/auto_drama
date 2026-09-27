"""Revoked attempts stop only their own lightweight real subprocesses."""
from __future__ import annotations

import subprocess
import sys
import threading
import time

import httpx
import pytest

from services.worker.client import WorkerClient
from services.worker.generation.cancellation import (
    CancellationToken,
    GenerationCancelled,
    cancellation_scope,
    check_cancelled,
    register_cancel_callback,
)
from services.worker.generation.processes import gpu_lock, owned_process
from services.worker.generation.voice_session import reuse_voice_runtime
from tests.unit.test_m2_worker import GenerationProtocol
from tests.unit.test_voice_process_session import RuntimeProbe


class InterruptedProtocol(GenerationProtocol):
    def __init__(self, *, disconnected=False):
        super().__init__()
        self.interrupt = threading.Event()
        self.observed = threading.Event()
        self.disconnected = disconnected

    def __call__(self, request):
        if request.url.path.endswith('/heartbeat') and self.interrupt.is_set():
            self.observed.set()
            if self.disconnected:
                raise httpx.ReadError('Coordinator connection dropped', request=request)
            return httpx.Response(409, json={'detail': 'Attempt was interrupted'})
        return super().__call__(request)


def run_worker(protocol, generate, tmp_path):
    with httpx.Client(base_url='http://coordinator', transport=httpx.MockTransport(protocol)) as client:
        return WorkerClient(client, generation_runner=generate, work_dir=tmp_path,
                            heartbeat_interval=0.01).run_once()


def test_revoked_heartbeat_stops_owned_child_releases_gpu_and_discards_result(tmp_path):
    protocol = InterruptedProtocol()
    owned = []
    unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])

    def generate(_job, directory):
        with gpu_lock(tmp_path / 'gpu.lock', 10), owned_process(
            [sys.executable, '-c', 'import time; time.sleep(30)'],
            directory / 'child.log', cwd=directory, timeout=30,
        ) as process:
            owned.append(process)
            protocol.interrupt.set()
            process.wait(timeout=3)
        return b'obsolete result'

    try:
        start = time.monotonic()
        assert run_worker(protocol, generate, tmp_path) == 'stale'
        assert time.monotonic() - start < 5
        assert owned[0].poll() is not None
        assert unrelated.poll() is None
        with gpu_lock(tmp_path / 'gpu.lock', 0.5):
            pass
        assert protocol.completions == protocol.failures == []
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=5)


def test_transport_failure_does_not_revoke_or_kill_the_child(tmp_path):
    protocol = InterruptedProtocol(disconnected=True)
    exit_codes = []

    def generate(_job, directory):
        with owned_process(
            [sys.executable, '-c', 'import time; time.sleep(0.3)'],
            directory / 'child.log', cwd=directory, timeout=5,
        ) as process:
            protocol.interrupt.set()
            assert protocol.observed.wait(2)
            assert process.poll() is None
            exit_codes.append(process.wait(timeout=3))
        return b'uncertain result'

    assert run_worker(protocol, generate, tmp_path) == 'deferred'
    assert exit_codes == [0]
    assert protocol.completions == protocol.failures == []


def test_revoked_voice_request_closes_resident_runtime_and_gpu_lease(tmp_path):
    runtime = RuntimeProbe(tmp_path)
    protocol = InterruptedProtocol()
    protocol.job['kind'] = 'm2_voice'
    with reuse_voice_runtime() as session:
        runtime.run(session, runtime.output('warmup'))
        process = session._process

        def generate(_job, _directory):
            output = runtime.output('interrupted', behavior='timeout')
            protocol.interrupt.set()
            runtime.run(session, output, timeout=30)
            return b'never upload'

        assert run_worker(protocol, generate, tmp_path) == 'stale'
        assert process.poll() is not None
        assert session._process is None
        assert session._lease is None
        assert not runtime.held
        assert protocol.completions == protocol.failures == []
        runtime.run(session, runtime.output('next-job'))
        assert session._process is not process


def test_late_revocation_after_voice_result_still_releases_gpu(tmp_path):
    runtime = RuntimeProbe(tmp_path)
    protocol = InterruptedProtocol()
    protocol.job['kind'] = 'm2_voice'
    with reuse_voice_runtime() as session:

        def generate(_job, _directory):
            runtime.run(session, runtime.output('done'))
            protocol.interrupt.set()
            assert protocol.observed.wait(2)
            for thread in threading.enumerate():
                if thread.name == 'worker-heartbeat':
                    thread.join(2)
            return b'completed too late'

        assert run_worker(protocol, generate, tmp_path) == 'stale'
        assert session._lease is None
        assert not runtime.held
        assert protocol.completions == protocol.failures == []


def test_old_token_cannot_kill_voice_runtime_reused_by_next_job(tmp_path):
    runtime = RuntimeProbe(tmp_path)
    with reuse_voice_runtime() as session:
        with cancellation_scope() as old:
            runtime.run(session, runtime.output('first'))
        process = session._process
        with cancellation_scope():
            old.cancel()
            assert process.poll() is None
            runtime.run(session, runtime.output('second'))
        assert session._process is process
        assert len(runtime.starts()) == 1


def test_cancelled_gpu_wait_does_not_release_another_jobs_lock(tmp_path):
    cancellation = CancellationToken()
    started = threading.Event()
    errors = []

    def wait_for_gpu():
        try:
            with cancellation_scope(cancellation):
                started.set()
                with gpu_lock(tmp_path / 'gpu.lock', 30):
                    pytest.fail('The cancelled job must not acquire the GPU')
        except GenerationCancelled as error:
            errors.append(error)

    with gpu_lock(tmp_path / 'gpu.lock', 2):
        thread = threading.Thread(target=wait_for_gpu)
        thread.start()
        assert started.wait(2)
        cancellation.cancel()
        thread.join(2)
        assert not thread.is_alive()
        assert len(errors) == 1
    with gpu_lock(tmp_path / 'gpu.lock', 0.5):
        pass


def test_callbacks_are_removed_before_scope_exit_and_nested_jobs_are_isolated():
    calls = []
    with cancellation_scope() as first:
        with (register_cancel_callback(lambda: calls.append('first')),
              cancellation_scope() as second,
              register_cancel_callback(lambda: calls.append('second'))):
            first.cancel()
            check_cancelled()
            assert calls == ['first']
            second.cancel()
            with pytest.raises(GenerationCancelled):
                check_cancelled()
        with pytest.raises(GenerationCancelled):
            check_cancelled()
    check_cancelled()
    assert calls == ['first', 'second']


def test_cancelled_attempt_does_not_spawn_a_new_subprocess(tmp_path, monkeypatch):
    monkeypatch.setattr(subprocess, 'Popen', lambda *args, **kwargs: pytest.fail('Unexpected child'))
    with cancellation_scope() as cancellation:
        cancellation.cancel()
        with pytest.raises(GenerationCancelled), owned_process(
            ['never-start'], tmp_path / 'child.log', cwd=tmp_path, timeout=1,
        ):
            pass
    assert not (tmp_path / 'child.log').exists()
