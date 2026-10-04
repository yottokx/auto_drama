import json
import os
from pathlib import Path

import pytest

from scripts.audio import json_io


class Clock:
    def __init__(self):
        self.elapsed = 0.0
        self.delays = []

    def monotonic(self):
        return self.elapsed

    def sleep(self, seconds):
        self.delays.append(seconds)
        self.elapsed += seconds


def windows_error(code):
    error = PermissionError("Windows destination is locked")
    error.winerror = code
    return error


def test_json_is_complete_before_atomic_publication_and_temps_are_unique(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    path.write_text('{"step": 0}', encoding="utf-8")
    legacy_temp = tmp_path / "status.json.tmp"
    legacy_temp.write_text("unrelated writer", encoding="utf-8")
    replace = os.replace
    sources = []

    def inspect_and_replace(source, destination):
        source = Path(source)
        sources.append(source)
        previous = json.loads(path.read_text(encoding="utf-8"))["step"]
        pending = json.loads(source.read_text(encoding="utf-8"))
        assert pending == {"step": previous + 1, "message": "音楽を生成しています。"}
        assert source.parent == destination.parent
        replace(source, destination)

    monkeypatch.setattr(json_io.os, "replace", inspect_and_replace)
    for step in (1, 2):
        json_io.write_json(path, {"step": step, "message": "音楽を生成しています。"})
    assert len(set(sources)) == 2
    assert all(not source.exists() for source in sources)
    assert legacy_temp.read_text(encoding="utf-8") == "unrelated writer"
    assert json.loads(path.read_text(encoding="utf-8"))["step"] == 2


@pytest.mark.parametrize("winerror", [5, 32, 33])
def test_transient_windows_locks_retry_and_publish_once(tmp_path, monkeypatch, winerror):
    path = tmp_path / "status.json"
    path.write_text('{"step": 0}', encoding="utf-8")
    replace = os.replace
    clock = Clock()
    sources = []

    def temporarily_locked(source, destination):
        sources.append(source)
        assert json.loads(path.read_text(encoding="utf-8")) == {"step": 0}
        if len(sources) <= 3:
            raise windows_error(winerror)
        replace(source, destination)

    monkeypatch.setattr(json_io, "time", clock)
    monkeypatch.setattr(json_io.os, "replace", temporarily_locked)
    json_io.write_json(path, {"step": 1})
    assert len(sources) == 4 and len(set(sources)) == 1
    assert clock.delays == [0.025, 0.05, 0.1]
    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 1}
    assert list(tmp_path.iterdir()) == [path]


def test_persistent_lock_raises_at_deadline_and_cleans_owned_temp(tmp_path, monkeypatch):
    path = tmp_path / "status.json"
    original = '{"step": 0}'
    path.write_text(original, encoding="utf-8")
    clock = Clock()
    sources = []
    failure = windows_error(5)

    def permanently_locked(source, destination):
        sources.append(source)
        raise failure

    monkeypatch.setattr(json_io, "time", clock)
    monkeypatch.setattr(json_io.os, "replace", permanently_locked)
    with pytest.raises(PermissionError) as caught:
        json_io.write_json(path, {"step": 1})
    assert caught.value is failure
    assert clock.elapsed == pytest.approx(json_io._REPLACE_TIMEOUT)
    assert 1 < len(sources) <= 15
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.parametrize("winerror", [None, 2])
def test_other_filesystem_errors_raise_without_retry(tmp_path, monkeypatch, winerror):
    path = tmp_path / "status.json"
    clock = Clock()
    attempts = []

    def rejected(source, destination):
        attempts.append(source)
        raise windows_error(winerror)

    monkeypatch.setattr(json_io, "time", clock)
    monkeypatch.setattr(json_io.os, "replace", rejected)
    with pytest.raises(PermissionError):
        json_io.write_json(path, {"step": 1})
    assert len(attempts) == 1 and not clock.delays
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), object()])
def test_invalid_json_preserves_previous_file_without_creating_temp(tmp_path, invalid):
    path = tmp_path / "status.json"
    original = '{"step": 0}'
    path.write_text(original, encoding="utf-8")
    with pytest.raises((ValueError, TypeError)):
        json_io.write_json(path, {"invalid": invalid})
    assert path.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [path]


@pytest.mark.skipif(os.name != "nt", reason="Windows reader sharing semantics")
def test_real_windows_reader_lock_is_retried_until_reader_closes(tmp_path, monkeypatch):
    import ctypes
    from concurrent.futures import ThreadPoolExecutor
    from ctypes import wintypes
    from threading import Event

    path = tmp_path / "status.json"
    path.write_text('{"step": 0}', encoding="utf-8")
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR, wintypes.DWORD, wintypes.DWORD, wintypes.LPVOID,
        wintypes.DWORD, wintypes.DWORD, wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    # Allow other reads/writes but hold off replacement until this reader closes.
    handle = kernel32.CreateFileW(str(path), 0x80000000, 0x3, None, 3, 0x80, None)
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    blocked = Event()
    replace = os.replace

    def signal_real_lock(source, destination):
        try:
            replace(source, destination)
        except OSError as exc:
            if getattr(exc, "winerror", None) in {5, 32, 33}:
                blocked.set()
            raise

    monkeypatch.setattr(json_io.os, "replace", signal_real_lock)
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(json_io.write_json, path, {"step": 1})
        try:
            assert blocked.wait(timeout=2), "Expected an actual Windows sharing violation"
        finally:
            kernel32.CloseHandle(handle)
        future.result(timeout=2)
    assert json.loads(path.read_text(encoding="utf-8")) == {"step": 1}
    assert list(tmp_path.iterdir()) == [path]
