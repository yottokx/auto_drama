"""Regression checks for GUI startup with Explorer's reduced PATH."""

import json
import subprocess

import pytest

from scripts.audio import setup_runtime as setup


def test_uv_found_without_path(monkeypatch, tmp_path):
    uv = tmp_path / ".local/bin/uv.exe"
    uv.parent.mkdir(parents=True)
    uv.touch()
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup.Path, "home", classmethod(lambda cls: tmp_path))
    assert setup.find_uv() == str(uv)


def test_missing_uv_reports_actual_dependency(monkeypatch, tmp_path):
    monkeypatch.setattr(setup.shutil, "which", lambda name: None)
    monkeypatch.setattr(setup.Path, "home", classmethod(lambda cls: tmp_path))
    monkeypatch.setattr(setup.sys, "executable", str(tmp_path / "python.exe"))
    with pytest.raises(FileNotFoundError, match="uv が見つかりません"):
        setup.find_uv()


def test_setup_preserves_isolation_and_locked_dependencies(monkeypatch, tmp_path):
    runtime = tmp_path / "audio runtime"
    runtime.mkdir()
    (runtime / "uv.lock").touch()
    uv = str(tmp_path / "uv tools/uv.exe")
    calls = []
    monkeypatch.setattr(setup, "find_uv", lambda: uv)
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **kwargs: calls.append((command, kwargs)))
    status = tmp_path / "status"
    setup.setup_runtime(runtime, status_dir=status)
    assert calls[0][0] == [uv, "python", "install", "3.12.13", "--no-bin", "--no-registry"]
    assert calls[1][0] == [uv, "sync", "--python", "3.12.13", "--project", str(runtime), "--locked"]
    assert calls[2][0][0] == str(runtime / ".venv/Scripts/python.exe")
    assert "StableAudio3Pipeline, AceStepPipeline" in calls[2][0][2]
    for _command, kwargs in calls:
        assert kwargs["stdin"] == subprocess.DEVNULL
        assert kwargs["check"] is True
        assert kwargs["env"]["UV_PYTHON_INSTALL_DIR"] == str(runtime / ".python")
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(runtime / ".venv")
        assert kwargs["env"]["UV_PYTHON_PREFERENCE"] == "only-managed"
    assert json.loads((status / "status.json").read_text(encoding="utf-8"))["phase"] == "preparing"


def test_failure_is_reported_to_gui(monkeypatch, tmp_path):
    def fail(**kwargs):
        raise FileNotFoundError("uv is missing")
    monkeypatch.setattr(setup, "setup_runtime", fail)
    assert setup.main(["--status-dir", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False
    assert "uv is missing" in result["error"]
