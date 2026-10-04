import json
import subprocess

from scripts.image_edit import setup_runtime as setup


def test_qwen_setup_preserves_dedicated_runtime_and_locked_dependencies(tmp_path, monkeypatch):
    runtime = tmp_path / "Qwen runtime"
    runtime.mkdir()
    (runtime / "uv.lock").touch()
    calls = []
    monkeypatch.setattr(setup, "find_uv", lambda: "uv.exe")
    monkeypatch.setattr(setup.subprocess, "run", lambda command, **kw: calls.append((command, kw)))
    setup.setup_runtime(runtime, status_dir=tmp_path / "status")
    assert calls[0][0] == ["uv.exe", "python", "install", "3.12.13", "--no-bin", "--no-registry"]
    assert calls[1][0] == ["uv.exe", "sync", "--python", "3.12.13", "--project", str(runtime), "--locked"]
    assert "QwenImage21Pipeline" in calls[2][0][-1]
    for _, kwargs in calls:
        assert kwargs["env"]["UV_PROJECT_ENVIRONMENT"] == str(runtime / ".venv")
        assert kwargs["env"]["UV_PYTHON_INSTALL_DIR"] == str(runtime / ".python")
        assert kwargs["stdin"] == subprocess.DEVNULL and kwargs["check"]


def test_setup_failure_writes_gui_result(tmp_path, monkeypatch):
    def failed(**kwargs):
        raise FileNotFoundError("uv unavailable")

    monkeypatch.setattr(setup, "setup_runtime", failed)
    assert setup.main(["--status-dir", str(tmp_path)]) == 1
    result = json.loads((tmp_path / "result.json").read_text(encoding="utf-8"))
    assert result["ok"] is False and "uv unavailable" in result["error"]
