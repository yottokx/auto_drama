"""Exercise the project dropdown and its launch path with a stub coordinator."""

import time
import tkinter

import pytest

from scripts.record import gui


def test_progress_clock_continues_during_mux_but_freezes_when_completed():
    state = {"status": "running", "phase": "muxing_chapter", "elapsed_seconds": 165,
             "updated_at_epoch": 1000, "recorded_seconds": 150, "current_chapter": 2,
             "utterance_current": 10, "utterance_total": 100}
    timing, position = gui.progress_text(state, now=1005)
    assert timing == "録画時間 00:02:30　実行時間 00:02:50"
    assert position == "録画2章目　台詞・地の文 （10/100）"
    state["status"] = "completed"
    assert gui.progress_text(state, now=2000)[0] == "録画時間 00:02:30　実行時間 00:02:45"
    assert gui.duration(3661) == "01:01:01"


def spin(root, condition):
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        root.update()
        if condition():
            return
        time.sleep(0.01)
    pytest.fail("GUI did not receive the background result")


def test_select_project_and_launch_recording(tmp_path, monkeypatch):
    try:
        root = tkinter.Tk()
    except tkinter.TclError:
        pytest.skip("Tk display is unavailable")
    root.withdraw()
    calls = []

    def fake_fetch(url):
        calls.append(url)
        if url.endswith("/api/projects"):
            return {"projects": [{"id": "project-1", "title": "テスト作品"}]}
        return {"production": {"player_url": "/player/build-1/"}}

    class Process:
        def poll(self):
            return None

    def fake_popen(command, **kwargs):
        calls.append((command, kwargs))
        return Process()

    monkeypatch.setattr(gui, "fetch_json", fake_fetch)
    monkeypatch.setattr(gui.subprocess, "Popen", fake_popen)
    try:
        app = gui.RecordingApp(root)
        spin(root, lambda: len(app.projects) == 1)
        assert "テスト作品" in app.project_box["values"][0]
        app.project_box.current(0)
        app.output_parent.set(str(tmp_path))
        app.start()
        spin(root, lambda: app.process is not None)
        command, kwargs = next(item for item in calls if isinstance(item, tuple))
        assert command[command.index("--url") + 1] == "http://127.0.0.1:8000/player/build-1/"
        assert gui.Path(command[command.index("--output-dir") + 1]).parent == tmp_path
        assert kwargs["creationflags"] == gui.subprocess.CREATE_NO_WINDOW
    finally:
        root.destroy()
