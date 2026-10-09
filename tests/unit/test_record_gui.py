"""Exercise the project dropdown and its launch path with a stub coordinator."""

import time
import tkinter

import pytest

from scripts.record import gui
from scripts.record import record_player as recorder


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


def test_viewing_options_reject_values_the_recorder_would_refuse():
    defaults = {option: default for _, option, _, _, _, default in gui.NUMBERS}
    options = gui.viewing_options("標準", "ゴシック", defaults)
    assert recorder.viewing_settings(recorder.parser().parse_args(["--url", "http://x/", *options])) == {
        "size": 24, "face": "gothic", "speed": 62, "wait": 14, "voice": 100, "bgm": 100, "alpha": 80}
    for option, value in (("--text-speed", "101"), ("--auto-wait", "0.1"), ("--voice-volume", "abc"),
                          ("--bgm-volume", "-1"), ("--window-opacity", "20"), ("--text-speed", "1.5")):
        with pytest.raises(ValueError, match="数値で指定"):
            gui.viewing_options("標準", "ゴシック", {**defaults, option: value})
    with pytest.raises(ValueError, match="選択"):
        gui.viewing_options("特大", "ゴシック", defaults)


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
        app.text_size.set("大")
        app.typeface.set("明朝")
        app.numbers["--bgm-volume"].set("40")
        app.start()
        spin(root, lambda: app.process is not None)
        command, kwargs = next(item for item in calls if isinstance(item, tuple))
        assert command[command.index("--url") + 1] == "http://127.0.0.1:8000/player/build-1/"
        output = gui.Path(command[command.index("--output-dir") + 1])
        assert output.parent == tmp_path
        assert kwargs["creationflags"] == gui.subprocess.CREATE_NO_WINDOW
        # The chosen viewing settings reach the recorder, and it accepts them as given.
        assert command[command.index("--text-size") + 1] == "large"
        assert command[command.index("--typeface") + 1] == "mincho"
        assert command[command.index("--bgm-volume") + 1] == "40"
        assert command[command.index("--auto-wait") + 1] == "1.4"
        parsed = recorder.parser().parse_args(command[command.index("--url"):])
        assert recorder.viewing_settings(parsed) == {"size": 28, "face": "mincho", "speed": 62, "wait": 14,
                                                     "voice": 100, "bgm": 40, "alpha": 80}

        # Stop is only available while recording. It waits for the recorder's own folder,
        # then leaves the request there; the recorder's final status re-enables Start.
        assert str(app.stop_button["state"]) == "normal" and str(app.start_button["state"]) == "disabled"
        app.stop()
        assert str(app.stop_button["state"]) == "disabled" and not output.exists()
        output.mkdir()
        (output / "status.json").write_text('{"status": "running", "phase": "recording"}', "utf-8")
        app.update_progress()
        assert (output / "stop.request").exists()
        assert "停止しています" in app.status.get()
        (output / "status.json").write_text(gui.json.dumps(
            {"status": "stopped", "phase": "stopped", "output": str(output / "recording.mp4")}), "utf-8")
        app.update_progress()
        assert "録画を停止しました" in app.status.get() and "recording.mp4" in app.status.get()
        assert app.process is None and str(app.start_button["state"]) == "normal"
        assert str(app.stop_button["state"]) == "disabled"
    finally:
        root.destroy()
