from types import SimpleNamespace

import pytest

from scripts.record import record_player as recorder
from scripts.record.record_player import player_url, successor


def test_successor_checks_lineage_and_order():
    ready = {"build_id": "first", "status": "ready",
             "next_build": {"id": "second", "chapter_number": 2}}
    assert successor(ready, "first", 1) == "second"
    with pytest.raises(ValueError):
        successor(ready, "other", 1)
    with pytest.raises(ValueError):
        successor(ready, "first", 2)
    for identifier in ["../evil", "first", "", "日本語"]:
        with pytest.raises(ValueError):
            successor({**ready, "next_build": {"id": identifier, "chapter_number": 2}}, "first", 1)
    for status in ["waiting", "complete"]:
        assert successor({"build_id": "first", "status": status}, "first", 1) is None


def test_player_url_rejects_non_http():
    assert player_url("http://127.0.0.1:8000/player/build/").endswith("/build/")
    for url in ["file:///tmp/index.html", "invalid", "http://localhost/?unexpected=1"]:
        with pytest.raises(ValueError):
            player_url(url)


@pytest.mark.parametrize("ending", ["chapter", "limit", "stop"])
def test_chapter_end_or_partial_stop_captures_one_bounded_fade_tail(tmp_path, monkeypatch, ending):
    """An indefinitely looping chapter outro cannot extend the recording."""
    probe = SimpleNamespace(now=0.0, started_at=None, fade_at=[], writes=[], events=[], waits=0)
    monkeypatch.setattr(recorder, "time", SimpleNamespace(monotonic=lambda: probe.now))

    class Locator:
        def __init__(self, selector):
            self.selector = selector

        def wait_for(self):
            pass

        def click(self):
            probe.started_at = probe.now

        def dispatch_event(self, event):
            assert event == "click"

        def evaluate(self, _expression):
            assert self.selector == "#adn-end"
            return ending == "chapter" and probe.now >= .55

    class Page:
        def goto(self, *_args, **_options):
            pass

        def wait_for_function(self, *_args, **_options):
            pass

        def add_style_tag(self, **_options):
            pass

        def expose_function(self, _name, callback):
            self.chunk = callback

        def screenshot(self, **_options):
            probe.now += .01
            return b"frame"

        def locator(self, selector):
            return Locator(selector)

        def evaluate(self, expression, argument=None):
            if expression == recorder.PLAYER_PROGRESS:
                return {"utterance_current": 1, "utterance_total": 1}
            if expression == recorder.AUDIO_START:
                self.chunk("AQI=")
                return
            if "fadeRecordingAudio" in expression:
                assert argument == 400
                probe.fade_at.append(probe.now)
                return
            assert "finishRecordingAudio" in expression
            probe.events.append("audio-finished")

        def wait_for_timeout(self, milliseconds):
            probe.waits += 1
            assert probe.waits < 30, "chapter-end BGM must not be recorded indefinitely"
            probe.now += milliseconds / 1000

    class Context:
        def add_init_script(self, script):
            probe.events.append(("settings", script))

        def new_page(self):
            return Page()

        def close(self):
            probe.events.append("context-closed")

    class Browser:
        def new_context(self, **_options):
            return Context()

    class Pipe:
        def write(self, data):
            probe.writes.append(data)

        def close(self):
            probe.events.append("video-closed")

    class Encoder:
        def __init__(self, command, **_options):
            self.stdin, self.returncode = Pipe(), 0
            recorder.Path(command[-1]).write_bytes(b"video")

        def wait(self, **_options):
            probe.events.append("video-finished")

        def poll(self):
            return 0

    monkeypatch.setattr(recorder.subprocess, "Popen", Encoder)
    monkeypatch.setattr(recorder, "encode", lambda command, _log: recorder.Path(command[-1]).write_bytes(b"mp4"))
    args = recorder.parser().parse_args(["--url", "http://localhost/player/fixture/", "--fps", "10",
                                         "--ffmpeg", "fake-ffmpeg", "--output-dir", str(tmp_path),
                                         "--max-seconds", ".8" if ending == "limit" else "10", "--keep-raw",
                                         "--text-size", "large", "--typeface", "mincho", "--auto-wait", "2.5",
                                         "--bgm-volume", "40"])
    if ending == "stop":
        (tmp_path / "stop.request").touch()
    states = []
    if ending == "chapter":
        result = recorder.record_chapter(Browser(), "http://localhost/player/fixture/", tmp_path / "chapter",
                                         args, lambda **state: states.append(state))
        assert result.name == "chapter.mp4"
    else:
        with pytest.raises(RuntimeError, match="途中の録画") as raised:
            recorder.record_chapter(Browser(), "http://localhost/player/fixture/", tmp_path / "chapter",
                                    args, lambda **state: states.append(state))
        outcome = raised.value
        assert (tmp_path / "chapter/partial.mp4").exists()
    assert len(probe.fade_at) == 1
    # Audio remains active while frame capture advances through the ramp.
    assert .4 <= probe.now - probe.fade_at[0] <= .51
    recorded = len(probe.writes) / args.fps
    assert recorded == states[-2]["chapter_recorded_seconds"]
    assert probe.fade_at[0] - probe.started_at + .4 <= recorded <= probe.fade_at[0] - probe.started_at + .51
    assert any(state.get("finishing") and state["fade_tail_seconds"] == .4 for state in states)
    assert "audio-finished" in probe.events and probe.events[-1] == "context-closed"
    # The chosen viewing settings are seeded before the page's own scripts run.
    seeded = next(event[1] for event in probe.events if isinstance(event, tuple))
    assert '"adn_settings_v1"' in seeded
    assert recorder.viewing_settings(args) == {"size": 28, "face": "mincho", "speed": 62, "wait": 25,
                                               "voice": 100, "bgm": 40, "alpha": 80}
    assert seeded == recorder.settings_script(args)
    assert recorder.json.dumps(recorder.json.dumps(recorder.viewing_settings(args))) in seeded
    # Only an explicit stop is reported as a stop; the time limit stays a failure.
    if ending != "chapter":
        assert isinstance(outcome, recorder.RecordingStopped) is (ending == "stop")
