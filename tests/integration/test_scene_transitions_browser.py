"""Opt-in native Tyrano/Web Audio check for scene continuity and cold preloading."""

import functools
import hashlib
import io
import os
import shutil
import subprocess
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zipfile import ZipFile

import pytest

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, demo_content
from packages.tyrano_export.demo import _png

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("AUTO_DRAMA_TRANSITION_SMOKE") != "1",
                                reason="Set AUTO_DRAMA_TRANSITION_SMOKE=1 for real-browser checks")


@pytest.fixture
def transition_site(tmp_path):
    script, assets = demo_content()
    value = script.model_dump(mode="json")
    value["directions"] = [d for d in value["directions"] if d["utterance_id"] == "line_001"]
    for line in value["utterances"]:
        line.update(display_text=line["id"], spoken_text=line["id"], speaker_id=None)
    # The old conversion emitted a background and full portrait reset at every boundary.
    for direction in list(value["directions"]):
        if direction["kind"] == "enter":
            value["directions"].append({"id": "reset_" + direction["character_id"], "kind": "exit",
                "utterance_id": "line_002", "character_id": direction["character_id"]})
    for direction in list(value["directions"]):
        if direction["utterance_id"] == "line_001":
            value["directions"].append({**direction, "id": "repeat_" + direction["id"],
                                        "utterance_id": "line_002"})
    assets["second_station"] = _png(96, 64, lambda x, y: (92, 30, 42, 255))
    value["assets"].append({"id": "second_station", "kind": "background", "artifact_id": "second_station",
                            "filename": "second_station.png",
                            "sha256": hashlib.sha256(assets["second_station"]).hexdigest()})
    value["directions"].extend([
        {"id": "new_background", "kind": "background", "utterance_id": "line_003", "asset_id": "second_station"},
        {"id": "ren_leaves", "kind": "exit", "utterance_id": "line_003", "character_id": "ren"},
        {"id": "old_background", "kind": "background", "utterance_id": "line_004", "asset_id": "station"},
    ])
    executable = shutil.which("ffmpeg") or "C:/ffmpeg/bin/ffmpeg.exe"
    for name, frequency in (("first_music", 220), ("second_music", 330)):
        path = tmp_path / (name + ".mp3")
        subprocess.run([executable, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
                        f"sine=frequency={frequency}:duration=12", "-ac", "2", "-ar", "44100",
                        "-codec:a", "libmp3lame", "-b:a", "192k", str(path)],
                       check=True, capture_output=True, timeout=20)
        assets[name] = path.read_bytes()
        value["assets"].append({"id": name, "kind": "music", "artifact_id": name,
                                "filename": name + ".mp3",
                                "sha256": hashlib.sha256(assets[name]).hexdigest()})
    value["music_cues"] = [
        {"id": "music_1", "utterance_id": "line_001", "action": "play", "asset_id": "first_music",
         "loop_start_seconds": 2.0, "loop_end_seconds": 10.0},
        {"id": "music_2", "utterance_id": "line_002", "action": "continue"},
        {"id": "music_3", "utterance_id": "line_003", "action": "play", "asset_id": "second_music",
         "loop_start_seconds": 2.0, "loop_end_seconds": 10.0},
        {"id": "music_4", "utterance_id": "line_004", "action": "stop"},
    ]
    value["scene_transitions"] = [
        {"id": "transition_1", "utterance_id": "line_001", "visual": "cut", "duration_ms": 0,
         "music_fade_out_ms": 0, "music_fade_in_ms": 0},
        {"id": "transition_2", "utterance_id": "line_002", "visual": "none", "duration_ms": 0,
         "music_fade_out_ms": 0, "music_fade_in_ms": 0},
        {"id": "transition_3", "utterance_id": "line_003", "visual": "fade", "duration_ms": 300,
         "music_fade_out_ms": 300, "music_fade_in_ms": 300},
        {"id": "transition_4", "utterance_id": "line_004", "visual": "dissolve", "duration_ms": 300,
         "music_fade_out_ms": 300, "music_fade_in_ms": 300},
    ]
    site = tmp_path / "site"
    site.mkdir()
    with ZipFile(io.BytesIO(compile_bundle(Script.model_validate(value), assets))) as archive:
        archive.extractall(site)  # Only our compiler's fixed fixture, never an uploaded ZIP.
    release_music = threading.Event()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith("second_music.mp3"):
                release_music.wait(timeout=15)
            try:
                return super().do_GET()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                return None  # Browser disposal deliberately cancels in-flight preloads.

        def translate_path(self, path):
            if path.startswith("/tyrano/"):
                return str(ROOT / "tyranoscript" / path.lstrip("/"))
            return super().translate_path(path)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(site)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", release_music
    finally:
        release_music.set()
        server.shutdown()
        server.server_close()
        worker.join()


def test_scene_continuity_cold_preload_and_one_advance_in_real_browser(transition_site):
    from playwright.sync_api import sync_playwright

    url, release_music = transition_site
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True,
                                              args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        page.add_init_script("""(() => {
          let factory;
          Object.defineProperty(window, 'AutoDramaMusic', {configurable:true,
            get: () => factory, set: value => {factory = {...value, create(...args) {
              const result = value.create(...args); window.testMusic = result; return result;
            }};}});
        })();""")
        try:
            page.goto(url)
            page.locator("#ad-start-button").click(timeout=20000)

            def stable_line(identifier):
                page.wait_for_function("""id => {
                  const k = window.TYRANO?.kag;
                  return k?.ftag?.array_tag[k.ftag.current_order_index]?.name === 'p' &&
                    !k.stat.is_adding_text && !k.tmp.auto_drama_transition &&
                    document.querySelector('.message_inner')?.textContent.includes(id);
                }""", arg=identifier, timeout=20000)

            stable_line("line_001")
            page.evaluate("window.testPortrait = document.querySelector('.tyrano_chara')")
            first = page.evaluate("window.testMusic.snapshot()")
            page.locator(".layer_event_click").click(force=True)
            stable_line("line_002")
            second = page.evaluate("window.testMusic.snapshot()")
            assert second["cue_id"] == first["cue_id"] == "music_1"
            assert second["position_seconds"] > first["position_seconds"]
            assert page.evaluate("window.testPortrait === document.querySelector('.tyrano_chara')")
            page.locator(".layer_event_click").click(force=True)
            page.wait_for_function("window.TYRANO.kag.tmp.auto_drama_transition === true")
            while_loading = page.evaluate("window.testMusic.snapshot()")
            assert while_loading["cue_id"] == "music_1", "Cold preload must retain old music"
            assert while_loading["position_seconds"] >= second["position_seconds"]
            page.wait_for_function("document.querySelector('#ad-save').disabled")
            # A second click while waiting must not skip the next line.
            page.keyboard.press("Enter")
            release_music.set()
            page.wait_for_selector(".ad-stage-transition")
            stable_line("line_003")
            third = page.evaluate("window.testMusic.snapshot()")
            assert third["cue_id"] == "music_3"
            assert 0 <= third["position_seconds"] < 3
            assert page.locator(".ad-stage-transition").count() == 0
            assert page.evaluate("window.TYRANO.kag.layer.getLayer('base','fore')[0].style.backgroundImage.includes('second_station.png')")
            assert page.locator(".tyrano_chara").count() == 1
            page.locator("#ad-save").click()
            page.wait_for_function("document.querySelector('#ad-status').textContent.includes('保存しました')")
            page.locator(".layer_event_click").click(force=True)
            stable_line("line_004")
            assert page.evaluate("window.testMusic.snapshot()") is None
            page.locator("#ad-load").click()
            stable_line("line_003")
            page.wait_for_function("window.testMusic.snapshot()?.cue_id === 'music_3'")
            assert not errors, errors
        finally:
            browser.close()
