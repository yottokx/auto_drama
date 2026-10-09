"""Opt-in native-player verification of music at interactive chapter boundaries."""

import functools
import hashlib
import io
import json
import os
import shutil
import subprocess
import threading
import time
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zipfile import ZipFile

import pytest

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, demo_content

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("AUTO_DRAMA_CHAPTER_MUSIC_SMOKE") != "1",
                                reason="Set AUTO_DRAMA_CHAPTER_MUSIC_SMOKE=1 for native browser checks")


@pytest.fixture
def chapter_site(tmp_path, request):
    site = tmp_path / "site"
    site.mkdir()
    for number, identifier in enumerate(("first", "second"), start=1):
        script, assets = demo_content()
        value = script.model_dump(mode="json")
        value.update(id=f"chapter_{identifier}", title=f"Chapter {number}")
        value["utterances"] = value["utterances"][:2]
        value["directions"] = [row for row in value["directions"] if row["utterance_id"] == "line_001"]
        for line in value["utterances"]:
            line.update(display_text=f"{identifier}_{line['id']}", spoken_text=f"{identifier}_{line['id']}",
                        speaker_id=None, audio_asset_id=None)
        name = f"music_{identifier}"
        mp3 = tmp_path / (name + ".mp3")
        executable = shutil.which("ffmpeg") or "C:/ffmpeg/bin/ffmpeg.exe"
        subprocess.run([executable, "-nostdin", "-v", "error", "-f", "lavfi", "-i",
            f"sine=frequency={220 * number}:duration=3", "-ac", "2", "-ar", "44100",
            "-codec:a", "libmp3lame", "-b:a", "192k", str(mp3)],
            check=True, capture_output=True, timeout=20)
        assets[name] = mp3.read_bytes()
        value["assets"].append({"id": name, "kind": "music", "artifact_id": name,
            "filename": name + ".mp3", "sha256": hashlib.sha256(assets[name]).hexdigest()})
        value["music_cues"] = [{"id": name, "utterance_id": "line_001", "action": "play", "asset_id": name,
            "volume": 0.5, "loop_start_seconds": 0.2, "loop_end_seconds": 2.8}]
        value["scene_transitions"] = [{"id": "first_stage", "utterance_id": "line_001", "visual": "none",
            "duration_ms": 0, "music_fade_out_ms": 0, "music_fade_in_ms": 200}]
        if getattr(request, "param", None) == "event_cg":
            cg = "cg_" + identifier
            assets[cg] = assets["station"]
            value["assets"].append({"id": cg, "kind": "event_cg", "artifact_id": cg,
                "filename": cg + ".png", "sha256": hashlib.sha256(assets[cg]).hexdigest()})
            value["event_cg_segments"] = [{"id": cg,
                "start_utterance_id": "line_001" if identifier == "first" else "line_002",
                "end_utterance_id": None, "base_asset_id": cg}]
        chapter = site / identifier
        chapter.mkdir()
        with ZipFile(io.BytesIO(compile_bundle(Script.model_validate(value), assets))) as archive:
            archive.extractall(chapter)  # A compiler-owned fixture, never an uploaded archive.
    requests = []
    state = {"fail_next": False}

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            requests.append(self.path)
            if self.path.endswith("player-context.json"):
                identifier = self.path.split("/")[2]
                body = {"mode": "live", "build_id": identifier, "production_id": f"production_{identifier}",
                    "project_id": "project", "storyline_id": "series", "chapter_number": 1 if identifier == "first" else 2,
                    "next_url": f"/api/m3/builds/{identifier}/next"}
            elif self.path.startswith("/api/m3/builds/"):
                if state["fail_next"]:
                    self.send_error(503)
                    return
                identifier = self.path.split("/")[4]
                body = {"build_id": identifier, "status": "ready" if identifier == "first" else "complete",
                    "next_build": {"id": "second", "chapter_number": 2} if identifier == "first" else None}
            else:
                try:
                    return super().do_GET()
                except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                    return None
            content = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)

        def translate_path(self, path):
            path = path.split("?", 1)[0]
            if path.startswith("/player/") and "/tyrano/" in path:
                return str(ROOT / "tyranoscript" / ("tyrano/" + path.split("/tyrano/", 1)[1]))
            if path.startswith("/tyrano/"):
                return str(ROOT / "tyranoscript" / path.lstrip("/"))
            if path.startswith("/player/"):
                path = path.removeprefix("/player/")
            return str(site / path.lstrip("/"))

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(site)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/player/first/", state, requests
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def observe_music(page):
    page.add_init_script("""(() => {
      let factory;
      Object.defineProperty(window, 'AutoDramaMusic', {configurable:true,
        get: () => factory, set: value => {factory = {...value, create(...args) {
          const result = value.create(...args); window.testMusic = result; return result;
        }};}});
    })();""")


STATE = "window.AutoDramaPlayer.state()"


def reach_chapter_end(page, url):
    page.goto(url)
    page.wait_for_function("document.getElementById('adn-begin')?.disabled === false", timeout=20000)
    page.locator("#adn-begin").click()
    for identifier in ("line_001", "line_002"):
        page.wait_for_function(
            "id => { const s = " + STATE + "; return s.line === id && s.tag === 'wait' && !s.busy; }",
            arg=identifier, timeout=15000)
        assert page.locator("#adn-body").text_content().startswith("first_" + identifier)
        page.keyboard.press("Enter")
    page.wait_for_function(STATE + ".ended")
    page.locator("#adn-next").wait_for(state="visible")


def test_end_music_keeps_playing_then_fades_only_on_next_click(chapter_site, tmp_path):
    from playwright.sync_api import sync_playwright

    url, _state, _requests = chapter_site
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--mute-audio"])
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        observe_music(page)
        try:
            reach_chapter_end(page, url)
            before = page.evaluate("window.testMusic.snapshot()")
            assert before and before["cue_id"] == "music_first" and not before["paused"]
            # The backlog stays available at the chapter end and never interrupts the music.
            page.locator("#adn-m-log").click()
            page.wait_for_function(STATE + ".log")
            page.wait_for_timeout(300)
            during = page.evaluate("window.testMusic.snapshot()")
            assert not during["paused"] and during["position_seconds"] != before["position_seconds"]
            page.locator("#adn-log-close").click()
            page.wait_for_function("!" + STATE + ".log")
            page.screenshot(path=str(tmp_path / "chapter-end-music.png"))
            # Two synchronous clicks: the second must not start another transition.
            clicked_at = time.monotonic()
            immediate = page.evaluate("""() => {
              document.querySelector('#adn-next').click();
              document.querySelector('#adn-next').click();
              return {disabled: document.querySelector('#adn-next').disabled, pathname: location.pathname};
            }""")
            assert immediate == {"disabled": True, "pathname": "/player/first/"}
            page.wait_for_url("**/player/second/", timeout=15000)
            assert time.monotonic() - clicked_at >= 0.35, "Navigation must wait for the audible fade"
            # A chapter entered from Next may start itself; if the browser denies
            # autoplay its normal explicit start control remains usable.
            page.wait_for_function(
                "window.AutoDramaPlayer?.state().started || document.getElementById('adn-begin')?.disabled === false")
            page.evaluate("""() => {
              if (!window.AutoDramaPlayer.state().started) document.getElementById('adn-begin').click();
            }""")
            page.wait_for_function("window.testMusic.snapshot()?.cue_id === 'music_second'")
            assert page.locator("#adn-chapter").text_content() == "第 2 章"
            # Returning offers the furthest position of the first chapter again.
            page.go_back()
            page.wait_for_function("document.getElementById('adn-begin')?.disabled === false", timeout=20000)
            page.locator("#adn-resume").click()
            page.wait_for_function(STATE + ".line === 'line_002' && window.testMusic.snapshot()?.cue_id === 'music_first'")
            assert not page.evaluate("window.testMusic.snapshot().paused")
            assert not errors, errors
        finally:
            browser.close()


def test_failed_next_chapter_restores_visible_waiting_scene_and_music(chapter_site):
    from playwright.sync_api import sync_playwright

    url, state, _requests = chapter_site
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--mute-audio"])
        page = browser.new_page()
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        observe_music(page)
        try:
            reach_chapter_end(page, url)
            state["fail_next"] = True
            page.locator("#adn-next").click()
            page.wait_for_function("document.querySelector('#adn-end-note').textContent.includes('進めませんでした')")
            page.wait_for_function(
                "window.testMusic.snapshot()?.paused === false && "
                "document.getElementById('adn-cover').style.opacity === '0' && "
                "!document.getElementById('adn-next').disabled")
            assert page.url == url
            assert page.evaluate("window.testMusic.snapshot().cue_id") == "music_first"
            # Once the successor answers again, the same button completes the move.
            state["fail_next"] = False
            page.locator("#adn-next").click()
            page.wait_for_url("**/player/second/", timeout=15000)
            assert not errors, errors
        finally:
            browser.close()
