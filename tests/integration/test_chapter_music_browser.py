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
def chapter_site(tmp_path):
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


def reach_chapter_end(page, url):
    page.goto(url)
    page.locator("#ad-start-button").click(timeout=20000)
    for identifier in ("first_line_001", "first_line_002"):
        page.wait_for_function("""id => {
          const k = window.TYRANO?.kag;
          return k?.ftag?.array_tag[k.ftag.current_order_index]?.name === 'p' &&
            !k.stat.is_adding_text && !k.tmp.auto_drama_transition &&
            document.querySelector('.message_inner')?.textContent.includes(id);
        }""", arg=identifier, timeout=15000)
        page.keyboard.press("Enter")
    page.wait_for_function("!document.querySelector('#ad-chapter-end').hidden")
    page.locator("#ad-next-button").wait_for(state="visible")


def test_end_music_is_saved_and_restored_then_fades_only_on_next_click(chapter_site, tmp_path):
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
            page.wait_for_timeout(300)
            after = page.evaluate("window.testMusic.snapshot()")
            assert after["position_seconds"] != before["position_seconds"]
            page.locator("#ad-backlog").click()
            assert page.evaluate("window.testMusic.snapshot().paused")
            page.locator("#ad-log-close").click()
            page.wait_for_function("window.testMusic.snapshot()?.paused === false")
            page.locator("#ad-save").click()
            page.reload()
            page.locator("#ad-resume-button").click(timeout=20000)
            page.wait_for_function("!document.querySelector('#ad-chapter-end').hidden && window.testMusic.snapshot()?.paused === false")
            page.locator("#ad-next-button").wait_for(state="visible")
            assert page.evaluate("window.testMusic.snapshot().cue_id") == "music_first"
            page.screenshot(path=str(tmp_path / "chapter-end-music.png"))
            # Native UI click is synchronous here, allowing observation during
            # the brief fade before navigation commits.
            clicked_at = time.monotonic()
            immediate = page.evaluate("""() => {
              document.querySelector('#ad-next-button').click();
              document.querySelector('#ad-next-button').click();
              return {overlay:!document.querySelector('#ad-chapter-transition').hidden,
                busy:!!window.TYRANO.kag.tmp.auto_drama_transition,
                disabled:document.querySelector('#ad-next-button').disabled,
                pathname:location.pathname};
            }""")
            assert immediate == {"overlay": True, "busy": True, "disabled": True, "pathname": "/player/first/"}
            page.wait_for_url("**/player/second/", timeout=15000)
            assert time.monotonic() - clicked_at >= 0.35, "Navigation must wait for the audible fade"
            # A chapter entered from Next may start itself; if the browser denies
            # autoplay its normal explicit start control remains usable.
            page.wait_for_function("document.querySelector('#ad-start').hidden || document.querySelector('#ad-start-button')?.disabled === false")
            # Auto-entry may hide Start between readiness and a scheduled click.
            # The same synchronous owned handler is idempotent in either case.
            page.evaluate("""() => {
              if (!document.querySelector('#ad-start').hidden && !document.querySelector('#ad-start-button').disabled)
                document.querySelector('#ad-start-button').click();
            }""")
            page.wait_for_function("window.testMusic.snapshot()?.cue_id === 'music_second'")
            page.go_back()
            page.wait_for_function("window.testMusic")
            if page.locator("#ad-start").is_visible():
                page.locator("#ad-resume-button").click(timeout=20000)
            page.wait_for_function("!document.querySelector('#ad-chapter-end').hidden && window.testMusic.snapshot()?.paused === false")
            assert page.evaluate("window.testMusic.snapshot().cue_id") == "music_first"
            assert page.locator("#ad-chapter-transition").is_hidden()
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
            page.locator("#ad-next-button").click()
            page.wait_for_function("document.querySelector('#ad-next-status').textContent.includes('進めませんでした')")
            page.wait_for_function("window.testMusic.snapshot()?.paused === false && !window.TYRANO.kag.tmp.auto_drama_transition")
            assert page.url == url
            assert page.evaluate("window.testMusic.snapshot().cue_id") == "music_first"
            assert page.locator("#ad-chapter-transition").is_hidden()
            assert not errors, errors
        finally:
            browser.close()
