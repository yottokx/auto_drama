"""Opt-in real-browser check of the viewing screen on the installed Tyrano engine."""

import functools
import hashlib
import io
import math
import os
import struct
import threading
import wave
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zipfile import ZipFile

import pytest

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, demo_content

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("AUTO_DRAMA_PLAYER_SMOKE") != "1",
                                reason="Set AUTO_DRAMA_PLAYER_SMOKE=1 for real-browser checks")
LONG = ("そう。じゃあ先に記録を済ませてしまおう。壁の図面は祖父が引いたもので、レンズの回転周期と油の残量を"
        "毎晩ここに書き足していく決まりなんだ。面倒に見えるけど、数字が一日でも抜けると、沖の船は灯りの癖を"
        "読めなくなる。だから嵐の晩ほど、誰かがここに座って鉛筆を握っていないといけない。")
STATE = "window.AutoDramaPlayer.state()"


def tone(seconds: float) -> bytes:
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(22050)
        wav.writeframes(b"".join(struct.pack("<h", int(6000 * math.sin(i / 12)))
                                 for i in range(int(22050 * seconds))))
    return output.getvalue()


@pytest.fixture
def site(tmp_path):
    script, assets = demo_content()
    value = script.model_dump(mode="json")
    # line_002 speaks for long enough to click while it plays; line_003 needs several pages.
    for line, seconds in (("line_002", 4.0), ("line_003", 6.0)):
        assets["voice_" + line] = tone(seconds)
        value["assets"].append({"id": "voice_" + line, "kind": "audio", "artifact_id": "voice_" + line,
                                "filename": line + ".wav",
                                "sha256": hashlib.sha256(assets["voice_" + line]).hexdigest()})
    for line in value["utterances"]:
        if line["id"] in {"line_002", "line_003"}:
            line["audio_asset_id"] = "voice_" + line["id"]
        if line["id"] == "line_003":
            line["display_text"] = LONG
    value["directions"].append({"id": "aki_leaves", "kind": "exit", "utterance_id": "line_003",
                                "timing": "after", "character_id": "aki"})
    directory = tmp_path / "site"
    directory.mkdir()
    with ZipFile(io.BytesIO(compile_bundle(Script.model_validate(value), assets))) as archive:
        archive.extractall(directory)  # Only our compiler's fixed fixture, never an uploaded ZIP.

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            try:
                return super().do_GET()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                return None

        def translate_path(self, path):
            if path.startswith("/tyrano/"):
                return str(ROOT / "tyranoscript" / path.lstrip("/"))
            return super().translate_path(path)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(directory)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/"
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def test_reading_paging_backlog_skip_and_saves_in_real_browser(site):
    from playwright.sync_api import sync_playwright

    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True,
                                              args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(site)

            def state():
                return page.evaluate(STATE)

            def until(condition, **arguments):
                page.wait_for_function("values => { const s = " + STATE + "; return " + condition + "; }",
                                       arg=arguments, timeout=20000)

            def stage_click():
                page.mouse.click(600, 250)  # Well above the window: anywhere on the stage advances.

            page.wait_for_function("document.getElementById('adn-begin')?.disabled === false", timeout=30000)
            assert page.locator("#adn-resume").is_hidden()
            # The engine's own message window must never show through.
            page.locator("#adn-begin").click()
            until("s.line === 'line_001' && s.tag === 'say'")
            assert page.locator("#adn-name").text_content() == ""
            assert page.locator(".tyrano_chara").count() == 2
            assert page.evaluate("getComputedStyle(document.querySelector('.message0_fore')).display") == "none"

            # Narration: one click finishes the text and shows the end mark, without advancing.
            stage_click()
            until("s.line === 'line_001' && s.tag === 'wait'")
            assert page.locator("#adn-body .mark.end").count() == 1
            stage_click()
            until("s.line === 'line_002' && s.tag === 'say' && s.voice")
            assert page.locator("#adn-name").text_content() == "レン"

            # Voiced line: the first click only enters the wait state and keeps the voice.
            stage_click()
            until("s.line === 'line_002' && s.phase === 'wait'")
            assert state()["voice"] is True and state()["tag"] == "say"
            stage_click()
            until("s.line === 'line_003' && s.tag === 'say' && s.voice")

            # A long line is split into pages of the same utterance; the voice is not restarted.
            pages = state()["pages"]
            assert pages >= 2
            assert page.locator("#adn-pages").text_content() == f"1 / {pages}"
            for number in range(1, pages):
                stage_click()
                until("s.phase === 'wait' && s.page === values.page", page=number - 1)
                assert page.locator("#adn-body .mark.more").count() == 1
                stage_click()
                until("s.page === values.page", page=number)
                assert state()["line"] == "line_003" and state()["voice"] is True
            assert page.locator(".tyrano_chara").count() == 2  # The exit after the line has not run yet.

            # Backlog: stops the voice, lists only lines reached so far, and can return to one.
            page.keyboard.press("ArrowUp")
            until("s.log")
            assert state()["voice"] is False
            assert page.locator("#adn-log-list .entry").count() == 3
            assert page.locator("#adn-log-list .entry.now p").text_content() == LONG
            page.locator("#adn-log-list .entry").nth(1).hover()
            page.locator("#adn-log-list .entry").nth(1).get_by_text("ここへ戻る").click()
            until("!s.log && !s.seeking && s.line === 'line_002' && s.tag === 'say'")
            assert page.locator(".tyrano_chara").count() == 2

            # Save here, hide and restore the UI, then skip to the end of the chapter.
            page.locator("#adn-m-save").click()
            page.wait_for_function("document.getElementById('adn-toast').textContent === 'セーブしました'")
            # The clicked button must not keep focus: Enter still belongs to the story.
            page.keyboard.press("Enter")
            until("s.line === 'line_002' && s.phase === 'wait'")
            page.locator("#adn-m-hide").click()
            page.wait_for_function("document.getElementById('adn').classList.contains('bare')")
            stage_click()
            page.wait_for_function("!document.getElementById('adn').classList.contains('bare')")
            assert state()["line"] == "line_002"  # Restoring the UI does not advance.
            page.keyboard.down("Control")
            until("s.ended")
            page.keyboard.up("Control")
            assert page.locator("#adn-end").is_visible()
            assert page.locator(".tyrano_chara").count() == 1  # Skipping still ran the exit.

            # Loading restores the saved line and the stage that belongs to it.
            page.locator("#adn-m-load").click()
            until("!s.ended && !s.seeking && s.line === 'line_002' && s.tag === 'say'")
            assert page.locator("#adn-end").is_hidden()
            assert page.locator(".tyrano_chara").count() == 2

            # Auto waits for the voice, then advances by itself; a click only cancels it.
            page.locator("#adn-m-auto").click()
            until("s.auto && s.line === 'line_003'")
            stage_click()
            until("!s.auto")
            assert state()["line"] == "line_003"

            # A reload offers the furthest position reached, not the one loaded afterwards.
            page.reload()
            page.wait_for_function("document.getElementById('adn-begin')?.disabled === false", timeout=30000)
            assert page.locator("#adn-resume").is_visible()
            page.locator("#adn-resume").click()
            until("s.started && !s.seeking && s.line === 'line_004'")
            assert not errors, errors
        finally:
            browser.close()


def test_browser_holding_an_earlier_cached_screen_reloads_into_the_current_one(tmp_path):
    """Earlier screens were served as immutable for a year under the same URLs."""
    from playwright.sync_api import sync_playwright

    from packages.tyrano_export.player import player_html

    script, assets = demo_content()
    directory = tmp_path / "site"
    directory.mkdir()
    with ZipFile(io.BytesIO(compile_bundle(script, assets))) as archive:
        archive.extractall(directory)
    # The earlier launcher: the engine without this screen's scripts, cached by the browser.
    earlier = "\n".join("<script>window.EARLIER_SCREEN = true</script>" if "auto_drama_player.js" in line
                        else "" if "auto_drama_" in line else line
                        for line in player_html().decode().splitlines()).encode()
    state = {"earlier": True}
    requests = []

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split("?", 1)[0]
            requests.append(path)
            if state["earlier"] and path in {"/", "/data/scenario/first.ks"}:
                content = earlier if path == "/" else b"[s]\n"
                self.send_response(200)
                self.send_header("Content-Type", "text/html" if path == "/" else "text/plain")
                self.send_header("Content-Length", str(len(content)))
                if path == "/":
                    self.send_header("Cache-Control", "private, max-age=31536000, immutable")
                self.end_headers()
                self.wfile.write(content)
                return None
            try:
                return super().do_GET()
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                return None

        def end_headers(self):
            if not state["earlier"]:
                self.send_header("Cache-Control", "no-store")
            super().end_headers()

        def translate_path(self, path):
            if path.startswith("/tyrano/"):
                return str(ROOT / "tyranoscript" / path.split("?", 1)[0].lstrip("/"))
            return super().translate_path(path)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(directory)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    url = f"http://127.0.0.1:{server.server_port}/"
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--mute-audio"])
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            try:
                page.goto(url)
                page.wait_for_function("window.EARLIER_SCREEN && window.TYRANO?.kag?.ftag?.array_tag?.length")
                # The server is updated; the browser comes back later with its cached launcher.
                state["earlier"] = False
                page.goto("about:blank")
                requests.clear()
                page.goto(url)
                page.wait_for_function("document.getElementById('adn-begin')?.disabled === false", timeout=30000)
                assert page.evaluate("!window.EARLIER_SCREEN && !!window.AutoDramaPlayer")
                # The launcher itself came from the cache first; only the reload asked for it.
                assert requests.count("/") == 1
                assert requests.index("/data/scenario/first.ks") < requests.index("/")
                page.locator("#adn-begin").click()
                page.wait_for_function(STATE + ".line === 'line_001' && " + STATE + ".tag === 'say'")
            finally:
                browser.close()
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
