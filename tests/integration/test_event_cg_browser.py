"""Opt-in native Tyrano checks for CG intervals, save/load and missing images."""

import functools
import hashlib
import io
import os
import threading
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from zipfile import ZipFile

import pytest

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, demo_content
from packages.tyrano_export.demo import _png
from tests.integration import test_chapter_music_browser

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("AUTO_DRAMA_CG_SMOKE") != "1",
                                reason="Set AUTO_DRAMA_CG_SMOKE=1 for real-browser checks")
chapter_site = test_chapter_music_browser.chapter_site


@pytest.fixture
def cg_site(tmp_path):
    script, assets = demo_content()
    value = script.model_dump(mode="json")
    for name, kind, color in (("cg_base", "event_cg", (180, 70, 90, 255)),
                              ("cg_variant", "event_cg", (90, 70, 180, 255)),
                              ("later_background", "background", (40, 90, 60, 255))):
        assets[name] = _png(96, 54, lambda x, y, color=color: color)
        value["assets"].append({"id": name, "kind": kind, "artifact_id": name,
                                "filename": name + ".png", "sha256": hashlib.sha256(assets[name]).hexdigest()})
    value["utterances"].extend({"id": f"line_{index:03}", "display_text": str(index),
                                "spoken_text": str(index)} for index in (5, 6))
    for line in value["utterances"]:
        line.update(display_text=line["id"], spoken_text=line["id"], speaker_id=None)
    value["event_cg_segments"] = [
        {"id": "event_one", "start_utterance_id": "line_002", "end_utterance_id": "line_004",
         "base_asset_id": "cg_base", "variants": [
             {"id": "change_one", "utterance_id": "line_003", "asset_id": "cg_variant"}]},
        {"id": "event_last", "start_utterance_id": "line_006", "end_utterance_id": None,
         "base_asset_id": "cg_base"},
    ]
    value["directions"].extend([
        {"id": "changed_bg", "kind": "background", "utterance_id": "line_003", "asset_id": "later_background"},
        {"id": "move_ren", "kind": "position", "utterance_id": "line_003", "character_id": "ren", "position": "center"},
        {"id": "aki_leaves", "kind": "exit", "utterance_id": "line_003", "timing": "after", "character_id": "aki"},
    ])
    site = tmp_path / "site"
    site.mkdir()
    with ZipFile(io.BytesIO(compile_bundle(Script.model_validate(value), assets))) as archive:
        archive.extractall(site)  # Only this compiler-owned fixture.
    missing = set()

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.rsplit("/", 1)[-1] in missing:
                self.send_error(404)
                return
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

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(site)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}/", missing
    finally:
        server.shutdown()
        server.server_close()
        worker.join()


def stable_line(page, identifier):
    page.wait_for_function("""id => {
      const k = window.TYRANO?.kag;
      return k?.ftag?.array_tag[k.ftag.current_order_index]?.name === 'p' &&
        !k.stat.is_adding_text && !k.tmp.auto_drama_transition &&
        document.querySelector('.message_inner')?.textContent.includes(id);
    }""", arg=identifier, timeout=20000)


def picture(page):
    return page.evaluate("""() => {
      const k = window.TYRANO.kag, base = k.layer.getLayer('base', 'fore')[0];
      return {source: base.style.backgroundImage, fit: base.style.backgroundSize,
        cast: [...document.querySelectorAll('.tyrano_chara')].map(item => item.className),
        cg: k.stat.auto_drama_event_cg};
    }""")


def test_cg_variant_restore_skip_and_chapter_end(cg_site, tmp_path):
    from playwright.sync_api import sync_playwright

    url, _ = cg_site
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--mute-audio"])
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        try:
            page.goto(url)
            page.locator("#ad-start-button").click(timeout=20000)
            stable_line(page, "line_001")
            assert len(picture(page)["cast"]) == 2
            page.locator(".layer_event_click").click(force=True)
            stable_line(page, "line_002")
            first = picture(page)
            assert "cg_base.png" in first["source"] and first["fit"] == "contain" and not first["cast"]
            page.locator(".layer_event_click").click(force=True)
            stable_line(page, "line_003")
            assert "cg_variant.png" in picture(page)["source"] and not picture(page)["cast"]
            page.screenshot(path=str(tmp_path / "cg-variant.png"))
            page.locator("#ad-save").click()
            page.wait_for_function("document.querySelector('#ad-status').textContent.includes('保存しました')")
            page.locator(".layer_event_click").click(force=True)
            stable_line(page, "line_004")
            normal = picture(page)
            assert "later_background.png" in normal["source"] and len(normal["cast"]) == 1
            assert "ad_ren" in normal["cast"][0]
            page.reload()
            page.locator("#ad-resume-button").click(timeout=20000)
            stable_line(page, "line_003")
            page.wait_for_function("window.TYRANO.kag.stat.auto_drama_event_cg?.variant_id === 'change_one'")
            assert "cg_variant.png" in picture(page)["source"] and not picture(page)["cast"]
            page.evaluate("window.TYRANO.kag.config.unReadTextSkip = 'true'; window.TYRANO.kag.key_mouse.skip()")
            page.wait_for_function("!document.querySelector('#ad-chapter-end').hidden", timeout=20000)
            last = picture(page)
            assert "cg_base.png" in last["source"] and not last["cast"]
            page.reload()
            page.locator("#ad-resume-button").click(timeout=20000)
            page.wait_for_function("!document.querySelector('#ad-chapter-end').hidden", timeout=20000)
            assert picture(page)["cg"]["segment_id"] == "event_last"
            assert not errors, errors
        finally:
            browser.close()


def test_missing_cg_uses_current_normal_stage_and_keeps_reading(cg_site):
    from playwright.sync_api import sync_playwright

    url, missing = cg_site
    missing.update(("cg_base.png", "cg_variant.png"))
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--mute-audio"])
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        try:
            page.goto(url)
            page.locator("#ad-start-button").click(timeout=20000)
            stable_line(page, "line_001")
            page.locator(".layer_event_click").click(force=True)
            stable_line(page, "line_002")
            assert "station.png" in picture(page)["source"] and len(picture(page)["cast"]) == 2
            page.locator(".layer_event_click").click(force=True)
            stable_line(page, "line_003")
            assert "later_background.png" in picture(page)["source"]
            assert len(picture(page)["cast"]) == 1
            assert "イベントCGを読み込めませんでした" in page.locator("#ad-status").inner_text()
            page.locator(".layer_event_click").click(force=True)
            stable_line(page, "line_004")
            assert len(picture(page)["cast"]) == 1 and picture(page)["cg"] is None
        finally:
            browser.close()


@pytest.mark.parametrize("chapter_site", ["event_cg"], indirect=True)
def test_next_chapter_replaces_old_cg_and_next_cg_keeps_music_running(chapter_site):
    from playwright.sync_api import sync_playwright

    url, _, _ = chapter_site
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel="msedge", headless=True,
                                              args=["--autoplay-policy=no-user-gesture-required", "--mute-audio"])
        page = browser.new_page(viewport={"width": 1200, "height": 900})
        errors = []
        page.on("pageerror", lambda error: errors.append(str(error)))
        test_chapter_music_browser.observe_music(page)
        try:
            test_chapter_music_browser.reach_chapter_end(page, url)
            assert "cg_first.png" in picture(page)["source"] and not picture(page)["cast"]
            page.locator("#ad-next-button").click()
            stable_line(page, "second_line_001")
            assert "station.png" in picture(page)["source"] and len(picture(page)["cast"]) == 2
            before = page.evaluate("window.testMusic.snapshot()")
            page.locator(".layer_event_click").click(force=True)
            stable_line(page, "second_line_002")
            after = page.evaluate("window.testMusic.snapshot()")
            assert "cg_second.png" in picture(page)["source"] and not picture(page)["cast"]
            assert before["cue_id"] == after["cue_id"] == "music_second"
            assert after["position_seconds"] >= before["position_seconds"]
            assert not errors, errors
        finally:
            browser.close()
