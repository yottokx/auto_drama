"""Opt-in native Tyrano check: the experiment preview starts at its scene and swaps one CG."""

import json
import mimetypes
import os
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from packages.contracts import Script
from packages.tyrano_export import demo_content
from packages.tyrano_export.compiler import canonical_json
from packages.tyrano_export.demo import _png
from scripts.image_edit import cg_preview
from tests.integration.test_event_cg_browser import picture, stable_line

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("AUTO_DRAMA_CG_SMOKE") != "1",
                                reason="Set AUTO_DRAMA_CG_SMOKE=1 for real-browser checks")
IMAGE = _png(96, 64, lambda x, y: (200, 40, 40, 255))


@pytest.fixture
def coordinator():
    """The published chapter and the engine, as /player/{build}/ serves them."""
    script, assets = demo_content()
    value = script.model_dump(mode="json")
    value["utterances"].extend({"id": f"line_{index:03}", "display_text": str(index),
                                "spoken_text": str(index)} for index in (5, 6, 7))
    for line in value["utterances"]:
        line.update(display_text=line["id"], spoken_text=line["id"], speaker_id=None)
    script = Script.model_validate(value)
    files = {"script.json": canonical_json(script.model_dump(mode="json"))}
    for asset in script.assets:
        folder = {"background": "bgimage", "character": "fgimage"}[asset.kind]
        files[f"data/{folder}/{asset.filename}"] = assets[asset.id]

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            relative = self.path.split("?", 1)[0].removeprefix("/player/build-1/")
            content = files.get(relative)
            if content is None and relative.startswith("tyrano/"):
                target = ROOT / "tyranoscript" / relative
                content = target.read_bytes() if target.is_file() else None
            if content is None:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header("Content-Type", mimetypes.guess_type(relative)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            try:
                self.wfile.write(content)
            except (ConnectionAbortedError, ConnectionResetError, BrokenPipeError):
                pass

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{server.server_port}"
    finally:
        server.shutdown()
        server.server_close()


def test_preview_resumes_at_the_scene_and_shows_the_cg_only_in_its_interval(coordinator, tmp_path):
    from playwright.sync_api import sync_playwright

    image = tmp_path / "image.png"
    image.write_bytes(IMAGE)
    record = {"context": {"published_build_id": "build-1",
                          "utterances": [{"id": f"line_{index:03}"} for index in (3, 4, 5, 6, 7)],
                          "cg_proposal": {"proposal": {"display_from": 2, "display_to": 3}}}}
    preview = cg_preview.PreviewServer()
    url = preview.add(*cg_preview.prepare(record, image, coordinator, ("dissolve", 400)))
    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch(channel="msedge", headless=True, args=["--mute-audio"])
            page = browser.new_page(viewport={"width": 1200, "height": 900})
            errors = []
            page.on("pageerror", lambda error: errors.append(str(error)))
            try:
                page.goto(url)
                page.wait_for_function("document.getElementById('adn-begin')?.disabled === false",
                                       timeout=20000)
                assert page.locator("#adn-resume").is_visible()
                page.locator("#adn-resume").click()
                # The scene's first line, with the ordinary background and portraits.
                stable_line(page, "line_003")
                before = picture(page)
                assert "station.png" in before["source"] and before["cg"] is None
                page.keyboard.press("Enter")
                # The old picture stays on top and fades while the CG is already underneath.
                page.wait_for_selector(".ad-stage-transition", timeout=5000)
                stable_line(page, "line_004")
                assert page.locator(".ad-stage-transition").count() == 0
                shown = picture(page)
                assert "cg_preview.png" in shown["source"] and not shown["cast"]
                assert shown["cg"]["segment_id"] == "cg_preview_segment"
                page.screenshot(path=str(tmp_path / "cg-preview.png"))
                page.keyboard.press("Enter")
                stable_line(page, "line_005")
                assert "cg_preview.png" in picture(page)["source"]
                page.keyboard.press("Enter")
                stable_line(page, "line_006")
                after = picture(page)
                assert "station.png" in after["source"] and after["cg"] is None
                assert after["cast"] == before["cast"]
                # The page runs as a standalone export: no published-build identity is asked for.
                assert json.loads(page.evaluate("JSON.stringify(window.AutoDramaPlayer.state())"))["total"] == 7
                assert not errors, errors
            finally:
                browser.close()
    finally:
        preview.close()
