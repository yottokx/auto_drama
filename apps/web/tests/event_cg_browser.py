"""Isolated browser smoke for CG controls; never connects to a real coordinator.

Run with the repository Python environment and an output directory for screenshots:
  .venv/Scripts/python.exe apps/web/tests/event_cg_browser.py --output outputs/cg-ui
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

from playwright.sync_api import expect, sync_playwright

WEB = Path(__file__).resolve().parents[1]
FIXTURE = """
import React, { useState } from 'react';
import { createRoot } from 'react-dom/client';
import { EventCgPolicyEditor } from '../../src/EventCgPolicyEditor';
import { SettingsDialog } from '../../src/SettingsDialog';
import '../../src/style.css';
import '../../src/previewExtras.css';
function Fixture() {
  const [status, setStatus] = useState({ canApprove: false, pending: false, policy: null });
  const [settings, setSettings] = useState(false);
  return <main style={{maxWidth: 940, margin: '40px auto', padding: '0 20px'}}>
    <h1>構成確認・イベントCG</h1>
    <EventCgPolicyEditor projectId="fixture" disabled={false} readonly={false} onStatusChange={setStatus}/>
    <button className="button button-primary" disabled={!status.canApprove}>構成を承認</button>
    <button className="button button-light" onClick={() => setSettings(true)}>設定を開く</button>
    {settings && <SettingsDialog onClose={() => setSettings(false)}/>}
  </main>;
}
createRoot(document.getElementById('root')).render(<React.StrictMode><Fixture/></React.StrictMode>);
"""


def run(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    policy = {"revision": 0, "max_cgs": 0, "max_variants_per_cg": 0}
    settings = {"revision": 0, "ready": False, "workers": [], "profile": {
        "backend": "qwen_image21", "model_revision": "d26bb61231c349cf6b7896fa83353113880e1ba3",
        "dtype": "bfloat16", "cpu_offload": True, "use_kv_cache": True,
        "steps": 40, "width": 960, "height": 640,
    }}
    calls, errors = [], []

    def route_api(route):
        path = urlparse(route.request.url).path
        calls.append((route.request.method, path))
        body = route.request.post_data_json if route.request.method == "PUT" else None
        if path.endswith("/policy"):
            if body:
                assert body["expected_revision"] == policy["revision"]
                policy.update(max_cgs=body["max_cgs"], max_variants_per_cg=body["max_variants_per_cg"],
                              revision=policy["revision"] + 1)
            value = policy
        elif path == "/api/event-cg/settings":
            if body:
                assert body["expected_revision"] == settings["revision"]
                settings.update(profile=body["profile"], revision=settings["revision"] + 1)
            value = settings
        else:
            route.fulfill(status=503, content_type="application/json", body=json.dumps({"detail": "Unrelated settings are outside this fixture."}))
            return
        route.fulfill(content_type="application/json", body=json.dumps(value))

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix="cg-browser-", dir=WEB / "tests") as folder:
        fixture = Path(folder)
        (fixture / "fixture.tsx").write_text(FIXTURE, encoding="utf-8")
        (fixture / "index.html").write_text('<html lang="ja"><meta charset="UTF-8"><div id="root"></div><script type="module" src="./fixture.tsx"></script></html>', encoding="utf-8")
        process = subprocess.Popen(["node", str(WEB / "node_modules/vite/bin/vite.js"), "--host", "127.0.0.1", "--port", str(port), "--strictPort"],
            cwd=WEB, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        try:
            base = f"http://127.0.0.1:{port}"
            for _ in range(100):
                try:
                    with urlopen(base, timeout=0.2):
                        break
                except OSError:
                    if process.poll() is not None:
                        raise RuntimeError("Isolated Vite server exited")
                    time.sleep(0.1)
            else:
                raise RuntimeError("Isolated Vite server did not start")
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(channel="msedge", headless=True)
                try:
                    page = browser.new_page(viewport={"width": 1200, "height": 1000})
                    page.route("**/api/**", route_api)
                    page.on("pageerror", lambda error: errors.append(str(error)))
                    page.goto(f"{base}/tests/{fixture.name}/index.html")
                    approve = page.get_by_role("button", name="構成を承認", exact=True)
                    expect(approve).to_be_enabled()
                    assert not any(path == "/api/event-cg/settings" for _, path in calls)
                    page.screenshot(path=str(output / "policy-disabled.png"), full_page=True)
                    page.get_by_label("作品全体の基本CG上限").fill("3")
                    page.get_by_label("CG1件あたりの追加差分上限").fill("2")
                    expect(approve).to_be_disabled()
                    expect(page.get_by_text("基本CG最大3件、差分込みで最大9枚", exact=False)).to_be_visible()
                    expect(page.get_by_text("実行環境の準備が必要です", exact=False)).to_be_visible()
                    page.get_by_role("button", name="CG設定を保存", exact=True).click()
                    expect(page.get_by_text("イベントCGの設定を保存しました", exact=False)).to_be_visible()
                    expect(approve).to_be_disabled()
                    page.screenshot(path=str(output / "policy-unready.png"), full_page=True)
                    settings.update(ready=True, workers=[{"id": "fixture-worker", "name": "テスト用GPU", "ready": True}])
                    page.get_by_role("button", name="準備状態を再確認", exact=True).click()
                    expect(approve).to_be_enabled()
                    page.screenshot(path=str(output / "policy-ready.png"), full_page=True)
                    page.get_by_role("button", name="設定を開く", exact=True).click()
                    page.get_by_role("tab", name="画像", exact=True).click()
                    expect(page.get_by_role("heading", name="Qwen-Image-2.1", exact=True)).to_be_visible()
                    expect(page.get_by_text("960 × 640", exact=True)).to_be_visible()
                    page.get_by_label("推論ステップ数").fill("25")
                    page.get_by_role("button", name="画像設定を保存", exact=True).click()
                    expect(page.get_by_text("画像設定を保存しました", exact=False)).to_be_visible()
                    assert settings["profile"]["steps"] == 25
                    page.screenshot(path=str(output / "image-settings.png"), full_page=True)
                    page.set_viewport_size({"width": 390, "height": 844})
                    assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
                    page.screenshot(path=str(output / "image-settings-mobile.png"), full_page=True)
                    assert not errors, errors
                finally:
                    browser.close()
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    print(f"CG UI browser checks passed. Screenshots: {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
