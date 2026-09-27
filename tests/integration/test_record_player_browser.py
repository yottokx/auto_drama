"""Opt-in real browser + FFmpeg smoke test, with a local fixture and no generation."""

import functools
import hashlib
import io
import json
import math
import os
import struct
import subprocess
import sys
import threading
import time
import wave
import zipfile
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from itertools import pairwise
from pathlib import Path

import pytest

from packages.contracts import Script
from packages.tyrano_export import compile_bundle, demo_content

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.skipif(os.environ.get("AUTO_DRAMA_RECORD_SMOKE") != "1",
                                reason="Set AUTO_DRAMA_RECORD_SMOKE=1 to run real browser recording")


@pytest.mark.parametrize("mode", ["single", "background_multi", "timeout"])
def test_headless_recording_contains_video_and_audio(tmp_path, mode):
    script, assets = demo_content()
    document = script.model_dump(mode="json")
    document["utterances"] = document["utterances"][:1]
    document["directions"] = []
    stream = io.BytesIO()
    with wave.open(stream, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(24000)
        wav.writeframes(b"".join(struct.pack("<h", int(12000 * math.sin(i * 2 * math.pi * 440 / 24000)))
                                 for i in range(24000)))
    assets["tone"] = stream.getvalue()
    document["assets"].append({"id": "tone", "kind": "audio", "filename": "tone.wav",
                               "artifact_id": "tone", "sha256": hashlib.sha256(assets["tone"]).hexdigest()})
    document["utterances"][0]["audio_asset_id"] = "tone"
    # Real Web Audio capture can omit packets between active sources. Include
    # unvoiced narration between two voices so timestamp gaps cannot go unnoticed.
    document["utterances"][0]["display_text"] = "最初の音声です。"
    document["utterances"].extend([
        {**document["utterances"][0], "id": "recording-narration", "speaker_id": None,
         "audio_asset_id": None,
         "display_text": "音声のない地の文を表示しています。この文章を読む間も動画の時刻は連続して進みます。"},
        {**document["utterances"][0], "id": "recording-second", "display_text": "次の音声です。"},
    ])
    bundle = compile_bundle(Script.model_validate(document), assets)
    site = tmp_path / "site"
    site.mkdir()
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        archive.extractall(site)

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path.endswith("player-context.json"):
                identifier = self.path.split("/")[2]
                body = {"mode": "live", "build_id": identifier, "production_id": "production",
                        "project_id": "project", "storyline_id": "storyline",
                        "chapter_number": 1 if identifier == "first" else 2,
                        "next_url": f"/api/m3/builds/{identifier}/next"}
            elif self.path.startswith("/api/m3/builds/"):
                identifier = self.path.split("/")[4]
                body = {"build_id": identifier, "status": "ready" if identifier == "first" else "complete",
                        "next_build": {"id": "second", "chapter_number": 2}}
            else:
                return super().do_GET()
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def translate_path(self, path):
            if path.startswith("/player/"):
                path = "/" + path.split("/", 3)[3]
            if path.startswith("/tyrano/"):
                return str(ROOT / "tyranoscript" / path.lstrip("/"))
            return super().translate_path(path)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(Handler, directory=str(site)))
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    output = tmp_path / "録画 test"
    snapshots = []
    finished = threading.Event()

    def watch_progress():
        while not finished.wait(0.1):
            try:
                snapshots.append(json.loads((output / "status.json").read_text("utf-8")))
            except (OSError, ValueError):
                pass

    watcher = threading.Thread(target=watch_progress, daemon=True)
    watcher.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}/player/first/"
        if mode == "background_multi":
            subprocess.run(["pwsh", "-NoProfile", "-File", str(ROOT / "scripts/record-player.ps1"),
                            "-Url", url, "-OutputDir", str(output), "-Fps", "10"],
                           check=True, timeout=20)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                try:
                    state = json.loads((output / "status.json").read_text("utf-8"))
                    if state["status"] != "running":
                        break
                except (OSError, ValueError):
                    pass
                time.sleep(0.5)
            else:
                (output / "stop.request").touch()
                pytest.fail("Background recording did not finish")
        else:
            result = subprocess.run([sys.executable, "-X", "utf8", str(ROOT / "scripts/record/record_player.py"),
                                     "--url", url, "--single-chapter", "--output-dir", str(output),
                                     "--max-seconds", "0.8" if mode == "timeout" else "30", "--fps", "10"],
                                    timeout=100, check=False)
            assert result.returncode == (1 if mode == "timeout" else 0)
    finally:
        finished.set()
        watcher.join()
        server.shutdown()
        server.server_close()
        worker.join()
    state = json.loads((output / "status.json").read_text("utf-8"))
    if mode == "timeout":
        assert state["status"] == "failed"
        assert (output / "chapter-001/partial.mp4").stat().st_size > 1000
        return
    assert state["status"] == "completed", state
    assert len(state["chapters"]) == (2 if mode == "background_multi" else 1)
    assert state["utterance_current"] == state["utterance_total"] == 3
    assert state["current_chapter"] == len(state["chapters"])
    assert any(snapshot.get("phase") == "recording"
               and 0 < snapshot.get("utterance_current", 0) < 3
               and snapshot.get("recorded_seconds", 0) > 0 for snapshot in snapshots)
    assert state["elapsed_seconds"] >= state["recorded_seconds"] - 0.5
    assert state["recorded_seconds"] == pytest.approx(sum(
        chapter["recorded_seconds"] for chapter in state["chapters"]))
    probe = subprocess.run(["ffprobe", "-v", "error", "-show_streams", "-of", "json",
                            str(output / "recording.mp4")], capture_output=True, check=True)
    streams = json.loads(probe.stdout)["streams"]
    assert {item["codec_type"] for item in streams} == {"video", "audio"}
    video = next(item for item in streams if item["codec_type"] == "video")
    assert abs(state["recorded_seconds"] - float(video["duration"])) < 0.15
    packets = subprocess.run([
        "ffprobe", "-v", "error", "-select_streams", "a", "-show_packets",
        "-show_entries", "packet=pts_time", "-of", "json", str(output / "recording.mp4"),
    ], capture_output=True, check=True)
    times = [float(packet["pts_time"]) for packet in json.loads(packets.stdout)["packets"]]
    assert max(right - left for left, right in pairwise(times)) < 0.05
    audio = subprocess.run(["ffmpeg", "-v", "error", "-i", str(output / "recording.mp4"),
                            "-f", "s16le", "-ac", "1", "pipe:1"], capture_output=True, check=True)
    samples = struct.unpack(f"<{len(audio.stdout) // 2}h", audio.stdout)
    assert max(abs(sample) for sample in samples) > 1000
    track = next(item for item in streams if item["codec_type"] == "audio")
    decoded_seconds = len(samples) / int(track["sample_rate"])
    assert abs(decoded_seconds - float(track["duration"])) < 0.1
