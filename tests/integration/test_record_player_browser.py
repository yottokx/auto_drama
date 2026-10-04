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


@pytest.mark.parametrize("mode", ["single", "background_multi", "timeout", "music_mix", "music_transitions"])
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
    if mode in {"music_mix", "music_transitions"}:
        music_wav = io.BytesIO()
        with wave.open(music_wav, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(b"".join(
                struct.pack("<h", int(7000 * math.sin(i * 2 * math.pi * 220 / 24000)))
                for i in range(24000)
            ))
        encoded = subprocess.run([
            "ffmpeg", "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0",
            "-codec:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "pipe:1",
        ], input=music_wav.getvalue(), capture_output=True, check=True)
        assets["scene_music"] = encoded.stdout
        document["assets"].append({"id": "scene_music", "kind": "music",
                                   "filename": "scene_music.mp3", "artifact_id": "scene_music",
                                   "sha256": hashlib.sha256(encoded.stdout).hexdigest()})
        document["music_cues"] = [{"id": "scene_music_start",
                                   "utterance_id": document["utterances"][0]["id"],
                                   "action": "play", "asset_id": "scene_music", "volume": 0.5,
                                   "loop_start_seconds": 0.1, "loop_end_seconds": 0.9}]
        if mode == "music_transitions":
            second_wav = io.BytesIO()
            with wave.open(second_wav, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(24000)
                wav.writeframes(b"".join(
                    struct.pack("<h", int(7000 * math.sin(i * 2 * math.pi * 330 / 24000)))
                    for i in range(24000)
                ))
            second = subprocess.run([
                "ffmpeg", "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0",
                "-codec:a", "libmp3lame", "-b:a", "192k", "-f", "mp3", "pipe:1",
            ], input=second_wav.getvalue(), capture_output=True, check=True)
            assets["next_music"] = second.stdout
            document["assets"].append({"id": "next_music", "kind": "music",
                                       "filename": "next_music.mp3", "artifact_id": "next_music",
                                       "sha256": hashlib.sha256(second.stdout).hexdigest()})
            document["music_cues"].extend([
                {"id": "scene_music_continue", "utterance_id": "recording-narration", "action": "continue"},
                {"id": "scene_music_change", "utterance_id": "recording-second", "action": "play",
                 "asset_id": "next_music", "volume": 0.5,
                 "loop_start_seconds": 0.1, "loop_end_seconds": 0.9},
            ])
            document["scene_transitions"] = [
                {"id": "first_stage", "utterance_id": document["utterances"][0]["id"],
                 "visual": "none", "duration_ms": 0, "music_fade_out_ms": 0, "music_fade_in_ms": 500},
                {"id": "same_stage", "utterance_id": "recording-narration", "visual": "none", "duration_ms": 0},
                {"id": "next_stage", "utterance_id": "recording-second", "visual": "fade", "duration_ms": 300,
                 "music_fade_out_ms": 800, "music_fade_in_ms": 1200},
            ]
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
        assert .8 <= state["chapter_recorded_seconds"] <= 1.5, "only a bounded 400ms ending tail is allowed"
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
    if mode in {"music_mix", "music_transitions"}:
        rate = int(track["sample_rate"])

        def tone_amplitude(frequency):
            # A one-second Fourier window distinguishes the 220 Hz BGM from
            # the 440 Hz voice; a voice-only recording must fail this check.
            amplitudes = []
            for offset in range(0, len(samples) - rate + 1, rate // 2):
                real = imaginary = 0.0
                for index, sample in enumerate(samples[offset:offset + rate]):
                    angle = index * 2 * math.pi * frequency / rate
                    real += sample * math.cos(angle)
                    imaginary += sample * math.sin(angle)
                amplitudes.append(2 * math.hypot(real, imaginary) / rate)
            return max(amplitudes)

        assert tone_amplitude(220) > 500
        assert tone_amplitude(440) > 1000
        # The reader's chapter-end BGM continues looping. Only the recorder mix
        # fades over its finite 400ms tail; the captured ending must settle to
        # silence rather than cutting a full-volume loop at the file boundary.
        ending_frequency = 330 if mode == "music_transitions" else 220
        window = rate // 20
        tail = samples[-int(rate * .8):]
        final_amplitudes = []
        for offset in range(0, len(tail) - window + 1, window):
            real = imaginary = 0.0
            for index, sample in enumerate(tail[offset:offset + window]):
                angle = index * 2 * math.pi * ending_frequency / rate
                real += sample * math.cos(angle)
                imaginary += sample * math.sin(angle)
            final_amplitudes.append(2 * math.hypot(real, imaginary) / window)
        steady = max(final_amplitudes)
        assert steady > 500
        assert final_amplitudes[-1] < steady * .15, "loop BGM must reach silence before recording stops"
        assert sum(steady * .15 < value < steady * .85 for value in final_amplitudes) >= 3
        if mode == "music_transitions":
            assert tone_amplitude(330) > 500, "The replacement BGM must enter the recording mix"
            # Short Fourier windows isolate the new BGM's actual captured
            # envelope from voice. A tap before the fade gain would jump to
            # full amplitude and produce fewer than four intermediate windows.
            window = rate // 10
            amplitudes = []
            for offset in range(0, len(samples) - window + 1, window):
                real = imaginary = 0.0
                for index, sample in enumerate(samples[offset:offset + window]):
                    angle = index * 2 * math.pi * 330 / rate
                    real += sample * math.cos(angle)
                    imaginary += sample * math.sin(angle)
                amplitudes.append(2 * math.hypot(real, imaginary) / window)
            peak = max(amplitudes)
            plateau = next(index for index, value in enumerate(amplitudes) if value >= peak * 0.85)
            assert sum(peak * 0.15 < value < peak * 0.85 for value in amplitudes[:plateau]) >= 4
