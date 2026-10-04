import json
import os
import subprocess
import sys
import time
import wave
from types import SimpleNamespace

import pytest

from scripts.audio import audio_files, player, process_tree


@pytest.fixture
def audio_sample(tmp_path):
    import numpy as np

    rate = 44100
    # Deliberately not aligned to the MP3 encoder's 1152-sample frames.
    count = rate + 237
    t = np.arange(count) / rate
    data = np.stack([0.25 * np.sin(2 * np.pi * 440 * t),
                     0.20 * np.sin(2 * np.pi * 659.25 * t)], axis=1).astype(np.float32)
    path = tmp_path / "音声 test & literal $(nothing).wav"
    with wave.open(str(path), "wb") as stream:
        stream.setnchannels(2)
        stream.setsampwidth(2)
        stream.setframerate(rate)
        stream.writeframes((data * 32767).astype("<i2").tobytes())
    return path, data, rate


@pytest.fixture
def ffmpeg_available():
    try:
        audio_files.find_ffmpeg()
        audio_files.find_ffprobe()
    except audio_files.AudioFileError:
        pytest.skip("FFmpeg/FFprobe are not available on this test host")


@pytest.mark.parametrize("bitrate", [192, 256, 320])
def test_real_mp3_round_trip_preserves_gapless_sample_count(
    audio_sample, tmp_path, ffmpeg_available, bitrate,
):
    import numpy as np

    source, expected, rate = audio_sample
    destination = tmp_path / f"出力 & music {bitrate}.mp3"
    result = audio_files.encode_mp3(source, destination, bitrate_kbps=bitrate, sample_rate=rate)
    info = audio_files.probe_audio(destination)
    decoded = audio_files.decode_audio(destination, sample_rate=rate)
    assert result["codec"] == info["codec"] == "mp3"
    assert result["encoded_sample_rate"] == info["sample_rate"] == rate
    assert result["encoded_channels"] == info["channels"] == 2
    assert result["bitrate_kbps"] == bitrate
    assert result["audio_bytes"] == destination.stat().st_size < source.stat().st_size
    assert result["encoder"] == "libmp3lame" and result["gapless_metadata"] is True
    assert decoded.dtype == np.float32 and decoded.shape == expected.shape
    assert np.isfinite(decoded).all()
    assert np.corrcoef(decoded[:, 0], expected[:, 0])[0, 1] > 0.995
    assert np.corrcoef(decoded[:, 1], expected[:, 1])[0, 1] > 0.995
    assert not list(tmp_path.glob(".*.mp3"))


def test_real_wav_decode_preserves_stereo_and_frame_count(audio_sample, ffmpeg_available):
    import numpy as np

    source, expected, rate = audio_sample
    decoded = audio_files.decode_audio(source, sample_rate=rate)
    assert decoded.shape == expected.shape
    quantized = (expected * 32767).astype("<i2").astype(np.float32) / 32768
    np.testing.assert_array_equal(decoded, quantized)


def test_real_mp3_cancelled_after_encoding_cleans_owned_temporary(
    audio_sample, tmp_path, ffmpeg_available,
):
    source, _, rate = audio_sample
    destination = tmp_path / "cancelled.mp3"

    def cancelled():
        return any(path.stat().st_size for path in tmp_path.glob(".cancelled.*.mp3"))

    with pytest.raises(audio_files.ExportCancelled):
        audio_files.encode_mp3(source, destination, sample_rate=rate, cancelled=cancelled)
    assert not destination.exists()
    assert not list(tmp_path.glob(".cancelled.*.mp3"))
    assert source.exists()


def test_existing_output_is_never_overwritten(audio_sample, tmp_path, ffmpeg_available):
    source, _, rate = audio_sample
    destination = tmp_path / "existing.mp3"
    destination.write_bytes(b"original music bytes")
    with pytest.raises(audio_files.AudioFileError, match="既に存在"):
        audio_files.encode_mp3(source, destination, sample_rate=rate)
    assert destination.read_bytes() == b"original music bytes"
    assert not list(tmp_path.glob(".existing.*.mp3"))


def test_concurrent_publication_cannot_replace_another_result(tmp_path, monkeypatch):
    source = tmp_path / "temporary.mp3"
    source.write_bytes(b"new result")
    destination = tmp_path / "result.mp3"
    link = audio_files.os.link

    def race(source, destination):
        destination.write_bytes(b"other writer")
        link(source, destination)

    monkeypatch.setattr(audio_files.os, "link", race)
    with pytest.raises(audio_files.AudioFileError, match="既に存在"):
        audio_files._publish_new(source, destination)
    assert destination.read_bytes() == b"other writer"
    assert source.read_bytes() == b"new result"


def test_immediate_cancel_does_not_start_a_process(tmp_path, monkeypatch):
    def unexpected(*_args, **_kwargs):
        pytest.fail("a cancelled export started a subprocess")

    monkeypatch.setattr(audio_files.subprocess, "Popen", unexpected)
    with pytest.raises(audio_files.ExportCancelled):
        audio_files.encode_mp3(tmp_path / "missing.wav", tmp_path / "out.mp3",
                               cancelled=lambda: True)


def test_mid_process_cancellation_terminates_and_reaps_only_owned_process(monkeypatch):
    state = {"cancelled": False}

    class RunningProcess:
        returncode = None
        terminated = False
        waited = False

        def communicate(self, timeout=None):
            if not self.terminated:
                state["cancelled"] = True
                raise subprocess.TimeoutExpired("ffmpeg", timeout)
            self.waited = True
            self.returncode = -1
            return b"", b""

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

    process = RunningProcess()
    commands = []

    class FakeJob:
        assigned = False
        closed = False

        def assign(self, child):
            assert child is process
            self.assigned = True

        def close(self):
            self.closed = True

    job = FakeJob()

    def spawn(command, **kwargs):
        commands.append((command, kwargs))
        return process

    monkeypatch.setattr(audio_files.subprocess, "Popen", spawn)
    monkeypatch.setattr(audio_files, "WindowsChildJob", lambda: job)
    with pytest.raises(audio_files.ExportCancelled):
        audio_files._run_program(["ffmpeg", "literal & path"], cancelled=lambda: state["cancelled"])
    assert process.terminated and process.waited
    assert job.assigned and job.closed
    assert commands[0][0] == ["ffmpeg", "literal & path"]
    assert "shell" not in commands[0][1]
    if audio_files.os.name == "nt":
        assert commands[0][1]["creationflags"] & subprocess.CREATE_NO_WINDOW


def test_job_assignment_failure_terminates_owned_child_and_closes_job(monkeypatch):
    class Child:
        returncode = None
        terminated = False
        reaped = False

        def poll(self):
            return self.returncode

        def terminate(self):
            self.terminated = True

        def communicate(self, timeout):
            self.reaped = True
            self.returncode = -1
            return b"", b""

    class BrokenJob:
        closed = False

        def assign(self, child):
            raise PermissionError("job assignment rejected")

        def close(self):
            self.closed = True

    child = Child()
    job = BrokenJob()
    monkeypatch.setattr(audio_files.subprocess, "Popen", lambda *_args, **_kwargs: child)
    monkeypatch.setattr(audio_files, "WindowsChildJob", lambda: job)
    with pytest.raises(audio_files.AudioFileError, match="job assignment"):
        audio_files._run_program(["ffmpeg"])
    assert child.terminated and child.reaped and job.closed


@pytest.mark.skipif(os.name != "nt", reason="Windows job process lifetime semantics")
def test_real_windows_job_close_terminates_only_assigned_child():
    assigned = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    other = subprocess.Popen(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    job = process_tree.WindowsChildJob()
    try:
        job.assign(assigned)
        job.close()
        assigned.wait(timeout=5)
        assert other.poll() is None
        job.close()
    finally:
        job.close()
        for child in (assigned, other):
            if child.poll() is None:
                child.terminate()
            child.wait(timeout=5)


@pytest.mark.skipif(os.name != "nt", reason="Windows forced-supervisor job handle closure")
def test_real_ffmpeg_child_dies_when_supervisor_is_forcibly_terminated(
    tmp_path, ffmpeg_available,
):
    import ctypes
    from ctypes import wintypes

    ready = tmp_path / "child-pid.txt"
    code = '''
import sys
from pathlib import Path
from scripts.audio import audio_files
from scripts.audio.process_tree import WindowsChildJob
assign = WindowsChildJob.assign
def assigned(self, process):
    assign(self, process)
    Path(sys.argv[1]).write_text(str(process.pid), encoding="utf-8")
WindowsChildJob.assign = assigned
audio_files._run_program([
    str(audio_files.find_ffmpeg()), "-nostdin", "-v", "error", "-re",
    "-f", "lavfi", "-i", "anullsrc=r=44100:cl=stereo", "-t", "30", "-f", "null", "-",
], timeout=40)
'''
    supervisor = subprocess.Popen(
        [sys.executable, "-c", code, str(ready)], stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NO_WINDOW,
    )
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel.OpenProcess.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.WaitForSingleObject.restype = wintypes.DWORD
    kernel.TerminateProcess.argtypes = [wintypes.HANDLE, wintypes.UINT]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    child_handle = None
    try:
        deadline = time.monotonic() + 5
        while not ready.exists() and supervisor.poll() is None and time.monotonic() < deadline:
            time.sleep(0.025)
        assert ready.exists(), "supervisor did not assign its FFmpeg child"
        child_pid = int(ready.read_text(encoding="utf-8"))
        child_handle = kernel.OpenProcess(0x100000 | 0x0001, False, child_pid)
        assert child_handle, "FFmpeg child did not stay alive for the forced-stop test"
        assert kernel.WaitForSingleObject(child_handle, 0) == 258  # WAIT_TIMEOUT
        supervisor.terminate()
        supervisor.communicate(timeout=5)
        assert kernel.WaitForSingleObject(child_handle, 5000) == 0
    finally:
        if supervisor.poll() is None:
            supervisor.terminate()
        supervisor.communicate(timeout=5)
        if child_handle:
            if kernel.WaitForSingleObject(child_handle, 0) == 258:
                kernel.TerminateProcess(child_handle, 1)
                kernel.WaitForSingleObject(child_handle, 5000)
            kernel.CloseHandle(child_handle)


@pytest.mark.parametrize("filename", ["empty.wav", "invalid.mp3", "invalid.wav"])
def test_invalid_audio_raises_clear_error(tmp_path, ffmpeg_available, filename):
    path = tmp_path / filename
    path.write_bytes(b"" if filename == "empty.wav" else b"this is not audio")
    with pytest.raises(audio_files.AudioFileError):
        audio_files.probe_audio(path)


@pytest.mark.parametrize("bitrate", [0, 128, 193, True])
def test_invalid_bitrate_fails_before_reading_source(tmp_path, bitrate):
    with pytest.raises(audio_files.AudioFileError, match="ビットレート"):
        audio_files.encode_mp3(tmp_path / "missing.wav", tmp_path / "out.mp3", bitrate_kbps=bitrate)


@pytest.mark.parametrize("rate", [0, 7999, 192001, 44100.5, True])
def test_invalid_sample_rate_fails_before_reading_source(tmp_path, rate):
    with pytest.raises(audio_files.AudioFileError, match="サンプリング"):
        audio_files.decode_audio(tmp_path / "missing.wav", sample_rate=rate)


def test_missing_program_override_is_actionable(tmp_path, monkeypatch):
    monkeypatch.setenv("STABLE_AUDIO_FFMPEG", str(tmp_path / "missing.exe"))
    with pytest.raises(audio_files.AudioFileError, match="指定先"):
        audio_files.find_ffmpeg()


@pytest.mark.parametrize("changes", [
    {"duration": "1801"}, {"duration": "nan"}, {"duration": "-1"},
    {"sample_rate": "7999"}, {"channels": 9}, {"codec_name": ""},
])
def test_probe_rejects_unsafe_or_invalid_metadata_before_decoding(tmp_path, monkeypatch, changes):
    path = tmp_path / "header.wav"
    path.write_bytes(b"header")
    stream = {"codec_name": "pcm_s16le", "sample_rate": "44100", "channels": 2,
              "duration": "1", "bit_rate": "1411200"}
    stream.update(changes)
    monkeypatch.setattr(audio_files, "find_ffprobe", lambda: "ffprobe")
    monkeypatch.setattr(audio_files, "_run_program",
                        lambda *_args, **_kwargs: json.dumps({"streams": [stream]}).encode())
    with pytest.raises(audio_files.AudioFileError):
        audio_files.probe_audio(path)


def test_probe_and_decode_disallow_nonlocal_audio_formats(tmp_path):
    path = tmp_path / "playlist.m3u"
    path.write_text("https://example.test/remote.mp3", encoding="utf-8")
    with pytest.raises(audio_files.AudioFileError, match="WAV または MP3"):
        audio_files.probe_audio(path)


def test_seam_preview_preserves_true_boundary_and_separates_auditions():
    import numpy as np

    data = np.arange(2000, dtype=np.float32).reshape(-1, 2)
    preview = player.seam_preview(data, 1000, seconds=0.1, repeats=3)
    assert preview.shape == (3 * 200 + 2 * 400, 2)
    for offset in (0, 600, 1200):
        np.testing.assert_array_equal(preview[offset + 85:offset + 100], data[-15:])
        np.testing.assert_array_equal(preview[offset + 100:offset + 115], data[:15])
        assert (preview[offset] == 0).all()
        assert (preview[offset + 199] == 0).all()
    assert (preview[200:600] == 0).all()


def test_loop_callback_wraps_multiple_times_without_silence_or_duplicate_samples():
    import numpy as np

    data = np.arange(10, dtype=np.float32).reshape(5, 2)
    callback = player.playback_callback(data, loop=True, stop_exception=RuntimeError)
    first = np.empty((13, 2), dtype=np.float32)
    second = np.empty((7, 2), dtype=np.float32)
    callback(first, 13, None, None)
    callback(second, 7, None, None)
    actual = np.concatenate([first, second])
    np.testing.assert_array_equal(actual, np.tile(data, (4, 1)))


def test_loop_callback_plays_original_intro_once_then_repeats_a_to_b():
    import numpy as np

    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    callback = player.playback_callback(
        data, loop=True, stop_exception=RuntimeError, loop_start=3, loop_end=8,
    )
    blocks = [np.empty((count, 2), dtype=np.float32) for count in (6, 17, 5)]
    for block in blocks:
        callback(block, len(block), None, None)
    expected = np.concatenate([data[:8], np.tile(data[3:8], (4, 1))])
    np.testing.assert_array_equal(np.concatenate(blocks), expected)
    # The source's final frames after exclusive B are never played in loop mode.
    assert not np.isin(np.concatenate(blocks)[:, 0], data[8:, 0]).any()


def test_nonloop_mode_plays_full_file_with_loop_markers():
    import numpy as np

    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    callback = player.playback_callback(
        data, loop=False, stop_exception=RuntimeError, loop_start=3, loop_end=8,
    )
    output = np.empty_like(data)
    with pytest.raises(RuntimeError):
        callback(output, len(output), None, None)
    np.testing.assert_array_equal(output, data)


def test_playback_controller_seeks_forward_and_back_without_reloading_pcm():
    import numpy as np

    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    controller = player.PlaybackController(data, loop=False, stop_exception=RuntimeError)
    output = np.empty((4, 2), dtype=np.float32)
    controller.callback(output, len(output), None, None)
    np.testing.assert_array_equal(output, data[:4])
    controller.seek(7)
    forward = np.empty((2, 2), dtype=np.float32)
    controller.callback(forward, len(forward), None, None)
    np.testing.assert_array_equal(forward, data[7:9])
    controller.seek(1)
    backward = np.empty((6, 2), dtype=np.float32)
    controller.callback(backward, len(backward), None, None)
    np.testing.assert_array_equal(backward, data[1:7])
    assert controller.data is data and controller.snapshot()["position_sample"] == 7


def test_playback_controller_seek_preserves_a_to_b_wrap_and_counts_each_wrap():
    import numpy as np

    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    controller = player.PlaybackController(
        data, loop=True, stop_exception=RuntimeError, loop_start=3, loop_end=8,
    )
    controller.seek(2)
    output = np.empty((21, 2), dtype=np.float32)
    controller.callback(output, len(output), None, None)
    np.testing.assert_array_equal(output, np.concatenate([data[2:8], np.tile(data[3:8], (3, 1))]))
    snapshot = controller.snapshot()
    assert snapshot["position_sample"] == 3 and snapshot["wrap_count"] == 4
    assert controller.seek(9) == 3


def test_seek_commands_acknowledge_each_sequence_once_and_ignore_stale_or_invalid():
    import numpy as np

    controller = player.PlaybackController(np.zeros((10, 2), dtype=np.float32),
                                            loop=False, stop_exception=RuntimeError)
    assert controller.apply_command({"sequence": 2, "seek_sample": 5}) is True
    output = np.empty((2, 2), dtype=np.float32)
    controller.callback(output, len(output), None, None)
    assert controller.apply_command({"sequence": 2, "seek_sample": 5}) is False
    assert controller.apply_command({"sequence": 1, "seek_sample": 0}) is False
    assert controller.snapshot()["position_sample"] == 7
    for invalid in ({"sequence": 3, "seek_sample": 10}, {"sequence": True, "seek_sample": 1},
                    {"sequence": 3, "seek_sample": False}, {"sequence": 3}, [], None):
        assert controller.apply_command(invalid) is False
    assert controller.snapshot()["command_sequence"] == 2
    assert controller.apply_command({"sequence": 4, "seek_sample": 1}) is True
    assert controller.snapshot()["position_sample"] == 1


def test_concurrent_seek_and_pcm_callbacks_keep_each_buffer_coherent():
    from concurrent.futures import ThreadPoolExecutor

    import numpy as np

    data = np.repeat(np.arange(2500, dtype=np.float32)[:, None], 2, axis=1)
    controller = player.PlaybackController(
        data, loop=True, stop_exception=RuntimeError, loop_start=100, loop_end=2000,
    )

    def render():
        for _index in range(300):
            output = np.empty((127, 2), dtype=np.float32)
            controller.callback(output, len(output), None, None)
            labels = output[:, 0]
            assert (0 <= labels).all() and (labels < 2000).all()
            following = np.where(labels[:-1] == 1999, 100, labels[:-1] + 1)
            np.testing.assert_array_equal(labels[1:], following)
            np.testing.assert_array_equal(output[:, 0], output[:, 1])

    def seek():
        for sequence in range(1, 301):
            assert controller.apply_command({"sequence": sequence,
                                             "seek_sample": sequence * 137 % 2500})

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [executor.submit(render), executor.submit(seek)]
        for future in futures:
            future.result(timeout=5)
    snapshot = controller.snapshot()
    assert snapshot["command_sequence"] == 300 and 0 <= snapshot["position_sample"] < 2000


def test_atomic_control_poll_and_status_publish_do_not_run_in_callback(tmp_path, monkeypatch):
    import numpy as np

    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    controller = player.PlaybackController(data, loop=False, stop_exception=RuntimeError)
    command = tmp_path / "commands.json"
    status = tmp_path / "status.json"
    player.write_json(command, {"sequence": 1, "seek_sample": 3})
    assert player.poll_control(command, controller) is True
    assert player.poll_control(command, controller) is False
    report = player.publish_status(status, {"sample_rate": 44100, "mode": "normal"}, controller)
    assert json.loads(status.read_text(encoding="utf-8")) == report
    assert report["position_sample"] == 3 and report["command_sequence"] == 1
    command.write_text('{"sequence":', encoding="utf-8")
    assert player.poll_control(command, controller) is False

    def unexpected_write(*_args, **_kwargs):
        pytest.fail("PortAudio callback attempted file I/O")

    monkeypatch.setattr(player, "write_json", unexpected_write)
    output = np.empty((2, 2), dtype=np.float32)
    controller.callback(output, 2, None, None)
    np.testing.assert_array_equal(output, data[3:5])
    assert controller.snapshot()["position_sample"] == 5


@pytest.mark.parametrize("index,expected", [(0, 700), (99, 799), (100, 200), (199, 299),
                                           (200, 300), (599, 300), (600, 700), (700, 200),
                                           (1199, 300), (1200, 700), (1399, 299), (1400, 300)])
def test_three_seam_previews_map_true_segments_pauses_and_end_to_source(index, expected):
    assert player.seam_source_position(index, loop_start=200, loop_end=800,
                                       sample_rate=1000, seconds=0.1, repeats=3) == expected


def test_seam_status_exposes_original_timeline_and_separate_buffer_cursor(tmp_path):
    import numpy as np

    source = np.zeros((1000, 2), dtype=np.float32)
    preview = player.seam_preview(source, 1000, loop_start=200, loop_end=800,
                                  seconds=0.1, repeats=3)
    controller = player.PlaybackController(preview, loop=False, stop_exception=RuntimeError)
    controller.seek(650)
    info = {"mode": "seam_preview", "sample_rate": 1000, "source_loop_start_sample": 200,
            "source_loop_end_sample": 800, "seam_seconds": 0.1, "seam_repeats": 3,
            "playback_sample_count": 1000, "sample_count": len(preview), "seek_enabled": False}
    status = player.publish_status(tmp_path / "seam.json", info, controller)
    assert status["position_sample"] == 750 and status["buffer_position_sample"] == 650
    assert status["total_samples"] == 1000 and status["sample_count"] == 1400
    assert status["loop_start_sample"] == 200 and status["loop_end_sample"] == 800


def test_main_status_records_loading_playing_and_end_using_fake_device(
    tmp_path, monkeypatch, capsys,
):
    import numpy as np

    data = np.arange(20, dtype=np.float32).reshape(10, 2)
    monkeypatch.setattr(player, "decode_audio", lambda *_args, **_kwargs: data)
    monkeypatch.setattr(player.signal, "signal", lambda *_args: None)
    reports = []
    write_json = player.write_json

    def record(path, report):
        reports.append(dict(report))
        write_json(path, report)

    monkeypatch.setattr(player, "write_json", record)
    output = np.empty((10, 2), dtype=np.float32)

    class FakeStream:
        latency = 0.05

        def __init__(self, *, callback, finished_callback, **_kwargs):
            self.callback = callback
            self.finished_callback = finished_callback

        def __enter__(self):
            try:
                self.callback(output, len(output), None, None)
            except RuntimeError:
                self.finished_callback()
            return self

        def __exit__(self, *_args):
            pass

    fake_device = SimpleNamespace(OutputStream=FakeStream, CallbackStop=RuntimeError,
                                  PortAudioError=OSError)
    monkeypatch.setitem(sys.modules, "sounddevice", fake_device)
    status = tmp_path / "playback.json"
    assert player.main(["--input", "fake.wav", "--start-sample", "2",
                        "--status-file", str(status)]) == 0
    np.testing.assert_array_equal(output[:8], data[2:])
    assert (output[8:] == 0).all()
    assert [report["state"] for report in reports] == ["loading", "playing", "ended"]
    final = json.loads(status.read_text(encoding="utf-8"))
    assert final["position_sample"] == 10 and final["total_samples"] == 10
    assert final["device_latency_seconds"] == 0.05
    capsys.readouterr()


@pytest.mark.parametrize("start,end", [(-1, 8), (8, 8), (9, 8), (0, 11), (True, 8), (0, False),
                                       (0.5, 8), (0, 8.5)])
def test_invalid_loop_markers_are_rejected_without_clamping(start, end):
    import numpy as np

    data = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(audio_files.AudioFileError, match="ループ範囲"):
        player.playback_callback(data, loop=True, stop_exception=RuntimeError,
                                 loop_start=start, loop_end=end)


def test_seam_preview_uses_exclusive_b_to_a_not_file_end_to_zero():
    import numpy as np

    data = np.arange(2000, dtype=np.float32).reshape(1000, 2)
    preview = player.seam_preview(data, 1000, seconds=0.1, repeats=1,
                                  loop_start=200, loop_end=800)
    np.testing.assert_array_equal(preview[85:100], data[785:800])
    np.testing.assert_array_equal(preview[100:115], data[200:215])


def test_short_loop_seam_preview_never_reads_intro_or_after_b():
    import numpy as np

    data = np.arange(2000, dtype=np.float32).reshape(1000, 2)
    preview = player.seam_preview(data, 1000, seconds=3, repeats=1,
                                  loop_start=200, loop_end=210)
    assert len(preview) == 20
    np.testing.assert_array_equal(preview[5:10], data[205:210])
    np.testing.assert_array_equal(preview[10:15], data[200:205])


def test_legacy_intro_restoration_preserves_prefix_gain_and_loop_once():
    import numpy as np

    source = np.arange(20, dtype=np.float32).reshape(10, 2)
    candidate = np.arange(100, 108, dtype=np.float32).reshape(4, 2)
    combined = player.prepend_intro(candidate, source, samples=3, gain=0.5)
    np.testing.assert_array_equal(combined[:3], source[:3] * 0.5)
    np.testing.assert_array_equal(combined[3:], candidate)
    callback = player.playback_callback(
        combined, loop=True, stop_exception=RuntimeError, loop_start=3, loop_end=7,
    )
    output = np.empty((15, 2), dtype=np.float32)
    callback(output, len(output), None, None)
    np.testing.assert_array_equal(output, np.concatenate([source[:3] * 0.5,
                                                         np.tile(candidate, (3, 1))]))


def test_legacy_playback_attenuation_preserves_relative_intro_and_loop_levels():
    import numpy as np

    source = np.full((3, 2), 2.0, dtype=np.float32)
    candidate = np.full((4, 2), 0.5, dtype=np.float32)
    composed = player.prepend_intro(candidate, source, samples=2, gain=0.8)
    original = composed.copy()
    attenuated, gain = player.attenuate_playback(composed)
    assert gain == pytest.approx(0.99 / 1.6)
    np.testing.assert_allclose(attenuated, original * gain, atol=1e-7)
    assert float(np.max(np.abs(attenuated))) == pytest.approx(0.99)
    assert attenuated[0, 0] / attenuated[2, 0] == pytest.approx(1.6 / 0.5)
    np.testing.assert_array_equal(composed, original)


@pytest.mark.parametrize("peak", [0, 0.1, 0.99, 1.0])
def test_legacy_playback_never_amplifies_quiet_audio(peak):
    import numpy as np

    data = np.full((10, 2), peak, dtype=np.float32)
    result, gain = player.attenuate_playback(data)
    assert gain == 1.0 and result is data


def test_player_validate_only_reports_uniform_legacy_playback_attenuation(monkeypatch, capsys):
    import numpy as np

    candidate = np.full((4, 2), 0.4, dtype=np.float32)
    original = np.full((6, 2), 1.5, dtype=np.float32)
    monkeypatch.setattr(player, "decode_audio",
                        lambda path, **_kwargs: original if path == "source.wav" else candidate)
    assert player.main([
        "--input", "candidate.mp3", "--validate-only", "--loop",
        "--intro-input", "source.wav", "--intro-samples", "2", "--intro-gain", "0.8",
        "--loop-start-sample", "2", "--loop-end-sample", "6",
    ]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["playback_gain"] == pytest.approx(0.99 / 1.2)
    assert info["loop_start_sample"] == 2 and info["loop_end_sample"] == 6


@pytest.mark.parametrize("samples,gain", [(-1, 1), (11, 1), (True, 1), (3, float("nan")),
                                          (3, float("inf")), (3, -1)])
def test_legacy_intro_rejects_invalid_length_and_gain(samples, gain):
    import numpy as np

    source = np.zeros((10, 2), dtype=np.float32)
    with pytest.raises(audio_files.AudioFileError):
        player.prepend_intro(source, source, samples, gain)


def test_nonloop_callback_fills_tail_then_finishes():
    import numpy as np

    data = np.ones((5, 2), dtype=np.float32)
    callback = player.playback_callback(data, loop=False, stop_exception=RuntimeError)
    output = np.empty((7, 2), dtype=np.float32)
    with pytest.raises(RuntimeError):
        callback(output, 7, None, None)
    np.testing.assert_array_equal(output[:5], data)
    assert (output[5:] == 0).all()


def test_player_validate_only_checks_real_decoder_without_output_device(
    audio_sample, ffmpeg_available, capsys,
):
    source, expected, rate = audio_sample
    assert player.main(["--input", str(source), "--validate-only", "--loop"]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["sample_count"] == len(expected) and info["sample_rate"] == rate
    assert info["loop"] is True


def test_player_validate_only_supports_real_intro_and_marked_loop(
    audio_sample, ffmpeg_available, capsys,
):
    source, expected, rate = audio_sample
    prefix = 231
    start, end = prefix + 12000, prefix + 33000
    assert player.main([
        "--input", str(source), "--validate-only", "--loop",
        "--intro-input", str(source), "--intro-samples", str(prefix), "--intro-gain", "0.5",
        "--loop-start-sample", str(start), "--loop-end-sample", str(end),
    ]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["sample_count"] == len(expected) + prefix
    assert info["intro_samples"] == prefix
    assert info["loop_start_sample"] == start and info["loop_end_sample"] == end
    assert info["period_seconds"] == (end - start) / rate


def test_validate_only_applies_latest_seek_command_and_publishes_position(
    tmp_path, monkeypatch, capsys,
):
    import numpy as np

    data = np.zeros((100, 2), dtype=np.float32)
    monkeypatch.setattr(player, "decode_audio", lambda *_args, **_kwargs: data)
    control, status = tmp_path / "control.json", tmp_path / "status.json"
    player.write_json(control, {"sequence": 7, "seek_sample": 42})
    assert player.main([
        "--input", "fake.wav", "--validate-only", "--start-sample", "12",
        "--status-file", str(status), "--control-file", str(control),
    ]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["position_sample"] == 42 and info["command_sequence"] == 7
    assert info["state"] == "ended" and info["cursor_basis"] == "next_buffer"
    assert json.loads(status.read_text(encoding="utf-8")) == info


@pytest.mark.parametrize("position", [-1, 100])
def test_invalid_initial_seek_reports_error_before_opening_device(
    tmp_path, monkeypatch, capsys, position,
):
    import numpy as np

    monkeypatch.setattr(player, "decode_audio",
                        lambda *_args, **_kwargs: np.zeros((100, 2), dtype=np.float32))
    status = tmp_path / "status.json"
    assert player.main(["--input", "fake.wav", "--validate-only", "--start-sample", str(position),
                        "--status-file", str(status)]) == 1
    report = json.loads(status.read_text(encoding="utf-8"))
    assert report["state"] == "error" and "再生位置" in report["error"]
    capsys.readouterr()


def test_player_validate_only_seam_preview_uses_markers(audio_sample, ffmpeg_available, capsys):
    source, expected, rate = audio_sample
    assert player.main([
        "--input", str(source), "--validate-only", "--seam-preview", "--seam-seconds", "0.1",
        "--loop-start-sample", "10000", "--loop-end-sample", "30000",
    ]) == 0
    info = json.loads(capsys.readouterr().out)
    assert info["sample_count"] == round(rate * 0.2) * 3 + round(rate * 0.4) * 2
    assert info["playback_sample_count"] == len(expected)
    assert info["loop_start_sample"] == 10000 and info["loop_end_sample"] == 30000


def test_module_import_keeps_gui_free_of_numerical_and_gpu_dependencies():
    result = subprocess.run([
        sys.executable, "-c",
        ("import sys; from scripts.audio import audio_files, player; "
         "assert not any(name in sys.modules for name in "
         "('numpy', 'torch', 'diffusers', 'sounddevice'))"),
    ], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
