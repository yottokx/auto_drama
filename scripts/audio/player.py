"""Independent CPU PCM player for MP3/WAV and uninterrupted loop comparison.

The GUI launches this with the audio runtime's Python. Terminating this owned
process stops playback. An MP3 is decoded once with its gapless metadata before
looping, so no codec or process restart occurs at a loop boundary.
"""

from __future__ import annotations

import argparse
import json
import math
import signal
import sys
import threading
from collections.abc import Mapping
from contextlib import suppress
from pathlib import Path
from typing import Any

try:
    from .audio_files import AudioFileError, decode_audio
    from .json_io import write_json
except ImportError:
    from audio_files import AudioFileError, decode_audio
    from json_io import write_json


def loop_bounds(data: Any, start: int = 0, end: int | None = None) -> tuple[int, int]:
    """Validate exclusive PCM sample markers without rounding or clamping."""
    if end is None:
        end = len(data)
    if (isinstance(start, bool) or isinstance(end, bool)
            or not isinstance(start, int) or not isinstance(end, int)
            or not 0 <= start < end <= len(data)):
        raise AudioFileError("ループ範囲は 0 <= A < B <= 音声サンプル数の整数で指定してください。")
    return start, end


def prepend_intro(data: Any, source: Any, samples: int, gain: float = 1.0) -> Any:
    """Restore a legacy candidate's original prefix once, without modifying it."""
    import numpy as np

    if (isinstance(samples, bool) or not isinstance(samples, int)
            or not 0 <= samples <= len(source)):
        raise AudioFileError("イントロのサンプル数が原音の範囲外です。")
    if isinstance(gain, bool) or not math.isfinite(gain) or not 0 <= gain <= 100:
        raise AudioFileError("イントロの音量係数は 0〜100 の有限値です。")
    if (data.ndim != 2 or source.ndim != 2 or data.shape[1] != 2 or source.shape[1] != 2):
        raise AudioFileError("イントロと候補にはステレオ音声が必要です。")
    if samples == 0:
        return data
    return np.concatenate([source[:samples] * gain, data]).astype(np.float32)


def attenuate_playback(data: Any) -> tuple[Any, float]:
    """Apply one uniform safety gain only when a composed legacy clip exceeds 1."""
    import numpy as np

    if data.size == 0 or not np.isfinite(data).all():
        raise AudioFileError("再生する音声が空、または非有限値を含んでいます。")
    peak = float(np.max(np.abs(data)))
    gain = 0.99 / peak if peak > 1.0 else 1.0
    return (data * gain if gain < 1.0 else data), gain


def seam_preview(
    data: Any, sample_rate: int, *, seconds: float = 3.0, repeats: int = 3,
    loop_start: int = 0, loop_end: int | None = None,
) -> Any:
    """Repeat the true tail/head seam; fade only the audition's outer edges.

Separate short auditions with silence so their own replay point cannot be
mistaken for a second loop boundary. The actual seam remains unmodified.
"""
    import numpy as np

    if not 0.1 <= seconds <= 15.0 or not 1 <= repeats <= 10:
        raise AudioFileError("継ぎ目試聴は前後 0.1〜15 秒、繰り返し 1〜10 回です。")
    if data.ndim != 2 or data.shape[1] != 2:
        raise AudioFileError("継ぎ目試聴に必要なステレオ音声がありません。")
    start, end = loop_bounds(data, loop_start, loop_end)
    window = min(max(1, round(seconds * sample_rate)), end - start)
    clip = np.concatenate([data[end - window:end], data[start:start + window]]).astype(np.float32)
    fade = min(int(sample_rate * 0.015), window // 2)
    if fade:
        ramp = np.linspace(0, 1, fade, dtype=np.float32)[:, None]
        clip[:fade] *= ramp
        clip[-fade:] *= ramp[::-1]
    pause = np.zeros((int(sample_rate * 0.4), 2), dtype=np.float32)
    chunks = []
    for index in range(repeats):
        if index:
            chunks.append(pause)
        chunks.append(clip)
    return np.concatenate(chunks)


def seam_source_position(
    sample: int, *, loop_start: int, loop_end: int, sample_rate: int = 44100,
    seconds: float = 3.0, repeats: int = 3,
) -> int:
    """Map a preview PCM cursor to the original B-tail/A-head timeline."""
    if not 0 <= loop_start < loop_end or not 0.1 <= seconds <= 15 or not 1 <= repeats <= 10:
        raise AudioFileError("継ぎ目試聴の範囲が不正です。")
    window = min(max(1, round(seconds * sample_rate)), loop_end - loop_start)
    pause = int(sample_rate * 0.4)
    total = 2 * window * repeats + pause * (repeats - 1)
    if isinstance(sample, bool) or not isinstance(sample, int) or not 0 <= sample <= total:
        raise AudioFileError("継ぎ目試聴の再生位置が範囲外です。")
    if sample == total:
        return loop_start + window
    local = sample % (2 * window + pause)
    if local < window:
        return loop_end - window + local
    if local < 2 * window:
        return loop_start + local - window
    return loop_start + window


class PlaybackController:
    """A PCM cursor shared by the main control thread and PortAudio callback.

The callback performs buffer copies and short lock-protected state updates. It
does no file I/O. Reported position is the next buffered PCM sample, so it can
lead audible playback by the output device's buffer latency.
"""

    def __init__(
        self, data: Any, *, loop: bool, stop_exception: type[Exception],
        loop_start: int = 0, loop_end: int | None = None, start_sample: int = 0,
    ) -> None:
        self.data = data
        self.loop = loop
        self.sample_count = len(data)
        self.loop_start, self.loop_end = loop_bounds(data, loop_start, loop_end)
        self.stop_exception = stop_exception
        self._lock = threading.Lock()
        self._position = self._seek_position(start_sample)
        self._wrap_count = 0
        self._command_sequence = 0
        self._ended = False

    def _seek_position(self, sample: int) -> int:
        if (isinstance(sample, bool) or not isinstance(sample, int)
                or not 0 <= sample < self.sample_count):
            raise AudioFileError("再生位置は 0 以上、音声サンプル数未満の整数です。")
        return self.loop_start if self.loop and sample >= self.loop_end else sample

    def seek(self, sample: int) -> int:
        position = self._seek_position(sample)
        with self._lock:
            self._position = position
            self._ended = False
        return position

    def apply_command(self, command: Any) -> bool:
        if not isinstance(command, Mapping):
            return False
        sequence = command.get("sequence")
        if isinstance(sequence, bool) or not isinstance(sequence, int) or sequence <= 0:
            return False
        try:
            position = self._seek_position(command.get("seek_sample"))
        except AudioFileError:
            return False
        with self._lock:
            if sequence <= self._command_sequence:
                return False
            self._position = position
            self._command_sequence = sequence
            self._ended = False
            return True

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "state": "ended" if self._ended else "playing",
                "position_sample": self._position, "total_samples": self.sample_count,
                "loop_start_sample": self.loop_start, "loop_end_sample": self.loop_end,
                "loop": self.loop, "wrap_count": self._wrap_count,
                "command_sequence": self._command_sequence,
            }

    def callback(self, outdata: Any, frames: int, _time: Any, _status: Any) -> None:
        end = self.loop_end if self.loop else self.sample_count
        with self._lock:
            written = 0
            while written < frames:
                if self._position == end:
                    if self.loop:
                        self._position = self.loop_start
                        self._wrap_count += 1
                    else:
                        outdata[written:] = 0
                        self._ended = True
                        raise self.stop_exception
                count = min(frames - written, end - self._position)
                outdata[written:written + count] = self.data[self._position:self._position + count]
                self._position += count
                written += count
            if self._position == end:
                if self.loop:
                    self._position = self.loop_start
                    self._wrap_count += 1
                else:
                    self._ended = True
                    raise self.stop_exception


def playback_callback(
    data: Any, *, loop: bool, stop_exception: type[Exception],
    loop_start: int = 0, loop_end: int | None = None,
) -> Any:
    """Compatible callback API: intro 0→B once, then continuously repeat A→B."""
    return PlaybackController(
        data, loop=loop, stop_exception=stop_exception, loop_start=loop_start, loop_end=loop_end,
    ).callback


def poll_control(path: Path | None, controller: PlaybackController) -> bool:
    """Apply each newest complete small seek command once, from the main thread."""
    if path is None:
        return False
    try:
        if path.stat().st_size > 4096:
            return False
        command = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    return controller.apply_command(command)


def publish_status(
    path: Path | None, info: Mapping[str, Any], controller: PlaybackController | None = None,
    *, state: str | None = None,
) -> dict[str, Any]:
    status = dict(info)
    if controller is not None:
        status.update(controller.snapshot())
        if info.get("mode") == "seam_preview":
            status["buffer_position_sample"] = status["position_sample"]
            status["position_sample"] = seam_source_position(
                status["buffer_position_sample"], loop_start=info["source_loop_start_sample"],
                loop_end=info["source_loop_end_sample"], sample_rate=info["sample_rate"],
                seconds=info["seam_seconds"], repeats=info["seam_repeats"],
            )
            status["total_samples"] = info["playback_sample_count"]
            status["loop_start_sample"] = info["source_loop_start_sample"]
            status["loop_end_sample"] = info["source_loop_end_sample"]
    if state is not None:
        status["state"] = state
    if path is not None:
        write_json(path, status)
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="BGM MP3/WAV PCM player")
    parser.add_argument("--input", required=True, help="MP3 or WAV input file")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--loop", action="store_true", help="Play intro once, then continuously repeat A-B")
    mode.add_argument("--seam-preview", action="store_true", help="Audition the tail/head seam")
    parser.add_argument("--seam-seconds", type=float, default=3.0)
    parser.add_argument("--repeat-count", type=int, default=3)
    parser.add_argument("--loop-start-sample", type=int, default=0, help="Loop A in decoded PCM samples")
    parser.add_argument("--loop-end-sample", type=int, help="Exclusive loop B in decoded PCM samples")
    parser.add_argument("--intro-input", help="Original MP3/WAV for restoring a legacy intro")
    parser.add_argument("--intro-samples", type=int, help="Samples to restore from the original start")
    parser.add_argument("--intro-gain", type=float, default=1.0)
    parser.add_argument("--start-sample", type=int, default=0, help="Initial decoded PCM position")
    parser.add_argument("--status-file", help="Atomic playback status JSON")
    parser.add_argument("--control-file", help="Atomic seek command JSON")
    parser.add_argument("--validate-only", action="store_true", help="Decode without playing")
    args = parser.parse_args(argv)
    status_path = Path(args.status_file) if args.status_file else None
    control_path = Path(args.control_file) if args.control_file else None
    controller = None
    rate = 44100
    info = {
        "sample_rate": rate, "position_sample": 0, "total_samples": 0,
        "loop_start_sample": args.loop_start_sample, "loop_end_sample": args.loop_end_sample,
        "loop": args.loop, "mode": "seam_preview" if args.seam_preview else "normal",
        "wrap_count": 0, "command_sequence": 0, "playback_gain": 1.0,
        "seek_enabled": not args.seam_preview, "cursor_basis": "next_buffer",
    }
    try:
        publish_status(status_path, info, state="loading")
        data = decode_audio(args.input, sample_rate=rate)
        intro_samples = 0
        playback_gain = 1.0
        if args.intro_input:
            if args.intro_samples is None:
                raise AudioFileError("イントロ復元には --intro-samples が必要です。")
            source = decode_audio(args.intro_input, sample_rate=rate)
            data = prepend_intro(data, source, args.intro_samples, args.intro_gain)
            intro_samples = args.intro_samples
            if intro_samples:
                data, playback_gain = attenuate_playback(data)
        elif args.intro_samples is not None or args.intro_gain != 1.0:
            raise AudioFileError("イントロ復元には --intro-input が必要です。")
        start, end = loop_bounds(data, args.loop_start_sample, args.loop_end_sample)
        playback_sample_count = len(data)
        if args.seam_preview:
            if args.start_sample:
                raise AudioFileError("継ぎ目試聴では再生位置の指定を利用できません。")
            data = seam_preview(
                data, rate, seconds=args.seam_seconds, repeats=args.repeat_count,
                loop_start=start, loop_end=end,
            )
        info.update({
            "sample_count": len(data), "duration_seconds": len(data) / rate,
            "channels": 2, "seam_preview": args.seam_preview,
            "loop_start_sample": start, "loop_end_sample": end,
            "period_seconds": (end - start) / rate, "intro_samples": intro_samples,
            "playback_gain": playback_gain, "playback_sample_count": playback_sample_count,
            "source_loop_start_sample": start, "source_loop_end_sample": end,
            "seam_seconds": args.seam_seconds, "seam_repeats": args.repeat_count,
        })
        controller = PlaybackController(
            data, loop=args.loop, stop_exception=RuntimeError,
            loop_start=start if not args.seam_preview else 0,
            loop_end=end if not args.seam_preview else None, start_sample=args.start_sample,
        )
        if not args.seam_preview:
            poll_control(control_path, controller)
        if args.validate_only:
            status = publish_status(status_path, info, controller, state="ended")
            print(json.dumps(status))
            return 0
        try:
            import sounddevice as sd
        except ImportError as exc:
            raise AudioFileError(
                "再生用 sounddevice がありません。Stable Audio 専用環境を準備してください。"
            ) from exc
        stopped = threading.Event()

        def stop(_signum: int, _frame: Any) -> None:
            stopped.set()

        signal.signal(signal.SIGINT, stop)
        signal.signal(signal.SIGTERM, stop)
        controller.stop_exception = sd.CallbackStop
        try:
            with sd.OutputStream(
                samplerate=rate, channels=2, dtype="float32", callback=controller.callback,
                finished_callback=stopped.set,
            ) as stream:
                info["device_latency_seconds"] = float(stream.latency)
                print(json.dumps(info), flush=True)
                publish_status(status_path, info, controller, state="playing")
                while not stopped.wait(0.1):
                    if not args.seam_preview:
                        poll_control(control_path, controller)
                    publish_status(status_path, info, controller)
        except sd.PortAudioError as exc:
            raise AudioFileError(f"再生デバイスを開けません: {exc}") from exc
        publish_status(status_path, info, controller, state="ended")
        return 0
    except KeyboardInterrupt:
        with suppress(OSError):
            publish_status(status_path, info, controller, state="ended")
        return 0
    except (AudioFileError, RuntimeError, OSError) as exc:
        info["error"] = str(exc)
        with suppress(OSError):
            publish_status(status_path, info, controller, state="error")
        print(f"再生に失敗しました: {exc}", file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
