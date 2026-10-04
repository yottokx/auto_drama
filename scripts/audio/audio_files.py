"""CPU audio conversion for the independent BGM experiment.

Imports stay in the standard library until decoding is requested, so the root
GUI can check FFmpeg availability without importing the audio runtime.
"""

from __future__ import annotations

import json
import math
import os
import shutil
import subprocess
import tempfile
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

try:
    from .process_tree import WindowsChildJob
except ImportError:
    from process_tree import WindowsChildJob

MAX_AUDIO_SECONDS = 1800.0
MAX_AUDIO_BYTES = 512 * 1024 * 1024
MP3_BITRATES = (192, 256, 320)


class AudioFileError(RuntimeError):
    """An audio file or codec error suitable for the test GUI."""


class ExportCancelled(AudioFileError):
    """The caller cancelled a conversion before publication."""


def _find_program(name: str) -> Path:
    override = os.environ.get(f"STABLE_AUDIO_{name.upper()}", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if not candidate.is_file():
            raise AudioFileError(f"{name} の指定先が見つかりません: {candidate}")
        return candidate.resolve()
    candidates = [shutil.which(name), f"C:/ffmpeg/bin/{name}.exe"]
    # FFprobe is often installed beside an FFmpeg executable outside PATH.
    if name == "ffprobe":
        ffmpeg = os.environ.get("STABLE_AUDIO_FFMPEG", "") or shutil.which("ffmpeg")
        if ffmpeg:
            candidates.append(str(Path(ffmpeg).with_name("ffprobe.exe" if os.name == "nt"
                                                       else "ffprobe")))
    for value in candidates:
        if value and Path(value).is_file():
            return Path(value).resolve()
    raise AudioFileError(
        f"{name} が見つかりません。FFmpeg を PATH または C:/ffmpeg/bin に配置してください。"
    )


def find_ffmpeg() -> Path:
    return _find_program("ffmpeg")


def find_ffprobe() -> Path:
    return _find_program("ffprobe")


def _check_cancelled(cancelled: Callable[[], bool] | None) -> None:
    if cancelled is not None and cancelled():
        raise ExportCancelled("音声の変換をキャンセルしました。")


def _run_program(
    command: Sequence[str], *, cancelled: Callable[[], bool] | None = None,
    timeout: float = 120.0,
) -> bytes:
    """Capture both pipes without deadlocking, and reap our own process on exit."""
    _check_cancelled(cancelled)
    options: dict[str, Any] = {}
    if os.name == "nt":
        options["creationflags"] = subprocess.CREATE_NO_WINDOW
    process: subprocess.Popen[bytes] | None = None
    job: WindowsChildJob | None = None
    try:
        job = WindowsChildJob()
        process = subprocess.Popen(
            list(command), stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, **options,
        )
        job.assign(process)
        deadline = time.monotonic() + timeout
        while True:
            _check_cancelled(cancelled)
            if time.monotonic() >= deadline:
                raise AudioFileError("FFmpeg の処理がタイムアウトしました。")
            try:
                output, error = process.communicate(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                continue
        _check_cancelled(cancelled)
        if process.returncode:
            message = error.decode("utf-8", errors="replace").strip()[-3000:]
            raise AudioFileError(f"音声変換に失敗しました: {message}")
        return output
    except OSError as exc:
        raise AudioFileError(f"FFmpeg を起動できません: {exc}") from exc
    finally:
        try:
            if process is not None and process.poll() is None:
                process.terminate()
                try:
                    process.communicate(timeout=2.0)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()
        finally:
            if job is not None:
                job.close()


def _audio_path(path: str | Path) -> Path:
    result = Path(path).expanduser().resolve()
    if not result.is_file():
        raise AudioFileError(f"音声ファイルが見つかりません: {result}")
    if result.suffix.lower() not in {".wav", ".mp3"}:
        raise AudioFileError("音声ファイルは WAV または MP3 を選択してください。")
    size = result.stat().st_size
    if size <= 0 or size > MAX_AUDIO_BYTES:
        raise AudioFileError("音声ファイルが空、または上限 512 MiB を超えています。")
    return result


def _finite_positive(value: Any, name: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise AudioFileError(f"音声の {name} を取得できません。") from exc
    if not math.isfinite(number) or number <= 0:
        raise AudioFileError(f"音声の {name} が不正です。")
    return number


def probe_audio(
    path: str | Path, *, cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Read bounded local-file metadata; no network protocols or playlist reads."""
    source = _audio_path(path)
    output = _run_program([
        str(find_ffprobe()), "-v", "error", "-protocol_whitelist", "file,pipe",
        "-select_streams", "a:0", "-show_entries",
        "stream=codec_name,sample_rate,channels,duration,bit_rate:format=duration,bit_rate",
        "-of", "json", str(source),
    ], cancelled=cancelled, timeout=30.0)
    try:
        metadata = json.loads(output)
        streams = metadata.get("streams", [])
        if not isinstance(streams, list) or len(streams) != 1:
            raise AudioFileError("音声ストリームが見つかりません。")
        stream = streams[0]
        container = metadata.get("format", {})
        duration = _finite_positive(stream.get("duration", container.get("duration")), "長さ")
        rate = _finite_positive(stream.get("sample_rate"), "サンプリング周波数")
        channels = _finite_positive(stream.get("channels"), "チャンネル数")
        if duration > MAX_AUDIO_SECONDS:
            raise AudioFileError("音声の長さが上限 30 分を超えています。")
        if not rate.is_integer() or not 8000 <= rate <= 192000:
            raise AudioFileError("音声のサンプリング周波数が対応範囲外です。")
        if not channels.is_integer() or not 1 <= channels <= 8:
            raise AudioFileError("音声のチャンネル数が対応範囲外です。")
        codec = stream.get("codec_name", "")
        if not isinstance(codec, str) or not codec:
            raise AudioFileError("音声コーデックを取得できません。")
        bit_rate = stream.get("bit_rate", container.get("bit_rate", "0"))
        return {
            "codec": codec, "sample_rate": int(rate), "channels": int(channels),
            "duration_seconds": duration, "audio_bytes": source.stat().st_size,
            "bit_rate": int(bit_rate or 0),
        }
    except (json.JSONDecodeError, TypeError, ValueError, AttributeError) as exc:
        raise AudioFileError("FFprobe の音声情報を読み取れません。") from exc


def _sample_rate(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 8000 <= value <= 192000:
        raise AudioFileError("サンプリング周波数は 8000〜192000 の整数です。")
    return value


def decode_audio(
    path: str | Path, *, sample_rate: int = 44100,
    cancelled: Callable[[], bool] | None = None,
) -> Any:
    """Decode once to stereo float32 PCM, respecting MP3 delay/padding metadata."""
    import numpy as np

    rate = _sample_rate(sample_rate)
    source = _audio_path(path)
    info = probe_audio(source, cancelled=cancelled)
    if info["duration_seconds"] * rate * 8 > MAX_AUDIO_BYTES:
        raise AudioFileError("展開後の音声が上限 512 MiB を超えます。")
    raw = _run_program([
        str(find_ffmpeg()), "-nostdin", "-v", "error", "-protocol_whitelist", "file,pipe",
        "-i", str(source), "-map", "0:a:0", "-vn", "-sn", "-dn", "-ac", "2",
        "-ar", str(rate), "-c:a", "pcm_f32le", "-f", "f32le", "pipe:1",
    ], cancelled=cancelled, timeout=max(60.0, info["duration_seconds"]))
    _check_cancelled(cancelled)
    if not raw or len(raw) % 8 or len(raw) > MAX_AUDIO_BYTES:
        raise AudioFileError("展開した PCM 音声が空、または不正な長さです。")
    data = np.frombuffer(raw, dtype="<f4").reshape(-1, 2).copy()
    if not np.isfinite(data).all():
        raise AudioFileError("展開した音声に非有限値が含まれています。")
    return data


def _publish_new(source: Path, destination: Path) -> None:
    """Publish a complete same-volume file atomically without ever overwriting."""
    deadline = time.monotonic() + 1.5
    while True:
        try:
            os.link(source, destination)
            return
        except FileExistsError as exc:
            raise AudioFileError(f"保存先が既に存在します: {destination}") from exc
        except OSError as exc:
            if getattr(exc, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                raise AudioFileError(f"音声ファイルを保存できません: {exc}") from exc
            time.sleep(0.025)


def encode_mp3(
    source_pcm_wav: str | Path, destination_mp3: str | Path, *, bitrate_kbps: int = 192,
    sample_rate: int = 44100, cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Encode a temporary PCM WAV, verify it, then publish the MP3 exactly once."""
    _check_cancelled(cancelled)
    rate = _sample_rate(sample_rate)
    if isinstance(bitrate_kbps, bool) or bitrate_kbps not in MP3_BITRATES:
        raise AudioFileError("MP3 ビットレートは 192 / 256 / 320 kbps です。")
    source = _audio_path(source_pcm_wav)
    if source.suffix.lower() != ".wav":
        raise AudioFileError("MP3 変換元には PCM WAV を指定してください。")
    original = probe_audio(source, cancelled=cancelled)
    if not original["codec"].startswith("pcm_"):
        raise AudioFileError("MP3 変換元が PCM WAV ではありません。")
    destination = Path(destination_mp3).expanduser().resolve()
    if destination.suffix.lower() != ".mp3":
        raise AudioFileError("MP3 の保存先には .mp3 を指定してください。")
    if destination.exists():
        raise AudioFileError(f"保存先が既に存在します: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=destination.parent, prefix=f".{destination.stem}.", suffix=".mp3", delete=False,
        ) as stream:
            temporary = Path(stream.name)
        _run_program([
            str(find_ffmpeg()), "-nostdin", "-v", "error", "-y",
            "-protocol_whitelist", "file,pipe", "-i", str(source), "-map", "0:a:0",
            "-vn", "-sn", "-dn", "-map_metadata", "-1", "-ac", "2", "-ar", str(rate),
            "-c:a", "libmp3lame", "-b:a", f"{bitrate_kbps}k", "-write_xing", "1",
            "-id3v2_version", "3", str(temporary),
        ], cancelled=cancelled, timeout=max(60.0, original["duration_seconds"]))
        result = probe_audio(temporary, cancelled=cancelled)
        if result["codec"] != "mp3" or result["sample_rate"] != rate or result["channels"] != 2:
            raise AudioFileError("作成した MP3 のコーデックまたは音声形式が不正です。")
        # The MP3 container includes its encoder delay; FFmpeg trims it on decode.
        if abs(result["duration_seconds"] - original["duration_seconds"]) > 0.2:
            raise AudioFileError("作成した MP3 の長さが原音と一致しません。")
        _check_cancelled(cancelled)
        _publish_new(temporary, destination)
        return {
            "codec": "mp3", "bitrate_kbps": bitrate_kbps, "audio_bytes": result["audio_bytes"],
            "encoded_duration_seconds": result["duration_seconds"],
            "encoded_sample_rate": result["sample_rate"], "encoded_channels": result["channels"],
            "encoder": "libmp3lame", "gapless_metadata": True,
        }
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
