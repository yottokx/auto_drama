"""Bounded, content-checked uploads for completed-work material candidates."""

from __future__ import annotations

import io
import shutil
import subprocess
import wave

from PIL import Image, ImageOps, UnidentifiedImageError

MAX_UPLOAD_BYTES = 32 * 1024 * 1024
MIN_REFERENCE_SECONDS = 0.25
MAX_REFERENCE_SECONDS = 30
REFERENCE_RATE = 24_000


def _bounded(data: bytes) -> None:
    if not data or len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("素材は空でない32MiB以内のファイルを選んでください。")


def normalize_portrait(data: bytes) -> tuple[bytes, dict]:
    """Decode the image, honor orientation, and preserve alpha without removing its background."""
    _bounded(data)
    try:
        with Image.open(io.BytesIO(data)) as source:
            original_format = source.format
            if original_format not in {"PNG", "WEBP", "JPEG"}:
                raise ValueError("立ち絵にはPNG・WebP・JPEGを使用してください。")
            if source.width > 4096 or source.height > 4096:
                raise ValueError("立ち絵の縦横はそれぞれ4096px以内にしてください。")
            if getattr(source, "is_animated", False):
                raise ValueError("立ち絵には静止画像を使用してください。")
            source.load()
            image = ImageOps.exif_transpose(source).convert("RGBA")
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise ValueError("画像を読み取れません。PNG・WebP・JPEGを選んでください。") from exc
    if image.getchannel("A").getbbox() is None:
        raise ValueError("立ち絵に表示できる画素がありません。")
    output = io.BytesIO()
    image.save(output, format="PNG")
    normalized = output.getvalue()
    _bounded(normalized)
    return normalized, {
        "source_format": original_format, "width": image.width, "height": image.height,
        "has_transparency": image.getchannel("A").getextrema()[0] < 255,
        "normalization": "oriented-rgba-png/1",
    }


def _wav_samples(data: bytes) -> bytes:
    # The coordinator is pinned to Python 3.12. audioop handles 8/16/24/32-bit
    # signedness, stereo downmix and sample-rate conversion without a GPU runtime.
    import audioop

    try:
        with wave.open(io.BytesIO(data), "rb") as source:
            channels, width, rate = source.getnchannels(), source.getsampwidth(), source.getframerate()
            frames = source.getnframes()
            if (channels not in {1, 2} or width not in {1, 2, 3, 4}
                    or not 8000 <= rate <= 192000 or source.getcomptype() != "NONE"):
                raise ValueError("音声はモノラルまたはステレオのPCM WAVを使用してください。")
            if not MIN_REFERENCE_SECONDS <= frames / rate <= MAX_REFERENCE_SECONDS:
                raise ValueError("基準音声は0.25〜30秒にしてください。")
            samples = source.readframes(frames)
            if len(samples) != frames * width * channels:
                raise ValueError("音声ファイルが途中で切れています。")
    except (wave.Error, EOFError) as exc:
        raise ValueError("PCM WAVを読み取れません。") from exc
    if width == 1:
        samples = audioop.bias(samples, 1, -128)
    samples = audioop.lin2lin(samples, width, 2)
    if channels == 2:
        samples = audioop.tomono(samples, 2, 0.5, 0.5)
    if rate != REFERENCE_RATE:
        samples, _ = audioop.ratecv(samples, 2, 1, rate, REFERENCE_RATE, None)
    return samples


def _mp3_samples(data: bytes) -> bytes:
    decoder = shutil.which("ffmpeg")
    if decoder is None:
        raise ValueError("この環境ではMP3を読み取れません。WAV形式でアップロードしてください。")
    try:
        decoded = subprocess.run([
            decoder, "-hide_banner", "-loglevel", "error", "-nostdin",
            "-protocol_whitelist", "pipe", "-f", "mp3", "-i", "pipe:0",
            "-map", "0:a:0", "-t", str(MAX_REFERENCE_SECONDS + 1),
            "-ac", "1", "-ar", str(REFERENCE_RATE), "-f", "s16le", "pipe:1",
        ], input=data, capture_output=True, timeout=20, check=True,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except (OSError, subprocess.SubprocessError) as exc:
        raise ValueError("MP3を読み取れません。音声ファイルを確認してください。") from exc
    return decoded.stdout


def normalize_reference_voice(data: bytes) -> tuple[bytes, dict]:
    """Keep the recording/transcript pairing while normalizing playable reference audio."""
    _bounded(data)
    is_wav = data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    is_mp3 = data[:3] == b"ID3" or (len(data) >= 2 and data[0] == 0xFF and data[1] & 0xE0 == 0xE0)
    if not is_wav and not is_mp3:
        raise ValueError("基準音声にはWAVまたはMP3を使用してください。")
    samples = _wav_samples(data) if is_wav else _mp3_samples(data)
    duration = len(samples) / 2 / REFERENCE_RATE
    if not MIN_REFERENCE_SECONDS - 1 / REFERENCE_RATE <= duration <= MAX_REFERENCE_SECONDS:
        raise ValueError("基準音声は0.25〜30秒にしてください。")
    import audioop

    if audioop.rms(samples, 2) < 1:
        raise ValueError("基準音声が無音です。声の入った音声を選んでください。")
    output = io.BytesIO()
    with wave.open(output, "wb") as target:
        target.setnchannels(1)
        target.setsampwidth(2)
        target.setframerate(REFERENCE_RATE)
        target.writeframes(samples)
    return output.getvalue(), {
        "source_format": "WAV" if is_wav else "MP3", "duration_seconds": duration,
        "sample_rate": REFERENCE_RATE, "channels": 1, "normalization": "mono-pcm16-wav/1",
    }
