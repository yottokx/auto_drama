import io
import math
import shutil
import struct
import subprocess
import wave

import pytest
from PIL import Image

from services.coordinator.adjustment_media import normalize_portrait, normalize_reference_voice


def recording(*, rate=44100, seconds=0.5, channels=2, silent=False):
    output = io.BytesIO()
    with wave.open(output, "wb") as audio:
        audio.setnchannels(channels)
        audio.setsampwidth(2)
        audio.setframerate(rate)
        audio.writeframes(b"".join(struct.pack("<h", 0 if silent else int(
            5000 * math.sin(2 * math.pi * 220 * index / rate))) * channels
            for index in range(int(rate * seconds))))
    return output.getvalue()


@pytest.mark.parametrize("format", ["PNG", "WEBP", "JPEG"])
def test_portrait_upload_is_decoded_and_normalized_without_removing_background(format):
    output = io.BytesIO()
    Image.new("RGB", (32, 48), "red").save(output, format=format)
    png, metadata = normalize_portrait(output.getvalue())
    with Image.open(io.BytesIO(png)) as image:
        assert image.format == "PNG" and image.mode == "RGBA" and image.size == (32, 48)
        assert image.getchannel("A").getextrema() == (255, 255)
    assert not metadata["has_transparency"]


def test_portrait_upload_preserves_alpha_and_rejects_empty_or_invalid_images():
    source = Image.new("RGBA", (32, 48))
    source.putpixel((12, 20), (100, 50, 20, 180))
    output = io.BytesIO()
    source.save(output, format="PNG")
    png, metadata = normalize_portrait(output.getvalue())
    assert metadata["has_transparency"]
    with Image.open(io.BytesIO(png)) as result:
        assert result.getpixel((12, 20)) == (100, 50, 20, 180)
    for data in (b"", b"not an image", b"RIFF\x00\x00\x00\x00WEBP"):
        with pytest.raises(ValueError):
            normalize_portrait(data)
    output = io.BytesIO()
    Image.new("RGBA", (12, 12)).save(output, format="PNG")
    with pytest.raises(ValueError, match="画素"):
        normalize_portrait(output.getvalue())


def test_voice_upload_resamples_downmixes_and_preserves_duration():
    result, metadata = normalize_reference_voice(recording())
    with wave.open(io.BytesIO(result)) as audio:
        assert audio.getnchannels() == 1 and audio.getsampwidth() == 2
        assert audio.getframerate() == 24000 and 11999 <= audio.getnframes() <= 12000
    assert metadata["source_format"] == "WAV"


@pytest.mark.parametrize("content", [b"bad", recording(seconds=0.1), recording(silent=True), recording()[:-4]],
                         ids=["invalid", "too-short", "silent", "truncated"])
def test_invalid_reference_upload_never_becomes_a_candidate(content):
    with pytest.raises(ValueError):
        normalize_reference_voice(content)


def test_mp3_is_decoded_as_audio_if_local_decoder_is_available():
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        pytest.skip("ffmpeg is optional for MP3 uploads")
    mp3 = subprocess.run([ffmpeg, "-v", "error", "-f", "wav", "-i", "pipe:0", "-f", "mp3", "pipe:1"],
        input=recording(), capture_output=True, check=True, timeout=20,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout
    wav, metadata = normalize_reference_voice(mp3)
    assert metadata["source_format"] == "MP3" and 0.4 < metadata["duration_seconds"] < 0.7
    assert wav.startswith(b"RIFF")


def test_mp3_has_actionable_error_without_decoder(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda _: None)
    with pytest.raises(ValueError, match="WAV形式"):
        normalize_reference_voice(b"ID3\x00\x00\x00")
