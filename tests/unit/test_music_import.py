"""Import boundary checks with a fake dedicated runtime, without inference or codecs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from services.coordinator import music_import

SCENE_ID = "scene_001"


def report(scene_id=SCENE_ID):
    return {"ok": True, "result": {"scene_id": scene_id,
        "prompt": "Imported instrumental background music.", "loop_start_seconds": 5.0,
        "loop_end_seconds": 25.0, "duration_seconds": 25.0,
        "source_duration_seconds": 30.0, "sample_rate": 44100,
        "quality": {"needs_review": False}}, "provenance": {"backend": "imported"}}


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.report = report()
        self.returncode = 0
        self.write_report = True
        self.timeout = False

    def run(self, command, **options):
        request_path = Path(command[command.index("--import-request") + 1])
        request = json.loads(request_path.read_text(encoding="utf-8"))
        source = Path(request["source_audio"])
        output = Path(command[command.index("--output-dir") + 1])
        self.calls.append({"command": command, "options": options, "request": request,
            "source": source, "data": source.read_bytes(), "temporary": request_path.parent})
        if self.timeout:
            raise subprocess.TimeoutExpired(command, options["timeout"])
        output.mkdir(parents=True, exist_ok=True)
        if self.write_report:
            (output / "result.json").write_text(json.dumps(self.report), encoding="utf-8")
        (output / "source.mp3").write_bytes(b"ID3normalized source")
        (output / "music.mp3").write_bytes(b"ID3normalized loop")
        return SimpleNamespace(returncode=self.returncode)


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    python = tmp_path / "runtime/python.exe"
    python.parent.mkdir()
    python.write_bytes(b"fake dedicated runtime")
    config = tmp_path / "config/m2-generation.json"
    config.parent.mkdir()
    config.write_text(json.dumps({"music": {"python": "runtime/python.exe"}}), encoding="utf-8")
    monkeypatch.setattr(music_import, "ROOT", tmp_path)
    fake = FakeRuntime()
    monkeypatch.setattr(music_import.subprocess, "run", fake.run)
    return SimpleNamespace(root=tmp_path, python=python, config=config, fake=fake)


@pytest.mark.parametrize(("data", "extension"), [
    (b"RIFF\x00\x00\x00\x00WAVEfmt fixture", ".wav"),
    (b"RF64\xff\xff\xff\xffWAVEds64 fixture", ".wav"),
    (b"ID3mp3 fixture", ".mp3"),
    (b"\xff\xfbaudio frame fixture", ".mp3"),
])
def test_header_selects_supported_temporary_extension(runtime, data, extension):
    assets, result = music_import.normalize_music(data, SCENE_ID)
    call = runtime.fake.calls[0]
    assert call["source"].name == "input" + extension
    assert call["data"] == data
    assert call["request"]["scene_id"] == SCENE_ID
    assert call["request"]["loop_target_seconds"] == 60
    assert call["request"]["crossfade_seconds"] == 0.5
    assert call["command"][:5] == [str(runtime.python), "-X", "utf8", "-m",
        "services.worker.generation.music_runner"]
    assert call["options"]["cwd"] == runtime.root
    assert call["options"]["timeout"] == 180
    assert call["options"]["capture_output"] is True
    assert call["options"]["check"] is False
    assert not call["temporary"].exists()
    assert assets == {"music.mp3": b"ID3normalized loop", "source.mp3": b"ID3normalized source"}
    assert result["scene_id"] == SCENE_ID


@pytest.mark.parametrize("scene_id", [None, 7, "s" * 129])
def test_invalid_scene_input_is_rejected_before_starting_runtime(runtime, scene_id):
    with pytest.raises(ValueError):
        music_import.normalize_music(b"ID3fixture", scene_id)
    assert not runtime.fake.calls


def test_result_cannot_be_assigned_to_another_scene(runtime):
    runtime.fake.report = report("scene_002")
    with pytest.raises(ValueError, match="場面が一致"):
        music_import.normalize_music(b"ID3fixture", SCENE_ID)
    assert not runtime.fake.calls[0]["temporary"].exists()


@pytest.mark.parametrize("mutation", [
    {"scene_id": "invalid/scene"}, {"loop_start_seconds": 26.0},
    {"quality": {"needs_review": True}},
])
def test_invalid_runtime_result_is_not_adopted(runtime, mutation):
    runtime.fake.report["result"].update(mutation)
    with pytest.raises(ValueError):
        music_import.normalize_music(b"ID3fixture", SCENE_ID)
    assert not runtime.fake.calls[0]["temporary"].exists()


def test_timeout_is_bounded_and_temporary_audio_is_removed(runtime):
    runtime.fake.timeout = True
    with pytest.raises(ValueError, match="時間内に完了") as error:
        music_import.normalize_music(b"ID3fixture", SCENE_ID)
    assert isinstance(error.value.__cause__, subprocess.TimeoutExpired)
    assert error.value.__cause__.timeout == 180
    assert not runtime.fake.calls[0]["temporary"].exists()


@pytest.mark.parametrize("failure", ["exit", "missing_report", "quality"])
def test_failed_cpu_runtime_never_returns_assets(runtime, failure):
    if failure == "exit":
        runtime.fake.returncode = 2
    elif failure == "missing_report":
        runtime.fake.write_report = False
    else:
        runtime.fake.report["ok"] = False
    with pytest.raises(ValueError):
        music_import.normalize_music(b"ID3fixture", SCENE_ID)
    assert not runtime.fake.calls[0]["temporary"].exists()


@pytest.mark.parametrize("configuration", ["relative", "absolute", "default"])
def test_dedicated_python_resolves_against_repository_root(runtime, configuration):
    python = runtime.python
    if configuration == "absolute":
        config = {"music": {"python": str(python)}}
    elif configuration == "default":
        python = runtime.root / "services/worker/runtimes/stable_audio3/.venv/Scripts/python.exe"
        python.parent.mkdir(parents=True)
        python.write_bytes(b"fake default runtime")
        config = {}
    else:
        config = {"music": {"python": "runtime/python.exe"}}
    runtime.config.write_text(json.dumps(config), encoding="utf-8")
    music_import.normalize_music(b"ID3fixture", SCENE_ID)
    assert runtime.fake.calls[0]["command"][0] == str(python)


def test_unprepared_runtime_fails_before_subprocess(runtime):
    runtime.python.unlink()
    with pytest.raises(ValueError, match="専用環境"):
        music_import.normalize_music(b"ID3fixture", SCENE_ID)
    assert not runtime.fake.calls
