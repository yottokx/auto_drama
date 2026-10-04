"""The planner cannot feed DiT until its child has exited with valid conditions."""

import json
import subprocess
import time

import pytest

from scripts.audio import ace_planner_bridge as bridge
from scripts.audio.ace_backend import instrumental_caption
from scripts.audio.engine import AudioGenerationError, GenerationCancelled, validate_request


def request(**kwargs):
    return validate_request({"model": "ace15_turbo", "model_path": "ace", "prompt": "Cold industrial synths.",
                             "duration": 10.01, "seed": 42, "bpm": 65, "keyscale": "Eb minor",
                             "timesignature": "4", "ace_planner": True, "planner_model_path": "lm", **kwargs})


def result(settings):
    return {"ok": True, "audio_codes": "<|audio_code_12|>" * 51, "code_count": 51,
            "planner_released": True, "actual_device": "cuda",
            "metadata": {"duration": settings.duration, "bpm": settings.bpm,
                         "keyscale": settings.keyscale, "timesignature": settings.timesignature},
            "planner_metadata": {"caption": instrumental_caption(settings.prompt, continuous=settings.ace_continuous), "lyrics": "[Instrumental]",
                                 "vocal_language": "unknown", "seed": settings.seed}}


@pytest.mark.parametrize("change", [
    {"audio_codes": "<|audio_code_12|>" * 50}, {"audio_codes": "a song"},
    {"audio_codes": "<|audio_code_12|>" * 51 + "[Verse]"},
    {"code_count": 51.0}, {"planner_released": False}, {"metadata": {}},
    {"audio_codes": "<|audio_code_64000|>" * 51}, {"actual_device": "bad"},
    {"planner_metadata": {"lyrics": "singing"}}, {"ok": False, "error": "OOM"},
])
def test_reject_invalid_codes_short_duration_rewrites_and_unreleased_model(change):
    settings = request()
    with pytest.raises(AudioGenerationError):
        bridge.validate_result({**result(settings), **change}, settings)


def test_fractional_duration_rounds_up_to_five_hz():
    settings = request()
    assert bridge.validate_result(result(settings), settings)["code_count"] == 51


def test_cpu_request_rejects_planner_result_generated_on_cuda():
    settings = request(device="cpu")
    with pytest.raises(AudioGenerationError, match="デバイス"):
        bridge.validate_result(result(settings), settings)


def test_continuity_constraint_must_reach_planner_caption():
    settings = request(ace_continuous=True)
    payload = result(settings)
    assert bridge.validate_result(payload, settings)["ok"]
    payload["planner_metadata"]["caption"] = instrumental_caption(settings.prompt)
    with pytest.raises(AudioGenerationError, match="caption"):
        bridge.validate_result(payload, settings)


def child_script(tmp_path, payload, *, wait_for_cancel=False):
    script = tmp_path / "child.py"
    script.write_text(
        "import pathlib,sys,time\n"
        "out=pathlib.Path(sys.argv[sys.argv.index('--output-dir')+1])\n"
        + ("while not (out/'stop.request').exists(): time.sleep(.01)\n" if wait_for_cancel else "")
        + f"(out/'result.json').write_text({json.dumps(json.dumps(payload))},encoding='utf-8')\n"
        + "time.sleep(.15)\n(out/'child-exited').touch()\n",
        encoding="utf-8",
    )
    return script


def use_child(monkeypatch, script):
    original = subprocess.Popen
    children = []

    def start(command, **kwargs):
        command = list(command)
        command[4] = str(script)
        child = original(command, **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(bridge.subprocess, "Popen", start)
    return children


def test_waits_for_child_exit_before_returning_result(tmp_path, monkeypatch):
    settings = request()
    children = use_child(monkeypatch, child_script(tmp_path, result(settings)))
    phases = []
    outcome = bridge.run_planner(settings, tmp_path, report=lambda phase, *_: phases.append(phase))
    assert outcome["planner_released"] is True
    assert (tmp_path / "planner" / "child-exited").exists()
    assert children[0].poll() == 0
    assert phases[-1] == "releasing"
    sent = json.loads((tmp_path / "planner" / "request.json").read_text(encoding="utf-8"))
    assert sent["prompt"] == instrumental_caption(settings.prompt)
    assert sent["seed"] == 42 and sent["bpm"] == 65


def test_cancel_marker_reaches_child_and_process_is_reaped(tmp_path, monkeypatch):
    settings = request()
    children = use_child(monkeypatch, child_script(tmp_path, result(settings), wait_for_cancel=True))
    started = time.monotonic()
    with pytest.raises(GenerationCancelled):
        bridge.run_planner(settings, tmp_path, report=lambda *_: None,
                           cancelled=lambda: time.monotonic() - started > .1)
    assert (tmp_path / "planner" / "stop.request").is_file()
    assert children[0].poll() is not None
