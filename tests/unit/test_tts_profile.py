import copy
import hashlib
import json
from pathlib import Path

import pytest

from packages.contracts import tts_profile
from services.worker.generation import pipeline

SMALL = {"provider_id": "irodori", "model_id": "irodori-v4.1-small", "precision": "fp32"}
LARGE = {"provider_id": "irodori", "model_id": "irodori-v4-large", "precision": "bf16"}


def test_frozen_purpose_profiles_keep_dependencies_and_ignore_later_catalog(monkeypatch):
    settings = {"voice_design": copy.deepcopy(LARGE), "voice_clone": copy.deepcopy(SMALL)}
    profile = tts_profile.build_tts_profile(settings)
    settings["voice_design"]["precision"] = "int4"
    monkeypatch.setattr(tts_profile, "resolve_bundle", lambda *_: pytest.fail("Unexpected re-resolution"))
    design = tts_profile.select_tts_profile(profile, "voice_design")
    clone = tts_profile.select_tts_profile(profile, "voice_clone")
    assert design["precision"] == "bf16"
    assert design["model_id"] == LARGE["model_id"] and clone["model_id"] == SMALL["model_id"]
    assert {file["repo_id"] for file in design["bundle"]["files"]} >= {
        "Aratako/Irodori-TTS-v4-Large", "sony/silentcipher", "Aratako/Semantic-DACVAE-Japanese-32dim"}
    design["bundle"]["files"].clear()
    assert profile["voice_design"]["bundle"]["files"]
    assert tts_profile.select_tts_profile(None, "voice_clone") is None


@pytest.mark.parametrize("tamper", ["dependency", "precision", "steps"])
def test_frozen_profiles_reject_modified_files_and_runtime_settings(tamper):
    profile = tts_profile.build_tts_profile({"voice_design": LARGE, "voice_clone": SMALL})
    entry = profile["voice_design"]
    if tamper == "dependency":
        entry["bundle"]["files"][-1]["revision"] = "a" * 40
    elif tamper == "precision":
        entry["precision"] = "int4"
    else:
        entry["num_steps"] = True
    with pytest.raises(ValueError):
        tts_profile.select_tts_profile(profile, "voice_design")


def test_production_design_and_clone_pass_their_independent_frozen_selection(tmp_path, monkeypatch):
    profile = tts_profile.build_tts_profile({"voice_design": LARGE, "voice_clone": SMALL})
    requests = []

    def runtime(command, *_args, **_kwargs):
        output = Path(command[command.index("--output-dir") + 1])
        requests.append(json.loads((output / "request.json").read_text(encoding="utf-8")))
        (output / "voice.wav").write_bytes(b"validated-waveform")
        (output / "result.json").write_text('{"decoded_and_non_silent":true}')

    monkeypatch.setattr(pipeline, "run_voice_process", runtime)
    config = pipeline.load_config()
    pipeline.generate_voice({"seed": 1, "character_contract_version": 2, "tts_profile": profile,
                             "character_result": {"voice": "若い女声話者。", "selfIntroduction": "こんにちは。"}},
                            tmp_path, config)
    reference = b"exact-reference"
    (tmp_path / "reference-voice.wav").write_bytes(reference)
    pipeline.generate_voice_clone({"seed": 2, "tts_profile": profile, "dialogue_text": "ありがとう。",
                                   "reference_voice": {"artifact_id": "ref", "text": "こんにちは。",
                                                       "sha256": hashlib.sha256(reference).hexdigest()}},
                                  tmp_path, config)
    assert requests[0]["tts_profile"] == profile["voice_design"]
    assert requests[0]["model_precision"] == "bf16"
    assert requests[1]["tts_profile"] == profile["voice_clone"]
    assert requests[1]["model_precision"] == "fp32"
    assert requests[1]["reference_text"] == "こんにちは。"
