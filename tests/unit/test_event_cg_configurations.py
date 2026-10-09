"""Selectable CG memory configurations: old productions keep their behavior, new ones reach the runtime."""

import pytest
from pydantic import ValidationError

from packages.contracts.event_cg import (
    CG_CONFIGURATION_FIELDS,
    CG_CONFIGURATIONS,
    EventCgProfile,
    configuration_id,
    image_input_sha256,
)
from scripts.image_edit import engine
from services.worker.generation import event_cg_pipeline as cg
from tests.unit.test_event_cg_worker import image_payload

LEGACY = {"backend": "qwen_image21", "model_revision": "d26bb61231c349cf6b7896fa83353113880e1ba3",
          "dtype": "bfloat16", "cpu_offload": True, "use_kv_cache": True, "steps": 40,
          "width": 960, "height": 640}


def test_profile_frozen_before_the_options_existed_keeps_its_behavior():
    profile = EventCgProfile.model_validate(LEGACY)
    assert profile == EventCgProfile()
    assert (profile.reference_resolution, profile.text_encoder_offload, profile.transformer_storage,
            profile.vae_tiling) == (1024, "model", "native", False)
    assert configuration_id(LEGACY) == "vram32"
    assert configuration_id({**LEGACY, "use_kv_cache": False}) == "vram24_detail"
    assert configuration_id({**LEGACY, "transformer_storage": "fp8"}) is None


def test_only_the_two_supported_sizes_are_accepted():
    assert EventCgProfile.model_validate({**LEGACY, "width": 1536, "height": 1024}).width == 1536
    for width, height in ((960, 1024), (1536, 640), (1920, 1280)):
        with pytest.raises(ValidationError):
            EventCgProfile.model_validate({**LEGACY, "width": width, "height": height})
    with pytest.raises(ValidationError):
        EventCgProfile.model_validate({**LEGACY, "reference_resolution": 512})


def test_every_configuration_is_a_distinct_valid_profile_with_guides():
    identifiers = [row["id"] for row in CG_CONFIGURATIONS]
    assert len(set(identifiers)) == len(identifiers) and identifiers[0] == "vram32"
    assert [row["vram_gb"] for row in CG_CONFIGURATIONS] == sorted(
        (row["vram_gb"] for row in CG_CONFIGURATIONS), reverse=True)
    for row in CG_CONFIGURATIONS:
        fields = {key: row[key] for key in CG_CONFIGURATION_FIELDS}
        profile = EventCgProfile.model_validate({**LEGACY, **fields, "text_encoder_offload": "layers"})
        assert configuration_id(profile) == row["id"]
        # The measured peak must leave room for the driver and desktop on that class of card.
        assert 0 < row["peak_vram_gib"] <= row["vram_gb"] - 2.5
        assert row["label"] and row["quality"] and 0 < row["time_ratio"] < 3
        # Seconds depend on the hardware; only a ratio to the standard configuration is offered.
        assert "inference_seconds" not in row and "秒" not in row["quality"]
        # Decoding in one piece is the peak once the transformer is small.
        assert row["vae_tiling"] is (row["transformer_storage"] == "fp8")
    assert CG_CONFIGURATIONS[0]["time_ratio"] == 1


def test_image_request_carries_the_frozen_configuration_to_the_runtime(tmp_path):
    payload = image_payload(tmp_path)
    payload["cg_profile"] = EventCgProfile.model_validate({
        **LEGACY, "width": 1536, "height": 1024, "reference_resolution": 768, "use_kv_cache": False,
        "text_encoder_offload": "layers", "transformer_storage": "fp8", "vae_tiling": True,
    }).model_dump(mode="json")
    payload["input_sha256"] = image_input_sha256(payload)
    request = cg._image_request(payload, tmp_path, {"model_path": str(tmp_path / "model")})
    checked = engine.validate_request(request)
    assert {key: checked[key] for key in (
        "width", "height", "reference_resolution", "use_kv_cache", "text_encoder_offload",
        "transformer_storage", "vae_tiling", "cpu_offload")} == {
        "width": 1536, "height": 1024, "reference_resolution": 768, "use_kv_cache": False,
        "text_encoder_offload": "layers", "transformer_storage": "fp8", "vae_tiling": True,
        "cpu_offload": True}
    # A production frozen earlier still asks for exactly what it always did.
    legacy = image_payload(tmp_path)
    legacy["cg_profile"] = dict(LEGACY)
    legacy["input_sha256"] = image_input_sha256(legacy)
    old = engine.validate_request(cg._image_request(legacy, tmp_path, {"model_path": str(tmp_path / "model")}))
    assert (old["width"], old["height"], old["reference_resolution"], old["text_encoder_offload"],
            old["transformer_storage"], old["vae_tiling"]) == (960, 640, 1024, "model", "native", False)
