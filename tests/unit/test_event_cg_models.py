"""The editing model is a choice: old works keep the standard model, a worker offers what it holds."""

import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from packages.contracts.event_cg import (
    CG_MODEL_CAPABILITIES,
    CG_MODELS,
    MODEL_REVISION,
    TURBO_REVISION,
    EventCgProfile,
    cg_model,
    configuration_id,
    model_capability,
    new_production_profile,
    worker_models,
)
from scripts.image_edit import backend, common, prepare
from services.worker.generation import event_cg_pipeline as cg

LEGACY = {"backend": "qwen_image21", "model_revision": MODEL_REVISION, "dtype": "bfloat16", "cpu_offload": True,
          "use_kv_cache": True, "steps": 40, "width": 960, "height": 640}


def test_profile_frozen_before_the_choice_existed_keeps_its_bytes_and_its_model():
    profile = EventCgProfile.model_validate(LEGACY)
    assert profile.model == "standard"
    # Hashes of frozen productions are taken over this dump: it must not gain a field.
    assert "model" not in profile.model_dump(mode="json")
    assert model_capability(LEGACY) == model_capability(None) == "qwen_image21"


def test_turbo_is_its_own_revision_with_its_own_fixed_schedule():
    profile = new_production_profile("turbo")
    value = profile.model_dump(mode="json")
    assert (value["model"], value["model_revision"], value["steps"]) == ("turbo", TURBO_REVISION, 8)
    assert model_capability(value) == "qwen_image21_turbo"
    # The memory settings are the same ones, whichever model runs with them.
    assert configuration_id(profile) == configuration_id(new_production_profile()) == "vram32"
    for wrong in ({"model_revision": MODEL_REVISION}, {"steps": 40}):
        with pytest.raises(ValidationError):
            EventCgProfile.model_validate({**value, **wrong})
    with pytest.raises(ValidationError):
        EventCgProfile.model_validate({**LEGACY, "model_revision": TURBO_REVISION})


def test_contract_and_runtime_name_the_same_checkpoints():
    assert {row["id"] for row in CG_MODELS} == set(common.MODELS) == set(cg.MODEL_DIRECTORIES)
    for row in CG_MODELS:
        assert common.MODELS[row["id"]]["model_id"] == "Qwen/" + row["name"]
        assert cg_model(row["id"]) is row
    assert (cg_model("turbo")["revision"], "standard") in {
        (revision, source) for (_, revision), (source, _) in prepare.SHARED_TEXT_ENCODER.items()}


def test_a_worker_holds_the_models_it_names_and_an_older_one_the_standard_model():
    kinds = {"m3_event_cg", "m3_event_cg_plan", "m3_event_cg_budget", "m2_outfit_image"}
    assert worker_models(kinds) == ["standard"]
    assert worker_models(kinds | {"qwen_image21_turbo"}) == ["turbo"]
    assert worker_models(kinds | set(CG_MODEL_CAPABILITIES)) == ["standard", "turbo"]
    assert worker_models({"tyrano_export", "qwen_image21_turbo"}) == []


def install(root, model, revision=None):
    """A verified model folder as the readiness check sees it: a manifest, no weights read."""
    directory = root / cg.RUNTIME / cg.MODEL_DIRECTORIES[model][1]
    directory.mkdir(parents=True)
    weights = directory / "weights.safetensors"
    weights.write_bytes(b"w")
    (directory / "model-manifest.json").write_text(json.dumps({
        "model_id": "Qwen/" + cg_model(model)["name"], "revision": revision or cg_model(model)["revision"],
        "files": [{"path": "weights.safetensors", "size_bytes": 1}]}), encoding="utf-8")


def test_readiness_lists_only_the_models_this_worker_downloaded(tmp_path):
    python = tmp_path / cg.RUNTIME / ".venv/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    assert cg.check_readiness({}, tmp_path) == {"ready": False, "errors": [], "models": []}
    install(tmp_path, "turbo")
    assert cg.check_readiness({}, tmp_path) == {"ready": True, "errors": [], "models": ["turbo"]}
    install(tmp_path, "standard")
    assert cg.check_readiness({}, tmp_path)["models"] == ["standard", "turbo"]
    assert cg.runtime_settings({}, tmp_path, "turbo")["model_path"].endswith("qwen-image-2.1-turbo")
    assert cg.runtime_settings({"event_cg": {"turbo_model_dir": "elsewhere"}}, tmp_path, "turbo")[
        "model_path"].endswith("elsewhere")


def test_a_model_folder_of_another_revision_is_not_offered(tmp_path):
    python = tmp_path / cg.RUNTIME / ".venv/Scripts/python.exe"
    python.parent.mkdir(parents=True)
    python.write_bytes(b"")
    install(tmp_path, "turbo", revision="a" * 40)
    status = cg.check_readiness({"event_cg": {"enabled": True}}, tmp_path)
    assert status["ready"] is False and status["models"] == [] and len(status["errors"]) == 2


def test_the_checkpoints_own_schedule_is_passed_to_the_pipeline(tmp_path):
    calls = []
    torch = SimpleNamespace(inference_mode=lambda: __import__("contextlib").nullcontext(),
                            Generator=lambda _device: SimpleNamespace(manual_seed=lambda seed: seed))

    def pipe(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(images=["image"])

    request = {"prompt": "p", "width": 64, "height": 64, "steps": 8, "reference_resolution": 1024,
               "use_kv_cache": True}
    (tmp_path / "model_index.json").write_text(json.dumps({"sample_sigmas": [1.0, 0.9, 0.5]}), encoding="utf-8")
    assert backend.sample_sigmas(tmp_path) == [1.0, 0.9, 0.5]
    pipe.sample_sigmas = backend.sample_sigmas(tmp_path)
    with pytest.raises(ValueError, match="3-step"):
        backend.render_image(pipe, torch, request, [], 1, None)
    assert backend.render_image(pipe, torch, {**request, "steps": 3}, [], 1, None) == "image"
    assert calls[-1]["sigmas"] == [1.0, 0.9, 0.5] and calls[-1]["num_inference_steps"] == 3
    # The standard model has no schedule of its own and is called exactly as before.
    (tmp_path / "model_index.json").write_text("{}", encoding="utf-8")
    pipe.sample_sigmas = backend.sample_sigmas(tmp_path)
    backend.render_image(pipe, torch, {**request, "steps": 40}, [], 1, None)
    assert "sigmas" not in calls[-1]
    (tmp_path / "model_index.json").write_text(json.dumps({"sample_sigmas": [2.0]}), encoding="utf-8")
    with pytest.raises(ValueError):
        backend.sample_sigmas(tmp_path)


def test_turbo_reuses_the_standard_text_encoder_only_for_the_compared_revisions(tmp_path, monkeypatch):
    standard, stage = tmp_path / "standard", tmp_path / "stage"
    (standard / "text_encoder").mkdir(parents=True)
    stage.mkdir()
    shards = []
    for name in ("model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors", "model.safetensors.index.json"):
        (standard / "text_encoder" / name).write_bytes(name.encode())
        shards.append({"path": "text_encoder/" + name, "size_bytes": len(name), "sha256": "a" * 64})
    (standard / prepare.MANIFEST).write_text(json.dumps({"revision": MODEL_REVISION, "files": [
        *shards, {"path": "text_encoder/config.json", "size_bytes": 2, "sha256": "b" * 64}]}), encoding="utf-8")
    monkeypatch.setitem(prepare.MODELS, "standard", {"model_id": "Qwen/Qwen-Image-2.1", "directory": standard})
    published = [{"path": "text_encoder/config.json", "size_bytes": 3},
                 {"path": "text_encoder/model.safetensors", "size_bytes": 99, "sha256": "c" * 64},
                 {"path": "transformer/config.json", "size_bytes": 4}]
    turbo = "Qwen/Qwen-Image-2.1-Turbo"
    # Another revision was never compared: it is fetched as published.
    assert prepare.shared_text_encoder(turbo, "f" * 40, published, stage) == published
    assert not (stage / "text_encoder").exists()
    entries = prepare.shared_text_encoder(turbo, TURBO_REVISION, published, stage)
    assert [entry["path"] for entry in entries] == [
        "text_encoder/config.json", "text_encoder/model-00001-of-00002.safetensors",
        "text_encoder/model-00002-of-00002.safetensors", "text_encoder/model.safetensors.index.json",
        "transformer/config.json"]
    # Turbo's own settings file is still fetched; only the weights are the standard model's files.
    assert entries[0] == published[0] and entries[1]["sha256"] == "a" * 64
    linked = stage / "text_encoder/model-00001-of-00002.safetensors"
    assert linked.samefile(standard / "text_encoder/model-00001-of-00002.safetensors")
    # Without the standard model at the compared revision, nothing is linked.
    (standard / prepare.MANIFEST).write_text(json.dumps({"revision": "e" * 40, "files": shards}), encoding="utf-8")
    assert prepare.shared_text_encoder(turbo, TURBO_REVISION, published, tmp_path / "other") == published


def test_turbo_names_its_processor_settings_differently():
    names = {name for name in prepare.REQUIRED_FILES if name != "processor/preprocessor_config.json"}
    assert prepare.missing_required(names) == ["processor/preprocessor_config.json"]
    assert prepare.missing_required(names | {"processor/processor_config.json"}) == []
