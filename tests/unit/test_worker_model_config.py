import json
from pathlib import Path

import pytest

from packages.contracts.llm_settings import LLMSettings
from services.worker.generation.llm import LocalLLM
from services.worker.generation.model_routing import (
    model_configuration_identity,
    resolve_purpose_profile,
)
from services.worker.generation.pipeline import load_config
from services.worker.model_config import REGISTRY, catalog, select_config


def registry(tmp_path):
    (tmp_path / "test.gguf").write_bytes(b"GGUFtest")
    (tmp_path / "server.exe").touch()
    (tmp_path / "base.json").write_text(json.dumps({
        "model": {"size_bytes": 8}, "server": {"executable": "server.exe"}}))
    entry = {"llm_config": "base.json", "path": str(tmp_path / "test.gguf"),
             "max_context_size": 32768, "reasoning_efforts": ["none", "low"],
             "reasoning_template": "qwen3"}
    path = tmp_path / REGISTRY
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps({"schema_version": 1, "models": {"test": entry}}))


def test_registry_paths_stay_local_and_missing_model_is_not_advertised(tmp_path):
    registry(tmp_path)
    models, errors = catalog(tmp_path)
    assert not errors and models == [{"model": "test",
                                     "reasoning_efforts": ["none", "low"]}]
    (tmp_path / "test.gguf").unlink()
    models, errors = catalog(tmp_path)
    assert models == [] and errors


@pytest.mark.parametrize("purpose,output", [("script-cast", 8192), ("script-outline", 8192),
    ("script-allocation", 2048), ("script-plan", 6144), ("script-handoff", 4096),
    ("script-scene", 8192), ("script-speech", 8192), ("script-staging", 8192)])
def test_common_settings_expand_request_while_preserving_stage_baseline(tmp_path, purpose, output):
    registry(tmp_path)
    common = LLMSettings(model="test", temperature=0.25, top_p=0.7,
                         reasoning_effort="low", ctx_size=32768)
    payload = {"seed": 1, "profile": common.profile(), "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1"}
    config = select_config(load_config(), payload, tmp_path)
    assert config["model_routing"] == {}
    _, profile = resolve_purpose_profile(config, payload, purpose)
    llm = LocalLLM(tmp_path, config, {**payload, "profile": profile}, tmp_path)
    request, _ = llm._chat_request(1, [], {})
    assert request["temperature"] == 0.25 and request["top_p"] == 0.7
    assert request["reasoning_effort"] == "low"
    assert profile["max_tokens"] == output
    assert request["max_tokens"] == output + 8192
    assert llm._startup_context_size() == 32768
    assert llm.base["model"]["relative_path"] == str((tmp_path / "test.gguf").resolve())


def test_model_identity_does_not_depend_on_worker_install_path(tmp_path):
    registry(tmp_path)
    payload = {"profile": {"common_settings_version": 1, "model_id": "test"}}
    config = select_config(load_config(), payload, tmp_path)
    original = model_configuration_identity(tmp_path, config)
    config["llm_base"]["model"]["relative_path"] = "E:/another-worker/model.gguf"
    config["llm_base"]["server"]["executable"] = "E:/another-worker/server.exe"
    assert model_configuration_identity(tmp_path, config) == original


def test_directory_discovery_add_remove_and_select_without_registration(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (tmp_path / "server.exe").touch()
    config_path = tmp_path / REGISTRY
    config_path.parent.mkdir(parents=True)
    config_path.write_text(json.dumps({"schema_version": 1, "model_dirs": ["models"],
                                      "server_executable": "server.exe"}))
    assert catalog(tmp_path) == ([], [])
    for name in ["first", "second"]:
        (folder / f"{name}.gguf").write_bytes(b"GGUFtest")
    (folder / "not-a-model.txt").write_text("ignored")
    models, errors = catalog(tmp_path)
    assert not errors and {model["model"] for model in models} == {"first", "second"}
    payload = {"seed": 1, "profile": {"common_settings_version": 1, "model_id": "second"}}
    config = select_config(load_config(), payload, tmp_path)
    llm = LocalLLM(tmp_path, config, payload, tmp_path)
    assert Path(llm.base["model"]["relative_path"]) == folder / "second.gguf"
    (folder / "first.gguf").unlink()
    assert [model["model"] for model in catalog(tmp_path)[0]] == ["second"]
    (folder / "invalid.gguf").write_bytes(b"invalid")
    models, errors = catalog(tmp_path)
    assert [model["model"] for model in models] == ["second"] and errors


def test_duplicate_filenames_are_not_silently_selected(tmp_path):
    config_path = tmp_path / REGISTRY
    config_path.parent.mkdir(parents=True)
    for directory in ["a", "b"]:
        (tmp_path / directory).mkdir()
        (tmp_path / directory / "same.gguf").write_bytes(b"GGUFtest")
    config_path.write_text(json.dumps({"schema_version": 1, "model_dirs": ["a", "b"]}))
    models, errors = catalog(tmp_path)
    assert not models and "Duplicate" in errors[0]


@pytest.mark.parametrize("context", [32768, 65536, 131072])
def test_user_context_is_not_capped_by_old_worker_limit(tmp_path, context):
    registry(tmp_path)
    payload = {"seed": 1, "profile": {"common_settings_version": 1,
        "model_id": "test", "context_size": context, "reasoning_level": "none"}}
    config = select_config(load_config(), payload, tmp_path)
    llm = LocalLLM(tmp_path, config, payload, tmp_path)
    assert llm._startup_context_size() == context
    assert llm.context_budget(output_tokens=8192)["context_size"] == context
    assert llm.profile["max_tokens"] == 3072


def test_legacy_job_uses_current_worker_paths_and_preserves_saved_identity(tmp_path, monkeypatch):
    from contextlib import contextmanager

    from services.worker.generation import llm as llm_module
    registry(tmp_path)
    config = load_config()
    config["llm"].update(model_id="test", context_size=65536)
    config["llm_base"] = {"model": {"relative_path": "old/model.gguf", "size_bytes": 8},
                          "server": {"executable": "old/server.exe"}}
    captured = []
    released = []

    @contextmanager
    def process(command, *_args, **_kwargs):
        captured.append(command)
        yield type("Process", (), {"poll": lambda self: None})()
        released.append(True)

    monkeypatch.setattr(llm_module, "owned_process", process)
    monkeypatch.setattr(LocalLLM, "request", lambda self, path, **kwargs:
                        {"status": "ok"} if path == "/health" else {"n_ctx": 65536})
    client = LocalLLM(tmp_path, config, {"seed": 1}, tmp_path / "job")
    identity = client.runtime_identity()
    with client:
        assert client.server_context_size == 65536
    command = captured[0]
    assert command[command.index("--model") + 1] == str(tmp_path / "test.gguf")
    assert command[command.index("--ctx-size") + 1] == "65536"
    assert command[0] == str(tmp_path / "server.exe")
    assert client.runtime_identity() == identity
    assert released == [True]


def test_explicit_server_path_also_applies_to_known_models(tmp_path):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "known.gguf").write_bytes(b"GGUFtest")
    (tmp_path / "new-server.exe").touch()
    (tmp_path / "config").mkdir()
    (tmp_path / "config/known-llm.json").write_text(json.dumps({
        "model": {"filename": "known.gguf", "size_bytes": 8},
        "server": {"executable": "missing-server.exe"}}))
    registry_path = tmp_path / REGISTRY
    registry_path.parent.mkdir(parents=True)
    registry_path.write_text(json.dumps({"schema_version": 1, "model_dirs": ["models"],
                                        "server_executable": "new-server.exe"}))
    models, errors = catalog(tmp_path)
    assert not errors and models[0]["model"] == "known"


@pytest.mark.parametrize("effort", ["high", "xhigh", "custom-value", " 任意の値 ", ""])
@pytest.mark.parametrize("template", [None, "qwen3"])
def test_free_text_effort_reaches_llm_request_and_script_purpose(tmp_path, effort, template):
    from services.worker.generation.narrative import _purpose

    registry(tmp_path)
    path = tmp_path / REGISTRY
    entries = json.loads(path.read_text())
    entries["models"]["test"].update(reasoning_efforts=["none"], reasoning_template=template)
    path.write_text(json.dumps(entries))
    common = LLMSettings(model="test", temperature=0.25, top_p=0.7,
                         reasoning_effort=effort, ctx_size=32768)
    payload = {"seed": 1, "profile": common.profile(), "story_workflow_version": 2,
               "workflow_policy": "script_continuation_v1"}
    config = select_config(load_config(), payload, tmp_path)
    llm = LocalLLM(tmp_path, config, payload, tmp_path)
    request, _ = llm._chat_request(1, [], {})
    assert request["reasoning_effort"] == effort
    for purpose in ("script-cast", "script-plan", "script-scene"):
        _purpose(llm, purpose)
        request, _ = llm._chat_request(1, [], {})
        assert request["reasoning_effort"] == effort
