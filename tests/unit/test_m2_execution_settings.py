"""M2 retries select new LLM settings without changing source inputs."""
from __future__ import annotations

import copy
import json
from contextlib import nullcontext

import pytest

from services.worker.generation import pipeline
from services.worker.generation.llm import LocalLLM
from tests.unit.test_m2_generation import job, unpack, world


def live_job(context=16384, revision=0, *, model="gemma-4-31B-it-UD-Q4_K_XL"):
    request = job()
    request["retry_generation"] = revision
    request["payload"].update(execution_settings_version=1, execution_settings_revision=revision)
    request["payload"]["profile"].update(
        common_settings_version=1, context_size=context, temperature=0.4,
        top_p=0.8, reasoning_level="none", model_id=model)
    return request


@pytest.fixture
def memory_runtime(monkeypatch):
    instances = []
    config = pipeline.load_config()

    def selected_config(current, payload, _root):
        selected = copy.deepcopy(current)
        profile = payload["profile"]
        selected["llm"].update(
            model_id=profile["model_id"], context_size=profile["context_size"],
            max_context_size=profile["context_size"], model_context_size=profile["context_size"])
        selected["model_routing"] = {}
        return selected

    class MemoryLLM:
        def __init__(self, _root, selected, payload, _output):
            self.config = copy.deepcopy(selected)
            self.profile = {**selected["llm"], **payload["profile"]}
            self.base = {"model": {"revision": "fixture"}}
            self.requests = 0
            self.trace = []
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    monkeypatch.setattr(pipeline, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(pipeline, "select_config", selected_config)
    monkeypatch.setattr(pipeline, "LocalLLM", MemoryLLM)
    monkeypatch.setattr(pipeline, "gpu_lock", lambda *_args: nullcontext())
    monkeypatch.setattr(pipeline, "generate_text", lambda *_args: world())
    return instances, config


def failed_run(monkeypatch, request, work):
    with monkeypatch.context() as failure:
        def interrupted(*_args):
            raise ValueError("fixture context shortage")

        failure.setattr(pipeline, "generate_text", interrupted)
        with pytest.raises(ValueError, match="fixture context shortage"):
            pipeline.generate_job(request, work)
    assert not (work / "result.zip").exists()


def test_m2_failed_job_uses_new_context_and_model_on_explicit_retry(
        tmp_path, monkeypatch, memory_runtime):
    instances, _ = memory_runtime
    failed_run(monkeypatch, live_job(), tmp_path)
    envelope, _ = unpack(pipeline.generate_job(live_job(32768, 1, model="new-model"), tmp_path))
    assert [llm.profile["context_size"] for llm in instances] == [16384, 32768]
    assert envelope["provenance"]["profile"]["model_id"] == "new-model"
    assert envelope["provenance"]["llm"]["model_id"] == "new-model"
    history = envelope["provenance"]["execution_settings_history"]
    assert len(history) == 1
    assert history[0]["payload"]["profile"]["context_size"] == 16384
    saved = json.loads((tmp_path / "job-request.json").read_text(encoding="utf-8"))
    assert saved["payload"]["execution_settings_revision"] == 1
    assert saved["generation_config"]["llm"]["context_size"] == 32768


def test_m2_automatic_retry_keeps_worker_llm_configuration_snapshot(
        tmp_path, monkeypatch, memory_runtime):
    instances, config = memory_runtime
    request = live_job()
    failed_run(monkeypatch, request, tmp_path)
    original_timeout = config["llm"]["request_timeout_seconds"]
    config["llm"]["request_timeout_seconds"] += 1
    pipeline.generate_job(request, tmp_path)
    assert len(instances) == 2
    assert instances[1].config["llm"]["request_timeout_seconds"] == original_timeout
    assert instances[1].profile["context_size"] == 16384


def test_existing_failed_m2_request_can_adopt_live_settings_on_explicit_retry(
        tmp_path, monkeypatch, memory_runtime):
    instances, _ = memory_runtime
    legacy = live_job()
    legacy["payload"].pop("execution_settings_version")
    legacy["payload"].pop("execution_settings_revision")
    failed_run(monkeypatch, legacy, tmp_path)
    previous = json.loads((tmp_path / "job-request.json").read_text(encoding="utf-8"))
    pipeline.generate_job(live_job(32768, 1), tmp_path)
    current = json.loads((tmp_path / "job-request.json").read_text(encoding="utf-8"))
    assert current["execution_history"] == [previous]
    assert instances[1].profile["context_size"] == 32768


def test_m2_setting_refresh_requires_explicit_retry_and_unchanged_source(
        tmp_path, monkeypatch, memory_runtime):
    instances, _ = memory_runtime
    failed_run(monkeypatch, live_job(), tmp_path)
    with pytest.raises(ValueError, match="explicit retry"):
        pipeline.generate_job(live_job(32768), tmp_path)
    changed_source = live_job(32768, 1)
    changed_source["payload"]["world_input"]["setting"] += "別の指定。"
    with pytest.raises(ValueError, match="another input snapshot"):
        pipeline.generate_job(changed_source, tmp_path)
    assert len(instances) == 1


def test_completed_m2_bundle_keeps_its_actual_provenance_on_settings_retry(
        tmp_path, memory_runtime):
    instances, _ = memory_runtime
    original = pipeline.generate_job(live_job(), tmp_path)
    assert pipeline.generate_job(live_job(32768, 1, model="new-model"), tmp_path) == original
    assert len(instances) == 1
    envelope, _ = unpack(original)
    assert envelope["provenance"]["profile"]["context_size"] == 16384


def test_changed_m2_runtime_does_not_replay_an_old_llm_response(tmp_path):
    config = pipeline.load_config()
    config.pop("model_routing", None)
    config["llm"].update(context_size=16384, max_context_size=32768, model_context_size=32768)
    first = LocalLLM(pipeline.ROOT, config, live_job()["payload"], tmp_path)
    first.request = lambda *_args: {
        "choices": [{"finish_reason": "stop", "message": {"content": "old-context"}}]}
    assert first.chat("final", [{"role": "user", "content": "same source"}])["content"] == "old-context"
    updated = copy.deepcopy(config)
    updated["llm"]["context_size"] = 32768
    second = LocalLLM(pipeline.ROOT, updated, live_job(32768, 1)["payload"], tmp_path)
    second.request = lambda *_args: {
        "choices": [{"finish_reason": "stop", "message": {"content": "new-context"}}]}
    assert second.chat("final", [{"role": "user", "content": "same source"}])["content"] == "new-context"
    assert len(list(tmp_path.glob("01-final-*.json"))) == 2
