"""Music and optional CG planners refresh only execution LLM settings on retry."""
from __future__ import annotations

import copy
import json
from contextlib import nullcontext

import pytest

from services.worker import model_config
from services.worker.generation import event_cg_pipeline as cg
from services.worker.generation import music_pipeline as music
from services.worker.generation.llm import ContextBudgetError
from tests.unit.test_event_cg_worker import payload as cg_payload
from tests.unit.test_event_cg_worker import unpack
from tests.unit.test_music_worker import context as music_context

KINDS = ["m3_music_plan", "m3_event_cg_budget", "m3_event_cg_plan"]


def request(kind, *, context=16384, revision=0, live=True):
    payload = (cg_payload() if kind.startswith("m3_event_cg") else
               {"schema_version": 1, "seed": 1, "context": music_context()})
    payload["profile"] = {"provider": "local", "common_settings_version": 1,
        "model_id": "new-model" if revision else "old-model", "context_size": context,
        "temperature": 0.4, "top_p": 0.8, "reasoning_level": "none"}
    if live:
        payload.update(execution_settings_version=1, execution_settings_revision=revision)
    return {"kind": kind, "retry_generation": revision, "payload": payload}


@pytest.fixture
def planner_runtime(monkeypatch):
    instances = []
    control = {"fail": False}
    config = {"gpu_lock_timeout_seconds": 1, "max_zip_bytes": 1_000_000,
              "llm": {"model_id": "old-model", "context_size": 16384,
                      "request_timeout_seconds": 120}, "model_routing": {}}

    def select_config(current, payload, _root):
        selected = copy.deepcopy(current)
        selected["llm"].update(payload["profile"])
        return selected

    def identity(_root, selected):
        return {"generation": {"settings": copy.deepcopy(selected["llm"]),
            "base": {"model": {"id": selected["llm"]["model_id"]}}}}

    class MemoryLLM:
        def __init__(self, _root, selected, payload, _output):
            self.config = copy.deepcopy(selected)
            self.profile = {**selected["llm"], **payload["profile"]}
            self.requests = 0
            self.trace = []
            instances.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            pass

    def plan(payload, llm):
        if control["fail"]:
            raise ContextBudgetError("fixture planner context shortage")
        return {"source": copy.deepcopy(payload["context"])}

    monkeypatch.setattr(music.pipeline, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(model_config, "select_config", select_config)
    for module in (music, cg):
        monkeypatch.setattr(module, "model_configuration_identity", identity)
        monkeypatch.setattr(module, "gpu_lock", lambda *_args: nullcontext())
        monkeypatch.setattr(module, "LocalLLM", MemoryLLM)
    monkeypatch.setattr(music, "plan_music", plan)
    monkeypatch.setattr(cg, "plan_budget", plan)
    monkeypatch.setattr(cg, "plan_cgs", plan)
    return instances, control


def generate(job, work):
    return (music if job["kind"] == "m3_music_plan" else cg).generate_job(job, work)


@pytest.mark.parametrize("kind", KINDS)
def test_existing_failed_planner_uses_current_settings_and_keeps_original_sources(
        kind, tmp_path, planner_runtime):
    instances, control = planner_runtime
    original = request(kind, live=False)
    control["fail"] = True
    with pytest.raises(ContextBudgetError, match="fixture planner context shortage"):
        generate(original, tmp_path)
    old_request = json.loads((tmp_path / "job-request.json").read_text(encoding="utf-8"))
    control["fail"] = False
    refreshed = request(kind, context=32768, revision=1)
    envelope, _ = unpack(generate(refreshed, tmp_path))
    assert [llm.profile["context_size"] for llm in instances] == [16384, 32768]
    assert envelope["provenance"]["llm"]["model_id"] == "new-model"
    assert envelope["result"]["source"] == original["payload"]["context"]
    current = json.loads((tmp_path / "job-request.json").read_text(encoding="utf-8"))
    assert current["payload"]["context"] == old_request["payload"]["context"]
    assert current["payload"]["seed"] == old_request["payload"]["seed"]
    assert current["execution_history"][0]["fingerprint"] == old_request["fingerprint"]


@pytest.mark.parametrize("kind", KINDS)
def test_planner_settings_retry_cannot_change_source_or_skip_explicit_revision(
        kind, tmp_path, planner_runtime):
    instances, control = planner_runtime
    control["fail"] = True
    with pytest.raises(ContextBudgetError):
        generate(request(kind), tmp_path)
    with pytest.raises(ValueError, match="explicit retry"):
        generate(request(kind, context=32768), tmp_path)
    changed = request(kind, context=32768, revision=1)
    changed["payload"]["context"]["overall_plot"]["chapters"][0]["summary"] = "別の原稿"
    with pytest.raises(ValueError, match="another input snapshot"):
        generate(changed, tmp_path)
    assert len(instances) == 1


@pytest.mark.parametrize("kind", KINDS)
def test_completed_planner_result_survives_settings_refresh_with_original_provenance(
        kind, tmp_path, planner_runtime):
    instances, _ = planner_runtime
    first = generate(request(kind), tmp_path)
    assert generate(request(kind, context=32768, revision=1), tmp_path) == first
    assert len(instances) == 1
    envelope, _ = unpack(first)
    assert envelope["provenance"]["llm"]["model_id"] == "old-model"
