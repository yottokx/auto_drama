"""Routing must remain visible at cache, usage and result boundaries."""

import copy

import pytest

from services.worker.generation import m3_pipeline, text_debug
from services.worker.generation.causal_runtime import CausalRun, digest
from services.worker.generation.llm import write_json
from services.worker.generation.model_routing import model_configuration_identity
from services.worker.generation.workflow_version import generator_protocol
from tests.unit.test_causal_runtime import FakeLLM, StageValue, forbidden, generated
from tests.unit.test_text_debug import snapshot


def configuration(tmp_path, monkeypatch):
    for name in ("gemma", "qwen"):
        write_json(tmp_path / f"{name}.json", {"model": {"revision": "original", "id": name},
                                              "template_sha256": "a" * 64})
    config = {"llm_config": "gemma.json", "llm": {"model_id": "gemma"},
              "model_routing": {"review": {"llm_config": "qwen.json", "llm": {"model_id": "qwen"}}}}
    monkeypatch.setattr(m3_pipeline.pipeline, "ROOT", tmp_path)
    monkeypatch.setattr(m3_pipeline.pipeline, "load_config", lambda: copy.deepcopy(config))
    return config


@pytest.mark.parametrize("field", ["revision", "template_sha256"])
def test_qwen_manifest_change_rejects_cached_result_before_opening_model(tmp_path, monkeypatch, field):
    config = configuration(tmp_path, monkeypatch)
    payload = {"schema_version": 1, "seed": 1, "story_workflow_version": 2,
               "generator_protocol": generator_protocol("causal")}
    job = {"kind": "m3_narrative", "payload": payload}
    work = tmp_path / "job"
    work.mkdir()
    write_json(work / "job-request.json", {"fingerprint": digest(job),
        "generation_config": {**config, "model_configuration_identity":
                              model_configuration_identity(tmp_path, config)}})
    (work / "result.zip").write_bytes(b"completed bundle")
    monkeypatch.setattr(m3_pipeline, "RoutedLLM", lambda *_args, **_kwargs: pytest.fail("No model load"))
    assert m3_pipeline.generate_job(job, work) == b"completed bundle"
    manifest = text_debug._read(tmp_path / "qwen.json")
    if field == "revision":
        manifest["model"][field] = "changed"
    else:
        manifest[field] = "b" * 64
    write_json(tmp_path / "qwen.json", manifest)
    with pytest.raises(ValueError, match="model configuration changed"):
        m3_pipeline.generate_job(job, work)
    assert (work / "result.zip").read_bytes() == b"completed bundle"


def test_debug_resume_rejects_changed_review_manifest_before_work(tmp_path, monkeypatch):
    configuration(tmp_path, monkeypatch)
    calls = []

    def stopped(*args):
        calls.append(args)
        raise RuntimeError("small intentional stop")

    monkeypatch.setattr(m3_pipeline, "generate_job", stopped)
    output = tmp_path / "experiment"
    with pytest.raises(RuntimeError, match="intentional stop"):
        text_debug.run_text_debug(snapshot(), output, chapter_limit=1)
    write_json(tmp_path / "qwen.json", {"model": {"revision": "changed", "id": "qwen"}})
    with pytest.raises(ValueError, match="config changed"):
        text_debug.run_text_debug(snapshot(), output, chapter_limit=1, resume=True)
    assert len(calls) == 1


class StageRoutedLLM(FakeLLM):
    def model_configuration_identity(self):
        return {"generation": "gemma", "review": "qwen"}

    def select_purpose(self, purpose):
        self.profile = {"model_id": "qwen" if purpose.endswith("review") else "gemma"}


def test_node_cache_selects_intended_model_before_key_and_restores_global_ordinals(tmp_path):
    first = StageRoutedLLM(tmp_path)
    with CausalRun(first, {}) as run:
        run.node("write-s1", {}, StageValue, lambda: generated(first, 1))
        run.node("chapter-review", {}, StageValue, lambda: generated(first, 2))
    resumed = StageRoutedLLM(tmp_path)
    resumed.profile = {"model_id": "qwen"}
    with CausalRun(resumed, {}) as run:
        assert run.node("write-s1", {}, StageValue, forbidden).value == 1
        assert resumed.profile["model_id"] == "gemma"
        assert run.node("chapter-review", {}, StageValue, forbidden).value == 2
        assert resumed.profile["model_id"] == "qwen"
        assert resumed.requests == 2
        run.node("write-s2", {}, StageValue, lambda: generated(resumed, 3))
        assert resumed.profile["model_id"] == "gemma" and resumed.requests == 3


def test_failure_metrics_count_every_model_load_and_release_once(tmp_path):
    work = tmp_path / "job"
    (work / "llm-v7").mkdir(parents=True)
    trace = []
    previous = None
    for number, route in enumerate(("generation", "review", "generation"), 1):
        trace.extend([{"type": "model_load", "elapsed_seconds": 2},
                      {"type": "model_switch", "status": "ready", "from_route": previous,
                       "to_route": route}, {"type": "model_release", "elapsed_seconds": 0.5}])
        write_json(work / "llm-v7" / f"{number:02d}-request.json", {
            "runtime": {"profile": {"model_id": route}},
            "response": {"usage": {"prompt_tokens": 10, "completion_tokens": 5}}})
        previous = route
    write_json(work / "generation-trace.json", trace[:-1])
    write_json(work / "llm-v7/llm-metrics.json", trace)
    metrics = text_debug._metrics(work, set(), None, 20, {}, cached_result=False)
    assert metrics["model_loads_this_run"] == 3
    assert metrics["model_switches_this_run"] == 2
    assert metrics["model_load_seconds_this_run"] == 6
    assert metrics["model_release_seconds_this_run"] == 1.5
    assert metrics["models_new"]["generation"]["requests"] == 2
    assert metrics["models_new"]["review"]["completion_tokens"] == 5
    replay = text_debug._metrics(work, set(text_debug._request_records(work)),
                                 {"trace": trace}, 0.01, {}, cached_result=True)
    assert replay["model_loads_this_run"] == replay["model_switches_this_run"] == 0
    assert replay["models_new"] == {}
