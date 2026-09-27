"""Two model routes share one request history and never overlap GPU ownership."""

import copy
import json
import time

import pytest

from services.worker.generation import model_routing as routing
from services.worker.generation.llm import LocalLLM
from services.worker.generation.narrative import _purpose
from services.worker.generation.pipeline import ROOT, load_config

PAYLOAD = {"seed": 31, "story_workflow_version": 2}
MESSAGES = [{"role": "user", "content": "原文を変更せず検査する。"}]


@pytest.fixture
def backend(monkeypatch):
    events, live, requests = [], [], []

    class Backend(LocalLLM):
        def __enter__(self):
            assert not live, "Two model runtimes must never overlap."
            self.output.mkdir(parents=True, exist_ok=True)
            events.append(("load", self.profile["model_id"]))
            live.append(self)
            self.base_url = "http://fake"
            self._launch_context_size = self._startup_context_size()
            self.server_context_size = self._launch_context_size
            self.deadline = time.monotonic() + 1000
            self.trace.append({"type": "model_load", "elapsed_seconds": 0.01})
            return self

        def __exit__(self, *args):
            events.append(("release", self.profile["model_id"]))
            assert live.pop() is self
            self.trace.append({"type": "model_release", "elapsed_seconds": 0.02})

        def request(self, path, value=None, timeout=None):
            if path == "/apply-template":
                return {"prompt": "exact-template"}
            if path == "/tokenize":
                return {"tokens": [1] * 500}
            assert path == "/v1/chat/completions"
            requests.append((self.profile["model_id"], copy.deepcopy(value)))
            return {"choices": [{"finish_reason": "stop", "message": {"content": "accepted"}}],
                    "usage": {"prompt_tokens": 500, "completion_tokens": 20},
                    "timings": {"predicted_n": 20, "predicted_ms": 350}}

    monkeypatch.setattr(routing, "LocalLLM", Backend)
    return events, live, requests


@pytest.mark.parametrize("purpose", ["blueprint-review", "chapter-intent-review",
    "scene-sequence-review", "scene-continuity-s1", "fact-comparison-s1-1",
    "state-comparison-s1", "extraction-review-s1", "quality_review", "chapter-progress-review"])
def test_semantic_purposes_use_qwen_low_and_reserve_its_output(purpose):
    category, profile = routing.resolve_purpose_profile(load_config(), PAYLOAD, purpose)
    assert category in {"quality_review", "continuity_review"}
    assert profile["model_id"] == "Qwen3.8-27B-UD-Q4_K_XL"
    assert (profile["context_size"], profile["max_tokens"], profile["reasoning_level"]) == (32768, 8192, "low")


@pytest.mark.parametrize("purpose", ["scene_text", "scene-text-s1", "staging", "story-blueprint",
    "scene-sequence", "extract-s1", "state-observation-s1", "information-s1"])
def test_writing_planning_and_source_observation_keep_gemma(purpose):
    _, profile = routing.resolve_purpose_profile(load_config(), PAYLOAD, purpose)
    assert profile["model_id"].startswith("gemma-")
    assert profile["reasoning_level"] == "none" and profile["context_size"] == 16384


def test_legacy_and_generation_overrides_do_not_leak_into_review(tmp_path):
    config = load_config()
    _, legacy = routing.resolve_purpose_profile(config, {"seed": 1}, "quality_review")
    assert legacy["model_id"] == config["llm"]["model_id"]
    payload = {**PAYLOAD, "profile": {"model_id": config["llm"]["model_id"], "max_tokens": 2048},
               "profiles": {"continuity_review": {"reasoning_level": "medium"}}}
    llm = routing.RoutedLLM(ROOT, config, payload, tmp_path)
    _purpose(llm, "scene-continuity-s1")
    assert llm.profile["reasoning_level"] == "medium" and llm.profile["max_tokens"] == 8192
    _purpose(llm, "scene_text")
    assert llm.profile["max_tokens"] == 2048 and llm.profile["reasoning_level"] == "none"


def test_selecting_profiles_and_inspecting_cache_never_loads_a_model(backend, tmp_path):
    events, _, _ = backend
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as llm:
        for purpose in ("scene_text", "scene-continuity-s1", "state-observation-s1"):
            llm.select_purpose(purpose)
            assert not llm.has_cached_chat(purpose, MESSAGES)
        assert events == [] and llm.requests == 0
    assert events == []


def test_switches_release_before_loading_and_keep_global_seed_trace_and_budget(backend, tmp_path):
    events, live, requests = backend
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as llm:
        for purpose in ("scene_text", "staging", "scene-continuity-s1", "quality_review", "extract-s1"):
            _purpose(llm, purpose)
            assert llm.chat(purpose, MESSAGES)["content"] == "accepted"
        assert llm.requests == 5
        budgets = [row for row in llm.trace if row["type"] == "context_budget"]
        assert [row["context_size"] for row in budgets] == [16384, 16384, 32768, 32768, 16384]
        assert [row["output_tokens"] for row in budgets] == [3072, 3072, 8192, 8192, 4096]
        assert llm.provenance()["requests"] == 5
    assert not live
    assert [kind for kind, _ in events] == ["load", "release", "load", "release", "load", "release"]
    assert [r["seed"] for _, r in requests] == [32, 33, 34, 35, 36]
    assert [r["reasoning_effort"] for _, r in requests] == ["none", "none", "low", "low", "none"]
    assert requests[2][1]["chat_template_kwargs"] == {"enable_thinking": True}
    assert len(json.loads((tmp_path / "llm-metrics.json").read_text(encoding="utf-8"))) == len(llm.trace)


def test_exact_replay_across_routes_needs_no_server_and_preserves_ordinals(backend, tmp_path):
    events, _, _ = backend
    purposes = ("scene_text", "scene-continuity-s1", "extract-s1")
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as first:
        for purpose in purposes:
            first.select_purpose(purpose)
            first.chat(purpose, MESSAGES)
    before = list(events)
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as resumed:
        for purpose in purposes:
            resumed.select_purpose(purpose)
            assert resumed.has_cached_chat(purpose, MESSAGES)
            assert resumed.chat(purpose, MESSAGES)["content"] == "accepted"
        assert resumed.requests == 3
        assert all(row["cache_hit"] for row in resumed.trace if row["type"] == "llm_generation")
    assert events == before


def test_registry_changes_in_unused_review_invalidate_generation_cache(backend, tmp_path):
    config = load_config()
    with routing.RoutedLLM(ROOT, config, PAYLOAD, tmp_path) as llm:
        llm.select_purpose("scene_text")
        llm.chat("scene_text", MESSAGES)
    changed = copy.deepcopy(config)
    changed["model_routing"]["review"]["llm"]["reasoning_level"] = "medium"
    other = routing.RoutedLLM(ROOT, changed, PAYLOAD, tmp_path)
    other.select_purpose("scene_text")
    assert not other.has_cached_chat("scene_text", MESSAGES)
    assert other.model_configuration_identity() != llm.model_configuration_identity()


def test_retries_keep_their_global_salt_across_route_changes(backend, tmp_path):
    _, _, requests = backend
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as llm:
        llm.select_purpose("quality_review")
        llm.chat("quality_review", MESSAGES)
        llm.retry_failed_from(1)
        llm.select_purpose("scene_text")
        llm.chat("scene_text", MESSAGES)
    assert requests[1][1]["seed"] == 31 + 2 + 104729
    restored = routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path)
    assert restored.retry_seed_segments == llm.retry_seed_segments


def test_xhigh_and_extra_thinking_override_are_rejected_before_loading(backend, tmp_path):
    events, _, _ = backend
    with pytest.raises(ValueError, match="reasoning_level"):
        routing.RoutedLLM(ROOT, load_config(), {**PAYLOAD, "profiles": {
            "continuity_review": {"reasoning_level": "xhigh"}}}, tmp_path)
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as llm:
        llm.select_purpose("quality_review")
        with pytest.raises(ValueError, match="bypass"):
            llm.chat("quality_review", MESSAGES, reasoning_effort="xhigh")
    assert events == []


def test_failure_still_releases_current_runtime(backend, tmp_path):
    _, live, _ = backend
    with (pytest.raises(KeyboardInterrupt),
          routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as llm):
        llm.select_purpose("quality_review")
        llm.chat("quality_review", MESSAGES)
        raise KeyboardInterrupt
    assert not live and (tmp_path / "llm-metrics.json").is_file()


def test_registry_identity_is_independent_of_selected_purpose(tmp_path):
    llm = routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path)
    identity = llm.model_configuration_identity()
    llm.select_purpose("quality_review")
    assert llm.model_configuration_identity() == identity
    assert routing.model_configuration_identity(ROOT, {"llm": {"model_id": "fake"}})["generation"]["base"] is None


def test_qwen_template_and_build_are_checked_before_generation(tmp_path):
    config = routing.route_configs(load_config())["review"]
    llm = LocalLLM(ROOT, config, {"seed": 1}, tmp_path)
    llm.request = lambda *_args, **_kwargs: {"n_ctx": 32768, "chat_template": "different"}
    with pytest.raises(ValueError, match="chat template"):
        llm._read_server_context(32768)


def test_actual_smaller_context_is_restored_for_offline_cache_identity(backend, tmp_path, monkeypatch):
    events, _, _ = backend
    start = routing.LocalLLM.__enter__

    def smaller(client):
        start(client)
        client.server_context_size = 24576
        return client

    monkeypatch.setattr(routing.LocalLLM, "__enter__", smaller)
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as first:
        first.select_purpose("quality_review")
        first.chat("quality_review", MESSAGES)
        budget = next(row for row in first.trace if row["type"] == "context_budget")
        assert budget["context_size"] == 24576
        identity = first.runtime_identity()
    before = list(events)
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as resumed:
        resumed.select_purpose("quality_review")
        assert resumed.runtime_identity() == identity
        assert resumed.has_cached_chat("quality_review", MESSAGES)
        resumed.chat("quality_review", MESSAGES)
    assert events == before


def test_medium_profile_changes_request_without_reloading_qwen(backend, tmp_path):
    events, _, requests = backend
    with routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as llm:
        llm.select_purpose("quality_review")
        llm.chat("quality_review", MESSAGES)
        llm.set_profile({**llm.profile, "reasoning_level": "medium"})
        llm.chat("quality_review", MESSAGES)
        assert [r["reasoning_effort"] for _, r in requests] == ["low", "medium"]
        assert len(events) == 1
    assert [kind for kind, _ in events] == ["load", "release"]


def test_failed_new_model_load_cannot_leave_previous_model_owned(backend, tmp_path, monkeypatch):
    events, live, _ = backend
    start = routing.LocalLLM.__enter__

    def failed_review(client):
        if client.profile["reasoning_level"] == "low":
            assert not live
            raise RuntimeError("review startup failed")
        return start(client)

    monkeypatch.setattr(routing.LocalLLM, "__enter__", failed_review)
    with (pytest.raises(RuntimeError, match="startup failed"),
          routing.RoutedLLM(ROOT, load_config(), PAYLOAD, tmp_path) as llm):
        llm.select_purpose("scene_text")
        llm.chat("scene_text", MESSAGES)
        llm.select_purpose("quality_review")
        llm.chat("quality_review", MESSAGES)
    assert not live
    assert [kind for kind, _ in events] == ["load", "release"]
    assert any(row["type"] == "model_switch" and row["status"] == "failed" for row in llm.trace)
