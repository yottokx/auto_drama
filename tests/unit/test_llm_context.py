"""Measure exact model prompts and preserve immutable successful request caches."""

import json
import time
from contextlib import contextmanager
from io import BytesIO

import pytest

from services.worker.generation.context_budget import ContextPolicy
from services.worker.generation.llm import ContextBudgetError, LocalLLM
from services.worker.generation.pipeline import ROOT, load_config


def reply(text="accepted"):
    return {"choices": [{"finish_reason": "stop", "message": {"content": text}}]}


def test_context_budget_counts_template_and_schema_reserves_full_output(tmp_path):
    config = load_config()
    config["llm"].update(context_size=4096, context_margin_tokens=128)
    llm = LocalLLM(ROOT, config, {"seed": 1}, tmp_path)
    llm.base_url = "http://model"
    llm.profile["max_tokens"] = 1024
    calls = []

    def request(path, value):
        calls.append((path, value))
        if path == "/apply-template":
            assert value["response_format"]["type"] == "json_schema"
            return {"prompt": "model template + schema + Japanese source"}
        assert path == "/tokenize"
        assert value == {"content": "model template + schema + Japanese source",
                         "add_special": True, "parse_special": True}
        return {"tokens": [42] * 2945}

    llm.request = request
    with pytest.raises(ContextBudgetError, match="story_state.*2945.*1024.*4096"):
        llm.chat("story_state", [{"role": "user", "content": "原文を保持する。"}],
                 response_format={"type": "json_schema"})
    assert [path for path, _ in calls] == ["/apply-template", "/tokenize"]
    assert not list(tmp_path.glob("01-story_state-*.json"))
    budget = json.loads((tmp_path / "last-context-budget.json").read_text("utf-8"))
    assert budget["prompt_tokens"] == 2945
    assert budget["margin_tokens"] == 128


def test_request_at_budget_boundary_uses_original_messages_and_full_answer(tmp_path):
    config = load_config()
    config["llm"].update(context_size=4096, context_margin_tokens=128)
    llm = LocalLLM(ROOT, config, {"seed": 1}, tmp_path)
    llm.base_url = "http://model"
    llm.profile["max_tokens"] = 1024
    messages = [{"role": "user", "content": "秘密と全文を省略しない。"}]
    calls = []

    def request(path, value):
        calls.append((path, value))
        if path == "/apply-template":
            return {"prompt": "formatted"}
        if path == "/tokenize":
            return {"tokens": [42] * 2944}
        return reply()

    llm.request = request
    assert llm.chat("review", messages)["content"] == "accepted"
    assert calls[-1][0] == "/v1/chat/completions"
    assert calls[-1][1]["messages"] == messages
    assert calls[-1][1]["max_tokens"] == 1024


def test_new_prompt_version_replays_only_exact_old_cache_and_never_changes_it(tmp_path):
    old = tmp_path / "llm-v5"
    new = tmp_path / "llm-v6"
    old.mkdir()
    new.mkdir()
    config = load_config()
    payload = {"seed": 9}
    messages = [{"role": "user", "content": "採用本文"}]
    first = LocalLLM(ROOT, config, payload, old)
    first.request = lambda *_: reply("保存済みの原文")
    first.chat("source", messages)
    first.retry_failed_from(1)
    first.chat("source", messages)
    before = {p.name: p.read_bytes() for p in old.iterdir()}
    # An independent prefix has no resampling salt; the inherited retry-state
    # must not cause an old response from a different request to be accepted.
    resumed = LocalLLM(ROOT, config, payload, new, replay_outputs=(old,))
    resumed.request = lambda *_: reply("新しい応答")
    assert resumed.cached_chat("source", messages) is None
    assert resumed.requests == 0
    assert resumed.chat("source", messages)["content"] == "新しい応答"
    assert resumed.chat("source", messages)["content"] == "保存済みの原文"
    assert {p.name: p.read_bytes() for p in old.iterdir()} == before
    changed = LocalLLM(ROOT, config, payload, new, replay_outputs=(old,))
    assert changed.cached_chat("source", [{"role": "user", "content": "別の本文"}]) is None


def test_exact_replay_skips_tokenization_and_generation(tmp_path):
    old = tmp_path / "llm-v5"
    new = tmp_path / "llm-v6"
    old.mkdir()
    config = load_config()
    messages = [{"role": "user", "content": "検査済みの原文"}]
    first = LocalLLM(ROOT, config, {"seed": 7}, old)
    first.request = lambda *_: reply()
    first.chat("source", messages)
    resumed = LocalLLM(ROOT, config, {"seed": 7}, new, replay_outputs=(old,))
    resumed.base_url = "http://model"
    resumed.request = lambda *_: pytest.fail("Exact cache replay must not contact the model")
    assert resumed.cached_chat("source", messages)["content"] == "accepted"
    assert not new.exists()


def scalable_config():
    config = load_config()
    config["llm"].update(max_context_size=32768, model_context_size=65536,
                         context_margin_tokens=512)
    return config


def test_stage_context_is_selected_before_loading_and_output_is_not_reduced(tmp_path):
    config = scalable_config()
    llm = LocalLLM(ROOT, config, {"seed": 1, "profiles": {
        "scene_text": {"context_size": 32768, "max_tokens": 6144}}}, tmp_path)
    assert llm._startup_context_size() == 32768
    llm.server_context_size = 32768
    llm._launch_context_size = 32768
    llm.set_profile({**llm.profile, "context_size": 32768, "max_tokens": 6144})
    llm.base_url = "http://model"
    calls = []

    def request(path, value):
        calls.append((path, value))
        if path == "/apply-template":
            return {"prompt": "original long source and exact schema"}
        if path == "/tokenize":
            return {"tokens": [1] * 26000}
        return reply()

    llm.request = request
    assert llm.chat("scene_text", [{"role": "user", "content": "long source"}])
    assert calls[-1][1]["max_tokens"] == 6144
    budget = next(item for item in llm.trace if item["type"] == "context_budget")
    assert budget["context_size"] == 32768
    assert budget["fits"] is True


@pytest.mark.parametrize("overrides", [
    {"context_size": 32768},
    {"max_context_size": 32768},
    {"allow_context_expansion": True},
    {"context_size": True},
    {"max_tokens": 9000},
    {"max_output_tokens": 16000},
])
def test_profile_cannot_raise_worker_resource_or_verified_model_limits(tmp_path, overrides):
    with pytest.raises(ValueError):
        LocalLLM(ROOT, load_config(), {"seed": 1, "profile": overrides}, tmp_path)


def test_resource_allowance_without_verified_model_limit_does_not_authorize_expansion(tmp_path):
    config = load_config()
    config["llm"]["max_context_size"] = 32768
    with pytest.raises(ValueError, match="verified context"):
        LocalLLM(ROOT, config, {"seed": 1, "profile": {"context_size": 32768}}, tmp_path)


def test_automatic_expansion_requires_preallocated_supported_server_and_preserves_output():
    settings = scalable_config()["llm"]
    settings["allow_context_expansion"] = True
    policy = ContextPolicy.resolve(settings, settings)
    assert policy.startup_size == 32768
    budget = policy.budget(19000, 6144, 24576)
    assert budget["context_size"] == 24576
    assert budget["expanded"] is True
    assert budget["output_tokens"] == 6144
    assert budget["fits"] is False
    larger = policy.budget(19000, 6144, 32768)
    assert larger["fits"] is True
    assert larger["output_tokens"] == 6144


def test_larger_server_cannot_silently_raise_profile_context():
    policy = ContextPolicy.resolve(scalable_config()["llm"], {"context_size": 16384})
    budget = policy.budget(14000, 3072, 32768)
    assert budget["context_size"] == 16384
    assert budget["fits"] is False
    assert budget["expanded"] is False


def test_reported_server_capacity_is_used_even_when_smaller_than_launch_setting(tmp_path):
    config = scalable_config()
    llm = LocalLLM(ROOT, config, {"seed": 1, "profile": {"context_size": 32768}}, tmp_path)
    llm.request = lambda *_args, **_kwargs: {"default_generation_settings": {"n_ctx": 8192}}
    llm._read_server_context(32768)
    assert llm.server_context_size == 8192
    assert llm.context_source == "server_properties"
    llm.request = lambda path, _value: (
        {"prompt": "full prompt"} if path == "/apply-template" else {"tokens": [1] * 5000})
    with pytest.raises(ContextBudgetError, match="8192"):
        llm.check_context("state", {"max_tokens": 3072})
    budget = json.loads((tmp_path / "last-context-budget.json").read_text("utf-8"))
    assert budget["server_context_size"] == 8192
    assert budget["output_tokens"] == 3072


def test_profile_context_cannot_be_changed_without_loading_appropriate_server(tmp_path):
    llm = LocalLLM(ROOT, scalable_config(), {"seed": 1}, tmp_path)
    llm.server_context_size = 16384
    llm._launch_context_size = 16384
    with pytest.raises(ValueError, match="running server"):
        llm.set_profile({"context_size": 32768})
    assert llm.profile["context_size"] == 16384


@pytest.mark.parametrize("change", ["configured", "server", "margin", "output_limit", "model"])
def test_cache_identity_changes_with_effective_context_and_model_settings(tmp_path, change):
    config = scalable_config()
    messages = [{"role": "user", "content": "the same source"}]
    first = LocalLLM(ROOT, config, {"seed": 1}, tmp_path)
    first.request = lambda *_: reply()
    first.chat("source", messages)
    replay = LocalLLM(ROOT, scalable_config(), {"seed": 1}, tmp_path)
    assert replay.cached_chat("source", messages)["content"] == "accepted"
    other = LocalLLM(ROOT, scalable_config(), {"seed": 1}, tmp_path)
    if change == "configured":
        other.set_profile({"context_size": 32768})
    elif change == "server":
        other.server_context_size = 8192
    elif change == "margin":
        other.set_profile({"context_margin_tokens": 768})
    elif change == "output_limit":
        other.set_profile({"max_output_tokens": 4096})
    else:
        other.base["model"]["revision"] = "another-revision"
    assert other.cached_chat("source", messages) is None
    assert other.requests == 0


def test_generation_measurements_distinguish_cache_cost_from_original_inference(tmp_path):
    response = {**reply(), "usage": {"prompt_tokens": 200, "completion_tokens": 20},
                "timings": {"prompt_ms": 120, "predicted_ms": 30, "predicted_n": 20}}
    llm = LocalLLM(ROOT, load_config(), {"seed": 1}, tmp_path)
    llm.request = lambda *_: response
    llm.chat("review", [{"role": "user", "content": "source"}])
    metric = llm.trace[-1]
    assert metric["cache_hit"] is False
    assert metric["generated_tokens_this_run"] == 20
    assert metric["server_timings"]["prompt_ms"] == 120
    assert metric["generation_http_seconds"] >= 0
    resumed = LocalLLM(ROOT, load_config(), {"seed": 1}, tmp_path)
    resumed.request = lambda *_: pytest.fail("cache hit cannot call the model")
    resumed.chat("review", [{"role": "user", "content": "source"}])
    metric = resumed.trace[-1]
    assert metric["cache_hit"] is True
    assert metric["generated_tokens_this_run"] == 0
    assert metric["generation_http_seconds"] == 0
    assert metric["server_timings_from_cache"] is True


def test_server_launch_release_and_http_timing_keep_owned_process_lifetime(tmp_path, monkeypatch):
    from services.worker.generation import llm as module

    config = scalable_config()
    model = tmp_path / "model.gguf"
    model.write_bytes(b"GGUF")
    server = tmp_path / "server.exe"
    server.touch()
    base = tmp_path / "base.json"
    base.write_text(json.dumps({"model": {"relative_path": str(model), "size_bytes": 4},
                                "server": {"executable": str(server)}}))
    config["llm_config"] = str(base)
    commands = []
    lifecycle = []

    @contextmanager
    def owner(command, *_args, **_kwargs):
        commands.append(command)
        lifecycle.append("start")
        try:
            yield type("Process", (), {"poll": lambda _: None})()
        finally:
            lifecycle.append("stop")

    class HTTP:
        def open(self, request, timeout):
            if request.full_url.endswith("/health"):
                result = {"status": "ok"}
            elif request.full_url.endswith("/props"):
                result = {"default_generation_settings": {"n_ctx": 24576}}
            else:
                raise AssertionError(request.full_url)
            response = BytesIO(json.dumps(result).encode())
            response.status = 200
            return response

    monkeypatch.setattr(module, "owned_process", owner)
    client = LocalLLM(ROOT, config, {"seed": 1, "profile": {"context_size": 32768}}, tmp_path)
    client.http = HTTP()
    with client as active:
        assert active.server_context_size == 24576
        assert lifecycle == ["start"]
    assert lifecycle == ["start", "stop"]
    assert commands[0][commands[0].index("--ctx-size") + 1] == "32768"
    trace = json.loads((tmp_path / "llm-metrics.json").read_text("utf-8"))
    assert {item["type"] for item in trace} >= {"model_load", "model_release", "llm_http"}
    assert all(item["elapsed_seconds"] >= 0 for item in trace if "elapsed_seconds" in item)


def test_failed_http_call_records_cost_and_does_not_hide_failure(tmp_path):
    client = LocalLLM(ROOT, load_config(), {"seed": 1}, tmp_path)
    client.deadline = time.monotonic() + 10
    client.base_url = "http://model"

    class FailingHTTP:
        def open(self, *_args, **_kwargs):
            raise TimeoutError("fixture timeout")

    client.http = FailingHTTP()
    with pytest.raises(TimeoutError, match="fixture timeout"):
        client.request("/v1/chat/completions", {})
    assert client.trace[-1]["status"] == "error"
    assert client.trace[-1]["elapsed_seconds"] >= 0
