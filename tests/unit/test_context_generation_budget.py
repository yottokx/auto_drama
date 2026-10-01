"""Share context growth between input and total generation without changing source profiles."""

import copy
import hashlib
import json
import time

import pytest

from services.worker.generation import model_routing as routing
from services.worker.generation.context_budget import OutputTokenPolicy
from services.worker.generation.llm import ContextBudgetError, LocalLLM
from services.worker.generation.pipeline import ROOT, load_config


MESSAGES = [{"role": "user", "content": "採用本文と設定を省略せず保持する。"}]


def configured(context=32768, *, common=True, effort="medium"):
    config = load_config()
    config.pop("model_routing", None)
    config["llm"].update(context_size=context, max_context_size=context,
                         model_context_size=context, context_margin_tokens=512,
                         max_output_tokens=8192)
    profile = {"context_size": context, "reasoning_level": effort if common else "none"}
    if common:
        profile["common_settings_version"] = 1
    return config, {"seed": 7, "profile": profile}


def reply():
    return {"choices": [{"finish_reason": "stop", "message": {"content": "accepted"}}],
            "usage": {"prompt_tokens": 500, "completion_tokens": 20}}


@pytest.mark.parametrize("context,extension", [(16384, 0), (32768, 8192),
                                               (65536, 24576), (32769, 8192)])
@pytest.mark.parametrize("client", [LocalLLM, routing.RoutedLLM])
def test_common_m2_output_grows_by_half_the_context_extension(tmp_path, context, extension, client):
    config, payload = configured(context)
    llm = client(ROOT, config, payload, tmp_path)
    request, _ = llm._chat_request(1, MESSAGES, {})
    budget = llm.context_budget()
    assert request["max_tokens"] == budget["output_tokens"] == 3072 + extension
    assert budget["input_token_limit"] == context - 3072 - extension - 512
    assert llm._output_limit(llm.profile) == 8192 + extension
    assert llm.profile["max_tokens"] == 3072
    assert payload["profile"] == {"context_size": context, "reasoning_level": "medium",
                                   "common_settings_version": 1}


@pytest.mark.parametrize("purpose,baseline", [("script-allocation", 2048),
                                              ("script-plan", 6144),
                                              ("script-outline", 8192),
                                              ("script-handoff", 4096)])
def test_script_purposes_keep_distinct_body_budgets_and_share_the_same_extension(tmp_path, purpose, baseline):
    config, payload = configured()
    payload.update(story_workflow_version=2, workflow_policy="script_continuation_v1")
    llm = routing.RoutedLLM(ROOT, config, payload, tmp_path)
    original_config = copy.deepcopy(config)
    original_payload = copy.deepcopy(payload)
    original_identity = llm.model_configuration_identity()
    for _ in range(3):
        llm.select_purpose(purpose)
        llm.set_profile(llm.profile)
        request, _ = llm._chat_request(1, MESSAGES, {})
        assert llm.profile["max_tokens"] == baseline
        assert request["max_tokens"] == baseline + 8192
        assert llm.context_budget()["output_tokens"] == baseline + 8192
    assert config == original_config and payload == original_payload
    assert llm.model_configuration_identity() == original_identity


@pytest.mark.parametrize("effort", ["none", "low", "medium", "high", "custom-effort"])
def test_reasoning_effort_is_forwarded_without_changing_the_allocation(tmp_path, effort):
    config, payload = configured(effort=effort)
    llm = LocalLLM(ROOT, config, payload, tmp_path)
    request, _ = llm._chat_request(1, MESSAGES, {})
    assert request["reasoning_effort"] == effort
    assert request["max_tokens"] == 11264


def test_legacy_profiles_do_not_gain_generation_tokens_from_context_alone(tmp_path):
    config, payload = configured(common=False)
    llm = LocalLLM(ROOT, config, payload, tmp_path)
    request, _ = llm._chat_request(1, MESSAGES, {})
    assert request["max_tokens"] == llm.context_budget()["output_tokens"] == 3072
    assert llm._output_limit(llm.profile) == 8192


def test_smaller_profile_allowance_limits_the_baseline_before_adding_extension(tmp_path):
    config, payload = configured()
    payload["profile"]["max_output_tokens"] = 4096
    llm = LocalLLM(ROOT, config, payload, tmp_path)
    assert llm._output_limit(llm.profile) == 12288
    request, _ = llm._chat_request(1, MESSAGES, {"max_tokens": 12288})
    assert request["max_tokens"] == 12288
    with pytest.raises(ValueError, match="output allowance"):
        llm._chat_request(1, MESSAGES, {"max_tokens": 12289})


@pytest.mark.parametrize("overrides", [{"max_output_tokens": 8193}, {"max_tokens": 8193},
                                        {"max_tokens": True}, {"max_output_tokens": True}])
def test_extension_does_not_let_profiles_bypass_the_baseline_allowance(tmp_path, overrides):
    config, payload = configured()
    payload["profile"].update(overrides)
    with pytest.raises(ValueError):
        LocalLLM(ROOT, config, payload, tmp_path)


def test_policy_requires_a_valid_baseline_and_keeps_the_default_allowance():
    policy = OutputTokenPolicy.resolve({}, {"common_settings_version": 1, "context_size": 32768})
    assert policy.base_limit == 8192 and policy.extension_tokens == 8192
    assert policy.limit == 16384 and policy.tokens(2048) == 10240
    with pytest.raises(ValueError):
        policy.tokens(8193)
    with pytest.raises(ValueError):
        policy.tokens(True)


@pytest.mark.parametrize("explicit", [2048, 10240, 16384])
def test_explicit_request_and_context_amounts_are_already_total(tmp_path, explicit):
    config, payload = configured()
    llm = LocalLLM(ROOT, config, payload, tmp_path)
    request, _ = llm._chat_request(1, MESSAGES, {"max_tokens": explicit})
    assert request["max_tokens"] == explicit
    assert llm.context_budget(output_tokens=explicit)["output_tokens"] == explicit


def test_exact_input_boundary_preserves_source_and_reserves_the_expanded_output(tmp_path):
    config, payload = configured()
    payload["profile"]["max_tokens"] = 2048
    llm = LocalLLM(ROOT, config, payload, tmp_path)
    llm.base_url = "http://fake"
    captured = []
    source = copy.deepcopy(MESSAGES)

    def request(path, value):
        captured.append((path, copy.deepcopy(value)))
        if path == "/apply-template":
            return {"prompt": "exact template including original source"}
        if path == "/tokenize":
            return {"tokens": [1] * 22016}
        assert path == "/v1/chat/completions"
        return reply()

    llm.request = request
    assert llm.chat("script-allocation", source)["content"] == "accepted"
    assert source == MESSAGES
    assert captured[-1][1]["messages"] == MESSAGES
    assert captured[-1][1]["max_tokens"] == 10240
    budget = next(row for row in llm.trace if row["type"] == "context_budget")
    assert budget["required_tokens"] == budget["context_size"] == 32768
    assert budget["fits"] is True


def test_smaller_actual_server_fails_without_shrinking_input_or_output(tmp_path):
    config, payload = configured()
    payload["profile"]["max_tokens"] = 2048
    llm = LocalLLM(ROOT, config, payload, tmp_path)
    llm.base_url = "http://fake"
    llm.server_context_size = 16384
    calls = []
    source = copy.deepcopy(MESSAGES)

    def request(path, value):
        calls.append((path, copy.deepcopy(value)))
        if path == "/apply-template":
            return {"prompt": "original source"}
        assert path == "/tokenize", "Generation must not run after a failed context check."
        return {"tokens": [1] * 6000}

    llm.request = request
    with pytest.raises(ContextBudgetError) as error:
        llm.chat("script-allocation", source)
    assert error.value.budget["output_tokens"] == 10240
    assert error.value.budget["context_size"] == 16384
    assert source == MESSAGES and calls[0][1]["messages"] == MESSAGES
    assert llm.profile["max_tokens"] == 2048
    assert not list(tmp_path.glob("01-script-allocation-*.json"))


def test_16k_existing_request_cache_remains_replayable(tmp_path):
    config, payload = configured(16384)
    llm = LocalLLM(ROOT, config, payload, tmp_path)
    # Build the pre-extension request shape, including its original baseline profile.
    original_request = {"model": "m2-local", "messages": MESSAGES, "stream": False,
                        "seed": 8, "temperature": llm.profile["temperature"],
                        "max_tokens": 3072, "top_p": llm.profile.get("top_p", 0.95),
                        "reasoning_effort": "medium"}
    original_runtime = llm.runtime_identity()
    assert original_runtime["version"] == 2
    assert original_runtime["profile"] == {**config["llm"], **payload["profile"]}
    assert original_runtime["output_limit"] == 8192
    fingerprint = hashlib.sha256(json.dumps({"request": original_request, "runtime": original_runtime},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    cache = tmp_path / f"01-source-{fingerprint[:16]}.json"
    cache.write_text(json.dumps({"request_sha256": fingerprint, "request": original_request,
                                 "runtime": original_runtime, "response": reply()}), encoding="utf-8")
    original_bytes = cache.read_bytes()
    llm.request = lambda *_args, **_kwargs: pytest.fail("Existing 16k cache must need no model call.")
    assert llm.cached_chat("source", MESSAGES)["content"] == "accepted"
    assert cache.read_bytes() == original_bytes


def test_routed_backend_receives_baseline_profile_and_wire_request_is_extended_once(tmp_path, monkeypatch):
    config, payload = configured()
    payload.update(story_workflow_version=2, workflow_policy="script_continuation_v1")
    loaded, sent = [], []

    class Backend(LocalLLM):
        def __enter__(self):
            loaded.append(copy.deepcopy(self.profile))
            self.base_url = "http://fake"
            self._launch_context_size = self.server_context_size = 32768
            self.deadline = time.monotonic() + 1000
            return self

        def __exit__(self, *_args):
            pass

        def request(self, path, value=None, timeout=None):
            if path == "/apply-template":
                return {"prompt": "exact template"}
            if path == "/tokenize":
                return {"tokens": [1] * 500}
            assert path == "/v1/chat/completions"
            sent.append(copy.deepcopy(value))
            return reply()

    monkeypatch.setattr(routing, "LocalLLM", Backend)
    with routing.RoutedLLM(ROOT, config, payload, tmp_path) as llm:
        for _ in range(2):
            llm.select_purpose("script-allocation")
            llm.chat("script-allocation", MESSAGES)
    assert len(loaded) == 1 and loaded[0]["max_tokens"] == 2048
    assert [row["max_tokens"] for row in sent] == [10240, 10240]
