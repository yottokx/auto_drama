"""External API models behind LocalLLM: conversion, transport and the worker's model list."""

import json
import urllib.error
import urllib.request
from io import BytesIO

import pytest

from packages.contracts.llm_settings import LLMSettings, model_provider
from services.worker.generation import llm_providers as providers
from services.worker.generation import narrative
from services.worker.generation.llm import ContextBudgetError, LocalLLM
from services.worker.generation.model_routing import RoutedLLM
from services.worker.generation.pipeline import ROOT, load_config
from services.worker.model_config import catalog, select_config

KEY = "test-secret-key-value"


class FakeOpener:
    """Answer queued replies; an int queues an HTTP error with that status."""

    def __init__(self, *replies):
        self.replies, self.requests = list(replies), []

    def open(self, request, timeout):
        self.requests.append({"url": request.full_url, "headers": dict(request.header_items()),
                              "body": json.loads(request.data)})
        reply = self.replies.pop(0)
        if isinstance(reply, int):
            raise urllib.error.HTTPError(request.full_url, reply, "error", {"retry-after": "0"},
                                         BytesIO(json.dumps({"error": {"message": f"status {reply}"}}).encode()))
        response = BytesIO(json.dumps(reply).encode())
        response.status = 200
        return response


def worker_root(tmp_path, *models):
    root = tmp_path / "root"
    (root / "services/worker/config").mkdir(parents=True)
    (root / "services/worker/config/models.local.json").write_text(json.dumps(
        {"schema_version": 1, "api_models": [{"id": model, "max_input_tokens": 1_000_000} for model in models]}))
    return root


def external_config(tmp_path, model, **settings):
    profile = LLMSettings(model=model, temperature=0.65, top_p=0.95,
                          reasoning_effort=settings.get("reasoning_effort", "none"),
                          ctx_size=settings.get("ctx_size", 16384)).profile()
    return select_config(load_config(), {"profile": profile}, worker_root(tmp_path, model)), profile


def anthropic_text(text, stop="end_turn"):
    return {"model": "claude-haiku-5-5", "stop_reason": stop, "content": [{"type": "text", "text": text}],
            "usage": {"input_tokens": 40, "output_tokens": 12, "output_tokens_details": {"thinking_tokens": 0}}}


def test_model_ids_name_their_provider():
    assert model_provider("openai/gpt-6-luna") == "openai"
    assert model_provider("anthropic/claude-haiku-5-5") == "anthropic"
    assert model_provider("gemma-4-31B-it-UD-Q4_K_XL") == "local"
    assert LLMSettings(model="openai/gpt-6-luna", temperature=1, top_p=1, reasoning_effort="none",
                       ctx_size=16384).profile()["provider"] == "openai"
    with pytest.raises(ValueError):
        providers.split_model_id("mistral/large")


def test_schemas_are_converted_for_each_provider_without_touching_property_names():
    schema = {"type": "object", "additionalProperties": False, "required": ["a"],
              "properties": {"a": {"type": "array", "minItems": 2, "maxItems": 5, "uniqueItems": True,
                                   "items": {"$ref": "#/$defs/Item"}},
                             "minimum": {"type": "integer", "minimum": 1, "maximum": 5},
                             "last": {"type": ["null", "string"], "enum": [None, "s1-u1"]}},
              "$defs": {"Item": {"type": "object", "additionalProperties": False, "required": [],
                                 "properties": {"x": {"type": "string", "maxLength": 9}}}}}
    openai = providers.openai_schema(schema)
    assert openai["required"] == ["a", "minimum", "last"]
    assert openai["$defs"]["Item"]["required"] == ["x"]
    assert "uniqueItems" not in openai["properties"]["a"] and openai["properties"]["a"]["maxItems"] == 5
    anthropic = providers.anthropic_schema(schema)
    assert anthropic["required"] == ["a"]
    assert anthropic["properties"]["a"] == {"type": "array", "minItems": 1, "items": {"$ref": "#/$defs/Item"}}
    assert anthropic["properties"]["minimum"] == {"type": "integer"}
    assert anthropic["properties"]["last"] == {"anyOf": [{"type": "string", "enum": ["s1-u1"]}, {"type": "null"}]}
    assert anthropic["$defs"]["Item"]["properties"]["x"]["maxLength"] == 9
    assert schema["properties"]["a"]["minItems"] == 2


def test_structured_requests_state_the_counts_and_ranges_a_grammar_would_enforce():
    schema = {"type": "object", "additionalProperties": False, "required": ["subs", "pairs"],
              "properties": {"subs": {"type": "array", "maxItems": 1, "items": {"$ref": "#/$defs/Sub"}},
                             "pairs": {"type": "array", "uniqueItems": True, "items": {
                                 "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"}}}},
              "$defs": {"Sub": {"type": "object", "additionalProperties": False, "required": ["lines", "age"],
                                "properties": {"lines": {"type": "array", "minItems": 3, "maxItems": 3,
                                                         "items": {"type": "string", "maxLength": 9}},
                                               "age": {"anyOf": [{"type": "integer", "minimum": 1},
                                                                 {"type": "null"}]}}}}}
    assert providers.schema_limits(schema) == [
        "subs: 最大1件", "subs[].lines: 3件", "subs[].age: 1以上", "pairs: 重複なし", "pairs[]: 2件"]
    request = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "作って"}],
               "max_tokens": 512, "response_format": {"type": "json_schema", "json_schema": {
                   "name": "cast", "strict": True, "schema": schema}}}
    for body in (providers.openai_request(request, "gpt-6-luna")["input"],
                 providers.anthropic_request(request, "claude-haiku-5-5")["messages"]):
        assert body[-1]["content"].startswith("作って\n\n回答のJSONの件数と数値の範囲（必ず守る）:\n- subs: 最大1件\n")
    assert request["messages"][-1]["content"] == "作って"
    plain = {**request, "response_format": None}
    plain.pop("response_format")
    assert providers.openai_request(plain, "gpt-6-luna")["input"][-1]["content"] == "作って"


def test_openai_request_keeps_sampling_only_without_reasoning():
    request = {"model": "m2-local", "messages": [{"role": "system", "content": "脚本家"},
                                                 {"role": "user", "content": "書いて"}],
               "stream": False, "seed": 3, "temperature": 0.65, "top_p": 0.95, "max_tokens": 2048,
               "reasoning_effort": "none",
               "response_format": {"type": "json_schema", "json_schema": {
                   "name": "script-plan", "strict": True, "schema": {
                       "type": "object", "properties": {"a": {"type": "string"}}, "required": []}}}}
    body = providers.openai_request(request, "gpt-6-luna")
    assert body["instructions"] == "脚本家"
    assert body["input"] == [{"role": "user", "content": "書いて"}]
    assert (body["max_output_tokens"], body["store"], body["reasoning"]) == (2048, False, {"effort": "none"})
    assert (body["temperature"], body["top_p"]) == (0.65, 0.95)
    assert body["text"]["format"]["name"] == "script-plan"
    assert body["text"]["format"]["schema"]["required"] == ["a"]
    assert "seed" not in body
    reasoning = providers.openai_request({**request, "reasoning_effort": "low"}, "gpt-6-luna")
    assert "temperature" not in reasoning and "top_p" not in reasoning
    with pytest.raises(ValueError, match="grammar"):
        providers.openai_request({**request, "grammar": "root ::= x"}, "gpt-6-luna")


def test_anthropic_request_maps_reasoning_and_never_sends_sampling():
    request = {"model": "m2-local", "messages": [{"role": "system", "content": "s"},
                                                 {"role": "user", "content": "u"}],
               "temperature": 0.65, "top_p": 0.95, "max_tokens": 1024, "reasoning_effort": "none"}
    body = providers.anthropic_request(request, "claude-haiku-5-5")
    assert body == {"model": "claude-haiku-5-5", "messages": [{"role": "user", "content": "u"}],
                    "max_tokens": 1024, "system": "s", "thinking": {"type": "disabled"}}
    thinking = providers.anthropic_request({**request, "reasoning_effort": "high"}, "claude-haiku-5-5")
    assert thinking["thinking"] == {"type": "adaptive"}
    assert thinking["output_config"] == {"effort": "high"}


def test_tool_rounds_convert_both_ways():
    tool = {"type": "function", "function": {"name": "random_integer", "description": "Draw.",
            "parameters": {"type": "object", "additionalProperties": False, "required": ["minimum"],
                           "properties": {"minimum": {"type": "integer"},
                                          "items": {"type": "array", "maxItems": 100, "items": {"type": "string"}}}}}}
    messages = [{"role": "user", "content": "引いて"},
                {"role": "assistant", "content": None, "tool_calls": [
                    {"id": "c1", "type": "function", "function": {"name": "random_integer",
                                                                  "arguments": "{\"minimum\": 1}"}}]},
                {"role": "tool", "tool_call_id": "c1", "content": "{\"value\": 4}"}]
    request = {"messages": messages, "max_tokens": 300, "tools": [tool], "tool_choice": "auto",
               "parallel_tool_calls": False}
    openai = providers.openai_request(request, "gpt-6-luna")
    assert openai["input"][1:] == [
        {"type": "function_call", "call_id": "c1", "name": "random_integer", "arguments": "{\"minimum\": 1}"},
        {"type": "function_call_output", "call_id": "c1", "output": "{\"value\": 4}"}]
    assert openai["tools"][0]["strict"] is True and openai["parallel_tool_calls"] is False
    anthropic = providers.anthropic_request(request, "claude-haiku-5-5")
    assert anthropic["messages"][1]["content"] == [
        {"type": "tool_use", "id": "c1", "name": "random_integer", "input": {"minimum": 1}}]
    assert anthropic["messages"][2] == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "c1", "content": "{\"value\": 4}"}]}
    assert "maxItems" not in anthropic["tools"][0]["input_schema"]["properties"]["items"]
    assert anthropic["tool_choice"] == {"type": "auto", "disable_parallel_tool_use": True}
    reply = providers.anthropic_reply({"stop_reason": "tool_use", "content": [
        {"type": "tool_use", "id": "c2", "name": "random_integer", "input": {"minimum": 2}}], "usage": {}})
    assert reply["choices"][0]["finish_reason"] == "tool_calls"
    assert json.loads(reply["choices"][0]["message"]["tool_calls"][0]["function"]["arguments"]) == {"minimum": 2}


def test_replies_become_chat_completions_with_finish_reasons_and_usage():
    complete = providers.openai_reply({"status": "completed", "model": "gpt-6-luna", "output": [
        {"type": "reasoning"}, {"type": "message", "content": [{"type": "output_text", "text": "本文"}]}],
        "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15,
                  "output_tokens_details": {"reasoning_tokens": 2}}})
    assert complete["choices"][0] == {"index": 0, "finish_reason": "stop",
                                      "message": {"role": "assistant", "content": "本文"}}
    assert complete["usage"]["completion_tokens"] == 5
    assert complete["usage"]["completion_tokens_details"] == {"reasoning_tokens": 2}
    cut = providers.openai_reply({"status": "incomplete", "incomplete_details": {"reason": "max_output_tokens"},
                                  "output": [{"type": "message", "content": [{"type": "output_text", "text": "途"}]}]})
    assert cut["choices"][0]["finish_reason"] == "length"
    assert providers.anthropic_reply(anthropic_text("途中", "max_tokens"))["choices"][0]["finish_reason"] == "length"
    with pytest.raises(providers.ExternalRefusal):
        providers.openai_reply({"status": "completed", "output": [
            {"type": "message", "content": [{"type": "refusal", "refusal": "no"}]}]})
    with pytest.raises(providers.ExternalRefusal):
        providers.anthropic_reply({"stop_reason": "refusal", "stop_details": {"category": "cyber"}})


def test_transport_retries_overload_and_never_reveals_the_key(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    monkeypatch.setattr(providers.time, "sleep", lambda _seconds: None)
    llm = LocalLLM(ROOT, load_config(), {"seed": 1}, tmp_path)
    llm.deadline = providers.time.monotonic() + 60
    opener = FakeOpener(529, anthropic_text("おはよう"), 400)
    remote = providers.ExternalLLM("anthropic", "claude-haiku-5-5", llm, opener=opener)
    request = {"messages": [{"role": "user", "content": "u"}], "max_tokens": 64, "reasoning_effort": "none"}
    assert remote.chat(request)["choices"][0]["message"]["content"] == "おはよう"
    assert opener.requests[0]["headers"]["X-api-key"] == KEY
    assert opener.requests[0]["headers"]["Anthropic-version"] == providers.ANTHROPIC_VERSION
    with pytest.raises(RuntimeError, match="HTTP 400: status 400") as failure:
        remote.chat(request)
    assert KEY not in str(failure.value)
    assert [row["http_status"] for row in llm.trace] == [529, 200, 400]
    assert KEY not in json.dumps(llm.trace)
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    with pytest.raises(RuntimeError, match="ANTHROPIC_API_KEY is not set"):
        remote.chat(request)


def test_worker_offers_api_models_only_while_their_key_is_set(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    root = worker_root(tmp_path, "openai/gpt-6-luna", "anthropic/claude-haiku-5-5")
    models, errors = catalog(root)
    assert models == [{"model": "anthropic/claude-haiku-5-5", "reasoning_efforts": providers.REASONING_EFFORTS[
        "anthropic"], "sampling_efforts": []}]
    assert errors == ["openai/gpt-6-luna: OPENAI_API_KEY is not set for this worker"]
    assert KEY not in json.dumps(models)


def test_external_chat_counts_against_the_model_limit_and_caches_like_local(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEY)
    config, profile = external_config(tmp_path, "anthropic/claude-haiku-5-5")
    opener = FakeOpener({"input_tokens": 20000}, anthropic_text("NARRATOR: 朝。"))
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: opener)
    output = tmp_path / "llm"
    with LocalLLM(ROOT, config, {"seed": 1, "profile": profile}, output) as llm:
        assert llm.process is None and llm.base_url == ""
        reply = llm.chat("scene", [{"role": "user", "content": "書いて"}], allow_truncated=True)
    assert reply["content"] == "NARRATOR: 朝。"
    assert [request["url"] for request in opener.requests] == [
        "https://api.anthropic.com/v1/messages/count_tokens", "https://api.anthropic.com/v1/messages"]
    assert "max_tokens" not in opener.requests[0]["body"]
    assert "temperature" not in opener.requests[1]["body"]
    budget = next(row for row in llm.trace if row["type"] == "context_budget")
    # 20,000 tokens do not fit the 16k tier the work is sized by, but fit the model.
    assert budget["fits"] and budget["expanded"] and budget["context_size"] == 1_000_000
    cache = next(output.glob("01-scene-*.json"))
    saved = json.loads(cache.read_text("utf-8"))
    assert saved["runtime"]["model"] == {"provider": "anthropic", "api_model": "claude-haiku-5-5",
                                         "protocol": providers.PROTOCOL_VERSION}
    assert saved["response"]["choices"][0]["finish_reason"] == "stop"
    assert KEY not in cache.read_text("utf-8")
    assert KEY not in (output / "llm-metrics.json").read_text("utf-8")
    replay = LocalLLM(ROOT, config, {"seed": 1, "profile": profile}, tmp_path / "again", replay_outputs=(output,))
    assert replay.cached_chat("scene", [{"role": "user", "content": "書いて"}],
                              allow_truncated=True)["content"] == "NARRATOR: 朝。"
    assert len(opener.requests) == 2


def test_external_input_beyond_the_model_limit_fails_before_generation(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    config, profile = external_config(tmp_path, "openai/gpt-6-luna")
    opener = FakeOpener({"input_tokens": 999_000})
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: opener)
    with (LocalLLM(ROOT, config, {"seed": 1, "profile": profile}, tmp_path / "llm") as llm,
          pytest.raises(ContextBudgetError)):
        llm.chat("scene", [{"role": "user", "content": "書いて"}])
    assert len(opener.requests) == 1
    assert opener.requests[0]["url"] == "https://api.openai.com/v1/responses/input_tokens"
    # The counting endpoint rejects generation-only fields.
    assert not {"temperature", "top_p", "max_output_tokens", "store"} & set(opener.requests[0]["body"])


def test_routed_script_runs_reach_the_external_model(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", KEY)
    config, profile = external_config(tmp_path, "openai/gpt-6-luna")
    opener = FakeOpener({"input_tokens": 50}, {"status": "completed", "model": "gpt-6-luna", "output": [
        {"type": "message", "content": [{"type": "output_text", "text": "{\"a\": \"x\"}"}]}], "usage": {}})
    monkeypatch.setattr(urllib.request, "build_opener", lambda *_args: opener)
    with RoutedLLM(ROOT, config, {"seed": 1, "profile": profile}, tmp_path / "llm") as llm:
        result = llm.structured("plan", [{"role": "user", "content": "計画"}], {
            "type": "object", "additionalProperties": False, "required": ["a"],
            "properties": {"a": {"type": "string"}}})
    assert result == {"a": "x"}
    assert opener.requests[1]["body"]["text"]["format"]["strict"] is True


def test_external_scene_source_names_the_speakers_a_grammar_would_allow():
    plan = type("Plan", (), {"character_ids": ["c1"]})()
    for effort in ("none", "low"):
        assert narrative._source_options({"reasoning_level": effort, "provider": "openai"}, plan) == {
            "allowed_speakers": ["NARRATOR", "c1"]}
    assert "grammar" in narrative._source_options({"reasoning_level": "none", "provider": "local"}, plan)
    request = {"messages": [{"role": "user", "content": "書いて"}], "max_tokens": 512,
               "allowed_speakers": ["NARRATOR", "c1"]}
    for body, turns in ((providers.openai_request(request, "gpt-6-luna"), "input"),
                        (providers.anthropic_request(request, "claude-haiku-5-5"), "messages")):
        assert "allowed_speakers" not in body
        assert body[turns][-1]["content"].startswith(
            "書いて\n\nこのシーンで発話できる人物IDは次の一覧のみです: [\"NARRATOR\", \"c1\"]。")
