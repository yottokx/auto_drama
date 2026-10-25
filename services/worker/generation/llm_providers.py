"""OpenAI Responses and Anthropic Messages behind the chat-completions shape LocalLLM keeps.

Requests are built as for llama-server and converted only here, just before sending; replies
are converted back to choices[0].message / finish_reason / usage. API keys are read from the
worker's environment for each request and never stored, traced, cached or put in errors.
"""
from __future__ import annotations

import copy
import json
import os
import re
import time
import urllib.error
import urllib.request

from packages.contracts.llm_settings import API_PROVIDERS as PROVIDERS

from .cancellation import check_cancelled

KEY_VARIABLES = {"openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}
ANTHROPIC_VERSION = "2026-10-21"
# Part of each cached request's model identity: a converter change must not replay old replies.
# 2: structured requests state their counts and numeric ranges in words.
PROTOCOL_VERSION = 2
# What each API accepts for reasoning (docs/experiments/external-llm-api-probe-20261025.md).
REASONING_EFFORTS = {"openai": ["none", "low", "medium", "high", "xhigh", "max"],
                     "anthropic": ["none", "low", "medium", "high", "xhigh", "max"]}
# The reasoning efforts with which temperature and top_p are accepted.
SAMPLING_EFFORTS = {"openai": ["none"], "anthropic": []}
RETRY_STATUSES = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
MAX_RETRIES = 2
# Fields of the llama-server request that have a counterpart (or are deliberately dropped).
REQUEST_FIELDS = frozenset({"model", "messages", "stream", "seed", "temperature", "top_p",
                            "max_tokens", "reasoning_effort", "response_format", "tools",
                            "tool_choice", "parallel_tool_calls"})


class ExternalRefusal(RuntimeError):
    """The provider declined to answer; nothing to validate or retry with feedback."""


def split_model_id(model_id: str) -> tuple[str, str]:
    """`openai/gpt-6-luna` -> ("openai", "gpt-6-luna"). GGUF file names cannot contain '/'."""
    provider, separator, model = model_id.partition("/")
    if not separator or provider not in PROVIDERS or not model:
        raise ValueError(f"External model IDs are provider/model with provider in {PROVIDERS}: {model_id}")
    return provider, model


def has_key(provider: str) -> bool:
    return bool(os.environ.get(KEY_VARIABLES[provider]))


# ----- JSON schema conversion (the original schema still validates every reply) -----

_SCHEMA_MAPS = ("properties", "$defs", "definitions")
_SCHEMA_LISTS = ("anyOf", "allOf", "oneOf", "prefixItems")
_SCHEMA_VALUES = ("items", "not", "additionalProperties")


def _map_schema(node, convert):
    if not isinstance(node, dict):
        return copy.deepcopy(node)
    result = {}
    for key, value in node.items():
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            result[key] = {name: _map_schema(child, convert) for name, child in value.items()}
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            result[key] = [_map_schema(child, convert) for child in value]
        elif key in _SCHEMA_VALUES and isinstance(value, dict):
            result[key] = _map_schema(value, convert)
        else:
            result[key] = copy.deepcopy(value)
    return convert(result)


def openai_schema(schema: dict) -> dict:
    """Strict mode lists every property as required and has no uniqueItems."""
    def convert(node):
        node.pop("uniqueItems", None)
        if isinstance(node.get("properties"), dict):
            node["required"] = list(node["properties"])
        return node
    return _map_schema(schema, convert)


def anthropic_schema(schema: dict) -> dict:
    """No numeric bounds, maxItems or uniqueItems; minItems only 0 or 1; null enums as anyOf."""
    def convert(node):
        for key in ("minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf",
                    "maxItems", "uniqueItems"):
            node.pop(key, None)
        if isinstance(node.get("minItems"), int) and node["minItems"] > 1:
            node["minItems"] = 1
        kinds = node.get("type")
        if isinstance(kinds, list) and "null" in kinds and None in node.get("enum", []):
            others = [kind for kind in kinds if kind != "null"]
            rest = {key: value for key, value in node.items() if key not in ("type", "enum")}
            return {**rest, "anyOf": [
                {"type": others[0] if len(others) == 1 else others,
                 "enum": [value for value in node["enum"] if value is not None]},
                {"type": "null"}]}
        return node
    return _map_schema(schema, convert)


def _count(low, high, unit: str) -> str:
    if low is not None and low == high:
        return f"{low}{unit}"
    if low is not None and high is not None:
        return f"{low}〜{high}{unit}"
    return f"{low}{unit}以上" if low is not None else f"最大{high}{unit}"


def schema_limits(schema: dict) -> list[str]:
    """Counts and numeric ranges, by field path, that the reply must meet.

    llama-server's grammar stops the local model at these bounds without it knowing them.
    Anthropic does not enforce them at all, and a model held back by OpenAI's strict mode
    without knowing why can derail mid-answer, so external models are told them in words.
    """
    definitions = schema.get("$defs", {})
    lines: list[str] = []

    def walk(node, path, depth=0):
        if not isinstance(node, dict) or depth > 32:
            return
        if "$ref" in node:
            walk(definitions.get(node["$ref"].rsplit("/", 1)[-1]), path, depth + 1)
            return
        for option in node.get("anyOf", []) + node.get("allOf", []):
            walk(option, path, depth + 1)
        if "minItems" in node or "maxItems" in node or node.get("uniqueItems"):
            parts = []
            if "minItems" in node or "maxItems" in node:
                parts.append(_count(node.get("minItems"), node.get("maxItems"), "件"))
            if node.get("uniqueItems"):
                parts.append("重複なし")
            lines.append(f"{path}: {'、'.join(parts)}")
        if "minimum" in node or "maximum" in node:
            lines.append(f"{path}: {_count(node.get('minimum'), node.get('maximum'), '')}")
        if isinstance(node.get("items"), dict):
            walk(node["items"], path + "[]", depth + 1)
        for name, child in (node.get("properties") or {}).items():
            walk(child, f"{path}.{name}" if path else name, depth + 1)

    walk(schema, "")
    return list(dict.fromkeys(line for line in lines if not line.startswith(": ")))


def _with_limits(turns: list[dict], schema: dict) -> list[dict]:
    limits = schema_limits(schema)
    if not limits or not turns or turns[-1]["role"] != "user" or not isinstance(turns[-1]["content"], str):
        return turns
    note = "\n\n回答のJSONの件数と数値の範囲（必ず守る）:\n" + "\n".join("- " + line for line in limits)
    return [*turns[:-1], {**turns[-1], "content": turns[-1]["content"] + note}]


# ----- requests -----

def _check_fields(request: dict) -> None:
    unknown = set(request) - REQUEST_FIELDS
    if unknown:
        # grammar and chat_template_kwargs are llama-server features; nothing may vanish silently.
        raise ValueError(f"External LLM requests cannot carry {sorted(unknown)}.")


def _system_and_turns(messages: list[dict]) -> tuple[str, list[dict]]:
    system = "\n\n".join(m["content"] for m in messages if m["role"] == "system")
    return system, [m for m in messages if m["role"] != "system"]


def _effort(request: dict) -> str:
    effort = request.get("reasoning_effort", "none")
    if not isinstance(effort, str) or not effort:
        raise ValueError("reasoning_effort must be a nonempty string.")
    return effort


def openai_request(request: dict, model: str) -> dict:
    _check_fields(request)
    system, turns = _system_and_turns(request["messages"])
    if request.get("response_format") is not None:
        turns = _with_limits(turns, request["response_format"]["json_schema"]["schema"])
    items = []
    for message in turns:
        if message["role"] == "tool":
            items.append({"type": "function_call_output", "call_id": message["tool_call_id"],
                          "output": message["content"]})
            continue
        if message.get("content"):
            items.append({"role": message["role"], "content": message["content"]})
        for call in message.get("tool_calls") or []:
            items.append({"type": "function_call", "call_id": call["id"],
                          "name": call["function"]["name"], "arguments": call["function"]["arguments"]})
    effort = _effort(request)
    body = {"model": model, "input": items, "max_output_tokens": request["max_tokens"],
            "reasoning": {"effort": effort}, "store": False}
    if system:
        body["instructions"] = system
    if effort in SAMPLING_EFFORTS["openai"]:
        for key in ("temperature", "top_p"):
            if key in request:
                body[key] = request[key]
    response_format = request.get("response_format")
    if response_format is not None:
        schema = response_format["json_schema"]
        body["text"] = {"format": {"type": "json_schema",
                                   "name": re.sub(r"[^A-Za-z0-9_-]", "_", schema.get("name", "result"))[:64],
                                   "schema": openai_schema(schema["schema"]), "strict": True}}
    if request.get("tools"):
        body["tools"] = [{"type": "function", "name": tool["function"]["name"],
                          "description": tool["function"].get("description", ""),
                          "parameters": openai_schema(tool["function"]["parameters"]), "strict": True}
                         for tool in request["tools"]]
        body["tool_choice"] = request.get("tool_choice", "auto")
        if "parallel_tool_calls" in request:
            body["parallel_tool_calls"] = request["parallel_tool_calls"]
    return body


def anthropic_request(request: dict, model: str) -> dict:
    _check_fields(request)
    system, turns = _system_and_turns(request["messages"])
    if request.get("response_format") is not None:
        turns = _with_limits(turns, request["response_format"]["json_schema"]["schema"])
    messages = []
    for message in turns:
        if message["role"] == "tool":
            block = {"type": "tool_result", "tool_use_id": message["tool_call_id"],
                     "content": message["content"]}
            if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list):
                messages[-1]["content"].append(block)
            else:
                messages.append({"role": "user", "content": [block]})
            continue
        if message.get("tool_calls"):
            blocks = [{"type": "text", "text": message["content"]}] if message.get("content") else []
            blocks += [{"type": "tool_use", "id": call["id"], "name": call["function"]["name"],
                        "input": json.loads(call["function"]["arguments"])}
                       for call in message["tool_calls"]]
            messages.append({"role": message["role"], "content": blocks})
        else:
            messages.append({"role": message["role"], "content": message["content"]})
    effort = _effort(request)
    # Sampling parameters are rejected on every model with this API version.
    body = {"model": model, "messages": messages, "max_tokens": request["max_tokens"]}
    if system:
        body["system"] = system
    config = {}
    if effort == "none":
        body["thinking"] = {"type": "disabled"}
    else:
        body["thinking"] = {"type": "adaptive"}
        config["effort"] = effort
    response_format = request.get("response_format")
    if response_format is not None:
        config["format"] = {"type": "json_schema",
                            "schema": anthropic_schema(response_format["json_schema"]["schema"])}
    if config:
        body["output_config"] = config
    if request.get("tools"):
        # Strict tool use rejects maxItems; arguments are checked by the tools themselves.
        body["tools"] = [{"name": tool["function"]["name"],
                          "description": tool["function"].get("description", ""),
                          "input_schema": anthropic_schema(tool["function"]["parameters"])}
                         for tool in request["tools"]]
        choice = {"type": "auto"}
        if request.get("parallel_tool_calls") is False:
            choice["disable_parallel_tool_use"] = True
        body["tool_choice"] = choice
    return body


# ----- replies -----

def openai_reply(body: dict) -> dict:
    text, calls, refusals = [], [], []
    for item in body.get("output", []):
        if item.get("type") == "message":
            for part in item.get("content", []):
                if part.get("type") == "output_text":
                    text.append(part.get("text", ""))
                elif part.get("type") == "refusal":
                    refusals.append(part.get("refusal", ""))
        elif item.get("type") == "function_call":
            calls.append({"id": item["call_id"], "type": "function",
                          "function": {"name": item["name"], "arguments": item.get("arguments", "")}})
    if refusals:
        raise ExternalRefusal("OpenAI declined to answer: " + " ".join(refusals)[:500])
    status = body.get("status")
    reason = (body.get("incomplete_details") or {}).get("reason")
    if status == "completed":
        finish = "tool_calls" if calls else "stop"
    elif status == "incomplete" and reason == "max_output_tokens":
        finish = "length"
    else:
        raise RuntimeError(f"OpenAI response ended as {status} ({reason}).")
    usage = body.get("usage") or {}
    message = {"role": "assistant", "content": "".join(text) if text or not calls else None}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": finish}],
            "model": body.get("model"), "provider": "openai",
            "usage": {"prompt_tokens": usage.get("input_tokens"),
                      "completion_tokens": usage.get("output_tokens"),
                      "total_tokens": usage.get("total_tokens"),
                      "prompt_tokens_details": {"cached_tokens": (usage.get("input_tokens_details") or {})
                                                .get("cached_tokens", 0)},
                      "completion_tokens_details": {"reasoning_tokens": (usage.get("output_tokens_details") or {})
                                                    .get("reasoning_tokens", 0)}}}


ANTHROPIC_FINISH = {"end_turn": "stop", "stop_sequence": "stop", "max_tokens": "length",
                    "tool_use": "tool_calls"}


def anthropic_reply(body: dict) -> dict:
    stop = body.get("stop_reason")
    if stop == "refusal":
        details = body.get("stop_details") or {}
        raise ExternalRefusal(f"Anthropic declined to answer ({details.get('category')}).")
    if stop not in ANTHROPIC_FINISH:
        raise RuntimeError(f"Anthropic response ended as {stop}.")
    text = [block.get("text", "") for block in body.get("content", []) if block.get("type") == "text"]
    calls = [{"id": block["id"], "type": "function",
              "function": {"name": block["name"], "arguments": json.dumps(block.get("input", {}),
                                                                          ensure_ascii=False)}}
             for block in body.get("content", []) if block.get("type") == "tool_use"]
    usage = body.get("usage") or {}
    prompt = sum(usage.get(key) or 0 for key in ("input_tokens", "cache_creation_input_tokens",
                                                 "cache_read_input_tokens"))
    message = {"role": "assistant", "content": "".join(text) if text or not calls else None}
    if calls:
        message["tool_calls"] = calls
    return {"choices": [{"index": 0, "message": message, "finish_reason": ANTHROPIC_FINISH[stop]}],
            "model": body.get("model"), "provider": "anthropic",
            "usage": {"prompt_tokens": prompt, "completion_tokens": usage.get("output_tokens"),
                      "total_tokens": prompt + (usage.get("output_tokens") or 0),
                      "prompt_tokens_details": {"cached_tokens": usage.get("cache_read_input_tokens") or 0},
                      "completion_tokens_details": {"reasoning_tokens": (usage.get("output_tokens_details") or {})
                                                    .get("thinking_tokens", 0)}}}


# ----- transport -----

class ExternalLLM:
    """Send one converted request for the LocalLLM that owns the deadline, trace and ordinal."""

    def __init__(self, provider: str, model: str, owner, *, opener=None):
        if provider not in PROVIDERS:
            raise ValueError(f"Unsupported LLM provider: {provider}")
        self.provider, self.model, self.owner = provider, model, owner
        # Unlike the local server, a remote API may sit behind the system proxy.
        self.http = opener or urllib.request.build_opener()

    def identity(self) -> dict:
        return {"provider": self.provider, "api_model": self.model}

    def chat(self, request: dict) -> dict:
        if self.provider == "openai":
            return openai_reply(self._post("/v1/responses", openai_request(request, self.model)))
        return anthropic_reply(self._post("/v1/messages", anthropic_request(request, self.model)))

    def count(self, request: dict) -> int:
        """Input tokens exactly as the provider will bill them, schema and tools included."""
        if self.provider == "openai":
            body = openai_request(request, self.model)
            # The counting endpoint refuses generation-only fields; they add no input tokens.
            for key in ("max_output_tokens", "store", "temperature", "top_p"):
                body.pop(key, None)
            tokens = self._post("/v1/responses/input_tokens", body).get("input_tokens")
        else:
            body = anthropic_request(request, self.model)
            body.pop("max_tokens")
            tokens = self._post("/v1/messages/count_tokens", body).get("input_tokens")
        if type(tokens) is not int or tokens < 0:
            raise TypeError(f"{self.provider} did not return an input token count.")
        return tokens

    def _headers(self) -> dict:
        key = os.environ.get(KEY_VARIABLES[self.provider])
        if not key:
            raise RuntimeError(f"{KEY_VARIABLES[self.provider]} is not set for this worker.")
        if self.provider == "openai":
            return {"Authorization": "Bearer " + key, "Content-Type": "application/json"}
        return {"x-api-key": key, "anthropic-version": ANTHROPIC_VERSION,
                "Content-Type": "application/json"}

    def _post(self, path: str, body: dict) -> dict:
        base = "https://api.openai.com" if self.provider == "openai" else "https://api.anthropic.com"
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        for attempt in range(MAX_RETRIES + 1):
            check_cancelled()
            remaining = self.owner.deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("External LLM generation deadline reached.")
            request = urllib.request.Request(base + path, data=data, headers=self._headers(), method="POST")
            started = time.monotonic()
            status, http_status, retry_after = "error", None, None
            try:
                with self.http.open(request, timeout=min(
                        remaining, self.owner.config["llm"]["request_timeout_seconds"])) as response:
                    http_status = getattr(response, "status", None)
                    result = json.load(response)
                    check_cancelled()
                    status = "ok"
                    return result
            except urllib.error.HTTPError as exc:
                check_cancelled()
                http_status = exc.code
                retry_after = exc.headers.get("retry-after") if exc.headers else None
                detail = exc.read().decode("utf-8", "replace")
                if exc.code not in RETRY_STATUSES or attempt == MAX_RETRIES:
                    raise RuntimeError(f"{self.provider} API HTTP {exc.code}: {_error_message(detail)}") from exc
            except urllib.error.URLError as exc:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(f"{self.provider} API unreachable: {exc.reason}") from exc
            finally:
                self.owner.trace.append({"type": "llm_http", "provider": self.provider, "path": path,
                                         "request": self.owner.requests, "attempt": attempt + 1,
                                         "elapsed_seconds": time.monotonic() - started,
                                         "status": status, "http_status": http_status})
            delay = min(60.0, _seconds(retry_after) if retry_after else 2.0 * 2 ** attempt)
            if time.monotonic() + delay >= self.owner.deadline:
                raise TimeoutError("External LLM generation deadline reached while waiting to retry.")
            time.sleep(delay)
        raise AssertionError("unreachable")


def _seconds(value: str) -> float:
    try:
        return max(0.0, float(value))
    except ValueError:
        return 2.0


def _error_message(detail: str) -> str:
    try:
        error = json.loads(detail).get("error")
    except ValueError:
        return detail[:500]
    if isinstance(error, dict):
        return str(error.get("message") or error)[:1000]
    return str(error or detail)[:1000]
