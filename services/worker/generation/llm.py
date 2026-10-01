"""Ephemeral pinned Gemma server, structured output and persisted request cache."""
from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path

from ..model_config import model_base, runtime_paths
from .cancellation import check_cancelled
from .context_budget import ContextPolicy, OutputTokenPolicy, positive_integer, server_context_from_properties
from .processes import owned_process
from .random_tools import TOOL_DEFINITIONS, RandomTools
from .schemas import validate_schema


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class ContextBudgetError(RuntimeError):
    """Keep source intact when the model cannot fit input and a complete answer."""


class LocalLLM:
    def __init__(self, root: Path, config: dict, payload: dict, output: Path, *,
                 replay_outputs: tuple[Path, ...] = ()):
        self.root, self.config, self.payload, self.output = root, config, payload, output
        self.base = model_base(root, config)
        self.profile = {**config["llm"], **payload.get("profile", {})}
        self._validate_profile(self.profile)
        self.http = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.base_url = ""
        self._owner = None
        self.process = None
        self.server_context_size: int | None = None
        self._launch_context_size: int | None = None
        self.context_source = "configured"
        self.requests = 0
        self.replay_outputs = replay_outputs
        retry_path = next((directory / "retry-state.json" for directory in (output, *replay_outputs)
                           if (directory / "retry-state.json").is_file()), output / "retry-state.json")
        self.retry_seed_segments = (
            json.loads(retry_path.read_text(encoding="utf-8"))["segments"]
            if retry_path.exists() else []
        )
        self.tools = RandomTools(payload["seed"], config["llm"]["max_tool_calls"])
        self.trace: list[dict] = []

    def _validate_profile(self, profile: dict) -> None:
        import math
        if not math.isfinite(profile.get("top_p", 0.95)) or not 0 < profile.get("top_p", 0.95) <= 1:
            raise ValueError("top_p must be finite and greater than zero, at most one.")
        settings = self._profile_settings(profile)
        if profile.get("provider", "local") != "local":
            raise ValueError("M2 supports the configured local provider only.")
        if profile["model_id"] != settings["model_id"]:
            raise ValueError("Requested model_id is not the configured pinned local model.")
        effort = profile.get("reasoning_level", "none")
        if not isinstance(effort, str):
            raise ValueError("reasoning_level must be a string.")
        if not profile.get("common_settings_version"):
            allowed = settings.get("reasoning_levels", ["none"])
            if effort not in allowed:
                raise ValueError("Unsupported reasoning_level for this configured local model.")
            if effort != "none" and settings.get("reasoning_template") != "qwen3":
                raise ValueError("Thinking requires an explicitly verified reasoning template.")
        budget = profile.get("reasoning_budget_tokens")
        if budget is not None and (settings.get("reasoning_template") != "qwen3"
                                   or type(budget) is not int or not 0 <= budget <= 8192):
            raise ValueError("Reasoning budget requires a supported Qwen template and a bounded token count.")
        output_limit = OutputTokenPolicy.resolve(settings, profile).base_limit
        if (not 0 <= profile["temperature"] <= 2
                or type(profile["max_tokens"]) is not int
                or not 256 <= profile["max_tokens"] <= output_limit):
            raise ValueError("Invalid local generation profile.")
        if "context_size" in settings:
            ContextPolicy.resolve(settings, profile)

    def _profile_settings(self, profile: dict) -> dict:
        return self.config["llm"]

    def _output_limit(self, profile: dict) -> int:
        return OutputTokenPolicy.resolve(self._profile_settings(profile), profile).limit

    def generation_tokens(self) -> int:
        """Effective total generation cap; profile.max_tokens stays the baseline."""
        return OutputTokenPolicy.resolve(self._profile_settings(self.profile), self.profile).tokens(
            self.profile["max_tokens"])

    def set_profile(self, profile: dict) -> None:
        """Select a stage profile without pretending to resize a running model."""
        merged = {**self.config["llm"], **profile}
        self._validate_profile(merged)
        if self._launch_context_size is not None:
            policy = ContextPolicy.resolve(self.config["llm"], merged)
            if policy.target > self._launch_context_size:
                raise ValueError("Task context_size exceeds the running server; start a new job "
                                 "with this profile before loading the model.")
        self.profile = merged

    def _startup_context_size(self) -> int:
        profiles = [self.profile]
        profiles.extend({**self.config["llm"], **self.payload.get("profile", {}), **overrides}
                        for overrides in self.payload.get("profiles", {}).values())
        for profile in profiles:
            self._validate_profile(profile)
        return max(ContextPolicy.resolve(self.config["llm"], profile).startup_size
                   for profile in profiles)

    def runtime_identity(self) -> dict:
        """Versioned cache identity, including the actual loaded context and model."""
        context = None
        if "context_size" in self.config["llm"]:
            policy = ContextPolicy.resolve(self.config["llm"], self.profile)
            context = {**policy.identity(), "server_context_size":
                       self.server_context_size or self._startup_context_size()}
        model = dict(self.base["model"])
        if "llm_base" in self.config:
            model.pop("relative_path", None)
        return {"version": 2, "model": model, "context": context,
                "profile": dict(self.profile), "output_limit": self._output_limit(self.profile)}

    def context_budget(self, *, prompt_tokens: int = 0, output_tokens: int | None = None) -> dict:
        """Share the same budget with evidence retrieval and scene splitting."""
        output = self.generation_tokens() if output_tokens is None else output_tokens
        if positive_integer(output, "output_tokens") > self._output_limit(self.profile):
            raise ValueError("Output tokens exceed the configured output allowance.")
        policy = ContextPolicy.resolve(self._profile_settings(self.profile), self.profile)
        return policy.budget(prompt_tokens, output,
                             self.server_context_size or self._startup_context_size())

    def retry_failed_from(self, request_number: int) -> None:
        """Retain successful prefix caches; resample only the definitively failed tail."""
        if type(request_number) is not int or not 1 <= request_number <= self.requests:
            raise ValueError("Invalid failed request checkpoint.")
        self.retry_seed_segments.append({"from_request": request_number,
                                        "salt": (len(self.retry_seed_segments) + 1) * 104729})
        write_json(self.output / "retry-state.json", {"segments": self.retry_seed_segments})

    def __enter__(self):
        check_cancelled()
        self.output.mkdir(parents=True, exist_ok=True)
        model, server = runtime_paths(self.root, self.config, self.base)
        if model.stat().st_size != self.base["model"]["size_bytes"]:
            raise ValueError("Pinned local model size mismatch.")
        with model.open("rb") as stream:
            if stream.read(4) != b"GGUF":
                raise ValueError("Local model has no GGUF header.")
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        self.base_url = f"http://127.0.0.1:{port}"
        startup_context = self._startup_context_size()
        self._launch_context_size = startup_context
        command = [str(server), "--model", str(model), "--host", "127.0.0.1", "--port", str(port),
            "--ctx-size", str(startup_context), "--n-gpu-layers", "all",
            "--parallel", "1", "--jinja", "--reasoning-format", "deepseek", "--alias", "m2-local"]
        reasoning_budget = self.profile.get("reasoning_budget_tokens")
        if reasoning_budget is not None:
            command.extend(("--reasoning-budget", str(reasoning_budget)))
        self.deadline = time.monotonic() + self.config["llm"]["total_timeout_seconds"]
        self._owner = owned_process(command, self.output / "llama-server.log", cwd=server.parent,
                                    timeout=self.config["llm"]["total_timeout_seconds"])
        load_started = time.monotonic()
        try:
            self.process = self._owner.__enter__()
            health_deadline = time.monotonic() + self.config["llm"]["startup_timeout_seconds"]
            while time.monotonic() < health_deadline:
                check_cancelled()
                if self.process.poll() is not None:
                    raise RuntimeError("Local Gemma server exited during startup; see llama-server.log.")
                try:
                    if self.request("/health", timeout=2).get("status") == "ok":
                        break
                except (OSError, urllib.error.URLError, RuntimeError, TimeoutError):
                    pass
                time.sleep(0.5)
            else:
                raise TimeoutError("Local Gemma server did not become ready.")
            self.trace.append({"type": "model_load", "elapsed_seconds":
                               time.monotonic() - load_started, "status": "ready",
                               "requested_context_size": startup_context})
            self._read_server_context(startup_context)
            self.trace.append({"type": "llm_runtime", **self.runtime_identity(),
                               "context_source": self.context_source})
            return self
        except BaseException:
            self.trace.append({"type": "model_startup_failed", "elapsed_seconds":
                               time.monotonic() - load_started})
            if self.process is not None:
                self._owner.__exit__(*__import__("sys").exc_info())
            write_json(self.output / "llm-metrics.json", self.trace)
            raise

    def __exit__(self, *args):
        started = time.monotonic()
        try:
            return self._owner.__exit__(*args)
        finally:
            self.trace.append({"type": "model_release", "elapsed_seconds":
                               time.monotonic() - started})
            write_json(self.output / "llm-metrics.json", self.trace)

    def _read_server_context(self, configured_context: int) -> None:
        self.server_context_size = configured_context
        self.context_source = "verified_launch_setting"
        try:
            properties = self.request("/props", timeout=5)
            actual = server_context_from_properties(properties)
        except RuntimeError as exc:
            # Older pinned servers may not expose /props. Only unsupported routes
            # fall back to our own launch flag; a broken server remains an error.
            if not any(f"HTTP {code}:" in str(exc) for code in (404, 405, 501)):
                raise
            if self.base["model"].get("chat_template_sha256"):
                raise ValueError("This reasoning model requires verified server template properties.") from exc
            actual = None
        else:
            expected = self.base["model"].get("chat_template_sha256")
            if expected:
                template = properties.get("chat_template")
                if not isinstance(template, str) or hashlib.sha256(template.encode()).hexdigest() != expected:
                    raise ValueError("Loaded model chat template differs from its pinned configuration.")
                build = self.base.get("server", {}).get("build_info")
                if build and properties.get("build_info") != build:
                    raise ValueError("Loaded server build differs from the verified reasoning runtime.")
                if (self._profile_settings(self.profile).get("reasoning_template") == "qwen3"
                        and not properties.get("chat_template_caps", {}).get("supports_reasoning_effort")):
                    raise ValueError("Loaded template does not support native reasoning effort.")
                self.trace.append({"type": "model_template", "model_id": self.profile["model_id"],
                                   "sha256": expected, "build_info": properties.get("build_info")})
        if actual is not None:
            self.server_context_size = actual
            self.context_source = "server_properties"
        self.trace.append({"type": "server_context", "requested": configured_context,
                           "actual": self.server_context_size, "source": self.context_source})

    def request(self, path: str, value: dict | None = None, timeout: float | None = None) -> dict:
        check_cancelled()
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Local Gemma generation deadline reached.")
        request = urllib.request.Request(self.base_url + path,
            data=None if value is None else json.dumps(value).encode("utf-8"),
            headers={"Content-Type": "application/json"})
        started = time.monotonic()
        status = "error"
        http_status = None
        try:
            with self.http.open(request, timeout=min(remaining, timeout or
                                self.config["llm"]["request_timeout_seconds"])) as response:
                http_status = getattr(response, "status", None)
                result = json.load(response)
                check_cancelled()
                status = "ok"
                return result
        except urllib.error.HTTPError as exc:
            check_cancelled()
            http_status = exc.code
            detail = exc.read().decode("utf-8", "replace")[:2000]
            raise RuntimeError(f"Local LLM HTTP {exc.code}: {detail}") from exc
        finally:
            self.trace.append({"type": "llm_http", "path": path, "request": self.requests,
                               "elapsed_seconds": time.monotonic() - started,
                               "status": status, "http_status": http_status})

    def _chat_request(self, number: int, messages: list[dict], extra: dict) -> tuple[dict, str]:
        self._validate_profile(self.profile)
        salt = 0
        for segment in self.retry_seed_segments:
            if number >= segment["from_request"]:
                salt = segment["salt"]
        seed = (self.payload["seed"] + number + salt) % (2**31)
        request = {"model": "m2-local", "messages": messages, "stream": False,
            "seed": seed, "temperature": self.profile["temperature"],
            "max_tokens": self.generation_tokens(), "top_p": self.profile.get("top_p", 0.95),
            "reasoning_effort": self.profile.get("reasoning_level", "none")}
        if self._profile_settings(self.profile).get("reasoning_template") == "qwen3":
            request["chat_template_kwargs"] = {
                "enable_thinking": self.profile.get("reasoning_level", "none") != "none"}
        # Explicit limits, e.g. durable scene budgets, already include expansion.
        request.update(extra)
        output = positive_integer(request["max_tokens"], "request.max_tokens")
        if output > self._output_limit(self.profile):
            raise ValueError("Request max_tokens exceeds the configured output allowance.")
        fingerprint = hashlib.sha256(json.dumps({"request": request,
            "runtime": self.runtime_identity()}, ensure_ascii=False,
            sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        return request, fingerprint

    def cached_chat(self, stage: str, messages: list[dict], **extra) -> dict | None:
        """Replay an exact older request if saved; a cache miss consumes no ordinal or HTTP call."""
        if not self.has_cached_chat(stage, messages, **extra):
            return None
        return self.chat(stage, messages, **extra)

    def has_cached_chat(self, stage: str, messages: list[dict], **extra) -> bool:
        """Let durable budgets distinguish replay from a fresh generation attempt."""
        extra.pop("allow_truncated", None)
        _, fingerprint = self._chat_request(self.requests + 1, messages, extra)
        return self._cache_path(self.requests + 1, stage, fingerprint) is not None

    def _cache_path(self, number: int, stage: str, fingerprint: str) -> Path | None:
        name = f"{number:02d}-{stage}-{fingerprint[:16]}.json"
        return next((directory / name for directory in (self.output, *self.replay_outputs)
                     if (directory / name).is_file()), None)

    def check_context(self, stage: str, request: dict) -> dict:
        """Use the loaded model's tokenizer and exact chat template, including schema/tools."""
        formatted = self.request("/apply-template", request)
        if not isinstance(formatted.get("prompt"), str):
            raise TypeError("Local LLM did not return its formatted prompt for token counting.")
        encoded = self.request("/tokenize", {"content": formatted["prompt"],
                                            "add_special": True, "parse_special": True})
        if not isinstance(encoded.get("tokens"), list):
            raise TypeError("Local LLM did not return prompt tokens.")
        prompt_tokens = len(encoded["tokens"])
        budget = {"type": "context_budget", "stage": stage, "request": self.requests,
                  **self.context_budget(prompt_tokens=prompt_tokens,
                                        output_tokens=request["max_tokens"])}
        self.trace.append(budget)
        if not budget["fits"]:
            write_json(self.output / "last-context-budget.json", budget)
            error = ContextBudgetError(
                f"{stage}: 入力{prompt_tokens} + 回答枠{budget['output_tokens']} + "
                f"余裕{budget['margin_tokens']} tokensが"
                f"コンテキスト{budget['context_size']} tokensを超えます。"
                "本文・知識と回答枠は切り捨てていません。工程の入力分割、"
                "または許可・検証された拡張設定での新しい実行が必要です。")
            error.budget = budget
            raise error
        return budget

    def chat(self, stage: str, messages: list[dict], *, allow_truncated: bool = False, **extra) -> dict:
        check_cancelled()
        self.requests += 1
        request, fingerprint = self._chat_request(self.requests, messages, extra)
        cache = self._cache_path(self.requests, stage, fingerprint)
        cache_hit = cache is not None
        started = time.monotonic()
        generation_elapsed = 0.0
        budget = None
        if cache is not None:
            saved = json.loads(cache.read_text(encoding="utf-8"))
            if saved.get("request_sha256") != fingerprint:
                raise ValueError("LLM response cache fingerprint mismatch.")
            response = saved["response"]
        else:
            if self.base_url:
                budget = self.check_context(stage, request)
            generation_started = time.monotonic()
            try:
                response = self.request("/v1/chat/completions", request)
            except BaseException:
                self.trace.append({"type": "llm_generation", "stage": stage,
                                   "request": self.requests, "cache_hit": False,
                                   "status": "error", "elapsed_seconds":
                                   time.monotonic() - generation_started})
                raise
            generation_elapsed = time.monotonic() - generation_started
            check_cancelled()
            cache = self.output / f"{self.requests:02d}-{stage}-{fingerprint[:16]}.json"
            write_json(cache, {"request_sha256": fingerprint, "request": request,
                               "runtime": self.runtime_identity(), "response": response,
                               "context_budget": budget,
                               "generation_elapsed_seconds": generation_elapsed})
        choices = response.get("choices", [])
        usage = response.get("usage", {})
        timings = response.get("timings", {})
        metric = {"type": "llm_generation", "stage": stage, "request": self.requests,
                  "cache_hit": cache_hit, "status": "response",
                  "elapsed_seconds": time.monotonic() - started,
                  "generation_http_seconds": generation_elapsed,
                  "usage": usage if isinstance(usage, dict) else {},
                  "generated_tokens_this_run": 0 if cache_hit else
                  usage.get("completion_tokens") if isinstance(usage, dict) else None,
                  "finish_reason": choices[0].get("finish_reason") if len(choices) == 1 else None}
        # llama.cpp timings separate prompt evaluation and token generation from
        # HTTP/load overhead. A missing measurement stays unknown, never zero.
        if isinstance(timings, dict):
            metric["server_timings"] = {key: value for key, value in timings.items()
                                        if type(value) in (int, float)}
            metric["server_timings_from_cache"] = cache_hit
        self.trace.append(metric)
        check_cancelled()
        accepted = ("stop", "tool_calls", "length") if allow_truncated else ("stop", "tool_calls")
        if len(choices) != 1 or choices[0].get("finish_reason") not in accepted:
            raise ValueError(f"{stage}: LLM output incomplete or truncated.")
        message = choices[0].get("message")
        if not isinstance(message, dict):
            raise TypeError(f"{stage}: Missing LLM message.")
        if allow_truncated:
            return {**message, "_finish_reason": choices[0]["finish_reason"]}
        return message

    def structured(self, stage: str, messages: list[dict], schema: dict) -> dict:
        message = self.chat(stage, messages, response_format={"type": "json_schema",
            "json_schema": {"name": "m2_result", "strict": True, "schema": schema}})
        if message.get("tool_calls"):
            raise ValueError(f"{stage}: Unexpected tool call in structured result.")
        try:
            result = json.loads(message.get("content") or "")
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{stage}: Invalid JSON output.") from exc
        validate_schema(result, schema)
        return result

    def random_context(self, stage: str, messages: list[dict]) -> list[dict]:
        """No schema grammar here: let Gemma choose whether it needs a tool."""
        history = list(messages)
        for _ in range(self.config["llm"]["max_tool_calls"] + 1):
            message = self.chat(stage, history, tools=TOOL_DEFINITIONS, tool_choice="auto",
                                parallel_tool_calls=False)
            calls = message.get("tool_calls") or []
            if not calls:
                if not isinstance(message.get("content"), str) or not message["content"].strip():
                    raise ValueError("Empty candidate selection response.")
                history.append({"role": "assistant", "content": message["content"]})
                return history
            history.append({"role": "assistant", "content": message.get("content"),
                            "tool_calls": calls})
            for call in calls:
                if not isinstance(call.get("id"), str) or not call["id"]:
                    raise ValueError("Missing random tool call ID.")
                function = call.get("function", {})
                arguments = json.loads(function.get("arguments", ""))
                result = self.tools.execute(function.get("name"), arguments)
                self.trace.append({**self.tools.trace[-1], "stage": stage})
                history.append({"role": "tool", "tool_call_id": call["id"],
                                "content": json.dumps(result, ensure_ascii=False)})
        raise ValueError("Random tool interaction did not finish within the limit.")
