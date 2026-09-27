"""Lazy, serial model routing under the caller's existing GPU ownership lock."""
from __future__ import annotations

import hashlib
import json
import time
from copy import deepcopy

from .cancellation import check_cancelled
from .context_budget import ContextPolicy, positive_integer
from .llm import LocalLLM, write_json


def _digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def profile_purpose(payload: dict, purpose: str) -> str:
    if payload.get("story_workflow_version") != 2:
        return purpose
    if payload.get("workflow_policy") == "story_draft_v1":
        return {"draft-outline": "draft_outline", "draft-handoff": "draft_handoff",
                "draft-chapter": "draft_writer"}.get(purpose, purpose)
    if payload.get("workflow_policy") == "script_continuation_v1":
        return {"script-cast": "script_planning", "script-outline": "script_planning", "script-plan": "script_planning",
                "script-allocation": "script_planning",
                "script-handoff": "script_handoff", "script-scene": "script_writer",
                "script-speech": "script_speech", "script-staging": "script_staging",
                "speech_separation": "script_speech", "staging": "script_staging",
                "scene_text": "script_writer", "story_outline": "script_planning",
                "story_design": "script_planning", "plot_outline": "script_planning",
                "chapter_plan": "script_planning",
                "supporting_character": "script_planning"}.get(purpose, purpose)
    if payload.get("workflow_policy") == "chapter_editor_v1":
        if purpose in {"initial-canon", "story-blueprint", "revise-blueprint",
                       "chapter-intent", "new-supporting-cast", "scene-sequence"}:
            return "editor_planning"
        if purpose.startswith(("extract-", "information-")):
            return "editor_extraction"
        if purpose.startswith(("adapt-plan-", "scene-text-")) or purpose in {
                "scene_text", "speech_separation", "staging"}:
            return "editor_writer"
        if purpose == "quality_review" or purpose.endswith("-review") or "-review-" in purpose:
            return "editor_review"
    if purpose.startswith("information-"):
        return "information_extraction"
    if purpose.startswith(("extract-", "state-observation-")):
        return "story_extraction"
    if purpose.startswith(("fact-comparison-", "state-comparison-", "scene-continuity-")):
        return "continuity_review"
    if purpose.startswith("adapt-plan-") or purpose == "scene-sequence":
        return "scene_planning"
    if purpose.endswith("-review") or "-review-" in purpose:
        return "continuity_review"
    return {"story-blueprint": "story_blueprint", "revise-blueprint": "story_blueprint",
            "chapter-intent": "chapter_intent"}.get(purpose, purpose)


def route_configs(config: dict) -> dict[str, dict]:
    routes = {"generation": deepcopy(config)}
    review = config.get("model_routing", {}).get("review")
    if review is not None:
        if not isinstance(review, dict) or not isinstance(review.get("llm"), dict) or not review.get("llm_config"):
            raise ValueError("Review model routing needs an llm_config and an llm profile.")
        routes["review"] = {**deepcopy(config), "llm_config": review["llm_config"],
                            "llm": {**config["llm"], **deepcopy(review["llm"])}}
        if routes["review"]["llm"]["model_id"] == config["llm"]["model_id"]:
            raise ValueError("Review and generation routes must have distinct configured model IDs.")
    return routes


def resolve_purpose_profile(config: dict, payload: dict, purpose: str) -> tuple[str, dict]:
    """Pure profile selection; global legacy overrides belong to generation only."""
    category = profile_purpose(payload, purpose)
    routes = route_configs(config)
    draft = (payload.get("story_workflow_version") == 2
             and payload.get("workflow_policy") == "story_draft_v1")
    script = (payload.get("story_workflow_version") == 2
              and payload.get("workflow_policy") == "script_continuation_v1")
    review = (payload.get("story_workflow_version") == 2
              and category in {"continuity_review", "quality_review", "editor_planning",
                               "editor_extraction", "editor_review"} and "review" in routes)
    review = review or (draft and category in {"draft_outline", "draft_handoff"}
                        and "review" in routes)
    if payload.get("workflow_policy") == "chapter_editor_v1" and "review" not in routes:
        raise ValueError("Chapter editor requires the configured Qwen route.")
    if draft and "review" not in routes:
        raise ValueError("Story draft requires the configured Qwen route.")
    settings = routes["review" if review else "generation"]["llm"]
    profile = deepcopy(settings)
    if draft and category in {"draft_outline", "draft_handoff", "draft_writer"}:
        profile["reasoning_level"] = "low" if category == "draft_outline" else "none"
        profile["max_tokens"] = min(4096 if category == "draft_handoff" else 8192,
                                    settings.get("max_output_tokens", 8192))
    if script and category in {"script_planning", "script_handoff", "script_writer",
                               "script_speech", "script_staging"}:
        profile["reasoning_level"] = "none"
        output_budget = {"script-allocation": 2048, "script-plan": 6144, "chapter_plan": 6144}.get(
            purpose, 4096 if category == "script_handoff" else 8192)
        profile["max_tokens"] = min(output_budget,
                                    settings.get("max_output_tokens", 8192))
    if review and payload.get("workflow_policy") == "chapter_editor_v1":
        profile["reasoning_budget_tokens"] = 2048
        if category == "editor_review":
            profile["reasoning_level"] = "none"
            profile["max_tokens"] = min(4096, settings.get("max_output_tokens", 8192))
    if not review:
        defaults = {"information_extraction": 6144, "story_extraction": 4096}
        if payload.get("story_workflow_version") == 2 and category in defaults:
            profile["max_tokens"] = min(defaults[category], settings.get("max_output_tokens", 8192))
        profile.update(payload.get("profile", {}))
    profile.update(payload.get("profiles", {}).get(category, {}))
    profile.update(payload.get("profiles", {}).get(purpose, {}))
    if profile["model_id"] != settings["model_id"]:
        raise ValueError("A task purpose cannot override its configured model route.")
    return category, profile


def model_configuration_identity(root, config: dict) -> dict:
    """Stable registry identity, independent of selected task and loaded process."""
    return {name: {"settings": deepcopy(value["llm"]),
                   "base": json.loads((root / value["llm_config"]).read_text(encoding="utf-8"))
                   if value.get("llm_config") else None}
            for name, value in route_configs(config).items()}


class RoutedLLM(LocalLLM):
    """One cache/ordinal/trace, at most one owned LocalLLM process at a time.

    Selecting a profile, entering the facade, and replaying caches never start a
    model. The facade owns the request stream; backends only own HTTP/processes.
    """

    def __init__(self, root, config, payload, output, *, replay_outputs=()):
        self._route_configs = route_configs(config)
        self._routing_identity = model_configuration_identity(root, config)
        self._route_bases = {name: value["base"] for name, value in self._routing_identity.items()}
        self._routing_hash = _digest(self._routing_identity)
        self._active = None
        self._active_route = None
        self._load_number = 0
        self._entered = False
        self._observed_contexts = {}
        super().__init__(root, config, payload, output, replay_outputs=replay_outputs)
        runtime_path = output / "routing-runtime.json"
        if runtime_path.is_file():
            saved = json.loads(runtime_path.read_text(encoding="utf-8"))
            if saved.get("routing_hash") == self._routing_hash:
                for name, value in saved.get("contexts", {}).items():
                    if name not in self._route_configs:
                        raise ValueError("Cached runtime context refers to an unknown model route.")
                    self._observed_contexts[name] = positive_integer(value, "saved server context")
        self.set_profile(self.profile)
        # Validate every declared purpose without starting either model.
        for purpose in payload.get("profiles", {}):
            _, profile = resolve_purpose_profile(config, payload, purpose)
            self._validate_profile(profile)

    def _route_name(self, profile):
        matches = [name for name, value in self._route_configs.items()
                   if value["llm"]["model_id"] == profile.get("model_id")]
        if len(matches) != 1:
            raise ValueError("Requested model_id is not a configured model route.")
        return matches[0]

    def _profile_settings(self, profile):
        return self._route_configs[self._route_name(profile)]["llm"]

    def select_purpose(self, purpose: str) -> None:
        _, profile = resolve_purpose_profile(self.config, self.payload, purpose)
        self.set_profile(profile)

    def set_profile(self, profile: dict) -> None:
        route = self._route_name(profile if "model_id" in profile else self.profile)
        merged = {**self._route_configs[route]["llm"], **profile}
        self._validate_profile(merged)
        self.profile, self.selected_route = merged, route
        self.base = self._route_bases[route]
        self.server_context_size = self._observed_contexts.get(route)
        self.base_url = self._active.base_url if self._active_route == route else ""

    def _startup_context_size(self):
        route = self._route_name(self.profile)
        settings = self._route_configs[route]["llm"]
        profiles = [self.profile]
        for purpose in self.payload.get("profiles", {}):
            _, profile = resolve_purpose_profile(self.config, self.payload, purpose)
            if self._route_name(profile) == route:
                profiles.append(profile)
        return max(ContextPolicy.resolve(settings, profile).startup_size for profile in profiles)

    def runtime_identity(self):
        settings = self._profile_settings(self.profile)
        policy = ContextPolicy.resolve(settings, self.profile)
        return {"version": 3, "model": self.base["model"], "profile": dict(self.profile),
                "output_limit": self._output_limit(self.profile),
                "context": {**policy.identity(), "server_context_size":
                            self.server_context_size or self._startup_context_size()},
                "routing": {"selected": self._route_name(self.profile),
                            "configuration_sha256": self._routing_hash,
                            "models": self._routing_identity}}

    def model_configuration_identity(self):
        return deepcopy(self._routing_identity)

    def provenance(self):
        profiles = {_digest(row["profile"]): row["profile"] for row in self.trace
                    if row.get("type") == "task_profile" and "profile" in row}
        return {"models": self.model_configuration_identity(),
                "configuration_sha256": self._routing_hash,
                "used_profiles": list(profiles.values()), "requests": self.requests,
                "retry_seed_segments": deepcopy(self.retry_seed_segments)}

    def _chat_request(self, number, messages, extra):
        request, fingerprint = super()._chat_request(number, messages, extra)
        level = self.profile.get("reasoning_level", "none")
        if request["reasoning_effort"] != level:
            raise ValueError("A routed request cannot bypass its selected reasoning profile.")
        kwargs = request.get("chat_template_kwargs", {})
        if kwargs.get("enable_thinking", level != "none") != (level != "none"):
            raise ValueError("Thinking mode conflicts with the selected model profile.")
        return request, fingerprint

    def __enter__(self):
        check_cancelled()
        self.output.mkdir(parents=True, exist_ok=True)
        self.deadline = time.monotonic() + self.config["llm"]["total_timeout_seconds"]
        self._entered = True
        return self

    def _release(self, args):
        active, route = self._active, self._active_route
        if active is None:
            return
        try:
            active.__exit__(*args)
        finally:
            self._active, self._active_route, self.base_url = None, None, ""
            self.trace.append({"type": "model_route_release", "route": route,
                               "model_id": active.profile["model_id"]})

    def __exit__(self, *args):
        try:
            self._release(args)
        finally:
            self._entered = False
            write_json(self.output / "llm-metrics.json", self.trace)

    def _ensure_runtime(self):
        if not self._entered:
            raise RuntimeError("RoutedLLM must own its context before generating uncached output.")
        check_cancelled()
        route = self._route_name(self.profile)
        startup = self._startup_context_size()
        if self._active_route == route and self._active._launch_context_size >= startup:
            self.base_url = self._active.base_url
            return
        previous = self._active_route
        self._release((None, None, None))
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Routed generation deadline reached before model load.")
        settings = deepcopy(self._route_configs[route])
        settings["llm"]["total_timeout_seconds"] = remaining
        backend_payload = {**self.payload, "profile": {**self.profile, "context_size": startup}, "profiles": {}}
        self._load_number += 1
        backend = LocalLLM(self.root, settings, backend_payload,
                           self.output / "runtimes" / f"{self._load_number:03d}-{route}")
        backend.trace, backend.requests = self.trace, self.requests
        self.trace.append({"type": "model_switch", "from_route": previous, "to_route": route,
                           "model_id": self.profile["model_id"], "status": "loading"})
        started = time.monotonic()
        try:
            backend.__enter__()
        except BaseException:
            self.trace.append({"type": "model_switch", "from_route": previous, "to_route": route,
                               "status": "failed", "elapsed_seconds": time.monotonic() - started})
            raise
        self._active, self._active_route = backend, route
        backend.deadline = min(backend.deadline, self.deadline)
        self.base_url, self.server_context_size = backend.base_url, backend.server_context_size
        self._observed_contexts[route] = self.server_context_size
        write_json(self.output / "routing-runtime.json", {"routing_hash": self._routing_hash,
                                                        "contexts": self._observed_contexts})
        self.trace.append({"type": "model_switch", "from_route": previous, "to_route": route,
                           "status": "ready", "elapsed_seconds": time.monotonic() - started,
                           "server_context_size": self.server_context_size})

    def request(self, path, value=None, timeout=None):
        if self._active is None or self._active_route != self._route_name(self.profile):
            raise RuntimeError("No active server for the selected model route.")
        self._active.requests = self.requests
        return self._active.request(path, value, timeout)

    def chat(self, stage, messages, *, allow_truncated=False, **extra):
        # CausalRun checks this same cache before charging a real request.
        if not self.has_cached_chat(stage, messages, **extra):
            self._ensure_runtime()
        return super().chat(stage, messages, allow_truncated=allow_truncated, **extra)
