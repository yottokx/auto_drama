"""Durable, input-addressed stages and a shared repair budget for the PoC writer."""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from pathlib import Path

from pydantic import BaseModel

from .cancellation import check_cancelled
from .llm import write_json
from .workflow_version import generator_protocol


def encoded(value) -> str:
    if isinstance(value, BaseModel):
        value = value.model_dump(mode="json")
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def digest(value) -> str:
    return hashlib.sha256(encoded(value).encode("utf-8")).hexdigest()


class RepairLimitError(RuntimeError):
    pass


class ResourceBudgetError(RepairLimitError):
    pass


_active_attempt = ContextVar("causal_resource_attempt", default=None)


def resource_limits(payload):
    limits = payload.get("workflow_limits", {})
    if not isinstance(limits, dict):
        raise TypeError("workflow_limits must be an object.")
    allowed = {"max_calls", "max_repairs", "max_tokens", "max_elapsed_seconds",
               "max_story_tokens", "max_story_elapsed_seconds"}
    if limits.keys() - allowed:
        raise ValueError("Unknown workflow limits: " + ", ".join(sorted(limits.keys() - allowed)))
    CausalRun._limit(limits.get("max_calls", 160), "max_calls", 1, 1000)
    default_repairs = 2 if payload.get("workflow_policy") == "chapter_editor_v1" else 6
    CausalRun._limit(limits.get("max_repairs", default_repairs), "max_repairs", 1, 30)
    world = payload.get("approval_snapshot", {}).get("world", {})
    count = world.get("result", world).get("chapterCount", 1)
    if type(count) is not int or not 1 <= count <= 100:
        raise ValueError("Invalid approved chapter count for workflow budget.")
    defaults = {"max_tokens": 4_000_000, "max_elapsed_seconds": 14_400}
    values = {}
    for key, default in defaults.items():
        values[key] = CausalRun._limit(limits.get(key, default), key, 1, 10**10)
    for key in defaults:
        story_key = key.replace("max_", "max_story_", 1)
        values[story_key] = CausalRun._limit(limits.get(story_key, values[key] * count), story_key, 1, 10**12)
    return values


class PersistentUsage:
    """One atomic story journal; chapter views never reset on a new job/restart.

    The existing GPU/output ownership lock serializes writers. Job wall time
    contains request time, so live and completed job attempts replace (never add
    to) the enclosed request durations. Time between runner invocations is not charged.
    """

    def __init__(self, work: Path | None, payload: dict, config: dict, *, previous_executions=()):
        from .execution_settings import enabled, source_config

        self.limits = resource_limits(payload)
        self.chapter = str(payload.get("chapter_number", 1))
        identity = {"protocol": generator_protocol("causal", payload.get("workflow_policy")),
            "storyline": payload.get("storyline_id"), "approval": payload.get("approval_snapshot"),
            "profile": payload.get("profile"), "profiles": payload.get("profiles"),
            "config": config, "limits": self.limits}
        if enabled(payload):
            identity.pop("profile")
            identity.pop("profiles")
            identity["config"] = source_config(config)
        self.identity = digest(identity)
        self.previous_identities = set()
        for execution in previous_executions:
            old_payload, old_config = execution["payload"], execution["generation_config"]
            old_identity = {"protocol": generator_protocol("causal", old_payload.get("workflow_policy")),
                "storyline": old_payload.get("storyline_id"), "approval": old_payload.get("approval_snapshot"),
                "profile": old_payload.get("profile"), "profiles": old_payload.get("profiles"),
                "config": old_config, "limits": resource_limits(old_payload)}
            self.previous_identities.add(digest(old_identity))
        self.path = None
        if work is not None:
            work = Path(work)
            self.path = (work.parent / "causal-resource-budgets" / (digest(payload["storyline_id"]) + ".json")
                         if payload.get("storyline_id") else work / "causal-stages" / "resources.json")
        self.data = {"schema_version": 1, "identity": self.identity, "limits": self.limits,
                     "requests": [], "attempts": []}
        self.reload()

    def reload(self):
        if self.path is not None and self.path.exists():
            value = json.loads(self.path.read_text(encoding="utf-8"))
            if (value.get("schema_version") != 1
                    or value.get("identity") not in {self.identity, *self.previous_identities}
                    or value.get("limits") != self.limits
                    or not isinstance(value.get("requests"), list)
                    or not isinstance(value.get("attempts"), list)):
                raise ValueError("Invalid or incompatible persistent workflow resource budget.")
            for group in ("requests", "attempts"):
                ids = []
                for item in value[group]:
                    if (not isinstance(item, dict) or not isinstance(item.get("id"), str)
                            or not isinstance(item.get("chapter"), str)
                            or not re.fullmatch(r"(?:[1-9][0-9]?|100)", item["chapter"])
                            or item.get("status") not in {"pending", "measured", "unmeasured"}):
                        raise ValueError("Invalid persistent resource measurement.")
                    ids.append(item["id"])
                    elapsed = item.get("elapsed_seconds")
                    if elapsed is not None and (type(elapsed) not in (int, float)
                            or not math.isfinite(elapsed) or elapsed < 0):
                        raise ValueError("Invalid persistent elapsed measurement.")
                    if group == "requests":
                        for key in ("prompt_tokens", "completion_tokens"):
                            count = item.get(key)
                            if count is not None and (type(count) is not int or count < 0):
                                raise ValueError("Invalid persistent token measurement.")
                        if item["status"] == "measured" and any(item.get(key) is None for key in
                                ("prompt_tokens", "completion_tokens", "elapsed_seconds")):
                            raise ValueError("Measured request is missing usage.")
                    elif item["status"] == "measured" and elapsed is None:
                        raise ValueError("Measured attempt is missing elapsed time.")
                if len(ids) != len(set(ids)):
                    raise ValueError("Duplicate persistent resource measurement ID.")
            attempts = {item["id"]: item for item in value["attempts"]}
            if any(r.get("attempt_id") is not None and (
                    r["attempt_id"] not in attempts or attempts[r["attempt_id"]]["chapter"] != r["chapter"])
                    for r in value["requests"]):
                raise ValueError("Persistent request refers to a different or unknown chapter attempt.")
            old_identity = value["identity"]
            if old_identity != self.identity:
                value.setdefault("execution_identity_history", []).append(old_identity)
                value["identity"] = self.identity
            self.data = value
            if old_identity != self.identity:
                self.save()

    def save(self):
        if self.path is not None:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            write_json(self.path, self.data)

    def totals(self, chapter=None):
        requests = [r for r in self.data["requests"] if chapter is None or r["chapter"] == chapter]
        attempts = [a for a in self.data["attempts"] if chapter is None or a["chapter"] == chapter]
        completed = {a["id"] for a in attempts if a["status"] == "measured"}
        active = _active_attempt.get()
        live = (active if active and active[0] == self.identity
                and any(a["id"] == active[1] and a["status"] == "pending" for a in attempts)
                else None)
        # Count model switches and other work between requests immediately. The
        # same context is visible to each chapter journal view in this process;
        # a stopped process's pending attempt still has unknown elapsed time.
        covered = completed | ({live[1]} if live else set())
        live_elapsed = max(0.0, time.monotonic() - live[2]) if live else 0.0
        return {"prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in requests),
                "completion_tokens": sum(r.get("completion_tokens") or 0 for r in requests),
                "elapsed_seconds": sum(a["elapsed_seconds"] for a in attempts if a["id"] in completed)
                    + live_elapsed
                    + sum(r.get("elapsed_seconds") or 0 for r in requests if r.get("attempt_id") not in covered),
                "unmeasured_requests": [r["id"] for r in requests if r["status"] != "measured"],
                "unfinished_attempts": [a["id"] for a in attempts if a["status"] != "measured"]}

    def summary(self):
        self.reload()
        return {"limits": self.limits, "chapter": self.totals(self.chapter), "story": self.totals()}

    def check(self, *, new_request=False):
        self.reload()
        active = _active_attempt.get()
        active = active[1] if active and active[0] == self.identity else None
        for scope, totals, prefix in (("chapter", self.totals(self.chapter), "max_"),
                                       ("story", self.totals(), "max_story_")):
            unknown_attempts = set(totals["unfinished_attempts"]) - ({active} if active else set())
            if totals["unmeasured_requests"] or unknown_attempts:
                raise ResourceBudgetError(f"Causal {scope} resource usage is unmeasured; resume cannot assume zero.")
            tokens = totals["prompt_tokens"] + totals["completion_tokens"]
            for used, key in ((tokens, prefix + "tokens"),
                              (totals["elapsed_seconds"], prefix + "elapsed_seconds")):
                if used > self.limits[key] or (new_request and used >= self.limits[key]):
                    raise ResourceBudgetError(f"Causal {scope} persistent {key} budget exhausted ({used}).")

    def begin_request(self):
        self.check(new_request=True)
        key = f"request-{len(self.data['requests']) + 1}"
        active = _active_attempt.get()
        self.data["requests"].append({"id": key, "chapter": self.chapter, "status": "pending",
            "attempt_id": active[1] if active and active[0] == self.identity else None, "prompt_tokens": None,
            "completion_tokens": None, "elapsed_seconds": None})
        self.save()
        return key

    def finish_request(self, key, trace, elapsed, error):
        self.reload()
        item = next(r for r in self.data["requests"] if r["id"] == key)
        metrics = [entry for entry in trace if entry.get("type") == "llm_generation"
                   and entry.get("cache_hit") is not True]
        usage = next((entry.get("usage") for entry in reversed(metrics)
                      if isinstance(entry.get("usage"), dict)
                      and all(type(entry["usage"].get(name)) is int and entry["usage"][name] >= 0
                              for name in ("prompt_tokens", "completion_tokens"))), None)
        if usage is not None:
            item.update(prompt_tokens=usage["prompt_tokens"], completion_tokens=usage["completion_tokens"])
        else:
            # A failed context check never dispatched generation. Its tokenizer
            # measurements describe input size, not consumed inference tokens.
            budgets = [entry for entry in trace if entry.get("type") == "context_budget"]
            if error is not None and not metrics and budgets and budgets[-1].get("fits") is False:
                item.update(prompt_tokens=0, completion_tokens=0)
                item["generation_not_dispatched"] = True
            elif budgets:
                item["prompt_tokens"] = budgets[-1].get("prompt_tokens")
        item.update(elapsed_seconds=elapsed, error=error,
                    status="measured" if item["prompt_tokens"] is not None
                    and item["completion_tokens"] is not None else "unmeasured")
        self.save()

    @contextmanager
    def job_attempt(self):
        self.check()
        key = f"attempt-{len(self.data['attempts']) + 1}"
        self.data["attempts"].append({"id": key, "chapter": self.chapter, "status": "pending",
                                      "elapsed_seconds": None})
        self.save()
        started = time.monotonic()
        token = _active_attempt.set((self.identity, key, started))
        try:
            yield
        finally:
            elapsed = time.monotonic() - started
            self.reload()
            item = next(a for a in self.data["attempts"] if a["id"] == key)
            item.update(status="measured", elapsed_seconds=elapsed)
            self.save()
            _active_attempt.reset(token)


class CausalRun:
    """Successful stages survive restarts; moving to a new stage cannot reset budgets."""

    def __init__(self, llm, payload):
        from .execution_settings import enabled, previous_executions, source_config, source_payload

        self.llm = llm
        self.payload = payload
        identity = {
            "protocol": generator_protocol("causal", payload.get("workflow_policy")), "payload": payload,
            "config": llm.config, "model": getattr(llm, "base", {}),
            "model_registry": (llm.model_configuration_identity()
                               if callable(getattr(llm, "model_configuration_identity", None)) else None),
        }
        self.live_settings = enabled(payload)
        if self.live_settings:
            identity = {"protocol": identity["protocol"], "payload": source_payload(payload),
                        "config": source_config(llm.config)}
        self.signature = digest(identity)
        output = getattr(llm, "output", None)
        previous = previous_executions(Path(output).parent if output is not None else None,
                                       payload, llm.config)
        self.previous_signatures = set()
        for execution in previous:
            old_config, old_payload = execution["generation_config"], execution["payload"]
            registry = old_config.get("model_configuration_identity")
            base = old_config.get("llm_base") or (registry or {}).get("generation", {}).get("base", {})
            self.previous_signatures.add(digest({
                "protocol": generator_protocol("causal", old_payload.get("workflow_policy")),
                "payload": old_payload, "config": old_config, "model": base,
                "model_registry": registry}))
        self.directory = Path(output).parent / "causal-stages" if output is not None else None
        self.state = {"signature": self.signature, "repairs": 0, "calls": 0, "issues": []}
        self.cache = {}
        self.repair_visits = {}
        self.active_nodes = []
        self.last_chat_identity = None
        self.original_chat = llm.chat
        if self.directory:
            self.directory.mkdir(parents=True, exist_ok=True)
            state_path = self.directory / "budget.json"
            if state_path.exists():
                state = json.loads(state_path.read_text(encoding="utf-8"))
                if state.get("signature") not in {self.signature, *self.previous_signatures}:
                    raise ValueError("Causal stage directory belongs to different inputs/profile.")
                if (type(state.get("calls")) is not int or state["calls"] < 0
                        or type(state.get("repairs")) is not int or state["repairs"] < 0
                        or not isinstance(state.get("issues"), list)
                        or len(state["issues"]) != state["repairs"]
                        or any(not isinstance(issue, dict) or not isinstance(issue.get("key"), str)
                               for issue in state["issues"])):
                    raise ValueError("Invalid persisted causal budget.")
                old_signature = state["signature"]
                if old_signature != self.signature:
                    state.setdefault("execution_identity_history", []).append(old_signature)
                    state["signature"] = self.signature
                self.state = state
        limits = payload.get("workflow_limits", {})
        default_repairs = 2 if payload.get("workflow_policy") == "chapter_editor_v1" else 6
        self.max_repairs = self._limit(limits.get("max_repairs", default_repairs), "max_repairs", 1, 30)
        self.max_calls = self._limit(limits.get("max_calls", 160), "max_calls", 1, 1000)
        self.usage = PersistentUsage(Path(output).parent if output is not None else None,
                                     payload, llm.config, previous_executions=previous)
        if self.state["calls"] and not any(r["chapter"] == self.usage.chapter for r in self.usage.data["requests"]):
            raise ValueError("Existing causal budget has no resource measurements; use a new protocol experiment.")

    @staticmethod
    def _limit(value, name, lower, upper):
        if type(value) is not int or not lower <= value <= upper:
            raise ValueError(f"Invalid workflow {name}.")
        return value

    def _save(self):
        if self.directory:
            write_json(self.directory / "budget.json", self.state)

    def __enter__(self):
        check_cancelled()
        self.usage.check()
        self._save()
        self.llm.chat = self.chat
        return self

    def __exit__(self, *args):
        self.llm.chat = self.original_chat
        self._save()

    def chat(self, *args, **kwargs):
        check_cancelled()
        probe = getattr(self.llm, "has_cached_chat", None)
        cached = callable(probe) and probe(*args, **kwargs)
        key = None
        if not cached:
            if self.state["calls"] >= self.max_calls:
                raise RepairLimitError("Causal chapter exhausted its persistent LLM call budget.")
            key = self.usage.begin_request()
            self.state["calls"] += 1
            self._save()
        start, trace_start, error = time.monotonic(), len(self.llm.trace), None
        try:
            response = self.original_chat(*args, **kwargs)
        except BaseException as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            if key is not None:
                self.usage.finish_request(key, self.llm.trace[trace_start:], time.monotonic() - start, error)
                self.state["resource_usage"] = self.usage.summary()
                self._save()
        self.usage.check()
        self.last_chat_identity = digest({"args": args, "kwargs": kwargs, "response": response,
            "profile": getattr(self.llm, "profile", None),
            "retry_seed_segments": getattr(self.llm, "retry_seed_segments", [])})
        return response

    def repair(self, scope: str, reason: str):
        check_cancelled()
        self.usage.check()
        # Replaying the same failed gate to reconstruct its feedback must not
        # charge the same repair again. Separate attempts (even identical errors)
        # have distinct request ordinals or occurrence indexes.
        visit = digest({"scope": scope, "reason": reason,
                        "request": getattr(self.llm, "requests", 0),
                        "last_chat": self.last_chat_identity,
                        "nodes": self.active_nodes})
        occurrence = self.repair_visits.get(visit, 0)
        self.repair_visits[visit] = occurrence + 1
        key = digest({"visit": visit, "occurrence": occurrence})
        if any(issue.get("key") == key for issue in self.state["issues"]):
            self.llm.trace.append({"type": "causal_repair_replay", "scope": scope,
                                   "reason": reason, "key": key})
            return
        if self.payload.get("workflow_policy") == "chapter_editor_v1":
            semantic = {"blueprint", "chapter_intent", "chapter_semantic", "scene"}
            if scope in semantic and any(issue["scope"] in semantic
                                          for issue in self.state["issues"]):
                raise RepairLimitError("Chapter editor semantic repair limit reached.")
            if scope == "record" and any(issue["scope"] == "record"
                                          for issue in self.state["issues"]):
                raise RepairLimitError("Chapter editor record correction limit reached.")
        if self.state["repairs"] >= self.max_repairs:
            raise RepairLimitError(f"Causal chapter repair budget exhausted ({scope}): {reason}")
        self.state["repairs"] += 1
        self.state["issues"].append({"scope": scope, "reason": reason, "key": key})
        self.llm.trace.append({"type": "causal_repair", "scope": scope, "reason": reason,
                               "repair_number": self.state["repairs"]})
        self._save()

    def node(self, name: str, inputs, model, generate, *, purpose: str | None = None):
        """Revalidate cached values; preserve request ordinals when skipping a stage."""
        check_cancelled()
        self.usage.check()
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,127}", name):
            raise ValueError("Invalid causal stage name.")
        # Explicit purposes keep a chapter editor's writer and review nodes
        # separate. Legacy stages retain their previous selection for replay.
        select = getattr(self.llm, "select_purpose", None)
        if callable(select):
            select(purpose or ("scene_text" if name.startswith("write-") else name))
        identity = getattr(self.llm, "runtime_identity", None)
        runtime = identity() if callable(identity) else {
            "profile": getattr(self.llm, "profile", None),
            "server_context_size": getattr(self.llm, "server_context_size", None)}
        identity = {"run": self.signature, "stage": name, "inputs": inputs,
                    "schema": model.model_json_schema() if model else None}
        if not self.live_settings:
            identity["runtime"] = runtime
        if purpose is not None:
            identity["purpose"] = purpose
        key = digest(identity)
        path = self.directory / f"{name}-{key[:20]}.json" if self.directory else None
        saved = self.cache.get(key)
        if saved is None and path and path.exists():
            saved = json.loads(path.read_text(encoding="utf-8"))
        if saved is None and self.live_settings and path and self.previous_signatures:
            # A legacy node did not store source identity separately. Rebuild its
            # exact key from validated prior settings and recorded LLM runtimes;
            # never accept a stage merely because its name matches.
            runtimes = {digest(runtime): runtime}
            output = getattr(self.llm, "output", None)
            if output is not None:
                for request_path in Path(output).glob("*.json"):
                    request = json.loads(request_path.read_text(encoding="utf-8"))
                    old_runtime = request.get("runtime") if isinstance(request, dict) else None
                    if isinstance(old_runtime, dict):
                        runtimes[digest(old_runtime)] = old_runtime
            for signature in sorted(self.previous_signatures):
                for old_runtime in runtimes.values():
                    old_identity = {**identity, "run": signature, "runtime": old_runtime}
                    old_key = digest(old_identity)
                    old_path = self.directory / f"{name}-{old_key[:20]}.json"
                    if not old_path.is_file():
                        continue
                    candidate = json.loads(old_path.read_text(encoding="utf-8"))
                    if candidate.get("key") != old_key:
                        raise ValueError("Causal stage fingerprint mismatch.")
                    saved = {**candidate, "key": key, "source_identity": identity,
                             "execution_runtime": old_runtime, "legacy_stage_key": old_key}
                    break
                if saved is not None:
                    break
        if saved is not None:
            if saved.get("key") != key:
                raise ValueError("Causal stage fingerprint mismatch.")
            if self.live_settings and saved.get("source_identity") != identity:
                raise ValueError("Causal stage source identity mismatch.")
            result = model.model_validate(saved["value"]) if model else deepcopy(saved["value"])
            if type(saved.get("requests")) is not int or saved["requests"] < 0:
                raise ValueError("Invalid cached causal stage request count.")
            check_cancelled()
            # Stage execution changes profiles. Reproduce that transition before
            # identifying the next stage, without changing the process lifetime.
            profile = saved.get("exit_profile")
            if profile is not None and not self.live_settings:
                select = getattr(self.llm, "set_profile", None)
                if callable(select):
                    select(deepcopy(profile))
                else:
                    self.llm.profile = deepcopy(profile)
            self.llm.requests = getattr(self.llm, "requests", 0) + saved["requests"]
            self.last_chat_identity = saved.get("last_chat_identity")
            self.llm.trace.append({"type": "causal_stage_cache", "stage": name, "key": key})
            if self.live_settings and path and not path.exists():
                write_json(path, saved)
            return result
        start = getattr(self.llm, "requests", 0)
        started = time.monotonic()
        self.active_nodes.append(key)
        try:
            result = generate()
            if model:
                result = model.model_validate(result)
            check_cancelled()
        except BaseException:
            self.llm.trace.append({"type": "causal_stage_failed", "stage": name, "key": key,
                                   "elapsed_seconds": time.monotonic() - started})
            raise
        finally:
            self.active_nodes.pop()
        value = result.model_dump(mode="json") if isinstance(result, BaseModel) else deepcopy(result)
        saved = {"key": key, "value": value,
                 "requests": getattr(self.llm, "requests", 0) - start,
                 "exit_profile": deepcopy(getattr(self.llm, "profile", None)),
                 "last_chat_identity": self.last_chat_identity}
        if self.live_settings:
            saved.update(source_identity=identity, execution_runtime=runtime)
        if path:
            write_json(path, saved)
        self.cache[key] = saved
        self.llm.trace.append({"type": "causal_stage", "stage": name, "key": key,
                               "elapsed_seconds": time.monotonic() - started})
        return result
