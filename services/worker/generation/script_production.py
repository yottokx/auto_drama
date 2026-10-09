"""Run the current script engine one app chapter at a time, with portable history.

Only source text, plans, factual notes and compact usage cross job boundaries.
The current job keeps the original request journal for cancellation/retry; media
and Tyrano publication remain the coordinator's responsibility.
"""
from __future__ import annotations

import copy
import time
from pathlib import Path

from packages.contracts.m3 import NarrativeResult
from packages.narrative import validate_narrative
from packages.narrative.continuity import narrative_hash

from .cancellation import GenerationCancelled
from .causal_runtime import digest
from .draft_story import _read
from .execution_settings import enabled, rebind_manifest, semantic_identity
from .llm import write_json
from .planning_state import approved_plan_identity, install_plan
from .script_cast import ScriptOptions
from .script_continuation import POLICY, ScriptRun
from .script_examples import example_catalog_identity
from .workflow_version import generator_protocol

HISTORY_KEYS = ("cast_plan", "plot", "plot_sha256", "outline", "accepted_plan", "story_chain",
                "story_chain_sha256", "chapter_allocation", "notes", "locations",
                "fallbacks", "request_ordinal", "sessions")
CHAPTER_KEYS = ("number", "text", "sha256", "scene_starts", "narrative_hash")
USAGE_KEYS = ("chapter", "purpose", "status", "dispatched", "charged_tokens",
              "reserved_tokens", "elapsed_seconds", "usage")


def _generation_identity_config(config: dict, payload: dict) -> dict:
    """Keep managed model identity portable without changing legacy checkpoints."""
    if not payload.get("profile", {}).get("common_settings_version"):
        return config
    identity = copy.deepcopy(config)
    if "llm_base" in identity:
        identity.pop("llm_config", None)
        identity["llm_base"]["model"].pop("relative_path", None)
        identity["llm_base"]["server"].pop("executable", None)
    return identity


class ProductionScriptRun(ScriptRun):
    def __init__(self, output: Path, config: dict, payload: dict):
        count = payload["approval_snapshot"]["world"]["result"]["chapterCount"]
        number = payload.get("chapter_number", 1)
        if type(number) is not int or not 1 <= number <= count or not payload.get("storyline_id"):
            raise ValueError("Script production needs a valid chapter and storyline.")
        identity = {"schema_version": 1, "execution_mode": "production", "workflow_policy": POLICY,
            "generator_protocol": generator_protocol("causal", POLICY),
            "approval_snapshot": payload["approval_snapshot"], "approved_chapter_count": count,
            "storyline_id": payload["storyline_id"], "seed": payload["seed"],
            "profile": payload.get("profile", {}), "profiles": payload.get("profiles", {}),
            "workflow_limits": payload.get("workflow_limits", {}),
            "generation_config": _generation_identity_config(config, payload),
            "script_options": ScriptOptions.model_validate(payload.get("script_options", {})).model_dump(),
            "script_examples": example_catalog_identity()}
        accepted = approved_plan_identity(payload)
        if accepted is not None:
            identity["approved_plan"] = accepted[1]
        if enabled(payload):
            identity.update(execution_settings_version=1,
                            execution_settings_revision=payload.get("execution_settings_revision", 0))
        manifest = {**identity, "input_sha256": digest(
            semantic_identity(identity) if enabled(payload) else identity)}
        self.inherited = self._history(payload, manifest, number)
        output.mkdir(parents=True, exist_ok=True)
        manifest_path = output / "experiment.json"
        if manifest_path.exists() and _read(manifest_path) != manifest:
            if not enabled(payload):
                raise ValueError("Script production input or model configuration changed.")
            manifest = rebind_manifest(output, _read(manifest_path), manifest)
        write_json(manifest_path, manifest)
        journal = output / "draft-state.json"
        if not journal.exists() and number == 1 and accepted is not None:
            if any(p.is_file() and p != manifest_path for p in output.rglob("*")):
                raise ValueError("Script journal is missing beside generated material.")
            state = {"input_sha256": manifest["input_sha256"], "steps": {}, "chapters": [],
                     "notes": [], "fallbacks": [], "sessions": [], "request_ordinal": 0, "partial": None}
            install_plan(state, output, manifest, *accepted)
            write_json(journal, state)
        if not journal.exists() and self.inherited is not None:
            # A missing journal beside generated material must never reset usage.
            if any(p.is_file() and p != manifest_path for p in output.rglob("*")):
                raise ValueError("Script journal is missing beside generated material.")
            state = {**copy.deepcopy(self.inherited), "input_sha256": manifest["input_sha256"],
                     "steps": {}, "partial": None}
            state["chapters"][-1].update(narrative=copy.deepcopy(payload["previous_narrative"]),
                                        artifact_id=payload["previous_narrative_artifact_id"])
            write_json(journal, state)
        if self.inherited is not None:
            for key, filename in (("cast_plan", "cast-plan.json"), ("plot", "plot.json"),
                                  ("outline", "outline.json"), ("story_chain", "story-chain.json"),
                                  ("chapter_allocation", "chapter-allocation.json")):
                if key in self.inherited and not (output / filename).exists():
                    write_json(output / filename, self.inherited[key])
        super().__init__(output, manifest, config, payload, number)

    def retry_failed_steps(self, generation: int) -> None:
        """Grant bounded fresh calls only after an explicit coordinator retry.

        This execution counter is separate from the immutable production input.
        Keep old replies, charges and request ordinals for auditing and budgets.
        """
        previous = self.state.get("retry_generation", 0)
        if type(generation) is not int or generation < previous:
            raise ValueError("Invalid or stale script retry generation.")
        if generation == previous:
            return
        for step in self.state["steps"].values():
            attempts = self._current_attempts(step)
            if not attempts or step.get("core_cast_replacement"):
                continue
            last = attempts[-1]
            if step.get("accepted_request") == last.get("request"):
                continue
            if not last.get("validation_error") and last["status"] not in {"failed", "interrupted"}:
                continue
            start = len(step["attempts"])
            step["retry_attempt_start"] = start
            step.setdefault("retry_generations", []).append({"generation": generation,
                                                            "start_attempt": start})
        self.state["retry_generation"] = generation
        self.persist()

    @staticmethod
    def _history(payload, manifest, number):
        checkpoint = payload.get("script_checkpoint")
        if number == 1:
            if checkpoint or payload.get("previous_narrative") or payload.get("previous_narrative_artifact_id"):
                raise ValueError("First script chapter cannot inherit a predecessor.")
            return None
        if not isinstance(checkpoint, dict) or not payload.get("previous_narrative_artifact_id"):
            raise ValueError("Script continuation requires its saved checkpoint and predecessor artifact.")
        previous = NarrativeResult.model_validate(payload.get("previous_narrative"))
        expected = {"schema_version": 1, "generator_protocol": manifest["generator_protocol"],
            "storyline_id": payload["storyline_id"], "approval_sha256": digest(payload["approval_snapshot"]),
            "chapter_number": number - 1, "narrative_hash": narrative_hash(previous),
            "generation_identity": manifest["input_sha256"]}
        if enabled(payload) and not enabled(checkpoint):
            # Old checkpoints bind model settings into an opaque hash. The
            # coordinator supplies their recorded source payload for migration;
            # all source, plan, seed and workflow checks still apply.
            source = payload.get("script_checkpoint_source")
            if not isinstance(source, dict):
                raise ValueError("Legacy script checkpoint requires its recorded source input.")
            for key in ("storyline_id", "seed", "workflow_policy", "generator_protocol"):
                if source.get(key) != payload.get(key):
                    raise ValueError("Script checkpoint source or generation seed was changed.")
            if (digest(source.get("approval_snapshot")) != digest(payload["approval_snapshot"])
                    or source.get("workflow_limits", {}) != payload.get("workflow_limits", {})
                    or ScriptOptions.model_validate(source.get("script_options", {})) !=
                    ScriptOptions.model_validate(payload.get("script_options", {}))
                    or approved_plan_identity(source) != approved_plan_identity(payload)):
                raise ValueError("Script checkpoint source plan or workflow was changed.")
            expected.pop("generation_identity")
        if "approved_plan" in manifest:
            expected.update(plan_approval_id=manifest["approved_plan"]["approval_id"],
                            plan_sha256=manifest["approved_plan"]["sha256"])
        if (any(checkpoint.get(k) != v for k, v in expected.items())
                or checkpoint.get("sha256") != digest({k: v for k, v in checkpoint.items() if k != "sha256"})
                or previous.chapter_number != number - 1 or previous.storyline_id != payload["storyline_id"]
                or previous.workflow_policy != POLICY):
            raise ValueError("Script checkpoint does not match the adopted predecessor or generation settings.")
        state = checkpoint["state"]
        chapters = state["chapters"]
        if (len(chapters) != number - 1 or chapters[-1]["narrative_hash"] != narrative_hash(previous)
                or chapters[-1]["text"] != ScriptRun.script_text(previous.scenes)
                or state["outline"] != previous.outline.model_dump(mode="json")):
            raise ValueError("Script checkpoint source history differs from its predecessor.")
        for index, row in enumerate(chapters, 1):
            if row["number"] != index or digest(row["text"]) != row["sha256"]:
                raise ValueError("Script checkpoint chapter sequence or source hash was changed.")
        for note in state["notes"]:
            if (not 1 <= note["number"] <= len(chapters)
                    or note["source_sha256"] != chapters[note["number"] - 1]["sha256"]):
                raise ValueError("Script checkpoint note has no matching source chapter.")
        if enabled(payload):
            state = copy.deepcopy(state)
            cast = state.get("cast_plan")
            if cast is not None:
                if (cast.get("input_sha256") != checkpoint.get("generation_identity")
                        or cast.get("sha256") != digest(cast.get("plan"))
                        or cast.get("protocol") != manifest["generator_protocol"]):
                    raise ValueError("Script checkpoint cast provenance was changed.")
                if cast["input_sha256"] != manifest["input_sha256"]:
                    cast.setdefault("source_input_sha256", cast["input_sha256"])
                    cast["input_sha256"] = manifest["input_sha256"]
        return state

    @property
    def storyline_id(self):
        return self.payload["storyline_id"]

    def _verify_narratives(self):
        accepted = approved_plan_identity(self.payload)
        if accepted is not None:
            content, identity = accepted
            if (self.state.get("accepted_plan") != identity
                    or self.state.get("plot") != content["plot"]
                    or self.state.get("cast_plan", {}).get("plan") != content["cast_plan"]):
                raise ValueError("Script execution plan differs from its approved common plan.")
        count = self.target - 1
        if len(self.state["chapters"]) < count:
            raise ValueError("Script journal lost its inherited chapter history.")
        if self.inherited is not None:
            for saved, inherited in zip(self.state["chapters"][:count], self.inherited["chapters"], strict=True):
                if any(saved.get(key) != inherited.get(key) for key in (*CHAPTER_KEYS, "scene_count")):
                    raise ValueError("Script journal inherited source was changed.")
            previous = self.state["chapters"][count - 1]
            if (previous["narrative"] != self.payload["previous_narrative"]
                    or previous["artifact_id"] != self.payload["previous_narrative_artifact_id"]):
                raise ValueError("Script journal predecessor was changed.")
        for row in self.state["chapters"][count:]:
            value = validate_narrative(row["narrative"], self.payload["approval_snapshot"],
                self.payload.get("previous_narrative"),
                expected_previous_artifact_id=self.payload.get("previous_narrative_artifact_id"))
            if (narrative_hash(value) != row["narrative_hash"] or self.script_text(value.scenes) != row["text"]
                    or value.storyline_id != self.storyline_id):
                raise ValueError("Saved production narrative was changed.")

    def export_chapter(self, result, export_dir):
        # Production exports must use real approved/generated media, downstream.
        return {"files": {}}

    def rows(self):
        return self.state.get("carried_usage", []) + super().rows()

    def writer_samples(self, model_id, *, excluding=None):
        return (self.state.get("writer_samples", {}).get(model_id, [])
                + super().writer_samples(model_id, excluding=excluding))

    def save(self, status, error=None):
        report = super().save(status, error)
        report.update(execution_mode="production", exported_chapter_count=0,
                      production_chapter=self.target)
        write_json(self.output / "report.json", report)
        return report

    def checkpoint(self):
        state = {key: copy.deepcopy(self.state[key]) for key in HISTORY_KEYS if key in self.state}
        state["chapters"] = [{**{key: row[key] for key in CHAPTER_KEYS},
            "scene_count": len(row["narrative"]["scenes"]) if "narrative" in row else row["scene_count"]}
            for row in self.state["chapters"]]
        state["carried_usage"] = [{key: row[key] for key in USAGE_KEYS if key in row} for row in self.rows()]
        model_ids = set(self.state.get("writer_samples", {})) | {
            row["profile"]["model_id"] for step in self.state["steps"].values() for row in step["attempts"]}
        state["writer_samples"] = {model: self.writer_samples(model) for model in sorted(model_ids)}
        value = {"schema_version": 1, "generator_protocol": self.manifest["generator_protocol"],
            "storyline_id": self.storyline_id, "approval_sha256": digest(self.payload["approval_snapshot"]),
            "chapter_number": self.target, "narrative_hash": self.state["chapters"][-1]["narrative_hash"],
            "generation_identity": self.manifest["input_sha256"], "state": state}
        if enabled(self.payload):
            value["execution_settings_version"] = 1
        if "approved_plan" in self.manifest:
            value.update(plan_approval_id=self.manifest["approved_plan"]["approval_id"],
                         plan_sha256=self.manifest["approved_plan"]["sha256"])
        return {**value, "sha256": digest(value)}

    def execute(self, llm):
        self.llm = llm
        llm.requests = self.state["request_ordinal"]
        llm._load_number = max((int(path.name.split("-", 1)[0]) for path in (llm.output / "runtimes").glob("*")
                               if path.name.split("-", 1)[0].isdigit()), default=0)
        self.started = time.monotonic()
        self.session = {"status": "running", "elapsed_seconds": 0, "chapter_limit": self.target}
        self.state["sessions"].append(self.session)
        status, error = "running", None
        try:
            self.save(status)
            self.restore_progress()
            self.check_budget(self.target)
            self.write_chapters()
            status = "script_complete" if self.target == self.manifest["approved_chapter_count"] else "chapter_limit_reached"
        except BaseException as exc:
            status = "interrupted" if isinstance(exc, (KeyboardInterrupt, GenerationCancelled)) else "failed"
            error = f"{type(exc).__name__}: {exc}"
            if not isinstance(exc, (KeyboardInterrupt, GenerationCancelled)):
                self.fail_progress()
            raise
        finally:
            self.session["status"] = status
            loads = sum(row.get("type") == "model_switch" and row.get("status") == "ready" for row in llm.trace)
            self.session.update(model_loads=loads, model_switches=max(0, loads - 1))
            self.save(status, error)
        return self.state["chapters"][-1]["narrative"], self.checkpoint()
