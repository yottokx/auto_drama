"""Separate immutable story inputs from settings selected for an execution."""
from __future__ import annotations

import copy
import json
from pathlib import Path

from .causal_runtime import digest
from .llm import write_json

_PAYLOAD_SETTINGS = frozenset({"profile", "profiles", "execution_settings_version",
                               "execution_settings_revision"})
_PAYLOAD_EVIDENCE = frozenset({"checkpoint_source", "script_checkpoint_source"})
_CONFIG_SETTINGS = frozenset({"llm", "llm_config", "llm_base", "model_routing",
                              "model_configuration_identity"})
_MANIFEST_METADATA = frozenset({"input_sha256", "execution_history"})
_CG_PLANNERS = frozenset({"m3_event_cg_budget", "m3_event_cg_plan"})


def enabled(payload: dict) -> bool:
    version = payload.get("execution_settings_version")
    if version is None:
        return False
    if type(version) is not int or version != 1:
        raise ValueError("Unsupported execution settings version.")
    return True


def execution_revision(payload: dict) -> int:
    revision = payload.get("execution_settings_revision", 0)
    if type(revision) is not int or revision < 0:
        raise ValueError("Invalid execution settings revision.")
    return revision


def source_payload(payload: dict) -> dict:
    """Retain story inputs; checkpoint_source is separately checked evidence."""
    return {key: copy.deepcopy(value) for key, value in payload.items()
            if key not in _PAYLOAD_SETTINGS | _PAYLOAD_EVIDENCE}


def source_config(config: dict) -> dict:
    music_identity = config.get("music_job_identity")
    llm_music_identity = isinstance(music_identity, dict) and "generation" in music_identity
    return {key: copy.deepcopy(value) for key, value in config.items()
            if key not in _CONFIG_SETTINGS and not (key == "music_job_identity" and llm_music_identity)}


def semantic_identity(identity: dict) -> dict:
    result = {key: copy.deepcopy(value) for key, value in identity.items()
              if key not in _PAYLOAD_SETTINGS | _MANIFEST_METADATA}
    if "generation_config" in result:
        result["generation_config"] = source_config(result["generation_config"])
    return result


def _validate_manifest(manifest: dict) -> None:
    identity = {key: value for key, value in manifest.items()
                if key not in _MANIFEST_METADATA}
    expected = digest(semantic_identity(identity) if enabled(manifest) else identity)
    if manifest.get("input_sha256") != expected:
        raise ValueError("Saved generation manifest hash was changed.")


def _execution_identity(value: dict) -> dict:
    return {key: copy.deepcopy(value.get(key)) for key in ("profile", "profiles", "generation_config")}


def bind_job_request(work: Path, kind: str, payload: dict, config: dict,
                     retry_generation: int = 0) -> tuple[dict, str, list[dict]]:
    """Bind live settings only at a new execution or explicit retry boundary."""
    if not enabled(payload):
        raise ValueError("Live job binding requires the execution settings protocol.")
    revision = execution_revision(payload)
    if type(retry_generation) is not int or retry_generation < 0 or revision != retry_generation:
        raise ValueError("Execution settings revision must match the explicit retry generation.")
    identity = {"kind": kind, "payload": source_payload(payload)}
    fingerprint = digest(identity)
    path = work / "job-request.json"
    current = {"fingerprint": fingerprint, "kind": kind, "payload": copy.deepcopy(payload),
               "generation_config": copy.deepcopy(config)}
    history: list[dict] = []
    if path.is_file():
        previous = json.loads(path.read_text(encoding="utf-8"))
        previous_payload = previous["payload"]
        previous_identity = {"kind": previous["kind"], "payload": previous_payload}
        saved_fingerprint = digest({"kind": previous["kind"], "payload": source_payload(previous_payload)}
                                   if enabled(previous_payload) else previous_identity)
        if previous.get("fingerprint") != saved_fingerprint:
            raise ValueError("Saved generation request fingerprint was changed.")
        previous_config = previous.get("generation_config")
        missing_cg_config = previous_config is None and kind in _CG_PLANNERS
        if ({"kind": previous["kind"], "payload": source_payload(previous_payload)} != identity
                or (previous_config is not None and source_config(previous_config) != source_config(config))):
            raise ValueError("Generation work directory belongs to another input snapshot.")
        old_revision = execution_revision(previous_payload)
        if previous_config is None:
            # These legacy planners saved their complete payload and LLM model
            # identity, but no worker config. Preserve precisely that evidence;
            # do not manufacture a historical non-LLM deployment configuration.
            registry = previous.get("model_identity")
            generation = registry.get("generation") if isinstance(registry, dict) else None
            if (not missing_cg_config or enabled(previous_payload) or revision <= old_revision
                    or not isinstance(generation, dict) or not isinstance(generation.get("settings"), dict)):
                raise ValueError("Legacy planner settings require recorded identity and an explicit retry.")
        if revision < old_revision:
            raise ValueError("Stale execution settings revision.")
        changed_profile = any(previous_payload.get(key) != payload.get(key)
                              for key in ("profile", "profiles"))
        if changed_profile and revision <= old_revision:
            raise ValueError("Changing model settings requires a new explicit retry revision.")
        history = copy.deepcopy(previous.get("execution_history", []))
        if revision > old_revision or not enabled(previous_payload):
            evidence = {key: copy.deepcopy(value) for key, value in previous.items()
                        if key != "execution_history"}
            history.append(evidence)
        elif not changed_profile:
            config = copy.deepcopy(previous["generation_config"])
            current["generation_config"] = config
        if history:
            current["execution_history"] = history
    write_json(path, current)
    return config, fingerprint, history


def previous_executions(work: Path | None, payload: dict, config: dict) -> list[dict]:
    """Return validated recorded settings for durable legacy budget migration."""
    if work is None or not enabled(payload):
        return []
    path = work / "job-request.json"
    if not path.is_file():
        return []
    request = json.loads(path.read_text(encoding="utf-8"))
    result = []
    for previous in request.get("execution_history", []):
        if "generation_config" not in previous and previous.get("kind") in _CG_PLANNERS:
            continue
        old_payload, old_config = previous["payload"], previous["generation_config"]
        expected = digest({"kind": previous["kind"],
                           "payload": source_payload(old_payload) if enabled(old_payload) else old_payload})
        if (previous.get("fingerprint") != expected
                or source_payload(old_payload) != source_payload(payload)
                or source_config(old_config) != source_config(config)
                or execution_revision(old_payload) >= execution_revision(payload)):
            raise ValueError("Recorded execution settings do not match the current source inputs.")
        result.append(copy.deepcopy(previous))
    return result


def rebind_manifest(output: Path, previous: dict, current: dict) -> dict:
    """Migrate identity references without discarding completed work or usage.

    The caller verifies the execution revision and atomically writes the returned
    manifest. Both old and new references are accepted during migration, so a
    process interruption between the individual atomic writes is recoverable.
    """
    if not enabled(current):
        raise ValueError("Changing generation settings requires the live settings protocol.")
    _validate_manifest(previous)
    _validate_manifest(current)
    if semantic_identity(previous) != semantic_identity(current):
        raise ValueError("Generation source inputs changed during settings migration.")
    revision, old_revision = execution_revision(current), execution_revision(previous)
    if revision < old_revision:
        raise ValueError("Stale generation settings revision.")
    if _execution_identity(previous) != _execution_identity(current) and revision <= old_revision:
        raise ValueError("Changing model settings requires a new explicit retry revision.")
    old_hash, new_hash = previous["input_sha256"], current["input_sha256"]
    allowed_hashes = {old_hash, new_hash}
    journal_path = output / "draft-state.json"
    journal = json.loads(journal_path.read_text(encoding="utf-8")) if journal_path.is_file() else None
    if journal is not None and journal.get("input_sha256") not in allowed_hashes:
        raise ValueError("Draft journal belongs to another experiment.")
    cast_path = output / "cast-plan.json"
    cast_file = json.loads(cast_path.read_text(encoding="utf-8")) if cast_path.is_file() else None
    cast_record = journal.get("cast_plan") if journal is not None else None
    for record in (cast_record, cast_file):
        if record is not None and (record.get("input_sha256") not in allowed_hashes
                                  or record.get("sha256") != digest(record.get("plan"))):
            raise ValueError("Saved cast plan provenance was changed.")
    if cast_record is not None and (cast_file is None
            or {key: value for key, value in cast_record.items() if key != "input_sha256"}
            != {key: value for key, value in cast_file.items() if key != "input_sha256"}):
        raise ValueError("Saved cast plan and journal disagree.")
    if cast_file is not None:
        if old_hash != new_hash:
            cast_file.setdefault("source_input_sha256", old_hash)
        cast_file["input_sha256"] = new_hash
        write_json(cast_path, cast_file)
    if journal is not None:
        if old_hash != new_hash:
            journal.setdefault("source_input_sha256", old_hash)
        journal["input_sha256"] = new_hash
        if cast_record is not None:
            if old_hash != new_hash:
                cast_record.setdefault("source_input_sha256", old_hash)
            cast_record["input_sha256"] = new_hash
        write_json(journal_path, journal)
    result = copy.deepcopy(current)
    history = copy.deepcopy(previous.get("execution_history", []))
    evidence = {key: copy.deepcopy(value) for key, value in previous.items()
                if key != "execution_history"}
    changed_execution = (enabled(previous) != enabled(current) or revision != old_revision
                         or _execution_identity(previous) != _execution_identity(current))
    if changed_execution and (not history or history[-1] != evidence):
        history.append(evidence)
    if history:
        result["execution_history"] = history
    return result
