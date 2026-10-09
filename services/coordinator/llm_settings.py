"""Coordinator-owned common settings, snapshotted when jobs are created."""
import json
from pathlib import Path

from packages.contracts.llm_settings import LLMCapability, LLMSettings

from .service import ServiceError

ROOT = Path(__file__).resolve().parents[2]


def saved_settings(coordinator, connection=None) -> LLMSettings | None:
    owned = connection is None
    connection = connection or coordinator.db.connect()
    try:
        row = connection.execute("SELECT value FROM llm_settings WHERE id=1").fetchone()
        return LLMSettings.model_validate_json(row["value"]) if row else None
    finally:
        if owned:
            connection.close()


LLM_JOB_KINDS = frozenset({
    "m2_world", "m2_character", "m2_relationships", "m2_image", "m3_narrative",
    "m3_image", "m3_plan", "m3_music_plan", "m3_event_cg_budget", "m3_event_cg_plan",
})
RETRY_LLM_JOB_KINDS = frozenset({
    "m2_world", "m2_character", "m2_relationships", "m2_image", "m3_image", "m3_narrative", "m3_plan",
    "m3_music_plan", "m3_event_cg_budget", "m3_event_cg_plan",
})
COMMON_PROFILE_KEYS = frozenset({
    "provider", "model_id", "temperature", "top_p", "reasoning_level", "context_size",
    "max_context_size", "model_context_size", "allow_context_expansion", "common_settings_version",
})


def generation_profile(coordinator, connection=None) -> dict:
    """Read current common settings without opening a second transaction."""
    common = saved_settings(coordinator, connection)
    if common is not None:
        return {**common.profile(), "max_tokens": 3072, "prompt_version": 4}
    llm = json.loads((ROOT / "config/m2-generation.json").read_text(encoding="utf-8"))["llm"]
    return {"provider": "local", "model_id": llm["model_id"],
            "temperature": llm["temperature"], "max_tokens": llm["max_tokens"],
            "reasoning_level": llm["reasoning_level"], "prompt_version": 4}


def execution_settings(coordinator, connection, payload, *, kind, revision=0, profile=None):
    """Snapshot a fresh execution; story, workflow and voice settings stay separate."""
    current = dict(profile if profile is not None else generation_profile(coordinator, connection))
    if payload.get("phase") == "planning" or payload.get("workflow_policy") == "script_continuation_v1":
        current = {key: value for key, value in current.items()
                   if key not in {"max_tokens", "prompt_version"}}
        if not current.get("common_settings_version"):
            llm = json.loads((ROOT / "config/m2-generation.json").read_text(encoding="utf-8"))["llm"]
            current.update(reasoning_level="none", context_size=llm.get("context_size", 16384))
    payload["profile"] = current
    result = {"profile": current}
    if kind in RETRY_LLM_JOB_KINDS:
        payload.update(execution_settings_version=1, execution_settings_revision=revision)
        result.update(execution_settings_version=1, execution_settings_revision=revision)
    else:
        payload.pop("execution_settings_version", None)
        payload.pop("execution_settings_revision", None)
    if current.get("common_settings_version") and "profiles" in payload:
        # Stage output limits stay with the workflow. Inherited inference choices
        # must not override a newly selected common model or context size.
        payload["profiles"] = {purpose: {key: value for key, value in overrides.items()
                                        if key not in COMMON_PROFILE_KEYS}
                               for purpose, overrides in payload["profiles"].items()}
        result["profiles"] = payload["profiles"]
    return result


def read_settings(coordinator) -> dict:
    settings = saved_settings(coordinator)
    if settings is None:
        defaults = json.loads((ROOT / "config/m2-generation.json").read_text(encoding="utf-8"))["llm"]
        settings = LLMSettings(model=defaults["model_id"], temperature=defaults["temperature"],
                               top_p=defaults.get("top_p", 0.95),
                               reasoning_effort=defaults["reasoning_level"],
                               ctx_size=defaults["context_size"])
    connection = coordinator.db.connect()
    try:
        models = [dict(model, worker_name=row["name"], worker_id=row["id"])
                  for row in connection.execute("SELECT * FROM worker WHERE last_seen_at>=?",
                                                (coordinator.clock() - 60,))
                  for model in json.loads(row["llm_models"])]
    finally:
        connection.close()
    return {"settings": settings.model_dump(), "models": models}


def save_settings(coordinator, settings: LLMSettings) -> dict:
    with coordinator.db.transaction() as connection:
        models = [LLMCapability.model_validate(model)
                  for row in connection.execute("SELECT llm_models FROM worker WHERE last_seen_at>=?",
                                                (coordinator.clock() - 60,))
                  for model in json.loads(row["llm_models"])]
        if not any(model.supports(settings.profile()) for model in models):
            raise ServiceError(422, "指定したモデルに対応するWorkerがありません。")
        connection.execute("INSERT INTO llm_settings(id,value) VALUES(1,?) "
                           "ON CONFLICT(id) DO UPDATE SET value=excluded.value",
                           (settings.model_dump_json(),))
    return read_settings(coordinator)
