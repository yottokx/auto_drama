"""Coordinator-owned common settings, snapshotted when jobs are created."""
import json
from pathlib import Path

from packages.contracts.llm_settings import LLMCapability, LLMSettings

from .service import ServiceError

ROOT = Path(__file__).resolve().parents[2]


def saved_settings(coordinator) -> LLMSettings | None:
    connection = coordinator.db.connect()
    try:
        row = connection.execute("SELECT value FROM llm_settings WHERE id=1").fetchone()
        return LLMSettings.model_validate_json(row["value"]) if row else None
    finally:
        connection.close()


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
