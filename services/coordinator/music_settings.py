"""BGM settings: the precision new works make their tracks with, kept per work by its first track."""
from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import Field

from packages.contracts.music import MUSIC_CONFIGURATIONS, MusicProfile
from packages.contracts.script import Contract

from .service import ServiceError


class ProfileUpdate(Contract):
    expected_revision: int = Field(ge=0, strict=True)
    profile: MusicProfile


class MusicSettings:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    @staticmethod
    def profile(connection) -> MusicProfile:
        row = connection.execute("SELECT profile FROM music_settings WHERE id=1").fetchone()
        return MusicProfile.model_validate_json(row["profile"]) if row else MusicProfile()

    def settings(self, connection):
        row = connection.execute("SELECT revision FROM music_settings WHERE id=1").fetchone()
        # Worker registrations remain for job history after the process exits.
        workers = [{"id": worker["id"], "name": worker["name"],
                    "ready": "m3_music" in json.loads(worker["capabilities"])}
                   for worker in connection.execute("SELECT * FROM worker WHERE last_seen_at>=? ORDER BY name,id",
                                                    (self.coordinator.clock() - 90,))]
        workers.sort(key=lambda worker: not worker["ready"])
        return {"profile": self.profile(connection).model_dump(mode="json"), "revision": row["revision"] if row else 0,
                "ready": any(worker["ready"] for worker in workers), "workers": workers,
                "configurations": [dict(value) for value in MUSIC_CONFIGURATIONS]}


def track_settings(connection, project_id) -> dict:
    """The model fields of a new track. A work keeps the precision of the tracks it already has.

    Half precision draws a related but different track from the same seed, so a work never mixes
    the two. Tracks requested before this setting existed carry no precision and stay float32.
    """
    earlier = connection.execute(
        "SELECT r.descriptor FROM m3_requirement r JOIN m3_production p ON p.id=r.production_id "
        "WHERE p.project_id=? AND r.kind='m3_music' "
        "UNION ALL SELECT payload FROM job WHERE project_id=? AND kind='m3_music' LIMIT 1",
        (project_id, project_id)).fetchone()
    precision = (json.loads(earlier[0]).get("precision", "float32") if earlier
                 else MusicSettings.profile(connection).precision)
    # float32 adds nothing, so requests made before the setting keep their exact bytes.
    return {"backend": "stable_audio3", "model": "medium",
            **({"precision": precision} if precision != "float32" else {})}


def router(coordinator):
    routes = APIRouter(prefix="/api/music", tags=["Music"])

    @routes.get("/settings")
    def settings():
        service = coordinator()
        with service.db.transaction() as connection:
            return MusicSettings(service).settings(connection)

    @routes.put("/settings")
    def update_settings(body: ProfileUpdate):
        service = coordinator()
        with service.db.transaction() as connection:
            manager = MusicSettings(service)
            previous = manager.settings(connection)
            if previous["revision"] != body.expected_revision:
                raise ServiceError(409, "BGM設定が更新されました。最新の設定を確認してください。")
            connection.execute("INSERT INTO music_settings VALUES (1,?,?) ON CONFLICT(id) "
                               "DO UPDATE SET profile=excluded.profile,revision=excluded.revision",
                               (body.profile.model_dump_json(), previous["revision"] + 1))
            return manager.settings(connection)

    return routes
