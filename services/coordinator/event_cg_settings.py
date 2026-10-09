"""Optional Qwen settings and per-work CG limits, frozen on production start."""
from __future__ import annotations

import json

from fastapi import APIRouter
from pydantic import Field

from packages.contracts.event_cg import (
    CG_CONFIGURATIONS,
    CG_KINDS,
    CG_SIZES,
    STAGING_RULES_VERSION,
    EventCgPolicy,
    EventCgProfile,
    configuration_id,
)
from packages.contracts.script import Contract

from .service import ServiceError, required


class PolicyUpdate(Contract):
    expected_revision: int = Field(ge=0, strict=True)
    max_cgs: int = Field(ge=0, le=100, strict=True)
    max_variants_per_cg: int = Field(ge=0, le=10, strict=True)


class ProfileUpdate(Contract):
    expected_revision: int = Field(ge=0, strict=True)
    profile: EventCgProfile


class EventCgSettings:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    @staticmethod
    def policy(connection, project_id):
        required(connection, "project", project_id)
        row = connection.execute("SELECT * FROM event_cg_policy WHERE project_id=?", (project_id,)).fetchone()
        return ({key: row[key] for key in ("max_cgs", "max_variants_per_cg", "revision")} if row else
                {"max_cgs": 0, "max_variants_per_cg": 0, "revision": 0})

    def settings(self, connection):
        row = connection.execute("SELECT * FROM event_cg_settings WHERE id=1").fetchone()
        profile = EventCgProfile.model_validate_json(row["profile"]) if row else EventCgProfile()
        # Worker registrations remain for job history after the process exits.
        # The settings inventory must describe only currently connected workers.
        workers = [{"id": worker["id"], "name": worker["name"],
                    "ready": CG_KINDS <= set(json.loads(worker["capabilities"]))}
                   for worker in connection.execute(
                       "SELECT * FROM worker WHERE last_seen_at>=? ORDER BY name,id",
                       (self.coordinator.clock() - 90,))]
        workers.sort(key=lambda worker: not worker["ready"])
        return {"profile": profile.model_dump(mode="json"), "revision": row["revision"] if row else 0,
                "ready": any(worker["ready"] for worker in workers), "workers": workers,
                "configuration": configuration_id(profile),
                "configurations": [dict(value) for value in CG_CONFIGURATIONS],
                "sizes": [{"width": width, "height": height} for width, height in CG_SIZES]}

    def freeze(self, connection, project_id, *, expected_revision=None):
        policy = self.policy(connection, project_id)
        if expected_revision is not None and expected_revision != policy["revision"]:
            raise ServiceError(409, "イベントCG設定が更新されました。最新の設定を確認してください。")
        if policy["max_cgs"] == 0:
            return EventCgPolicy(planning_version=STAGING_RULES_VERSION,
                                 max_variants_per_cg=policy["max_variants_per_cg"])
        settings = self.settings(connection)
        if not settings["ready"]:
            raise ServiceError(409, "イベントCGを使うにはQwenの準備済みWorkerを起動してください。CG上限0なら準備不要です。")
        return EventCgPolicy(planning_version=STAGING_RULES_VERSION, max_cgs=policy["max_cgs"],
                             max_variants_per_cg=policy["max_variants_per_cg"],
                             generation_profile=settings["profile"])


def router(coordinator):
    routes = APIRouter(prefix="/api/event-cg", tags=["Event CG"])

    @routes.get("/settings")
    def settings():
        service = coordinator()
        with service.db.transaction() as connection:
            return EventCgSettings(service).settings(connection)

    @routes.put("/settings")
    def update_settings(body: ProfileUpdate):
        service = coordinator()
        with service.db.transaction() as connection:
            manager = EventCgSettings(service)
            previous = manager.settings(connection)
            if previous["revision"] != body.expected_revision:
                raise ServiceError(409, "画像設定が更新されました。最新の設定を確認してください。")
            connection.execute("INSERT INTO event_cg_settings VALUES (1,?,?) ON CONFLICT(id) "
                               "DO UPDATE SET profile=excluded.profile,revision=excluded.revision",
                               (body.profile.model_dump_json(), previous["revision"] + 1))
            return manager.settings(connection)

    @routes.get("/projects/{project_id}/policy")
    def policy(project_id: str):
        with coordinator().db.transaction() as connection:
            return EventCgSettings.policy(connection, project_id)

    @routes.put("/projects/{project_id}/policy")
    def update_policy(project_id: str, body: PolicyUpdate):
        with coordinator().db.transaction() as connection:
            previous = EventCgSettings.policy(connection, project_id)
            if previous["revision"] != body.expected_revision:
                raise ServiceError(409, "イベントCG設定が更新されました。最新の設定を確認してください。")
            connection.execute("INSERT INTO event_cg_policy VALUES (?,?,?,?) ON CONFLICT(project_id) "
                               "DO UPDATE SET max_cgs=excluded.max_cgs,"
                               "max_variants_per_cg=excluded.max_variants_per_cg,revision=excluded.revision",
                               (project_id, body.max_cgs, body.max_variants_per_cg, previous["revision"] + 1))
            return EventCgSettings.policy(connection, project_id)

    return routes
