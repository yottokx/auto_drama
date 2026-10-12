"""Expression pictures of any portrait of a work, made on request and never required.

A main character's looks keep theirs in the STEP3 draft. Every other portrait - a planned
supporting character's, or one chosen while adjusting a finished work - keeps them here,
keyed by the portrait they were edited from. A chapter looks both up when it is built.
"""
from __future__ import annotations

import json
import zipfile
import zlib
from uuid import uuid4

from fastapi import APIRouter
from pydantic import Field, model_validator

from packages.contracts.expressions import Expression
from packages.contracts.script import Contract

from .service import Coordinator, ServiceError, public, required

PORTRAIT_KINDS = frozenset({"character", "m2_image", "m2_outfit_image"})
FaceFraction = tuple[float, float, float, float]


class ExpressionRequest(Contract):
    base_artifact_id: str = Field(pattern=r"^[A-Za-z0-9]{1,64}$")
    character_id: str = Field(min_length=1, max_length=128)
    expressions: list[Expression] = Field(min_length=1, max_length=11)
    # Draw again even what this portrait already has.
    redraw: bool = False
    # The face marked by hand, as fractions of the picture, when it could not be found.
    face_box: FaceFraction | None = None

    @model_validator(mode="after")
    def drawable(self):
        if "neutral" in self.expressions or len(set(self.expressions)) != len(self.expressions):
            raise ValueError("name each expression once; the ordinary face is the portrait itself")
        if self.face_box is not None and (
                any(not 0 <= value <= 1 for value in self.face_box)
                or self.face_box[2] - self.face_box[0] < 0.02 or self.face_box[3] - self.face_box[1] < 0.02):
            raise ValueError("a marked face lies inside the picture and has a size")
        return self


class ExpressionRemoval(Contract):
    base_artifact_id: str = Field(pattern=r"^[A-Za-z0-9]{1,64}$")
    expression: Expression


class ExpressionService:
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.db, self.store, self.clock = coordinator.db, coordinator.store, coordinator.clock

    @staticmethod
    def portrait(connection, artifact_id: str) -> str:
        """The portrait itself: a chapter and an adjustment show it under names of their own."""
        seen = set()
        while artifact_id not in seen:
            seen.add(artifact_id)
            origin = json.loads(required(connection, "artifact", artifact_id)["provenance"] or "{}").get(
                "approved_artifact_id")
            if not origin:
                break
            artifact_id = origin
        return artifact_id

    @staticmethod
    def _cast_pictures(connection, project_id, base_artifact_id) -> dict[str, str]:
        """Expressions a main character's look already has from the cast review."""
        from .m2_service import M2Service

        return {picture["expression"]: picture["imageArtifactId"]
                for character in M2Service._load(connection, project_id)["draft"]["characters"]
                for picture in character.get("expressions", [])
                if picture["baseImageArtifactId"] == base_artifact_id}

    @classmethod
    def of_portrait(cls, connection, project_id, artifact_id) -> dict[str, str]:
        """Every finished expression of the portrait an artifact shows, wherever it was made."""
        base = cls.portrait(connection, artifact_id)
        return {**cls._cast_pictures(connection, project_id, base), **cls.current(connection, project_id, [base])}

    @staticmethod
    def _newest(connection, project_id, base_artifact_id) -> dict[str, dict]:
        """The latest request of each expression of one portrait, finished or not."""
        result = {}
        for row in connection.execute(
                "SELECT * FROM portrait_expression WHERE project_id=? AND base_artifact_id=? "
                "ORDER BY created_at,rowid", (project_id, base_artifact_id)):
            result[row["expression"]] = dict(row)
        return {name: row for name, row in result.items() if not row["removed"]}

    @classmethod
    def current(cls, connection, project_id, base_artifact_ids) -> dict[str, str]:
        """The finished picture of each expression that any of these portraits has."""
        result = {}
        for base in base_artifact_ids:
            for row in connection.execute(
                    "SELECT * FROM portrait_expression WHERE project_id=? AND base_artifact_id=? "
                    "ORDER BY created_at,rowid", (project_id, base)):
                # A later request that has not finished leaves the earlier picture in use.
                if row["removed"]:
                    result.pop(row["expression"], None)
                elif row["artifact_id"]:
                    result[row["expression"]] = row["artifact_id"]
        return result

    def pictures(self, connection, project_id, base_artifact_ids) -> dict[str, list[dict]]:
        view = {}
        for asked in dict.fromkeys(base_artifact_ids):
            base = self.portrait(connection, asked)
            finished = self.current(connection, project_id, [base])
            rows = []
            newest = self._newest(connection, project_id, base)
            for name, row in newest.items():
                job = required(connection, "job", row["job_id"]) if row["job_id"] else None
                artifact = row["artifact_id"] or finished.get(name)
                rows.append({
                    "expression": name, "base_artifact_id": asked,
                    "status": "ready" if row["artifact_id"] else job["status"] if job else "missing",
                    "url": f"/api/artifacts/{artifact}/content" if artifact else None,
                    "artifact_id": artifact, "job_id": job["id"] if job else None,
                    "error": job["error"] if job and not row["artifact_id"] else None})
            # What the cast review made for this portrait is shown too, and is not made twice.
            for name, artifact in self._cast_pictures(connection, project_id, base).items():
                if name not in newest:
                    rows.append({"expression": name, "base_artifact_id": asked, "status": "ready",
                                 "url": f"/api/artifacts/{artifact}/content", "artifact_id": artifact,
                                 "job_id": None, "error": None, "from_cast": True})
            view[asked] = rows
        return view

    def project(self, project_id, base_artifact_ids):
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            from .image_editing import enabled
            from .m2_service import M2Service

            return {"project_id": project_id,
                    # A work that forgoes the editing model has no expressions to make.
                    "enabled": enabled(M2Service._load(connection, project_id)["draft"]),
                    "portraits": self.pictures(connection, project_id, base_artifact_ids)}

    def generate(self, project_id, body: ExpressionRequest):
        from .event_cg_settings import EventCgSettings
        from .llm_settings import execution_settings
        from .m2_service import M2Service

        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            required(connection, "project", project_id)
            asked = required(connection, "artifact", body.base_artifact_id)
            if asked["project_id"] != project_id:
                raise ServiceError(422, "この作品の立ち絵を指定してください。")
            base = required(connection, "artifact", self.portrait(connection, asked["id"]))
            if (base["project_id"] != project_id or base["media_type"] != "image/png"
                    or base["kind"] not in PORTRAIT_KINDS):
                raise ServiceError(422, "この作品の立ち絵を指定してください。")
            self.store.read(base)
            m2 = M2Service(self.coordinator)
            if not m2._qwen_ready(connection, M2Service._load(connection, project_id)["draft"]):
                raise ServiceError(409, "表情の作成には、画像編集の準備ができたWorkerが必要です。")
            newest = self._newest(connection, project_id, base["id"])
            made = self._cast_pictures(connection, project_id, base["id"])
            profile = m2._profile(connection)
            cg_profile = EventCgSettings(self.coordinator).settings(connection)["profile"]
            started = 0
            for expression in body.expressions:
                row = newest.get(expression)
                job = required(connection, "job", row["job_id"]) if row and row["job_id"] else None
                if job and job["status"] in {"pending", "running"}:
                    continue
                if ((row and row["artifact_id"]) or (row is None and expression in made)) and not body.redraw:
                    continue
                identifier, job_id, run_id, now = uuid4().hex, uuid4().hex, uuid4().hex, self.clock()
                payload = {"schema_version": 1, "seed": int(job_id[:8], 16) & 0x7FFFFFFF,
                           "character_id": body.character_id, "profile": profile, "cg_profile": cg_profile,
                           "outfit": {"id": "default", "name": "通常の姿", "description": "", "occasion": ""},
                           "reference_image": {"artifact_id": base["id"], "sha256": base["sha256"]},
                           "expression": {"id": expression,
                                          **({"face_box": list(body.face_box)} if body.face_box else {})},
                           "portrait_expression": {"id": identifier}}
                settings = {"schema_version": 1, "seed": payload["seed"], **execution_settings(
                    self.coordinator, connection, payload, kind="m2_outfit_image", profile=profile)}
                connection.execute(
                    "INSERT INTO generation_run(id,project_id,settings_version,story_revision_id,policy,status,created_at) "
                    "VALUES (?,?,?,?,?,'pending',?)",
                    (run_id, project_id, 0, base["id"], json.dumps({"mode": "portrait-expression"}), now))
                connection.execute(
                    "INSERT INTO job(id,project_id,run_id,kind,payload,settings_snapshot,priority,status,max_attempts,"
                    "created_at,updated_at) VALUES (?,?,?,'m2_outfit_image',?,?,100,'pending',3,?,?)",
                    (job_id, project_id, run_id, json.dumps(payload, ensure_ascii=False),
                     json.dumps(settings, ensure_ascii=False), now, now))
                connection.execute("INSERT INTO portrait_expression VALUES (?,?,?,?,?,?,NULL,0,?)", (
                    identifier, project_id, body.character_id, base["id"], expression, job_id, now))
                started += 1
            if not started:
                raise ServiceError(409, "作成する表情がありません。すでに作成済みか、作成中です。")
            return {"project_id": project_id, "portraits": self.pictures(connection, project_id, [asked["id"]])}

    def usage(self, project_id):
        """How often the finished work asks each character for each expression, pictured or not."""
        from packages.narrative.validation import script_character_id

        from .adjustment_service import AdjustmentService

        with self.db.transaction() as connection:
            adjustments = AdjustmentService(self.coordinator)
            root, chapters = adjustments._context(connection, project_id)
            counts: dict[str, dict[str, int]] = {}
            if root is None:
                return {"project_id": project_id, "characters": counts}
            names = {script_character_id(row["id"]): row["id"]
                     for row in adjustments.m3._snapshot(connection, root)["characters"]}
            for chapter in chapters:
                narrative = adjustments.m3._load_narrative(connection, chapter)
                names.update({script_character_id(value.id): value.id for value in narrative.supporting_characters})
                script = json.loads(self.store.read(required(connection, "artifact", chapter["build"]["script_artifact_id"])))
                for direction in script["directions"]:
                    if direction["kind"] == "expression" and direction["expression"] != "neutral":
                        person = counts.setdefault(names.get(direction["character_id"], direction["character_id"]), {})
                        person[direction["expression"]] = person.get(direction["expression"], 0) + 1
            return {"project_id": project_id, "characters": counts}

    def remove(self, project_id, body: ExpressionRemoval):
        with self.db.transaction() as connection:
            base = self.portrait(connection, body.base_artifact_id)
            row = self._newest(connection, project_id, base).get(body.expression)
            if row is None:
                if body.expression in self._cast_pictures(connection, project_id, base):
                    raise ServiceError(409, "この表情はメインキャラの確認画面で作ったものです。そちらで削除してください。")
                raise ServiceError(404, "その表情の絵が見つかりません。")
            job = required(connection, "job", row["job_id"]) if row["job_id"] else None
            if job and job["status"] in {"pending", "running"}:
                raise ServiceError(409, "この表情は作成中です。")
            # A removal is recorded rather than erased: pictures a published chapter already shows stay.
            connection.execute("INSERT INTO portrait_expression VALUES (?,?,?,?,?,NULL,NULL,1,?)", (
                uuid4().hex, project_id, row["character_id"], base, body.expression, self.clock()))
            return {"project_id": project_id,
                    "portraits": self.pictures(connection, project_id, [body.base_artifact_id])}

    def validate_retry(self, connection, job):
        link = json.loads(job["payload"])["portrait_expression"]
        row = required(connection, "portrait_expression", link["id"])
        newest = self._newest(connection, row["project_id"], row["base_artifact_id"]).get(row["expression"])
        if newest is None or newest["id"] != row["id"]:
            raise ServiceError(409, "この表情は作り直されたため、以前の依頼は再試行できません。")

    def complete(self, job_id, worker_id, lease_id, data):
        from .m2_bundle import validate_bundle
        from .m2_service import M2Service

        with self.db.transaction() as connection:
            job, _ = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            payload = json.loads(job["payload"])
        try:
            envelope, files = validate_bundle(data, job["kind"], payload["character_id"])
            made = envelope["provenance"].get("outfit", {})
            if (made.get("reference_sha256") != payload["reference_image"]["sha256"]
                    or (made.get("expression") or {}).get("id") != payload["expression"]["id"]):
                raise ValueError("the picture is of another portrait or expression")
        except (ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile, EOFError, zlib.error,
                RuntimeError) as exc:
            raise ServiceError(422, "生成結果の表情・立ち絵の情報が入力と一致しません。") from exc
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            row = required(connection, "portrait_expression", payload["portrait_expression"]["id"])
            record = M2Service(self.coordinator)._artifact(
                connection, job["project_id"], "portrait-expression-" + row["id"], "m2_expression_image",
                "image.png", "image/png", files["image.png"], provenance=envelope["provenance"],
                job=job, attempt=attempt)
            bundle = self.coordinator._register_artifact(
                connection, job["project_id"], "portrait-expression-job-" + job_id, "m2_bundle", "m2-result.zip",
                "application/zip", self.store.put(data), source_job_id=job_id, source_attempt_id=attempt["id"])
            now = self.clock()
            connection.execute("UPDATE portrait_expression SET artifact_id=? WHERE id=?", (record["id"], row["id"]))
            connection.execute("UPDATE job_attempt SET status='completed',ended_at=? WHERE id=?", (now, attempt["id"]))
            connection.execute("UPDATE job SET status='completed',result_artifact_id=?,updated_at=?,error=NULL WHERE id=?",
                               (bundle["id"], now, job_id))
            connection.execute("UPDATE generation_run SET status='completed' WHERE id=?", (job["run_id"],))
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            return {"job": public(required(connection, "job", job_id)), "artifact": public(bundle)}


def router(coordinator):
    routes = APIRouter(prefix="/api/expressions", tags=["Expressions"])

    @routes.get("/projects/{project_id}")
    def project(project_id: str, base: str = ""):
        identifiers = [value for value in base.split(",") if value][:60]
        return ExpressionService(coordinator()).project(project_id, identifiers)

    @routes.get("/projects/{project_id}/usage")
    def usage(project_id: str):
        return ExpressionService(coordinator()).usage(project_id)

    @routes.post("/projects/{project_id}")
    def generate(project_id: str, body: ExpressionRequest):
        return ExpressionService(coordinator()).generate(project_id, body)

    @routes.post("/projects/{project_id}/remove")
    def remove(project_id: str, body: ExpressionRemoval):
        return ExpressionService(coordinator()).remove(project_id, body)

    return routes
