"""Start another work from a chosen step of an existing one.

Making a step again in place means deciding what of the later steps to throw away. A fork has
no such question: the new work carries what was settled up to the chosen step and has nothing
after it, and the work it came from stays exactly as it was. Stored files are shared by their
content, so carrying pictures and voices over adds records, not copies.
"""
from __future__ import annotations

import copy
import json
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter

from packages.contracts.outfits import DEFAULT_OUTFIT
from packages.contracts.script import Contract

from .m2_service import M2Service
from .service import Coordinator, ServiceError, required
from .storage import StoredObject

LEVELS = ("brief", "world", "cast", "plan")


class ForkRequest(Contract):
    # The last step whose results the new work starts with.
    through: Literal["brief", "world", "cast", "plan"]


class ForkService:
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.db, self.store, self.clock = coordinator.db, coordinator.store, coordinator.clock
        self.m2 = M2Service(coordinator)

    def _available(self, connection, project_id, state) -> list[str]:
        """How far this work has got, which is how much of it a fork can carry."""
        from .planning_service import PlanningService

        draft = state["draft"]
        levels = ["brief"]
        if not draft["worldResult"] or draft["worldPendingChanges"]:
            return levels
        levels.append("world")
        cast = draft["characters"]
        complete = draft["worldConfirmed"] and cast and all(
            row["result"] and not row["pendingChanges"] and row["imageArtifactId"] and row["voiceArtifactId"]
            and not row["imagePendingChanges"] and not row["voicePendingChanges"] for row in cast)
        relationships = draft["relationships"]
        if len(cast) >= 2 and (relationships["pendingChanges"] or not relationships["artifactId"]):
            complete = False
        if complete and self.m2._uses_looks(draft) and any(self.m2._stale_looks(row) for row in cast):
            complete = False
        if not complete or state["activeJobId"] or state["queue"]:
            return levels
        levels.append("cast")
        planning = PlanningService(self.coordinator).selected(connection, project_id)
        if draft["approved"] and planning and planning["content"] and planning["status"] in {"ready", "approved"}:
            levels.append("plan")
        return levels

    def options(self, project_id):
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            return {"project_id": project_id,
                    "levels": self._available(connection, project_id, M2Service._load(connection, project_id))}

    def fork(self, project_id, body: ForkRequest):
        from .material_service import MaterialService
        from .project_history import HistoryService

        level = LEVELS.index(body.through)
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            source_project = required(connection, "project", project_id)
            source = M2Service._load(connection, project_id)
            if body.through not in self._available(connection, project_id, source):
                raise ServiceError(409, "そのステップはまだ出来上がっていないため、引き継げません。")
            target = self.coordinator._create_project(
                connection, source_project["title"], source_project["instructions"], source_project["chapter_count"])
            copies: dict[str, dict] = {}

            def carry(artifact_id):
                """The same stored file as an artifact of the new work."""
                if not artifact_id:
                    return None
                if artifact_id not in copies:
                    record = required(connection, "artifact", artifact_id)
                    if record["project_id"] != project_id:
                        raise ServiceError(409, "引き継ぐ素材がもとの作品と一致しません。")
                    made = self.coordinator._register_artifact(
                        connection, target, record["logical_id"], record["kind"], record["filename"],
                        record["media_type"],
                        StoredObject(record["sha256"], record["size_bytes"], record["storage_key"]),
                        source_job_id=record["source_job_id"], source_attempt_id=record["source_attempt_id"])
                    connection.execute("UPDATE artifact SET provenance=? WHERE id=?", (record["provenance"], made["id"]))
                    copies[artifact_id] = required(connection, "artifact", made["id"])
                return copies[artifact_id]

            state = {"wizardStepVersion": source.get("wizardStepVersion", 3),
                     "appliedInstructions": copy.deepcopy(source.get("appliedInstructions", [])),
                     "draft": copy.deepcopy(source["draft"]), "activeJobId": None, "queue": []}
            draft = state["draft"]
            draft.update(revision=1, approved=False, approval=None, planningRequired=True, planApproved=False)
            sheets: dict[str, str] = {}
            materials = MaterialService(self.coordinator)
            if level == 0:
                # Only what the user asked for: the world, the cast and the materials are made anew.
                self.m2._discard_cast(state)
                draft.update(worldResult=None, worldArtifactId=None, worldVersion=None, worldConfirmed=False,
                             worldConfirmationArtifactId=None, worldPendingChanges=True, requests=[],
                             step="world-input")
                state["appliedInstructions"] = []
                materials.sync_brief(connection, target, draft.get("materials", []))
            else:
                world = carry(draft["worldArtifactId"])
                draft.update(worldArtifactId=world["id"], worldVersion=world["version"], worldPendingChanges=False)
                sheets = self._carry_materials(connection, project_id, target, carry)
            if level == 1:
                self.m2._discard_cast(state)
                draft.update(worldConfirmed=False, worldConfirmationArtifactId=None, step="world-review")
            if level >= 2:
                draft.update(worldConfirmed=True, step="character-review",
                             worldConfirmationArtifactId=carry(draft["worldConfirmationArtifactId"])["id"])
                state["appliedInstructions"] = copy.deepcopy(source.get("appliedInstructions", []))
                for character in draft["characters"]:
                    result = carry(character["resultArtifactId"])
                    character.update(resultArtifactId=result["id"], resultVersion=result["version"],
                                     imageArtifactId=carry(character["imageArtifactId"])["id"],
                                     voiceArtifactId=carry(character["voiceArtifactId"])["id"], voiceTests=[])
                    for outfit in character.get("outfits", []):
                        if outfit["id"] != DEFAULT_OUTFIT:
                            image = carry(outfit.get("imageArtifactId"))
                            outfit["imageArtifactId"] = image["id"] if image else None
                            # The look was derived from the portrait that is this work's base now.
                            outfit["baseImageArtifactId"] = (character["imageArtifactId"]
                                                             if outfit.get("baseImageArtifactId") else None)
                        for key in ("materialArtifactId", "plainArtifactId"):
                            if outfit.get(key):
                                outfit[key] = carry(outfit[key])["id"]
                        if outfit.get("materialImageId"):
                            outfit["materialImageId"] = sheets.get(outfit["materialImageId"])
                            if not outfit["materialImageId"]:
                                outfit.pop("materialImageId")
                    looks = {DEFAULT_OUTFIT: character["imageArtifactId"],
                             **{outfit["id"]: outfit.get("imageArtifactId") for outfit in character.get("outfits", [])
                                if outfit["id"] != DEFAULT_OUTFIT}}
                    pictures = []
                    for picture in character.pop("expressions", []):
                        # An expression comes along with the portrait it was edited from; one whose
                        # portrait had already been replaced is left behind.
                        base = copies.get(picture["baseImageArtifactId"])
                        if base is not None and base["id"] == looks.get(picture["outfitId"]):
                            pictures.append({**picture, "baseImageArtifactId": base["id"],
                                             "imageArtifactId": carry(picture["imageArtifactId"])["id"]})
                    if pictures:
                        character["expressions"] = pictures
                relationships = draft["relationships"]
                if relationships.get("artifactId"):
                    record = carry(relationships["artifactId"])
                    relationships.update(artifactId=record["id"], version=record["version"])
            connection.execute("INSERT INTO m2_draft VALUES (?,?,?)",
                               (target, json.dumps(state, ensure_ascii=False), self.clock()))
            HistoryService(self.coordinator).ensure(connection, target)
            policy = connection.execute("SELECT * FROM event_cg_policy WHERE project_id=?", (project_id,)).fetchone()
            if policy:
                connection.execute("INSERT INTO event_cg_policy VALUES (?,?,?,1)",
                                   (target, policy["max_cgs"], policy["max_variants_per_cg"]))
            if level == 3:
                self._carry_plan(connection, project_id, target, state, carry)
            self.m2._save(connection, target, state)
            return self.m2._view(connection, target, state)

    def _carry_materials(self, connection, project_id, target, carry) -> dict[str, str]:
        """The shared materials with their finished sheets; returns old sheet ids to new ones."""
        sheets = {}
        for material in connection.execute(
                "SELECT * FROM shared_material WHERE project_id=? AND removed=0 ORDER BY created_at,rowid",
                (project_id,)).fetchall():
            identifier = uuid4().hex
            connection.execute(
                "INSERT INTO shared_material (id,project_id,name,description,created_at,brief_id) VALUES (?,?,?,?,?,?)",
                (identifier, target, material["name"], material["description"], self.clock(), material["brief_id"]))
            for image in connection.execute(
                    "SELECT * FROM shared_material_image WHERE material_id=? AND removed=0 AND artifact_id IS NOT NULL "
                    "ORDER BY created_at,rowid", (material["id"],)).fetchall():
                source = json.loads(image["source"])
                if source.get("artifact_id"):
                    # The picture a sheet was made from may not come along; the sheet itself does.
                    source.pop("artifact_id")
                    source.pop("origin", None)
                new = uuid4().hex
                connection.execute(
                    "INSERT INTO shared_material_image (id,material_id,label,wish,mode,source,artifact_id,detail,created_at) "
                    "VALUES (?,?,?,?,?,?,?,?,?)",
                    (new, identifier, image["label"], image["wish"], image["mode"],
                     json.dumps(source, ensure_ascii=False), carry(image["artifact_id"])["id"], image["detail"],
                     self.clock()))
                sheets[image["id"]] = new
        return sheets

    def _carry_plan(self, connection, project_id, target, state, carry) -> None:
        """Approve the carried cast and hand its plan over as it stands, with the planned cast's assets."""
        from .planning_service import PlanningService, asset_source

        planning = PlanningService(self.coordinator)
        source = planning.selected(connection, project_id)
        self.m2._approve(connection, target, state)
        row = planning.selected(connection, target)
        # Approval asked for a new plan; the carried one takes its place, so that request is settled
        # here and its frozen settings serve as this plan's own.
        now = self.clock()
        connection.execute("UPDATE job SET status='completed',updated_at=? WHERE id=?", (now, row["active_job_id"]))
        connection.execute("UPDATE generation_run SET status='completed' WHERE id=(SELECT run_id FROM job WHERE id=?)",
                           (row["active_job_id"],))
        connection.execute("UPDATE planning_draft SET content=?,status='ready',active_job_id=NULL,revision=revision+1,"
                           "updated_at=? WHERE id=?", (source["content"], now, row["id"]))
        content = json.loads(source["content"])
        from .expression_service import ExpressionService

        for (cid, kind), (character, asset) in planning._assets(connection, source, content).items():
            if asset and asset["artifact_id"]:
                carried = carry(asset["artifact_id"])
                connection.execute("INSERT INTO planning_asset VALUES (?,?,?,?,?,NULL,?,?)", (
                    uuid4().hex, row["id"], cid, kind, asset_source(character, kind), carried["id"], now))
                if kind == "image":
                    # A supporting portrait brings the expressions that were made for it.
                    for expression, artifact_id in ExpressionService.current(
                            connection, project_id, [asset["artifact_id"]]).items():
                        connection.execute("INSERT INTO portrait_expression VALUES (?,?,?,?,?,NULL,?,0,?)", (
                            uuid4().hex, target, cid, carried["id"], expression, carry(artifact_id)["id"], now))
        state["draft"]["step"] = "planning-review"


def router(coordinator):
    routes = APIRouter(prefix="/api/forks", tags=["Forks"])

    @routes.get("/projects/{project_id}")
    def options(project_id: str):
        return ForkService(coordinator()).options(project_id)

    @routes.post("/projects/{project_id}")
    def fork(project_id: str, body: ForkRequest):
        return ForkService(coordinator()).fork(project_id, body)

    return routes
