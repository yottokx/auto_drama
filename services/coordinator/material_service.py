"""Shared materials: outfits and items several characters wear in the same design.

A material belongs to the work, not to a step: it can be added while the cast or the plan is
reviewed. Each of its images is a design sheet without a person, drawn from words, varied from
another sheet, or taken from a finished portrait. Which sheet a character wears is always chosen
by the user; nothing here assigns one.
"""
from __future__ import annotations

import json
import zipfile
import zlib
from typing import Literal
from uuid import uuid4

from fastapi import APIRouter
from pydantic import Field

from packages.contracts.outfits import DEFAULT_OUTFIT
from packages.contracts.script import Contract

from .m2_bundle import validate_bundle
from .m2_service import M2Service
from .service import Coordinator, ServiceError, public, required

TEXT_KIND, EDIT_KIND = "m2_material_image", "m2_outfit_image"
MAX_MATERIALS, MAX_IMAGES = 12, 6


class MaterialSource(Contract):
    """Where an image comes from when it is not drawn from words alone."""

    image_id: str | None = Field(default=None, max_length=64)
    character_id: str | None = Field(default=None, max_length=128)
    outfit_id: str | None = Field(default=None, max_length=64)
    planned: bool = False


class MaterialAction(Contract):
    action: Literal["add-material", "save-material", "delete-material", "add-image", "retake-image", "delete-image"]
    material_id: str | None = Field(default=None, max_length=64)
    image_id: str | None = Field(default=None, max_length=64)
    name: str | None = Field(default=None, max_length=40)
    description: str | None = Field(default=None, max_length=400)
    label: str | None = Field(default=None, max_length=40)
    wish: str | None = Field(default=None, max_length=400)
    mode: Literal["text", "derive", "extract"] = "text"
    source: MaterialSource | None = None


class MaterialService:
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.db, self.store, self.clock = coordinator.db, coordinator.store, coordinator.clock

    def _capable(self, connection, kind: str) -> bool:
        return any(kind in json.loads(worker["capabilities"]) for worker in connection.execute(
            "SELECT capabilities FROM worker WHERE last_seen_at>=?", (self.clock() - 90,)))

    def _material(self, connection, project_id, identifier):
        row = required(connection, "shared_material", identifier or "")
        if row["project_id"] != project_id or row["removed"]:
            raise ServiceError(404, "共通の素材が見つかりません。")
        return row

    def _image(self, connection, project_id, identifier):
        row = required(connection, "shared_material_image", identifier or "")
        if row["removed"]:
            raise ServiceError(404, "共通の素材の絵が見つかりません。")
        return row, self._material(connection, project_id, row["material_id"])

    def reference(self, connection, project_id, image_id) -> dict | None:
        """What a dressing job needs of a finished sheet; None while it is not usable."""
        row = connection.execute(
            "SELECT i.*,m.project_id,m.removed AS material_removed FROM shared_material_image i "
            "JOIN shared_material m ON m.id=i.material_id WHERE i.id=?", (image_id,)).fetchone()
        if (row is None or row["project_id"] != project_id or row["removed"] or row["material_removed"]
                or not row["artifact_id"] or not row["detail"]):
            return None
        artifact, detail = required(connection, "artifact", row["artifact_id"]), json.loads(row["detail"])
        return {"image_id": row["id"], "artifact_id": artifact["id"], "sha256": artifact["sha256"],
                "outfit_en": detail["outfit_en"], "summary_ja": detail["summary"], "category": detail["category"]}

    @staticmethod
    def _redesign(connection, image: dict) -> None:
        """The next drawing of this sheet decides its design anew; the present picture stays usable."""
        if image["detail"]:
            connection.execute("UPDATE shared_material_image SET detail=? WHERE id=?", (
                json.dumps({**json.loads(image["detail"]), "version": 0}, ensure_ascii=False), image["id"]))

    def _describe(self, connection, material: dict, name: str, description: str) -> None:
        if (name, description) != (material["name"], material["description"]):
            # Sheets that took their design from the material's own words follow its new words.
            for image in connection.execute(
                    "SELECT * FROM shared_material_image WHERE material_id=? AND removed=0 AND wish='' "
                    "AND mode='text'", (material["id"],)).fetchall():
                self._redesign(connection, dict(image))
        connection.execute("UPDATE shared_material SET name=?,description=? WHERE id=?",
                           (name, description, material["id"]))

    def sync_brief(self, connection, project_id, wishes: list[dict]) -> None:
        """Make the materials declared in the brief exist; their sheets are drawn once the world is known."""
        declared = {row["brief_id"]: dict(row) for row in connection.execute(
            "SELECT * FROM shared_material WHERE project_id=? AND removed=0 AND brief_id IS NOT NULL", (project_id,))}
        for wish in wishes:
            current = declared.pop(wish["id"], None)
            if current is None:
                connection.execute(
                    "INSERT INTO shared_material (id,project_id,name,description,created_at,brief_id) VALUES (?,?,?,?,?,?)",
                    (uuid4().hex, project_id, wish["name"], wish["description"], self.clock(), wish["id"]))
            else:
                self._describe(connection, current, wish["name"], wish["description"])
        for row in declared.values():
            connection.execute("UPDATE shared_material SET removed=1 WHERE id=?", (row["id"],))

    def ensure_sheets(self, connection, project_id, *, world=None, redraw=False) -> None:
        """Draw the first sheet of every declared material that has none, when a worker can.

        With ``redraw`` the world itself was made anew, so the sheets that were drawn from the
        material's own words are designed and drawn again for it.
        """
        if not self._capable(connection, TEXT_KIND):
            return
        if redraw:
            for image in connection.execute(
                    "SELECT i.*,m.id AS owner FROM shared_material_image i JOIN shared_material m ON m.id=i.material_id "
                    "WHERE m.project_id=? AND m.removed=0 AND m.brief_id IS NOT NULL AND i.removed=0 "
                    "AND i.mode='text' AND i.wish='' ORDER BY i.created_at,i.rowid", (project_id,)).fetchall():
                job = required(connection, "job", image["job_id"]) if image["job_id"] else None
                if job and job["status"] in {"pending", "running"}:
                    continue
                self._redesign(connection, dict(image))
                self._enqueue(connection, project_id, required(connection, "shared_material", image["owner"]),
                              required(connection, "shared_material_image", image["id"]), world=world)
        for material in connection.execute(
                "SELECT * FROM shared_material m WHERE project_id=? AND removed=0 AND brief_id IS NOT NULL "
                "AND NOT EXISTS (SELECT 1 FROM shared_material_image i WHERE i.material_id=m.id AND i.removed=0) "
                "ORDER BY created_at,rowid", (project_id,)).fetchall():
            self._add_image(connection, project_id, dict(material), label="", wish="", mode="text", source=None,
                            world=world)

    def brief_reference(self, connection, project_id, brief_id) -> dict | None:
        """The first finished sheet of a declared material, which is the one the brief means."""
        for row in connection.execute(
                "SELECT i.id FROM shared_material_image i JOIN shared_material m ON m.id=i.material_id "
                "WHERE m.project_id=? AND m.brief_id=? AND m.removed=0 AND i.removed=0 "
                "ORDER BY i.created_at,i.rowid", (project_id, brief_id)):
            found = self.reference(connection, project_id, row["id"])
            if found is not None:
                return found
        return None

    def _view(self, connection, project_id):
        materials = []
        for material in connection.execute(
                "SELECT * FROM shared_material WHERE project_id=? AND removed=0 ORDER BY created_at,rowid", (project_id,)):
            images = []
            for image in connection.execute(
                    "SELECT * FROM shared_material_image WHERE material_id=? AND removed=0 ORDER BY created_at,rowid",
                    (material["id"],)):
                job = required(connection, "job", image["job_id"]) if image["job_id"] else None
                working = bool(job and job["status"] in {"pending", "running"})
                detail = json.loads(image["detail"]) if image["detail"] else {}
                images.append({
                    "id": image["id"], "label": image["label"], "wish": image["wish"], "mode": image["mode"],
                    "status": ("working" if working else "failed" if job and job["status"] == "failed"
                               else "ready" if image["artifact_id"] else "missing"),
                    "url": f"/api/artifacts/{image['artifact_id']}/content" if image["artifact_id"] else None,
                    "summary": detail.get("summary"), "category": detail.get("category"),
                    "error": job["error"] if job and job["status"] == "failed" else None})
            materials.append({"id": material["id"], "name": material["name"], "brief_id": material["brief_id"],
                              "description": material["description"], "images": images})
        from .image_editing import edit_model

        return {"project_id": project_id, "materials": materials,
                "can_draw": self._capable(connection, TEXT_KIND),
                "can_edit": self._capable(connection, EDIT_KIND) and self._editing(connection, project_id),
                "edit_caution": bool(edit_model()["output_caution"])}

    @staticmethod
    def _editing(connection, project_id) -> bool:
        from .image_editing import enabled

        return enabled(M2Service._load(connection, project_id)["draft"])

    def project(self, project_id):
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            return self._view(connection, project_id)

    def _source(self, connection, project_id, mode, source):
        """The picture an edit starts from, and what is known about it in words."""
        if mode == "derive":
            found = self.reference(connection, project_id, source.image_id if source else None)
            if found is None:
                raise ServiceError(422, "もとにする絵を選んでください。")
            return found["artifact_id"], {"base_outfit_en": found["outfit_en"]}, {"image_id": found["image_id"]}
        if source is None or not source.character_id:
            raise ServiceError(422, "もとにする人物を選んでください。")
        if source.planned:
            from .planning_service import PlanningService

            planning = PlanningService(self.coordinator)
            row = planning.selected(connection, project_id)
            found = planning._assets(connection, row).get((source.character_id, "image")) if row else None
            if not found or not found[1] or not found[1]["artifact_id"]:
                raise ServiceError(409, "この人物の立ち絵がまだありません。")
            artifact, appearance = found[1]["artifact_id"], found[0]["appearance"]
        else:
            state = M2Service._load(connection, project_id)
            character = next((row for row in state["draft"]["characters"] if row["id"] == source.character_id), None)
            if character is None or not character.get("result"):
                raise ServiceError(422, "もとにする人物を選んでください。")
            outfit = next((row for row in character.get("outfits", [])
                           if row["id"] == (source.outfit_id or DEFAULT_OUTFIT)), None)
            if source.outfit_id in (None, DEFAULT_OUTFIT):
                artifact = character["imageArtifactId"]
                appearance = (outfit or {}).get("generatedDescription") or character["result"]["appearance"]
            else:
                artifact = (outfit or {}).get("imageArtifactId")
                appearance = (outfit or {}).get("generatedDescription") or (outfit or {}).get("name", "")
            if not artifact:
                raise ServiceError(409, "この人物の立ち絵がまだありません。")
        if required(connection, "artifact", artifact)["kind"] == "portrait_omission":
            raise ServiceError(409, "この人物には立ち絵がありません。")
        return artifact, {"appearance": appearance}, {
            "character_id": source.character_id, "outfit_id": source.outfit_id, "planned": source.planned}

    def _enqueue(self, connection, project_id, material, image, *, world=None):
        from .event_cg_settings import EventCgSettings
        from .llm_settings import execution_settings

        kind = TEXT_KIND if image["mode"] == "text" else EDIT_KIND
        if kind == EDIT_KIND and not self._editing(connection, project_id):
            raise ServiceError(409, "この作品は画像編集を使わない設定です。")
        if not self._capable(connection, kind):
            raise ServiceError(409, "立ち絵を作れるWorkerを起動してください。" if kind == TEXT_KIND
                               else "絵をもとにした作成には、画像編集の準備ができたWorkerが必要です。")
        state = M2Service._load(connection, project_id)
        source = json.loads(image["source"])
        sheet = {"mode": image["mode"], "name": material["name"], "label": image["label"],
                 # What a portrait wears is taken as drawn; the material's own words would overrule it.
                 "wish": image["wish"] or ("" if image["mode"] == "extract" else material["description"]),
                 **source.get("words", {})}
        if image["detail"]:
            # Drawing again keeps what the sheet was decided to be; a changed wish clears this first.
            sheet["design"] = json.loads(image["detail"])
        identifier, run_id, now = uuid4().hex, uuid4().hex, self.clock()
        profile = M2Service(self.coordinator)._profile(connection)
        payload = {"schema_version": 1, "seed": int(identifier[:8], 16) & 0x7FFFFFFF, "character_id": None,
                   # A caller inside an unsaved world update passes the world it is about to save.
                   "world_result": world if world is not None else state["draft"].get("worldResult"),
                   "profile": profile,
                   "material_sheet": sheet, "material_image_job": {"image_id": image["id"]}}
        if kind == EDIT_KIND:
            artifact = required(connection, "artifact", source["artifact_id"])
            payload["reference_image"] = {"artifact_id": artifact["id"], "sha256": artifact["sha256"]}
            payload["cg_profile"] = EventCgSettings(self.coordinator).settings(connection)["profile"]
        settings = {"schema_version": 1, "seed": payload["seed"],
                    **execution_settings(self.coordinator, connection, payload, kind=kind, profile=profile)}
        project = required(connection, "project", project_id)
        connection.execute(
            "INSERT INTO generation_run(id,project_id,settings_version,story_revision_id,policy,status,created_at) "
            "VALUES (?,?,?,?,?,'pending',?)", (run_id, project_id, project["settings_version"], material["id"],
                json.dumps({"mode": "shared-material"}), now))
        connection.execute(
            "INSERT INTO job(id,project_id,run_id,kind,payload,settings_snapshot,priority,status,max_attempts,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,100,'pending',3,?,?)",
            (identifier, project_id, run_id, kind, json.dumps(payload, ensure_ascii=False),
             json.dumps(settings, ensure_ascii=False), now, now))
        connection.execute("UPDATE shared_material_image SET job_id=? WHERE id=?", (identifier, image["id"]))

    def _add_image(self, connection, project_id, material, *, label, wish, mode, source, world=None):
        count = connection.execute("SELECT COUNT(*) FROM shared_material_image WHERE material_id=? AND removed=0",
                                   (material["id"],)).fetchone()[0]
        if count >= MAX_IMAGES:
            raise ServiceError(422, f"1つの素材に持てる絵は{MAX_IMAGES}枚までです。")
        stored = {}
        if mode != "text":
            artifact, words, origin = self._source(connection, project_id, mode, source)
            stored = {"artifact_id": artifact, "words": words, "origin": origin}
        identifier = uuid4().hex
        connection.execute("INSERT INTO shared_material_image (id,material_id,label,wish,mode,source,created_at) "
                           "VALUES (?,?,?,?,?,?,?)", (identifier, material["id"], (label or "").strip(),
                               (wish or "").strip(), mode, json.dumps(stored, ensure_ascii=False), self.clock()))
        self._enqueue(connection, project_id, material, required(connection, "shared_material_image", identifier),
                      world=world)

    def action(self, project_id, body: MaterialAction):
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            required(connection, "project", project_id)
            if body.action == "add-material":
                name = (body.name or "").strip()
                if not name:
                    raise ServiceError(422, "素材の呼び名を入力してください。")
                if connection.execute("SELECT COUNT(*) FROM shared_material WHERE project_id=? AND removed=0",
                                      (project_id,)).fetchone()[0] >= MAX_MATERIALS:
                    raise ServiceError(422, f"共通の素材は{MAX_MATERIALS}個までです。")
                identifier = uuid4().hex
                connection.execute("INSERT INTO shared_material (id,project_id,name,description,created_at) VALUES (?,?,?,?,?)",
                                   (identifier, project_id, name, (body.description or "").strip(), self.clock()))
                # The first sheet comes with the material; from words unless a source was chosen.
                self._add_image(connection, project_id, required(connection, "shared_material", identifier),
                                label=body.label, wish=body.wish, mode=body.mode, source=body.source)
            elif body.action in {"save-material", "delete-material", "add-image"}:
                material = self._material(connection, project_id, body.material_id)
                if body.action == "save-material":
                    name = (body.name or "").strip()
                    if not name:
                        raise ServiceError(422, "素材の呼び名を入力してください。")
                    self._describe(connection, material, name, (body.description or "").strip())
                elif body.action == "delete-material":
                    connection.execute("UPDATE shared_material SET removed=1 WHERE id=?", (material["id"],))
                else:
                    self._add_image(connection, project_id, material, label=body.label, wish=body.wish,
                                    mode=body.mode, source=body.source)
            else:
                image, material = self._image(connection, project_id, body.image_id)
                if body.action == "delete-image":
                    connection.execute("UPDATE shared_material_image SET removed=1 WHERE id=?", (image["id"],))
                else:
                    job = required(connection, "job", image["job_id"]) if image["job_id"] else None
                    if job and job["status"] in {"pending", "running"}:
                        raise ServiceError(409, "この絵は作成中です。")
                    if body.wish is not None or body.label is not None:
                        wish = (body.wish if body.wish is not None else image["wish"]).strip()
                        label = (body.label if body.label is not None else image["label"]).strip()
                        # A different wish asks for a different design, not for the same one drawn again.
                        if (wish, label) != (image["wish"], image["label"]):
                            self._redesign(connection, image)
                        connection.execute("UPDATE shared_material_image SET wish=?,label=? WHERE id=?",
                                           (wish, label, image["id"]))
                    self._enqueue(connection, project_id, material,
                                  required(connection, "shared_material_image", image["id"]))
            return self._view(connection, project_id)

    def validate_retry(self, connection, job):
        link = json.loads(job["payload"])["material_image_job"]
        image = required(connection, "shared_material_image", link["image_id"])
        if image["removed"] or image["job_id"] != job["id"]:
            raise ServiceError(409, "この絵は削除されたか、作り直されています。")

    def complete(self, job_id, worker_id, lease_id, data):
        with self.db.transaction() as connection:
            job, _ = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            payload = json.loads(job["payload"])
        try:
            envelope, files = validate_bundle(data, job["kind"], None)
            detail = envelope["provenance"]["material"]
            if (detail["mode"] != payload["material_sheet"]["mode"] or detail["category"] not in {"outfit", "item"}
                    or not detail["outfit_en"].strip() or not detail["summary"].strip()):
                raise ValueError("the sheet differs from its request")
            reference = payload.get("reference_image")
            if reference and detail.get("reference_sha256") != reference["sha256"]:
                raise ValueError("the sheet was made from another picture")
        except (ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile, EOFError, zlib.error,
                RuntimeError) as exc:
            raise ServiceError(422, "生成結果が依頼した素材と一致しません。") from exc
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            image = required(connection, "shared_material_image", payload["material_image_job"]["image_id"])
            store = M2Service(self.coordinator)._artifact
            record = store(connection, job["project_id"], "shared-material-" + image["id"], "shared_material",
                           "image.png", "image/png", files["image.png"], provenance=envelope["provenance"],
                           job=job, attempt=attempt)
            bundle = self.coordinator._register_artifact(connection, job["project_id"],
                "shared-material-job-" + job_id, "m2_bundle", "result.zip", "application/zip", self.store.put(data),
                source_job_id=job_id, source_attempt_id=attempt["id"])
            now = self.clock()
            if image["job_id"] == job_id:
                connection.execute("UPDATE shared_material_image SET artifact_id=?,detail=? WHERE id=?", (
                    record["id"], json.dumps({key: detail[key] for key in
                                              ("category", "outfit_en", "summary", "change_en", "version")
                                              if key in detail}, ensure_ascii=False), image["id"]))
            connection.execute("UPDATE job_attempt SET status='completed',ended_at=? WHERE id=?", (now, attempt["id"]))
            connection.execute("UPDATE job SET status='completed',result_artifact_id=?,updated_at=?,error=NULL WHERE id=?",
                               (bundle["id"], now, job_id))
            connection.execute("UPDATE generation_run SET status='completed' WHERE id=?", (job["run_id"],))
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            # A look that was waiting for this sheet can be drawn now.
            M2Service(self.coordinator).material_ready(connection, job["project_id"])
            return {"job": public(required(connection, "job", job_id)), "artifact": public(bundle)}


def router(coordinator):
    routes = APIRouter(prefix="/api/materials", tags=["Materials"])

    @routes.get("/projects/{project_id}")
    def project(project_id: str):
        return MaterialService(coordinator()).project(project_id)

    @routes.post("/projects/{project_id}/actions")
    def action(project_id: str, body: MaterialAction):
        return MaterialService(coordinator()).action(project_id, body)

    return routes
