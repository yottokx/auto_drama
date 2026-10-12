"""What a work owes to the image editing model, so its output is never used unknowingly.

Whether that output needs care is a property of the model in use, not of this code: with a
different editing model the labels and the list below simply report nothing to mind.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import APIRouter

from packages.contracts.outfits import DEFAULT_OUTFIT

from .service import required

ROOT = Path(__file__).resolve().parents[2]
# Job kinds whose pictures come out of the editing model.
EDIT_KINDS = frozenset({"m2_outfit_image", "m3_event_cg"})
EDIT_MODEL = {"name": "Qwen-Image-2.1", "output_caution": True}


def edit_model(in_use: str | None = None) -> dict:
    """The editing model's registration; the generation config may replace it."""
    try:
        configured = json.loads((ROOT / "config/m2-generation.json").read_text(encoding="utf-8")).get("image_editing")
    except (OSError, ValueError):
        configured = None
    return {**EDIT_MODEL, **({"name": in_use} if in_use else {}),
            **(configured if isinstance(configured, dict) else {})}


def enabled(draft: dict) -> bool:
    """A work may forgo the editing model altogether: no looks, no dressing, no event CGs."""
    return draft.get("imageEditing", True) is not False


def _edited(connection, artifact_id) -> bool:
    if not artifact_id:
        return False
    artifact = required(connection, "artifact", artifact_id)
    approved = json.loads(artifact["provenance"] or "{}").get("approved_artifact_id")
    if approved:
        # A chapter publishes approved portraits under its own name; the origin is the approved one.
        return _edited(connection, approved)
    if not artifact["source_job_id"]:
        return False
    return required(connection, "job", artifact["source_job_id"])["kind"] in EDIT_KINDS


def usage(coordinator, connection, project_id: str) -> dict:
    from .m2_service import M2Service
    from .planning_service import PlanningService

    state = M2Service._load(connection, project_id)
    draft = state["draft"]
    portraits = []
    expressions = 0
    for character in draft["characters"]:
        name = (character.get("result") or character["input"]).get("name") or "キャラクター"
        # Every expression picture is an edit; they are counted rather than listed one by one.
        expressions += len(character.get("expressions", []))
        if _edited(connection, character.get("imageArtifactId")):
            portraits.append({"name": name, "look": "通常の姿"})
        for outfit in character.get("outfits", []):
            if outfit["id"] != DEFAULT_OUTFIT and _edited(connection, outfit.get("imageArtifactId")):
                portraits.append({"name": name, "look": outfit["name"]})
    planning = PlanningService(coordinator)
    row = planning.selected(connection, project_id)
    if row is not None:
        for (_cid, kind), (character, asset) in planning._assets(connection, row).items():
            if kind == "image" and asset and _edited(connection, asset["artifact_id"]):
                portraits.append({"name": character["name"], "look": "通常の姿"})
    selection = connection.execute(
        "SELECT production_id,edition_id FROM project_history_state WHERE project_id=?", (project_id,)).fetchone()
    if selection and selection["edition_id"]:
        # A finished work shows what its adjustments chose, which may differ from the cast review.
        names = {character["id"]: (character.get("result") or character["input"]).get("name") or "キャラクター"
                 for character in draft["characters"]}
        looks = {(character["id"], outfit["id"]): outfit["name"]
                 for character in draft["characters"] for outfit in character.get("outfits", [])}
        if row is not None and row["content"]:
            names.update({value["id"]: value["name"] for value in
                          json.loads(row["content"])["cast_plan"]["supporting_characters"]})
        if selection["production_id"]:
            # The names the work was produced with outlast later changes to the cast review.
            from .m3_service import M3Service

            snapshot = M3Service(coordinator)._snapshot(
                connection, required(connection, "m3_production", selection["production_id"]))
            for character in snapshot["characters"]:
                names[character["id"]] = character["result"]["name"]
                looks.update({(character["id"], outfit["id"]): outfit["name"] for outfit in character.get("outfits", [])})
        edition = required(connection, "publication_edition", selection["edition_id"])
        for value in json.loads(edition["state"])["characters"]:
            if not value.get("image_candidate_id"):
                continue
            candidate = required(connection, "adjustment_candidate", value["image_candidate_id"])
            if not _edited(connection, candidate["artifact_id"]):
                continue
            cid, _, oid = value["character_id"].partition("--")
            item = {"name": names.get(cid, cid), "look": looks.get((cid, oid), oid) if oid else "通常の姿"}
            if item not in portraits:
                portraits.append(item)
    event_cgs = 0
    if selection and selection["production_id"]:
        event_cgs = connection.execute(
            "SELECT COUNT(*) FROM m3_requirement r JOIN m3_production p ON p.id=r.production_id "
            "WHERE p.storyline_id=? AND r.kind='m3_event_cg' AND r.artifact_id IS NOT NULL",
            (selection["production_id"],)).fetchone()[0]
    from packages.contracts.event_cg import cg_model

    from .event_cg_settings import EventCgSettings

    # Name the model new pictures are drawn with; the standard one and Turbo share their terms of use.
    model = edit_model(cg_model(EventCgSettings(coordinator).settings(connection)["profile"].get("model", "standard"))["name"])
    return {"project_id": project_id, "enabled": enabled(draft), "model": model["name"],
            "caution": bool(model["output_caution"]), "portraits": portraits, "event_cgs": event_cgs,
            "expressions": expressions}


def router(coordinator):
    routes = APIRouter(prefix="/api/image-editing", tags=["Image editing"])

    @routes.get("/projects/{project_id}")
    def project(project_id: str):
        service = coordinator()
        with service.db.transaction() as connection:
            required(connection, "project", project_id)
            return usage(service, connection, project_id)

    return routes
