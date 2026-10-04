"""Read-only, edition-pinned material for the standalone image experiment GUI."""

from __future__ import annotations

from collections.abc import Callable

from fastapi import APIRouter

from .m3_service import M3Service
from .service import Coordinator, public, required


def image_catalog(coordinator: Coordinator, project_id: str) -> dict:
    # Deliberately do not call project(), recover(), advance(), or settle_control().
    # All chapter and portrait selections come from one database snapshot.
    service = M3Service(coordinator)
    with coordinator.db.transaction() as connection:
        project = public(required(connection, "project", project_id))
        root = service._selected_production(connection, project_id)
        if root is None:
            return {"project_id": project_id, "project": project,
                    "production": None, "portraits": []}
        selection = service.history.selection(connection, project_id)
        production = public(dict(root))
        production["history_frozen"] = bool(selection and selection["production_frozen"])
        production["chapters"] = service._chapters(connection, root)
        portraits = {}
        for chapter in production["chapters"]:
            build = chapter.get("build")
            if not build:
                continue
            row = required(connection, "m3_production", chapter["production_id"])
            for value in service._published_portraits(connection, row, build):
                key = (value["character_id"], value["image_artifact_id"])
                if key not in portraits:
                    artifact = required(connection, "artifact", value["image_artifact_id"])
                    portraits[key] = {**value, "artifact_id": value["image_artifact_id"],
                        "source": {"kind": "published_portrait", "project_id": project_id,
                            "storyline_id": root["id"], "production_id": row["id"],
                            "build_id": build["id"], "artifact_id": artifact["id"],
                            "sha256": artifact["sha256"],
                            "edition_id": selection.get("edition_id") if selection else None,
                            "history_frozen": production["history_frozen"]},
                        "chapter_numbers": []}
                portraits[key]["chapter_numbers"].append(chapter["chapter_number"])
        return {"project_id": project_id, "project": project,
                "production": production, "portraits": list(portraits.values())}


def router(coordinator: Callable[[], Coordinator]) -> APIRouter:
    routes = APIRouter(prefix="/api/image-experiments", tags=["Image experiments"])

    @routes.get("/projects/{project_id}")
    def project(project_id: str):
        return image_catalog(coordinator(), project_id)

    return routes
