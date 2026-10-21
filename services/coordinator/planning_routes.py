"""STEP4 common-planning review and approval API."""
from typing import Literal

from fastapi import APIRouter
from pydantic import Field

from packages.contracts.script import Contract

from .planning_service import PlanningService


class PlanningAction(Contract):
    # "plot": plan a long-form plan's plot from its cast at chapter_count chapters.
    action: Literal["generate", "revise", "save", "approve", "plot"]
    expected_revision: int = Field(ge=0, strict=True)
    expected_event_cg_policy_revision: int | None = Field(default=None, ge=0, strict=True)
    reconfirm: bool = Field(default=False, strict=True)
    content: dict | None = None
    target: Literal["all", "plot", "character", "relationships"] = "all"
    character_id: str | None = Field(default=None, max_length=100)
    chapter_number: int | None = Field(default=None, ge=1, le=100, strict=True)
    instruction: str | None = Field(default=None, max_length=30000)
    chapter_count: int | None = Field(default=None, ge=1, le=100, strict=True)


class AssetRetake(Contract):
    expected_revision: int = Field(ge=0, strict=True)
    character_id: str = Field(min_length=1, max_length=100)
    kind: Literal["image", "voice"]
    instruction: str | None = Field(default=None, max_length=2000)


class AssetDress(Contract):
    expected_revision: int = Field(ge=0, strict=True)
    character_id: str = Field(min_length=1, max_length=100)
    material_image_id: str = Field(min_length=1, max_length=64)


def router(coordinator):
    routes = APIRouter(prefix="/api/planning", tags=["Planning"])

    @routes.get("/projects/{project_id}")
    def project(project_id: str):
        return PlanningService(coordinator()).project(project_id)

    @routes.post("/projects/{project_id}/actions")
    def action(project_id: str, body: PlanningAction):
        return PlanningService(coordinator()).action(project_id, body)

    @routes.post("/projects/{project_id}/assets")
    def retake_asset(project_id: str, body: AssetRetake):
        return PlanningService(coordinator()).retake_asset(project_id, body)

    @routes.post("/projects/{project_id}/assets/dress")
    def dress_asset(project_id: str, body: AssetDress):
        return PlanningService(coordinator()).dress_asset(project_id, body)

    return routes
