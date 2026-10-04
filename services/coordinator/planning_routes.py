"""STEP4 common-planning review and approval API."""
from typing import Literal

from fastapi import APIRouter
from pydantic import Field

from packages.contracts.script import Contract

from .planning_service import PlanningService


class PlanningAction(Contract):
    action: Literal["generate", "revise", "save", "approve"]
    expected_revision: int = Field(ge=0, strict=True)
    content: dict | None = None
    target: Literal["all", "plot", "character", "relationships"] = "all"
    character_id: str | None = Field(default=None, max_length=100)
    chapter_number: int | None = Field(default=None, ge=1, le=100, strict=True)
    instruction: str | None = Field(default=None, max_length=30000)


def router(coordinator):
    routes = APIRouter(prefix="/api/planning", tags=["Planning"])

    @routes.get("/projects/{project_id}")
    def project(project_id: str):
        return PlanningService(coordinator()).project(project_id)

    @routes.post("/projects/{project_id}/actions")
    def action(project_id: str, body: PlanningAction):
        return PlanningService(coordinator()).action(project_id, body)

    return routes
