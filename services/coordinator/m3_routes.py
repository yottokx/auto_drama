"""M3 production progress, idempotent start, and bounded worker result adoption."""

import json
from collections.abc import Callable
from typing import Literal

from fastapi import APIRouter, Query, Request
from pydantic import BaseModel, ConfigDict
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from packages.contracts.m3 import PortraitSettingsUpdate, ProductionPlot, ProductionPortraits

from .m2_bundle import MAX_BUNDLE_BYTES
from .m3_service import M3Service
from .service import Coordinator, ServiceError


class StopRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    mode: Literal["graceful", "immediate"] = "graceful"


def router(coordinator: Callable[[], Coordinator]) -> APIRouter:
    routes = APIRouter(prefix="/api/m3", tags=["M3"])

    @routes.get("/projects/{project_id}")
    def project(project_id: str):
        return M3Service(coordinator()).project(project_id)

    @routes.get("/projects/{project_id}/plot", response_model=ProductionPlot)
    def plot(project_id: str):
        return M3Service(coordinator()).plot(project_id)

    @routes.post("/projects/{project_id}/start")
    def start(project_id: str):
        return M3Service(coordinator()).start(project_id)

    @routes.post("/projects/{project_id}/stop")
    def stop(project_id: str, request: StopRequest):
        return M3Service(coordinator()).stop(project_id, request.mode)

    @routes.post("/projects/{project_id}/resume")
    def resume(project_id: str):
        return M3Service(coordinator()).resume(project_id)

    @routes.get("/builds/{build_id}/next")
    def next_build(build_id: str):
        return Response(content=json.dumps(M3Service(coordinator()).next_build(build_id)),
                        media_type="application/json", headers={"Cache-Control": "no-store"})

    @routes.get("/builds/{build_id}/export")
    def export_chapters(build_id: str):
        return Response(content=M3Service(coordinator()).export_chapters(build_id),
                        media_type="application/zip",
                        headers={"Content-Disposition": 'attachment; filename="chapters.zip"',
                                 "Cache-Control": "no-store"})

    @routes.post("/projects/{project_id}/rebuild")
    def rebuild(project_id: str):
        return M3Service(coordinator()).rebuild(project_id)

    @routes.get("/projects/{project_id}/portraits", response_model=ProductionPortraits)
    def portraits(project_id: str):
        return M3Service(coordinator()).portraits(project_id)

    @routes.put("/projects/{project_id}/portraits")
    def update_portraits(project_id: str, request: PortraitSettingsUpdate):
        return M3Service(coordinator()).update_portraits(project_id, request)

    @routes.post("/jobs/{job_id}/complete")
    async def complete(
        job_id: str,
        request: Request,
        worker_id: str = Query(min_length=1),
        lease_id: str = Query(min_length=1),
    ):
        if request.headers.get("content-type", "").split(";", 1)[0] != "application/zip":
            raise ServiceError(415, "application/zip で送信してください。")
        content = bytearray()
        async for chunk in request.stream():
            content.extend(chunk)
            if len(content) > MAX_BUNDLE_BYTES:
                raise ServiceError(413, "各生成結果は32MiB以内にしてください。")
        service = coordinator()
        with service.db.transaction() as connection:
            from .service import required
            job = required(connection, "job", job_id)
            kind = job["kind"]
            adjustment = json.loads(job["payload"]).get("adjustment")
        if adjustment:
            from .adjustment_service import AdjustmentService
            complete_job = AdjustmentService(service).complete
        elif kind == "m3_plan":
            from .planning_service import PlanningService
            complete_job = PlanningService(service).complete
        else:
            complete_job = M3Service(service).complete
        return await run_in_threadpool(complete_job, job_id, worker_id, lease_id, bytes(content))

    return routes
