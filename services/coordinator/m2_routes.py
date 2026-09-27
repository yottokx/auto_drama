"""M2 wizard API; M1 owns shared jobs, leases, workers and artifact downloads."""

from collections.abc import Callable

from fastapi import APIRouter, Query, Request
from starlette.concurrency import run_in_threadpool

from packages.contracts.m2 import M2Action, M2ProjectInput

from .m2_bundle import MAX_BUNDLE_BYTES
from .m2_service import M2Service
from .project_history import HistoryService, RestoreRequest
from .service import Coordinator, ServiceError


def router(coordinator: Callable[[], Coordinator]) -> APIRouter:
    routes = APIRouter(prefix="/api/m2", tags=["M2"])

    def service() -> M2Service:
        return M2Service(coordinator())

    @routes.get("/projects")
    def projects():
        return {"projects": service().projects()}

    @routes.post("/projects", status_code=201)
    def create_project(body: M2ProjectInput):
        return service().create(body)

    @routes.get("/projects/{project_id}")
    def project(project_id: str):
        return service().project(project_id)

    @routes.get("/projects/{project_id}/history")
    def history(project_id: str):
        return HistoryService(coordinator()).list(project_id)

    @routes.post("/projects/{project_id}/history/{revision_id}/restore")
    def restore(project_id: str, revision_id: str, body: RestoreRequest):
        return HistoryService(coordinator()).restore(project_id, revision_id, body)

    @routes.post("/projects/{project_id}/actions")
    def action(project_id: str, body: M2Action):
        return service().action(project_id, body)

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
                raise ServiceError(413, "M2の出力は32MiB以内にしてください。")
        return await run_in_threadpool(
            service().complete, job_id, worker_id, lease_id, bytes(content)
        )

    return routes
