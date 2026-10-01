"""M1 coordinator: persistent local API. Launch on loopback only."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator
from starlette.concurrency import run_in_threadpool

from packages.contracts import Script
from packages.contracts.llm_settings import LLMCapability, LLMSettings

from .service import Coordinator, ServiceError

ROOT = Path(__file__).resolve().parents[2]
MAX_RESULT_BYTES = 32 * 1024 * 1024
Label = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=120)]


class InputModel(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="after")
    @classmethod
    def valid_text(cls, value):
        values = value if isinstance(value, list) else [value]
        for item in values:
            if isinstance(item, str) and any(
                (ord(char) < 32 and char not in "\t\n\r") or 0xD800 <= ord(char) <= 0xDFFF
                for char in item
            ):
                raise ValueError("control characters and unpaired surrogates are not supported")
        return value


class ProjectInput(InputModel):
    title: Label
    instructions: str = Field(default="", max_length=10000)
    chapter_count: int = Field(default=1, ge=1, le=100)


class DemoInput(InputModel):
    title: Label = "灯台の約束"


class WorkerInput(InputModel):
    name: Label
    capabilities: list[Label] = Field(default_factory=lambda: ["tyrano_export"], max_length=32)
    llm_models: list[LLMCapability] = Field(default_factory=list, max_length=256)


class WorkerCapabilitiesInput(InputModel):
    capabilities: list[Label] = Field(max_length=32)
    llm_models: list[LLMCapability] = Field(max_length=256)


class LeaseInput(InputModel):
    worker_id: Label
    lease_id: Label


class FailureInput(LeaseInput):
    error: str = Field(min_length=1, max_length=2000)


class JobInput(InputModel):
    script_artifact_id: Label
    depends_on: list[Label] = Field(default_factory=list, max_length=100)
    priority: int = Field(default=0, ge=-100, le=100)
    max_attempts: int = Field(default=3, ge=1, le=10)


def create_app(
    data_dir: Path | str | None = None,
    *,
    lease_seconds: float = 60,
    coordinator: Coordinator | None = None,
) -> FastAPI:
    directory = Path(data_dir or os.environ.get("AUTO_DRAMA_DATA_DIR", ROOT / "data"))
    if not directory.is_absolute():
        directory = ROOT / directory

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        service = coordinator or Coordinator(directory, lease_seconds=lease_seconds)
        await run_in_threadpool(service.initialize)
        app.state.coordinator = service
        yield

    app = FastAPI(title="AI Auto Drama", version="0.4.0", lifespan=lifespan)

    def service() -> Coordinator:
        return app.state.coordinator

    from .m2_routes import router as m2_router
    from .m3_player import install_player_routes
    from .m3_routes import router as m3_router
    from .tts_settings import register_routes as register_tts_routes

    app.include_router(m2_router(service))
    app.include_router(m3_router(service))
    install_player_routes(app, service)
    register_tts_routes(app, service)

    @app.exception_handler(ServiceError)
    async def service_error(request: Request, exc: ServiceError):
        return JSONResponse(status_code=exc.status, content={"detail": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def invalid_input(request: Request, exc: RequestValidationError):
        # Do not echo invalid request values (including unpaired surrogates or
        # secret-looking fields) into an error response or application log.
        return JSONResponse(
            status_code=422,
            content={
                "detail": "入力の形式・文字・参照を確認してください。API仕様は /docs で確認できます。"
            },
        )

    @app.exception_handler(OSError)
    async def storage_error(request: Request, exc: OSError):
        return JSONResponse(
            status_code=503,
            content={
                "detail": "保存済みファイルの読み書き・整合性を確認できません。保存先を確認してください。"
            },
        )

    @app.get("/api/health")
    def health():
        return {"status": "ok", "service": "coordinator", "stage": "m4", "schema_version": 4}

    @app.get("/api/projects")
    def projects():
        return {"projects": service().projects()}

    @app.post("/api/projects", status_code=201)
    def create_project(body: ProjectInput):
        return service().create_project(**body.model_dump())

    @app.post("/api/projects/demo", status_code=201)
    def create_demo(body: DemoInput):
        return service().create_demo(body.title)

    @app.get("/api/projects/{project_id}")
    def project(project_id: str):
        return service().project(project_id)

    @app.post("/api/projects/{project_id}/scripts", status_code=201)
    def save_script(project_id: str, body: Script):
        return service().save_script(project_id, body)

    @app.post("/api/projects/{project_id}/jobs", status_code=201)
    def enqueue(project_id: str, body: JobInput):
        return service().enqueue(
            project_id,
            body.script_artifact_id,
            list(dict.fromkeys(body.depends_on)),
            body.priority,
            body.max_attempts,
        )

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        return service().job(job_id)

    @app.post("/api/jobs/{job_id}/retry")
    def retry(job_id: str):
        return service().retry(job_id)

    @app.get("/api/artifacts/{artifact_id}")
    def artifact(artifact_id: str):
        return service().artifact(artifact_id)[0]

    @app.get("/api/artifacts/{artifact_id}/content")
    def content(artifact_id: str):
        record, data = service().artifact(artifact_id)
        return Response(
            data,
            media_type=record["media_type"],
            headers={
                "Content-Disposition": f'attachment; filename="{record["filename"]}"',
                "ETag": f'"{record["sha256"]}"',
                "X-Content-Type-Options": "nosniff",
            },
        )

    @app.get("/api/workers")
    def workers():
        return {"workers": service().workers()}

    @app.post("/api/workers", status_code=201)
    def register_worker(body: WorkerInput):
        return service().register_worker(body.name, body.capabilities,
                                         [model.model_dump() for model in body.llm_models])

    @app.post("/api/workers/{worker_id}/capabilities")
    def update_worker_capabilities(worker_id: str, body: WorkerCapabilitiesInput):
        return service().update_worker_capabilities(worker_id, body.capabilities,
            [model.model_dump() for model in body.llm_models])

    @app.get("/api/settings/llm")
    def llm_settings():
        from .llm_settings import read_settings
        return read_settings(service())

    @app.post("/api/settings/llm")
    def update_llm_settings(body: LLMSettings):
        from .llm_settings import save_settings
        return save_settings(service(), body)

    @app.post("/api/workers/{worker_id}/claim")
    def claim(worker_id: str):
        return {"job": service().claim(worker_id)}

    @app.post("/api/jobs/{job_id}/heartbeat")
    def heartbeat(job_id: str, body: LeaseInput):
        return service().heartbeat(job_id, body.worker_id, body.lease_id)

    @app.post("/api/jobs/{job_id}/fail")
    def fail(job_id: str, body: FailureInput):
        return service().fail(job_id, body.worker_id, body.lease_id, body.error)

    @app.post("/api/jobs/{job_id}/complete")
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
            if len(content) > MAX_RESULT_BYTES:
                raise ServiceError(413, "M1の出力は32MiB以内にしてください。")
        return await run_in_threadpool(
            service().complete, job_id, worker_id, lease_id, bytes(content)
        )

    return app


app = create_app()
