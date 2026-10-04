"""Presentation candidates and complete-story adjustment publication APIs."""
from collections.abc import Callable

from fastapi import APIRouter, Request
from starlette.concurrency import run_in_threadpool
from starlette.exceptions import HTTPException

from packages.contracts.adjustments import (
    AdjustmentGenerate,
    AdjustmentMusicGenerate,
    AdjustmentRevision,
    AdjustmentSample,
    AdjustmentSave,
    AdjustmentStart,
)
from packages.contracts.music import MusicReplanRequest

from .adjustment_service import LIMITS, AdjustmentService
from .service import Coordinator, ServiceError


def router(coordinator: Callable[[], Coordinator]):
    routes = APIRouter(prefix="/api/m3/projects/{project_id}/adjustments", tags=["Adjustments"])

    @routes.get("")
    def project(project_id: str):
        return AdjustmentService(coordinator()).project(project_id)

    @routes.post("/start")
    def start(project_id: str, body: AdjustmentStart):
        return AdjustmentService(coordinator()).start(project_id, body)

    @routes.put("")
    def save(project_id: str, body: AdjustmentSave):
        return AdjustmentService(coordinator()).save(project_id, body)

    @routes.post("/preview")
    def preview(project_id: str, body: AdjustmentSave):
        return AdjustmentService(coordinator()).save(project_id, body, preview=True)

    @routes.post("/generate")
    def generate(project_id: str, body: AdjustmentGenerate):
        return AdjustmentService(coordinator()).generate(project_id, body)

    @routes.post("/music/generate")
    def generate_music(project_id: str, body: AdjustmentMusicGenerate):
        from .music_adjustments import MusicAdjustmentService

        return MusicAdjustmentService(coordinator()).generate(project_id, body)

    @routes.post("/music/upload")
    async def upload_music(project_id: str, request: Request):
        try:
            size = int(request.headers.get("content-length", "0"))
        except ValueError as exc:
            raise ServiceError(422, "アップロードのサイズが不正です。") from exc
        if size > LIMITS["upload_bytes"] + 65536:
            raise ServiceError(413, "アップロードは32MiB以内にしてください。")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > LIMITS["upload_bytes"] + 65536:
                raise ServiceError(413, "アップロードは32MiB以内にしてください。")
        request._body = bytes(data)
        try:
            form = await request.form(max_files=1, max_fields=3)
        except (HTTPException, ValueError) as exc:
            raise ServiceError(422, "アップロードのフォームを読み取れませんでした。") from exc
        try:
            revision = int(form.get("expected_revision", ""))
            pid, sid, file = form.get("production_id"), form.get("scene_id"), form.get("file")
            if revision < 1 or not isinstance(pid, str) or not isinstance(sid, str) or not hasattr(file, "read"):
                raise ValueError("章・場面と音楽ファイルを指定してください。")
            original = await file.read(LIMITS["upload_bytes"] + 1)
            if len(original) > LIMITS["upload_bytes"]:
                raise ServiceError(413, "アップロードは32MiB以内にしてください。")
            from .music_adjustments import MusicAdjustmentService
            from .music_import import normalize_music

            music = MusicAdjustmentService(coordinator())
            await run_in_threadpool(music.validate_upload, project_id, revision, pid, sid)
            files, result = await run_in_threadpool(normalize_music, original, sid)
            return await run_in_threadpool(music.upload,
                                          project_id, revision, pid, sid, files, result)
        except (ValueError, TypeError) as exc:
            raise ServiceError(422, str(exc)) from exc
        finally:
            await form.close()

    @routes.post("/music/replan")
    def replan_music(project_id: str, body: MusicReplanRequest):
        from .music_adjustments import MusicAdjustmentService

        return MusicAdjustmentService(coordinator()).replan(project_id, body)

    @routes.post("/sample")
    def sample(project_id: str, body: AdjustmentSample):
        return AdjustmentService(coordinator()).sample(project_id, body)

    @routes.post("/apply")
    def apply(project_id: str, body: AdjustmentRevision):
        return AdjustmentService(coordinator()).apply(project_id, body)

    @routes.post("/retry")
    def retry(project_id: str, body: AdjustmentRevision):
        return AdjustmentService(coordinator()).retry(project_id, body)

    @routes.post("/upload")
    async def upload(project_id: str, request: Request):
        # Bound the whole multipart request before parsing or decoding it.
        try:
            content_length = int(request.headers.get("content-length", "0"))
        except ValueError as exc:
            raise ServiceError(422, "アップロードのサイズが不正です。") from exc
        if content_length > LIMITS["upload_bytes"] + 65536:
            raise ServiceError(413, "アップロードは32MiB以内にしてください。")
        data = bytearray()
        async for chunk in request.stream():
            data.extend(chunk)
            if len(data) > LIMITS["upload_bytes"] + 65536:
                raise ServiceError(413, "アップロードは32MiB以内にしてください。")
        request._body = bytes(data)
        try:
            form = await request.form(max_files=1, max_fields=5)
        except (HTTPException, ValueError) as exc:
            raise ServiceError(422, "アップロードのフォームを読み取れませんでした。") from exc
        file = form.get("file")
        try:
            revision = int(form.get("expected_revision", ""))
            cid, kind = form.get("character_id"), form.get("kind")
            reference_text = form.get("reference_text")
            if (revision < 1 or kind not in {"image", "voice"} or not isinstance(cid, str)
                    or not hasattr(file, "read") or (reference_text is not None
                        and (not isinstance(reference_text, str) or len(reference_text) > 2000))):
                raise ValueError("人物・素材の種類とファイルを指定してください。")
            original = await file.read(LIMITS["upload_bytes"] + 1)
            if len(original) > LIMITS["upload_bytes"]:
                raise ServiceError(413, "アップロードは32MiB以内にしてください。")
            from .adjustment_media import normalize_portrait, normalize_reference_voice

            normalized, metadata = await run_in_threadpool(
                normalize_portrait if kind == "image" else normalize_reference_voice, original)
            return await run_in_threadpool(AdjustmentService(coordinator()).upload,
                project_id, revision, cid, kind, original, normalized, metadata, reference_text)
        except (ValueError, TypeError) as exc:
            raise ServiceError(422, str(exc)) from exc
        finally:
            await form.close()

    return routes
