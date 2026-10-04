"""Leased HTTP worker protocol for export and isolated M2 generation."""

from __future__ import annotations

import hashlib
import io
import logging
import math
import re
import threading
import time
import wave
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal, Self

import httpx

from packages.contracts import Script
from packages.contracts.m3 import M3_KINDS
from packages.tyrano_export import compile_bundle
from services.worker.generation.cancellation import (
    CancellationToken,
    GenerationCancelled,
    cancellation_scope,
    check_cancelled,
)
from services.worker.generation.event_cg_session import current_session as current_event_cg_session
from services.worker.generation.image_session import current_session as current_image_session
from services.worker.generation.llm_session import current_session as current_llm_session
from services.worker.generation.music_session import current_session as current_music_session
from services.worker.generation.progress import progress_scope
from services.worker.generation.voice_session import current_session

logger = logging.getLogger(__name__)
RunResult = Literal["idle", "completed", "failed", "stale", "deferred"]
M2_JOB_KINDS = (
    "m2_world",
    "m2_character",
    "m2_image",
    "m2_voice",
    "m2_relationships",
    "m2_voice_clone",
)
GENERATION_JOB_KINDS = (*M2_JOB_KINDS, *M3_KINDS)


class LeaseLost(Exception):
    """The coordinator no longer accepts this attempt."""


class _LeaseHeartbeat:
    """Keep the lease alive while fetching inputs and compiling the bundle."""

    def __init__(self, worker: WorkerClient, job: dict[str, Any], cancellation: CancellationToken):
        self.worker = worker
        self.job = job
        self.cancellation = cancellation
        self.stopped = threading.Event()
        self.error: Exception | None = None
        self.thread: threading.Thread | None = None
        self.interval = worker.heartbeat_interval

    def __enter__(self) -> Self:
        response = self.worker._heartbeat(self.job)
        expires_at = response.get("lease_expires_at", self.job.get("lease_expires_at"))
        if expires_at:
            expiry = datetime.fromisoformat(expires_at)
            if expiry.tzinfo is None:
                expiry = expiry.replace(tzinfo=UTC)
            remaining = (expiry - datetime.now(UTC)).total_seconds()
            self.interval = min(self.interval, max(0.05, remaining / 3))
        self.thread = threading.Thread(target=self._run, name="worker-heartbeat", daemon=True)
        self.thread.start()
        return self

    def _run(self) -> None:
        while not self.stopped.wait(self.interval):
            try:
                self.worker._heartbeat(self.job)
            except Exception as error:  # noqa: BLE001 - forward thread errors to the job executor
                self.error = error
                if isinstance(error, LeaseLost):
                    self.cancellation.cancel()
                return

    def check(self) -> None:
        if self.error is not None:
            raise self.error

    def __exit__(self, *_: object) -> None:
        self.stopped.set()
        if self.thread is not None:
            self.thread.join()


class WorkerClient:
    """Execute one claimed job at a time using only the coordinator API.

    The caller owns ``client`` and its lifetime. It can be an httpx.Client or a
    compatible in-process test client. Each process has its own worker identity.
    """

    def __init__(
        self,
        client: httpx.Client,
        *,
        name: str = "local",
        heartbeat_interval: float = 10.0,
        generation_runner: Callable[[dict, Path], bytes] | None = None,
        generation_kinds: Sequence[str] = M2_JOB_KINDS,
        work_dir: Path | None = None,
        before_job: Callable[[str], None] | None = None,
        llm_models: list[dict] | None = None,
        refresh_capabilities: Callable[[], tuple[list[str], list[dict]]] | None = None,
        extra_capabilities: Sequence[str] = (),
    ) -> None:
        if not name.strip():
            raise ValueError("Worker name cannot be empty")
        if not math.isfinite(heartbeat_interval) or heartbeat_interval <= 0:
            raise ValueError("Heartbeat interval must be positive and finite")
        self.client = client
        self.name = name
        self.heartbeat_interval = heartbeat_interval
        if any(kind not in GENERATION_JOB_KINDS for kind in generation_kinds):
            raise ValueError("Unsupported generation capability")
        self.generation_runner = generation_runner
        self.before_job = before_job
        self.generation_kinds = list(dict.fromkeys(generation_kinds)) if generation_runner else []
        namespace = hashlib.sha256(str(client.base_url).encode()).hexdigest()[:16]
        cache = work_dir or Path(__file__).resolve().parent / "cache" / "m2" / "jobs"
        self.work_dir = cache.resolve() / namespace
        self.worker_id: str | None = None
        self.llm_models = llm_models or []
        self.refresh_capabilities = refresh_capabilities
        self.extra_capabilities = list(dict.fromkeys(extra_capabilities))
        self._next_catalog_refresh = 0.0

    def register(self) -> str:
        response = self.client.post(
            "/api/workers",
            json={"name": self.name, "capabilities": ["tyrano_export", *self.generation_kinds,
                                                     *self.extra_capabilities],
                  **({"llm_models": self.llm_models} if self.llm_models else {})},
        )
        response.raise_for_status()
        self.worker_id = response.json()["id"]
        return self.worker_id

    def _lease_fields(self, job: dict[str, Any]) -> dict[str, str]:
        assert self.worker_id is not None
        return {"worker_id": self.worker_id, "lease_id": job["lease_id"]}

    @staticmethod
    def _check_response(response: httpx.Response) -> None:
        if response.status_code == 409:
            raise LeaseLost("Job lease expired or was replaced")
        response.raise_for_status()

    def _heartbeat(self, job: dict[str, Any]) -> dict[str, Any]:
        response = self.client.post(
            f"/api/jobs/{job['id']}/heartbeat", json=self._lease_fields(job)
        )
        self._check_response(response)
        return response.json()

    def _artifact(self, artifact_id: str) -> bytes:
        response = self.client.get(f"/api/artifacts/{artifact_id}/content")
        response.raise_for_status()
        return response.content

    def _prepare_reference(self, job: dict[str, Any], directory: Path) -> None:
        """Fetch a clone source from our coordinator, never a job-provided URL/path."""
        reference = job["payload"].get("reference_voice")
        if not isinstance(reference, dict):
            raise TypeError("Voice clone requires a reference artifact")
        artifact_id, expected_hash = reference.get("artifact_id"), reference.get("sha256")
        if not isinstance(artifact_id, str) or not re.fullmatch(
            r"[A-Za-z0-9_-]{1,128}", artifact_id
        ):
            raise ValueError("Invalid voice reference artifact identifier")
        if not isinstance(expected_hash, str) or not re.fullmatch(r"[a-f0-9]{64}", expected_hash):
            raise ValueError("Voice reference requires its SHA256")
        if not isinstance(reference.get("text"), str) or not reference["text"].strip():
            raise ValueError("Voice reference requires its original spoken text")
        path = directory / "reference-voice.wav"
        cached = (
            path.read_bytes() if path.is_file() and path.stat().st_size <= 32 * 1024 * 1024 else b""
        )
        content = (
            cached
            if hashlib.sha256(cached).hexdigest() == expected_hash
            else self._artifact(artifact_id)
        )
        if not content or len(content) > 32 * 1024 * 1024:
            raise ValueError("Voice reference has an invalid size")
        if hashlib.sha256(content).hexdigest() != expected_hash:
            raise ValueError("Voice reference content hash mismatch")
        try:
            with wave.open(io.BytesIO(content), "rb") as audio:
                if (
                    audio.getcomptype() != "NONE"
                    or audio.getnchannels() not in (1, 2)
                    or audio.getsampwidth() not in (1, 2, 3, 4)
                    or not 8000 <= audio.getframerate() <= 192000
                    or not 1 <= audio.getnframes() <= audio.getframerate() * 300
                ):
                    raise ValueError("Voice reference must be a nonempty PCM WAV")
                expected_size = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
                if len(audio.readframes(audio.getnframes())) != expected_size:
                    raise ValueError("Voice reference is truncated")
        except (wave.Error, EOFError) as error:
            raise ValueError("Voice reference is not a valid PCM WAV") from error
        if content != cached:
            temporary = path.with_suffix(".tmp")
            temporary.write_bytes(content)
            temporary.replace(path)

    def _execute(self, job: dict[str, Any], heartbeat: _LeaseHeartbeat) -> bytes:
        if self.before_job is not None:
            self.before_job(job["kind"])
        if job["kind"] in self.generation_kinds and self.generation_runner is not None:
            if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job["id"]):
                raise ValueError("Invalid job identifier")
            heartbeat.check()
            directory = self.work_dir / job["id"]
            directory.mkdir(parents=True, exist_ok=True)
            if job["kind"] in {"m2_voice_clone", "m3_voice_clone"}:
                self._prepare_reference(job, directory)
                heartbeat.check()
            if job["kind"] == "m3_event_cg":
                self._prepare_cg_references(job, directory, heartbeat)
            # Job identity, rather than attempt identity, preserves the request
            # and tool cache across lease recovery and transport retries.
            def publish_progress(progress):
                check_cancelled()
                heartbeat.check()
                for attempt in range(2):
                    try:
                        response = self.client.post(f"/api/jobs/{job['id']}/progress",
                            json={**self._lease_fields(job), "progress": progress})
                        self._check_response(response)
                        return
                    except LeaseLost:
                        heartbeat.cancellation.cancel()
                        raise
                    except httpx.HTTPError:
                        if attempt == 1:
                            # Progress is observational. A transport outage must
                            # not discard accepted inference or rerun it.
                            logger.warning("Could not report progress for job %s", job["id"])

            with progress_scope(publish_progress):
                result = self.generation_runner(job, directory)
            heartbeat.check()
            return result
        if job["kind"] != "tyrano_export":
            raise ValueError(f"Unsupported job kind: {job['kind']}")
        raw_script = self._artifact(job["payload"]["script_artifact_id"])
        script = Script.model_validate_json(raw_script)
        assets: dict[str, bytes] = {}
        for asset in script.assets:
            heartbeat.check()
            content = self._artifact(asset.artifact_id)
            if hashlib.sha256(content).hexdigest() != asset.sha256:
                raise ValueError(f"Asset content hash mismatch: {asset.id}")
            assets[asset.id] = content
        heartbeat.check()
        return compile_bundle(script, assets)

    def _prepare_cg_references(self, job, directory, heartbeat):
        from PIL import Image

        from packages.contracts.event_cg import image_input_sha256

        payload = job["payload"]
        if payload.get("input_sha256") != image_input_sha256(payload):
            raise ValueError("CG frozen input hash mismatch.")
        references = payload.get("references")
        if not isinstance(references, list) or not 1 <= len(references) <= 10:
            raise ValueError("CG references must contain one to ten images.")
        for index, reference in enumerate(references, 1):
            heartbeat.check()
            check_cancelled()
            identifier, expected = reference.get("artifact_id"), reference.get("sha256")
            if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", identifier)
                    or not isinstance(expected, str) or not re.fullmatch(r"[a-f0-9]{64}", expected)
                    or reference.get("reference_index") != index):
                raise ValueError("Invalid CG reference identity or order.")
            path = directory / f"cg-reference-{index:02d}.png"
            cached = path.read_bytes() if path.is_file() and path.stat().st_size <= 32 * 1024 * 1024 else b""
            content = cached if hashlib.sha256(cached).hexdigest() == expected else self._artifact(identifier)
            if not content or len(content) > 32 * 1024 * 1024 or hashlib.sha256(content).hexdigest() != expected:
                raise ValueError("CG reference content hash or size mismatch.")
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"PNG", "WEBP", "JPEG"} or getattr(image, "n_frames", 1) != 1:
                    raise ValueError("CG reference must be a static image.")
                if not 1 <= min(image.size) <= max(image.size) <= 4096:
                    raise ValueError("CG reference dimensions are invalid.")
                image.load()
            if content != cached:
                temporary = path.with_suffix(".tmp")
                temporary.write_bytes(content)
                temporary.replace(path)

    def _complete(self, job: dict[str, Any], bundle: bytes) -> None:
        # A transport failure may happen after the server has committed the
        # result. Retry the exact request once; never turn uncertainty into fail.
        prefix = "m2/" if job["kind"] in M2_JOB_KINDS else (
            "m3/" if job["kind"] in M3_KINDS else ""
        )
        for attempt in range(2):
            try:
                response = self.client.post(
                    f"/api/{prefix}jobs/{job['id']}/complete",
                    params=self._lease_fields(job),
                    content=bundle,
                    headers={"Content-Type": "application/zip"},
                )
                self._check_response(response)
                return
            except httpx.TransportError:
                if attempt == 1:
                    raise

    def _fail(self, job: dict[str, Any], error: Exception) -> RunResult:
        try:
            response = self.client.post(
                f"/api/jobs/{job['id']}/fail",
                json={**self._lease_fields(job), "error": (
                    str(error) if len(str(error)) <= 2000
                    else str(error)[:450] + "\n...\n" + str(error)[-1540:]
                )},
            )
            self._check_response(response)
        except LeaseLost:
            return "stale"
        except httpx.HTTPError:
            logger.exception("Could not report failure for job %s; lease will recover", job["id"])
            return "deferred"
        logger.warning("Job %s failed: %s", job["id"], error)
        return "failed"

    def run_once(self) -> RunResult:
        """Register if necessary, claim at most one job, and return its outcome.

        ``deferred`` leaves uncertain work to lease recovery. No partial output
        is uploaded, and stale attempts never submit a failure for a new lease.
        Registration/claim errors propagate so callers can choose retry policy.
        """
        if self.refresh_capabilities is not None and time.monotonic() >= self._next_catalog_refresh:
            kinds, models = self.refresh_capabilities()
            if self.worker_id is not None and (kinds != self.generation_kinds or models != self.llm_models):
                updated = self.client.post(f"/api/workers/{self.worker_id}/capabilities", json={
                    "capabilities": ["tyrano_export", *kinds, *self.extra_capabilities],
                    "llm_models": models})
                updated.raise_for_status()
            self.generation_kinds, self.llm_models = kinds, models
            self._next_catalog_refresh = time.monotonic() + 10
        if self.worker_id is None:
            self.register()
        response = self.client.post(f"/api/workers/{self.worker_id}/claim", json={})
        response.raise_for_status()
        job = response.json()["job"]
        if job is None:
            return "idle"
        logger.info("Job %s (%s) started", job["id"], job["kind"])

        try:
            with cancellation_scope() as cancellation, _LeaseHeartbeat(
                self, job, cancellation
            ) as heartbeat:
                try:
                    bundle = self._execute(job, heartbeat)
                    heartbeat.check()
                except (LeaseLost, GenerationCancelled):
                    raise
                except httpx.TransportError:
                    raise
                except Exception as error:  # noqa: BLE001 - persist any executor failure via the API
                    # Revocation can surface as a pipe/socket error from the
                    # child we just terminated. Never fail the replacement lease.
                    heartbeat.check()
                    return self._fail(job, error)
                # Completion is deliberately outside the input/compile error
                # handler: its response can be lost after adoption succeeds.
                try:
                    self._complete(job, bundle)
                except httpx.HTTPStatusError as error:
                    if error.response.status_code in {400, 413, 415, 422}:
                        # A definitive validation rejection has not adopted the
                        # output. Unlike an uncertain transport failure it can
                        # safely consume this attempt and surface its failure.
                        return self._fail(
                            job, ValueError("生成結果がサーバーの検証を通りませんでした。")
                        )
                    raise
        except (LeaseLost, GenerationCancelled):
            voices = current_session()
            if voices is not None:
                voices.close()
            images = current_image_session()
            if images is not None:
                images.close()
            llms = current_llm_session()
            if llms is not None:
                llms.close()
            music = current_music_session()
            if music is not None:
                music.close()
            cgs = current_event_cg_session()
            if cgs is not None:
                cgs.close()
            logger.info("Job %s no longer belongs to this worker", job["id"])
            return "stale"
        except httpx.HTTPError:
            logger.exception("Job %s deferred to lease recovery", job["id"])
            return "deferred"
        logger.info("Job %s completed", job["id"])
        return "completed"
