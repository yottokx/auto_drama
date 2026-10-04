"""Run with the worker venv, without loading coordinator or GPU libraries."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from contextlib import ExitStack
from pathlib import Path

# Direct file execution is supported by the isolated-mode PowerShell launcher.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from services.worker.client import WorkerClient
from services.worker.generation import image_session, llm_session
from services.worker.generation.voice_session import prepare_job, reuse_voice_runtime

logger = logging.getLogger(__name__)


def prepare_media_job(kind: str):
    prepare_job(kind)
    image_session.prepare_job(kind)
    llm_session.prepare_job(kind)


def maintain_runtimes(result: str, *sessions):
    """Keep healthy resident models while idle; release unsafe attempts."""
    for session in sessions:
        if session is not None:
            if result in {"failed", "stale"}:
                session.close()
            else:
                session.expire_if_needed()


def positive_seconds(value: str) -> float:
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise argparse.ArgumentTypeError("must be a positive, finite number")
    return seconds


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the local Auto Drama generation/export worker"
    )
    parser.add_argument("--coordinator", default="http://127.0.0.1:8000")
    parser.add_argument("--name", default="local")
    parser.add_argument("--once", action="store_true", help="Claim at most one job, then exit")
    parser.add_argument("--poll-interval", type=positive_seconds, default=2.0)
    parser.add_argument(
        "--no-voice-reuse", action="store_true", help="Reload Irodori for every voice job"
    )
    parser.add_argument("--no-image-reuse", action="store_true", help="Reload Anima for every image job")
    parser.add_argument(
        "--export-only", action="store_true", help="Run M1 export without loading M2 configuration"
    )
    parser.add_argument(
        "--work-dir", type=Path, help="Persistent generation request and result cache"
    )
    args = parser.parse_args()
    if not args.name.strip():
        parser.error("--name cannot be empty")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)
    root = Path(__file__).resolve().parents[2]
    generation_runner = None
    generation_kinds = []
    llm_models = []
    refresh_capabilities = None
    if not args.export_only:
        from services.worker.generation import check_readiness, generate_job

        readiness = check_readiness()
        from services.worker.model_config import catalog
        llm_models, model_errors = catalog(Path(__file__).resolve().parents[2])
        for error in model_errors:
            logger.warning("Worker model registry: %s", error)
        generation_kinds = readiness["available_job_kinds"]
        generation_runner = generate_job

        def refresh_capabilities():
            state = check_readiness()
            models, _ = catalog(Path(__file__).resolve().parents[2])
            return state["available_job_kinds"], models
        for error in readiness["errors"]:
            logger.warning("Local model configuration: %s", error)
        logger.info(
            "Available generation jobs: %s", ", ".join(generation_kinds) or "none (export only)"
        )
    try:
        with (
            reuse_voice_runtime(enabled=not args.no_voice_reuse and not args.export_only) as voices,
            image_session.reuse_image_runtime(enabled=not args.no_image_reuse and not args.export_only) as images,
            llm_session.reuse_llm_runtime(enabled=not args.export_only) as llms,
            httpx.Client(base_url=args.coordinator, timeout=30.0, trust_env=False) as client,
            ExitStack() as download_stack,
        ):
            worker = WorkerClient(
                client,
                name=args.name,
                generation_runner=generation_runner,
                generation_kinds=generation_kinds,
                work_dir=args.work_dir,
                before_job=prepare_media_job,
                llm_models=llm_models,
                refresh_capabilities=refresh_capabilities,
                extra_capabilities=["tts_download"] if not args.export_only and not args.once else [],
            )
            downloads_started = False
            while True:
                maintain_runtimes("idle", voices, images, llms)
                try:
                    if not args.export_only and not args.once and not downloads_started:
                        from services.worker.tts_download_client import TTSDownloadAgent

                        if worker.worker_id is None:
                            worker.register()
                        download_client = download_stack.enter_context(httpx.Client(
                            base_url=args.coordinator, timeout=15, trust_env=False
                        ))
                        download_stack.enter_context(TTSDownloadAgent(
                            download_client, worker.worker_id, root
                        ))
                        downloads_started = True
                    result = worker.run_once()
                except httpx.HTTPError as error:
                    logger.warning("Coordinator unavailable: %s", error)
                    result = "deferred"
                maintain_runtimes(result, voices, images, llms)
                if args.once:
                    return 0 if result in {"idle", "completed"} else 1
                if result != "completed":
                    # Service resident-model deadlines even with a long poll interval.
                    remaining = args.poll_interval
                    while remaining > 0:
                        interval = min(remaining, 1.0)
                        time.sleep(interval)
                        remaining -= interval
                        maintain_runtimes("idle", voices, images, llms)
    except KeyboardInterrupt:
        logger.info("Worker stopped; unfinished leases will recover on the coordinator")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
