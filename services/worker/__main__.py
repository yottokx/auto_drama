"""Run with the worker venv, without loading coordinator or GPU libraries."""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from pathlib import Path

# Direct file execution is supported by the isolated-mode PowerShell launcher.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import httpx

from services.worker.client import WorkerClient
from services.worker.generation.voice_session import prepare_job, reuse_voice_runtime

logger = logging.getLogger(__name__)


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
    generation_runner = None
    generation_kinds = []
    if not args.export_only:
        from services.worker.generation import check_readiness, generate_job

        readiness = check_readiness()
        generation_kinds = readiness["available_job_kinds"]
        if generation_kinds:
            generation_runner = generate_job
        for error in readiness["errors"]:
            logger.warning("Local model configuration: %s", error)
        logger.info(
            "Available generation jobs: %s", ", ".join(generation_kinds) or "none (export only)"
        )
    try:
        with (
            reuse_voice_runtime(enabled=not args.no_voice_reuse and not args.export_only) as voices,
            httpx.Client(base_url=args.coordinator, timeout=30.0, trust_env=False) as client,
        ):
            worker = WorkerClient(
                client,
                name=args.name,
                generation_runner=generation_runner,
                generation_kinds=generation_kinds,
                work_dir=args.work_dir,
                before_job=prepare_job,
            )
            while True:
                try:
                    result = worker.run_once()
                except httpx.HTTPError as error:
                    logger.warning("Coordinator unavailable: %s", error)
                    result = "deferred"
                if result != "completed" and voices is not None:
                    voices.close()
                if args.once:
                    return 0 if result in {"idle", "completed"} else 1
                if result != "completed":
                    time.sleep(args.poll_interval)
    except KeyboardInterrupt:
        logger.info("Worker stopped; unfinished leases will recover on the coordinator")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
