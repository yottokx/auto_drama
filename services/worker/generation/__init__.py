"""Local generation worker entry points. No GPU libraries in this process."""
from pathlib import Path

from . import pipeline


def check_readiness() -> dict:
    from .m3_pipeline import generation_kinds

    status = pipeline.check_readiness()
    status["available_job_kinds"].extend(generation_kinds(status["available_job_kinds"]))
    return status


def available_job_kinds() -> list[str]:
    return check_readiness()["available_job_kinds"]


def generate_job(job: dict, work_dir: Path) -> bytes:
    if job.get("kind", "").startswith("m3_"):
        from .m3_pipeline import generate_job as generate_m3

        return generate_m3(job, work_dir)
    return pipeline.generate_job(job, work_dir)

__all__ = ["available_job_kinds", "check_readiness", "generate_job"]
