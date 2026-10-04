"""Local generation worker entry points. No GPU libraries in this process."""
from pathlib import Path


def check_readiness() -> dict:
    from . import pipeline
    from .event_cg_pipeline import check_readiness as event_cg_readiness
    from .m3_pipeline import generation_kinds
    from .music_pipeline import check_readiness as music_readiness

    status = pipeline.check_readiness()
    status["available_job_kinds"].extend(generation_kinds(status["available_job_kinds"]))
    try:
        music = music_readiness(pipeline.load_config(), pipeline.ROOT)
    except (OSError, ValueError, KeyError) as exc:
        music = {"ready": False, "errors": [str(exc)]}
    if music["ready"]:
        status["available_job_kinds"].append("m3_music")
    status["errors"].extend(music["errors"])
    status["music_ready"] = music["ready"]
    try:
        event_cg = event_cg_readiness(pipeline.load_config(), pipeline.ROOT)
    except (OSError, ValueError, KeyError):
        event_cg = {"ready": False, "errors": []}
    if event_cg["ready"]:
        status["available_job_kinds"].append("m3_event_cg")
    status["errors"].extend(event_cg["errors"])
    status["event_cg_ready"] = event_cg["ready"]
    return status


def available_job_kinds() -> list[str]:
    return check_readiness()["available_job_kinds"]


def generate_job(job: dict, work_dir: Path) -> bytes:
    from . import pipeline
    from .event_cg_session import prepare_job as prepare_event_cg
    from .music_session import prepare_job

    prepare_job(job.get("kind"))
    prepare_event_cg(job.get("kind"))
    if job.get("kind", "").startswith("m3_"):
        from .m3_pipeline import generate_job as generate_m3

        return generate_m3(job, work_dir)
    return pipeline.generate_job(job, work_dir)

__all__ = ["available_job_kinds", "check_readiness", "generate_job"]
