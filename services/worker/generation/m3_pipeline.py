"""Persisted M3 local jobs, sharing the M2 GPU ownership boundary and media runtimes."""

from __future__ import annotations

import hashlib
import io
import json
import tempfile
import zipfile
from pathlib import Path

from packages.contracts.m3 import EMOTION_TAGS, M3_KINDS, GenerationEnvelope, Location

from . import image_session, llm_session, music_session, pipeline
from .cancellation import check_cancelled
from .causal_runtime import digest
from .llm import LocalLLM, write_json
from .model_routing import RoutedLLM, model_configuration_identity
from .narrative import (
    CONTINUITY_PROMPT_VERSION,
    PROMPT_VERSION,
    StructuredGenerationError,
    generate_narrative,
)
from .processes import gpu_lock, run_process
from .workflow_version import generator_protocol


def generation_kinds(available: list[str]) -> list[str]:
    kinds = []
    if "m2_world" in available:
        kinds.extend(("m3_plan", "m3_narrative", "m3_music_plan"))
    if "m2_image" in available:
        kinds.extend(("m3_background", "m3_image"))
    if "m2_voice" in available:
        kinds.append("m3_voice")
    if "m2_voice_clone" in available:
        kinds.append("m3_voice_clone")
    return kinds


def generate_background(payload: dict, work: Path, config: dict) -> tuple[bytes, dict]:
    location = Location.model_validate(payload["location"])
    settings = config["image"]
    output = Path(tempfile.mkdtemp(prefix="background-", dir=work))
    prompt = ", ".join(filter(None, (
        settings.get("positive_prompt_prefix", ""),
        "scenery, no humans, no people, visual novel background", location.image_prompt,
    )))
    negative = settings.get("negative_prompt", "") + ", person, human, girl, boy, face, silhouette"
    command = [str(pipeline.ROOT / settings["python"]),
        str(pipeline.ROOT / "scripts/m0/image_smoke.py"), "--mode", "background",
        "--prompt", prompt, "--negative-prompt", negative,
        "--model-dir", str(pipeline.ROOT / settings["model_dir"]),
        "--output-dir", str(output), "--width", "1280", "--height", "720",
        "--steps", str(settings["steps"]), "--guidance-scale", str(settings["guidance_scale"]),
        "--seed", str(payload["seed"])]
    runner = run_process if image_session.current_session() is None else image_session.run_image_process
    runner(command, output / "runtime.log", cwd=pipeline.ROOT, timeout=settings["timeout_seconds"])
    report = json.loads((output / "result.json").read_text(encoding="utf-8"))
    if (report.get("mode") != "background" or report.get("prompt") != prompt
            or report.get("negative_prompt") != negative
            or report.get("width") != 1280 or report.get("height") != 720):
        raise ValueError("Background runtime did not honor the background request.")
    return (output / "image.png").read_bytes(), {
        "model": "Anima", "location_id": location.id, "source_description": location.description,
        **{key: report[key] for key in ("official_diffusers_revision", "runtime_commit", "versions",
            "prompt", "negative_prompt", "seed", "width", "height", "steps", "guidance_scale")},
        "model_reused": report.get("model_reused", False),
        **({"initialization_seconds": report["initialization_seconds"]}
           if "initialization_seconds" in report else {}),
    }


def _media_job(job: dict, work: Path) -> bytes:
    """Reuse proven portraits/references/clones; wrap their result in the M3 contract."""
    payload = {**job["payload"], "character_contract_version": 2}
    payload.setdefault("instruction", "")
    payload.setdefault("scope", {"m3_image": "image-retake", "m3_voice": "voice-retake",
                                 "m3_voice_clone": "voice-clone"}[job["kind"]])
    original_text = payload.get("dialogue_text")
    emotion = payload.get("voice_emotion", "neutral")
    if job["kind"] == "m3_voice_clone":
        if emotion not in EMOTION_TAGS:
            raise ValueError("Unsupported voice emotion.")
        if not isinstance(original_text, str) or not original_text.strip():
            raise ValueError("Dialogue text is required.")
        payload["tts_emotion"] = emotion
    content = pipeline.generate_job(
        {**job, "kind": job["kind"].replace("m3_", "m2_", 1), "payload": payload}, work,
        **({"supporting_portrait": True}
           if job["kind"] == "m3_image" and not payload.get("adjustment") else {}))
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        envelope = json.loads(archive.read("result.json"))
        assets = {name: archive.read(name) for name in archive.namelist() if name != "result.json"}
    envelope["kind"] = job["kind"]
    purpose = {"m3_image": "supporting_image", "m3_voice": "supporting_reference_voice",
               "m3_voice_clone": "dialogue_voice"}[job["kind"]]
    if payload.get("adjustment"):
        purpose = "adjustment_" + {
            "m3_image": "portrait", "m3_voice": "reference_voice",
            "m3_voice_clone": "sample_voice" if payload["adjustment"].get("purpose") == "sample"
            else "dialogue_voice",
        }[job["kind"]]
        envelope["provenance"]["adjustment"] = payload["adjustment"]
    envelope["provenance"]["purpose"] = purpose
    if job["kind"] == "m3_voice_clone":
        envelope["provenance"]["voice"].update({
            "display_text": original_text, "spoken_text": original_text,
            "voice_emotion": emotion, "emotion_tag": EMOTION_TAGS[emotion],
        })
    envelope["trace"].append({"type": "task_profile", "purpose": purpose,
                              "profile": job["payload"].get("profile", {})})
    return pipeline._bundle(GenerationEnvelope.model_validate(envelope).model_dump(), assets)


def generate_job(job: dict, work_dir: Path) -> bytes:
    check_cancelled()
    kind, payload = job.get("kind"), job.get("payload")
    music_session.prepare_job(kind)
    if kind in {"m3_music_plan", "m3_music"}:
        from .music_pipeline import generate_job as generate_music

        return generate_music(job, work_dir)
    pipeline.voice_session.prepare_job(kind)
    image_session.prepare_job(kind)
    pipeline.llm_session.prepare_job(kind)
    if kind not in M3_KINDS or not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError("Unsupported M3 job or payload schema.")
    if type(payload.get("seed")) is not int or not 0 <= payload["seed"] < 2**63:
        raise ValueError("Invalid M3 generation seed.")
    script = kind == "m3_narrative" and payload.get("workflow_policy") == "script_continuation_v1"
    plan = kind == "m3_plan"
    if plan:
        from packages.contracts.planning import planning_protocol

        if (payload.get("planning_protocol") != planning_protocol()
                or payload.get("workflow_policy") != "script_continuation_v1"
                or payload.get("story_workflow_version") != 2
                or payload.get("generator_protocol") != generator_protocol("causal", "script_continuation_v1")):
            raise ValueError("Common planning requires its versioned planning and generator protocol.")
    if kind == "m3_narrative" and payload.get("workflow_policy") not in (None, "chapter_editor_v1", "script_continuation_v1"):
        raise ValueError("Unknown narrative workflow policy.")
    if script and (payload.get("story_workflow_version") != 2 or "generator_protocol" not in payload):
        raise ValueError("Script production requires its workflow version and generator protocol.")
    if kind == "m3_narrative" and "generator_protocol" in payload:
        expected_protocol = generator_protocol("causal" if payload.get("story_workflow_version") == 2 else "legacy",
                                               payload.get("workflow_policy"))
        if payload["generator_protocol"] != expected_protocol:
            raise ValueError("Narrative generator protocol changed; use a new experiment.")
    if (payload.get("workflow_policy") == "chapter_editor_v1"
            and (payload.get("execution_mode") != "text_only"
                 or payload.get("story_workflow_version") != 2
                 or "generator_protocol" not in payload)):
        raise ValueError("The chapter editor requires a versioned text-only causal experiment.")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True, exist_ok=True)
    if kind in {"m3_image", "m3_voice", "m3_voice_clone"}:
        return _media_job(job, work)
    from ..model_config import select_config
    config = pipeline.load_config()
    if kind in {"m3_plan", "m3_narrative"}:
        config = select_config(config, payload, pipeline.ROOT)
    causal = kind in {"m3_plan", "m3_narrative"} and payload.get("story_workflow_version") == 2
    if causal:
        config = {**config, "model_configuration_identity": model_configuration_identity(pipeline.ROOT, config)}
    fingerprint = hashlib.sha256(json.dumps({"kind": kind, "payload": payload},
        ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    request_path = work / "job-request.json"
    if request_path.exists():
        previous = json.loads(request_path.read_text(encoding="utf-8"))
        if previous.get("fingerprint") != fingerprint:
            raise ValueError("Generation work directory belongs to another input snapshot.")
        if causal and previous["generation_config"].get("model_configuration_identity") != config["model_configuration_identity"]:
            raise ValueError("Narrative model configuration changed; use a new experiment.")
        config = previous["generation_config"]
    else:
        write_json(request_path, {"fingerprint": fingerprint, "kind": kind, "payload": payload,
                                  "generation_config": config})
    cache = work / "result.zip"
    if cache.exists():
        check_cancelled()
        return cache.read_bytes()
    continuous = kind == "m3_narrative" and (payload.get("m4") is True or payload.get("chapter_number", 1) > 1)
    narrative_version = (payload["generator_protocol"]["prompt"] if causal and
                         payload.get("workflow_policy") in {"chapter_editor_v1", "script_continuation_v1"} else
                         7 if causal else CONTINUITY_PROMPT_VERSION if continuous else PROMPT_VERSION)
    provenance = {"provider": "local", "seed": payload["seed"],
                  "prompt_version": narrative_version if kind == "m3_narrative" else 1,
                  "input_sha256": fingerprint, "profile": payload.get("profile", {}),
                  "purpose": "story_planning" if plan else "story_narrative" if kind == "m3_narrative" else "background_image"}
    if kind in {"m3_plan", "m3_narrative"}:
        provenance["generator_protocol"] = {
            **generator_protocol("causal" if causal else "legacy", payload.get("workflow_policy")),
            "prompt": narrative_version}
    if plan:
        provenance["generator_protocol"] = planning_protocol()
        provenance["planning_protocol"] = planning_protocol()
        provenance["planning_id"] = payload.get("planning_id")
        provenance["planning_revision"] = payload.get("planning_revision", 0)
    assets, trace, result = {}, [], {}
    script_run = None
    if script:
        from .script_production import ProductionScriptRun

        script_run = ProductionScriptRun(work / "script", config, payload)
        script_run.retry_failed_steps(job.get("retry_generation", 0))
    elif plan:
        from .planning_run import CommonPlanRun

        script_run = CommonPlanRun(work / "script", config, payload)
        script_run.retry_failed_steps(job.get("retry_generation", 0))
    with llm_session.gpu_scope(kind, image_session.gpu_scope(kind, gpu_lock(
            pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]))):
        if kind in {"m3_plan", "m3_narrative"}:
            # Existing M3 jobs replay their exact v4 request cache; state-aware
            # prompts have an independent namespace and never overwrite it.
            llm_work = work / "script" / "llm" if script or plan else work / (f"llm-v{narrative_version}" if continuous or causal else "llm")
            replay = (work / "llm-v5",) if continuous and narrative_version == 6 else ()
            llm_type = RoutedLLM if causal else LocalLLM
            with llm_type(pipeline.ROOT, config, payload, llm_work, replay_outputs=replay) as llm:
                try:
                    if plan:
                        result = script_run.execute(llm)
                        provenance["plan_sha256"] = digest(result)
                    elif script_run is not None:
                        result, provenance["script_checkpoint"] = script_run.execute(llm)
                    else:
                        result = generate_narrative(payload, llm)
                except (ValueError, StructuredGenerationError):
                    # Semantic rejection is definitive. A recovered lease or
                    # uncertain upload does not enter this branch, so successful
                    # source/cache records remain stable across transport retries.
                    if llm.requests and not causal:
                        llm.retry_failed_from(getattr(llm, "failure_request", llm.requests))
                    raise
                finally:
                    write_json(work / "generation-trace.json", list(llm.trace))
                provenance["llm"] = llm.provenance() if causal else {
                    "model_id": config["llm"]["model_id"], "revision": llm.base["model"].get("revision"),
                    "sha256": llm.base["model"].get("publisher_sha256", llm.base["model"].get("local_sha256")), "requests": llm.requests,
                    "reasoning_level": llm.profile.get("reasoning_level", "none"),
                    "top_p": llm.profile.get("top_p", 0.95),
                    "retry_seed_segments": llm.retry_seed_segments,
                }
            trace = list(llm.trace)
            write_json(work / "generation-trace.json", trace)
        else:
            assets["image.png"], provenance["image"] = generate_background(payload, work, config)
        envelope = GenerationEnvelope(schema_version=1, kind=kind, result=result,
                                      provenance=provenance, trace=trace).model_dump(mode="json")
        bundle = pipeline._bundle(envelope, assets)
        if len(bundle) > config["max_zip_bytes"]:
            raise ValueError("Generated result exceeds upload size limit.")
        check_cancelled()
        temporary = cache.with_suffix(".tmp")
        temporary.write_bytes(bundle)
        temporary.replace(cache)
        write_json(work / "result.json", envelope)
        check_cancelled()
        return bundle
