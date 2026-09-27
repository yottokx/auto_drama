"""Offline text-only experiments through the production narrative generation path.

An experiment owns its output directory, never a coordinator database or public
build. A chapter becomes a predecessor only after normal narrative validation and
an atomic commit record. LLM request caches remain available after failure.
"""

from __future__ import annotations

import copy
import hashlib
import html
import io
import json
import os
import time
import zipfile
from contextlib import contextmanager, nullcontext
from pathlib import Path

from packages.contracts.m2 import CharacterResult, RelationshipsResult, WorldResult
from packages.contracts.m3 import GenerationEnvelope
from packages.narrative import story_state_hash, validate_narrative

from . import m3_pipeline, pipeline
from .causal_runtime import PersistentUsage, ResourceBudgetError, resource_limits
from .llm import write_json
from .model_routing import model_configuration_identity
from .schemas import validate_relationships
from .workflow_version import generator_protocol


def _hash(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":")).encode()).hexdigest()


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _input_payload(value: dict) -> dict:
    if not isinstance(value, dict):
        raise TypeError("Input must be an approval snapshot or M3 narrative job JSON object.")
    if "payload" in value:
        if value.get("kind", "m3_narrative") != "m3_narrative":
            raise ValueError("Only m3_narrative job requests can seed a text experiment.")
        value = value["payload"]
    if not isinstance(value, dict):
        raise TypeError("M3 job payload must be an object.")
    return value


def extract_input(value: dict) -> tuple[dict, dict, int]:
    """Accept an approval snapshot or a saved M3 job request, without media reads."""
    value = _input_payload(value)
    source = value.get("approval_snapshot", value.get("snapshot", value))
    if not isinstance(source, dict):
        raise TypeError("Approval snapshot must be an object.")
    snapshot = copy.deepcopy(source)
    world = snapshot.get("world", {})
    if not isinstance(world, dict):
        raise TypeError("Approval world must be an object.")
    world_result = world.get("result", world)
    WorldResult.model_validate(world_result)
    snapshot["world"] = world if "result" in world else {"result": world}
    characters = snapshot.get("characters")
    if not isinstance(characters, list) or not characters:
        raise ValueError("At least one approved main character is required.")
    cast = []
    for item in characters:
        if not isinstance(item, dict):
            raise TypeError("Each approved character must be an object.")
        character = item.get("result", item)
        CharacterResult.model_validate(character)
        cast.append(character)
    if len({item["id"] for item in cast}) != len(cast):
        raise ValueError("Approved character IDs must be unique.")
    snapshot["characters"] = [item if "result" in item else {"result": item}
                              for item in characters]
    relationships = snapshot.get("relationships", {"result": {"pairs": []}})
    if not isinstance(relationships, dict):
        raise TypeError("Approved relationships must be an object.")
    relations_result = relationships.get("result", relationships)
    if len(cast) == 1:
        if RelationshipsResult.model_validate(relations_result).pairs:
            raise ValueError("A single main character cannot have a main-character pair.")
    else:
        validate_relationships(relations_result, cast)
    snapshot["relationships"] = (relationships if "result" in relationships
                                 else {"result": relationships})
    profile, seed = copy.deepcopy(value.get("profile", {})), value.get("seed", 1)
    if not isinstance(profile, dict):
        raise TypeError("Generation profile must be an object.")
    return snapshot, profile, seed


@contextmanager
def _output_lock(output: Path):
    """OS-owned lock: a killed runner can resume without deleting a stale marker."""
    lock = (output / ".text-debug.lock").open("a+b")
    try:
        if lock.tell() == 0:
            lock.write(b"0")
            lock.flush()
        lock.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("Another text-debug runner owns this output directory.") from exc
        yield
    finally:
        # Closing the handle releases the OS lock even when a run failed.
        lock.close()


def _payload(manifest: dict, number: int, previous: dict | None) -> dict:
    payload = {
        "schema_version": 1, "m4": True, "chapter_number": number,
        "production_id": f"{manifest['storyline_id']}-{number:03d}",
        "storyline_id": manifest["storyline_id"],
        "approval_snapshot": manifest["approval_snapshot"],
        "seed": (manifest["seed"] + number - 1) % (2**63),
        "profile": manifest["profile"], "execution_mode": "text_only",
        "generator_protocol": manifest["generator_protocol"],
    }
    if manifest.get("profiles"):
        payload["profiles"] = copy.deepcopy(manifest["profiles"])
    if manifest.get("workflow_limits"):
        payload["workflow_limits"] = copy.deepcopy(manifest["workflow_limits"])
    if manifest["workflow"] == "causal":
        payload["story_workflow_version"] = 2
        if manifest.get("workflow_policy"):
            payload["workflow_policy"] = manifest["workflow_policy"]
    if previous is not None:
        payload.update(previous_narrative=previous["envelope"]["result"],
                       previous_narrative_artifact_id=previous["artifact_id"],
                       previous_state_hash=previous["end_state_hash"])
    return payload


def _validate_result(envelope: dict, payload: dict, *, workflow: str) -> dict:
    checked = GenerationEnvelope.model_validate(envelope)
    if checked.kind != "m3_narrative":
        raise ValueError("Text-only execution returned a non-narrative result.")
    expected_provenance = {
        "input_sha256": _hash({"kind": "m3_narrative", "payload": payload}),
        "generator_protocol": payload["generator_protocol"],
        "prompt_version": payload["generator_protocol"]["prompt"],
        "seed": payload["seed"], "profile": payload["profile"],
    }
    if any(checked.provenance.get(key) != value for key, value in expected_provenance.items()):
        raise ValueError("Narrative provenance does not match the input/profile/seed/generator protocol.")
    result = validate_narrative(checked.result, payload["approval_snapshot"],
                                payload.get("previous_narrative"), require_state=True)
    if (result.chapter_number != payload["chapter_number"]
            or result.storyline_id != payload["storyline_id"]
            or result.previous_narrative_artifact_id != payload.get("previous_narrative_artifact_id")
            or result.previous_state_hash != payload.get("previous_state_hash")):
        raise ValueError("Generated narrative does not match the exact predecessor lineage.")
    if result.workflow_version != payload["generator_protocol"]["workflow"]:
        raise ValueError("Requested workflow does not match the generated narrative version.")
    if result.workflow_policy != payload.get("workflow_policy", "scene_gate_v1"):
        raise ValueError("Requested chapter policy does not match the generated narrative.")
    # Keep the generated JSON intact. Validation is not a second serialization pass.
    return checked.result


def _request_records(work: Path) -> dict[str, dict]:
    records = {}
    for path in sorted(work.glob("llm*/**/*.json")):
        try:
            value = _read(path)
        except (OSError, ValueError):
            continue
        if isinstance(value, dict) and isinstance(value.get("response"), dict):
            records[path.relative_to(work).as_posix()] = value
    return records


def _trace_files(work: Path) -> dict[str, str]:
    return {path.relative_to(work).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in [work / "generation-trace.json", *work.glob("llm*/llm-metrics.json")]
            if path.is_file()}


def _metrics(work: Path, previous_files: set[str], envelope: dict | None,
             elapsed: float, previous_traces: dict[str, str], *, cached_result: bool) -> dict:
    records = _request_records(work)
    fresh = [value for key, value in records.items() if key not in previous_files]

    def token_total(key: str, values: list[dict]):
        counts = [(item["response"].get("usage") or {}).get(key) for item in values]
        measured = [count for count in counts if type(count) is int]
        return sum(measured) if measured else None

    def timing_seconds(key: str):
        counts = [(item["response"].get("timings") or {}).get(key) for item in fresh]
        measured = [count for count in counts if type(count) in {int, float}]
        return round(sum(measured) / 1000, 3) if measured else None

    llm = (envelope or {}).get("provenance", {}).get("llm", {})
    trace = (envelope or {}).get("trace", [])
    trace_origin = "saved_result" if cached_result else "current_result" if envelope else "unavailable"
    traces = _trace_files(work)
    # LocalLLM writes its complete trace including release on failure. The pipeline
    # trace may be captured before release; prefer the complete LLM record.
    changed = [name for name, digest in traces.items() if digest != previous_traces.get(name)]
    changed.sort(key=lambda name: name.endswith("llm-metrics.json"), reverse=True)
    if not cached_result:
        for name in changed:
            value = _read(work / name)
            if isinstance(value, list):
                trace, trace_origin = value, f"current_file:{name}"
                break
    this_run = trace if trace_origin.startswith("current_") else []

    def elapsed_total(kind: str):
        measured = [item["elapsed_seconds"] for item in this_run if isinstance(item, dict)
                    and item.get("type") == kind and type(item.get("elapsed_seconds")) in {int, float}]
        if cached_result:
            return 0.0
        return round(sum(measured), 3) if measured else None
    per_model = {}
    for item in fresh:
        runtime = item.get("runtime", {})
        model_id = runtime.get("profile", {}).get("model_id", "unknown")
        per_model.setdefault(model_id, []).append(item)
    return {
        "elapsed_seconds": round(elapsed, 3),
        "request_cache_records": len(records), "new_request_cache_records": len(fresh),
        "prompt_tokens_new": token_total("prompt_tokens", fresh),
        "completion_tokens_new": token_total("completion_tokens", fresh),
        "prompt_tokens_saved": token_total("prompt_tokens", list(records.values())),
        "completion_tokens_saved": token_total("completion_tokens", list(records.values())),
        "prompt_processing_seconds_new": timing_seconds("prompt_ms"),
        "token_generation_seconds_new": timing_seconds("predicted_ms"),
        "requests_in_adopted_run": llm.get("requests"),
        "retry_seed_segments": llm.get("retry_seed_segments", []),
        "result_cache_hit": cached_result, "trace_origin": trace_origin,
        "model_load_seconds_this_run": elapsed_total("model_load"),
        "model_release_seconds_this_run": elapsed_total("model_release"),
        "model_loads_this_run": sum(item.get("type") == "model_load" for item in this_run),
        "model_switches_this_run": sum(item.get("type") == "model_switch"
            and item.get("status") == "ready" and item.get("from_route") is not None
            for item in this_run),
        "models_new": {model_id: {"requests": len(values),
            "prompt_tokens": token_total("prompt_tokens", values),
            "completion_tokens": token_total("completion_tokens", values)}
            for model_id, values in per_model.items()},
        "timing_trace_this_run": [item for item in this_run if isinstance(item, dict)
                                 and ("seconds" in json.dumps(item)
                                      or "timing" in item.get("type", ""))],
        "timing_trace": [item for item in trace if isinstance(item, dict)
                         and ("seconds" in json.dumps(item) or "timing" in item.get("type", ""))],
        "measurement_note": "null means unavailable; saved tokens include prior attempts and caches",
    }


def _draft_report(work: Path) -> None:
    entries = ["# 未採用の生成途中資料", "", "以下は検査・章採用前の候補です。通読本文には含めません。", ""]
    saved = {}
    stages = work / "causal-stages"
    for prefix in ("editor-raw-", "editor-draft-"):
        for path in stages.glob(prefix + "*.json"):
            try:
                record = _read(path)
                value = record.get("value", {})
                scene_id = path.stem[len(prefix):].rsplit("-", 1)[0]
                prose = value.get("raw_text" if prefix == "editor-draft-" else "text")
                if not scene_id or not isinstance(prose, str) or not prose:
                    continue
                key = (scene_id, prefix)
                if key not in saved or path.stat().st_mtime_ns > saved[key][0]:
                    saved[key] = (path.stat().st_mtime_ns, prose)
            except (OSError, ValueError, TypeError):
                continue
    if saved:
        entries.extend(["## 保存済み本文下書き", "", "章末の採用判定前に保存された原文です。", ""])
        for scene_id in sorted({scene_id for scene_id, _ in saved}):
            raw = saved.get((scene_id, "editor-raw-"))
            mapped = saved.get((scene_id, "editor-draft-"))
            if mapped:
                entries.extend([f"### {scene_id}（表示用原文）", "", "````text",
                                mapped[1], "````", ""])
            if raw and (not mapped or raw[1] != mapped[1]):
                entries.extend([f"### {scene_id}（執筆直後の原稿）", "", "````text",
                                raw[1], "````", ""])
    for name, record in _request_records(work).items():
        choices = record.get("response", {}).get("choices", [])
        if choices:
            content = choices[0].get("message", {}).get("content")
            if isinstance(content, str):
                entries.extend([f"## {name}", "", "````text", content, "````", ""])
    (work / "draft.md").write_text("\n".join(entries), encoding="utf-8")


def _render(output: Path, manifest: dict, commits: list[dict], report: dict) -> None:
    title = manifest["approval_snapshot"]["world"]["result"]["title"]
    quality_note = "PoC実験です。自動検査の完了と、人による作劇品質の受入は別です。品質受入は未実施です。"
    lines = [f"# {title}", "", quality_note, "",
             f"本文検証：{report['status']}（採用済み {len(commits)} 章）", ""]
    sections = [f"<h1>{html.escape(title)}</h1>",
                f"<p>{quality_note}</p>",
                f"<p>本文検証：{html.escape(report['status'])}（採用済み {len(commits)} 章）</p>"]
    notes = ["# 作者用の計画・実績・検査記録", "", "先の展開・秘密を含みます。", ""]
    for commit in commits:
        result = commit["envelope"]["result"]
        chapter = f"第{result['chapter_number']}章　{result['title']}"
        lines.extend([f"## {chapter}", ""])
        sections.append(f"<section><h2>{html.escape(chapter)}</h2>")
        for scene in result["scenes"]:
            heading = f"シーン {scene['id']}"
            lines.extend([f"### {heading}", "", "````text", scene["raw_text"], "````", ""])
            sections.append(f"<h3>{html.escape(heading)}</h3><pre>{html.escape(scene['raw_text'])}</pre>")
        sections.append("</section>")
        note = {key: value for key, value in result.items() if key != "scenes"}
        note["scenes"] = [{key: value for key, value in scene.items()
                           if key not in {"raw_text", "utterances"}} for scene in result["scenes"]]
        note["provenance"] = commit["envelope"]["provenance"]
        note["trace"] = commit["envelope"]["trace"]
        notes.extend([f"## {chapter}", "", "````json",
                      json.dumps(note, ensure_ascii=False, indent=2), "````", ""])
    if report.get("error"):
        lines.extend(["---", "", "続きの生成は停止しています。詳細は report.json と章の draft.md を参照してください。"])
    (output / "story.md").write_text("\n".join(lines), encoding="utf-8")
    (output / "author-notes.md").write_text("\n".join(notes), encoding="utf-8")
    page = ("<!doctype html><html lang='ja'><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            "<meta http-equiv='Content-Security-Policy' content=\"default-src 'none'; style-src 'unsafe-inline'\">"
            f"<title>{html.escape(title)}</title><style>body{{max-width:52rem;margin:3rem auto;"
            "padding:0 1rem;line-height:1.9;font-family:system-ui,sans-serif}}"
            "pre{white-space:pre-wrap;overflow-wrap:anywhere;font-family:inherit}"
            "section{margin:3rem 0;border-top:1px solid #aaa}</style>"
            + "\n".join(sections) + "</html>")
    (output / "story.html").write_text(page, encoding="utf-8")


def run_text_debug(input_value: dict, output_dir: Path, *, workflow: str = "causal",
                   context_size: int | None = None, chapter_limit: int | None = None,
                   resume: bool = False, seed: int | None = None,
                   policy: str | None = None, plot_only: bool = False) -> dict:
    """Generate only approved chapter text; resume verifies every adopted predecessor."""
    if policy == "script_continuation_v1":
        if workflow != "causal":
            raise ValueError("Script continuation requires the causal text-only workflow.")
        from .script_continuation import run_script_debug
        return run_script_debug(input_value, output_dir, context_size=context_size,
                                chapter_limit=chapter_limit, resume=resume, seed=seed,
                                plot_only=plot_only)
    if plot_only:
        raise ValueError("plot_only is available only for script_continuation_v1.")
    if policy == "story_draft_v1":
        if workflow != "causal":
            raise ValueError("Story draft requires the causal text-only workflow.")
        from .draft_story import run_draft_debug
        return run_draft_debug(input_value, output_dir, context_size=context_size,
                               chapter_limit=chapter_limit, resume=resume, seed=seed)
    snapshot, profile, source_seed = extract_input(input_value)
    limits = copy.deepcopy(_input_payload(input_value).get("workflow_limits", {}))
    resource_limits({"workflow_limits": limits, "approval_snapshot": snapshot})
    profiles = copy.deepcopy(_input_payload(input_value).get("profiles", {}))
    if not isinstance(profiles, dict) or any(
            not isinstance(purpose, str) or not purpose.strip() or not isinstance(value, dict)
            for purpose, value in profiles.items()):
        raise TypeError("Purpose profiles must map nonempty purpose names to profile objects.")
    if workflow not in {"legacy", "causal"}:
        raise ValueError("workflow must be legacy or causal.")
    protocol = generator_protocol(workflow, policy) if policy else generator_protocol(workflow)
    if context_size is not None:
        if type(context_size) is not int or context_size < 1024:
            raise ValueError("context_size must be an integer of at least 1024 tokens.")
        profile["context_size"] = context_size
    seed = source_seed if seed is None else seed
    if type(seed) is not int or not 0 <= seed < 2**63:
        raise ValueError("seed must be an integer between 0 and 2**63 - 1.")
    count = snapshot["world"]["result"]["chapterCount"]
    target = count if chapter_limit is None else chapter_limit
    if type(target) is not int or not 1 <= target <= count:
        raise ValueError("chapter_limit must be within the approved chapter count.")
    config = pipeline.load_config()
    if workflow == "causal" and config.get("llm_config"):
        config = {**config, "model_configuration_identity": model_configuration_identity(pipeline.ROOT, config)}
    model_config = (_read(pipeline.ROOT / config["llm_config"])
                    if config.get("llm_config") else None)
    identity = {"schema_version": 1, "execution_mode": "text_only", "workflow": workflow,
                "generator_protocol": protocol,
                "approval_snapshot": snapshot, "seed": seed, "profile": profile,
                "configured_profile": config.get("llm", {}),
                "model_config_sha256": _hash(model_config),
                "generation_config_sha256": _hash(config)}
    # Empty overrides are equivalent to omission. Keep existing default-profile
    # experiments resumable; explicit purpose settings become part of identity.
    if profiles:
        identity["profiles"] = profiles
    if policy is not None:
        identity["workflow_policy"] = policy
    if limits:
        identity["workflow_limits"] = limits
    fingerprint = _hash(identity)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with _output_lock(output):
        manifest_path = output / "experiment.json"
        if manifest_path.exists():
            if not resume:
                raise ValueError("Use a new output directory or --resume.")
            manifest = _read(manifest_path)
            if (manifest.get("profiles", {}) != profiles
                    or manifest.get("input_sha256") != fingerprint or _hash({
                    key: manifest.get(key) for key in identity}) != fingerprint):
                raise ValueError("Experiment input, seed, workflow, generator protocol, profile, or config changed.")
        else:
            if any(path.name != ".text-debug.lock" for path in output.iterdir()):
                raise ValueError("A new experiment requires an empty output directory.")
            manifest = {**identity, "profiles": profiles, "input_sha256": fingerprint,
                        "storyline_id": f"text-debug-{fingerprint[:32]}",
                        "approved_chapter_count": count}
            write_json(manifest_path, manifest)
        report_path = output / "report.json"
        report = _read(report_path) if report_path.exists() else {
            "schema_version": 1, "execution_mode": "text_only", "input_sha256": fingerprint,
            "storyline_id": manifest["storyline_id"], "workflow": workflow,
            "approved_chapter_count": count, "attempts": [],
            "public_build_created": False,
        }
        commits = []
        usage = (PersistentUsage(output / "jobs" / "chapter-001", _payload(manifest, 1, None), config)
                 if workflow == "causal" else None)
        if usage is not None:
            usage.check()
        chapters_dir = output / "chapters"
        chapters_dir.mkdir(exist_ok=True)
        for path in sorted(chapters_dir.glob("chapter-*.json")):
            number = len(commits) + 1
            if path.name != f"chapter-{number:03d}.json":
                raise ValueError("Adopted chapter sequence has a gap or unexpected file.")
            commit = _read(path)
            payload = _payload(manifest, number, commits[-1] if commits else None)
            result = _validate_result(commit["envelope"], payload, workflow=workflow)
            digest = _hash(result)
            if (commit.get("result_sha256") != digest
                    or commit.get("artifact_id") != f"text-debug-{digest}"
                    or commit.get("input_sha256") != _hash(payload)
                    or commit.get("end_state_hash") != story_state_hash(result["end_state"])):
                raise ValueError("Adopted chapter was changed or belongs to another input.")
            commits.append(commit)
        if len(commits) > target:
            raise ValueError("chapter_limit cannot precede an already adopted chapter.")

        def save(status: str, **details):
            report.update(status=status, target_chapter_count=target,
                          adopted_chapter_count=len(commits), development_stage="poc",
                          quality_acceptance="not_evaluated", human_review="not_performed",
                          generator_protocol=manifest["generator_protocol"],
                          automated_checks={"text_verified": "passed",
                              "chapter_limit_reached": "partial_passed"}.get(status, status), **details)
            aggregate = {}
            for key in ("elapsed_seconds", "new_request_cache_records", "prompt_tokens_new",
                        "completion_tokens_new", "prompt_processing_seconds_new",
                        "token_generation_seconds_new", "model_load_seconds_this_run",
                        "model_release_seconds_this_run", "model_loads_this_run",
                        "model_switches_this_run"):
                observed = [item["metrics"][key] for item in report["attempts"]
                            if item.get("metrics", {}).get(key) is not None]
                aggregate[key] = round(sum(observed), 3) if observed else None
            report["metrics"] = aggregate
            if usage is not None:
                report["resource_budget"] = usage.summary()
            write_json(report_path, report)
            _render(output, manifest, commits, report)

        report.pop("error", None)
        save("running")
        for number in range(len(commits) + 1, target + 1):
            work = output / "jobs" / f"chapter-{number:03d}"
            work.mkdir(parents=True, exist_ok=True)
            payload = _payload(manifest, number, commits[-1] if commits else None)
            job = {"id": payload["production_id"], "kind": "m3_narrative", "payload": payload}
            if usage is not None:
                usage = PersistentUsage(work, payload, config)
            write_json(work / "debug-job.json", job)
            previous_files = set(_request_records(work))
            previous_traces = _trace_files(work)
            cached_result = (work / "result.zip").is_file()
            started, envelope = time.monotonic(), None
            attempt = {"chapter_number": number, "attempt": 1 + sum(
                item["chapter_number"] == number for item in report["attempts"])}
            try:
                with usage.job_attempt() if usage is not None else nullcontext():
                    bundle = m3_pipeline.generate_job(job, work)
                if usage is not None:
                    usage.check()
                with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
                    envelope = json.loads(archive.read("result.json"))
                    if archive.namelist() != ["result.json"]:
                        raise ValueError("Text-only narrative result unexpectedly contains media.")
                result = _validate_result(envelope, payload, workflow=workflow)
                if usage is not None and not any(r["chapter"] == str(number) for r in usage.data["requests"]):
                    raise ResourceBudgetError("Causal chapter has no measured LLM request records.")
                digest = _hash(result)
                commit = {"schema_version": 1, "execution_mode": "text_only",
                          "input_sha256": _hash(payload), "result_sha256": digest,
                          "artifact_id": f"text-debug-{digest}",
                          "end_state_hash": story_state_hash(result["end_state"]),
                          "envelope": envelope}
                write_json(chapters_dir / f"chapter-{number:03d}.json", commit)
                commits.append(commit)
                attempt.update(status="adopted")
            except BaseException as exc:
                attempt.update(status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                               error=f"{type(exc).__name__}: {exc}")
                _draft_report(work)
                raise
            finally:
                attempt["metrics"] = _metrics(work, previous_files, envelope,
                                               time.monotonic() - started, previous_traces,
                                               cached_result=cached_result)
                report["attempts"].append(attempt)
                save(attempt["status"] if attempt["status"] != "adopted" else "running",
                     **({"error": attempt["error"]} if "error" in attempt else {}))
        save("text_verified" if target == count else "chapter_limit_reached")
        return report
