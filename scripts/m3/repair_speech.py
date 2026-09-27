"""Prepare a speech-only chapter repair, then atomically publish with --apply.

Original approval, source artifacts, jobs and published builds remain immutable.
Only the current production's adopted narrative and media requirements advance.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sqlite3
import sys
import zipfile
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from packages.contracts.m3 import EMOTION_TAGS, NarrativeResult
from packages.narrative.repair import remap_scene, voice_reuse_map
from packages.narrative.speech import separate_stage_directions
from packages.narrative.validation import validate_narrative
from services.coordinator.m3_bundle import validate_bundle
from services.coordinator.m3_service import M3Service
from services.coordinator.service import Coordinator, encode_json, required
from services.worker.generation import m3_pipeline, pipeline
from services.worker.generation.llm import LocalLLM, write_json
from services.worker.generation.narrative import _review_scene, separate_speech
from services.worker.generation.processes import gpu_lock
from services.worker.generation.voice_session import reuse_voice_runtime


def prepare(coordinator: Coordinator, project_id: str, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    path = output / "repair.json"
    if path.exists():
        report = json.loads(path.read_text(encoding="utf-8"))
        if report["project_id"] != project_id:
            raise ValueError("Repair directory belongs to another project.")
        return report
    service = M3Service(coordinator)
    with coordinator.db.transaction() as connection:
        row = connection.execute("SELECT * FROM m3_production WHERE project_id=? ORDER BY created_at DESC",
                                 (project_id,)).fetchone()
        if row is None:
            raise ValueError("Project has no production.")
        production = dict(row)
        build = service._latest_build(connection, production["id"])
        if build is None:
            raise ValueError("Repair requires a published chapter.")
        build = dict(build)
        snapshot = service._snapshot(connection, production)
        original_record = required(connection, "artifact", production["narrative_artifact_id"])
        original = validate_narrative(json.loads(coordinator.store.read(original_record)), snapshot)
        requirements = [dict(row) for row in connection.execute(
            "SELECT * FROM m3_requirement WHERE production_id=? ORDER BY rowid", (production["id"],))]
        audio_jobs = {r["target_id"]: required(connection, "job", r["job_id"])
                      if r["job_id"] else {"payload": r["descriptor"]}
                      for r in requirements if r["kind"] == "m3_voice_clone"}
        artifacts = {r["artifact_id"]: required(connection, "artifact", r["artifact_id"])
                     for r in requirements}
        narrative_job = dict(connection.execute(
            "SELECT job.* FROM job JOIN m3_production_job p ON p.job_id=job.id "
            "WHERE p.production_id=? AND kind='m3_narrative'", (production["id"],)).fetchone())
    write_json(output / "before.json", {"production": production, "build": build,
        "requirements": requirements, "narrative": original.model_dump(mode="json")})
    payload = json.loads(narrative_job["payload"])
    config = pipeline.load_config()
    names = {c["id"]: c["result"]["name"] for c in snapshot["characters"]}
    names.update({c.id: c.name for c in original.supporting_characters})
    context = json.dumps({"approval": snapshot,
        "supporting_characters": [c.model_dump(mode="json") for c in original.supporting_characters]},
        ensure_ascii=False)
    scenes, reuses, origins, changes = [], {}, {}, []
    with gpu_lock(ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]):  # noqa: SIM117
        with LocalLLM(ROOT, config, payload, output / "llm") as llm:
            for scene in original.scenes:
                separation, hints = separate_speech(llm, context, scene.plan, scene.raw_text, names)
                repaired = remap_scene(scene, separation, hints)
                if separation.raw_text != scene.raw_text:
                    review = _review_scene(llm, context, scene.plan, repaired.utterances)
                    repaired = repaired.model_copy(update={"review": review})
                scenes.append(repaired)
                reuses.update(voice_reuse_map(scene, repaired, separation))
                for segment in separation.segments:
                    if segment.utterance_id and segment.speaker_id:
                        origins[segment.utterance_id] = segment.original_utterance_id
                changes.append({"scene_id": scene.id, "raw_text": separation.raw_text,
                    "utterance_map": separation.utterance_map,
                    "segments": [asdict(s) for s in separation.segments], "delivery_hints": hints})
            write_json(output / "classification-trace.json", llm.trace)
    candidate = validate_narrative(original.model_copy(update={"scenes": scenes}), snapshot)
    write_json(output / "candidate-narrative.json", candidate.model_dump(mode="json"))
    write_json(output / "changes.json", changes)
    originals = {r["target_id"]: r for r in requirements if r["kind"] == "m3_voice_clone"}
    audios = []
    with reuse_voice_runtime():
        for scene in candidate.scenes:
            for utterance in scene.utterances:
                if utterance.speaker_id is None:
                    continue
                old_id = origins[utterance.id]
                old = originals[old_id]
                source_payload = json.loads(audio_jobs[old_id]["payload"])
                audio_payload = {**source_payload, "dialogue_text": utterance.spoken_text,
                    "voice_emotion": utterance.voice_emotion, "delivery": utterance.delivery}
                record = {"target_id": utterance.id, "original_target_id": old_id,
                    "descriptor": audio_payload, "original_artifact_id": old["artifact_id"]}
                if reuses.get(utterance.id) == old_id:
                    record["reuse_artifact_id"] = old["artifact_id"]
                else:
                    work = output / "voice" / utterance.id
                    work.mkdir(parents=True, exist_ok=True)
                    reference = source_payload["reference_voice"]
                    reference_record = artifacts[reference["artifact_id"]]
                    if reference_record["sha256"] != reference["sha256"]:
                        raise ValueError("Immutable voice reference changed.")
                    (work / "reference-voice.wav").write_bytes(coordinator.store.read(reference_record))
                    content = m3_pipeline.generate_job({"id": "speech-repair-" + utterance.id,
                        "kind": "m3_voice_clone", "payload": audio_payload}, work)
                    envelope, _ = validate_bundle(content, "m3_voice_clone")
                    with zipfile.ZipFile(io.BytesIO(content)) as bundle:
                        audio = bundle.read("voice.wav")
                    (work / "repaired.wav").write_bytes(audio)
                    record.update(file=str((work / "repaired.wav").relative_to(output)),
                        sha256=hashlib.sha256(audio).hexdigest(), provenance=envelope["provenance"])
                    print(f"Repaired voice: {utterance.id}", flush=True)
                audios.append(record)
    report = {"project_id": project_id, "production_id": production["id"],
        "original_build_id": build["id"], "original_narrative_id": original_record["id"],
        "candidate_sha256": hashlib.sha256((output / "candidate-narrative.json").read_bytes()).hexdigest(),
        "prepared_file_sha256": {name: hashlib.sha256((output / name).read_bytes()).hexdigest()
            for name in ("candidate-narrative.json", "before.json", "changes.json", "classification-trace.json")},
        "audios": audios, "reused_audio_count": sum("reuse_artifact_id" in a for a in audios),
        "regenerated_audio_count": sum("file" in a for a in audios),
        "removed_audio_count": len(originals) - len(audios)}
    write_json(path, report)
    return report


def _prepared_files(output: Path, report: dict) -> dict[str, bytes]:
    names = {"candidate-narrative.json", "before.json", "changes.json", "classification-trace.json"}
    hashes = report.get("prepared_file_sha256")
    if not isinstance(hashes, dict) or set(hashes) != names:
        raise ValueError("Prepared repair must contain every audited file digest.")
    files = {name: (output / name).read_bytes() for name in names}
    if any(hashlib.sha256(data).hexdigest() != hashes[name] for name, data in files.items()):
        raise ValueError("Prepared repair files changed after validation.")
    return files


def _voice_provenance(voice: dict, payload: dict) -> None:
    reference = payload["reference_voice"]
    expected = {
        "spoken_text": payload["dialogue_text"],
        "text": EMOTION_TAGS[payload["voice_emotion"]] + payload["dialogue_text"],
        "voice_emotion": payload["voice_emotion"],
        "reference_artifact_id": reference["artifact_id"],
        "reference_sha256": reference["sha256"],
        "reference_text": reference["text"],
    }
    if any(voice.get(key) != value for key, value in expected.items()):
        raise ValueError("Repair audio provenance differs from its speech or reference.")


def _publication_preflight(coordinator, connection, production, latest, candidate, output,
                           report, files) -> dict[str, bytes]:
    service = M3Service(coordinator)
    snapshot = service._snapshot(connection, production)
    validate_narrative(candidate, snapshot)
    original_record = required(connection, "artifact", production["narrative_artifact_id"])
    original = validate_narrative(json.loads(coordinator.store.read(original_record)), snapshot)
    requirements = [dict(row) for row in connection.execute(
        "SELECT * FROM m3_requirement WHERE production_id=? ORDER BY rowid", (production["id"],))]
    before = json.loads(files["before.json"])
    if (before.get("production") != production or before.get("build") != dict(latest)
            or before.get("requirements") != requirements
            or before.get("narrative") != original.model_dump(mode="json")):
        raise ValueError("Repair source snapshot no longer matches the current production.")
    changes = json.loads(files["changes.json"])
    if [change.get("scene_id") for change in changes] != [scene.id for scene in original.scenes]:
        raise ValueError("Repair source mappings must cover every scene exactly once.")
    names = {c["id"]: c["result"]["name"] for c in snapshot["characters"]}
    names.update({c.id: c.name for c in original.supporting_characters})
    scenes, origins, reuses = [], {}, {}
    for old_scene, new_scene, change in zip(original.scenes, candidate.scenes, changes, strict=True):
        segments = change["segments"]
        directions = {s["candidate_id"]: s["display_text"] for s in segments
                      if s["kind"] == "narration" and s["candidate_id"] is not None}
        deliveries = [s["candidate_id"] for s in segments if s["kind"] == "delivery"]
        separation = separate_stage_directions(
            old_scene.raw_text, old_scene.id, set(old_scene.plan.character_ids), directions, names,
            direction_narrations=directions, delivery_candidate_ids=deliveries,
        )
        if (separation.raw_text != change["raw_text"]
                or separation.utterance_map != change["utterance_map"]
                or [asdict(s) for s in separation.segments] != segments):
            raise ValueError("Repair source mapping changed or rewrote unselected dialogue.")
        repaired = remap_scene(old_scene, separation, change["delivery_hints"])
        scenes.append(repaired.model_copy(update={"review": new_scene.review}))
        reuses.update(voice_reuse_map(old_scene, repaired, separation))
        for segment in separation.segments:
            if segment.utterance_id and segment.speaker_id:
                origins[segment.utterance_id] = segment.original_utterance_id
    if original.model_copy(update={"scenes": scenes}).model_dump() != candidate.model_dump():
        raise ValueError("Candidate narrative changed beyond the audited speech repair.")
    spoken = {u.id: u for scene in candidate.scenes for u in scene.utterances if u.speaker_id}
    audios = report["audios"]
    targets = [audio["target_id"] for audio in audios]
    if len(targets) != len(set(targets)) or set(targets) != set(spoken):
        raise ValueError("Repair audio must cover every spoken utterance exactly once.")
    originals = {r["target_id"]: r for r in requirements if r["kind"] == "m3_voice_clone"}
    references = {r["target_id"]: r for r in requirements if r["kind"] == "m3_voice"}
    generated = {}
    for audio in audios:
        utterance = spoken[audio["target_id"]]
        old_id = origins[utterance.id]
        old = originals[old_id]
        if (audio["original_target_id"] != old_id
                or audio["original_artifact_id"] != old["artifact_id"]):
            raise ValueError("Repair audio references a different original utterance.")
        source_payload = json.loads(required(connection, "job", old["job_id"])["payload"]
                                    if old["job_id"] else old["descriptor"])
        expected_payload = {**source_payload, "dialogue_text": utterance.spoken_text,
                            "voice_emotion": utterance.voice_emotion, "delivery": utterance.delivery}
        if (audio["descriptor"] != expected_payload
                or expected_payload["character_id"] != utterance.speaker_id):
            raise ValueError("Repair audio descriptor differs from its source and candidate speech.")
        reference = expected_payload["reference_voice"]
        reference_record = required(connection, "artifact", reference["artifact_id"])
        reference_metadata = json.loads(reference_record["provenance"]).get("voice", {})
        if (references[utterance.speaker_id]["artifact_id"] != reference_record["id"]
                or reference_record["project_id"] != production["project_id"]
                or reference_record["sha256"] != reference["sha256"]
                or (reference_metadata.get("reference_text") or reference_metadata.get("text"))
                != reference["text"]):
            raise ValueError("Repair uses a changed or mismatched immutable voice reference.")
        coordinator.store.read(reference_record)
        reuse = audio.get("reuse_artifact_id")
        if bool(reuse) == ("file" in audio):
            raise ValueError("Repair audio must be either reused or regenerated.")
        if reuse:
            if reuse != old["artifact_id"] or reuses.get(utterance.id) != old_id:
                raise ValueError("Reused audio must exactly match its original speech inputs.")
            reused = required(connection, "artifact", reuse)
            if reused["project_id"] != production["project_id"] or reused["kind"] != "audio":
                raise ValueError("Reused audio must belong to the repaired project.")
            _voice_provenance(json.loads(reused["provenance"]).get("voice", {}), expected_payload)
            coordinator.store.read(reused)
        else:
            audio_path = (output / audio["file"]).resolve()
            if not audio_path.is_relative_to(output.resolve()):
                raise ValueError("Repair audio must remain inside its output directory.")
            data = audio_path.read_bytes()
            if hashlib.sha256(data).hexdigest() != audio["sha256"]:
                raise ValueError("Prepared audio changed after validation.")
            _voice_provenance(audio["provenance"].get("voice", {}), expected_payload)
            generated[utterance.id] = data
    counts = {"reused_audio_count": len(audios) - len(generated),
              "regenerated_audio_count": len(generated),
              "removed_audio_count": len(originals) - len(audios)}
    if any(report.get(key) != value for key, value in counts.items()):
        raise ValueError("Repair audio counts differ from the validated media plan.")
    return generated


def publish(coordinator: Coordinator, output: Path, report: dict) -> dict:
    service = M3Service(coordinator)
    files = _prepared_files(output, report)
    raw = files["candidate-narrative.json"]
    if hashlib.sha256(raw).hexdigest() != report["candidate_sha256"]:
        raise ValueError("Prepared narrative changed after validation.")
    candidate = NarrativeResult.model_validate_json(raw)
    with coordinator.db.transaction() as connection:
        production = required(connection, "m3_production", report["production_id"])
        latest = service._latest_build(connection, production["id"])
        if (production["project_id"] != report["project_id"]
                or production["narrative_artifact_id"] != report["original_narrative_id"]
                or latest is None or latest["id"] != report["original_build_id"]):
            raise ValueError("Production changed while repair was being prepared.")
        if connection.execute("SELECT 1 FROM job WHERE project_id=? AND status IN ('pending','running')",
                              (production["project_id"],)).fetchone():
            raise ValueError("Cannot publish while project jobs are active.")
        generated = _publication_preflight(
            coordinator, connection, production, latest, candidate, output, report, files,
        )
        repair_provenance = {"producer": "speech-separation-repair/1",
            "original_narrative_id": report["original_narrative_id"],
            "original_build_id": report["original_build_id"]}
        service._artifact(connection, production, "speech-repair-audit-" + production["id"],
            "speech_repair_audit", "speech-repair.json", encode_json({
                "report": report, "changes": json.loads(files["changes.json"]),
                "before": json.loads(files["before.json"]),
                "trace": json.loads(files["classification-trace.json"]),
            }), provenance=repair_provenance)
        record = service._artifact(connection, production, "narrative-" + production["id"],
            "m3_narrative", "narrative.json", encode_json(candidate.model_dump(mode="json")),
            provenance=repair_provenance)
        connection.execute("UPDATE m3_production SET narrative_artifact_id=? WHERE id=?",
                           (record["id"], production["id"]))
        production["narrative_artifact_id"] = record["id"]
        # The old requirement table is captured in the immutable audit artifact.
        # Published builds keep their original manifest, files and voice references.
        connection.execute("DELETE FROM m3_requirement WHERE production_id=? AND kind='m3_voice_clone'",
                           (production["id"],))
        for audio in report["audios"]:
            artifact_id = audio.get("reuse_artifact_id")
            if not artifact_id:
                data = generated[audio["target_id"]]
                artifact = service._artifact(connection, production,
                    "speech-repair-" + production["id"] + "-" + audio["target_id"], "audio", "voice.wav", data,
                    provenance={**repair_provenance, **audio["provenance"],
                                "original_artifact_id": audio["original_artifact_id"]})
                artifact_id = artifact["id"]
            service._requirement(connection, production, "m3_voice_clone", audio["target_id"],
                                 audio["descriptor"], artifact_id)
        requirements = [dict(row) for row in connection.execute(
            "SELECT * FROM m3_requirement WHERE production_id=? ORDER BY rowid", (production["id"],))]
        service._publish(connection, production, requirements)
        result = dict(service._latest_build(connection, production["id"]))
    write_json(output / "published.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    coordinator = Coordinator(args.data_dir.resolve())
    output = args.output_dir.resolve()
    report = prepare(coordinator, args.project_id, output)
    print(json.dumps({k: v for k, v in report.items() if k != "audios"}, ensure_ascii=False), flush=True)
    if args.apply:
        with sqlite3.connect(args.data_dir / "auto_drama.db") as source:  # noqa: SIM117
            with sqlite3.connect(output / "before-publication.db") as destination:
                source.backup(destination)
        print(json.dumps(publish(coordinator, output, report), ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
