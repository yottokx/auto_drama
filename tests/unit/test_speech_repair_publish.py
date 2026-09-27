"""Prepared speech repairs must validate completely before replacing adopted media."""

import copy
import hashlib
import io
import json
from dataclasses import asdict
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m3 import NarrativeResult
from packages.narrative import parse_scene_text, validate_narrative
from packages.narrative.repair import remap_scene, voice_reuse_map
from packages.narrative.speech import separate_stage_directions
from scripts.m3.repair_speech import publish
from services.coordinator.app import create_app
from services.coordinator.m3_service import M3Service
from services.coordinator.service import encode_json, required
from tests.integration.test_m2 import claim
from tests.integration.test_m3 import approved, complete, finish, narrative, output


def database_state(coordinator):
    with coordinator.db.transaction() as connection:
        return {table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")]
                for table in ("m3_production", "m3_requirement", "chapter_build", "artifact", "job")}


@pytest.fixture
def repair(tmp_path):
    with TestClient(create_app(tmp_path / "data")) as client:
        project, worker = approved(client)
        job = claim(client, worker)
        value = narrative(job["payload"]["approval_snapshot"])
        scene = value["scenes"][0]
        prefix = scene["plan"]["character_ids"][0] + ": "
        scene["raw_text"] = scene["raw_text"].replace(prefix, prefix + "（眼鏡を直す）", 1)
        scene["utterances"] = [u.model_dump() for u in parse_scene_text(
            scene["raw_text"], scene["id"], set(scene["plan"]["character_ids"]),
        )]
        response = complete(client, worker, job, output(job, value))
        assert response.status_code == 200, response.text
        finish(client, worker)
        coordinator = client.app.state.coordinator
        service = M3Service(coordinator)
        with coordinator.db.transaction() as connection:
            production = dict(connection.execute(
                "SELECT * FROM m3_production WHERE project_id=?", (project,),
            ).fetchone())
            build = dict(service._latest_build(connection, production["id"]))
            snapshot = service._snapshot(connection, production)
            record = required(connection, "artifact", production["narrative_artifact_id"])
            original = NarrativeResult.model_validate_json(coordinator.store.read(record))
            requirements = [dict(row) for row in connection.execute(
                "SELECT * FROM m3_requirement WHERE production_id=? ORDER BY rowid",
                (production["id"],),
            )]
            jobs = {r["target_id"]: json.loads(required(connection, "job", r["job_id"])["payload"])
                    for r in requirements if r["kind"] == "m3_voice_clone"}
        names = {c["id"]: c["result"]["name"] for c in snapshot["characters"]}
        names.update({c.id: c.name for c in original.supporting_characters})
        changes, scenes, reuses, origins = [], [], {}, {}
        for scene in original.scenes:
            selected = ["scene-1-u1-p1"] if scene.id == "scene-1" else []
            narrations = {uid: "主人公が眼鏡を直した。" for uid in selected}
            separation = separate_stage_directions(
                scene.raw_text, scene.id, set(scene.plan.character_ids), selected, names,
                direction_narrations=narrations,
            )
            repaired = remap_scene(scene, separation)
            scenes.append(repaired)
            reuses.update(voice_reuse_map(scene, repaired, separation))
            origins.update({s.utterance_id: s.original_utterance_id for s in separation.segments
                            if s.utterance_id and s.speaker_id})
            changes.append({"scene_id": scene.id, "raw_text": separation.raw_text,
                            "utterance_map": separation.utterance_map,
                            "segments": [asdict(s) for s in separation.segments],
                            "delivery_hints": {}})
        candidate = validate_narrative(original.model_copy(update={"scenes": scenes}), snapshot)
        originals = {r["target_id"]: r for r in requirements if r["kind"] == "m3_voice_clone"}
        directory = tmp_path / "repair"
        directory.mkdir()
        audios = []
        for scene in candidate.scenes:
            for utterance in scene.utterances:
                if not utterance.speaker_id:
                    continue
                old_id = origins[utterance.id]
                descriptor = {**jobs[old_id], "dialogue_text": utterance.spoken_text,
                              "voice_emotion": utterance.voice_emotion, "delivery": utterance.delivery}
                audio = {"target_id": utterance.id, "original_target_id": old_id,
                         "original_artifact_id": originals[old_id]["artifact_id"],
                         "descriptor": descriptor}
                if utterance.id in reuses:
                    audio["reuse_artifact_id"] = originals[old_id]["artifact_id"]
                else:
                    with ZipFile(io.BytesIO(output({"kind": "m3_voice_clone", "payload": descriptor}))) as bundle:
                        data = bundle.read("voice.wav")
                        provenance = json.loads(bundle.read("result.json"))["provenance"]
                    filename = utterance.id + ".wav"
                    (directory / filename).write_bytes(data)
                    audio.update(file=filename, sha256=hashlib.sha256(data).hexdigest(),
                                 provenance=provenance)
                audios.append(audio)
        prepared = {"candidate-narrative.json": candidate.model_dump(mode="json"),
                    "before.json": {"production": production, "build": build,
                                    "requirements": requirements,
                                    "narrative": original.model_dump(mode="json")},
                    "changes.json": changes, "classification-trace.json": []}
        for filename, content in prepared.items():
            (directory / filename).write_bytes(encode_json(content))
        hashes = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                  for name in prepared}
        report = {"project_id": project, "production_id": production["id"],
                  "original_build_id": build["id"], "original_narrative_id": record["id"],
                  "candidate_sha256": hashes["candidate-narrative.json"],
                  "prepared_file_sha256": hashes, "audios": audios,
                  "reused_audio_count": 7, "regenerated_audio_count": 1, "removed_audio_count": 0}
        yield coordinator, directory, report, client


def test_publish_preserves_original_build_artifacts_jobs_and_replay(repair):
    coordinator, directory, report, client = repair
    before = database_state(coordinator)
    result = publish(coordinator, directory, report)
    assert result["revision"] == 2
    after = database_state(coordinator)
    assert after["job"] == before["job"]
    assert all(row in after["artifact"] for row in before["artifact"])
    assert all(row in after["chapter_build"] for row in before["chapter_build"])
    assert len(after["chapter_build"]) == len(before["chapter_build"]) + 1
    assert len(after["m3_requirement"]) == len(before["m3_requirement"])
    for build in after["chapter_build"]:
        response = client.get(f"/api/artifacts/{build['export_artifact_id']}/content")
        assert response.status_code == 200
        with ZipFile(io.BytesIO(response.content)) as archive:
            assert "narrative.json" in archive.namelist()
    assert json.loads((directory / "published.json").read_text(encoding="utf-8")) == result


@pytest.mark.parametrize("mutation", [
    lambda r: r["audios"][1].update(reuse_artifact_id=r["audios"][2]["reuse_artifact_id"]),
    lambda r: r["audios"][0]["descriptor"].update(dialogue_text="別の台詞です。"),
    lambda r: r["audios"][0]["provenance"]["voice"].update(reference_artifact_id="wrong"),
    lambda r: r["audios"][0]["provenance"]["voice"].update(text="別の台詞です。"),
    lambda r: r["audios"][0].update(original_target_id=r["audios"][1]["original_target_id"]),
    lambda r: r["audios"].pop(),
    lambda r: r["audios"].append(copy.deepcopy(r["audios"][0])),
    lambda r: r.update(reused_audio_count=100),
    lambda r: r.pop("prepared_file_sha256"),
])
def test_malformed_report_rejected_before_any_database_change(repair, mutation):
    coordinator, directory, report, _ = repair
    before = database_state(coordinator)
    mutation(report)
    with pytest.raises(ValueError):
        publish(coordinator, directory, report)
    assert database_state(coordinator) == before


@pytest.mark.parametrize("filename", [
    "candidate-narrative.json", "before.json", "changes.json", "classification-trace.json",
])
def test_changed_prepared_files_are_rejected(repair, filename):
    coordinator, directory, report, _ = repair
    before = database_state(coordinator)
    with (directory / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="files changed"):
        publish(coordinator, directory, report)
    assert database_state(coordinator) == before


def test_changed_live_requirements_are_rejected(repair):
    coordinator, directory, report, _ = repair
    with coordinator.db.transaction() as connection:
        connection.execute("UPDATE m3_requirement SET descriptor='{}' WHERE target_id='scene-1-u1'")
    before = database_state(coordinator)
    with pytest.raises(ValueError, match="snapshot"):
        publish(coordinator, directory, report)
    assert database_state(coordinator) == before


def test_failure_during_bundle_publication_rolls_back_adoption(repair, monkeypatch):
    coordinator, directory, report, _ = repair
    before = database_state(coordinator)

    def fail_publication(*args):
        raise RuntimeError("export failed")

    monkeypatch.setattr(M3Service, "_publish", fail_publication)
    with pytest.raises(RuntimeError, match="export failed"):
        publish(coordinator, directory, report)
    assert database_state(coordinator) == before
