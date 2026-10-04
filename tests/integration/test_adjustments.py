"""Complete-story editions preserve original URLs and regenerate exactly changed voices."""
from __future__ import annotations

import copy
import io
import json
import sqlite3
import struct
import wave
from contextlib import nullcontext
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m3 import EMOTION_TAGS, M3_KINDS
from services.coordinator.app import create_app
from services.coordinator.service import Coordinator
from tests.integration.planning_fixtures import complete_and_approve_plan
from tests.integration.test_m2 import action, claim, png, report_voice_inventory
from tests.integration.test_m2_cast import ready_cast
from tests.integration.test_m3 import approved, complete, finish, output, production
from tests.integration.test_project_history import history, restore


@pytest.fixture
def story(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        endpoint = f"/api/m3/projects/{project}/adjustments"
        yield client, coordinator, project, worker, endpoint


def start(client, endpoint):
    response = client.post(endpoint + "/start", json={})
    assert response.status_code == 200, response.text
    return response.json()


def save(client, endpoint, value, characters):
    response = client.put(endpoint, json={"expected_revision": value["draft"]["revision"], "characters": characters})
    assert response.status_code == 200, response.text
    return response.json()


def apply(client, endpoint, value):
    response = client.post(endpoint + "/apply", json={"expected_revision": value["draft"]["revision"]})
    assert response.status_code == 200, response.text
    return response.json()


def recording():
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(struct.pack("<h", 700) * 12000)
    return data.getvalue()


def upload(client, endpoint, value, *, kind="image", cid="support-1"):
    response = client.post(endpoint + "/upload", data={"expected_revision": value["draft"]["revision"],
        "character_id": cid, "kind": kind, **({"reference_text": "新しい基準音声です。"} if kind == "voice" else {})},
        files={"file": ("candidate.wav" if kind == "voice" else "candidate.png",
                        recording() if kind == "voice" else png(225))})
    assert response.status_code == 200, response.text
    return response.json()


def ids(value):
    return [row["build"]["id"] for row in value["chapters"]]


def zip_chapters(client, build):
    response = client.get(f"/api/m3/builds/{build}/export")
    assert response.status_code == 200
    with ZipFile(io.BytesIO(response.content)) as archive:
        return json.loads(archive.read("chapters.json"))["chapters"]


def test_adjustments_require_all_chapters_and_reads_do_not_create_drafts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        endpoint = f"/api/m3/projects/{project}/adjustments"
        assert client.get(endpoint).json()["readonly"]
        assert client.post(endpoint + "/start", json={}).status_code == 409
        finish(client, worker)
        value = client.get(endpoint).json()
        assert value["complete"] and not value["readonly"] and value["draft"] is None
        with client.app.state.coordinator.db.transaction() as connection:
            assert connection.execute("SELECT COUNT(*) FROM adjustment_draft").fetchone()[0] == 0
            assert connection.execute("SELECT COUNT(*) FROM publication_edition").fetchone()[0] == 0


def test_editions_pin_next_zip_and_history_without_mutating_original(story):
    client, coordinator, project, _worker, endpoint = story
    original = production(client, project)
    originals = ids(original)
    old_zip = zip_chapters(client, originals[0])
    before_history = history(client, project)["current_revision_id"]
    original_bytes = {c["build"]["id"]: client.get(c["export_url"]).content for c in original["chapters"]}
    value = start(client, endpoint)
    changes = copy.deepcopy(value["draft"]["characters"])
    changes[0].update(offset_y=24, scale=1.15)
    value = save(client, endpoint, value, changes)
    value = apply(client, endpoint, value)
    assert value["draft"]["status"] == "applied"
    assert not value["busy"]
    adjusted = ids(production(client, project))
    assert all(new != old for new, old in zip(adjusted, originals, strict=True))
    assert [row["build_id"] for row in zip_chapters(client, adjusted[0])] == adjusted
    assert zip_chapters(client, originals[0]) == old_zip
    assert client.get(f"/api/m3/builds/{originals[0]}/next").json()["next_build"]["id"] == originals[1]
    assert client.get(f"/api/m3/builds/{adjusted[0]}/next").json()["next_build"]["id"] == adjusted[1]
    for c in original["chapters"]:
        assert client.get(c["export_url"]).content == original_bytes[c["build"]["id"]]
    applied_history = history(client, project)["current_revision_id"]
    restore(client, project, before_history)
    assert ids(production(client, project)) == originals
    assert client.get(endpoint).json()["draft"] is None
    restore(client, project, applied_history)
    restored = client.get(endpoint).json()
    assert ids(production(client, project)) == adjusted
    assert restored["edition"]["id"] == value["edition"]["id"]
    assert restored["draft"]["status"] == "applied"
    with coordinator.db.transaction() as connection, pytest.raises(sqlite3.IntegrityError):
        connection.execute("UPDATE publication_edition SET state='{}' WHERE id=?", (restored["edition"]["id"],))


def test_voice_upload_regenerates_every_affected_utterance_before_atomic_switch(story):
    client, coordinator, project, worker, endpoint = story
    original = production(client, project)
    value = start(client, endpoint)
    value = upload(client, endpoint, value, kind="voice")
    candidate = next(row for row in value["candidates"] if row["source"] == "upload")
    assert candidate["reference_text"] == "新しい基準音声です。"
    assert candidate["original_artifact_id"] and candidate["artifact_id"]
    sample = claim(client, worker)
    assert sample["payload"]["adjustment"]["purpose"] == "sample"
    assert complete(client, worker, sample, output(sample, voice_patch={"reference_sha256": "0" * 64})).status_code == 422
    assert complete(client, worker, sample).status_code == 200
    assert client.get(endpoint).json()["candidates"][-1]["sample_url"] is not None
    changes = copy.deepcopy(value["draft"]["characters"])
    selected = next(row for row in changes if row["character_id"] == "support-1")
    selected["voice_candidate_id"] = candidate["id"]
    value = save(client, endpoint, value, changes)
    value = apply(client, endpoint, value)
    assert value["draft"]["status"] == "applying"
    assert ids(production(client, project)) == ids(original)
    with coordinator.db.transaction() as connection:
        expected = sum(1 for row in connection.execute("SELECT descriptor FROM m3_requirement r JOIN m3_production p ON p.id=r.production_id "
            "WHERE p.project_id=? AND r.kind='m3_voice_clone'", (project,)) if json.loads(row[0])["character_id"] == "support-1")
        original_requirements = [tuple(row) for row in connection.execute("SELECT id,artifact_id FROM m3_requirement ORDER BY id")]
    jobs = []
    while job := claim(client, worker):
        assert job["kind"] == "m3_voice_clone"
        assert job["payload"]["adjustment"]["purpose"] == "dialogue"
        assert job["payload"]["character_id"] == "support-1"
        assert job["payload"]["reference_voice"]["artifact_id"] == candidate["artifact_id"]
        if len(jobs) < expected - 1:
            assert ids(production(client, project)) == ids(original)
        assert complete(client, worker, job).status_code == 200
        jobs.append(job)
    assert len(jobs) == expected
    value = client.get(endpoint).json()
    assert value["draft"]["status"] == "applied" and value["edition"]["id"] != value["draft"]["base_edition_id"]
    assert ids(production(client, project)) != ids(original)
    with coordinator.db.transaction() as connection:
        assert [tuple(row) for row in connection.execute("SELECT id,artifact_id FROM m3_requirement ORDER BY id")] == original_requirements
    for old, new in zip(original["chapters"], production(client, project)["chapters"], strict=True):
        with ZipFile(io.BytesIO(client.get(old["export_url"]).content)) as archive:
            old_text = archive.read("narrative.json")
        with ZipFile(io.BytesIO(client.get(new["export_url"]).content)) as archive:
            assert archive.read("narrative.json") == old_text


def test_candidate_revision_conflicts_upload_validation_and_stale_sample(story):
    client, _coordinator, _project, worker, endpoint = story
    value = start(client, endpoint)
    voice = upload(client, endpoint, value, kind="voice")
    candidate = next(row for row in voice["candidates"] if row["source"] == "upload")
    job = claim(client, worker)
    # Saving newer controls cannot let an old sample callback overwrite the candidate.
    voice = save(client, endpoint, voice, voice["draft"]["characters"])
    assert complete(client, worker, job).status_code == 200
    assert next(row for row in client.get(endpoint).json()["candidates"] if row["id"] == candidate["id"])["sample_url"] is None
    request = {"expected_revision": voice["draft"]["revision"], "candidate_id": candidate["id"]}
    sample = client.post(endpoint + "/sample", json=request)
    assert sample.status_code == 200
    assert sample.json()["draft"]["revision"] == request["expected_revision"] + 1
    assert client.post(endpoint + "/sample", json=request).status_code == 409
    assert complete(client, worker, claim(client, worker)).status_code == 200
    assert next(row for row in client.get(endpoint).json()["candidates"] if row["id"] == candidate["id"])["sample_url"]
    assert client.post(endpoint + "/upload", content=b"bad", headers={"Content-Length": "broken"}).status_code == 422
    invalid = client.post(endpoint + "/upload", data={"expected_revision": 1, "character_id": "support-1", "kind": "image"},
                          files={"file": ("bad.png", b"invalid")})
    assert invalid.status_code == 422
    edits = copy.deepcopy(voice["draft"]["characters"])
    edits[0]["voice_candidate_id"] = candidate["id"]
    current = client.get(endpoint).json()
    assert client.put(endpoint, json={"expected_revision": current["draft"]["revision"], "characters": edits}).status_code == 422


def test_failed_candidate_retry_keeps_one_time_settings_unchanged(story):
    client, coordinator, project, worker, endpoint = story
    value = start(client, endpoint)
    settings = copy.deepcopy(value["cast"])
    response = client.post(endpoint + "/generate", json={"expected_revision": value["draft"]["revision"],
        "character_id": "support-1", "kind": "voice", "instruction": "今回だけささやく声にしてください。"})
    assert response.status_code == 200, response.text
    value = response.json()
    job = claim(client, worker)
    assert job["payload"]["instruction"] == "今回だけささやく声にしてください。"
    assert "source_prompt" not in job["payload"]
    pending = next(row for row in value["candidates"] if row["source"] == "generated")
    source = job["payload"]["character_result"]["voice"]
    assert pending["prompt_details"] == {"source": source, "input": source,
        "instruction": job["payload"]["instruction"], "effective": None, "baseline": None}
    with coordinator.db.transaction() as connection:
        connection.execute("UPDATE job SET max_attempts=1 WHERE id=?", (job["id"],))
    failed = client.post(f"/api/jobs/{job['id']}/fail", json={"worker_id": worker, "lease_id": job["lease_id"], "error": "test failure"})
    assert failed.status_code == 200
    retry = client.post(endpoint + "/retry", json={"expected_revision": value["draft"]["revision"]})
    assert retry.status_code == 200, retry.text
    assert retry.json()["draft"]["revision"] > value["draft"]["revision"]
    retried = claim(client, worker)
    assert retried["id"] == job["id"] and retried["attempt"] == 2
    assert complete(client, worker, retried, b"bad zip").status_code == 422
    assert complete(client, worker, retried).status_code == 200
    sample = claim(client, worker)
    assert sample["payload"]["adjustment"]["purpose"] == "sample"
    assert complete(client, worker, sample).status_code == 200
    current = client.get(endpoint).json()
    assert [row["result"] for row in current["cast"]] == [row["result"] for row in settings]
    assert ids(production(client, project)) == [row["build_id"] for row in value["edition"]["chapters"]]


@pytest.mark.parametrize("kind,cid", [("image", "character-1"), ("voice", "support-1")])
def test_direct_prompt_candidate_preserves_source_effective_prompt_and_history(story, kind, cid):
    client, coordinator, project, worker, endpoint = story
    value = start(client, endpoint)
    person = next(row for row in value["cast"] if row["character_id"] == cid)
    original = copy.deepcopy(person["result"])
    source = original["appearance" if kind == "image" else "voice"]
    assert person["source_prompts"][kind] == source
    selected = next(row for row in value["draft"]["characters"] if row["character_id"] == cid)
    original_candidate = next(row for row in value["candidates"] if row["id"] == selected[kind + "_candidate_id"])
    assert original_candidate["prompt_details"]["effective"] is None
    assert original_candidate["prompt_details"]["baseline"] is None
    baseline = "original full-body prompt" if kind == "image" else "元の穏やかな声の説明"
    # M3 aliases retain an approved-artifact reference instead of duplicating
    # the original image prompt. Both representations must resolve correctly.
    with coordinator.db.transaction() as connection:
        artifact = connection.execute("SELECT id,provenance FROM artifact WHERE id=?",
                                      (original_candidate["artifact_id"],)).fetchone()
        provenance = json.loads(artifact["provenance"])
        identifier = provenance.get("approved_artifact_id", artifact["id"])
        provenance = json.loads(connection.execute("SELECT provenance FROM artifact WHERE id=?", (identifier,)).fetchone()[0])
        provenance.setdefault(kind, {})["prompt" if kind == "image" else "caption"] = baseline
        connection.execute("UPDATE artifact SET provenance=? WHERE id=?", (json.dumps(provenance), identifier))
    value = client.get(endpoint).json()
    original_candidate = next(row for row in value["candidates"] if row["id"] == original_candidate["id"])
    assert original_candidate["prompt_details"] == {
        "source": source, "input": source, "instruction": "", "effective": baseline, "baseline": baseline,
    }
    edited = "足元まで見える青いドレス姿。" if kind == "image" else "ゆっくり話す落ち着いた低い声。"
    response = client.post(endpoint + "/generate", json={"expected_revision": value["draft"]["revision"],
        "character_id": cid, "kind": kind, "source_prompt": edited})
    assert response.status_code == 200, response.text
    pending = response.json()
    candidate = next(row for row in pending["candidates"] if row["source"] == "generated")
    expected = {"source": source, "input": edited, "instruction": "", "effective": None, "baseline": baseline}
    assert candidate["prompt"] == "" and candidate["prompt_details"] == expected
    job = claim(client, worker)
    assert job["payload"]["source_prompt"] == edited
    assert job["payload"]["instruction"] == ""
    assert job["payload"]["character_result"] == original
    effective = "full body, blue dress, shoes visible" if kind == "image" else edited
    data = io.BytesIO()
    with ZipFile(io.BytesIO(output(job))) as prior, ZipFile(data, "w") as archive:
        for name in prior.namelist():
            content = prior.read(name)
            if name == "result.json":
                envelope = json.loads(content)
                envelope["provenance"]["prompt_details"] = {"source": source, "input": edited,
                    "instruction": "", "effective": effective}
                content = json.dumps(envelope).encode()
            archive.writestr(name, content)
    assert complete(client, worker, job, data.getvalue()).status_code == 200
    if kind == "voice":
        sample = claim(client, worker)
        assert sample["payload"]["adjustment"]["purpose"] == "sample"
        assert complete(client, worker, sample).status_code == 200
    value = client.get(endpoint).json()
    expected["effective"] = effective
    candidate = next(row for row in value["candidates"] if row["id"] == candidate["id"])
    assert candidate["prompt_details"] == expected
    assert candidate["metadata"]["prompt_details"] == expected
    original_selection = copy.deepcopy(value["draft"]["characters"])
    changes = copy.deepcopy(original_selection)
    next(row for row in changes if row["character_id"] == cid)[kind + "_candidate_id"] = candidate["id"]
    value = save(client, endpoint, value, changes)
    selected_history = history(client, project)["current_revision_id"]
    save(client, endpoint, value, original_selection)
    restore(client, project, selected_history)
    value = client.get(endpoint).json()
    assert next(row for row in value["draft"]["characters"] if row["character_id"] == cid)[kind + "_candidate_id"] == candidate["id"]
    assert next(row for row in value["candidates"] if row["id"] == candidate["id"])["prompt_details"] == expected
    assert next(row for row in value["cast"] if row["character_id"] == cid)["result"] == original


@pytest.mark.parametrize("fields", [{}, {"instruction": " \n\t"}, {"source_prompt": ""},
    {"source_prompt": " \n\t", "instruction": "新しい候補を作る"}, {"source_prompt": "x" * 10001}])
def test_candidate_generation_requires_nonempty_instruction_or_edited_prompt(story, fields):
    client, _coordinator, _project, _worker, endpoint = story
    value = start(client, endpoint)
    response = client.post(endpoint + "/generate", json={"expected_revision": value["draft"]["revision"],
        "character_id": "support-1", "kind": "image", **fields})
    assert response.status_code == 422
    current = client.get(endpoint).json()
    assert current["draft"]["revision"] == value["draft"]["revision"]
    assert all(row["source"] == "original" for row in current["candidates"])


def test_voice_apply_failure_holds_original_and_retries_only_failed_job(story):
    client, coordinator, _project, worker, endpoint = story
    value = start(client, endpoint)
    original_edition = value["edition"]["id"]
    value = upload(client, endpoint, value, kind="voice")
    assert complete(client, worker, claim(client, worker)).status_code == 200
    candidate = next(row for row in value["candidates"] if row["source"] == "upload")
    changes = copy.deepcopy(value["draft"]["characters"])
    next(row for row in changes if row["character_id"] == "support-1")["voice_candidate_id"] = candidate["id"]
    value = apply(client, endpoint, save(client, endpoint, value, changes))
    failed_job = claim(client, worker)
    with coordinator.db.transaction() as connection:
        connection.execute("UPDATE job SET max_attempts=1 WHERE id=?", (failed_job["id"],))
    assert client.post(f"/api/jobs/{failed_job['id']}/fail", json={"worker_id": worker,
        "lease_id": failed_job["lease_id"], "error": "failed utterance"}).status_code == 200
    completed = finish(client, worker)
    value = client.get(endpoint).json()
    assert value["draft"]["status"] == "failed" and value["edition"]["id"] == original_edition
    retry = client.post(endpoint + "/retry", json={"expected_revision": value["draft"]["revision"]})
    assert retry.status_code == 200, retry.text
    remaining = finish(client, worker)
    assert len(remaining) == 1 and remaining[0]["id"] == failed_job["id"]
    assert completed
    value = client.get(endpoint).json()
    assert value["draft"]["status"] == "applied" and value["edition"]["id"] != original_edition


def test_unused_approved_main_characters_remain_available_for_adjustment(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready_cast(client, count=2)
        action(client, project, "approve")
        pid = project["project"]["id"]
        complete_and_approve_plan(client, project)
        worker = client.post("/api/workers", json={"name": "adjustment fixture", "capabilities": [*M3_KINDS, "tts_download"]}).json()["id"]
        report_voice_inventory(client, worker)
        finish(client, worker)
        value = start(client, f"/api/m3/projects/{pid}/adjustments")
        person = next(row for row in value["cast"] if row["character_id"] == "person-2")
        assert person["role"] == "main" and person["chapter_numbers"] == []
        assert person["image_url"] and person["voice_url"] and person["reference_text"]
        assert "person-2" in value["draft"]["geometry"]["characters"]


def test_candidate_callbacks_preserve_stale_material_but_never_enqueue_stale_samples(story):
    client, coordinator, _project, worker, endpoint = story
    value = start(client, endpoint)
    response = client.post(endpoint + "/generate", json={"expected_revision": value["draft"]["revision"],
        "character_id": "support-1", "kind": "voice", "instruction": "今回だけ明るい声にする。"})
    assert response.status_code == 200
    value = response.json()
    job = claim(client, worker)
    moment = [coordinator.clock()]
    coordinator.clock = lambda: moment[0]
    moment[0] += coordinator.lease_seconds + 1
    assert complete(client, worker, job).status_code == 409
    renewed = claim(client, worker)
    assert renewed["id"] == job["id"] and renewed["lease_id"] != job["lease_id"]
    save(client, endpoint, value, value["draft"]["characters"])
    assert complete(client, worker, renewed).status_code == 200
    current = client.get(endpoint).json()
    candidate = next(row for row in current["candidates"] if row["source"] == "generated")
    assert candidate["artifact_id"] and candidate["sample_url"] is None
    assert claim(client, worker) is None


def test_pending_apply_resumes_after_restart_and_new_draft_preserves_published_editions(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        finish(client, worker)
        endpoint = f"/api/m3/projects/{project}/adjustments"
        value = upload(client, endpoint, start(client, endpoint), kind="voice")
        assert complete(client, worker, claim(client, worker)).status_code == 200
        candidate = next(row for row in value["candidates"] if row["source"] == "upload")
        changes = copy.deepcopy(value["draft"]["characters"])
        next(row for row in changes if row["character_id"] == "support-1")["voice_candidate_id"] = candidate["id"]
        pending = apply(client, endpoint, save(client, endpoint, value, changes))
        source = pending["edition"]["id"]
        baseline = ids(production(client, project))
        assert pending["draft"]["status"] == "applying"
    with TestClient(create_app(tmp_path)) as client:
        assert ids(production(client, project)) == baseline
        restored = client.get(endpoint).json()
        assert restored["draft"]["status"] == "applying" and restored["edition"]["id"] == source
        remaining = finish(client, worker)
        assert remaining and all(job["payload"]["adjustment"]["purpose"] == "dialogue" for job in remaining)
        adjusted = client.get(endpoint).json()
        assert adjusted["draft"]["status"] == "applied"
        edition = adjusted["edition"]["id"]
        first_adjustment = ids(production(client, project))
        opened = client.post(endpoint + "/start", json={"expected_edition_id": edition})
        assert opened.status_code == 200, opened.text
        opened = opened.json()
        assert opened["draft"]["id"] != adjusted["draft"]["id"]
        assert opened["draft"]["characters"] == adjusted["draft"]["characters"]
        changes = copy.deepcopy(opened["draft"]["characters"])
        changes[0]["offset_y"] = -13
        second = apply(client, endpoint, save(client, endpoint, opened, changes))
        assert second["edition"]["source_edition_id"] == edition
        assert second["edition"]["id"] != edition
        assert [c["build_id"] for c in zip_chapters(client, first_adjustment[0])] == first_adjustment
    with TestClient(create_app(tmp_path)) as client:
        assert client.get(endpoint).json()["edition"]["id"] == second["edition"]["id"]
        assert [c["build_id"] for c in zip_chapters(client, baseline[0])] == baseline


def test_real_pipeline_retry_preserves_input_fingerprint_and_adopts_candidate_and_sample(
    story, tmp_path, monkeypatch,
):
    # Keep the actual worker request fingerprints and disk caches. Only model
    # inference and its GPU lease are replaced; coordinator leases stay real.
    from services.worker.generation import generate_job, pipeline

    client, coordinator, _project, worker, endpoint = story
    attempts = {"candidate": 0, "sample": 0}

    def voice(payload, _work, _config):
        attempts["candidate"] += 1
        if attempts["candidate"] == 1:
            raise RuntimeError("inference failed after writing the immutable request")
        return recording(), {"reference_text": payload["reference_text"]}

    def clone(payload, _work, _config):
        attempts["sample"] += 1
        if attempts["sample"] == 1:
            raise RuntimeError("sample inference failed after writing the immutable request")
        reference = payload["reference_voice"]
        return recording(), {"text": EMOTION_TAGS[payload["tts_emotion"]] + payload["dialogue_text"],
            "reference_artifact_id": reference["artifact_id"], "reference_sha256": reference["sha256"],
            "reference_text": reference["text"]}

    monkeypatch.setattr(pipeline, "load_config", lambda: {"gpu_lock_timeout_seconds": 1, "max_zip_bytes": 1_000_000})
    monkeypatch.setattr(pipeline, "gpu_lock", lambda *_args: nullcontext())
    monkeypatch.setattr(pipeline, "generate_voice", voice)
    monkeypatch.setattr(pipeline, "generate_voice_clone", clone)
    value = start(client, endpoint)
    generated = client.post(endpoint + "/generate", json={"expected_revision": value["draft"]["revision"],
        "character_id": "support-1", "kind": "voice", "instruction": "この回だけ明るくしてください。"})
    assert generated.status_code == 200
    value = generated.json()
    for purpose in ("candidate", "sample"):
        claimed = claim(client, worker)
        assert claimed["payload"]["adjustment"]["purpose"] == purpose
        frozen_payload = copy.deepcopy(claimed["payload"])
        work = tmp_path / "worker" / claimed["id"]
        with pytest.raises(RuntimeError, match="immutable request"):
            generate_job(claimed, work)
        fingerprint = (work / "job-request.json").read_bytes()
        with coordinator.db.transaction() as connection:
            connection.execute("UPDATE job SET max_attempts=1 WHERE id=?", (claimed["id"],))
        assert client.post(f"/api/jobs/{claimed['id']}/fail", json={"worker_id": worker,
            "lease_id": claimed["lease_id"], "error": "runtime failed"}).status_code == 200
        retried = client.post(endpoint + "/retry", json={"expected_revision": value["draft"]["revision"]})
        assert retried.status_code == 200, retried.text
        value = retried.json()
        next_attempt = claim(client, worker)
        assert next_attempt["id"] == claimed["id"]
        assert next_attempt["payload"] == frozen_payload
        with coordinator.db.transaction() as connection:
            adoption_revision = connection.execute("SELECT adoption_revision FROM adjustment_job WHERE job_id=?",
                                                   (claimed["id"],)).fetchone()[0]
        assert adoption_revision == value["draft"]["revision"]
        assert adoption_revision > frozen_payload["adjustment"]["revision"]
        bundle = generate_job(next_attempt, work)
        assert (work / "job-request.json").read_bytes() == fingerprint
        assert generate_job(next_attempt, work) == bundle  # valid disk cache reused
        assert complete(client, worker, next_attempt, bundle).status_code == 200
        value = client.get(endpoint).json()
    candidate = next(row for row in value["candidates"] if row["source"] == "generated")
    assert candidate["artifact_id"] and candidate["sample_url"]
    assert attempts == {"candidate": 2, "sample": 2}


def test_candidate_only_upload_and_sample_receive_history_entries_without_selecting_assets(story):
    client, _coordinator, project, worker, endpoint = story
    value = start(client, endpoint)
    selected = value["draft"]["characters"]
    baseline = history(client, project)
    value = upload(client, endpoint, value)
    uploaded = history(client, project)
    assert len(uploaded["entries"]) == len(baseline["entries"]) + 1
    assert next(entry for entry in uploaded["entries"] if entry["current"])["label"] == "素材候補をアップロード"
    candidate = next(row for row in value["candidates"] if row["kind"] == "voice")
    sample = client.post(endpoint + "/sample", json={"expected_revision": value["draft"]["revision"], "candidate_id": candidate["id"]})
    assert sample.status_code == 200
    assert complete(client, worker, claim(client, worker)).status_code == 200
    sampled = history(client, project)
    assert len(sampled["entries"]) == len(uploaded["entries"]) + 1
    assert next(entry for entry in sampled["entries"] if entry["current"])["label"] == "選択した声で台詞を試聴"
    current = client.get(endpoint).json()
    assert current["draft"]["characters"] == selected
    assert any(row["sample_url"] for row in current["candidates"])
