"""Multi-chapter production and recovery through real API leases with fixture media."""

import copy
import hashlib
import io
import json
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m3 import M3_KINDS
from packages.narrative import story_state_hash
from services.coordinator.app import create_app
from services.coordinator.service import Coordinator
from tests.integration.test_m2 import action, claim, ready, report_voice_inventory
from tests.integration.test_m3 import complete, narrative, output, pin_legacy_production, production
from tests.integration.test_m3 import finish as finish_default
from tests.integration.test_project_history import history, restore


def approve(client):
    project, _ = ready(client)
    action(client, project, "approve")
    # These fixtures exercise the historical StoryState/semantic-review contract.
    # New script production is covered separately without rewriting old histories.
    pin_legacy_production(client, project["project"]["id"])
    worker = client.post("/api/workers", json={
        "name": "M4 fixture", "capabilities": [*M3_KINDS, "tts_download"],
    }).json()["id"]
    report_voice_inventory(client, worker)
    return project["project"]["id"], worker


def chapter(job):
    payload = job["payload"]
    result = narrative(payload["approval_snapshot"])
    number = payload["chapter_number"]
    previous = payload.get("previous_narrative")
    if previous:
        result["outline"] = copy.deepcopy(previous["outline"])
        result["supporting_characters"] = copy.deepcopy(previous["supporting_characters"])
        start = copy.deepcopy(previous["end_state"])
    else:
        result["outline"]["foreshadowing"] = [{
            "setup_chapter": 1, "payoff_chapter": 3, "detail": "鍵の刻印は家族の印。",
        }]
        start = {
            "schema_version": 1, "chapter_number": 0, "summary": "閉じた図書館を訪れる。",
            "facts": [{"id": "family", "detail": "鍵の刻印は記録官の家族の印。"}],
            "characters": [{
                "character_id": cid, "location": "図書館", "relationships": [],
                "possessions": [], "injuries": [], "promises": [], "inner_emotion": "緊張",
                "knowledge": [{"fact_id": "family", "acquired_chapter": 0,
                               "evidence_utterance_ids": []}] if cid != "support-1" else [],
            } for cid in (payload["approval_snapshot"]["characters"][0]["id"], "support-1")],
            "foreshadowing": [{"outline_index": 0, "status": "pending", "changed_chapter": 0,
                               "evidence_utterance_ids": []}],
        }
    end = copy.deepcopy(start)
    end.update(chapter_number=number, summary=f"第{number}章の話し合いが終わった。")
    evidence = result["scenes"][0]["utterances"][0]["id"]
    if number in (1, 3):
        end["foreshadowing"] = [{
            "outline_index": 0, "status": "resolved" if number == 3 else "planted",
            "changed_chapter": number, "evidence_utterance_ids": [evidence],
        }]
    if number == 2:
        end["characters"][1]["knowledge"] = [{
            "fact_id": "family", "acquired_chapter": 2, "evidence_utterance_ids": [evidence],
        }]
    result.update(
        chapter_number=number, title=f"第{number}章の記録", storyline_id=payload["storyline_id"],
        previous_narrative_artifact_id=payload.get("previous_narrative_artifact_id"),
        previous_state_hash=payload.get("previous_state_hash"), start_state=start, end_state=end,
        continuity_review={"passed": True, "issues": [],
                           "checked_character_ids": [c["character_id"] for c in end["characters"]],
                           "checked_foreshadowing_indices": [0]},
    )
    return result


def finish_job(client, worker, job):
    data = output(job, result=chapter(job)) if job["kind"] == "m3_narrative" else output(job)
    response = complete(client, worker, job, data)
    assert response.status_code == 200, response.text
    return data


def finish(client, worker):
    completed = []
    for _ in range(150):
        job = claim(client, worker)
        if not job:
            return completed
        finish_job(client, worker, job)
        completed.append(job)
    pytest.fail("M4 fixture did not drain within the expected job budget")


def test_three_chapters_publish_independently_keep_lineage_and_export_a_completed_prefix(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approve(client)
        first_job = claim(client, worker)
        assert first_job["payload"]["m4"] is True
        finish_job(client, worker, first_job)
        state = production(client, project)
        assert len(state["chapters"]) == 3
        assert state["chapters"][0]["status"] == "generating_assets"
        assert state["chapters"][0]["build"] is None
        next_job = state["chapters"][1]["jobs"][0]
        assert next_job["kind"] == "m3_narrative" and next_job["status"] == "pending"
        assert next_job["payload"]["previous_narrative_artifact_id"] == state["chapters"][0]["narrative_artifact_id"]
        assert next_job["payload"]["previous_state_hash"] == story_state_hash(chapter(first_job)["end_state"])
        writer = client.post("/api/workers", json={
            "name": "parallel writer", "capabilities": ["m3_narrative"],
        }).json()["id"]
        second_job = claim(client, writer)
        assert second_job["id"] == next_job["id"]  # May run before the first chapter's media.
        finish(client, worker)
        first_state = production(client, project)
        first_build = first_state["chapters"][0]["build"]
        assert first_build and first_state["chapters"][1]["build"] is None
        first_zip = client.get(first_state["chapters"][0]["export_url"]).content
        next_url = f"/api/m3/builds/{first_build['id']}/next"
        assert client.get(next_url).json() == {
            "build_id": first_build["id"], "status": "waiting", "next_build": None,
        }
        with ZipFile(io.BytesIO(client.get(f"/api/m3/builds/{first_build['id']}/export").content)) as archive:
            entries = json.loads(archive.read("chapters.json"))
            assert len(entries["chapters"]) == 1 and entries["terminal"]
            assert archive.read("chapter-001.zip") == first_zip
        finish_job(client, writer, second_job)
        finish(client, worker)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert [c["status"] for c in state["chapters"]] == ["published"] * 3
        assert state["chapters"][0]["build"] == first_build
        assert client.get(first_state["chapters"][0]["export_url"]).content == first_zip
        for index, current in enumerate(state["chapters"]):
            response = client.get(f"/api/m3/builds/{current['build']['id']}/next").json()
            assert set(response) == {"build_id", "status", "next_build"}
            assert response["status"] == ("complete" if index == 2 else "ready")
            if index < 2:
                assert response["next_build"]["id"] == state["chapters"][index + 1]["build"]["id"]
                assert set(response["next_build"]) == {"id", "chapter_number", "player_url", "export_url"}
                validation = state["chapters"][index + 1]["build"]["validation"]
                assert validation["previous_narrative_artifact_id"] == current["narrative_artifact_id"]
                assert validation["previous_state_hash"] == current["build"]["validation"]["end_state_hash"]
        with ZipFile(io.BytesIO(client.get(f"/api/m3/builds/{first_build['id']}/export").content)) as archive:
            entries = json.loads(archive.read("chapters.json"))["chapters"]
            assert [e["chapter_number"] for e in entries] == [1, 2, 3]
            for entry in entries:
                assert hashlib.sha256(archive.read(entry["filename"])).hexdigest() == entry["sha256"]
        assert sum(j["kind"] == "m3_narrative" for j in state["jobs"]) == 3
        # Shared supporting cast and backgrounds are adopted across chapters.
        for kind, expected in (("m3_image", 1), ("m3_voice", 1), ("m3_background", 2)):
            assert sum(j["kind"] == kind for j in state["jobs"]) == expected


def test_graceful_stop_adopts_inflight_result_without_assigning_more_work(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approve(client)
        job = claim(client, worker)
        response = client.post(f"/api/m3/projects/{project}/stop", json={"mode": "graceful"})
        assert response.status_code == 200, response.text
        assert response.json()["production"]["control_state"] == "stopping"
        assert claim(client, worker) is None
        finish_job(client, worker, job)
        paused = production(client, project)
        assert paused["control_state"] == "paused"
        assert paused["chapters"][0]["narrative_artifact_id"]
        assert len(paused["jobs"]) == 1 and paused["jobs"][0]["status"] == "completed"
        assert claim(client, worker) is None
        adopted = paused["chapters"][0]["narrative_artifact_id"]
        response = client.post(f"/api/m3/projects/{project}/resume")
        assert response.status_code == 200, response.text
        finish(client, worker)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert state["chapters"][0]["narrative_artifact_id"] == adopted
        assert client.get(f"/api/jobs/{job['id']}").json()["job"]["attempt_count"] == 1


def test_immediate_interrupt_rejects_stale_result_and_restarts_only_unfinished_attempt(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approve(client)
        story = claim(client, worker)
        finish_job(client, worker, story)
        adopted = production(client, project)["chapters"][0]["narrative_artifact_id"]
        media = claim(client, worker)
        assert media["kind"] != "m3_narrative"
        response = client.post(f"/api/m3/projects/{project}/stop", json={"mode": "immediate"})
        assert response.status_code == 200, response.text
        assert response.json()["production"]["control_state"] == "interrupted"
        assert complete(client, worker, media).status_code == 409
        assert claim(client, worker) is None
        assert client.post(f"/api/m3/projects/{project}/resume").status_code == 200
        retried = claim(client, worker)
        assert retried["id"] == media["id"] and retried["lease_id"] != media["lease_id"]
        assert complete(client, worker, media).status_code == 409
        finish_job(client, worker, retried)
        finish(client, worker)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert state["chapters"][0]["narrative_artifact_id"] == adopted
        assert client.get(f"/api/jobs/{story['id']}").json()["job"]["attempt_count"] == 1
        assert client.get(f"/api/jobs/{media['id']}").json()["job"]["attempt_count"] == 2


def test_restart_keeps_completed_outputs_and_persistent_pause_without_regeneration(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approve(client)
        finish_job(client, worker, claim(client, worker))
        finish_job(client, worker, claim(client, worker))
        assert client.post(f"/api/m3/projects/{project}/stop", json={"mode": "graceful"}).status_code == 200
        before = production(client, project)
        completed = {j["id"] for j in before["jobs"] if j["status"] == "completed"}
        assert before["control_state"] == "paused" and len(completed) == 2
    with TestClient(create_app(coordinator=Coordinator(tmp_path))) as client:
        paused = production(client, project)
        assert paused["control_state"] == "paused"
        assert {j["id"] for j in paused["jobs"] if j["status"] == "completed"} == completed
        assert claim(client, worker) is None
        assert client.post(f"/api/m3/projects/{project}/resume").status_code == 200
        following = finish(client, worker)
        assert not completed.intersection(j["id"] for j in following)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        builds = [c["build"] for c in state["chapters"]]
        for identifier in completed:
            assert client.get(f"/api/jobs/{identifier}").json()["job"]["attempt_count"] == 1
    with TestClient(create_app(tmp_path)) as client:
        assert [c["build"] for c in production(client, project)["chapters"]] == builds
        assert claim(client, worker) is None


@pytest.mark.parametrize("damage", ["storyline", "artifact", "state_hash", "start_state", "knowledge"])
def test_continuation_cannot_adopt_wrong_lineage_or_unearned_character_knowledge(tmp_path, damage):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approve(client)
        first = claim(client, worker)
        finish_job(client, worker, first)
        writer = client.post("/api/workers", json={
            "name": "continuity writer", "capabilities": ["m3_narrative"],
        }).json()["id"]
        second = claim(client, writer)
        result = chapter(second)
        if damage == "storyline":
            result["storyline_id"] = "wrong-storyline"
        elif damage == "artifact":
            result["previous_narrative_artifact_id"] = "wrong-artifact"
        elif damage == "state_hash":
            result["previous_state_hash"] = "0" * 64
        elif damage == "start_state":
            result["start_state"]["summary"] = "前章の終わりを書き換えた。"
        else:
            result["end_state"]["characters"][1]["knowledge"][0]["acquired_chapter"] = 0
        assert complete(client, writer, second, output(second, result=result)).status_code == 422
        state = production(client, project)
        assert state["chapters"][0]["narrative_artifact_id"]
        assert state["chapters"][1]["narrative_artifact_id"] is None
        assert state["chapters"][1]["build"] is None


@pytest.mark.parametrize("damage", ["previous_narrative_artifact_id", "previous_state_hash"])
def test_next_chapter_skips_newer_builds_with_incompatible_predecessor(tmp_path, damage):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approve(client)
        finish(client, worker)
        state = production(client, project)
        first, second = [c["build"] for c in state["chapters"][:2]]
        # Simulate another published revision. The reader must compare immutable
        # lineage rather than selecting the numerically newest chapter build.
        with coordinator.db.transaction() as connection:
            row = dict(connection.execute("SELECT * FROM chapter_build WHERE id=?", (second["id"],)).fetchone())
            row["id"] = "incompatible-build"
            row["revision"] += 1
            validation = json.loads(row["validation"])
            validation[damage] = "incompatible"
            row["validation"] = json.dumps(validation)
            columns = ",".join(row)
            placeholders = ",".join("?" for _ in row)
            connection.execute(f"INSERT INTO chapter_build ({columns}) VALUES ({placeholders})", tuple(row.values()))
        response = client.get(f"/api/m3/builds/{first['id']}/next")
        assert response.status_code == 200
        assert response.json()["next_build"]["id"] == second["id"]
        assert response.headers["cache-control"] == "no-store"


@pytest.mark.parametrize("legacy_manifest", [False, True])
def test_legacy_first_chapter_requires_explicit_resume_and_keeps_original_outputs(tmp_path, legacy_manifest):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approve(client)
        # Build the pre-M4 shape through the real lease/adoption pipeline. This
        # models a migrated production whose old narrative has no StoryState.
        with coordinator.db.transaction() as connection:
            root = connection.execute("SELECT * FROM m3_production WHERE project_id=?", (project,)).fetchone()
            connection.execute("UPDATE m3_production SET m4_enabled=0 WHERE id=?", (root["id"],))
            job = connection.execute("SELECT * FROM job WHERE project_id=? AND kind='m3_narrative'", (project,)).fetchone()
            payload = json.loads(job["payload"])
            payload["m4"] = False
            connection.execute("UPDATE job SET payload=? WHERE id=?", (json.dumps(payload), job["id"]))
        first_jobs = finish_default(client, worker)
        if legacy_manifest:
            # Pre-M4 builds do not carry lineage in validation. Insert that old
            # row shape as a fixture, retaining exactly the same source ZIP.
            with coordinator.db.transaction() as connection:
                row = dict(connection.execute(
                    "SELECT * FROM chapter_build WHERE project_id=?", (project,)).fetchone())
                row["id"] = "legacy-first-build"
                row["revision"] += 1
                validation = json.loads(row["validation"])
                row["validation"] = json.dumps({key: validation[key] for key in (
                    "schema_version", "raw_text_mapping", "references", "media",
                    "content_review", "required_assets")})
                connection.execute(
                    f"INSERT INTO chapter_build ({','.join(row)}) VALUES ({','.join('?' for _ in row)})",
                    tuple(row.values()))
                connection.execute("UPDATE project_history_state SET build_id=? WHERE project_id=?",
                                   (row["id"], project))
        state = production(client, project)
        first = state["chapters"][0]
        assert first["build"] and first["build"]["chapter_number"] == 1
        assert all(c["production_id"] is None for c in state["chapters"][1:])
        legacy_text = client.get(f"/api/artifacts/{first['narrative_artifact_id']}/content").content
        assert json.loads(legacy_text)["end_state"] is None
        old_zip = client.get(first["export_url"]).content
        with coordinator.db.transaction() as connection:
            old_artifacts = {row["id"]: row["sha256"] for row in connection.execute(
                "SELECT id,sha256 FROM artifact WHERE project_id=?", (project,))}
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        state = production(client, project)
        assert state["chapters"][0]["build"] == first["build"]
        assert all(c["production_id"] is None for c in state["chapters"][1:])
        assert claim(client, worker) is None
        assert client.get(f"/api/m3/builds/{first['build']['id']}/next").json()["status"] == "waiting"
        response = client.post(f"/api/m3/projects/{project}/resume")
        assert response.status_code == 200, response.text
        next_job = claim(client, worker)
        assert next_job["kind"] == "m3_narrative" and next_job["payload"]["chapter_number"] == 2
        assert next_job["payload"]["previous_narrative"]["end_state"] is None
        assert next_job["payload"]["previous_state_hash"] is None
        assert complete(client, worker, next_job).status_code == 200
        later_jobs = finish_default(client, worker)
        completed = production(client, project)
        assert completed["status"] == "published", completed["error"]
        assert completed["chapters"][0]["build"] == first["build"]
        assert client.get(first["export_url"]).content == old_zip
        assert client.get(f"/api/artifacts/{first['narrative_artifact_id']}/content").content == legacy_text
        assert not {j["id"] for j in first_jobs}.intersection(j["id"] for j in later_jobs)
        second = completed["chapters"][1]
        second_text = client.get(f"/api/artifacts/{second['narrative_artifact_id']}/content").json()
        assert second_text["predecessor_state_inferred"] is True
        assert second_text["previous_state_hash"] == story_state_hash(second_text["start_state"])
        next_build = client.get(f"/api/m3/builds/{first['build']['id']}/next").json()
        assert next_build["status"] == "ready" and next_build["next_build"]["id"] == second["build"]["id"]
        with coordinator.db.transaction() as connection:
            artifacts = {row["id"]: row["sha256"] for row in connection.execute(
                "SELECT id,sha256 FROM artifact WHERE project_id=?", (project,))}
        assert all(artifacts.get(identifier) == digest for identifier, digest in old_artifacts.items())
        for old_job in first_jobs:
            assert client.get(f"/api/jobs/{old_job['id']}").json()["job"]["attempt_count"] == 1


def test_restoring_stopped_chapter_snapshot_hides_later_content_and_does_not_resume(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approve(client)
        finish_job(client, worker, claim(client, worker))
        # Publish chapter one while chapter two is still waiting for its writer.
        for _ in range(40):
            if production(client, project)["chapters"][0]["build"]:
                break
            job = claim(client, worker)
            assert job is not None and job["payload"]["chapter_number"] == 1
            finish_job(client, worker, job)
        assert client.post(f"/api/m3/projects/{project}/stop", json={"mode": "graceful"}).status_code == 200
        stopped = production(client, project)
        assert stopped["control_state"] == "paused"
        assert stopped["chapters"][0]["build"] and stopped["chapters"][1]["build"] is None
        assert stopped["chapters"][1]["jobs"][0]["status"] == "pending"
        stopped_history = history(client, project)
        assert not stopped_history["busy"] and stopped_history["pending_operation"] is None
        old_revision = stopped_history["current_revision_id"]
        assert client.post(f"/api/m3/projects/{project}/resume").status_code == 200
        finish(client, worker)
        completed = production(client, project)
        assert completed["status"] == "published", completed["error"]
        assert history(client, project)["current_revision_id"] != old_revision
        later_urls = [c["export_url"] for c in completed["chapters"][1:]]
        assert client.post(f"/api/m3/projects/{project}/stop", json={"mode": "graceful"}).status_code == 200
        with coordinator.db.transaction() as connection:
            before = {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                      for name in ("artifact", "job", "chapter_build")}
        restore(client, project, old_revision)
        restored = production(client, project)
        assert restored["history_frozen"] and restored["control_state"] == "paused"
        assert restored["chapters"] == stopped["chapters"]
        assert restored["total_jobs"] == stopped["total_jobs"]
        assert restored["completed_jobs"] == stopped["completed_jobs"]
        assert {j["id"]: j["status"] for j in restored["jobs"]} == {
            j["id"]: j["status"] for j in stopped["jobs"]}
        assert claim(client, worker) is None
        # A newer live result cannot silently become the restored old chapter.
        assert client.post(f"/api/m3/projects/{project}/resume").status_code == 409
        assert all(client.get(url).status_code == 200 for url in later_urls)
        with coordinator.db.transaction() as connection:
            after = {name: connection.execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                     for name in before}
        assert before == after
    with TestClient(create_app(tmp_path)) as client:
        assert production(client, project)["chapters"] == stopped["chapters"]
        assert claim(client, worker) is None
