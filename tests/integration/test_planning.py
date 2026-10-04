"""STEP4 gates, immutable approved seeds and stale planning operations."""
import copy
import io
import json
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from services.coordinator.m2_service import M2Service
from services.coordinator.planning_service import PlanningService
from services.coordinator.service import Coordinator
from tests.integration.planning_fixtures import plan_content
from tests.integration.test_m2 import action, claim, ready


def endpoint(project):
    return f"/api/planning/projects/{project['project']['id']}"


def planning(client, project):
    response = client.get(endpoint(project))
    assert response.status_code == 200, response.text
    return response.json()["planning"]


def plan_action(client, project, name, status=200, **values):
    revision = planning(client, project)["revision"]
    response = client.post(endpoint(project) + "/actions", json={
        "action": name, "expected_revision": revision, **values,
    })
    assert response.status_code == status, response.text
    return response.json()


def plan_worker(client):
    return client.post("/api/workers", json={
        "name": "common planning fixture", "capabilities": ["m3_plan"],
    }).json()["id"]


def bundle(job, content=None, protocol=None):
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr("result.json", json.dumps({
            "schema_version": 1, "kind": "m3_plan",
            "result": content or plan_content(job["payload"]["approval_snapshot"]),
            "provenance": {"planning_protocol": protocol or job["payload"]["planning_protocol"]},
            "trace": [],
        }, ensure_ascii=False))
    return data.getvalue()


def complete_plan(client, job, worker, *, content=None, status=200, protocol=None):
    response = client.post(f"/api/m3/jobs/{job['id']}/complete",
        params={"worker_id": worker, "lease_id": job["lease_id"]},
        content=bundle(job, content, protocol), headers={"Content-Type": "application/zip"})
    assert response.status_code == status, response.text
    return response


def reviewed(client):
    project, _ = ready(client)
    project = action(client, project, "approve")
    worker = plan_worker(client)
    job = claim(client, worker)
    assert job["kind"] == "m3_plan"
    complete_plan(client, job, worker)
    return project, worker, job


def test_step3_only_enqueues_common_plan_and_all_public_starts_require_approval(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        project = action(client, project, "approve")
        pid = project["project"]["id"]
        state = planning(client, project)
        assert state["status"] == "generating" and state["content"] is None
        assert project["draft"]["planningRequired"] and not project["draft"]["planApproved"]
        assert project["draft"]["step"] == "planning-review"
        assert [job["kind"] for job in project["jobs"] if job["status"] == "pending"] == ["m3_plan"]
        assert client.get(f"/api/m3/projects/{pid}").json()["production"] is None
        assert client.post(f"/api/m3/projects/{pid}/start").status_code == 409
        assert client.post(f"/api/m3/projects/{pid}/resume").status_code == 409
        action(client, project, "go-to", step="production", status=409)
        duplicate = action(client, project, "approve")
        assert len(duplicate["jobs"]) == len(project["jobs"])
        assert planning(client, project)["revision"] == state["revision"]


def test_direct_edit_approval_is_idempotent_and_seed_is_immutable(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(tmp_path, coordinator=coordinator)) as client:
        project, _, job = reviewed(client)
        original = planning(client, project)
        content = copy.deepcopy(original["content"])
        content["plot"]["core"]["central_question"] = "編集済みの問い。"
        saved = plan_action(client, project, "save", content=content)["planning"]
        assert saved["revision"] > original["revision"]
        stale = client.post(endpoint(project) + "/actions", json={
            "action": "save", "expected_revision": original["revision"], "content": original["content"],
        })
        assert stale.status_code == 409
        approved = plan_action(client, project, "approve")["planning"]
        duplicate = plan_action(client, project, "approve")["planning"]
        assert duplicate["approval_id"] == approved["approval_id"]
        pid = project["project"]["id"]
        detail = client.get(f"/api/m2/projects/{pid}").json()
        assert detail["draft"]["planApproved"] and detail["draft"]["step"] == "production"
        with coordinator.db.transaction() as connection:
            roots = connection.execute("SELECT * FROM m3_production WHERE project_id=?", (pid,)).fetchall()
            assert len(roots) == 1 and roots[0]["sequential_publication"] == 1
            narrative_jobs = connection.execute("SELECT payload FROM job WHERE project_id=? AND kind='m3_narrative'", (pid,)).fetchall()
            assert len(narrative_jobs) == 1
            seed = json.loads(narrative_jobs[0]["payload"])["approved_plan"]
            assert seed["approval_id"] == approved["approval_id"] and seed["content"] == saved["content"]
            assert json.loads(narrative_jobs[0]["payload"])["approval_snapshot"] == job["payload"]["approval_snapshot"]
            artifact = connection.execute("SELECT artifact_id FROM planning_approval WHERE id=?", (seed["approval_id"],)).fetchone()[0]
        assert json.loads(client.get(f"/api/artifacts/{artifact}/content").content)["content"] == saved["content"]


def test_failed_revision_preserves_plan_and_retries_same_frozen_instruction(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, original_job = reviewed(client)
        original = planning(client, project)["content"]
        plan_action(client, project, "revise", target="plot", chapter_number=1, instruction="交流を増やしてください。")
        for _ in range(3):
            job = claim(client, worker)
            assert job["payload"]["plan_content"] == original
            assert job["payload"]["instruction"] == "交流を増やしてください。"
            response = client.post(f"/api/jobs/{job['id']}/fail", json={
                "worker_id": worker, "lease_id": job["lease_id"], "error": "fixture failure",
            })
            assert response.status_code == 200, response.text
        failed = planning(client, project)
        assert failed["status"] == "failed" and failed["content"] == original
        response = client.post(f"/api/jobs/{job['id']}/retry")
        assert response.status_code == 200, response.text
        retry = claim(client, worker)
        assert retry["payload"] == job["payload"]
        assert retry["payload"]["profile"] == original_job["payload"]["profile"]
        repaired = copy.deepcopy(original)
        repaired["plot"]["chapters"][0]["role"] = "会話と交流を描く。"
        complete_plan(client, retry, worker, content=repaired)
        assert planning(client, project)["content"]["plot"]["chapters"][0]["role"] == "会話と交流を描く。"


def test_main_edit_rejects_late_plan_result_and_deselects_pending_work(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        project = action(client, project, "approve")
        worker = plan_worker(client)
        job = claim(client, worker)
        action(client, project, "edit-character", character_id="character-1", patch={"settings": "新設定。"})
        complete_plan(client, job, worker, status=409)
        assert planning(client, project)["content"] is None
        response = client.post(f"/api/jobs/{job['id']}/fail", json={
            "worker_id": worker, "lease_id": job["lease_id"], "error": "stale job",
        })
        assert response.status_code == 200, response.text
        assert claim(client, worker) is None


def test_restore_plan_content_advances_revision_and_never_revives_old_jobs(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, _ = reviewed(client)
        initial = planning(client, project)
        pid = project["project"]["id"]
        history = client.get(f"/api/m2/projects/{pid}/history").json()
        saved_revision = history["current_revision_id"]
        changed = copy.deepcopy(initial["content"])
        changed["plot"]["chapters"][0]["role"] = "別の役割。"
        plan_action(client, project, "save", content=changed)
        current = planning(client, project)
        history = client.get(f"/api/m2/projects/{pid}/history").json()
        draft = client.get(f"/api/m2/projects/{pid}").json()["draft"]
        response = client.post(f"/api/m2/projects/{pid}/history/{saved_revision}/restore", json={
            "expected_version": history["version"], "expected_revision": draft["revision"],
        })
        assert response.status_code == 200, response.text
        restored = planning(client, project)
        assert restored["revision"] > current["revision"]
        assert restored["content"] == initial["content"] and restored["active_job_id"] is None
        assert claim(client, worker) is None


def test_invalid_save_or_wrong_protocol_preserves_valid_plan(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, _ = reviewed(client)
        original = planning(client, project)
        damaged = copy.deepcopy(original["content"])
        damaged["plot"]["core"]["characters"][0]["character_id"] = "missing-character"
        plan_action(client, project, "save", content=damaged, status=422)
        assert planning(client, project)["content"] == original["content"]
        plan_action(client, project, "revise", target="plot", instruction="結末を具体化。")
        job = claim(client, worker)
        complete_plan(client, job, worker, protocol={"invalid": True}, status=422)
        assert planning(client, project)["content"] == original["content"]


def test_plan_generation_freezes_profiles_before_step5(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path)) as client:
        project, _, job = reviewed(client)
        monkeypatch.setattr(M2Service, "_profile", lambda *_: {"provider": "local", "model_id": "changed"})
        plan_action(client, project, "approve")
        response = client.get(f"/api/m3/projects/{project['project']['id']}").json()
        narrative = next(row for row in response["production"]["jobs"] if row["kind"] == "m3_narrative")
        assert narrative["payload"]["profile"] == job["payload"]["profile"]
        assert narrative["payload"]["tts_profile"] == job["payload"]["tts_profile"]
        assert narrative["payload"]["seed"] == job["payload"]["seed"]


def test_second_plan_approval_creates_separate_series_from_same_main_snapshot(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(tmp_path, coordinator=coordinator)) as client:
        project, _, _ = reviewed(client)
        first = plan_action(client, project, "approve")["planning"]
        worker = client.post("/api/workers", json={
            "name": "failed writer", "capabilities": ["m3_narrative"],
        }).json()["id"]
        for _ in range(3):
            old_job = claim(client, worker)
            assert old_job["kind"] == "m3_narrative"
            response = client.post(f"/api/jobs/{old_job['id']}/fail", json={
                "worker_id": worker, "lease_id": old_job["lease_id"], "error": "failed old series",
            })
            assert response.status_code == 200, response.text
        edited = copy.deepcopy(first["content"])
        edited["plot"]["core"]["central_question"] = "別の構成の問い。"
        plan_action(client, project, "save", content=edited)
        second = plan_action(client, project, "approve")["planning"]
        assert first["approval_id"] != second["approval_id"]
        with coordinator.db.transaction() as connection:
            roots = connection.execute("SELECT * FROM m3_production WHERE project_id=?",
                                       (project["project"]["id"],)).fetchall()
            assert len(roots) == 2
            assert len({row["approval_id"] for row in roots}) == 1
            assert len({row["storyline_id"] for row in roots}) == 2
            assert {row["plan_approval_id"] for row in roots} == {first["approval_id"], second["approval_id"]}
        assert client.post(f"/api/jobs/{old_job['id']}/retry").status_code == 409
        new_job = claim(client, worker)
        assert new_job["payload"]["approved_plan"]["approval_id"] == second["approval_id"]


def test_step4_edit_of_paused_series_blocks_resume_and_old_worker_claim(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(tmp_path, coordinator=coordinator)) as client:
        project, _, _ = reviewed(client)
        first = plan_action(client, project, "approve")["planning"]
        pid = project["project"]["id"]
        response = client.post(f"/api/m3/projects/{pid}/stop", json={"mode": "graceful"})
        assert response.status_code == 200, response.text
        edited = copy.deepcopy(first["content"])
        edited["plot"]["chapters"][0]["role"] = "構成を修正してから執筆。"
        plan_action(client, project, "save", content=edited)
        response = client.post(f"/api/m3/projects/{pid}/resume")
        assert response.status_code == 409
        writer = client.post("/api/workers", json={
            "name": "isolated writer", "capabilities": ["m3_narrative"],
        }).json()["id"]
        # Even a stale control value cannot bypass the applicable-plan check.
        with coordinator.db.transaction() as connection:
            connection.execute("UPDATE m3_production SET control_state='running' WHERE project_id=?", (pid,))
        assert claim(client, writer) is None
        state = client.get(f"/api/m3/projects/{pid}").json()["production"]
        assert state and state["id"] == state["chapters"][0]["production_id"]
        second = plan_action(client, project, "approve")["planning"]
        assert first["approval_id"] != second["approval_id"]
        assert claim(client, writer)["payload"]["approved_plan"]["approval_id"] == second["approval_id"]


def test_unstarted_legacy_approval_can_explicitly_generate_step4_without_changing_main_snapshot(tmp_path, monkeypatch):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(tmp_path, coordinator=coordinator)) as client:
        project, _ = ready(client)
        # Model a persisted approval saved before either planning or production.
        original_start = PlanningService.start_approved
        monkeypatch.setattr(PlanningService, "start_approved", lambda *_: None)
        project = action(client, project, "approve")
        monkeypatch.setattr(PlanningService, "start_approved", original_start)
        pid = project["project"]["id"]
        with coordinator.db.transaction() as connection:
            state = M2Service._load(connection, pid)
            state["draft"].pop("planningRequired")
            state["draft"].pop("planApproved")
            state["draft"]["step"] = "character-review"
            state["wizardStepVersion"] = 2
            M2Service(coordinator)._save(connection, pid, state)
        before = client.get(f"/api/artifacts/{project['draft']['approval']['artifactId']}/content").content
        assert planning(client, project) is None
        response = client.post(endpoint(project) + "/actions", json={"action": "generate", "expected_revision": 0})
        assert response.status_code == 200, response.text
        assert response.json()["planning"]["status"] == "generating"
        assert client.get(f"/api/m3/projects/{pid}").json()["production"] is None
        assert client.get(f"/api/artifacts/{project['draft']['approval']['artifactId']}/content").content == before
        assert client.post(f"/api/m3/projects/{pid}/start").status_code == 409
        worker = plan_worker(client)
        job = claim(client, worker)
        assert job["kind"] == "m3_plan"
        complete_plan(client, job, worker)
        plan_action(client, project, "approve")
        assert client.get(f"/api/m3/projects/{pid}").json()["production"]["plan_approval_id"]


@pytest.mark.parametrize("action_name", ["save", "revise", "approve"])
def test_active_planning_job_blocks_conflicting_edits(tmp_path, action_name):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        project = action(client, project, "approve")
        before = planning(client, project)
        plan_action(client, project, action_name, status=409,
                    content={}, instruction="この応答は競合する。")
        assert planning(client, project)["revision"] == before["revision"]
