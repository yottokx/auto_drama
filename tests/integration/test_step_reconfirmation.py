"""Reconfirming an earlier step forks subsequent work without losing history."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from tests.integration.test_event_cg import cg_output, cg_result, setup_story
from tests.integration.test_m2 import action, claim, complete, detail
from tests.integration.test_m2 import worker as cast_worker
from tests.integration.test_m3 import approved, finish, production
from tests.integration.test_m3 import complete as complete_production
from tests.integration.test_planning import complete_plan, plan_worker
from tests.integration.test_project_history import history, restore


def selected_plan(client, project):
    return client.get(f"/api/planning/projects/{project}").json()["planning"]


def settle_reconfirmation(client, project, step):
    if step == "world-review":
        worker = cast_worker(client)
        for _ in range(15):
            job = claim(client, worker)
            if job is None:
                return
            result = complete(client, job, worker)
            assert result.status_code == 200, result.text
        raise AssertionError("Character regeneration did not settle")
    if step == "character-review":
        worker = plan_worker(client)
        job = claim(client, worker)
        assert job and job["kind"] == "m3_plan"
        complete_plan(client, job, worker)
    else:
        result = client.post(f"/api/m3/projects/{project}/stop", json={"mode": "immediate"})
        assert result.status_code == 200, result.text


@pytest.mark.parametrize("step", ["world-review", "character-review", "planning-review"])
def test_reconfirm_replaces_downstream_selection_and_history_restores_completed_story(tmp_path, step):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        finish(client, worker)
        original = production(client, project)
        assert original["status"] == "published"
        original_draft = copy.deepcopy(detail(client, project)["draft"])
        original_plan = copy.deepcopy(selected_plan(client, project))
        old_history = history(client, project)
        old_id = old_history["current_revision_id"]
        export_url = original["chapters"][0]["export_url"]
        export_bytes = client.get(export_url).content

        action(client, project, "go-to", step=step)
        if step == "planning-review":
            endpoint = f"/api/planning/projects/{project}/actions"
            body = {"action": "approve", "reconfirm": True,
                    "expected_revision": original_plan["revision"]}
        else:
            endpoint = f"/api/m2/projects/{project}/actions"
            body = {"action": "confirm-world" if step == "world-review" else "approve",
                    "reconfirm": True, "expected_revision": detail(client, project)["draft"]["revision"]}
        result = client.post(endpoint, json=body)
        assert result.status_code == 200, result.text
        assert client.post(endpoint, json=body).status_code == 409
        changed = detail(client, project)["draft"]
        if step == "world-review":
            assert changed["worldConfirmed"] and not changed["approved"]
            assert changed["worldResult"] == original_draft["worldResult"]
            assert changed["worldConfirmationArtifactId"] != original_draft["worldConfirmationArtifactId"]
            assert changed["activeJobId"]
            assert selected_plan(client, project) is None
            assert production(client, project) is None
        elif step == "character-review":
            assert changed["approved"] and not changed["planApproved"]
            assert changed["approval"]["id"] != original_draft["approval"]["id"]
            assert changed["characters"] == original_draft["characters"]
            assert selected_plan(client, project)["id"] != original_plan["id"]
            assert production(client, project) is None
        else:
            assert changed["planApproved"] and changed["step"] == "production"
            assert selected_plan(client, project)["approval_id"] != original_plan["approval_id"]
            assert production(client, project)["id"] != original["id"]
        assert client.get(export_url).content == export_bytes

        settle_reconfirmation(client, project, step)
        next_history = history(client, project)
        next_id = next_history["current_revision_id"]
        assert next_id != old_id and not next_history["busy"]
        restored = restore(client, project, old_id)
        assert restored["draft"]["approval"]["id"] == original_draft["approval"]["id"]
        assert selected_plan(client, project)["approval_id"] == original_plan["approval_id"]
        assert production(client, project)["id"] == original["id"]
        assert production(client, project)["chapters"][0]["export_url"] == export_url
        assert client.get(export_url).content == export_bytes
        assert claim(client, worker) is None
        # Undo keeps the newer branch available, too.
        restore(client, project, next_id)
        assert history(client, project)["entries"]
        if step != "planning-review":
            assert production(client, project) is None
        else:
            assert production(client, project)["id"] != original["id"]
        assert claim(client, worker) is None


@pytest.mark.parametrize("step", ["world-review", "character-review", "planning-review"])
def test_reconfirmation_refuses_active_downstream_generation_without_losing_current_state(tmp_path, step):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        running = claim(client, worker)
        assert running and running["kind"] == "m3_narrative"
        before = production(client, project)
        old_plan = selected_plan(client, project)
        before_revision = detail(client, project)["draft"]["revision"]
        if step == "planning-review":
            result = client.post(f"/api/planning/projects/{project}/actions", json={
                "action": "approve", "reconfirm": True, "expected_revision": old_plan["revision"]})
        else:
            result = client.post(f"/api/m2/projects/{project}/actions", json={
                "action": "confirm-world" if step == "world-review" else "approve",
                "reconfirm": True, "expected_revision": before_revision})
        assert result.status_code == 409, result.text
        assert detail(client, project)["draft"]["revision"] == before_revision
        assert selected_plan(client, project)["approval_id"] == old_plan["approval_id"]
        assert production(client, project)["id"] == before["id"]


def test_history_restores_project_cg_limits_after_reconfirming_a_step(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        finish(client, worker)
        original = history(client, project)["current_revision_id"]
        endpoint = f"/api/event-cg/projects/{project}/policy"
        changed = client.put(endpoint, json={"max_cgs": 3, "max_variants_per_cg": 2,
                                            "expected_revision": 0})
        assert changed.status_code == 200, changed.text
        action(client, project, "approve", reconfirm=True)
        settle_reconfirmation(client, project, "character-review")
        restore(client, project, original)
        restored = client.get(endpoint).json()
        assert restored["max_cgs"] == 0 and restored["max_variants_per_cg"] == 0
        assert restored["revision"] > changed.json()["revision"]
        assert client.put(endpoint, json={"max_cgs": 3, "max_variants_per_cg": 2,
                                         "expected_revision": changed.json()["revision"]}).status_code == 409


def test_reconfirm_replans_legacy_zero_cg_budget_and_keeps_original_history(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client, max_cgs=3)
        budget = claim(client, worker)
        assert budget and budget["kind"] == "m3_event_cg_budget"
        # Model the already adopted all-zero result from the previous protocol.
        budget["payload"].pop("budget_allocation_version")
        with client.app.state.coordinator.db.transaction() as connection:
            connection.execute("UPDATE job SET payload=? WHERE id=?",
                               (json.dumps(budget["payload"]), budget["id"]))
        zero = cg_result(budget)
        for chapter in zero["chapters"]:
            chapter.update(limit=0, reason="後半のために温存")
        result = complete_production(client, worker, budget, cg_output(budget, result=zero))
        assert result.status_code == 200, result.text
        assert client.post(f"/api/m3/projects/{project}/stop", json={"mode": "immediate"}).status_code == 200
        old = production(client, project)
        old_history = history(client, project)["current_revision_id"]
        old_plan = selected_plan(client, project)
        assert [row["limit"] for row in old["event_cg"]["chapter_budgets"]] == [0, 0, 0]

        result = client.post(f"/api/planning/projects/{project}/actions", json={
            "action": "approve", "reconfirm": True, "expected_revision": old_plan["revision"],
            "expected_event_cg_policy_revision": 1})
        assert result.status_code == 200, result.text
        assert production(client, project)["id"] != old["id"]
        new_budget = claim(client, worker)
        assert new_budget and new_budget["kind"] == "m3_event_cg_budget"
        assert new_budget["payload"]["budget_allocation_version"] == 2
        assert new_budget["payload"]["context"]["overall_plot"] == budget["payload"]["context"]["overall_plot"]
        # A new worker cannot silently adopt the old all-zero normal result.
        result = complete_production(client, worker, new_budget, cg_output(new_budget, result=zero))
        assert result.status_code == 422, result.text
        valid = cg_result(new_budget)
        result = complete_production(client, worker, new_budget, cg_output(new_budget, result=valid))
        assert result.status_code == 200, result.text
        assert sum(row["limit"] for row in production(client, project)["event_cg"]["chapter_budgets"]) == 3
        assert client.post(f"/api/m3/projects/{project}/stop", json={"mode": "immediate"}).status_code == 200
        restore(client, project, old_history)
        assert production(client, project)["id"] == old["id"]
        assert [row["limit"] for row in production(client, project)["event_cg"]["chapter_budgets"]] == [0, 0, 0]
        assert claim(client, worker) is None
