"""Current LLM choices apply at execution boundaries without changing story inputs."""
import copy
import json

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m3 import M3_KINDS
from services.coordinator.app import create_app
from tests.integration.test_event_cg import cg_output, setup_story
from tests.integration.test_job_retry_generation import fail
from tests.integration.test_llm_settings import SETTINGS
from tests.integration.test_m2 import action, claim, create, report_voice_inventory
from tests.integration.test_m3 import approved, complete, production
from tests.integration.test_planning import plan_action, planning, reviewed
from tests.integration.test_scene_music import music_output
from tests.integration.test_script_production import publish_first_chapter


def current_worker(client, model="gemma-test", capabilities=None):
    result = client.post("/api/workers", json={
        "name": "live settings fixture", "capabilities": capabilities or [*M3_KINDS, "tts_download"],
        "llm_models": [{"model": model, "max_context_size": 16384, "reasoning_efforts": ["none"]}],
    })
    assert result.status_code == 201, result.text
    identifier = result.json()["id"]
    if capabilities is None:
        report_voice_inventory(client, identifier)
    return identifier


def save_current(client, **changes):
    response = client.post("/api/settings/llm", json={**SETTINGS, **changes})
    assert response.status_code == 200, response.text


def test_m2_explicit_retry_refreshes_context_but_automatic_retry_keeps_input(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        worker = current_worker(client, capabilities=["m2_world"])
        save_current(client, ctx_size=16384)
        project = create(client)
        original = next(job for job in action(client, project, "generate-world")["jobs"]
                        if job["kind"] == "m2_world")
        initial = claim(client, worker)
        save_current(client, ctx_size=32768, temperature=0.75)
        assert client.get(f"/api/jobs/{initial['id']}").json()["job"]["payload"] == initial["payload"]
        fail(client, initial, worker)
        for _ in range(initial["max_attempts"] - 1):
            automatic = claim(client, worker)
            assert automatic["payload"] == initial["payload"]
            fail(client, automatic, worker)
        retried = client.post(f"/api/jobs/{original['id']}/retry").json()
        assert retried["payload"]["profile"]["context_size"] == 32768
        assert retried["payload"]["profile"]["temperature"] == 0.75
        assert retried["payload"]["execution_settings_revision"] == retried["retry_generation"] == 1
        assert retried["settings_snapshot"]["profile"] == retried["payload"]["profile"]
        for key in ("seed", "world_input", "world_result", "instruction", "base_revision"):
            assert retried["payload"][key] == initial["payload"][key]
        assert claim(client, worker)["payload"] == retried["payload"]


def test_planning_revision_and_retry_take_current_model_without_losing_plan(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _old_worker, first = reviewed(client)
        saved_plan = copy.deepcopy(planning(client, project)["content"])
        worker = current_worker(client)
        save_current(client, ctx_size=16384)
        plan_action(client, project, "revise", target="plot", instruction="交流を増やす。")
        original = claim(client, worker)
        assert original["payload"]["profile"]["model_id"] == SETTINGS["model"]
        assert original["payload"]["seed"] == first["payload"]["seed"]
        save_current(client, ctx_size=32768)
        for index in range(original["max_attempts"]):
            job = original if index == 0 else claim(client, worker)
            assert job["payload"] == original["payload"]
            fail(client, job, worker)
        response = client.post(f"/api/jobs/{original['id']}/retry")
        assert response.status_code == 200, response.text
        retried = claim(client, worker)
        assert retried["payload"]["profile"]["context_size"] == 32768
        assert "max_tokens" not in retried["payload"]["profile"]
        for key in ("plan_content", "instruction", "approval_snapshot", "seed", "tts_profile", "planning_revision"):
            assert retried["payload"][key] == original["payload"][key]
        assert planning(client, project)["content"] == saved_plan


def test_next_chapter_uses_current_model_and_context_preserving_checkpoint(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, original_worker = approved(client)
        first = claim(client, original_worker)
        with client.app.state.coordinator.db.transaction() as connection:
            payload = copy.deepcopy(first["payload"])
            payload["profiles"] = {"script_writer": {
                "model_id": payload["profile"]["model_id"], "context_size": 16384,
                "temperature": 0.1, "max_tokens": 4096,
            }}
            connection.execute("UPDATE job SET payload=? WHERE id=?", (json.dumps(payload), first["id"]))
            first["payload"] = payload
        current = current_worker(client)
        save_current(client, ctx_size=32768)
        assert client.get(f"/api/jobs/{first['id']}").json()["job"]["payload"] == first["payload"]
        assert complete(client, original_worker, first).status_code == 200
        state = publish_first_chapter(client, project, current)
        second = state["chapters"][1]["jobs"][0]
        assert second["payload"]["profile"]["model_id"] == SETTINGS["model"]
        assert second["payload"]["profile"]["context_size"] == 32768
        assert second["payload"]["profiles"] == {"script_writer": {"max_tokens": 4096}}
        for key in ("seed", "approval_snapshot", "tts_profile", "generator_protocol", "approved_plan"):
            assert second["payload"][key] == first["payload"][key]
        assert second["payload"]["execution_settings_version"] == 1
        assert second["payload"]["execution_settings_revision"] == 0
        source = second["payload"]["script_checkpoint_source"]
        assert source["seed"] == first["payload"]["seed"]
        assert source["approval_snapshot"] == first["payload"]["approval_snapshot"]
        assert "profile" not in source


def test_retry_existing_third_chapter_enriches_legacy_source_and_preserves_finished_chapters(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        while True:
            job = claim(client, worker)
            assert job is not None
            if job["kind"] == "m3_narrative" and job["payload"]["chapter_number"] == 3:
                break
            assert complete(client, worker, job).status_code == 200
        before = production(client, project)
        with client.app.state.coordinator.db.transaction() as connection:
            payload = copy.deepcopy(job["payload"])
            for key in ("execution_settings_version", "execution_settings_revision", "script_checkpoint_source"):
                payload.pop(key, None)
            snapshot = copy.deepcopy(job["settings_snapshot"])
            for key in ("execution_settings_version", "execution_settings_revision"):
                snapshot.pop(key, None)
            connection.execute("UPDATE job SET payload=?,settings_snapshot=? WHERE id=?",
                               (json.dumps(payload), json.dumps(snapshot), job["id"]))
            job["payload"] = payload
        for index in range(job["max_attempts"]):
            attempt = job if index == 0 else claim(client, worker)
            fail(client, attempt, worker)
        current_worker(client)
        save_current(client, ctx_size=32768)
        response = client.post(f"/api/jobs/{job['id']}/retry")
        assert response.status_code == 200, response.text
        retried = response.json()
        assert retried["payload"]["profile"]["context_size"] == 32768
        assert retried["payload"]["execution_settings_version"] == 1
        assert retried["payload"]["execution_settings_revision"] == 1
        assert retried["payload"]["script_checkpoint"] == payload["script_checkpoint"]
        assert retried["payload"]["script_checkpoint_source"]["seed"] == payload["seed"]
        after = production(client, project)
        for original, continued in zip(before["chapters"][:2], after["chapters"][:2], strict=True):
            assert continued["narrative_artifact_id"] == original["narrative_artifact_id"]
            assert continued["status"] == original["status"] == "published"


@pytest.mark.parametrize("kind", ["m3_music_plan", "m3_event_cg_budget", "m3_event_cg_plan"])
def test_planner_retry_refreshes_model_and_context_without_changing_semantic_inputs(tmp_path, kind):
    with TestClient(create_app(tmp_path)) as client:
        if kind == "m3_music_plan":
            project, worker = approved(client)
            with client.app.state.coordinator.db.transaction() as connection:
                connection.execute("UPDATE m3_production SET music_enabled=1 WHERE project_id=?", (project,))
        else:
            project, worker = setup_story(client)
        for _ in range(30):
            original = claim(client, worker)
            assert original is not None
            if original["kind"] == kind:
                break
            data = cg_output(original) if original["kind"].startswith("m3_event_cg") else None
            response = complete(client, worker, original, data)
            assert response.status_code == 200, response.text
        else:
            pytest.fail("The planner was not queued.")
        assert original["payload"]["execution_settings_version"] == 1
        source = copy.deepcopy(original["payload"]["context"])
        before = production(client, project)
        current = current_worker(client)
        save_current(client, ctx_size=32768)
        for index in range(original["max_attempts"]):
            attempt = original if index == 0 else claim(client, worker)
            assert attempt["payload"] == original["payload"]
            fail(client, attempt, worker)
        response = client.post(f"/api/jobs/{original['id']}/retry")
        assert response.status_code == 200, response.text
        retried = claim(client, current)
        assert retried["id"] == original["id"]
        assert retried["payload"]["profile"]["model_id"] == SETTINGS["model"]
        assert retried["payload"]["profile"]["context_size"] == 32768
        assert retried["payload"]["execution_settings_revision"] == retried["retry_generation"] == 1
        assert retried["settings_snapshot"]["profile"] == retried["payload"]["profile"]
        mutable = {"profile", "profiles", "execution_settings_revision"}
        assert {key: value for key, value in retried["payload"].items() if key not in mutable} == {
            key: value for key, value in original["payload"].items() if key not in mutable}
        assert retried["payload"]["context"] == source
        after = production(client, project)
        for initial, continued in zip(before["chapters"], after["chapters"], strict=True):
            assert continued["narrative_artifact_id"] == initial["narrative_artifact_id"]
        data = music_output(retried, {}) if kind == "m3_music_plan" else cg_output(retried)
        response = complete(client, current, retried, data)
        assert response.status_code == 200, response.text
