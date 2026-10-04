import json

import pytest
from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from services.coordinator.m2_service import M2Service
from tests.integration.test_m2 import action, create

SETTINGS = {"model": "gemma-test", "temperature": 0.4, "top_p": 0.8,
            "reasoning_effort": "none", "ctx_size": 32768}


def worker(client, model="gemma-test", maximum=32768):
    response = client.post("/api/workers", json={"name": model,
        "capabilities": ["m2_world"], "llm_models": [{"model": model,
        "max_context_size": maximum, "reasoning_efforts": ["none"]}]})
    assert response.status_code == 201, response.text
    return response.json()["id"]


def test_settings_persist_snapshot_and_filter_workers(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        right = worker(client)
        wrong = worker(client, "other")
        worker(client, maximum=16384)
        assert client.post("/api/settings/llm", json=SETTINGS).status_code == 200
        project = create(client)
        result = action(client, project, "generate-world")
        original = next(job for job in result["jobs"] if job["kind"] == "m2_world")
        original = client.get(f"/api/jobs/{original['id']}").json()["job"]
        assert original["payload"]["profile"]["top_p"] == 0.8
        changed = {**SETTINGS, "temperature": 1.0}
        assert client.post("/api/settings/llm", json=changed).status_code == 200
        for identifier in (wrong,):
            assert client.post(f"/api/workers/{identifier}/claim", json={}).json()["job"] is None
        claimed = client.post(f"/api/workers/{right}/claim", json={}).json()["job"]
        assert claimed["payload"]["profile"] == original["payload"]["profile"]
        assert claimed["payload"]["profile"]["temperature"] == 0.4
        assert claimed["payload"]["profile"]["context_size"] == 32768
        assert M2Service(client.app.state.coordinator)._profile()["temperature"] == 1.0
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/settings/llm").json()["settings"] == changed


@pytest.mark.parametrize("patch", [
    {"ctx_size": 8192}, {"ctx_size": 16384.5},
    {"temperature": -0.1}, {"temperature": 2.1}, {"top_p": 0}, {"top_p": 1.1},
    {"reasoning_effort": 123}, {"model": "missing"}, {"max_tokens": 100},
])
def test_invalid_or_unsupported_settings_do_not_replace_saved_value(tmp_path, patch):
    with TestClient(create_app(tmp_path)) as client:
        worker(client)
        assert client.post("/api/settings/llm", json=SETTINGS).status_code == 200
        assert client.post("/api/settings/llm", json={**SETTINGS, **patch}).status_code == 422
        assert client.get("/api/settings/llm").json()["settings"] == SETTINGS


def test_script_production_preserves_common_settings_and_owns_output_budget(tmp_path):
    from tests.integration.planning_fixtures import complete_and_approve_plan
    from tests.integration.test_m2 import ready
    with TestClient(create_app(tmp_path)) as client:
        # Finish setup using the legacy fixture worker, then select new common settings.
        project, _ = ready(client)
        worker(client)
        assert client.post("/api/settings/llm", json=SETTINGS).status_code == 200
        action(client, project, "approve")
        complete_and_approve_plan(client, project, llm_models=[{
            "model": SETTINGS["model"], "max_context_size": SETTINGS["ctx_size"],
            "reasoning_efforts": [SETTINGS["reasoning_effort"]],
        }])
        connection = client.app.state.coordinator.db.connect()
        try:
            row = connection.execute("SELECT payload FROM job WHERE kind='m3_narrative'").fetchone()
        finally:
            connection.close()
        profile = json.loads(row["payload"])["profile"]
        assert profile["model_id"] == SETTINGS["model"]
        assert profile["temperature"] == SETTINGS["temperature"]
        assert profile["top_p"] == SETTINGS["top_p"]
        assert profile["context_size"] == SETTINGS["ctx_size"]
        assert "max_tokens" not in profile


def test_worker_rescans_models_without_reregistering(tmp_path):
    from services.worker.client import WorkerClient
    inventory = []
    with TestClient(create_app(tmp_path)) as client:
        worker_client = WorkerClient(client, generation_runner=lambda *_: b"",
            refresh_capabilities=lambda: (["m2_world"], list(inventory)))
        assert worker_client.run_once() == "idle"
        worker_id = worker_client.worker_id
        assert client.get("/api/settings/llm").json()["models"] == []
        inventory.append({"model": "new-file", "max_context_size": 16384,
                          "reasoning_efforts": ["none"]})
        worker_client._next_catalog_refresh = 0
        assert worker_client.run_once() == "idle"
        assert worker_client.worker_id == worker_id
        assert client.get("/api/settings/llm").json()["models"][0]["model"] == "new-file"
        inventory.clear()
        worker_client._next_catalog_refresh = 0
        assert worker_client.run_once() == "idle"
        assert client.get("/api/settings/llm").json()["models"] == []


@pytest.mark.parametrize("context", [32768, 65536, 131072])
def test_expanded_context_saves_and_dispatches_with_legacy_worker_advertisement(tmp_path, context):
    with TestClient(create_app(tmp_path)) as client:
        identifier = worker(client, maximum=16384)
        settings = {**SETTINGS, "ctx_size": context}
        response = client.post("/api/settings/llm", json=settings)
        assert response.status_code == 200, response.text
        project = create(client)
        action(client, project, "generate-world")
        job = client.post(f"/api/workers/{identifier}/claim", json={}).json()["job"]
        assert job["payload"]["profile"]["context_size"] == context


@pytest.mark.parametrize("effort", ["low", "high", "xhigh", "custom-value", " 任意の値 ", ""])
def test_free_text_effort_persists_and_dispatches(tmp_path, effort):
    settings = {**SETTINGS, "reasoning_effort": effort}
    with TestClient(create_app(tmp_path)) as client:
        identifier = worker(client)
        response = client.post("/api/settings/llm", json=settings)
        assert response.status_code == 200, response.text
        assert response.json()["settings"] == settings
        project = create(client)
        action(client, project, "generate-world")
        job = client.post(f"/api/workers/{identifier}/claim", json={}).json()["job"]
        assert job["payload"]["profile"]["reasoning_level"] == effort
    with TestClient(create_app(tmp_path)) as client:
        assert client.get("/api/settings/llm").json()["settings"] == settings
