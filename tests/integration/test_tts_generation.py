"""Installed, runtime-capable TTS choices and immutable voice job profiles."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m3 import M3_KINDS
from packages.contracts.tts_catalog import resolve_bundle
from services.coordinator.app import create_app
from tests.integration.test_m2 import (
    action,
    claim,
    complete,
    ready,
    report_voice_inventory,
    worker,
)

SMALL = "irodori-v4.1-small"
LARGE = "irodori-v4-large"


def choice(model, precision):
    return {"provider_id": "irodori", "model_id": model, "precision": precision}


def settings(design=(LARGE, "bf16"), clone=(SMALL, "fp32")):
    return {"schema_version": 1, "generation_active": True,
            "voice_design": choice(*design), "voice_clone": choice(*clone)}


def installed(model, precision, *, purpose, runtime=True):
    return {"model_id": model, "precision": precision,
            "manifest_id": resolve_bundle(model, precision)["manifest_id"],
            "file_download_ready": True, "generation_ready": runtime,
            "generation_purposes": [purpose] if runtime else []}


def register(client, purposes):
    kinds = {"voice_design": "m2_voice", "voice_clone": "m2_voice_clone"}
    return client.post("/api/workers", json={"name": "fixture", "capabilities": [
        "tts_download", *(kinds[purpose] for purpose in purposes),
    ]}).json()["id"]


@pytest.mark.parametrize("missing", ["files", "runtime", "purpose", "offline", "manifest"])
def test_unavailable_selections_cannot_be_saved(tmp_path, missing):
    with TestClient(create_app(tmp_path)) as client:
        before = client.get("/api/settings/tts").json()["settings"]
        target = register(client, ["voice_design", "voice_clone"])
        items = [installed(LARGE, "bf16", purpose="voice_design"),
                 installed(SMALL, "fp32", purpose="voice_clone")]
        if missing == "files":
            items = items[1:]
        elif missing == "runtime":
            items[0].update(generation_ready=False, generation_purposes=[])
        elif missing == "purpose":
            items[0]["generation_purposes"] = ["voice_clone"]
        assert client.post(f"/api/workers/{target}/tts-models", json={"models": items}).status_code == 200
        with client.app.state.coordinator.db.transaction() as connection:
            if missing == "offline":
                connection.execute("UPDATE worker SET last_seen_at=0 WHERE id=?", (target,))
            elif missing == "manifest":
                connection.execute("UPDATE tts_worker_inventory SET manifest_id='old' WHERE model_id=?", (LARGE,))
        assert client.post("/api/settings/tts", json=settings()).status_code == 422
        assert client.get("/api/settings/tts").json()["settings"] == before


def test_purposes_can_use_different_installed_workers_and_reject_forged_runtime_readiness(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        design = register(client, ["voice_design"])
        clone = register(client, ["voice_clone"])
        for target, item in ((design, installed(LARGE, "bf16", purpose="voice_design")),
                             (clone, installed(SMALL, "fp32", purpose="voice_clone"))):
            assert client.post(f"/api/workers/{target}/tts-models", json={"models": [item]}).status_code == 200
        bad = installed(LARGE, "bf16", purpose="voice_clone")
        assert client.post(f"/api/workers/{design}/tts-models", json={"models": [bad]}).status_code == 422
        response = client.post("/api/settings/tts", json=settings())
        assert response.status_code == 200, response.text
        assert response.json()["settings"] == settings()
        assert response.json()["stage"] == "active"


def test_previous_draft_selection_is_preserved_when_generation_becomes_active(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        defaults = client.get("/api/settings/tts").json()["settings"]
        assert defaults["voice_design"] == defaults["voice_clone"] == choice(SMALL, "fp32")
        old = {**settings(), "generation_active": False}
        with client.app.state.coordinator.db.transaction() as connection:
            connection.execute("INSERT INTO tts_settings(id,value) VALUES(1,?)", (json.dumps(old),))
        active = client.get("/api/settings/tts").json()
        assert active["generation_active"] is True
        assert active["settings"] == settings()


def test_revoked_generation_capabilities_immediately_hide_runtime_readiness(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        target = register(client, ["voice_design", "voice_clone"])
        item = installed(LARGE, "bf16", purpose="voice_design")
        assert client.post(f"/api/workers/{target}/tts-models", json={"models": [item]}).status_code == 200
        with client.app.state.coordinator.db.transaction() as connection:
            connection.execute("UPDATE worker SET capabilities=? WHERE id=?", (json.dumps(["tts_download"]), target))
        entry = client.get("/api/tts-downloads").json()["workers"][0]["inventory"][0]
        assert entry["file_download_ready"] is True
        assert entry["generation_ready"] is False
        assert entry["generation_purposes"] == []


def test_voice_jobs_freeze_independent_selections_and_only_matching_worker_can_claim(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, target = ready(client)
        report_voice_inventory(client, target, [(SMALL, "fp32"), (LARGE, "bf16")])
        assert client.post("/api/settings/tts", json=settings()).status_code == 200
        action(client, project, "retake", character_id="character-1", scope="voice-retake")
        jobs = client.get(f"/api/m2/projects/{project['project']['id']}").json()["jobs"]
        design = next(job for job in jobs if job["kind"] == "m2_voice" and job["status"] == "pending")
        frozen = client.get(f"/api/jobs/{design['id']}").json()["job"]
        profile = frozen["payload"]["tts_profile"]
        assert profile == frozen["settings_snapshot"]["tts_profile"]
        assert profile["voice_design"]["model_id"] == LARGE
        assert profile["voice_design"]["precision"] == "bf16"
        assert profile["voice_clone"]["model_id"] == SMALL
        assert profile["voice_design"]["bundle"] == resolve_bundle(LARGE, "bf16")
        changed = settings((SMALL, "fp32"), (LARGE, "bf16"))
        assert client.post("/api/settings/tts", json=changed).status_code == 200
        assert client.get(f"/api/jobs/{design['id']}").json()["job"] == frozen
        wrong = worker(client)
        assert claim(client, wrong) is None
        claimed = claim(client, target)
        assert claimed["payload"]["tts_profile"] == profile
        assert complete(client, claimed, target).status_code == 200
        action(client, project, "clone-voice", character_id="character-1", text="別モデルで試聴します。")
        cloned = claim(client, target)
        assert cloned["kind"] == "m2_voice_clone"
        assert cloned["payload"]["tts_profile"]["voice_clone"]["model_id"] == LARGE
        assert cloned["payload"]["tts_profile"]["voice_design"]["model_id"] == SMALL


def test_all_production_assets_and_continuation_retain_initial_tts_profile(tmp_path):
    from tests.integration.planning_fixtures import complete_and_approve_plan
    from tests.integration.test_m3 import pin_legacy_production, production
    from tests.integration.test_m4 import finish_job

    with TestClient(create_app(tmp_path)) as client:
        project, target = ready(client)
        report_voice_inventory(client, target, [(SMALL, "fp32"), (LARGE, "bf16")])
        assert client.post("/api/settings/tts", json=settings()).status_code == 200
        action(client, project, "approve")
        project_id = project["project"]["id"]
        complete_and_approve_plan(client, project_id)
        pin_legacy_production(client, project_id)
        runtime = client.post("/api/workers", json={"name": "production fixture",
            "capabilities": [*M3_KINDS, "tts_download"]}).json()["id"]
        report_voice_inventory(client, runtime, [(SMALL, "fp32"), (LARGE, "bf16")])
        initial = claim(client, runtime)
        profile = copy.deepcopy(initial["payload"]["tts_profile"])
        assert client.post("/api/settings/tts", json=settings((SMALL, "fp32"), (LARGE, "bf16"))).status_code == 200
        finish_job(client, runtime, initial)
        state = production(client, project_id)
        jobs = [job for chapter in state["chapters"] for job in chapter["jobs"]]
        assert any(job["kind"] == "m3_voice_clone" for job in jobs)
        assert any(job["payload"]["chapter_number"] == 2 for job in jobs)
        assert all(job["payload"]["tts_profile"] == profile for job in jobs)
        for job in jobs:
            full = client.get(f"/api/jobs/{job['id']}").json()["job"]
            assert full["settings_snapshot"]["tts_profile"] == profile


def test_legacy_voice_job_without_tts_profile_keeps_compatible_capability_dispatch(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        action(client, project, "retake", character_id="character-1", scope="voice-retake")
        with client.app.state.coordinator.db.transaction() as connection:
            row = connection.execute("SELECT id,payload FROM job WHERE status='pending' AND kind='m2_voice'").fetchone()
            payload = json.loads(row["payload"])
            payload.pop("tts_profile")
            connection.execute("UPDATE job SET payload=? WHERE id=?", (json.dumps(payload), row["id"]))
        legacy = client.post("/api/workers", json={"name": "legacy", "capabilities": ["m2_voice"]}).json()["id"]
        assert claim(client, legacy)["id"] == row["id"]
