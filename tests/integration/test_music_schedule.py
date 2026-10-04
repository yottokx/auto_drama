"""Music direction follows adopted text before any chapter media load."""

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m3 import M3_KINDS
from services.coordinator.app import create_app
from tests.integration.planning_fixtures import complete_and_approve_plan
from tests.integration.test_m2 import action, claim, ready, report_voice_inventory
from tests.integration.test_m3 import complete, production
from tests.integration.test_scene_music import grouped_output


def start_story(client, *, music_enabled=True):
    project, _ = ready(client)
    action(client, project, "approve")
    complete_and_approve_plan(client, project, music_enabled=music_enabled)
    worker = client.post("/api/workers", json={
        "name": "Music scheduling fixture", "capabilities": [*M3_KINDS, "tts_download"],
    }).json()["id"]
    report_voice_inventory(client, worker)
    return project["project"]["id"], worker


@pytest.mark.parametrize("music_enabled", [True, False])
def test_music_visibility_is_known_before_narrative_or_media_jobs(tmp_path, music_enabled):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = start_story(client, music_enabled=music_enabled)
        state = production(client, project)
        assert state["music_enabled"] is music_enabled
        assert all(chapter["music_enabled"] is music_enabled for chapter in state["chapters"])
        assert all(chapter["narrative_artifact_id"] is None for chapter in state["chapters"])
        assert not any(job["kind"].startswith("m3_music") for job in state["jobs"])


def test_music_design_is_next_after_narrative_and_blocks_queued_media_until_adoption(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = start_story(client)
        narrative = claim(client, worker)
        assert narrative["kind"] == "m3_narrative"
        assert complete(client, worker, narrative).status_code == 200
        state = production(client, project)
        chapter = state["chapters"][0]
        text_id = chapter["narrative_artifact_id"]
        assert text_id
        assert not any(row["kind"] == "m3_music" for row in chapter["requirements"])
        # Media queued by older versions can be older/higher priority than the
        # plan. The claim gate also protects them, without rewriting their input.
        with client.app.state.coordinator.db.transaction() as connection:
            connection.execute(
                "UPDATE job SET priority=1000 WHERE project_id=? "
                "AND kind IN ('m3_image','m3_background','m3_voice','m3_voice_clone')",
                (project,),
            )
        other_worker = client.post("/api/workers", json={
            "name": "Media-only worker", "capabilities": [
                "m3_image", "m3_background", "m3_voice", "m3_voice_clone", "m3_music", "tts_download",
            ],
        }).json()["id"]
        report_voice_inventory(client, other_worker)
        assert claim(client, other_worker) is None
        plan = claim(client, worker)
        assert plan["kind"] == "m3_music_plan"
        assert plan["payload"]["profile"] == narrative["payload"]["profile"]
        assert claim(client, other_worker) is None
        result = grouped_output(plan, {}, ("stop", "stop"))
        assert complete(client, worker, plan, result).status_code == 200
        after = production(client, project)["chapters"][0]
        assert after["narrative_artifact_id"] == text_id
        assert next(row for row in after["requirements"] if row["kind"] == "m3_music_plan")["artifact_id"]
        assert not any(row["kind"] == "m3_music" for row in after["requirements"])
        assert claim(client, worker)["kind"] in {"m3_image", "m3_background"}


def test_failed_music_design_retries_before_media_without_repeating_narrative(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = start_story(client)
        narrative = claim(client, worker)
        assert complete(client, worker, narrative).status_code == 200
        plan = claim(client, worker)
        assert plan["kind"] == "m3_music_plan"
        for attempt in range(3):
            assert client.post(f"/api/jobs/{plan['id']}/fail", json={
                "worker_id": worker, "lease_id": plan["lease_id"], "error": "Planning fixture failure",
            }).status_code == 200
            if attempt < 2:
                retried = claim(client, worker)
                assert retried["id"] == plan["id"]
                plan = retried
        assert claim(client, worker) is None
        assert client.post(f"/api/jobs/{plan['id']}/retry").status_code == 200
        retried = claim(client, worker)
        assert retried["id"] == plan["id"]
        assert complete(client, worker, retried, grouped_output(retried, {}, ("stop", "stop"))).status_code == 200
        state = production(client, project)
        assert next(job for job in state["jobs"] if job["id"] == narrative["id"])["attempt_count"] == 1
        assert claim(client, worker)["kind"] in {"m3_image", "m3_background"}
