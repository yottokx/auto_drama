"""Planned supporting characters are drawn and voiced in STEP4 and reused by the chapters."""
import copy

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from tests.integration.planning_fixtures import plan_content
from tests.integration.test_m2 import action, claim, ready, report_voice_inventory
from tests.integration.test_m3 import CHARACTER, M3_KINDS, complete, finish
from tests.integration.test_planning import (
    complete_plan,
    endpoint,
    plan_action,
    plan_worker,
    planning,
)

SUPPORT = {**CHARACTER, "id": "support-1", "name": "エマ"}


def with_support(snapshot):
    content = plan_content(snapshot)
    content["cast_plan"]["supporting_characters"] = [copy.deepcopy(SUPPORT)]
    content["cast_plan"]["everyday_context"] = [{
        "character_id": "support-1", "personal_concern": "記録室の鍵を預かっている。",
        "contact": "記録室で顔を合わせる。", "initial_knowledge": "記録の場所を知っている。"}]
    return content


def reviewed_with_support(client):
    project, _ = ready(client)
    project = action(client, project, "approve")
    worker = plan_worker(client)
    job = claim(client, worker)
    complete_plan(client, job, worker, content=with_support(job["payload"]["approval_snapshot"]))
    media = client.post("/api/workers", json={
        "name": "media", "capabilities": [*M3_KINDS, "tts_download"]}).json()["id"]
    report_voice_inventory(client, media)
    return project, media


def assets(client, project):
    return planning(client, project)["assets"]["support-1"]


def test_supporting_assets_are_made_in_review_frozen_at_approval_and_reused(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, media = reviewed_with_support(client)
        assert {kind: row["status"] for kind, row in assets(client, project).items()} == {
            "image": "pending", "voice": "pending"}
        plan_action(client, project, "approve", status=409)
        made = finish(client, media)
        assert sorted(job["kind"] for job in made) == ["m3_image", "m3_voice"]
        assert all(job["payload"]["character_id"] == "support-1" and "production_id" not in job["payload"]
                   for job in made)
        ready_assets = assets(client, project)
        assert all(row["status"] == "ready" for row in ready_assets.values())
        assert client.get(ready_assets["image"]["url"]).headers["content-type"] == "image/png"
        plan_action(client, project, "approve")
        with client.app.state.coordinator.db.transaction() as connection:
            connection.execute("UPDATE m3_production SET music_enabled=0")
        narrative = claim(client, media)
        assert narrative["kind"] == "m3_narrative"
        assert complete(client, media, narrative).status_code == 200
        # Only the lines and backgrounds remain; the supporting portrait and voice are already there.
        rest = finish(client, media)
        assert not [job for job in rest if job["kind"] in {"m3_image", "m3_voice"}]
        clone = next(job for job in rest if job["kind"] == "m3_voice_clone"
                     and job["payload"]["character_id"] == "support-1")
        with client.app.state.coordinator.db.transaction() as connection:
            voice = connection.execute(
                "SELECT artifact_id FROM planning_asset WHERE character_id='support-1' AND kind='voice'").fetchone()[0]
        assert clone["payload"]["reference_voice"]["artifact_id"] == voice


def test_retake_and_edit_redo_only_what_changed(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, media = reviewed_with_support(client)
        finish(client, media)
        before = assets(client, project)
        response = client.post(endpoint(project) + "/assets", json={
            "expected_revision": planning(client, project)["revision"], "character_id": "support-1",
            "kind": "image", "instruction": "髪を短く"})
        assert response.status_code == 200, response.text
        (job,) = finish(client, media)
        assert job["kind"] == "m3_image" and job["payload"]["instruction"] == "髪を短く"
        after = assets(client, project)
        assert after["image"]["url"] != before["image"]["url"] and after["voice"] == before["voice"]
        content = planning(client, project)["content"]
        content["cast_plan"]["supporting_characters"][0]["voice"] = "低く落ち着いた声。ゆっくり話す。"
        plan_action(client, project, "save", content=content)
        assert {kind: row["status"] for kind, row in assets(client, project).items()} == {
            "image": "ready", "voice": "pending"}
        (job,) = finish(client, media)
        assert job["kind"] == "m3_voice"
