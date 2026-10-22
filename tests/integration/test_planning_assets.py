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


def test_a_new_plan_asks_for_every_planned_character_and_an_earlier_one_keeps_its_wording(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        project = action(client, project, "approve")
        worker = plan_worker(client)
        job = claim(client, worker)
        # Without a chosen character count, the guideline for the current settings is used.
        assert job["payload"]["script_options"] == {
            "planned_cast_appears": True, "open_scene_cast": True, "cast_limit": project["castLimit"]["guide"],
            "long_form": True, "chapter_scope_guard": True}
        complete_plan(client, job, worker)
        # A plan begun before this option existed has none; revising it must not introduce it.
        with client.app.state.coordinator.db.transaction() as connection:
            import json

            row = connection.execute("SELECT payload,settings_snapshot FROM job WHERE id=?", (job["id"],)).fetchone()
            payload, settings = json.loads(row["payload"]), json.loads(row["settings_snapshot"])
            payload.pop("script_options")
            settings.pop("script_options", None)
            connection.execute("UPDATE job SET payload=?,settings_snapshot=? WHERE id=?",
                               (json.dumps(payload, ensure_ascii=False), json.dumps(settings), job["id"]))
        plan_action(client, project, "revise", target="plot", instruction="第1章をもう少し穏やかに。")
        revised = claim(client, worker)
        assert "script_options" not in revised["payload"]


def test_the_character_count_chosen_at_approval_reaches_the_plan(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        # The default settings run at 16k, whose tier holds four characters.
        assert project["castLimit"] == {"guide": 4, "max": 4, "looks": 3}
        mains = len(project["draft"]["characters"])
        # Fewer than the mains is refused: by the contract at zero, by the service otherwise.
        action(client, project, "approve", cast_limit=mains - 1, status=400 if mains > 1 else 422)
        # So is more than the tier holds.
        action(client, project, "approve", cast_limit=5, status=400)
        project = action(client, project, "approve", cast_limit=4)
        job = claim(client, plan_worker(client))
        assert job["payload"]["script_options"]["cast_limit"] == 4
        assert job["payload"]["script_options"]["open_scene_cast"] is True
