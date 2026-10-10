"""An approved alternate look is planned, published as its own portrait and shown on stage."""

import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from packages.contracts.m3 import M3_KINDS
from packages.narrative.validation import script_character_id
from services.coordinator.app import create_app
from tests.integration.planning_fixtures import complete_and_approve_plan
from tests.integration.test_m2 import action, claim, png, ready, report_voice_inventory
from tests.integration.test_m2 import complete as complete_m2
from tests.integration.test_m2_outfits import LOOK, outfit_bundle, outfit_worker
from tests.integration.test_m3 import complete, finish, narrative, output, production


def approved_with_look(client):
    project, _ = ready(client)
    qwen = outfit_worker(client)
    action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
    action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
    job = claim(client, qwen)
    assert complete_m2(client, job, qwen, outfit_bundle(job)).status_code == 200
    action(client, project, "approve")
    planned = complete_and_approve_plan(client, project)
    worker = client.post("/api/workers", json={
        "name": "M3 fixture", "capabilities": [*M3_KINDS, "tts_download"]}).json()["id"]
    report_voice_inventory(client, worker)
    return project["project"]["id"], worker, planned


def test_planned_look_is_published_and_changed_on_stage(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, planned = approved_with_look(client)
        chapters = planned["planning"]["content"]["plot"]["chapters"]
        assert chapters[0]["outfit_appearances"][0]["outfit_id"] == "look1"
        job = claim(client, worker)
        assert job["kind"] == "m3_narrative"
        assert job["payload"]["approval_snapshot"]["characters"][0]["outfits"][0]["id"] == "look1"
        result = narrative(job["payload"]["approval_snapshot"])
        first, second = result["scenes"]
        first["plan"]["outfits"] = [{"character_id": "character-1", "outfit_id": "default", "change_to": "look1"}]
        first["directions"].append({
            "id": "scene-1-o1", "utterance_id": first["utterances"][3]["id"], "kind": "outfit",
            "timing": "before", "character_id": "character-1", "position": None, "duration_ms": 400,
            "outfit_id": "look1"})
        second["plan"]["outfits"] = [{"character_id": "character-1", "outfit_id": "look1"}]
        assert complete(client, worker, job, output(job, result)).status_code == 200
        finish(client, worker)
        chapter = production(client, project)["chapters"][0]
        assert chapter["status"] == "published"
        with ZipFile(io.BytesIO(client.get(chapter["export_url"]).content)) as archive:
            script = json.loads(archive.read("script.json"))
            names = archive.namelist()
            scenario = archive.read("data/scenario/first.ks").decode("utf-8")
        main = next(row for row in script["characters"] if row["id"] == script_character_id("character-1"))
        (look,) = main["outfits"]
        files = {row["id"]: row["filename"] for row in script["assets"]}
        assert look["id"] == "look1" and f"data/fgimage/{files[look['image_asset_id']]}" in names
        entrances = [row.get("outfit_id") for row in script["directions"]
                     if row["kind"] == "enter" and row["character_id"] == main["id"]]
        assert entrances == [None, "look1"]
        assert [(row["utterance_id"], row["outfit_id"]) for row in script["directions"]
                if row["kind"] == "outfit"] == [(first["utterances"][3]["id"], "look1")]
        assert f'storage="{files[look["image_asset_id"]]}" time="400"]' in scenario
        asset = next(row for row in script["assets"] if row["id"] == look["image_asset_id"])
        assert client.get(f"/api/artifacts/{asset['artifact_id']}/content").content == png(200)


def test_chapter_that_does_not_show_the_look_publishes_without_it(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, _ = approved_with_look(client)
        finish(client, worker)
        chapter = production(client, project)["chapters"][0]
        assert chapter["status"] == "published"
        with ZipFile(io.BytesIO(client.get(chapter["export_url"]).content)) as archive:
            script = json.loads(archive.read("script.json"))
        assert all("outfits" not in row for row in script["characters"])
        assert all("outfit_id" not in row for row in script["directions"])
