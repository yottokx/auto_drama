"""A published alternate look is placed, replaced and regenerated like any other portrait."""

import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from packages.narrative.validation import script_character_id
from services.coordinator.app import create_app
from tests.integration.test_adjustments import apply, save, start
from tests.integration.test_m2 import claim, png
from tests.integration.test_m2_outfits import outfit_bundle, outfit_worker
from tests.integration.test_m3 import complete, finish, narrative, output, production
from tests.integration.test_m3_outfits import approved_with_look

LOOK_ROW = "character-1--look1"


def published_with_look(client):
    project, worker, _ = approved_with_look(client)
    job = claim(client, worker)
    result = narrative(job["payload"]["approval_snapshot"])
    result["scenes"][1]["plan"]["outfits"] = [{"character_id": "character-1", "outfit_id": "look1"}]
    assert complete(client, worker, job, output(job, result)).status_code == 200
    finish(client, worker)
    return project, f"/api/m3/projects/{project}/adjustments"


def first_script(client, project):
    chapter = production(client, project)["chapters"][0]
    with ZipFile(io.BytesIO(client.get(chapter["export_url"]).content)) as archive:
        return json.loads(archive.read("script.json"))


def look_of(script):
    main = next(row for row in script["characters"] if row["id"] == script_character_id("character-1"))
    return main, main["outfits"][0]


def test_look_has_its_own_row_and_placement_reaches_every_chapter(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, endpoint = published_with_look(client)
        value = start(client, endpoint)
        row = next(item for item in value["cast"] if item["character_id"] == LOOK_ROW)
        assert row["role"] == "look" and row["name"].endswith("（浴衣）") and row["chapter_numbers"] == [1]
        assert row["voice_candidate_id"] is None and row["image_url"]
        assert row["source_prompts"]["image"] == "藍色の浴衣に白い帯を締めた姿。"
        assert LOOK_ROW in value["draft"]["baseline"]["layouts"]
        characters = [{key: item[key] for key in ("character_id", "image_candidate_id", "voice_candidate_id",
                                                  "framing", "height_cm", "body_bounds", "offset_y", "scale")}
                      for item in value["draft"]["characters"]]
        edited = next(item for item in characters if item["character_id"] == LOOK_ROW)
        edited.update(offset_y=12, scale=1.1)
        value = save(client, endpoint, value, characters)
        applied = apply(client, endpoint, value)
        assert applied["draft"]["status"] == "applied", applied["draft"]
        script = first_script(client, project)
        main, look = look_of(script)
        assert (look["offset_y"], look["scale"]) == (12, 1.1) and "scale" not in main
        key = f"{main['id']}--look1"
        assert key in script["portrait_baseline"]["layouts"] and main["id"] in script["portrait_baseline"]["layouts"]


def test_look_is_regenerated_from_the_chosen_base_portrait_and_replaced(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, endpoint = published_with_look(client)
        value = start(client, endpoint)
        qwen = outfit_worker(client)
        response = client.post(endpoint + "/generate", json={
            "expected_revision": value["draft"]["revision"], "character_id": LOOK_ROW, "kind": "image",
            "instruction": "帯を赤にする。"})
        assert response.status_code == 200, response.text
        job = claim(client, qwen)
        base = next(item for item in value["draft"]["characters"] if item["character_id"] == "character-1")
        candidates = {item["id"]: item for item in value["candidates"]}
        assert job["kind"] == "m2_outfit_image"
        assert job["payload"]["outfit"]["description"] == "藍色の浴衣に白い帯を締めた姿。\n帯を赤にする。"
        assert job["payload"]["reference_image"]["artifact_id"] == candidates[base["image_candidate_id"]]["artifact_id"]
        done = client.post(f"/api/m2/jobs/{job['id']}/complete", params={"worker_id": qwen, "lease_id": job["lease_id"]},
                           content=outfit_bundle(job, media=png(77)), headers={"Content-Type": "application/zip"})
        assert done.status_code == 200, done.text
        value = client.get(endpoint).json()
        candidate = next(item for item in value["candidates"]
                         if item["character_id"] == LOOK_ROW and item["source"] == "generated")
        assert client.get(candidate["url"]).content == png(77)
        characters = [{key: item[key] for key in ("character_id", "image_candidate_id", "voice_candidate_id",
                                                  "framing", "height_cm", "body_bounds", "offset_y", "scale")}
                      for item in value["draft"]["characters"]]
        next(item for item in characters if item["character_id"] == LOOK_ROW)["image_candidate_id"] = candidate["id"]
        value = save(client, endpoint, value, characters)
        applied = apply(client, endpoint, value)
        assert applied["draft"]["status"] == "applied", applied["draft"]
        script = first_script(client, project)
        _, look = look_of(script)
        asset = next(item for item in script["assets"] if item["id"] == look["image_asset_id"])
        assert client.get(f"/api/artifacts/{asset['artifact_id']}/content").content == png(77)
