"""Alternate looks of a main character: defined, generated from the base portrait, approved."""

import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from tests.integration.test_m2 import action, claim, complete, detail, png, ready, run_one

LOOK = {"name": "浴衣", "description": "藍色の浴衣に白い帯。", "occasion": "夏祭りの夜。"}


def outfit_worker(client):
    return client.post("/api/workers", json={"name": "qwen", "capabilities": ["m2_outfit_image", "m2_image"]}).json()["id"]


def outfit_bundle(job, *, sha256=None, media=None):
    payload = job["payload"]
    envelope = {"schema_version": 1, "kind": "m2_outfit_image", "result": {}, "trace": [],
                "provenance": {"seed": payload["seed"], "outfit": {
                    "id": payload["outfit"]["id"], "summary": "藍色の浴衣に白い帯を締めた姿。",
                    "reference_sha256": sha256 or payload["reference_image"]["sha256"]}}}
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr("result.json", json.dumps(envelope, ensure_ascii=False))
        archive.writestr("image.png", media if media is not None else png(200))
    return data.getvalue()


def test_look_is_generated_from_the_base_portrait_and_enters_the_approval(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        qwen = outfit_worker(client)
        saved = action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
        (look,) = saved["draft"]["characters"][0]["outfits"]
        assert look["id"] == "look1" and look["imagePendingChanges"] and look["imageUrl"] is None
        action(client, project, "approve", status=409)
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        job = claim(client, qwen)
        base = saved["draft"]["characters"][0]["imageArtifactId"]
        assert job["kind"] == "m2_outfit_image" and job["payload"]["outfit"] == {
            "id": "look1", "name": "浴衣", "description": "藍色の浴衣に白い帯。", "occasion": "夏祭りの夜。"}
        assert job["payload"]["character_result"]["appearance"]
        assert job["payload"]["reference_image"]["artifact_id"] == base
        assert complete(client, job, qwen, outfit_bundle(job, sha256="0" * 64)).status_code == 422
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        (look,) = detail(client, project)["draft"]["characters"][0]["outfits"]
        assert not look["imagePendingChanges"] and look["baseImageArtifactId"] == base
        assert client.get(look["imageUrl"]).content == png(200)
        action(client, project, "save-outfit", character_id="character-1",
               outfit={"id": "default", "name": "旅装", "description": "青い布。"})
        approved = action(client, project, "approve")
        outfits = approved["draft"]["approval"]["characters"][0]["outfits"]
        assert [row["id"] for row in outfits] == ["default", "look1"]
        assert "image" not in outfits[0] and outfits[1]["image"]["artifactId"] == look["imageArtifactId"]
        assert outfits[1]["occasion"] == "夏祭りの夜。"
        # The story receives what the adopted image shows rather than the user's wording.
        assert look["generatedDescription"] == outfits[1]["description"] == "藍色の浴衣に白い帯を締めた姿。"


def test_changed_description_or_base_portrait_requires_a_new_image(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        qwen = outfit_worker(client)
        action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        job = claim(client, qwen)
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        edited = action(client, project, "save-outfit", character_id="character-1",
                        outfit={**LOOK, "id": "look1", "occasion": "花火大会の夜。"})
        assert not edited["draft"]["characters"][0]["outfits"][0]["imagePendingChanges"]
        edited = action(client, project, "save-outfit", character_id="character-1",
                        outfit={**LOOK, "id": "look1", "description": "白い浴衣。"})
        assert edited["draft"]["characters"][0]["outfits"][0]["imagePendingChanges"]
        action(client, project, "approve", status=409)
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        job = claim(client, qwen)
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        action(client, project, "retake", character_id="character-1", scope="image-retake")
        run_one(client, worker_id, "m2_image")
        assert detail(client, project)["draft"]["characters"][0]["outfits"][0]["imagePendingChanges"]
        action(client, project, "approve", status=409)
        action(client, project, "delete-outfit", character_id="character-1", outfit_id="look1")
        approved = action(client, project, "approve")
        assert "outfits" not in approved["draft"]["approval"]["characters"][0]


def test_character_without_looks_keeps_the_original_approval_shape(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        action(client, project, "save-outfit", character_id="character-1",
               outfit={"id": "default", "name": "旅装"})
        approved = action(client, project, "approve")
        assert set(approved["draft"]["approval"]["characters"][0]) == {
            "id", "result", "locked", "artifactId", "version", "sha256", "image", "voice"}
        named = action(client, project, "save-outfit", character_id="character-1",
                       outfit={"name": "浴衣", "description": ""})
        assert named["draft"]["characters"][0]["outfits"][-1]["imagePendingChanges"]
