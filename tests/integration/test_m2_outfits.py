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


def outfit_bundle(job, *, sha256=None, media=None, occasion=None):
    payload = job["payload"]
    envelope = {"schema_version": 1, "kind": "m2_outfit_image", "result": {}, "trace": [],
                "provenance": {"seed": payload["seed"], "outfit": {
                    "id": payload["outfit"]["id"],
                    "summary": payload["outfit"].get("appearance", "藍色の浴衣に白い帯を締めた姿。"),
                    "occasion": occasion or payload["outfit"].get("occasion"),
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
        # The look is redrawn from the new portrait automatically; until then nothing else may change.
        action(client, project, "approve", status=409)
        job = claim(client, qwen)
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        action(client, project, "delete-outfit", character_id="character-1", outfit_id="look1")
        approved = action(client, project, "approve")
        assert "outfits" not in approved["draft"]["approval"]["characters"][0]


def test_unstated_occasion_is_filled_in_and_a_direct_edit_is_drawn_as_written(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        qwen = outfit_worker(client)
        action(client, project, "save-outfit", character_id="character-1",
               outfit={"name": "浴衣", "description": "涼しげに"})
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        job = claim(client, qwen)
        assert "appearance" not in job["payload"]["outfit"]
        assert complete(client, job, qwen, outfit_bundle(job, occasion="夏祭りに出かける夜")).status_code == 200
        (look,) = detail(client, project)["draft"]["characters"][0]["outfits"]
        # The wish stays what the user wrote; what was made is kept beside it.
        assert (look["description"], look["occasion"]) == ("涼しげに", "")
        assert look["generatedOccasion"] == "夏祭りに出かける夜"
        shown = {"id": "look1", "name": "夏の浴衣", "description": "涼しげに",
                 "appearance": look["generatedDescription"]}
        renamed = action(client, project, "save-outfit", character_id="character-1", outfit=shown)
        assert not renamed["draft"]["characters"][0]["outfits"][0]["imagePendingChanges"]
        edited = action(client, project, "save-outfit", character_id="character-1",
                        outfit={**shown, "appearance": "白地に朝顔柄の浴衣と赤い帯。"})
        (look,) = edited["draft"]["characters"][0]["outfits"]
        assert look["imagePendingChanges"] and look["description"] == "涼しげに"
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        job = claim(client, qwen)
        assert job["payload"]["outfit"]["appearance"] == "白地に朝顔柄の浴衣と赤い帯。"
        assert complete(client, job, qwen, outfit_bundle(job, occasion="夏祭りに出かける夜")).status_code == 200
        approved = action(client, project, "approve")
        row = approved["draft"]["approval"]["characters"][0]["outfits"][-1]
        assert (row["description"], row["occasion"]) == ("白地に朝顔柄の浴衣と赤い帯。", "夏祭りに出かける夜")
        # A new wish replaces the edited text with a fresh design.
        wished = action(client, project, "save-outfit", character_id="character-1",
                        outfit={"id": "look1", "name": "夏の浴衣", "description": "もっと華やかに"})
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        assert wished["draft"]["characters"][0]["outfits"][0]["imagePendingChanges"]
        assert "appearance" not in claim(client, qwen)["payload"]["outfit"]


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


def brief(client, project, outfits):
    current = detail(client, project)
    return action(client, project, "save-brief", world=current["draft"]["worldInput"],
                  characters=[row["input"] for row in current["draft"]["characters"]], outfits=outfits)


def test_look_wished_for_in_the_brief_is_drawn_after_the_base_portrait(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        from tests.integration.test_m2 import create, worker

        project, identifier = create(client, prompt="異星の物語を自由に考える。"), worker(client)
        qwen = outfit_worker(client)
        saved = brief(client, project, {"character-1": [{"name": "浴衣", "description": "夏祭りっぽく"}]})
        (look,) = saved["draft"]["characters"][0]["outfits"]
        assert look["id"] == "look1" and look["imagePendingChanges"]
        action(client, project, "generate-world")
        run_one(client, identifier, "m2_world")
        action(client, project, "confirm-world")
        for kind in ("m2_character", "m2_image", "m2_voice"):
            run_one(client, identifier, kind)
        job = claim(client, qwen)
        assert job["kind"] == "m2_outfit_image" and job["payload"]["outfit"]["description"] == "夏祭りっぽく"
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        current = detail(client, project)
        assert not current["draft"]["characters"][0]["outfits"][0]["imagePendingChanges"]
        assert claim(client, qwen) is None
        approved = action(client, project, "approve")
        assert approved["draft"]["approval"]["characters"][0]["outfits"][-1]["id"] == "look1"
        # The story is told what the base look shows, taken from the character's appearance.
        base = approved["draft"]["approval"]["characters"][0]["outfits"][0]
        assert base["id"] == "default" and base["description"] == approved["draft"]["characters"][0]["result"]["appearance"]
        # Retaking the base portrait redraws the look from it without being asked.
        action(client, project, "retake", character_id="character-1", scope="image-retake")
        run_one(client, identifier, "m2_image")
        again = claim(client, qwen)
        assert again["kind"] == "m2_outfit_image"
        assert again["payload"]["reference_image"]["artifact_id"] != job["payload"]["reference_image"]["artifact_id"]


def test_brief_looks_are_edited_removed_and_kept_without_touching_the_cast(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        results = [row["resultVersion"] for row in detail(client, project)["draft"]["characters"]]
        saved = brief(client, project, {"character-1": [{"name": "浴衣"}, {"name": "鎧", "occasion": "戦う時"}]})
        looks = saved["draft"]["characters"][0]["outfits"]
        assert [row["id"] for row in looks] == ["look1", "look2"]
        assert not saved["draft"]["characters"][0]["pendingChanges"] and not saved["draft"]["worldPendingChanges"]
        saved = brief(client, project, {"character-1": [{"id": "look2", "name": "鎧", "occasion": "決戦の時"}]})
        (look,) = saved["draft"]["characters"][0]["outfits"]
        assert (look["id"], look["occasion"]) == ("look2", "決戦の時")
        untouched = brief(client, project, None)
        assert untouched["draft"]["characters"][0]["outfits"] == saved["draft"]["characters"][0]["outfits"]
        assert [row["resultVersion"] for row in untouched["draft"]["characters"]] == results
        # No Qwen worker: the look waits, and the cast cannot be approved with it undrawn.
        action(client, project, "confirm-world")
        assert all(job["kind"] != "m2_outfit_image" for job in detail(client, project)["jobs"])
        action(client, project, "approve", status=409)


def test_a_look_waits_for_a_worker_that_holds_the_editing_model_in_use(tmp_path):
    from packages.contracts.event_cg import new_production_profile

    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        standard = outfit_worker(client)
        action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
        settings = client.get("/api/event-cg/settings").json()
        assert [(row["id"], row["available"]) for row in settings["models"]] == [("standard", True), ("turbo", False)]
        turbo_profile = new_production_profile("turbo").model_dump(mode="json")
        saved = client.put("/api/event-cg/settings", json={"expected_revision": settings["revision"],
                                                           "profile": turbo_profile})
        assert saved.status_code == 200 and saved.json()["ready"] is False
        # No connected worker holds Turbo, so the look cannot be drawn yet.
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1", status=409)
        turbo = client.post("/api/workers", json={
            "name": "turbo", "capabilities": ["m2_outfit_image", "qwen_image21_turbo"]}).json()["id"]
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        # The worker with only the standard model leaves the job for the one that has Turbo.
        assert claim(client, standard) is None
        job = claim(client, turbo)
        assert job["kind"] == "m2_outfit_image"
        assert (job["payload"]["cg_profile"]["model"], job["payload"]["cg_profile"]["steps"]) == ("turbo", 8)
