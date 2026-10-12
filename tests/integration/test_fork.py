"""Another work started from a chosen step of an existing one, which stays as it was."""

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from tests.integration.test_m2 import action, claim, complete, detail, ready
from tests.integration.test_m2_outfits import LOOK, outfit_bundle
from tests.integration.test_materials import made_material, materials, workers
from tests.integration.test_planning import plan_worker, planning


def fork(client, project, through, status=200):
    response = client.post(f"/api/forks/projects/{project['project']['id']}", json={"through": through})
    assert response.status_code == status, response.text
    return response.json()


def levels(client, project):
    return client.get(f"/api/forks/projects/{project['project']['id']}").json()["levels"]


def finished_cast(client):
    project, main = ready(client)
    anima, qwen = workers(client)
    sheet = made_material(client, project["project"]["id"], anima)["images"][0]
    action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
    action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1",
           material_image_id=sheet["id"])
    job = claim(client, qwen)
    assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
    return detail(client, project), main


def test_a_fork_carries_what_was_settled_up_to_its_step_and_leaves_the_source_alone(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, main = finished_cast(client)
        before = detail(client, project)
        assert levels(client, project) == ["brief", "world", "cast"]
        fork(client, project, "plan", status=409)

        brief = fork(client, project, "brief")
        draft = brief["draft"]
        assert brief["project"]["id"] != project["project"]["id"] and draft["step"] == "world-input"
        assert draft["worldInput"] == before["draft"]["worldInput"] and draft["worldResult"] is None
        assert draft["characters"][0]["input"] == before["draft"]["characters"][0]["input"]
        assert draft["characters"][0]["result"] is None and draft["characters"][0]["imageArtifactId"] is None
        # What the look is meant to be comes along; its picture does not.
        (look,) = draft["characters"][0]["outfits"]
        assert look["name"] == "浴衣" and look["imageArtifactId"] is None
        assert materials(client, brief["project"]["id"]) == []

        world = fork(client, project, "world")
        assert world["draft"]["step"] == "world-review" and world["draft"]["worldResult"] == before["draft"]["worldResult"]
        assert not world["draft"]["worldConfirmed"] and world["draft"]["characters"][0]["result"] is None
        (material,) = materials(client, world["project"]["id"])
        assert material["images"][0]["status"] == "ready"
        # The carried world is confirmed in the new work and its own cast is made from it.
        action(client, world, "confirm-world")
        job = claim(client, main)
        assert job["kind"] == "m2_character" and job["project_id"] == world["project"]["id"]
        assert job["payload"]["character_result"] is None

        cast = fork(client, project, "cast")
        character, old = cast["draft"]["characters"][0], before["draft"]["characters"][0]
        assert cast["draft"]["step"] == "character-review" and not cast["draft"]["approved"]
        assert character["result"] == old["result"] and character["imageArtifactId"] != old["imageArtifactId"]
        assert client.get(character["imageUrl"]).content == client.get(old["imageUrl"]).content
        assert client.get(character["voiceUrl"]).content == client.get(old["voiceUrl"]).content
        (look,) = [row for row in character["outfits"] if row["id"] == "look1"]
        assert client.get(look["imageUrl"]).content and not look["imagePendingChanges"]
        assert look["baseImageArtifactId"] == character["imageArtifactId"]
        assert look["materialImageId"] == materials(client, cast["project"]["id"])[0]["images"][0]["id"]
        assert detail(client, project) == before


def test_a_fork_through_the_plan_starts_at_its_review_with_the_planned_cast_drawn(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        from tests.integration.test_m3 import finish
        from tests.integration.test_planning_assets import assets, reviewed_with_support

        project, media = reviewed_with_support(client)
        finish(client, media)
        source = planning(client, project)
        assert levels(client, project) == ["brief", "world", "cast", "plan"]
        copy = fork(client, project, "plan")
        assert copy["draft"]["step"] == "planning-review" and copy["draft"]["approved"]
        carried = planning(client, copy)
        assert carried["content"] == source["content"] and carried["status"] == "ready"
        assert {kind: row["status"] for kind, row in assets(client, copy).items()} == {"image": "ready", "voice": "ready"}
        # Nothing is made again for the carried plan, and it can be approved as it stands.
        assert claim(client, plan_worker(client)) is None and claim(client, media) is None
        response = client.post(f"/api/planning/projects/{copy['project']['id']}/actions",
                               json={"action": "approve", "expected_revision": carried["revision"]})
        assert response.status_code == 200, response.text
        job = claim(client, media)
        assert job["kind"] == "m3_narrative" and job["project_id"] == copy["project"]["id"]
        assert job["payload"]["approved_plan"]["content"] == source["content"]
