"""Height edits obey appearance locks without regenerating approved image/audio assets."""

from fastapi.testclient import TestClient
from test_m2 import action, claim, complete, detail, ready

from services.coordinator.app import create_app


def test_height_edit_versions_result_preserves_media_and_immutable_approval(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        original = project["draft"]["characters"][0]
        approved = action(client, project, "approve")
        approval_id = approved["draft"]["approval"]["artifactId"]
        frozen = client.get(f"/api/artifacts/{approval_id}/content").content
        edited = action(client, project, "edit-character", character_id="character-1",
                        patch={"height_cm": 115, "body_type": "humanoid"})
        result = edited["draft"]["characters"][0]
        assert result["result"]["height_cm"] == 115
        assert result["result"]["body_type"] == "humanoid"
        assert result["imageArtifactId"] == original["imageArtifactId"]
        assert result["voiceArtifactId"] == original["voiceArtifactId"]
        assert not result["imagePendingChanges"] and not result["voicePendingChanges"]
        assert result["resultArtifactId"] != original["resultArtifactId"]
        assert client.get(f"/api/artifacts/{approval_id}/content").content == frozen
        action(client, project, "toggle-lock", character_id="character-1", scope="appearance")
        for patch in ({"height_cm": 170}, {"body_type": "nonhumanoid"}):
            action(client, project, "edit-character", character_id="character-1",
                   patch=patch, status=409)


def test_coordinator_preserves_dimensions_when_worker_changes_unrequested_scope(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = ready(client)
        edited = action(client, project, "edit-character", character_id="character-1",
                        patch={"height_cm": 115, "body_type": "humanoid"})
        source = edited["draft"]["characters"][0]["result"]
        action(client, project, "revise-character", character_id="character-1", scope="voice",
               instruction="より穏やかな声にする")
        job = claim(client, worker)
        generated = {**source, "voice": "穏やかな声。", "height_cm": 180,
                     "body_type": "nonhumanoid"}
        response = complete(client, job, worker, result=generated)
        assert response.status_code == 200, response.text
        actual = detail(client, project)["draft"]["characters"][0]["result"]
        assert actual["height_cm"] == 115 and actual["body_type"] == "humanoid"
