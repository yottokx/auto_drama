"""Character revision routing and adoption must keep the requested cast member."""

import copy

import pytest
from fastapi.testclient import TestClient
from test_m2 import action, claim, complete, detail
from test_m2_cast import ready_cast

from services.coordinator.app import create_app


def _distinct_cast(client):
    project, worker_id = ready_cast(client, count=2)
    project = action(
        client, project, "edit-character", character_id="person-2",
        patch={"name": "セレナ", "role": "主人公を支える調律師",
               "settings": "慎重な記録官とは異なり、即断を好む旅の調律師。"},
    )
    return project, worker_id


def _artifacts(client, project):
    return client.get(f"/api/projects/{project['project']['id']}").json()["artifacts"]


def test_second_character_revision_payload_keeps_its_input_result_and_scope(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = _distinct_cast(client)
        first, second = copy.deepcopy(project["draft"]["characters"])
        instruction = "名前だけを和洋折衷にしてください。ほかは維持してください。"
        action(client, project, "revise-character", character_id=second["id"],
               scope="settings", instruction=instruction)

        job = claim(client, worker_id)
        payload = job["payload"]
        assert job["kind"] == "m2_character"
        assert payload["character_id"] == second["id"]
        assert payload["character_input"] == second["input"]
        assert payload["character_result"] == second["result"]
        assert payload["scope"] == "settings"
        assert payload["instruction"] == instruction
        assert payload["cast_results"] == [first["result"], second["result"]]
        draft = detail(client, project)["draft"]
        assert draft["requests"][-1]["characterId"] == second["id"]
        assert draft["characters"][0] == first
        assert draft["characters"][1]["result"] == second["result"]


@pytest.mark.parametrize("scope", ["all", "settings"])
def test_first_character_output_cannot_be_adopted_as_second_character(tmp_path, scope):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = _distinct_cast(client)
        first, second = copy.deepcopy(project["draft"]["characters"])
        action(client, project, "revise-character", character_id=second["id"],
               scope=scope, instruction="名前を和洋折衷にしてください。")
        job = claim(client, worker_id)
        before = detail(client, project)["draft"]
        artifacts = _artifacts(client, project)

        rejected = complete(client, job, worker_id, result=first["result"])

        assert rejected.status_code == 422
        assert detail(client, project)["draft"] == before
        assert _artifacts(client, project) == artifacts
        job_state = client.get(f"/api/jobs/{job['id']}").json()["job"]
        assert job_state["status"] == "running"
        assert job_state["result_artifact_id"] is None
        after = detail(client, project)["draft"]["characters"]
        assert after[0] == first
        assert after[1]["result"] == second["result"]
        for field in ("resultArtifactId", "imageArtifactId", "voiceArtifactId"):
            assert after[1][field] == second[field]


def test_second_character_name_revision_adopts_only_target_and_preserves_old_artifacts(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = _distinct_cast(client)
        first, second = copy.deepcopy(project["draft"]["characters"])
        old_artifact = second["resultArtifactId"]
        old_content = client.get(f"/api/artifacts/{old_artifact}/content").content
        action(client, project, "revise-character", character_id=second["id"],
               scope="settings", instruction="名前だけを和洋折衷にしてください。")
        job = claim(client, worker_id)
        revised = {**second["result"], "name": "月代セレナ"}

        response = complete(client, job, worker_id, result=revised)

        assert response.status_code == 200, response.text
        draft = detail(client, project)["draft"]
        assert draft["characters"][0] == first
        target = draft["characters"][1]
        assert target["id"] == second["id"]
        assert target["result"] == revised
        assert target["resultVersion"] == second["resultVersion"] + 1
        assert target["resultArtifactId"] != old_artifact
        assert client.get(f"/api/artifacts/{old_artifact}/content").content == old_content
        for field in ("imageArtifactId", "voiceArtifactId"):
            assert target[field] == second[field]
        assert not target["pendingChanges"]
        assert not target["imagePendingChanges"]
        assert not target["voicePendingChanges"]
        # A name change refreshes cast relationships without regenerating either person's media.
        next_job = claim(client, worker_id)
        assert next_job["kind"] == "m2_relationships"
        assert next_job["payload"]["cast_results"] == [first["result"], revised]
