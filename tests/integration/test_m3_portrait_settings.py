"""Display-only height changes must retain story, media and immutable old builds."""

import io
import json
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from services.coordinator.m3_service import M3Service
from services.coordinator.service import Coordinator
from tests.integration.test_m3 import approved, finish, production
from tests.integration.test_m3_rebuild import counts


def settings(build_id):
    return {"expected_build_id": build_id, "characters": [
        {"character_id": "character-1", "framing": "upper_body", "height_cm": 170},
        {"character_id": "support-1", "framing": "upper_body", "height_cm": 120},
    ]}


def test_portrait_settings_publish_new_layout_without_regeneration(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        old = production(client, project)
        old_zip = client.get(old["export_url"]).content
        request = settings(old["build"]["id"])
        response = client.put(f"/api/m3/projects/{project}/portraits", json=request)
        assert response.status_code == 200, response.text
        new = response.json()["production"]
        assert new["build"]["revision"] == 2
        assert new["jobs"] == old["jobs"]
        assert new["narrative_artifact_id"] == old["narrative_artifact_id"]
        assert new["build"]["manifest"] == old["build"]["manifest"]
        assert new["build"]["validation"]["portrait_settings_artifact_id"]
        assert client.get(old["export_url"]).content == old_zip
        with ZipFile(io.BytesIO(old_zip)) as left, ZipFile(io.BytesIO(
            client.get(new["export_url"]).content)) as right:
            changed = {name for name in left.namelist() if left.read(name) != right.read(name)}
            assert changed <= {"manifest.json", "script.json", "data/scenario/first.ks",
                               "data/system/Config.tjs", "data/others/auto_drama_player.json",
                               # Portrait rectangles are also recorded per line and per scene.
                               "data/others/auto_drama_states.json", "data/others/auto_drama_stages.json"}
            # Scene-entry portraits are placed from the stage targets, not from scenario tags.
            assert "data/others/auto_drama_stages.json" in changed
            script = json.loads(right.read("script.json"))
            assert sorted(c["height_cm"] for c in script["characters"]) == [120, 170]
            assert all(c["framing"] == "upper_body" for c in script["characters"])
        before = counts(coordinator)
        assert client.put(f"/api/m3/projects/{project}/portraits", json=request).status_code == 409
        request["expected_build_id"] = new["build"]["id"]
        assert client.put(f"/api/m3/projects/{project}/portraits", json=request).status_code == 200
        assert counts(coordinator) == before
        assert client.post(f"/api/m3/projects/{project}/rebuild").json()["production"] == new
    with TestClient(create_app(tmp_path)) as client:
        assert production(client, project)["build"] == new["build"]


@pytest.mark.parametrize("characters", [
    [{"character_id": "unknown", "framing": "upper_body", "height_cm": 120}],
    [{"character_id": "character-1", "framing": "upper_body", "height_cm": 0}],
    [{"character_id": "character-1", "framing": "upper_body", "height_cm": True}],
    [{"character_id": "character-1", "framing": "full_body", "height_cm": 120}],
    [{"character_id": "character-1", "framing": "upper_body", "height_cm": 170}] * 2,
])
def test_invalid_portrait_settings_leave_current_build_unchanged(tmp_path, characters):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        old = production(client, project)
        before = counts(coordinator)
        request = {"expected_build_id": old["build"]["id"], "characters": characters}
        response = client.put(f"/api/m3/projects/{project}/portraits", json=request)
        assert response.status_code == 422, response.text
        assert production(client, project) == old
        assert counts(coordinator) == before


def test_failed_portrait_recompile_rolls_back_settings_and_build(tmp_path, monkeypatch):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        old = production(client, project)
        before = counts(coordinator)

        def fail(*args, **kwargs):
            raise ValueError("Cannot compile new layout")

        monkeypatch.setattr(M3Service, "_publish", fail)
        response = client.put(f"/api/m3/projects/{project}/portraits",
                              json=settings(old["build"]["id"]))
        assert response.status_code == 422
        assert counts(coordinator) == before
        assert production(client, project) == old
