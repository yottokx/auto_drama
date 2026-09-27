"""Portrait bounds are presentation-only and tied to the exact published image."""

import io
import json
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from packages.contracts.script import PortraitBounds
from services.coordinator.app import create_app
from services.coordinator.m3_service import M3Service
from services.coordinator.service import Coordinator
from tests.integration.test_m2 import action, create, png
from tests.integration.test_m3 import approved, finish, production
from tests.integration.test_m3_rebuild import counts

BOUNDS = {"left": 0.0, "top": 0.0, "right": 0.5, "bottom": 1.0}


def inspect(client, project):
    response = client.get(f"/api/m3/projects/{project}/portraits")
    assert response.status_code == 200, response.text
    return response.json()


def request_for(view, *, first_only=False):
    selected = view["characters"][:1] if first_only else view["characters"]
    return {"expected_build_id": view["build_id"], "characters": [
        {"character_id": character["character_id"], "framing": "upper_body",
         "height_cm": 170 - index * 50, "body_bounds": BOUNDS,
         "image_artifact_id": character["image_artifact_id"]}
        for index, character in enumerate(selected)
    ]}


def db_state(coordinator):
    with coordinator.db.transaction() as connection:
        return tuple(connection.iterdump())


def test_portraits_empty_before_publication_and_read_only_after_publication(tmp_path, monkeypatch):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        empty = create(client)["project"]["id"]
        assert inspect(client, empty) == {
            "project_id": empty, "production_id": None, "build_id": None, "characters": [],
        }
        project, worker = approved(client)
        pending = inspect(client, project)
        assert pending["production_id"] and pending["build_id"] is None
        assert pending["characters"] == []
        finish(client, worker)
        published = production(client, project)
        # Later edits to the M2 draft must not change the published names or images.
        action(client, project, "edit-character", character_id="character-1",
               patch={"name": "公開版とは異なる名前"})
        before = db_state(coordinator)

        def unexpected(*args):
            raise AssertionError("Portrait inspection must not recover or advance work")

        monkeypatch.setattr(M3Service, "advance", unexpected)
        monkeypatch.setattr(coordinator, "_recover", unexpected)
        view = inspect(client, project)
        assert view["build_id"] == published["build"]["id"]
        assert [value["character_id"] for value in view["characters"]] == [
            "character-1", "support-1",
        ]
        assert [value["name"] for value in view["characters"]] == ["リオ", "エマ"]
        for character in view["characters"]:
            assert character["body_bounds"] is None
            assert character["image_url"] == (
                f"/api/artifacts/{character['image_artifact_id']}/content"
            )
            assert client.get(character["image_url"]).status_code == 200
        assert inspect(client, project) == view
        assert db_state(coordinator) == before
        assert client.get("/api/m3/projects/missing/portraits").status_code == 404


def test_bounds_publish_and_reset_preserve_story_media_jobs_and_previous_build(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        original = production(client, project)
        old_zip = client.get(original["export_url"]).content
        original_view = inspect(client, project)
        before = counts(coordinator)
        request = request_for(original_view)
        response = client.put(f"/api/m3/projects/{project}/portraits", json=request)
        assert response.status_code == 200, response.text
        updated = response.json()["production"]
        view = inspect(client, project)
        assert view["build_id"] == updated["build"]["id"] != original_view["build_id"]
        assert updated["jobs"] == original["jobs"]
        assert updated["narrative_artifact_id"] == original["narrative_artifact_id"]
        assert updated["approval_artifact_id"] == original["approval_artifact_id"]
        assert counts(coordinator)["job"] == before["job"]
        assert counts(coordinator)["job_attempt"] == before["job_attempt"]
        assert client.get(original["export_url"]).content == old_zip
        for index, character in enumerate(view["characters"]):
            assert character["body_bounds"] == BOUNDS
            assert character["height_cm"] == 170 - index * 50
            assert character["framing"] == "upper_body"
            assert character["image_artifact_id"] == original_view["characters"][index]["image_artifact_id"]
        with ZipFile(io.BytesIO(old_zip)) as left, ZipFile(io.BytesIO(
            client.get(updated["export_url"]).content)) as right:
            for name in left.namelist():
                if name in {"approval.json", "narrative.json"} or name.startswith((
                    "sources/", "data/fgimage/", "data/bgimage/", "data/sound/",
                )):
                    assert right.read(name) == left.read(name)
            script = json.loads(right.read("script.json"))
            assert all(value["body_bounds"] == BOUNDS for value in script["characters"])
            assert all("image_artifact_id" not in value for value in script["characters"])
        assert client.put(f"/api/m3/projects/{project}/portraits", json=request).status_code == 409
        request["expected_build_id"] = view["build_id"]
        unchanged_counts = counts(coordinator)
        assert client.put(f"/api/m3/projects/{project}/portraits", json=request).status_code == 200
        assert counts(coordinator) == unchanged_counts

        # PUT replaces the complete setting set; omitted characters regain their defaults.
        response = client.put(f"/api/m3/projects/{project}/portraits",
                              json=request_for(view, first_only=True))
        assert response.status_code == 200, response.text
        partial = inspect(client, project)
        assert partial["characters"][0]["body_bounds"] == BOUNDS
        assert partial["characters"][1] == original_view["characters"][1]
        response = client.put(f"/api/m3/projects/{project}/portraits", json={
            "expected_build_id": partial["build_id"], "characters": [],
        })
        assert response.status_code == 200, response.text
        assert inspect(client, project)["characters"] == original_view["characters"]
        assert counts(coordinator)["job"] == before["job"]


@pytest.mark.parametrize("patch", [
    {"body_bounds": {**BOUNDS, "left": -0.1}},
    {"body_bounds": {**BOUNDS, "right": 1.1}},
    {"body_bounds": {**BOUNDS, "left": 0.6}},
    {"body_bounds": {**BOUNDS, "right": 0.049}},
    {"body_bounds": {**BOUNDS, "bottom": 0.049}},
    {"body_bounds": {**BOUNDS, "left": "0"}},
    {"body_bounds": {**BOUNDS, "left": False}},
    {"image_artifact_id": None},
    {"image_artifact_id": "a-different-image"},
])
def test_invalid_bounds_or_image_identity_cannot_change_publication(tmp_path, patch):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        view = inspect(client, project)
        request = request_for(view, first_only=True)
        request["characters"][0].update(patch)
        before = db_state(coordinator)
        response = client.put(f"/api/m3/projects/{project}/portraits", json=request)
        assert response.status_code == 422, response.text
        assert db_state(coordinator) == before
        assert inspect(client, project) == view


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_portrait_bounds_reject_nonfinite_coordinates(value):
    with pytest.raises(ValidationError):
        PortraitBounds.model_validate({**BOUNDS, "left": value})


def test_portrait_bounds_accept_five_percent_size_with_float_rounding():
    assert PortraitBounds(left=0.1, top=0.5, right=0.15, bottom=0.55)


def test_image_replacement_ignores_old_bounds_but_keeps_height_and_framing(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        view = inspect(client, project)
        response = client.put(f"/api/m3/projects/{project}/portraits",
                              json=request_for(view, first_only=True))
        assert response.status_code == 200, response.text
        bounded = inspect(client, project)
        old = production(client, project)
        with coordinator.db.transaction() as connection:
            replacement = coordinator._register_artifact(
                connection, project, "replacement-portrait", "character", "replacement.png",
                "image/png", coordinator.store.put(png(color=180)),
            )
            connection.execute(
                "UPDATE m3_requirement SET artifact_id=? WHERE production_id=? "
                "AND kind='m3_image' AND target_id='character-1'",
                (replacement["id"], old["id"]),
            )
        # Until rebuilding, GET must still describe the image in the published build.
        assert inspect(client, project) == bounded
        stale = request_for(bounded, first_only=True)
        assert client.put(f"/api/m3/projects/{project}/portraits", json=stale).status_code == 422
        jobs_before = counts(coordinator)["job"]
        response = client.post(f"/api/m3/projects/{project}/rebuild")
        assert response.status_code == 200, response.text
        current = inspect(client, project)["characters"][0]
        assert current["image_artifact_id"] == replacement["id"]
        assert current["body_bounds"] is None
        assert current["height_cm"] == 170
        assert current["framing"] == "upper_body"
        assert counts(coordinator)["job"] == jobs_before


def test_bounds_recompile_failure_rolls_back_setting_and_build(tmp_path, monkeypatch):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        view = inspect(client, project)
        before = db_state(coordinator)

        def fail(*args):
            raise ValueError("Invalid body layout")

        monkeypatch.setattr(M3Service, "_publish", fail)
        response = client.put(f"/api/m3/projects/{project}/portraits", json=request_for(view))
        assert response.status_code == 422
        assert db_state(coordinator) == before
        assert inspect(client, project) == view
