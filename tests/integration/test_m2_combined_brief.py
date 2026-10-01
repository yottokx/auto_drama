"""Combined instructions precede world approval and automatic cast generation."""

import copy
import json

import pytest
from fastapi.testclient import TestClient
from test_m2 import WORLD, action, claim, complete, create, detail, run_one, worker

from packages.contracts.m2 import CharacterBrief
from services.coordinator.app import create_app
from services.coordinator.service import Coordinator


def brief_values(project):
    draft = project["draft"]
    return {
        "world": copy.deepcopy(draft["worldInput"]),
        "characters": [copy.deepcopy(value["input"]) for value in draft["characters"]],
        "relationshipInputs": copy.deepcopy(draft["relationshipInputs"]),
    }


def generate_combined(client, count=2):
    project, worker_id = create(client), worker(client)
    values = {
        "world": {**WORLD, "prompt": "機械の街で、指定した二人の物語。"},
        "characters": [
            CharacterBrief(id=f"person-{index}", freeform=f"指定された人物{index}").model_dump()
            for index in range(1, count + 1)
        ],
        "relationshipInputs": [
            {"characterIds": ["person-1", "person-2"], "instruction": "かつての同僚"}
        ]
        if count >= 2
        else [],
    }
    saved = action(client, project, "save-brief", **values)
    assert saved["draft"]["step"] == "world-input"
    assert not saved["jobs"]
    action(client, project, "generate-characters", status=409)
    action(client, project, "generate-world")
    job, _ = run_one(client, worker_id, "m2_world")
    assert job["payload"]["cast_inputs"] == values["characters"]
    assert job["payload"]["relationship_inputs"] == values["relationshipInputs"]
    reviewed = detail(client, project)
    assert reviewed["draft"]["step"] == "world-review"
    assert not reviewed["draft"]["worldConfirmed"]
    assert not reviewed["draft"]["activeJobId"]
    assert len(reviewed["jobs"]) == 1
    confirmed = action(client, project, "confirm-world")
    assert confirmed["draft"]["step"] == "character-review"
    assert confirmed["draft"]["worldConfirmed"]
    assert confirmed["jobs"][-1]["kind"] == "m2_character"
    return confirmed, worker_id


@pytest.mark.parametrize("count", [1, 3])
def test_combined_brief_world_confirmation_automatically_generates_complete_cast(tmp_path, count):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = generate_combined(client, count)
        original_brief = brief_values(project)
        confirmation = project["draft"]["worldConfirmationArtifactId"]
        confirmation_bytes = client.get(f"/api/artifacts/{confirmation}/content").content
        # Repeated confirmation during the generated pipeline is harmless.
        repeated = action(client, project, "confirm-world")
        assert repeated == project
        jobs = []
        while job := claim(client, worker_id):
            jobs.append(job)
            assert job["payload"]["world_result"] == WORLD
            assert job["payload"]["world_input"] == original_brief["world"]
            assert job["payload"]["cast_inputs"] == original_brief["characters"]
            assert job["payload"]["relationship_inputs"] == original_brief["relationshipInputs"]
            if job["payload"]["character_id"]:
                assert job["payload"]["character_input"] == next(
                    person for person in original_brief["characters"]
                    if person["id"] == job["payload"]["character_id"]
                )
            assert complete(client, job, worker_id).status_code == 200
        kinds = [job["kind"] for job in jobs]
        assert kinds == [
            kind for _ in range(count) for kind in ("m2_character", "m2_image", "m2_voice")
        ] + (["m2_relationships"] if count > 1 else [])
        ready = detail(client, project)
        assert all(not character["pendingChanges"] for character in ready["draft"]["characters"])
        approved = action(client, project, "approve")
        assert approved["draft"]["approved"]
        assert client.get(f"/api/artifacts/{confirmation}/content").content == confirmation_bytes


@pytest.mark.parametrize("changed", ["world", "character", "relationship", "cast"])
def test_combined_input_change_invalidates_confirmation_and_keeps_published_snapshot(
    tmp_path, changed
):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = generate_combined(client)
        while job := claim(client, worker_id):
            assert complete(client, job, worker_id).status_code == 200
        approved = action(client, project, "approve")
        artifact = approved["draft"]["approval"]["artifactId"]
        original = client.get(f"/api/artifacts/{artifact}/content").content
        values = brief_values(approved)
        if changed == "world":
            values["world"]["prompt"] = "違う時代へ。"
        elif changed == "character":
            values["characters"][0]["freeform"] = "街に来たばかりの時計職人。"
        elif changed == "relationship":
            values["relationshipInputs"][0]["instruction"] = "幼なじみ。"
        else:
            values["characters"].append(CharacterBrief(id="person-3").model_dump())
        saved = action(client, project, "save-brief", **values)
        draft = saved["draft"]
        assert not draft["approved"]
        assert not draft["approval"]
        assert not draft["worldConfirmed"]
        assert not draft["worldConfirmationArtifactId"]
        assert draft["worldPendingChanges"]
        assert all(value["pendingChanges"] for value in draft["characters"])
        assert draft["relationships"]["pendingChanges"]
        assert draft["worldArtifactId"] == approved["draft"]["worldArtifactId"]
        assert (
            draft["characters"][0]["imageArtifactId"]
            == approved["draft"]["characters"][0]["imageArtifactId"]
        )
        action(client, project, "confirm-world", status=409)
        action(client, project, "generate-characters", status=409)
        assert client.get(f"/api/artifacts/{artifact}/content").content == original


def test_unchanged_combined_brief_and_confirmation_preserve_approval_and_jobs(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = generate_combined(client)
        while job := claim(client, worker_id):
            assert complete(client, job, worker_id).status_code == 200
        approved = action(client, project, "approve")
        saved = action(client, project, "save-brief", **brief_values(approved))
        assert saved["draft"]["approved"]
        assert saved["draft"]["approval"] == approved["draft"]["approval"]
        assert saved["draft"]["revision"] == approved["draft"]["revision"]
        assert saved["jobs"] == approved["jobs"]
        confirmed = action(client, project, "confirm-world")
        assert confirmed["draft"]["approval"] == approved["draft"]["approval"]
        assert confirmed["draft"]["revision"] == approved["draft"]["revision"]
        assert confirmed["jobs"] == approved["jobs"]


@pytest.mark.parametrize("bad", ["empty", "duplicate", "unknown-pair", "too-many"])
def test_combined_brief_validation_is_atomic(tmp_path, bad):
    with TestClient(create_app(tmp_path)) as client:
        project = create(client)
        values = brief_values(project)
        values["world"]["prompt"] = "この変更も不正な人物指定と一緒には保存されない。"
        if bad == "empty":
            values["characters"] = []
        elif bad == "duplicate":
            values["characters"] *= 2
        elif bad == "unknown-pair":
            values["relationshipInputs"] = [
                {"characterIds": ["character-1", "missing"], "instruction": "不明な人物"}
            ]
        else:
            values["characters"] = [
                CharacterBrief(id=f"p-{index}").model_dump() for index in range(4)
            ]
        action(client, project, "save-brief", status=422, **values)
        assert detail(client, project) == project


def test_legacy_character_input_step_maps_without_rewriting_saved_content(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project = create(client)
        project_id = project["project"]["id"]
        with coordinator.db.transaction() as connection:
            state = json.loads(connection.execute("SELECT state FROM m2_draft").fetchone()[0])
            state["draft"]["step"] = "character-input"
            state["draft"]["characters"][0]["input"]["freeform"] = "以前保存した指示"
            original = json.dumps(state)
            connection.execute("UPDATE m2_draft SET state=?", (original,))
        restored = detail(client, project)
        assert restored["draft"]["step"] == "world-input"
        assert restored["draft"]["characters"][0]["input"]["freeform"] == "以前保存した指示"
        with coordinator.db.transaction() as connection:
            assert connection.execute("SELECT state FROM m2_draft").fetchone()[0] == original
        navigated = action(client, project_id, "go-to", step="character-input")
        assert navigated["draft"]["step"] == "world-input"
        assert navigated["draft"]["revision"] == restored["draft"]["revision"]


def test_legacy_confirmed_world_with_missing_cast_continues_existing_confirmation(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = generate_combined(client, count=1)
        confirmation = project["draft"]["worldConfirmationArtifactId"]
        with coordinator.db.transaction() as connection:
            # Previously confirmation stopped at character-input before queueing cast work.
            connection.execute("DELETE FROM job WHERE kind='m2_character'")
            state = json.loads(connection.execute("SELECT state FROM m2_draft").fetchone()[0])
            state["draft"]["step"] = "character-input"
            state["activeJobId"] = None
            state["queue"] = []
            connection.execute("UPDATE m2_draft SET state=?", (json.dumps(state),))
        confirmed = action(client, project, "confirm-world")
        assert confirmed["draft"]["step"] == "character-review"
        assert confirmed["draft"]["worldConfirmationArtifactId"] == confirmation
        for kind in ("m2_character", "m2_image", "m2_voice"):
            run_one(client, worker_id, kind)


def test_locked_cast_does_not_bypass_world_regeneration_after_combined_edit(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = generate_combined(client, count=1)
        for kind in ("m2_character", "m2_image", "m2_voice"):
            run_one(client, worker_id, kind)
        for scope in ("settings", "appearance", "voice"):
            project = action(client, project, "toggle-lock", character_id="person-1", scope=scope)
        previous = project["draft"]["characters"][0]
        values = brief_values(project)
        values["world"]["prompt"] = "新たな時代の世界。"
        saved = action(client, project, "save-brief", **values)
        assert saved["draft"]["characters"][0]["locked"] == previous["locked"]
        assert saved["draft"]["characters"][0]["resultArtifactId"] == previous["resultArtifactId"]
        action(client, project, "confirm-world", status=409)
        action(client, project, "generate-characters", status=409)
        action(client, project, "approve", status=409)
        action(client, project, "generate-world")
        action(client, project, "save-brief", status=409, **values)
        run_one(client, worker_id, "m2_world")
        confirmed = action(client, project, "confirm-world")
        assert confirmed["draft"]["worldConfirmed"]
        assert not confirmed["draft"]["activeJobId"]
        assert (
            confirmed["draft"]["characters"][0]["resultArtifactId"] == previous["resultArtifactId"]
        )
