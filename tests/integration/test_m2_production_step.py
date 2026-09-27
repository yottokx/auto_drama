"""The production step follows approval while backward navigation preserves its inputs."""

import copy
import json

import pytest
from fastapi.testclient import TestClient
from test_m2 import action, create, detail, ready

from services.coordinator.app import create_app
from services.coordinator.service import Coordinator


def test_approval_enters_production_and_revisiting_settings_preserves_existing_work(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        assert project["draft"]["step"] == "character-review"
        approved = action(client, project, "approve")
        assert approved["draft"]["step"] == "production"
        assert approved["draft"]["hasProduction"]
        snapshot = approved["draft"]["approval"]
        frozen = client.get(f"/api/artifacts/{snapshot['artifactId']}/content").content
        revision, jobs = approved["draft"]["revision"], approved["jobs"]
        assert len([job for job in jobs if job["kind"] == "m3_narrative"]) == 1

        for step in ("world-input", "world-review", "character-review", "production"):
            current = action(client, project, "go-to", step=step)
            assert current["draft"]["step"] == step
            assert detail(client, project)["draft"]["step"] == step
            assert current["draft"]["approved"] and current["draft"]["approval"] == snapshot
            assert current["draft"]["revision"] == revision
            assert current["jobs"] == jobs

        action(client, project, "go-to", step="world-review")
        confirmed = action(client, project, "confirm-world")
        assert confirmed["draft"]["step"] == "character-review"
        assert detail(client, project)["draft"]["step"] == "character-review"
        approved_again = action(client, project, "approve")
        assert approved_again["draft"]["step"] == "production"
        assert approved_again["draft"]["approval"] == snapshot
        assert approved_again["draft"]["revision"] == revision
        assert approved_again["jobs"] == jobs
        assert client.get(f"/api/artifacts/{snapshot['artifactId']}/content").content == frozen


@pytest.mark.parametrize("has_results", [False, True])
def test_production_requires_approval_even_when_all_results_are_ready(tmp_path, has_results):
    with TestClient(create_app(tmp_path)) as client:
        project = ready(client)[0] if has_results else create(client)
        before = copy.deepcopy(project)
        assert not before["draft"]["hasProduction"]
        action(client, project, "go-to", step="production", status=409)
        assert detail(client, project) == before


@pytest.mark.parametrize("saved_step,expected", [
    ("character-review", "production"),
    ("world-input", "world-input"),
    ("world-review", "world-review"),
])
def test_legacy_approval_moves_to_production_only_from_its_old_default_step(
    tmp_path, saved_step, expected,
):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _ = ready(client)
        approved = action(client, project, "approve")
        identifier = project["project"]["id"]
        snapshot = approved["draft"]["approval"]
        with coordinator.db.transaction() as connection:
            state = json.loads(connection.execute(
                "SELECT state FROM m2_draft WHERE project_id=?", (identifier,),
            ).fetchone()[0])
            state.pop("wizardStepVersion")
            state["draft"]["step"] = saved_step
            original = json.dumps(state)
            connection.execute("UPDATE m2_draft SET state=? WHERE project_id=?", (original, identifier))
        restored = detail(client, project)
        assert restored["draft"]["step"] == expected
        assert restored["draft"]["approval"] == snapshot
        assert restored["jobs"] == approved["jobs"]
        with coordinator.db.transaction() as connection:
            assert connection.execute(
                "SELECT state FROM m2_draft WHERE project_id=?", (identifier,),
            ).fetchone()[0] == original
        returned = action(client, project, "go-to", step="character-review")
        assert returned["draft"]["step"] == "character-review"
        assert returned["draft"]["approved"]
    with TestClient(create_app(tmp_path)) as client:
        restored = detail(client, identifier)
        assert restored["draft"]["step"] == "character-review"
        assert restored["draft"]["approval"] == snapshot


@pytest.mark.parametrize("change,expected", [
    ("character", "character-review"),
    ("world-result", "world-review"),
    ("brief", "world-input"),
])
def test_editing_from_production_returns_to_the_relevant_setup_step(tmp_path, change, expected):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        approved = action(client, project, "approve")
        snapshot = approved["draft"]["approval"]
        frozen = client.get(f"/api/artifacts/{snapshot['artifactId']}/content").content
        if change == "character":
            result = action(client, project, "edit-character", character_id="character-1",
                            patch={"settings": "人々の思い出を慎重に記録する研究者。"})
        elif change == "world-result":
            result = action(client, project, "edit-world",
                            world={**approved["draft"]["worldResult"], "setting": "別の図書館。"})
        else:
            result = action(client, project, "save-brief",
                            world={**approved["draft"]["worldInput"], "prompt": "別の物語。"},
                            characters=[value["input"] for value in approved["draft"]["characters"]],
                            relationshipInputs=approved["draft"]["relationshipInputs"])
        assert result["draft"]["step"] == expected
        assert not result["draft"]["approved"] and result["draft"]["approval"] is None
        assert result["draft"]["hasProduction"]
        assert result["jobs"] == approved["jobs"]
        assert detail(client, project)["draft"]["step"] == expected
        # Editing the next approval must not hide the chapter using the previous one.
        returned = action(client, project, "go-to", step="production")
        assert returned["draft"]["step"] == "production"
        assert not returned["draft"]["approved"] and returned["draft"]["approval"] is None
        assert returned["jobs"] == approved["jobs"]
        assert client.get(f"/api/artifacts/{snapshot['artifactId']}/content").content == frozen


def test_old_unapproved_draft_with_existing_production_retains_access(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _ = ready(client)
        approved = action(client, project, "approve")
        action(client, project, "edit-character", character_id="character-1",
               patch={"settings": "次の版に向けた新しい設定。"})
        identifier = project["project"]["id"]
        with coordinator.db.transaction() as connection:
            state = json.loads(connection.execute(
                "SELECT state FROM m2_draft WHERE project_id=?", (identifier,),
            ).fetchone()[0])
            state.pop("wizardStepVersion")
            connection.execute("UPDATE m2_draft SET state=? WHERE project_id=?",
                               (json.dumps(state), identifier))
        restored = detail(client, project)
        assert not restored["draft"]["approved"] and restored["draft"]["approval"] is None
        assert restored["draft"]["hasProduction"]
        assert restored["draft"]["step"] == "production"
        assert restored["jobs"] == approved["jobs"]
        returned = action(client, project, "go-to", step="character-review")
        assert returned["draft"]["step"] == "character-review"
        assert detail(client, project)["draft"]["step"] == "character-review"
