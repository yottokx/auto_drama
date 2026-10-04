"""Plot inspection projects immutable adopted inputs without advancing generation."""

import pytest
from fastapi.testclient import TestClient

from packages.contracts.planning import validate_plan_content
from services.coordinator.app import create_app
from services.coordinator.m3_service import M3Service
from services.coordinator.service import Coordinator, encode_json
from tests.integration.test_m2 import action, claim, create
from tests.integration.test_m3 import approved, complete, finish, narrative, output, production


def adopted(client):
    project, worker = approved(client)
    job = claim(client, worker)
    result = narrative(job["payload"]["approval_snapshot"])
    result["outline"]["foreshadowing"] = [
        {"setup_chapter": 1, "payoff_chapter": 3, "detail": "余白に残された文字。"},
    ]
    result["outline"]["character_arcs"].append({"character_id": "support-1", "change": "扉を開く。"})
    result["outline"] = validate_plan_content(
        job["payload"]["approved_plan"]["content"], job["payload"]["approval_snapshot"],
    ).plot.as_outline().model_dump(mode="json")
    response = complete(client, worker, job, output(job, result))
    assert response.status_code == 200, response.text
    return project, worker, result, job["payload"]["approval_snapshot"]


def db_state(coordinator):
    with coordinator.db.transaction() as connection:
        return tuple(connection.iterdump())


def test_plot_returns_empty_until_narrative_is_adopted_and_missing_project_is_404(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project = create(client)["project"]["id"]
        before = db_state(coordinator)
        response = client.get(f"/api/m3/projects/{project}/plot")
        assert response.status_code == 200
        assert response.json() == {
            "project_id": project, "production_id": None, "narrative_artifact_id": None, "plot": None,
        }
        assert db_state(coordinator) == before
        project, _ = approved(client)
        pending = production(client, project)
        response = client.get(f"/api/m3/projects/{project}/plot")
        assert response.status_code == 200
        assert response.json() == {
            "project_id": project, "production_id": pending["id"],
            "narrative_artifact_id": None, "plot": None,
        }
        assert client.get("/api/m3/projects/missing/plot").status_code == 404


@pytest.mark.parametrize("published", [False, True])
def test_adopted_plot_is_available_before_and_after_publication_without_side_effects(
    tmp_path, monkeypatch, published,
):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker, result, snapshot = adopted(client)
        if published:
            finish(client, worker)
        current = production(client, project)
        assert bool(current["build"]) is published
        assert current["narrative_artifact_id"]
        before = db_state(coordinator)

        def unexpected(*args):
            raise AssertionError("Plot inspection must not run recovery or schedule generation")

        monkeypatch.setattr(M3Service, "advance", unexpected)
        monkeypatch.setattr(coordinator, "_recover", unexpected)
        for _ in range(2):
            response = client.get(f"/api/m3/projects/{project}/plot")
            assert response.status_code == 200, response.text
            assert response.json() == {
                "project_id": project, "production_id": current["id"],
                "narrative_artifact_id": current["narrative_artifact_id"],
                "plot": {
                    "outline": result["outline"],
                    "characters": [
                        {"id": character["id"], "name": character["result"]["name"]}
                        for character in snapshot["characters"]
                    ] + [{"id": "support-1", "name": "エマ"}],
                },
            }
            assert "private_runtime_path" not in response.text
            assert "記録を閉じないでください" not in response.text
        assert db_state(coordinator) == before


def test_plot_character_names_follow_approved_inputs_after_draft_edit(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _, _, snapshot = adopted(client)
        before = client.get(f"/api/m3/projects/{project}/plot").json()
        changed = action(client, project, "edit-character", character_id="character-1",
                         patch={"name": "未承認の別名"})
        assert not changed["draft"]["approved"]
        response = client.get(f"/api/m3/projects/{project}/plot")
        assert response.status_code == 200
        assert response.json() == before
        assert response.json()["plot"]["characters"][0]["name"] == snapshot["characters"][0]["result"]["name"]
        assert "未承認の別名" not in response.text


def test_selected_production_has_no_plot_fallback_even_when_timestamps_tie(tmp_path):
    coordinator = Coordinator(tmp_path, clock=lambda: 1000.0)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker, _, _ = adopted(client)
        old = production(client, project)
        action(client, project, "edit-character", character_id="character-1",
               patch={"name": "次の承認の名前"})
        finish(client, worker)
        action(client, project, "approve")
        from tests.integration.planning_fixtures import complete_and_approve_plan

        complete_and_approve_plan(client, project)
        with coordinator.db.transaction() as connection:
            rows = list(connection.execute(
                "SELECT * FROM m3_production WHERE project_id=? AND chapter_number=1 ORDER BY created_at DESC,id DESC",
                (project,),
            ))
            assert len(rows) == 2 and rows[0]["created_at"] == rows[1]["created_at"]
            latest = next(dict(row) for row in rows if row["id"] != old["id"])
        response = client.get(f"/api/m3/projects/{project}/plot").json()
        assert response["production_id"] == latest["id"]
        assert response["narrative_artifact_id"] is None
        with coordinator.db.transaction() as connection:
            connection.execute("UPDATE m3_production SET created_at=2000 WHERE id=?", (old["id"],))
        response = client.get(f"/api/m3/projects/{project}/plot").json()
        assert response == {
            "project_id": project, "production_id": latest["id"],
            "narrative_artifact_id": None, "plot": None,
        }


@pytest.mark.parametrize("target", ["narrative_artifact_id", "approval_artifact_id"])
def test_plot_rejects_references_to_another_projects_artifacts(tmp_path, target):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        first, worker, _, _ = adopted(client)
        finish(client, worker)
        second, _, _, _ = adopted(client)
        first_production, second_production = production(client, first), production(client, second)
        with coordinator.db.transaction() as connection:
            connection.execute(f"UPDATE m3_production SET {target}=? WHERE id=?",
                               (second_production[target], first_production["id"]))
        before = db_state(coordinator)
        response = client.get(f"/api/m3/projects/{first}/plot")
        assert response.status_code == 422, response.text
        assert set(response.json()) == {"detail"}
        assert "plot" not in response.json()
        assert db_state(coordinator) == before


def test_plot_rejects_wrong_narrative_artifact_kind(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _, _, _ = adopted(client)
        current = production(client, project)
        with coordinator.db.transaction() as connection:
            connection.execute("UPDATE m3_production SET narrative_artifact_id=? WHERE id=?",
                               (current["approval_artifact_id"], current["id"]))
        response = client.get(f"/api/m3/projects/{project}/plot")
        assert response.status_code == 422
        assert set(response.json()) == {"detail"}


@pytest.mark.parametrize("target", ["narrative_artifact_id", "approval_artifact_id"])
def test_plot_rejects_corrupt_saved_bytes_without_queuing_replacement(tmp_path, target):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _, _, _ = adopted(client)
        current = production(client, project)
        with coordinator.db.transaction() as connection:
            record = dict(connection.execute("SELECT * FROM artifact WHERE id=?", (current[target],)).fetchone())
        before = db_state(coordinator)
        (coordinator.store.root / record["storage_key"]).write_bytes(b"corrupt")
        response = client.get(f"/api/m3/projects/{project}/plot")
        assert response.status_code == 503, response.text
        assert set(response.json()) == {"detail"}
        assert db_state(coordinator) == before


def test_malformed_adopted_plot_returns_generic_error_not_source_content(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _, _, _ = adopted(client)
        current = production(client, project)
        with coordinator.db.transaction() as connection:
            record = M3Service(coordinator)._artifact(
                connection, current, "narrative-" + current["id"], "m3_narrative",
                "narrative.json", encode_json({"private_path": "C:/private/invalid"}),
            )
            connection.execute("UPDATE m3_production SET narrative_artifact_id=? WHERE id=?",
                               (record["id"], current["id"]))
        response = client.get(f"/api/m3/projects/{project}/plot")
        assert response.status_code == 422
        assert set(response.json()) == {"detail"}
        assert "private" not in response.text
