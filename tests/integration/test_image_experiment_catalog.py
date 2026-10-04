"""An experiment catalog reads all selected chapters without progressing production."""

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from services.coordinator.m3_service import M3Service
from services.coordinator.service import Coordinator
from tests.integration.test_m3 import approved, finish


def test_empty_project_catalog_does_not_start_production(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project = client.post("/api/projects", json={"title": "画像テスト"}).json()
        response = client.get(f"/api/image-experiments/projects/{project['id']}")
        assert response.status_code == 200, response.text
        assert response.json()["production"] is None
        assert response.json()["portraits"] == []


def test_catalog_is_readonly_and_contains_published_chapter_sources(tmp_path, monkeypatch):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        with coordinator.db.transaction() as connection:
            before = [tuple(row) for row in connection.execute("SELECT * FROM job ORDER BY id")]

        def forbidden(*args, **kwargs):
            raise AssertionError("A material browser must not change production")

        monkeypatch.setattr(coordinator, "_recover", forbidden)
        monkeypatch.setattr(M3Service, "advance", forbidden)
        monkeypatch.setattr(M3Service, "_settle_control", forbidden)
        response = client.get(f"/api/image-experiments/projects/{project}")
        assert response.status_code == 200, response.text
        value = response.json()
        published = [row for row in value["production"]["chapters"] if row["build"]]
        assert published
        assert value["portraits"]
        assert any(row["character_id"] == "support-1" for row in value["portraits"])
        for row in value["portraits"]:
            assert row["source"]["project_id"] == project
            assert len(row["source"]["sha256"]) == 64
            assert row["chapter_numbers"]
        with coordinator.db.transaction() as connection:
            after = [tuple(row) for row in connection.execute("SELECT * FROM job ORDER BY id")]
        assert before == after
