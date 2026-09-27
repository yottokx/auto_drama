"""Player upgrades reuse adopted content and retain every published build."""

import io
import sqlite3
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.tyrano_export import compiler
from services.coordinator import database
from services.coordinator.app import create_app
from services.coordinator.m3_service import M3Service
from services.coordinator.service import Coordinator
from tests.integration.test_m3 import approved, finish, production


def revise_player(monkeypatch):
    original = compiler.player_files

    def revised(script_bytes):
        files = original(script_bytes)
        files["data/others/auto_drama_player.css"] += b"\n/* Revised rendering. */\n"
        return files

    monkeypatch.setattr(compiler, "player_files", revised)


def counts(coordinator):
    with coordinator.db.transaction() as connection:
        return {
            table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
            for table in ("job", "job_attempt", "artifact", "chapter_build")
        }


def test_rebuild_preserves_old_build_content_and_jobs_and_is_idempotent(tmp_path, monkeypatch):
    # A fixed clock verifies that latest selection cannot depend on timestamp ties.
    coordinator = Coordinator(tmp_path, clock=lambda: 1_000.0)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        finish(client, worker)
        old = production(client, project)
        old_bundle = client.get(old["export_url"]).content
        before = counts(coordinator)
        assert old["build"]["revision"] == 1
        assert client.post(f"/api/m3/projects/{project}/rebuild").json()["production"] == old
        assert counts(coordinator) == before

        revise_player(monkeypatch)
        response = client.post(f"/api/m3/projects/{project}/rebuild")
        assert response.status_code == 200, response.text
        new = response.json()["production"]
        assert new["build"]["revision"] == 2
        assert new["build"]["id"] != old["build"]["id"]
        assert new["jobs"] == old["jobs"]
        assert new["narrative_artifact_id"] == old["narrative_artifact_id"]
        assert M3Service(coordinator).build(old["build"]["id"]) == old["build"]
        assert client.get(old["export_url"]).content == old_bundle
        new_bundle = client.get(new["export_url"]).content
        assert new_bundle != old_bundle
        with ZipFile(io.BytesIO(old_bundle)) as left, ZipFile(io.BytesIO(new_bundle)) as right:
            assert left.namelist() == right.namelist()
            changed = {name for name in left.namelist() if left.read(name) != right.read(name)}
            assert changed == {"manifest.json", "data/others/auto_drama_player.css"}
            assert client.get(old["player_url"] + "data/others/auto_drama_player.css").content == (
                left.read("data/others/auto_drama_player.css")
            )
            assert client.get(new["player_url"] + "data/others/auto_drama_player.css").content == (
                right.read("data/others/auto_drama_player.css")
            )
        after = counts(coordinator)
        assert after == {
            **before,
            "chapter_build": before["chapter_build"] + 1,
            "artifact": before["artifact"] + 2,
        }
        assert client.post(f"/api/m3/projects/{project}/rebuild").json()["production"] == new
        assert counts(coordinator) == after
        with coordinator.db.transaction() as connection, pytest.raises(
            sqlite3.IntegrityError, match="immutable"
        ):
            connection.execute("UPDATE chapter_build SET status='published'")
    with TestClient(create_app(tmp_path)) as client:
        assert production(client, project)["build"] == new["build"]
        assert client.get(old["export_url"]).content == old_bundle


def test_failed_rebuild_keeps_existing_publication_and_rolls_back(tmp_path, monkeypatch):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        assert client.post(f"/api/m3/projects/{project}/rebuild").status_code == 409
        finish(client, worker)
        old = production(client, project)
        bundle = client.get(old["export_url"]).content
        before = counts(coordinator)
        revise_player(monkeypatch)
        original = M3Service._artifact

        def fail_after_script(self, connection, production, logical_id, kind, *args, **kwargs):
            if kind == "tyrano_export":
                raise ValueError("simulated failure after registering the new script")
            return original(self, connection, production, logical_id, kind, *args, **kwargs)

        monkeypatch.setattr(M3Service, "_artifact", fail_after_script)
        assert client.post(f"/api/m3/projects/{project}/rebuild").status_code == 422
        assert production(client, project) == old
        assert counts(coordinator) == before
        assert client.get(old["export_url"]).content == bundle


def test_migration_preserves_existing_published_build(tmp_path, monkeypatch):
    # Seed genuine adopted content, then represent it using the M3 database
    # schema. Current service code is not expected to run against an old schema.
    data_dir = tmp_path / "data"
    coordinator = Coordinator(data_dir)
    with monkeypatch.context() as patch:
        patch.setattr(M3Service, "_next_chapter", lambda *args: None)
        with TestClient(create_app(coordinator=coordinator)) as client:
            project, worker = approved(client)
            finish(client, worker)
            old = production(client, project)
            bundle = client.get(old["export_url"]).content
    original = next(database.MIGRATIONS.glob("003_*.sql")).read_text(encoding="utf-8")
    with coordinator.db.transaction(migration=True) as connection:
        connection.execute("DROP TABLE project_history_state")
        connection.execute("DROP TABLE project_revision")
        definitions = {
            "m3_production": original[:original.index("CREATE TABLE m3_requirement")],
            "chapter_build": original[original.index("CREATE TABLE chapter_build"):
                                      original.index("CREATE INDEX m3_production_project")],
        }
        for table, definition in definitions.items():
            connection.execute(definition.replace("CREATE TABLE " + table,
                                                  "CREATE TABLE " + table + "_legacy", 1))
            columns = ",".join(row[1] for row in connection.execute(f"PRAGMA table_info({table}_legacy)"))
            connection.execute(f"INSERT INTO {table}_legacy ({columns}) SELECT {columns} FROM {table}")
            connection.execute(f"DROP TABLE {table}")
            connection.execute(f"ALTER TABLE {table}_legacy RENAME TO {table}")
        connection.execute("DELETE FROM schema_migration WHERE version>=4")
    with TestClient(create_app(data_dir)) as client:
        migrated = production(client, project)
        assert migrated["build"] == old["build"]
        assert migrated["jobs"] == old["jobs"]
        assert migrated["control_state"] == "paused"  # Legacy chapters never auto-continue.
        assert client.get(migrated["export_url"]).content == bundle
        with client.app.state.coordinator.db.transaction() as connection:
            assert not connection.execute("PRAGMA foreign_key_check").fetchall()
        revise_player(monkeypatch)
        response = client.post(f"/api/m3/projects/{project}/rebuild")
        assert response.status_code == 200, response.text
        assert response.json()["production"]["build"]["revision"] == 2
