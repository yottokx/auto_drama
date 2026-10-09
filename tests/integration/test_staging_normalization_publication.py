"""Publish safe staging while keeping accepted sources and old builds immutable."""

from __future__ import annotations

import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from packages.contracts import Script
from packages.narrative.continuity import narrative_hash
from services.coordinator import m3_service
from services.coordinator.app import create_app
from tests.integration.test_m2 import claim
from tests.integration.test_m3 import approved, complete, finish, narrative, output, production


def repeated_waits(snapshot):
    value = narrative(snapshot)
    scene = value["scenes"][0]
    scene["directions"].extend({
        "id": f"{scene['id']}-pause-{index}",
        "utterance_id": scene["utterances"][1]["id"],
        "kind": "pause", "timing": "after", "duration_ms": 500,
        "character_id": None, "position": None,
    } for index in range(49))
    return value


def start_repeated_waits(client):
    project, worker = approved(client)
    job = claim(client, worker)
    value = repeated_waits(job["payload"]["approval_snapshot"])
    response = complete(client, worker, job, output(job, result=value))
    assert response.status_code == 200, response.text
    finish(client, worker)
    state = production(client, project)
    assert state["status"] == "published", state["error"]
    return project, state


def bundle_files(client, state):
    response = client.get(state["export_url"])
    assert response.status_code == 200, response.text
    with ZipFile(io.BytesIO(response.content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def waits(value):
    return [direction for direction in value["directions"]
            if direction["kind"] == "pause" and direction["utterance_id"] == "scene-1-u2"]


def database_state(client):
    with client.app.state.coordinator.db.transaction() as connection:
        return {table: [dict(row) for row in connection.execute(f"SELECT * FROM {table} ORDER BY id")]
                for table in ("job", "m3_requirement", "chapter_build", "artifact")}


def test_adoption_preserves_raw_staging_and_checkpoint_but_publication_removes_repeated_waits(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        _, state = start_repeated_waits(client)
        files = bundle_files(client, state)
        script = json.loads(files["script.json"])
        source = json.loads(files["narrative.json"])
        assert len(waits(source["scenes"][0])) == 49
        assert sum(row["duration_ms"] for row in waits(source["scenes"][0])) == 24500
        assert [row["duration_ms"] for row in waits(script)] == [500]
        assert script["utterances"][1]["display_text"] == source["scenes"][0]["utterances"][1]["display_text"]
        assert files["sources/scene-1.txt"].decode("utf-8") == source["scenes"][0]["raw_text"]
        scenario = files["data/scenario/first.ks"].decode("utf-8")
        assert scenario.count('[ad_pause time="500"]') == 1
        assert "staging-normalization.json" in files
        audit = json.loads(files["staging-normalization.json"])
        assert state["build"]["validation"]["staging_normalization"] == audit
        assert audit["schema_version"] == 1
        assert len(audit["reports"]) == 1
        report = audit["reports"][0]
        assert report["scene_id"] == "scene-1"
        assert report["removed_count"] == report["duplicate_count"] == 48
        assert len(report["changes"]) == 48
        assert len(waits({"directions": report["original_directions"]})) == 49
        with client.app.state.coordinator.db.transaction() as connection:
            record = connection.execute("SELECT * FROM artifact WHERE id=?",
                                        (state["narrative_artifact_id"],)).fetchone()
            adopted = client.app.state.coordinator.store.read(record)
            provenance = json.loads(record["provenance"])
        assert json.loads(adopted) == source
        assert provenance["script_checkpoint"]["narrative_hash"] == narrative_hash(source)
        assert client.get(state["player_url"] + "data/scenario/first.ks").content == files["data/scenario/first.ks"]


def test_rebuild_repairs_legacy_waits_without_regeneration_or_overwriting_old_build(tmp_path, monkeypatch):
    conversion = m3_service.narrative_to_script

    def legacy_conversion(source, snapshot, references, **kwargs):
        # Model a previously published export containing accepted duplicate waits.
        # Its narrative/checkpoint remain the real accepted source, and every
        # asset still passes the actual publication and bundle validators.
        kwargs.pop("normalization_reports", None)
        script = conversion(source, snapshot, references, **kwargs).model_dump(mode="json")
        raw_waits = [
            {key: getattr(direction, key) for key in
             ("id", "utterance_id", "kind", "timing", "duration_ms")}
            for scene in source.scenes for direction in scene.directions
            if direction.kind == "pause" and direction.utterance_id == "scene-1-u2"
        ]
        if raw_waits:
            first = next(index for index, row in enumerate(script["directions"])
                         if row["kind"] == "pause" and row["utterance_id"] == "scene-1-u2")
            script["directions"][first:first + 1] = raw_waits
        return Script.model_validate(script)

    with TestClient(create_app(tmp_path)) as client:
        with monkeypatch.context() as legacy:
            legacy.setattr(m3_service, "narrative_to_script", legacy_conversion)
            project, original = start_repeated_waits(client)
        before = database_state(client)
        original_files = bundle_files(client, original)
        old_bundle = client.get(original["export_url"]).content
        assert len(waits(json.loads(original_files["script.json"]))) == 49
        assert "staging-normalization.json" not in original_files

        response = client.post(f"/api/m3/projects/{project}/rebuild")
        assert response.status_code == 200, response.text
        current = production(client, project)
        after = database_state(client)
        files = bundle_files(client, current)
        assert current["build"]["id"] != original["build"]["id"]
        assert current["build"]["revision"] == original["build"]["revision"] + 1
        assert current["narrative_artifact_id"] == original["narrative_artifact_id"]
        assert files["narrative.json"] == original_files["narrative.json"]
        assert files["sources/scene-1.txt"] == original_files["sources/scene-1.txt"]
        assert [row["duration_ms"] for row in waits(json.loads(files["script.json"]))] == [500]
        assert "staging-normalization.json" in files
        assert after["job"] == before["job"]
        assert after["m3_requirement"] == before["m3_requirement"]
        assert all(row in after["chapter_build"] for row in before["chapter_build"])
        assert all(row in after["artifact"] for row in before["artifact"])
        assert len(after["chapter_build"]) == len(before["chapter_build"]) + 1
        assert client.get(original["export_url"]).content == old_bundle
        assert client.get(original["player_url"] + "data/scenario/first.ks").content == original_files["data/scenario/first.ks"]
        assert client.get(current["player_url"] + "data/scenario/first.ks").content == files["data/scenario/first.ks"]
        repeated = client.post(f"/api/m3/projects/{project}/rebuild")
        assert repeated.status_code == 200, repeated.text
        assert production(client, project)["build"]["id"] == current["build"]["id"]
        assert database_state(client) == after
