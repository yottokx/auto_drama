"""Script generation keeps the existing production, media and publication path."""

import copy
import io
import json
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m3 import NarrativeResult
from packages.narrative.continuity import narrative_hash
from services.coordinator.app import create_app
from services.coordinator.service import Coordinator
from services.worker.generation import pipeline
from services.worker.generation.causal_runtime import digest
from services.worker.generation.script_production import ProductionScriptRun
from services.worker.generation.workflow_version import generator_protocol
from tests.integration.test_m2 import claim
from tests.integration.test_m3 import (
    approved,
    complete,
    finish,
    output,
    pin_legacy_production,
    production,
)


def envelope(data):
    with ZipFile(io.BytesIO(data)) as archive:
        return json.loads(archive.read("result.json"))


def replace_envelope(data, value):
    changed = io.BytesIO()
    with ZipFile(io.BytesIO(data)) as source, ZipFile(changed, "w") as target:
        for name in source.namelist():
            target.writestr(name, json.dumps(value, ensure_ascii=False) if name == "result.json"
                            else source.read(name))
    return changed.getvalue()


def publish_first_chapter(client, project, worker):
    """A new storyline waits for publication before queuing its next writer."""
    state = production(client, project)
    assert not state["chapters"][1]["jobs"]
    while not state["chapters"][1]["jobs"]:
        job = claim(client, worker)
        assert job is not None and job["kind"] != "m3_narrative"
        response = complete(client, worker, job)
        assert response.status_code == 200, response.text
        state = production(client, project)
    assert state["chapters"][0]["status"] == "published"
    return state


def test_single_main_api_approval_reaches_script_setting_without_changing_snapshot(tmp_path):
    with TestClient(create_app(tmp_path / "coordinator")) as client:
        project, worker = approved(client)
        job = claim(client, worker)
        snapshot = job["payload"]["approval_snapshot"]
        assert len(snapshot["characters"]) == 1
        assert "relationships" not in snapshot
        original = copy.deepcopy(snapshot)
        approval_id = job["settings_snapshot"]["approval_artifact_id"]
        before = client.get(f"/api/artifacts/{approval_id}/content").content

        run = ProductionScriptRun(tmp_path / "worker", pipeline.load_config(), job["payload"])
        setting = run.setting()
        assert setting["relationships"] == {"pairs": []}
        assert setting["main_character_ids"] == [snapshot["characters"][0]["id"]]
        assert setting["world"] == snapshot["world"]["result"]
        assert run.payload["approval_snapshot"] == original == snapshot
        assert digest(run.payload["approval_snapshot"]) == digest(original)
        assert client.get(f"/api/artifacts/{approval_id}/content").content == before
        assert len(production(client, project)["jobs"]) == 1


def test_script_chapters_pin_profile_checkpoint_and_publish_with_media_after_restart(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        first = claim(client, worker)
        payload = first["payload"]
        protocol = generator_protocol("causal", "script_continuation_v1")
        assert payload["workflow_policy"] == "script_continuation_v1"
        assert payload["story_workflow_version"] == 2 and payload["generator_protocol"] == protocol
        assert payload["profile"]["reasoning_level"] == "none"
        assert payload["profile"]["context_size"] == 16384
        assert "max_tokens" not in payload["profile"]
        data = output(first)
        saved = envelope(data)
        assert complete(client, worker, first, data).status_code == 200
        state = production(client, project)
        first_artifact = state["chapters"][0]["narrative_artifact_id"]
        first_bytes = client.get(f"/api/artifacts/{first_artifact}/content").content
        # Reading the approved plot does not start another generation.
        assert client.get(f"/api/m3/projects/{project}/plot").json()["plot"]
        assert production(client, project)["total_jobs"] == state["total_jobs"]
        state = publish_first_chapter(client, project, worker)
        second = state["chapters"][1]["jobs"][0]
        assert second["payload"]["script_checkpoint"] == saved["provenance"]["script_checkpoint"]
        assert second["payload"]["profile"] == payload["profile"]
        assert second["payload"]["seed"] == payload["seed"]
        assert second["payload"]["generator_protocol"] == protocol
        assert second["payload"]["previous_state_hash"] is None
        assert second["payload"]["previous_narrative_artifact_id"] == first_artifact

    with TestClient(create_app(tmp_path)) as client:
        finish(client, worker)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert [chapter["status"] for chapter in state["chapters"]] == ["published"] * 3
        assert client.get(f"/api/artifacts/{first_artifact}/content").content == first_bytes
        previous = None
        for chapter in state["chapters"]:
            value = client.get(f"/api/artifacts/{chapter['narrative_artifact_id']}/content").json()
            result = NarrativeResult.model_validate(value)
            assert result.workflow_policy == "script_continuation_v1"
            if previous:
                assert result.previous_narrative_hash == narrative_hash(previous)
            previous = result
            validation = chapter["build"]["validation"]
            assert validation["content_review"] is False
            assert validation["content_review_status"] == "not_evaluated"
            with ZipFile(io.BytesIO(client.get(chapter["export_url"]).content)) as archive:
                assert json.loads(archive.read("narrative.json"))["workflow_policy"] == "script_continuation_v1"
                script = json.loads(archive.read("script.json"))
                assert {asset["kind"] for asset in script["assets"]} == {"character", "background", "audio"}
                assert all(row["audio_asset_id"] for row in script["utterances"] if row["speaker_id"])
        writers = [job for job in state["jobs"] if job["kind"] == "m3_narrative"]
        assert len(writers) == 3
        assert all(job["payload"]["generator_protocol"] == protocol for job in writers)


@pytest.mark.parametrize("damage", ["policy", "version", "protocol", "checkpoint", "empty_checkpoint",
                                   "checkpoint_storyline", "checkpoint_chapter", "checkpoint_source",
                                   "checkpoint_approval", "checkpoint_state"])
def test_script_job_rejects_wrong_workflow_or_missing_checkpoint_before_assets(tmp_path, damage):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        job = claim(client, worker)
        data = output(job)
        value = envelope(data)
        if damage == "policy":
            legacy = copy.deepcopy(job)
            legacy["payload"].pop("workflow_policy")
            value["result"] = envelope(output(legacy))["result"]
        elif damage == "version":
            value["result"]["workflow_version"] = 1
        elif damage == "protocol":
            value["provenance"]["generator_protocol"]["prompt"] -= 1
        elif damage == "empty_checkpoint":
            value["provenance"]["script_checkpoint"] = {}
        elif damage.startswith("checkpoint_"):
            key = {"checkpoint_storyline": "storyline_id", "checkpoint_chapter": "chapter_number",
                   "checkpoint_source": "narrative_hash", "checkpoint_approval": "approval_sha256",
                   "checkpoint_state": "state"}[damage]
            checkpoint = value["provenance"]["script_checkpoint"]
            checkpoint[key] = {"changed": True} if key == "state" else "changed"
        else:
            value["provenance"].pop("script_checkpoint")
        assert complete(client, worker, job, replace_envelope(data, value)).status_code == 422
        state = production(client, project)
        assert state["narrative_artifact_id"] is None
        assert len(state["jobs"]) == 1 and state["build"] is None


@pytest.mark.parametrize("damage", ["artifact", "source_hash"])
def test_script_second_chapter_rejects_wrong_adopted_predecessor(tmp_path, damage):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        first = claim(client, worker)
        assert complete(client, worker, first).status_code == 200
        publish_first_chapter(client, project, worker)
        writer = client.post("/api/workers", json={
            "name": "script writer", "capabilities": ["m3_narrative"],
        }).json()["id"]
        second = claim(client, writer)
        data = output(second)
        value = envelope(data)
        key = "previous_narrative_artifact_id" if damage == "artifact" else "previous_narrative_hash"
        value["result"][key] = "wrong-artifact" if damage == "artifact" else "0" * 64
        assert complete(client, writer, second, replace_envelope(data, value)).status_code == 422
        state = production(client, project)
        assert state["chapters"][0]["narrative_artifact_id"]
        assert state["chapters"][1]["narrative_artifact_id"] is None


def test_script_workflow_options_and_seed_are_frozen_for_later_chapters(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        with client.app.state.coordinator.db.transaction() as connection:
            row = connection.execute(
                "SELECT id,payload FROM job WHERE project_id=? AND kind='m3_narrative'", (project,),
            ).fetchone()
            payload = json.loads(row["payload"])
            payload.update(profiles={"script_writer": {"temperature": 0.55}},
                           workflow_limits={"max_calls": 90},
                           script_options={"target_body_characters": 4500})
            connection.execute("UPDATE job SET payload=? WHERE id=?", (json.dumps(payload), row["id"]))
        first = claim(client, worker)
        assert complete(client, worker, first).status_code == 200
        state = publish_first_chapter(client, project, worker)
        continued = state["chapters"][1]["jobs"][0]["payload"]
        for key in ("profiles", "workflow_limits", "script_options", "seed"):
            assert continued[key] == first["payload"][key]


def test_saved_legacy_production_cannot_accept_script_result(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        pin_legacy_production(client, project)
        job = claim(client, worker)
        script_job = copy.deepcopy(job)
        script_job["payload"].update(workflow_policy="script_continuation_v1", story_workflow_version=2,
            generator_protocol=generator_protocol("causal", "script_continuation_v1"))
        assert complete(client, worker, job, output(script_job)).status_code == 422
        state = production(client, project)
        assert state["narrative_artifact_id"] is None
        assert len(state["jobs"]) == 1
