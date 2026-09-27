import io
import json
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts import Script
from packages.narrative.validation import script_character_id
from services.coordinator.app import create_app
from services.coordinator.m3_bundle import validate_bundle
from tests.integration.test_m2 import claim
from tests.integration.test_m3 import approved, complete, narrative, output, production

APPEARANCE = "姿は見えず、声としてのみ現れる。"


def omitted_output(job, *, quote=APPEARANCE):
    with ZipFile(io.BytesIO(output(job))) as archive:
        envelope = json.loads(archive.read("result.json"))
    envelope["result"] = {"portrait": {
        "status": "omitted", "character_id": job["payload"]["character_id"],
        "failed_stage": "background_removal", "failure": "Invalid background removal output: alpha=(0,234)",
        "diagnosis": {"action": "omit_portrait", "reason": "元設定に姿がないと明記されています。",
                      "source_field": "appearance", "source_quote": quote, "revised_prompt": ""},
    }}
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("result.json", json.dumps(envelope, ensure_ascii=False))
    return buffer.getvalue()


def test_omitted_portrait_publishes_with_dialogue_and_voice_and_survives_restart(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        story_job = claim(client, worker)
        story = narrative(story_job["payload"]["approval_snapshot"])
        story["supporting_characters"][0]["appearance"] = APPEARANCE
        response = complete(client, worker, story_job, output(story_job, story))
        assert response.status_code == 200, response.text
        image_seen = False
        while job := claim(client, worker):
            if job["kind"] == "m3_image":
                image_seen = True
                # A fabricated quotation is rejected without consuming the valid lease.
                assert complete(client, worker, job, omitted_output(job, quote="made up")).status_code == 422
                response = complete(client, worker, job, omitted_output(job))
            else:
                response = complete(client, worker, job)
            assert response.status_code == 200, response.text
        assert image_seen
        state = production(client, project)
        assert state["build"], state
        coordinator = client.app.state.coordinator
        with coordinator.db.transaction() as connection:
            record = dict(connection.execute("SELECT * FROM artifact WHERE id=?",
                          (state["build"]["script_artifact_id"],)).fetchone())
        script = Script.model_validate_json(coordinator.store.read(record))
        support_id = script_character_id("support-1")
        support = next(value for value in script.characters if value.id == support_id)
        assert support.image_asset_id is None
        assert support.name == "エマ"
        spoken = [value for value in script.utterances if value.speaker_id == support_id]
        assert spoken and all(value.audio_asset_id for value in spoken)
        assert not any(getattr(value, "character_id", None) == support_id for value in script.directions)
        assert client.get(state["player_url"]).status_code == 200
        assert client.get(state["export_url"]).status_code == 200
        assert all(value["character_id"] != "support-1" for value in
                   client.get(f"/api/m3/projects/{project}/portraits").json()["characters"])
    with TestClient(create_app(tmp_path)) as client:
        assert production(client, project)["build"]["id"] == state["build"]["id"]


def test_missing_image_without_explicit_omission_is_not_success():
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("result.json", json.dumps({"schema_version": 1, "kind": "m3_image",
                                                   "result": {}, "provenance": {}, "trace": []}))
    with pytest.raises(ValueError, match="explicit portrait omission"):
        validate_bundle(buffer.getvalue(), "m3_image")
