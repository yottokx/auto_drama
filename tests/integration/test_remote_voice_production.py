"""A chapter with a voice heard over the phone finds its speaker's face and frames them in the export."""

import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from packages.contracts import Script
from services.coordinator.app import create_app
from tests.integration.test_m2 import claim
from tests.integration.test_m3 import approved, complete, narrative, output, production

FACE = {"left": 10, "top": 6, "right": 30, "bottom": 28}


def with_a_call(job):
    """In the first scene Emma is only heard over the phone; in the second she is there."""
    result = narrative(job["payload"]["approval_snapshot"])
    first = result["scenes"][0]
    for line in first["utterances"]:
        if line["speaker_id"] == "support-1":
            line["remote"] = {"source": "remote", "channel": "phone"}
    first["directions"] = [row for row in first["directions"] if row["character_id"] != "support-1"]
    return output(job, result=result)


def face(job, found=FACE):
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr("result.json", json.dumps({
            "schema_version": 1, "kind": "m3_portrait_face", "provenance": {"model": "fixture"}, "trace": [],
            "result": {"portrait_sha256": job["payload"]["portrait"]["sha256"], "width": 64, "height": 64,
                       "face": found}}))
    return data.getvalue()


def test_a_remote_speakers_face_is_found_and_frames_them(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        kinds = []
        while job := claim(client, worker):
            data = (with_a_call(job) if job["kind"] == "m3_narrative"
                    else face(job) if job["kind"] == "m3_portrait_face" else None)
            response = complete(client, worker, job, data)
            assert response.status_code == 200, response.text
            kinds.append(job["kind"])
            if job["kind"] == "m3_portrait_face":
                # The face is looked for only in a portrait that exists.
                assert {"artifact_id", "sha256"} <= set(job["payload"]["portrait"])
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        # Only Emma is heard remotely, and she has one portrait: one face for each chapter.
        assert kinds.count("m3_portrait_face") == kinds.count("m3_narrative") >= 1
        bundle = client.get(state["export_url"]).content
        with ZipFile(io.BytesIO(bundle)) as archive:
            script = Script.model_validate_json(archive.read("script.json"))
            frames = json.loads(archive.read("data/others/auto_drama_remote.json"))["lines"]
            states = json.loads(archive.read("data/others/auto_drama_states.json"))["entries"]
    emma = next(value for value in script.characters if value.name == "エマ")
    assert emma.face.model_dump() == FACE
    heard = [line.id for line in script.utterances if line.remote is not None]
    assert heard and set(frames) == set(heard)
    assert all(frames[line]["speaker"] == emma.id and frames[line]["channel"] == "phone"
               and not frames[line]["on_stage"] for line in heard)
    # She is on stage in the second scene, where she is there in person, and never in the first.
    on_stage = {line: {value["id"] for value in entry["characters"]} for line, entry in states.items()}
    assert not any(emma.id in shown for line, shown in on_stage.items() if line.startswith("scene-1-"))
    assert emma.id in on_stage["scene-2-u2"]
