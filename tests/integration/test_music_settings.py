"""BGM settings: saved with a revision, and each work keeps the precision of its first track."""

import json

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from services.coordinator.music_settings import track_settings
from tests.integration.test_m2 import claim
from tests.integration.test_m3 import complete
from tests.integration.test_music_schedule import start_story
from tests.integration.test_scene_music import grouped_output


def save(client, revision, precision):
    return client.put("/api/music/settings", json={"expected_revision": revision,
                                                     "profile": {"precision": precision}})


def test_settings_offer_card_sizes_and_reject_stale_saves(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        settings = client.get("/api/music/settings").json()
        assert settings["profile"] == {"precision": "float32"} and settings["revision"] == 0
        assert [row["label"] for row in settings["configurations"]] == ["16GB以上", "12GB"]
        assert settings["ready"] is False and settings["workers"] == []
        client.post("/api/workers", json={"name": "Music", "capabilities": ["m3_music"]})
        client.post("/api/workers", json={"name": "Text", "capabilities": ["m3_narrative"]})
        settings = client.get("/api/music/settings").json()
        assert settings["ready"] is True
        assert [(row["name"], row["ready"]) for row in settings["workers"]] == [("Music", True), ("Text", False)]
        saved = save(client, 0, "float16")
        assert saved.status_code == 200 and saved.json()["revision"] == 1
        assert saved.json()["profile"] == {"precision": "float16"}
        assert save(client, 0, "float32").status_code == 409
        assert save(client, 1, "bfloat16").status_code == 422
        assert client.get("/api/music/settings").json()["profile"] == {"precision": "float16"}


def first_tracks(client, project, worker):
    """Complete the first chapter's text and music plan; return the music requirement descriptors."""
    narrative = claim(client, worker)
    assert complete(client, worker, narrative).status_code == 200
    plan = claim(client, worker)
    assert plan["kind"] == "m3_music_plan"
    assert complete(client, worker, plan, grouped_output(plan, {}, ("play", "continue"))).status_code == 200
    with client.app.state.coordinator.db.transaction() as connection:
        return [json.loads(row[0]) for row in connection.execute(
            "SELECT r.descriptor FROM m3_requirement r JOIN m3_production p ON p.id=r.production_id "
            "WHERE p.project_id=? AND r.kind='m3_music'", (project,))]


def test_a_work_keeps_the_precision_of_its_first_track(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = start_story(client)
        assert save(client, 0, "float16").status_code == 200
        descriptors = first_tracks(client, project, worker)
        assert descriptors and all(row["precision"] == "float16" for row in descriptors)
        # Changing the setting later never mixes precisions inside the work.
        assert save(client, 1, "float32").status_code == 200
        with client.app.state.coordinator.db.transaction() as connection:
            assert track_settings(connection, project)["precision"] == "float16"


def test_reference_precision_adds_nothing_and_older_tracks_stay_float32(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = start_story(client)
        descriptors = first_tracks(client, project, worker)
        # Requests made at float32 keep the exact fields they had before the setting existed.
        assert descriptors and all("precision" not in row for row in descriptors)
        assert all((row["backend"], row["model"]) == ("stable_audio3", "medium") for row in descriptors)
        assert save(client, 0, "float16").status_code == 200
        with client.app.state.coordinator.db.transaction() as connection:
            assert track_settings(connection, project) == {"backend": "stable_audio3", "model": "medium"}
