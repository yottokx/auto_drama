"""User-visible history restores coherent references without regenerating media."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from services.coordinator.service import Coordinator
from tests.integration.test_m2 import (
    action,
    claim,
    complete,
    create,
    detail,
    png,
    ready,
    run_one,
    wav,
)
from tests.integration.test_m3 import approved, finish, production
from tests.integration.test_m3 import complete as complete_m3
from tests.integration.test_m3_portrait_settings import settings
from tests.integration.test_m3_rebuild import counts


def project_id(project):
    return project if isinstance(project, str) else project["project"]["id"]


def history(client, project):
    response = client.get(f"/api/m2/projects/{project_id(project)}/history")
    assert response.status_code == 200, response.text
    value = response.json()
    assert value["project_id"] == project_id(project)
    assert isinstance(value["version"], int)
    current = [entry for entry in value["entries"] if entry["current"]]
    assert len(current) == 1
    assert current[0]["id"] == value["current_revision_id"]
    assert all(entry["label"] and entry["created_at"] for entry in value["entries"])
    return value


def current_entry(value):
    return next(entry for entry in value["entries"] if entry["current"])


def restore(client, project, revision_id, *, expected=None, draft_revision=None, status=200):
    before = history(client, project) if expected is None else expected
    draft_revision = (
        detail(client, project)["draft"]["revision"] if draft_revision is None else draft_revision
    )
    response = client.post(
        f"/api/m2/projects/{project_id(project)}/history/{revision_id}/restore",
        json={"expected_version": before["version"], "expected_revision": draft_revision},
    )
    assert response.status_code == status, response.text
    return response.json()


def test_restore_appends_history_preserves_future_and_does_not_generate(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _ = ready(client)
        original = copy.deepcopy(project["draft"])
        baseline = history(client, project)
        baseline_id = baseline["current_revision_id"]
        edited = action(
            client, project, "edit-character", character_id="character-1",
            patch={"name": "月代リオ", "settings": "人の話を最後まで聞く記録官。"},
        )
        after_edit = history(client, project)
        assert len(after_edit["entries"]) == len(baseline["entries"]) + 1
        edited_id = after_edit["current_revision_id"]
        action(client, project, "toggle-lock", character_id="character-1", scope="appearance")
        after_lock = history(client, project)
        assert len(after_lock["entries"]) == len(baseline["entries"]) + 2
        before_counts = counts(coordinator)
        before_revision = detail(client, project)["draft"]["revision"]

        restored = restore(client, project, baseline_id)

        assert restored["draft"]["revision"] > before_revision
        assert restored["draft"]["characters"] == original["characters"]
        assert restored["draft"]["worldResult"] == original["worldResult"]
        assert restored["draft"]["relationships"] == original["relationships"]
        assert restored["draft"]["activeJobId"] is None
        assert counts(coordinator) == before_counts
        reverted = history(client, project)
        assert len(reverted["entries"]) == len(after_lock["entries"]) + 1
        entry = current_entry(reverted)
        assert entry["restored_from_id"] == baseline_id
        assert entry["number"] > max(value["number"] for value in after_lock["entries"])
        assert {value["id"] for value in after_lock["entries"]} < {
            value["id"] for value in reverted["entries"]
        }
        # Forward history is still usable after undo; it is never discarded.
        restored_edit = restore(client, project, edited_id)
        assert restored_edit["draft"]["characters"] == edited["draft"]["characters"]
        assert restored_edit["draft"]["revision"] > restored["draft"]["revision"]
        assert counts(coordinator) == before_counts
        final_history = history(client, project)
        identifier = project_id(project)
    with TestClient(create_app(tmp_path)) as client:
        assert history(client, identifier) == final_history
        assert detail(client, identifier)["draft"] == restored_edit["draft"]


def test_navigation_noop_and_rejected_edit_do_not_add_history(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        before = history(client, project)
        action(client, project, "go-to", step="world-review")
        action(client, project, "go-to", step="character-review")
        action(
            client, project, "edit-character", character_id="character-1",
            patch={"name": project["draft"]["characters"][0]["result"]["name"]},
        )
        action(
            client, project, "edit-character", character_id="missing-person",
            patch={"name": "対象外"}, status=404,
        )
        after = history(client, project)
        assert after["entries"] == before["entries"]
        assert after["current_revision_id"] == before["current_revision_id"]
        assert after["pending_operation"] is None


def test_character_generation_chain_is_one_history_operation(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        before = history(client, project)
        original = copy.deepcopy(project["draft"]["characters"][0])
        action(
            client, project, "revise-character", character_id="character-1", scope="all",
            instruction="服と声を変える。",
        )
        pending = history(client, project)
        assert pending["busy"]
        assert pending["pending_operation"] is not None
        assert pending["entries"] == before["entries"]
        run_one(
            client, worker_id, "m2_character",
            result={**original["result"], "appearance": "赤いマント。", "voice": "明るい高い声。"},
        )
        assert history(client, project)["entries"] == before["entries"]
        run_one(client, worker_id, "m2_image", media=png(180))
        assert history(client, project)["entries"] == before["entries"]
        run_one(client, worker_id, "m2_voice", media=wav(900))
        after = history(client, project)
        assert not after["busy"]
        assert after["pending_operation"] is None
        assert len(after["entries"]) == len(before["entries"]) + 1
        changed = detail(client, project)["draft"]["characters"][0]
        for field in ("resultArtifactId", "imageArtifactId", "voiceArtifactId"):
            assert changed[field] != original[field]
        restored = restore(client, project, before["current_revision_id"])
        assert restored["draft"]["characters"][0] == original
        assert claim(client, worker_id) is None


@pytest.mark.parametrize("scope,kind,field,media", [
    ("image-retake", "m2_image", "imageArtifactId", png(190)),
    ("voice-retake", "m2_voice", "voiceArtifactId", wav(1100)),
])
def test_media_retake_can_restore_exact_old_asset_without_regeneration(
    tmp_path, scope, kind, field, media,
):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = ready(client)
        old = project["draft"]["characters"][0]
        old_bytes = client.get(f"/api/artifacts/{old[field]}/content").content
        before = history(client, project)
        action(client, project, "retake", character_id="character-1", scope=scope)
        run_one(client, worker_id, kind, media=media)
        changed = detail(client, project)["draft"]["characters"][0]
        assert changed[field] != old[field]
        assert len(history(client, project)["entries"]) == len(before["entries"]) + 1
        before_counts = counts(coordinator)

        restored = restore(client, project, before["current_revision_id"])

        assert restored["draft"]["characters"][0] == old
        assert counts(coordinator) == before_counts
        assert client.get(f"/api/artifacts/{old[field]}/content").content == old_bytes
        assert client.get(f"/api/artifacts/{changed[field]}/content").content == media
        assert claim(client, worker_id) is None


@pytest.mark.parametrize("stale_field", ["history", "draft"])
def test_restore_rejects_stale_clients_without_changing_state(tmp_path, stale_field):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _ = ready(client)
        old_history = history(client, project)
        old_revision = project["draft"]["revision"]
        action(client, project, "edit-character", character_id="character-1", patch={"name": "別名"})
        latest = history(client, project)
        before = detail(client, project)
        before_counts = counts(coordinator)
        restore(
            client, project, old_history["current_revision_id"],
            expected=old_history if stale_field == "history" else latest,
            draft_revision=old_revision if stale_field == "draft" else before["draft"]["revision"],
            status=409,
        )
        assert detail(client, project) == before
        assert history(client, project) == latest
        assert counts(coordinator) == before_counts


def test_restore_rejects_cross_project_revision(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _ = ready(client)
        other = create(client)
        other_history = history(client, other)
        before = detail(client, project)
        before_counts = counts(coordinator)
        own_history = history(client, project)
        response = client.post(
            f"/api/m2/projects/{project_id(project)}/history/"
            f"{other_history['current_revision_id']}/restore",
            json={"expected_version": own_history["version"],
                  "expected_revision": before["draft"]["revision"]},
        )
        assert response.status_code in (404, 422), response.text
        assert detail(client, project) == before
        assert history(client, project) == own_history
        assert counts(coordinator) == before_counts


def test_missing_old_asset_cannot_partially_restore_the_project(tmp_path, monkeypatch):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = ready(client)
        old_image = project["draft"]["characters"][0]["imageArtifactId"]
        baseline = history(client, project)
        action(client, project, "retake", character_id="character-1", scope="image-retake")
        run_one(client, worker_id, "m2_image", media=png(155))
        before = detail(client, project)
        before_history = history(client, project)
        before_counts = counts(coordinator)
        read = coordinator.store.read

        def missing_old_image(record):
            if record["id"] == old_image:
                raise FileNotFoundError("Old media is unavailable")
            return read(record)

        monkeypatch.setattr(coordinator.store, "read", missing_old_image)

        restore(client, project, baseline["current_revision_id"], status=422)

        assert detail(client, project) == before
        assert history(client, project) == before_history
        assert counts(coordinator) == before_counts


@pytest.mark.parametrize("stage", ["m2", "m3"])
def test_restore_rejects_pending_and_running_generation(tmp_path, stage):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = ready(client)
        before = history(client, project)
        if stage == "m2":
            action(client, project, "retake", character_id="character-1", scope="image-retake")
        else:
            action(client, project, "approve")
            worker_id = client.post(
                "/api/workers", json={"name": "history fixture", "capabilities": ["m3_narrative"]},
            ).json()["id"]
        for running in (False, True):
            if running:
                assert claim(client, worker_id) is not None
            current = detail(client, project)
            pending_history = history(client, project)
            assert pending_history["busy"]
            before_counts = counts(coordinator)
            restore(client, project, before["current_revision_id"], status=409)
            assert detail(client, project) == current
            assert history(client, project) == pending_history
            assert counts(coordinator) == before_counts


def test_failed_partial_chain_can_restore_and_old_job_cannot_resume(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = ready(client)
        original = copy.deepcopy(project["draft"]["characters"][0])
        before = history(client, project)
        action(
            client, project, "revise-character", character_id="character-1", scope="all",
            instruction="外見と声を変える。",
        )
        run_one(
            client, worker_id, "m2_character",
            result={**original["result"], "appearance": "青い鎧。", "voice": "高めの声。"},
        )
        for _ in range(3):
            failed = claim(client, worker_id)
            assert failed["kind"] == "m2_image"
            response = client.post(
                f"/api/jobs/{failed['id']}/fail",
                json={"worker_id": worker_id, "lease_id": failed["lease_id"], "error": "failed"},
            )
            assert response.status_code == 200, response.text
        assert not history(client, project)["busy"]
        before_counts = counts(coordinator)

        restored = restore(client, project, before["current_revision_id"])

        assert restored["draft"]["characters"][0] == original
        assert restored["draft"]["activeJobId"] is None
        assert counts(coordinator) == before_counts
        with coordinator.db.transaction() as connection:
            state = json.loads(connection.execute(
                "SELECT state FROM m2_draft WHERE project_id=?", (project_id(project),),
            ).fetchone()["state"])
            assert state["queue"] == []
            assert state["activeJobId"] is None
        assert client.post(f"/api/jobs/{failed['id']}/retry").status_code == 409
        assert complete(client, failed, worker_id).status_code == 409
        assert claim(client, worker_id) is None
        assert history(client, project)["pending_operation"] is None


def test_published_build_restore_selects_old_settings_and_new_edit_stays_monotonic(tmp_path):
    coordinator = Coordinator(tmp_path, clock=lambda: 1_000.0)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = approved(client)
        finish(client, worker_id)
        original = production(client, project)
        baseline = history(client, project)
        original_portraits = client.get(f"/api/m3/projects/{project}/portraits").json()
        original_zip = client.get(original["export_url"]).content
        request = settings(original["build"]["id"])
        response = client.put(f"/api/m3/projects/{project}/portraits", json=request)
        assert response.status_code == 200, response.text
        changed = response.json()["production"]
        changed_history = history(client, project)
        assert changed["build"]["revision"] == 2
        assert len(changed_history["entries"]) == len(baseline["entries"]) + 1
        before_counts = counts(coordinator)

        restore(client, project, baseline["current_revision_id"])

        assert production(client, project)["build"] == original["build"]
        assert client.get(f"/api/m3/projects/{project}/portraits").json() == original_portraits
        assert client.get(original["export_url"]).content == original_zip
        assert client.get(changed["export_url"]).status_code == 200
        assert counts(coordinator) == before_counts
        # Rebuild must reuse the restored settings, not the newest settings artifact.
        rebuilt = client.post(f"/api/m3/projects/{project}/rebuild")
        assert rebuilt.status_code == 200, rebuilt.text
        assert rebuilt.json()["production"]["build"] == original["build"]
        assert counts(coordinator) == before_counts
        request = settings(original["build"]["id"])
        request["characters"][0]["height_cm"] = 165
        response = client.put(f"/api/m3/projects/{project}/portraits", json=request)
        assert response.status_code == 200, response.text
        third = response.json()["production"]
        assert third["build"]["revision"] == 3
        assert third["build"]["manifest"] == original["build"]["manifest"]
        assert third["jobs"] == original["jobs"]
        assert third["narrative_artifact_id"] == original["narrative_artifact_id"]
        assert current_entry(history(client, project))["number"] > current_entry(
            changed_history,
        )["number"]


def test_restore_preproduction_revision_keeps_future_chapter_unselected(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _ = ready(client)
        baseline = history(client, project)
        identifier = project_id(project)
        action(client, project, "approve")
        worker_id = client.post(
            "/api/workers", json={"name": "history fixture", "capabilities": [
                "m3_narrative", "m3_background", "m3_image", "m3_voice", "m3_voice_clone",
                "tts_download",
            ]},
        ).json()["id"]
        from tests.integration.test_m2 import report_voice_inventory

        report_voice_inventory(client, worker_id)
        finish(client, worker_id)
        published = production(client, identifier)
        before_counts = counts(coordinator)

        restored = restore(client, project, baseline["current_revision_id"])

        assert not restored["draft"]["approved"]
        assert production(client, identifier) is None
        portraits = client.get(f"/api/m3/projects/{identifier}/portraits").json()
        assert portraits["production_id"] is None
        assert portraits["build_id"] is None
        assert portraits["characters"] == []
        plot = client.get(f"/api/m3/projects/{identifier}/plot").json()
        assert plot["production_id"] is None
        assert plot["narrative_artifact_id"] is None
        assert plot["plot"] is None
        assert client.get(published["export_url"]).status_code == 200
        assert counts(coordinator) == before_counts


def test_restored_project_rejects_retry_of_unselected_failed_production(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _ = ready(client)
        baseline = history(client, project)
        action(client, project, "approve")
        worker_id = client.post(
            "/api/workers", json={"name": "history fixture", "capabilities": ["m3_narrative"]},
        ).json()["id"]
        for _ in range(3):
            job = claim(client, worker_id)
            assert job["kind"] == "m3_narrative"
            response = client.post(
                f"/api/jobs/{job['id']}/fail",
                json={"worker_id": worker_id, "lease_id": job["lease_id"], "error": "failed"},
            )
            assert response.status_code == 200, response.text
        assert not history(client, project)["busy"]
        restore(client, project, baseline["current_revision_id"])
        before_counts = counts(coordinator)
        before = detail(client, project)
        assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 409
        assert production(client, project_id(project)) is None
        assert detail(client, project) == before
        assert counts(coordinator) == before_counts
        assert claim(client, worker_id) is None


@pytest.mark.parametrize("failure_stage", ["narrative", "assets"])
def test_failed_production_history_does_not_show_later_completed_content(tmp_path, failure_stage):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = approved(client)
        if failure_stage == "assets":
            narrative_job = claim(client, worker_id)
            response = complete_m3(client, worker_id, narrative_job)
            assert response.status_code == 200, response.text
        for _ in range(3):
            job = claim(client, worker_id)
            response = client.post(
                f"/api/jobs/{job['id']}/fail",
                json={"worker_id": worker_id, "lease_id": job["lease_id"], "error": "停止した工程"},
            )
            assert response.status_code == 200, response.text
        # Other queued media can finish independently after one asset fails.
        finish(client, worker_id)
        failed = production(client, project)
        assert failed["status"] == "failed"
        failed_history = history(client, project)
        assert not failed_history["busy"]
        failed_plot = client.get(f"/api/m3/projects/{project}/plot").json()
        assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 200
        finish(client, worker_id)
        published = production(client, project)
        assert published["status"] == "published"
        before_counts = counts(coordinator)

        restore(client, project, failed_history["current_revision_id"])

        restored = production(client, project)
        for field in (
            "id", "narrative_artifact_id", "build", "status", "stage", "error", "completed_jobs",
            "total_jobs", "player_url", "export_url",
        ):
            assert restored[field] == failed[field], field
        assert {value["id"]: value["status"] for value in restored["jobs"]} == {
            value["id"]: value["status"] for value in failed["jobs"]
        }
        assert client.get(f"/api/m3/projects/{project}/plot").json() == failed_plot
        assert client.get(published["export_url"]).status_code == 200
        assert counts(coordinator) == before_counts
        assert claim(client, worker_id) is None
