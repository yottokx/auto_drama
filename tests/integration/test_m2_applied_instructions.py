"""Only accepted explicit edits become requirements for later M2 stages."""

import json

import pytest
from fastapi.testclient import TestClient
from test_m2 import CHARACTER, action, claim, complete, create, detail, ready, run_one, worker
from test_m2_combined_brief import brief_values, generate_combined

from services.coordinator.app import create_app
from services.coordinator.service import Coordinator


def finish_queue(client, worker_id):
    jobs = []
    while job := claim(client, worker_id):
        jobs.append(job)
        response = complete(client, job, worker_id)
        assert response.status_code == 200, response.text
    return jobs


def ready_pair(client):
    project, worker_id = generate_combined(client)
    finish_queue(client, worker_id)
    return detail(client, project), worker_id


def test_accepted_world_revision_is_in_first_character_job_after_restart(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = create(client), worker(client)
        instruction = "この世界では魔法が使える設定に変更してください。"
        action(client, project, "generate-world", instruction=instruction)
        job, _ = run_one(client, worker_id, "m2_world")
        assert job["payload"]["applied_instructions"] == []
        project_id = project["project"]["id"]

    with TestClient(create_app(tmp_path)) as client:
        confirmed = action(client, project_id, "confirm-world")
        assert "appliedInstructions" not in confirmed["draft"]
        payload = confirmed["jobs"][-1]["payload"]
        assert payload["instruction"] == ""
        assert payload["applied_instructions"] == [{
            "kind": "m2_world", "character_id": None, "scope": "world",
            "instruction": instruction,
        }]


def test_accepted_character_revision_reaches_automatic_relationships_and_next_revision(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready_pair(client)
        instruction = "名前だけをミナに変更してください。"
        action(client, project, "revise-character", character_id="person-1", scope="all",
               instruction=instruction)
        run_one(client, worker_id, "m2_character",
                result={**CHARACTER, "id": "person-1", "name": "ミナ"})
        jobs = finish_queue(client, worker_id)
        relationship = next(job for job in jobs if job["kind"] == "m2_relationships")
        expected = [{"kind": "m2_character", "character_id": "person-1", "scope": "all",
                     "instruction": instruction}]
        assert relationship["payload"]["instruction"] == ""
        assert relationship["payload"]["applied_instructions"] == expected
        revised = action(client, project, "revise-character", character_id="person-1",
                         scope="all", instruction="年齢だけを30歳に変更してください。")
        assert revised["jobs"][-1]["payload"]["applied_instructions"] == expected


def test_accepted_relationship_revision_reaches_later_character_generation(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready_pair(client)
        instruction = "二人は幼なじみに変更してください。"
        action(client, project, "generate-relationships", instruction=instruction)
        run_one(client, worker_id, "m2_relationships")
        generated = action(client, project, "generate-characters")
        assert generated["jobs"][-1]["payload"]["applied_instructions"] == [{
            "kind": "m2_relationships", "character_id": None, "scope": "relationships",
            "instruction": instruction,
        }]


def test_failed_world_requests_never_become_authoritative(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = create(client), worker(client)
        action(client, project, "generate-world", instruction="採用されなかった指示")
        for _ in range(3):
            failed = claim(client, worker_id)
            assert failed["payload"]["applied_instructions"] == []
            response = client.post(f"/api/jobs/{failed['id']}/fail", json={
                "worker_id": worker_id, "lease_id": failed["lease_id"], "error": "fixture",
            })
            assert response.status_code == 200, response.text
        action(client, project, "generate-world")
        run_one(client, worker_id, "m2_world")
        confirmed = action(client, project, "confirm-world")
        assert confirmed["draft"]["requests"][0]["instruction"] == "採用されなかった指示"
        assert confirmed["jobs"][-1]["payload"]["applied_instructions"] == []


def test_direct_edits_record_only_changed_fields_and_ignore_locks_and_media(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        original = project["draft"]["characters"][0]["result"]
        action(client, project, "edit-character", character_id="character-1",
               patch={"name": "ミナ", "settings": original["settings"]})
        action(client, project, "toggle-lock", character_id="character-1", scope="appearance")
        action(client, project, "toggle-lock", character_id="character-1", scope="appearance")
        action(client, project, "retake", character_id="character-1", scope="image-retake",
               instruction="立ち絵の光を明るくする")
        run_one(client, worker_id, "m2_image")
        world = detail(client, project)["draft"]["worldResult"]
        action(client, project, "edit-world", world={**world, "title": "新しい作品名"})
        generated = action(client, project, "generate-world")
        assert generated["jobs"][-1]["payload"]["applied_instructions"] == [
            {"kind": "m2_character", "character_id": "character-1", "scope": "all",
             "changes": {"name": "ミナ"}},
            {"kind": "m2_world", "character_id": None, "scope": "world",
             "changes": {"title": "新しい作品名"}},
        ]


@pytest.mark.parametrize("save", ["save-brief", "save-world", "save-characters", "save-relationships"])
@pytest.mark.parametrize("changed", [False, True])
def test_only_changed_step1_input_resets_accepted_history(tmp_path, save, changed):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready_pair(client)
        edited = action(client, project, "edit-character", character_id="person-1",
                        patch={"name": "ミナ"})
        values = brief_values(edited)
        if changed:
            if save in ("save-brief", "save-world"):
                values["world"]["prompt"] = "新しく保存した指示"
            elif save == "save-characters":
                values["characters"][0]["freeform"] = "新しく保存した人物指示"
            else:
                values["relationshipInputs"][0]["instruction"] = "新しく保存した関係性指示"
        fields = {
            "save-brief": values,
            "save-world": {"world": values["world"]},
            "save-characters": {"characters": values["characters"]},
            "save-relationships": {"relationshipInputs": values["relationshipInputs"]},
        }
        action(client, project, save, **fields[save])
        generated = action(client, project, "generate-world")
        cleared = changed and save in ("save-brief", "save-characters")
        assert generated["jobs"][-1]["payload"]["applied_instructions"] == ([] if cleared else [
            {"kind": "m2_character", "character_id": "person-1", "scope": "all",
             "changes": {"name": "ミナ"}},
        ])


def test_legacy_state_without_applied_history_still_generates(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project = create(client)
        with coordinator.db.transaction() as connection:
            state = json.loads(connection.execute("SELECT state FROM m2_draft").fetchone()[0])
            state.pop("appliedInstructions", None)
            connection.execute("UPDATE m2_draft SET state=?", (json.dumps(state),))
        generated = action(client, project, "generate-world")
        assert generated["jobs"][-1]["payload"]["applied_instructions"] == []


@pytest.mark.parametrize("save", ["save-world", "save-characters", "save-relationships"])
def test_source_changes_preserve_unrelated_accepted_edits(tmp_path, save):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready_pair(client)
        action(client, project, "edit-world",
               world={**project["draft"]["worldResult"], "title": "新しい作品名"})
        action(client, project, "confirm-world")
        finish_queue(client, worker_id)
        action(client, project, "edit-character", character_id="person-1", patch={"name": "ミナ"})
        action(client, project, "generate-relationships", instruction="二人は幼なじみ")
        run_one(client, worker_id, "m2_relationships")
        values = brief_values(detail(client, project))
        if save == "save-world":
            action(client, project, save, world={**values["world"], "prompt": "新しい世界の指示"})
        elif save == "save-characters":
            values["characters"][1]["freeform"] = "二人目にだけ新しい指示"
            action(client, project, save, characters=values["characters"])
        else:
            values["relationshipInputs"][0]["instruction"] = "今回から仕事仲間"
            action(client, project, save, relationshipInputs=values["relationshipInputs"])
        generated = action(client, project, "generate-world")
        expected = [
            {"kind": "m2_world", "character_id": None, "scope": "world",
             "changes": {"title": "新しい作品名"}},
            {"kind": "m2_character", "character_id": "person-1", "scope": "all",
             "changes": {"name": "ミナ"}},
            {"kind": "m2_relationships", "character_id": None, "scope": "relationships",
             "instruction": "二人は幼なじみ"},
        ]
        if save == "save-world":
            expected.pop(0)
        elif save == "save-relationships":
            expected.pop()
        assert generated["jobs"][-1]["payload"]["applied_instructions"] == expected


def test_restoring_history_restores_only_accepted_edits_on_that_branch(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        identifier = project["project"]["id"]
        endpoint = f"/api/m2/projects/{identifier}/history"
        action(client, project, "edit-character", character_id="character-1", patch={"name": "ミナ"})
        first_history = client.get(endpoint).json()
        action(client, project, "edit-character", character_id="character-1", patch={"age": "30歳"})
        latest_history = client.get(endpoint).json()
        response = client.post(f"{endpoint}/{first_history['current_revision_id']}/restore", json={
            "expected_version": latest_history["version"],
            "expected_revision": detail(client, project)["draft"]["revision"],
        })
        assert response.status_code == 200, response.text
        generated = action(client, project, "generate-characters")
        assert generated["jobs"][-1]["payload"]["applied_instructions"] == [{
            "kind": "m2_character", "character_id": "character-1", "scope": "all",
            "changes": {"name": "ミナ"},
        }]
