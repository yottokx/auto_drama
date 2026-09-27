"""Cast-wide M2 contracts, voice clones and compatibility with saved M2 work."""

import copy
import io
import json
from itertools import combinations
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from test_m2 import (
    CHARACTER,
    action,
    claim,
    complete,
    create,
    detail,
    output,
    ready,
    run_one,
    worker,
)

from packages.contracts.m2 import CharacterBrief
from packages.contracts.m3 import M3_KINDS
from services.coordinator.app import create_app
from services.coordinator.service import Coordinator


def ready_cast(client, count=3, *, inputs=None):
    project, worker_id = ready(client)
    action(
        client,
        project,
        "save-characters",
        characters=[
            project["draft"]["characters"][0]["input"],
            *[CharacterBrief(id=f"person-{index}").model_dump() for index in range(2, count + 1)],
        ],
        **({"relationshipInputs": inputs} if inputs is not None else {}),
    )
    action(client, project, "generate-characters")
    while job := claim(client, worker_id):
        response = complete(client, job, worker_id)
        assert response.status_code == 200, response.text
    return detail(client, project), worker_id


def test_main_cast_limit_applies_to_creation_and_saved_inputs(tmp_path):
    characters = [CharacterBrief(id=f"person-{index}").model_dump() for index in range(4)]
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/m2/projects", json={"world": {}, "characters": characters})
        assert response.status_code == 422
        project, _ = ready(client)
        before = detail(client, project)
        action(client, project, "save-characters", characters=characters, status=422)
        assert detail(client, project)["draft"] == before["draft"]


def test_legacy_four_person_cast_cannot_queue_unsupported_relationship_generation(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, _ = ready_cast(client)
        with coordinator.db.transaction() as connection:
            row = connection.execute(
                "SELECT state FROM m2_draft WHERE project_id=?", (project["project"]["id"],)
            ).fetchone()
            state = json.loads(row[0])
            fourth = copy.deepcopy(state["draft"]["characters"][0])
            fourth["id"] = fourth["input"]["id"] = fourth["result"]["id"] = "person-4"
            state["draft"]["characters"].append(fourth)
            state["draft"]["relationships"]["pendingChanges"] = True
            connection.execute(
                "UPDATE m2_draft SET state=? WHERE project_id=?",
                (json.dumps(state), project["project"]["id"]),
            )
        before = detail(client, project)
        rejected = action(client, project, "generate-relationships", status=409)
        assert "3人以下" in rejected["detail"]
        # These revisions also enqueue cast relationships automatically.
        for scope in ("all", "settings"):
            rejected = action(
                client,
                project,
                "revise-character",
                character_id="character-1",
                scope=scope,
                instruction="慎重な性格に",
                status=409,
            )
            assert "3人以下" in rejected["detail"]
        after = detail(client, project)
        assert after["draft"] == before["draft"]
        assert after["jobs"] == before["jobs"]


def test_generated_character_remains_viewable_during_next_character_job(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        saved = action(
            client,
            project,
            "save-characters",
            characters=[
                project["draft"]["characters"][0]["input"],
                CharacterBrief(id="person-2").model_dump(),
            ],
        )
        generated_before = saved["draft"]["characters"][0]["result"]
        current = action(client, project, "generate-characters")
        assert current["draft"]["characters"][0]["result"] == generated_before
        assert current["draft"]["characters"][1]["result"] is None
        navigated = action(client, project, "go-to", step="character-review")
        assert navigated["draft"]["revision"] == current["draft"]["revision"]
        run_one(client, worker_id, "m2_character")
        assert detail(client, project)["draft"]["characters"][0]["result"]


def test_full_mesh_generated_without_instructions_and_approval_tracks_immutable_relationships(
    tmp_path,
):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready_cast(client)
        draft = project["draft"]
        expected = set(combinations(sorted(value["id"] for value in draft["characters"]), 2))
        relationship = draft["relationships"]
        assert draft["relationshipInputs"] == []
        assert {
            tuple(value["characterIds"]) for value in relationship["result"]["pairs"]
        } == expected
        assert not relationship["pendingChanges"]
        jobs = [value for value in project["jobs"] if value["kind"] == "m2_relationships"]
        assert len(jobs) == 1
        assert len(jobs[0]["payload"]["cast_results"]) == 3
        approved = action(client, project, "approve")["draft"]["approval"]
        reference = approved["relationships"]
        assert reference["artifactId"] == relationship["artifactId"]
        original = client.get(f"/api/artifacts/{approved['artifactId']}/content").content
        changed = action(
            client,
            project,
            "save-relationships",
            relationshipInputs=[
                {"characterIds": ["person-2", "character-1"], "instruction": "かつて競い合った仲間"}
            ],
        )
        assert changed["draft"]["relationshipInputs"][0]["characterIds"] == [
            "character-1",
            "person-2",
        ]
        assert changed["draft"]["relationships"]["pendingChanges"]
        assert changed["draft"]["relationships"]["result"] == relationship["result"]
        action(client, project, "approve", status=409)
        action(client, project, "generate-relationships", instruction="互いの敬意を強める")
        job, _ = run_one(client, worker_id, "m2_relationships")
        assert job["payload"]["instruction"] == "互いの敬意を強める"
        current = detail(client, project)
        assert current["draft"]["relationships"]["version"] == 2
        assert client.get(f"/api/artifacts/{approved['artifactId']}/content").content == original
        # Keep the first chapter operation intact until it settles; editing the
        # cast remains independent while production uses its frozen approval.
        action(client, project, "approve", status=409)
        from tests.integration.test_m3 import complete as complete_chapter
        from tests.integration.test_m3 import finish, narrative
        from tests.integration.test_m3 import output as chapter_output

        chapter_worker = client.post(
            "/api/workers", json={"name": "chapter fixture", "capabilities": list(M3_KINDS)},
        ).json()["id"]
        chapter_job = claim(client, chapter_worker)
        chapter_result = narrative(chapter_job["payload"]["approval_snapshot"])
        chapter_result["outline"]["character_arcs"] = [
            {"character_id": character["id"], "change": "仲間を信頼する。"}
            for character in draft["characters"]
        ]
        response = complete_chapter(client, chapter_worker, chapter_job, chapter_output(chapter_job, chapter_result))
        assert response.status_code == 200, response.text
        finish(client, chapter_worker)
        assert action(client, project, "approve")["draft"]["approved"]


@pytest.mark.parametrize("bad", ["missing", "duplicate", "unknown", "reversed", "blank"])
def test_invalid_relationship_results_never_replace_valid_complete_mesh(tmp_path, bad):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready_cast(client)
        previous = project["draft"]["relationships"]
        action(client, project, "generate-relationships")
        job = claim(client, worker_id)
        result = copy.deepcopy(previous["result"])
        if bad == "missing":
            result["pairs"].pop()
        elif bad == "duplicate":
            result["pairs"][-1] = result["pairs"][0]
        elif bad == "unknown":
            result["pairs"][0]["characterIds"] = ["character-1", "unknown"]
        elif bad == "reversed":
            result["pairs"][0]["characterIds"].reverse()
        else:
            result["pairs"][0]["firstToSecond"] = "  "
        assert complete(client, job, worker_id, result=result).status_code == 422
        current = detail(client, project)["draft"]["relationships"]
        assert current["result"] == previous["result"]
        assert current["artifactId"] == previous["artifactId"]
        assert current["pendingChanges"]


def test_relation_inputs_are_atomic_and_membership_prunes_only_input_pairs(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready_cast(
            client,
            inputs=[
                {"characterIds": ["person-2", "character-1"], "instruction": "師弟"},
                {"characterIds": ["person-2", "person-3"], "instruction": "姉妹"},
            ],
        )
        before = project["draft"]
        for pairs in [
            [{"characterIds": ["person-2", "person-2"], "instruction": "self"}],
            [{"characterIds": ["character-1", "missing"], "instruction": "unknown"}],
            [
                {"characterIds": ["character-1", "person-2"], "instruction": "x"},
                {"characterIds": ["person-2", "character-1"], "instruction": "y"},
            ],
        ]:
            action(client, project, "save-relationships", relationshipInputs=pairs, status=422)
            assert detail(client, project)["draft"] == before
        saved = action(
            client,
            project,
            "save-characters",
            characters=[value["input"] for value in before["characters"][:2]],
        )
        assert saved["draft"]["relationshipInputs"] == [
            {"characterIds": ["character-1", "person-2"], "instruction": "師弟"}
        ]
        assert saved["draft"]["relationships"]["pendingChanges"]
        assert saved["draft"]["relationships"]["result"] == before["relationships"]["result"]


def test_settings_revision_refreshes_relations_but_voice_only_revision_does_not(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready_cast(client, count=2)
        original = project["draft"]["relationships"]
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="settings",
            instruction="几帳面に",
        )
        run_one(
            client, worker_id, "m2_character", result={**CHARACTER, "settings": "几帳面な記録官。"}
        )
        run_one(client, worker_id, "m2_relationships")
        refreshed = detail(client, project)["draft"]["relationships"]
        assert refreshed["version"] == original["version"] + 1
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="voice",
            instruction="ゆっくり",
        )
        run_one(
            client, worker_id, "m2_character", result={**CHARACTER, "voice": "ゆっくり話す声。"}
        )
        run_one(client, worker_id, "m2_voice")
        assert claim(client, worker_id) is None
        assert detail(client, project)["draft"]["relationships"] == refreshed
        edited = action(
            client,
            project,
            "edit-character",
            character_id="character-1",
            patch={"settings": "直接修正した背景。"},
        )
        assert edited["draft"]["relationships"]["pendingChanges"]
        assert claim(client, worker_id) is None
        action(client, project, "approve", status=409)


def test_intro_changes_rebuild_sample_voice_and_voice_lock_protects_spoken_text(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        original = project["draft"]["characters"][0]
        assert original["voiceReferenceText"] == CHARACTER["selfIntroduction"]
        edited = action(
            client,
            project,
            "edit-character",
            character_id="character-1",
            patch={"sampleLines": ["一つ目。", "二つ目。", "三つ目。"]},
        )
        assert edited["draft"]["characters"][0]["voiceArtifactId"] == original["voiceArtifactId"]
        action(client, project, "toggle-lock", character_id="character-1", scope="voice")
        action(
            client,
            project,
            "edit-character",
            character_id="character-1",
            patch={"selfIntroduction": "別の自己紹介です。"},
            status=409,
        )
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="settings",
            instruction="慎重に",
        )
        run_one(
            client,
            worker_id,
            "m2_character",
            result={**CHARACTER, "selfIntroduction": "変えてはいけない自己紹介。"},
        )
        assert claim(client, worker_id) is None
        protected = detail(client, project)["draft"]["characters"][0]
        assert protected["result"]["selfIntroduction"] == CHARACTER["selfIntroduction"]
        assert protected["voiceArtifactId"] == original["voiceArtifactId"]
        action(client, project, "toggle-lock", character_id="character-1", scope="voice")
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="settings",
            instruction="自己紹介も改める",
        )
        run_one(
            client,
            worker_id,
            "m2_character",
            result={**CHARACTER, "selfIntroduction": "改めまして、記録官のリオと申します。"},
        )
        invalidated = detail(client, project)["draft"]["characters"][0]
        assert invalidated["voiceArtifactId"] is None
        job, _ = run_one(client, worker_id, "m2_voice")
        assert (
            job["payload"]["character_result"]["selfIntroduction"]
            == "改めまして、記録官のリオと申します。"
        )
        assert (
            detail(client, project)["draft"]["characters"][0]["voiceReferenceText"]
            == "改めまして、記録官のリオと申します。"
        )


def test_voice_clone_keeps_reference_and_approval_and_persists_trial_history(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        approved = action(client, project, "approve")
        original = approved["draft"]["characters"][0]
        snapshot = approved["draft"]["approval"]
        started = action(
            client,
            project,
            "clone-voice",
            character_id="character-1",
            text="  この物語を、あなたにも届けましょう。  ",
        )
        assert started["draft"]["approved"]
        assert started["draft"]["approval"] == snapshot
        job = claim(client, worker_id)
        assert job["kind"] == "m2_voice_clone"
        assert job["payload"]["reference_voice"]["artifact_id"] == original["voiceArtifactId"]
        assert job["payload"]["reference_voice"]["text"] == original["voiceReferenceText"]
        assert job["payload"]["dialogue_text"] == "この物語を、あなたにも届けましょう。"
        data = output(job)
        assert complete(client, job, worker_id, data).status_code == 200
        assert complete(client, job, worker_id, data).status_code == 200
        current = detail(client, project)
        character = current["draft"]["characters"][0]
        assert current["draft"]["approval"] == snapshot
        assert current["draft"]["approved"]
        assert character["voiceArtifactId"] == original["voiceArtifactId"]
        assert character["resultArtifactId"] == original["resultArtifactId"]
        assert len(character["voiceTests"]) == 1
        trial = character["voiceTests"][0]
        assert client.get(trial["url"]).status_code == 200
        action(client, project, "retake", character_id="character-1", scope="voice-retake")
        run_one(client, worker_id, "m2_voice")
        changed = detail(client, project)["draft"]["characters"][0]
        assert changed["voiceArtifactId"] != trial["sourceVoiceArtifactId"]
        assert changed["voiceTests"] == [trial]
        project_id = project["project"]["id"]
    with TestClient(create_app(tmp_path)) as client:
        assert detail(client, project_id)["draft"]["characters"][0]["voiceTests"] == [trial]


def test_clone_failure_does_not_invalidate_approval_and_text_is_bounded(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        approved = action(client, project, "approve")["draft"]["approval"]
        for invalid in ("  ", "あ" * 1001):
            action(
                client, project, "clone-voice", character_id="character-1", text=invalid, status=422
            )
        action(client, project, "clone-voice", character_id="character-1", text="失敗を試す台詞。")
        for _ in range(3):
            job = claim(client, worker_id)
            assert (
                client.post(
                    f"/api/jobs/{job['id']}/fail",
                    json={"worker_id": worker_id, "lease_id": job["lease_id"], "error": "fixture"},
                ).status_code
                == 200
            )
        current = detail(client, project)["draft"]
        assert current["approved"] and current["approval"] == approved
        assert current["characters"][0]["voiceTests"] == []
        assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 200
        run_one(client, worker_id, "m2_voice_clone")
        assert detail(client, project)["draft"]["approved"]


def test_trial_cannot_discard_failed_cast_generation_queue_or_retry(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = ready_cast(client)
        action(client, project, "generate-characters")
        first, _ = run_one(client, worker_id, "m2_character")
        assert first["payload"]["character_id"] == "character-1"
        for _ in range(3):
            failed = claim(client, worker_id)
            assert failed["payload"]["character_id"] == "person-2"
            response = client.post(
                f"/api/jobs/{failed['id']}/fail",
                json={
                    "worker_id": worker_id,
                    "lease_id": failed["lease_id"],
                    "error": "fixture failure",
                },
            )
            assert response.status_code == 200
        before = detail(client, project)
        with coordinator.db.transaction() as connection:
            state_before = json.loads(
                connection.execute(
                    "SELECT state FROM m2_draft WHERE project_id=?",
                    (project["project"]["id"],),
                ).fetchone()[0]
            )
        assert [value["kind"] for value in state_before["queue"]] == [
            "m2_character",
            "m2_relationships",
        ]
        rejected = action(
            client,
            project,
            "clone-voice",
            character_id="character-1",
            text="完成済みの人物で試聴。",
            status=409,
        )
        assert "再試行" in rejected["detail"]
        assert detail(client, project)["draft"] == before["draft"]
        with coordinator.db.transaction() as connection:
            state_after = json.loads(
                connection.execute(
                    "SELECT state FROM m2_draft WHERE project_id=?",
                    (project["project"]["id"],),
                ).fetchone()[0]
            )
        assert state_after == state_before
        assert client.post(f"/api/jobs/{failed['id']}/retry").status_code == 200
        retried, _ = run_one(client, worker_id, "m2_character")
        assert retried["id"] == failed["id"]
        assert retried["payload"] == failed["payload"]
        third, _ = run_one(client, worker_id, "m2_character")
        assert third["payload"]["character_id"] == "person-3"
        run_one(client, worker_id, "m2_relationships")
        assert claim(client, worker_id) is None
        assert action(client, project, "approve")["draft"]["approved"]


def test_new_trial_can_supersede_a_failed_trial_without_changing_approval(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        approved = action(client, project, "approve")["draft"]["approval"]
        action(client, project, "clone-voice", character_id="character-1", text="失敗する台詞。")
        for _ in range(3):
            old = claim(client, worker_id)
            assert (
                client.post(
                    f"/api/jobs/{old['id']}/fail",
                    json={
                        "worker_id": worker_id,
                        "lease_id": old["lease_id"],
                        "error": "fixture failure",
                    },
                ).status_code
                == 200
            )
        changed = action(
            client, project, "clone-voice", character_id="character-1", text="新しい試聴の台詞。"
        )
        assert changed["draft"]["approved"]
        assert changed["draft"]["approval"] == approved
        new, _ = run_one(client, worker_id, "m2_voice_clone")
        assert new["id"] != old["id"]
        assert new["payload"]["dialogue_text"] == "新しい試聴の台詞。"
        assert detail(client, project)["draft"]["approval"] == approved


@pytest.mark.parametrize(
    "field,value",
    [
        ("selfIntroduction", ""),
        ("sampleLines", []),
        ("sampleLines", ["一つだけ。"]),
        ("sampleLines", ["一。", "二。", "  "]),
    ],
)
def test_new_character_output_requires_spoken_intro_and_three_real_lines(tmp_path, field, value):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        original = project["draft"]["characters"][0]
        action(client, project, "generate-characters")
        job = claim(client, worker_id)
        assert job["payload"]["character_contract_version"] == 2
        assert (
            complete(client, job, worker_id, result={**CHARACTER, field: value}).status_code == 422
        )
        current = detail(client, project)["draft"]["characters"][0]
        assert current["resultArtifactId"] == original["resultArtifactId"]


@pytest.mark.parametrize(
    "kind,metadata",
    [
        ("m2_voice", {"reference_text": "別の台詞。"}),
        ("m2_voice_clone", {"text": "別の台詞。"}),
        ("m2_voice", []),
        ("m2_voice_clone", "bad"),
        ("m2_voice_clone", {"text": "正しい台詞。"}),
    ],
)
def test_voice_metadata_must_match_requested_speech_before_adoption(tmp_path, kind, metadata):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        original = project["draft"]["characters"][0]
        if kind == "m2_voice":
            action(client, project, "retake", character_id="character-1", scope="voice-retake")
        else:
            action(client, project, "clone-voice", character_id="character-1", text="正しい台詞。")
        job = claim(client, worker_id)
        with ZipFile(io.BytesIO(output(job))) as archive:
            files = {name: archive.read(name) for name in archive.namelist()}
        envelope = json.loads(files["result.json"])
        envelope["provenance"]["voice"] = metadata
        files["result.json"] = json.dumps(envelope, ensure_ascii=False).encode()
        changed = io.BytesIO()
        with ZipFile(changed, "w") as archive:
            for name, data in files.items():
                archive.writestr(name, data)
        assert complete(client, job, worker_id, data=changed.getvalue()).status_code == 422
        current = detail(client, project)["draft"]["characters"][0]
        assert current["voiceArtifactId"] == original["voiceArtifactId"]
        assert current["voiceTests"] == []


def test_legacy_results_read_edit_lock_and_queued_generation_remain_compatible(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker_id = create(client), worker(client)
        action(client, project, "generate-world")
        run_one(client, worker_id, "m2_world")
        action(client, project, "confirm-world")
        with coordinator.db.transaction() as connection:
            row = connection.execute(
                "SELECT id,payload FROM job WHERE kind='m2_character'"
            ).fetchone()
            payload = json.loads(row["payload"])
            payload.pop("character_contract_version")
            connection.execute(
                "UPDATE job SET payload=? WHERE id=?", (json.dumps(payload), row["id"])
            )
        legacy = {
            key: value
            for key, value in CHARACTER.items()
            if key not in ("selfIntroduction", "sampleLines")
        }
        run_one(client, worker_id, "m2_character", result=legacy)
        run_one(client, worker_id, "m2_image")
        legacy_voice_job, _ = run_one(client, worker_id, "m2_voice")
        assert legacy_voice_job["payload"]["character_contract_version"] == 1
        approved = action(client, project, "approve")
        original_artifact = approved["draft"]["approval"]["artifactId"]
        original_bytes = client.get(f"/api/artifacts/{original_artifact}/content").content
        with coordinator.db.transaction() as connection:
            state = json.loads(connection.execute("SELECT state FROM m2_draft").fetchone()[0])
            state["draft"].pop("relationshipInputs")
            state["draft"].pop("relationships")
            state["draft"]["characters"][0].pop("voiceTests")
            connection.execute("UPDATE m2_draft SET state=?", (json.dumps(state),))
        current = detail(client, project)
        assert current["draft"]["characters"][0]["result"] == legacy
        assert current["draft"]["approved"]
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="appearance",
            instruction="青い衣装",
            status=409,
        )
        edited = action(
            client,
            project,
            "edit-character",
            character_id="character-1",
            patch={"settings": "旧結果の直接修正。"},
        )
        assert "selfIntroduction" not in edited["draft"]["characters"][0]["result"]
        action(client, project, "toggle-lock", character_id="character-1", scope="voice")
        action(client, project, "generate-characters", status=409)
        action(
            client,
            project,
            "clone-voice",
            character_id="character-1",
            text="旧サンプルからも作れます。",
        )
        run_one(client, worker_id, "m2_voice_clone")
        action(client, project, "toggle-lock", character_id="character-1", scope="voice")
        action(client, project, "generate-characters")
        run_one(client, worker_id, "m2_character")
        run_one(client, worker_id, "m2_voice")
        assert detail(client, project)["draft"]["characters"][0]["result"]["selfIntroduction"]
        assert client.get(f"/api/artifacts/{original_artifact}/content").content == original_bytes
