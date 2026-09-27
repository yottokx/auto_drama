"""First-chapter production uses API leases and real artifact validation, without GPUs."""

import copy
import io
import json
import sqlite3
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts import Script
from packages.contracts.m3 import EMOTION_TAGS, M3_KINDS
from packages.narrative.validation import parse_scene_text
from packages.tyrano_export import compile_bundle
from services.coordinator.app import create_app
from services.coordinator.m3_service import M3Service
from services.coordinator.service import Coordinator
from tests.integration.test_m2 import CHARACTER, action, claim, create, png, ready, wav


def narrative(snapshot):
    main_id = snapshot["characters"][0]["id"]
    support = {**CHARACTER, "id": "support-1", "name": "エマ"}
    scenes = []
    for number in (1, 2):
        identifier = f"scene-{number}"
        raw = (
            f"{main_id}: 記録を閉じないでください。まだ確かめたいことがあります。\n"
            "NARRATOR: エマは扉にかけた手を止め、記録の消えかけた文字を見つめた。\n"
            "support-1: 危険です。それでも読む理由を、私に教えてください。\n"
            f"{main_id}: この余白に、あなたの家族の言葉が残っています。\n"
            "NARRATOR: エマは鍵を握り直し、ためらいながら扉を開けた。\n"
            "support-1: 分かりました。一緒に確かめましょう。"
        )
        utterances = [
            value.model_dump()
            for value in parse_scene_text(raw, identifier, {main_id, "support-1"})
        ]
        utterances[0]["voice_emotion"] = "sad"
        scenes.append(
            {
                "id": identifier,
                "plan": {
                    "id": identifier,
                    "location_id": f"location-{number}",
                    "character_ids": [main_id, "support-1"],
                    "objectives": "鍵を借りる。",
                    "start_state": "エマが拒んでいる。",
                    "end_state": "一緒に調べる約束をする。",
                    "atmosphere": "緊張の中の信頼",
                    "required_events": [
                        {
                            "id": "persuasion",
                            "description": "反対を聞き、理由を示して判断を変える。",
                        }
                    ],
                },
                "raw_text": raw,
                "utterances": utterances,
                "directions": [
                    {
                        "id": f"{identifier}-enter-{index}",
                        "utterance_id": utterances[0]["id"],
                        "kind": "enter",
                        "timing": "before",
                        "character_id": cid,
                        "position": position,
                        "duration_ms": 0,
                    }
                    for index, (cid, position) in enumerate(
                        ((main_id, "left"), ("support-1", "right"))
                    )
                ],
                "review": {
                    "passed": True,
                    "issues": [],
                    "events": [
                        {
                            "event_id": "persuasion",
                            "dramatized": True,
                            "evidence_utterance_ids": [u["id"] for u in utterances],
                            "reason": "拒否・理由の提示・手の変化と判断を実際に描いている。",
                        }
                    ],
                },
            }
        )
    return {
        "schema_version": 1,
        "chapter_number": 1,
        "title": "開かれた記録",
        "outline": {
            "ending": "ふたりで家族の記録を守る。",
            "character_arcs": [{"character_id": main_id, "change": "他者に任せる。"}],
            "chapters": [
                {
                    "number": number,
                    "title": f"第{number}章",
                    "role": "対話",
                    "summary": "記録を守る。",
                }
                for number in range(1, snapshot["world"]["result"]["chapterCount"] + 1)
            ],
            "foreshadowing": [],
        },
        "supporting_characters": [support],
        "locations": [
            {
                "id": f"location-{n}",
                "name": "図書館",
                "description": "古い記録室。",
                "time_of_day": "夜",
                "atmosphere": "静謐",
                "image_prompt": "empty library, night",
            }
            for n in (1, 2)
        ],
        "scenes": scenes,
    }


def output(job, result=None, voice_patch=None):
    payload, kind = job["payload"], job["kind"]
    if result is None:
        result = narrative(payload["approval_snapshot"]) if kind == "m3_narrative" else {}
    script_policy = kind == "m3_narrative" and payload.get("workflow_policy") == "script_continuation_v1"
    if script_policy and "workflow_policy" not in result:
        from packages.narrative.continuity import narrative_hash

        result = copy.deepcopy(result)
        previous = payload.get("previous_narrative")
        for key in ("start_state", "end_state", "previous_state_hash", "predecessor_state_inferred",
                    "continuity_review"):
            result.pop(key, None)
        result.update(workflow_version=2, workflow_policy="script_continuation_v1",
                      chapter_number=payload["chapter_number"], storyline_id=payload["storyline_id"],
                      previous_narrative_artifact_id=payload.get("previous_narrative_artifact_id"),
                      previous_narrative_hash=narrative_hash(previous) if previous else None)
        if previous:
            result["outline"] = copy.deepcopy(previous["outline"])
            result["supporting_characters"] = copy.deepcopy(previous["supporting_characters"])
        for scene in result["scenes"]:
            scene["review"] = {"policy": "not_evaluated", "passed": False, "issues": [], "events": []}
    elif kind == "m3_narrative" and not script_policy and payload.get("m4") and "start_state" not in result:
        from packages.narrative import story_state_hash

        result = copy.deepcopy(result)
        number = payload["chapter_number"]
        previous = payload.get("previous_narrative")
        result["chapter_number"] = number
        if previous:
            result["outline"] = previous["outline"]
            result["supporting_characters"] = previous["supporting_characters"]
        cast = [value["id"] for value in payload["approval_snapshot"]["characters"]]
        cast += [value["id"] for value in result["supporting_characters"]]

        def state(chapter):
            return {"schema_version": 1, "chapter_number": chapter, "summary": "記録を確かめる。",
                    "facts": [], "characters": [{"character_id": cid, "location": "図書館",
                        "relationships": [], "possessions": [], "injuries": [], "promises": [],
                        "inner_emotion": "慎重", "knowledge": []} for cid in cast],
                    "foreshadowing": [{"outline_index": i,
                        "status": "resolved" if chapter >= clue["payoff_chapter"] else
                            "planted" if chapter >= clue["setup_chapter"] else "pending",
                        "changed_chapter": clue["payoff_chapter"] if chapter >= clue["payoff_chapter"] else
                            clue["setup_chapter"] if chapter >= clue["setup_chapter"] else 0,
                        "evidence_utterance_ids": [result["scenes"][0]["utterances"][0]["id"]]
                            if chapter >= clue["setup_chapter"] else []}
                        for i, clue in enumerate(result["outline"]["foreshadowing"])]}

        start = previous["end_state"] if previous and previous.get("end_state") else state(number - 1)
        result.update(start_state=start, end_state=state(number), storyline_id=payload["storyline_id"],
                      previous_narrative_artifact_id=payload.get("previous_narrative_artifact_id"),
                      previous_state_hash=story_state_hash(start) if number > 1 else None,
                      predecessor_state_inferred=bool(previous and not previous.get("end_state")),
                      continuity_review={"passed": True, "issues": [], "checked_character_ids": cast,
                          "checked_foreshadowing_indices": list(range(len(result["outline"]["foreshadowing"])))})
    provenance = {"producer": "fixture", "private_runtime_path": "C:/private/runtime"}
    if script_policy:
        from packages.narrative.continuity import narrative_hash
        from services.coordinator.m3_service import _checkpoint_digest

        checkpoint = {
            "schema_version": 1, "generator_protocol": payload["generator_protocol"],
            "chapter_number": payload["chapter_number"], "narrative_hash": narrative_hash(result),
            "storyline_id": payload["storyline_id"],
            "approval_sha256": _checkpoint_digest(payload["approval_snapshot"]),
            "generation_identity": _checkpoint_digest({"fixture": True}), "state": {"fixture": True},
        }
        checkpoint["sha256"] = _checkpoint_digest(checkpoint)
        provenance.update(generator_protocol=payload["generator_protocol"], script_checkpoint=checkpoint)
    if kind == "m3_voice":
        provenance["voice"] = {"reference_text": payload["reference_text"]}
    elif kind == "m3_voice_clone":
        reference = payload["reference_voice"]
        provenance["voice"] = {
            "text": EMOTION_TAGS[payload["voice_emotion"]] + payload["dialogue_text"],
            "spoken_text": payload["dialogue_text"],
            "voice_emotion": payload["voice_emotion"],
            "reference_artifact_id": reference["artifact_id"],
            "reference_sha256": reference["sha256"],
            "reference_text": reference["text"],
            **(voice_patch or {}),
        }
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr(
            "result.json",
            json.dumps(
                {
                    "schema_version": 1,
                    "kind": kind,
                    "result": result,
                    "provenance": provenance,
                    "trace": [],
                },
                ensure_ascii=False,
            ),
        )
        if kind in ("m3_image", "m3_background"):
            archive.writestr("image.png", png(123))
        elif kind in ("m3_voice", "m3_voice_clone"):
            archive.writestr("voice.wav", wav())
    return data.getvalue()


def complete(client, worker, job, data=None):
    return client.post(
        f"/api/m3/jobs/{job['id']}/complete",
        params={"worker_id": worker, "lease_id": job["lease_id"]},
        content=output(job) if data is None else data,
        headers={"Content-Type": "application/zip"},
    )


def production(client, project_id):
    response = client.get(f"/api/m3/projects/{project_id}")
    assert response.status_code == 200, response.text
    return response.json()["production"]


def approved(client):
    project, _ = ready(client)
    action(client, project, "approve")
    worker = client.post(
        "/api/workers", json={"name": "M3 fixture", "capabilities": list(M3_KINDS)}
    ).json()["id"]
    return project["project"]["id"], worker


def pin_legacy_production(client, project_id):
    """Model a production queued before the script engine became the default."""
    with client.app.state.coordinator.db.transaction() as connection:
        rows = connection.execute(
            "SELECT id,payload,settings_snapshot FROM job WHERE project_id=? AND kind='m3_narrative'",
            (project_id,),
        ).fetchall()
        for row in rows:
            payload = json.loads(row["payload"])
            for key in ("story_workflow_version", "workflow_policy", "generator_protocol", "script_options"):
                payload.pop(key, None)
            connection.execute("UPDATE job SET payload=? WHERE id=?",
                               (json.dumps(payload, ensure_ascii=False), row["id"]))


def finish(client, worker):
    jobs = []
    while job := claim(client, worker):
        response = complete(client, worker, job)
        assert response.status_code == 200, response.text
        jobs.append(job)
    return jobs


def test_approval_gates_start_and_automatically_starts_exactly_once(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project = create(client)["project"]["id"]
        assert production(client, project) is None
        assert client.post(f"/api/m3/projects/{project}/start").status_code == 409
        project, _ = approved(client)
        before = production(client, project)
        assert before["stage"] == "narrative" and len(before["jobs"]) == 1
        assert (
            before["build"] is None
            and before["player_url"] is None
            and before["export_url"] is None
        )
        for _ in range(2):
            assert (
                client.post(f"/api/m3/projects/{project}/start").json()["production"]["id"]
                == before["id"]
            )
        assert len(production(client, project)["jobs"]) == 1
        frozen = before["jobs"][0]["payload"]["approval_snapshot"]
        action(
            client,
            project,
            "edit-character",
            character_id="character-1",
            patch={"settings": "別の下書き。"},
        )
        assert production(client, project)["jobs"][0]["payload"]["approval_snapshot"] == frozen
        assert client.post(f"/api/m3/projects/{project}/start").status_code == 409


@pytest.mark.parametrize(
    "damage", ["text", "speaker", "span", "missing", "duplicate", "review", "reference"]
)
def test_invalid_narrative_never_enqueues_assets_or_publishes(tmp_path, damage):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        if damage == "review":
            pin_legacy_production(client, project)
        job = claim(client, worker)
        result = narrative(job["payload"]["approval_snapshot"])
        scene = result["scenes"][0]
        if damage == "text":
            scene["utterances"][0]["display_text"] += "改変"
        elif damage == "speaker":
            scene["utterances"][0]["speaker_id"] = "support-1"
        elif damage == "span":
            scene["utterances"][0]["source_start"] += 1
        elif damage == "missing":
            scene["utterances"].pop()
        elif damage == "duplicate":
            scene["utterances"].append(copy.deepcopy(scene["utterances"][-1]))
        elif damage == "review":
            scene["review"]["events"][0]["dramatized"] = False
        else:
            scene["plan"]["location_id"] = "unknown"
        assert complete(client, worker, job, output(job, result=result)).status_code == 422
        state = production(client, project)
        assert (
            len(state["jobs"]) == 1
            and state["narrative_artifact_id"] is None
            and state["build"] is None
        )


def test_complete_build_is_immutable_reproducible_and_survives_reinitialization(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        job = claim(client, worker)
        data = output(job)
        assert complete(client, worker, job, data).status_code == 200
        assert complete(client, worker, job, data).status_code == 200
        assert complete(client, worker, job, data + b"changed").status_code == 409
        before = production(client, project)
        assert before["build"] is None and before["stage"] == "assets"
        media = claim(client, worker)
        assert complete(client, worker, media).status_code == 200
        adopted_media = production(client, project)["completed_jobs"]
    with TestClient(create_app(coordinator=Coordinator(tmp_path))) as client:
        assert production(client, project)["completed_jobs"] == adopted_media
        jobs = finish(client, worker)
        assert "m3_voice_clone" in {job["kind"] for job in jobs}
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert state["completed_jobs"] == state["total_jobs"]
        bundle = client.get(state["export_url"]).content
        with ZipFile(io.BytesIO(bundle)) as archive:
            script = Script.model_validate_json(archive.read("script.json"))
            assert all(u.audio_asset_id for u in script.utterances if u.speaker_id)
            assert len(script.characters) == 2
            assert {asset.kind for asset in script.assets} == {"character", "background", "audio"}
            assert "sources/scene-1.txt" in archive.namelist()
            assert "narrative.json" in archive.namelist() and "approval.json" in archive.namelist()
            assert not any(name.startswith("tyrano/") for name in archive.namelist())
            assert b"private_runtime_path" not in bundle and b"C:/private/runtime" not in bundle
            manifest = json.loads(archive.read("manifest.json"))
            assert manifest["engine_included"] is False
            documents = {
                name: archive.read(name)
                for name in (
                    "approval.json",
                    "narrative.json",
                    "chapter-manifest.json",
                    "sources/scene-1.txt",
                    "sources/scene-2.txt",
                )
            }
        assets = {
            asset.id: client.get(f"/api/artifacts/{asset.artifact_id}/content").content
            for asset in script.assets
        }
        assert compile_bundle(script, assets, documents=documents) == bundle
        assert (
            client.post(f"/api/m3/projects/{project}/start").json()["production"]["build"]
            == state["build"]
        )
        with (
            coordinator.db.transaction() as connection,
            pytest.raises(sqlite3.IntegrityError, match="immutable"),
        ):
            connection.execute("UPDATE chapter_build SET status='published'")
    with TestClient(create_app(tmp_path)) as client:
        restored = production(client, project)
        assert restored["build"] == state["build"]
        assert client.get(restored["export_url"]).content == bundle


def test_missing_ready_asset_blocks_publication_without_regenerating_completed_media(tmp_path):
    coordinator = Coordinator(tmp_path)
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        job = claim(client, worker)
        assert complete(client, worker, job).status_code == 200
        with coordinator.db.transaction() as connection:
            row = dict(
                connection.execute(
                    "SELECT artifact.* FROM m3_requirement r JOIN artifact ON artifact.id=r.artifact_id WHERE r.kind='m3_image'"
                ).fetchone()
            )
        target = coordinator.store.root / row["storage_key"]
        original = target.read_bytes()
        target.unlink()
        # Finish only this chapter's media while the shared source is absent.
        # Later writers correctly refuse to adopt another chapter using a missing source.
        jobs = []
        later = None
        while job := claim(client, worker):
            if job["payload"]["chapter_number"] > 1:
                later = job
                break
            assert complete(client, worker, job).status_code == 200
            jobs.append(job)
        state = production(client, project)
        assert state["status"] == "failed" and state["build"] is None
        assert all(job["status"] == "completed" for job in state["chapters"][0]["jobs"])
        target.write_bytes(original)
        if later:
            assert complete(client, worker, later).status_code == 200
        finish(client, worker)
        restored = client.post(f"/api/m3/projects/{project}/start").json()["production"]
        assert restored["status"] == "published", restored["error"]
        assert len(restored["chapters"][0]["jobs"]) == len(jobs) + 1
        assert claim(client, worker) is None


def test_stale_lease_and_wrong_voice_reference_never_adopt(tmp_path):
    now = [1_000.0]
    coordinator = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=coordinator)) as client:
        project, worker = approved(client)
        stale = claim(client, worker)
        now[0] += 11
        replacement = claim(client, worker)
        assert replacement["id"] == stale["id"] and replacement["lease_id"] != stale["lease_id"]
        assert complete(client, worker, stale).status_code == 409
        assert complete(client, worker, replacement).status_code == 200
        while job := claim(client, worker):
            if job["kind"] == "m3_voice_clone":
                bad = output(job, voice_patch={"reference_sha256": "0" * 64})
                assert complete(client, worker, job, bad).status_code == 422
                assert production(client, project)["build"] is None
                break
            assert complete(client, worker, job).status_code == 200
        else:
            pytest.fail("dialogue clone job missing")


def test_restart_recovers_adopted_result_before_next_jobs_and_before_publication(
    tmp_path, monkeypatch
):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        job = claim(client, worker)
        with monkeypatch.context() as patch:
            patch.setattr(M3Service, "advance", lambda *args: None)
            assert complete(client, worker, job).status_code == 200
        assert claim(client, worker) is None  # Crash window: result adopted, next jobs not queued.
    with TestClient(create_app(tmp_path)) as client, monkeypatch.context() as patch:
        patch.setattr(M3Service, "_publish", lambda *args: None)
        finish(client, worker)
        state = production(client, project)
        assert state["completed_jobs"] == state["total_jobs"] and state["build"] is None
    with TestClient(create_app(tmp_path)) as client:
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert claim(client, worker) is None


def test_failed_media_can_retry_without_repeating_narrative_or_completed_assets(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        narrative_job = claim(client, worker)
        assert complete(client, worker, narrative_job).status_code == 200
        failed_id = None
        for _ in range(3):
            job = claim(client, worker)
            failed_id = failed_id or job["id"]
            assert job["id"] == failed_id
            response = client.post(
                f"/api/jobs/{failed_id}/fail",
                json={
                    "worker_id": worker,
                    "lease_id": job["lease_id"],
                    "error": "一時的なモデル失敗",
                },
            )
            assert response.status_code == 200
        assert production(client, project)["status"] == "failed"
        assert client.post(f"/api/jobs/{failed_id}/retry").status_code == 200
        finish(client, worker)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert sum(job["kind"] == "m3_narrative" for job in state["jobs"]) == 3
        assert client.get(f"/api/jobs/{narrative_job['id']}").json()["job"]["attempt_count"] == 1
        assert client.get(f"/api/jobs/{failed_id}").json()["job"]["attempt_count"] == 4
