"""M2 API integration tests use deterministic fixtures, never expensive model inference."""

import binascii
import copy
import io
import json
import struct
import wave
import zlib
from concurrent.futures import ThreadPoolExecutor
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m2 import CharacterBrief, WorldBrief
from services.coordinator.app import create_app
from services.coordinator.service import Coordinator

WORLD = {
    **WorldBrief().model_dump(),
    "title": "銀河の法典",
    "genre": "民俗学×宇宙法廷劇",
    "mood": "乾いたユーモア",
    "setting": "異星の図書館。記録を読み解き文化間の争いを解決する。",
}
CHARACTER = {
    **CharacterBrief(id="character-1").model_dump(),
    "name": "リオ",
    "age": "不明",
    "gender": "無性別",
    "role": "記録官",
    "settings": "異星の民話を収集する慎重な研究者。",
    "appearance": "青い布をまとう半透明の異星人。",
    "voice": "穏やかで低めの声。",
    "selfIntroduction": "私はリオ、民話を記録する者です。知らない文化の話を、ひとつずつ聞かせてください。",
    "sampleLines": [
        "記録には、まだ余白があります。",
        "その物語の続きを聞かせてください。",
        "少し考える時間をいただけますか。",
    ],
}


def png(color=90):
    def chunk(kind, data):
        return (
            struct.pack(">I", len(data))
            + kind
            + data
            + struct.pack(">I", binascii.crc32(kind + data))
        )

    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 1, 8, 6, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(bytes([0, color, 30, 40, 255, 0, 0, 0, 0])))
        + chunk(b"IEND", b"")
    )


def wav(tone=500):
    data = io.BytesIO()
    with wave.open(data, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16_000)
        audio.writeframes(struct.pack("<h", tone) * 160)
    return data.getvalue()


def output(job, *, result=None, media=None, extra=None):
    kind = job["kind"]
    if result is None:
        result = (
            WORLD
            if kind == "m2_world"
            else {**CHARACTER, "id": job["payload"]["character_id"]}
            if kind == "m2_character"
            else {}
        )
    provenance = {"model": "fixture", "seed": job["payload"]["seed"]}
    if kind == "m2_voice":
        provenance["voice"] = {
            "reference_text": job["payload"]["character_result"].get("selfIntroduction")
            or "旧サンプル台詞。"
        }
    elif kind == "m2_voice_clone":
        reference = job["payload"]["reference_voice"]
        provenance["voice"] = {
            "text": job["payload"]["dialogue_text"],
            "reference_artifact_id": reference["artifact_id"],
            "reference_sha256": reference["sha256"],
            "reference_text": reference["text"],
        }
    if kind == "m2_relationships" and not result:
        from itertools import combinations

        result = {
            "pairs": [
                {
                    "characterIds": list(pair),
                    "summary": "記録をめぐる仲間。",
                    "firstToSecond": "仕事ぶりを認める。",
                    "secondToFirst": "知識を頼りにする。",
                }
                for pair in combinations(
                    sorted(value["id"] for value in job["payload"]["cast_inputs"]), 2
                )
            ]
        }
    envelope = {
        "schema_version": 1,
        "kind": kind,
        "result": result,
        "provenance": provenance,
        "trace": [{"stage": "fixture", "selected": "test"}],
    }
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr("result.json", json.dumps(envelope, ensure_ascii=False))
        if kind == "m2_image":
            archive.writestr("image.png", media if media is not None else png())
        if kind in ("m2_voice", "m2_voice_clone"):
            archive.writestr("voice.wav", media if media is not None else wav())
        if extra:
            archive.writestr(extra, "not accepted")
    return data.getvalue()


def create(client, **world):
    response = client.post(
        "/api/m2/projects", json={"world": {**WorldBrief().model_dump(), **world}}
    )
    assert response.status_code == 201, response.text
    return response.json()


def detail(client, project):
    identifier = project if isinstance(project, str) else project["project"]["id"]
    return client.get(f"/api/m2/projects/{identifier}").json()


def action(client, project, name, *, status=200, **values):
    current = detail(client, project)
    response = client.post(
        f"/api/m2/projects/{current['project']['id']}/actions",
        json={
            "action": name,
            "expected_revision": current["draft"]["revision"],
            **values,
        },
    )
    assert response.status_code == status, response.text
    return response.json()


def worker(client):
    identifier = client.post(
        "/api/workers",
        json={
            "name": "fixture",
            "capabilities": [
                "m2_world",
                "m2_character",
                "m2_image",
                "m2_voice",
                "m2_relationships",
                "m2_voice_clone",
                "tts_download",
            ],
        },
    ).json()["id"]
    report_voice_inventory(client, identifier)
    return identifier


def report_voice_inventory(client, identifier, models=None):
    """Fixture runtimes advertise installed models without loading real weights."""
    from packages.contracts.tts_catalog import resolve_bundle

    selections = models or [("irodori-v4.1-small", "fp32")]
    response = client.post(f"/api/workers/{identifier}/tts-models", json={"models": [
        {"model_id": model, "precision": precision,
         "manifest_id": resolve_bundle(model, precision)["manifest_id"],
         "file_download_ready": True, "generation_ready": True,
         "generation_purposes": ["voice_design", "voice_clone"]}
        for model, precision in selections
    ]})
    assert response.status_code == 200, response.text


def claim(client, identifier):
    return client.post(f"/api/workers/{identifier}/claim").json()["job"]


def complete(client, job, worker_id, data=None, **values):
    return client.post(
        f"/api/m2/jobs/{job['id']}/complete",
        params={
            "worker_id": worker_id,
            "lease_id": job["lease_id"],
        },
        content=data if data is not None else output(job, **values),
        headers={"Content-Type": "application/zip"},
    )


def run_one(client, identifier, kind=None, **values):
    job = claim(client, identifier)
    assert job
    if kind:
        assert job["kind"] == kind
    response = complete(client, job, identifier, **values)
    assert response.status_code == 200, response.text
    return job, response.json()


def ready(client):
    project, identifier = create(client, prompt="異星の物語を自由に考える。"), worker(client)
    action(client, project, "generate-world")
    run_one(client, identifier, "m2_world")
    action(client, project, "confirm-world")
    for kind in ("m2_character", "m2_image", "m2_voice"):
        run_one(client, identifier, kind)
    return detail(client, project), identifier


def test_full_wizard_empty_optional_input_atomic_pipeline_and_immutable_approval(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        draft = project["draft"]
        assert draft["worldInput"]["title"] == ""
        assert draft["worldResult"]["title"] == WORLD["title"]
        assert draft["characters"][0]["input"]["name"] == ""
        assert draft["characters"][0]["result"]["name"] == CHARACTER["name"]
        assert not draft["worldPendingChanges"]
        assert not draft["characters"][0]["pendingChanges"]
        for job in project["jobs"]:
            assert job["status"] == "completed"
            assert job["payload"]["profile"]["provider"] == "local"
        bases = [job["payload"]["base_revision"] for job in project["jobs"]]
        assert len(bases) == len(set(bases))
        assert claim(client, worker_id) is None
        approved = action(client, project, "approve")
        snapshot = approved["draft"]["approval"]
        assert approved["project"]["status"] == "ready"
        assert len(approved["jobs"]) == 5
        assert approved["jobs"][-1]["kind"] == "m3_plan"
        assert approved["draft"]["step"] == "planning-review"
        assert not approved["draft"]["hasProduction"]
        assert snapshot["world"]["version"] == 1
        assert snapshot["characters"][0]["image"]["version"] == 1
        saved = client.get(f"/api/artifacts/{snapshot['artifactId']}/content").content
        changed = action(
            client,
            project,
            "edit-character",
            character_id="character-1",
            patch={"settings": "新しい設定。"},
        )
        assert not changed["draft"]["approved"]
        assert changed["draft"]["characters"][0]["resultVersion"] == 2
        assert client.get(f"/api/artifacts/{snapshot['artifactId']}/content").content == saved
        assert json.loads(saved)["characters"][0]["result"]["settings"] == CHARACTER["settings"]
        identifier = project["project"]["id"]
    with TestClient(create_app(tmp_path)) as client:
        restarted = detail(client, identifier)
        assert restarted["draft"]["characters"][0]["result"]["settings"] == "新しい設定。"
        assert client.get(f"/api/artifacts/{snapshot['artifactId']}/content").content == saved


def test_locks_server_enforced_for_whole_and_partial_revisions_and_asset_only_retake(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        original = project["draft"]["characters"][0]
        action(client, project, "toggle-lock", character_id="character-1", scope="appearance")
        action(
            client,
            project,
            "edit-character",
            character_id="character-1",
            patch={"appearance": "changed"},
            status=409,
        )
        action(
            client, project, "retake", character_id="character-1", scope="image-retake", status=409
        )
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="all",
            instruction="全体を大胆に変更",
        )
        run_one(
            client,
            worker_id,
            "m2_character",
            result={
                **CHARACTER,
                "settings": "大胆な研究者。",
                "appearance": "赤い鎧。",
                "voice": "明るい高めの声。",
            },
        )
        current = detail(client, project)["draft"]["characters"][0]
        assert current["result"]["appearance"] == original["result"]["appearance"]
        assert current["imageArtifactId"] == original["imageArtifactId"]
        assert current["voiceArtifactId"] is None
        run_one(client, worker_id, "m2_voice", media=wav(900))
        assert claim(client, worker_id) is None
        before = detail(client, project)["draft"]["characters"][0]
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="settings",
            instruction="礼儀正しい人物に",
        )
        run_one(
            client,
            worker_id,
            "m2_character",
            result={
                **CHARACTER,
                "settings": "礼儀正しい。",
                "appearance": "悪意ある変更",
                "voice": "悪意ある変更",
            },
        )
        after = detail(client, project)["draft"]["characters"][0]
        assert after["result"]["settings"] == "礼儀正しい。"
        assert after["result"]["voice"] == before["result"]["voice"]
        assert after["voiceArtifactId"] == before["voiceArtifactId"]
        assert claim(client, worker_id) is None
        action(
            client,
            project,
            "retake",
            character_id="character-1",
            scope="voice-retake",
            instruction="少しゆっくり",
        )
        run_one(client, worker_id, "m2_voice", media=wav(1200))
        retaken = detail(client, project)["draft"]["characters"][0]
        assert retaken["result"] == after["result"]
        assert retaken["resultArtifactId"] == after["resultArtifactId"]
        assert retaken["imageArtifactId"] == after["imageArtifactId"]
        assert retaken["voiceArtifactId"] != after["voiceArtifactId"]
        assert detail(client, project)["draft"]["requests"][0]["protectedScopes"] == ["appearance"]


def test_busy_conflict_gate_and_inputs_cannot_be_approved_before_regeneration(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project = create(client)
        action(client, project, "confirm-world", status=409)
        action(client, project, "generate-characters", status=409)
        action(client, project, "approve", status=409)
        current = action(client, project, "generate-world")
        assert current["draft"]["step"] == "world-review"
        action(client, project, "save-world", world=WORLD, status=409)
        navigated = action(client, project, "go-to", step="world-input")
        assert navigated["draft"]["revision"] == current["draft"]["revision"]
        assert navigated["draft"]["activeJobId"] == current["draft"]["activeJobId"]
        worker_id = worker(client)
        run_one(client, worker_id)
        action(client, project, "confirm-world")
        for kind in ("m2_character", "m2_image", "m2_voice"):
            run_one(client, worker_id, kind)
        old = detail(client, project)
        changed = action(client, project, "save-world", world={**WORLD, "prompt": "海の物語"})
        assert changed["draft"]["worldResult"] == WORLD
        assert not changed["draft"]["worldConfirmed"]
        assert changed["draft"]["worldPendingChanges"]
        action(client, project, "confirm-world", status=409)
        response = client.post(
            f"/api/m2/projects/{project['project']['id']}/actions",
            json={
                "expected_revision": old["draft"]["revision"],
                "action": "generate-world",
            },
        )
        assert response.status_code == 409


def test_restart_recover_stale_and_duplicate_completion(tmp_path):
    now = [1_800_000_000.0]
    coordinator = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=coordinator)) as client:
        project = create(client)
        action(client, project, "generate-world")
        old_worker = worker(client)
        old = claim(client, old_worker)
        data = output(old)
    now[0] += 11
    with TestClient(
        create_app(coordinator=Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0]))
    ) as client:
        new_worker = worker(client)
        current = claim(client, new_worker)
        assert current["payload"]["seed"] == old["payload"]["seed"]
        assert complete(client, old, old_worker, data).status_code == 409
        with ThreadPoolExecutor(max_workers=2) as pool:
            responses = list(
                pool.map(lambda _: complete(client, current, new_worker, data), range(2))
            )
        assert [response.status_code for response in responses] == [200, 200]
        assert responses[0].json()["artifact"]["id"] == responses[1].json()["artifact"]["id"]
        assert (
            complete(
                client, current, new_worker, output(current, result={**WORLD, "title": "別"})
            ).status_code
            == 409
        )
        current_draft = detail(client, project)["draft"]
        assert current_draft["worldVersion"] == 1
        assert current_draft["worldResult"] == WORLD


def test_validation_rechecks_lease_and_revision_after_io(tmp_path, monkeypatch):
    import services.coordinator.m2_service as module

    now = [1_800_000_000.0]
    service = Coordinator(tmp_path, lease_seconds=10, clock=lambda: now[0])
    with TestClient(create_app(coordinator=service)) as client:
        project, worker_id = create(client), worker(client)
        action(client, project, "generate-world")
        job = claim(client, worker_id)
        original = module.validate_bundle

        def slow(*args):
            value = original(*args)
            now[0] += 11
            return value

        monkeypatch.setattr(module, "validate_bundle", slow)
        assert complete(client, job, worker_id).status_code == 409
        assert detail(client, project)["draft"]["worldResult"] is None
        monkeypatch.setattr(module, "validate_bundle", original)
        current = claim(client, worker_id)
        with service.db.transaction() as connection:
            state = json.loads(connection.execute("SELECT state FROM m2_draft").fetchone()[0])
            state["draft"]["revision"] += 1
            connection.execute("UPDATE m2_draft SET state=?", (json.dumps(state),))
        assert complete(client, current, worker_id).status_code == 409
        assert detail(client, project)["draft"]["worldResult"] is None


def test_failed_retry_same_seed_and_new_intent_supersedes_failed_history(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = create(client), worker(client)
        action(client, project, "generate-world")
        for _ in range(3):
            job = claim(client, worker_id)
            assert (
                client.post(
                    f"/api/jobs/{job['id']}/fail",
                    json={
                        "worker_id": worker_id,
                        "lease_id": job["lease_id"],
                        "error": "fixture failure",
                    },
                ).status_code
                == 200
            )
        assert detail(client, project)["jobs"][0]["status"] == "failed"
        assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 200
        retry = claim(client, worker_id)
        assert retry["payload"] == job["payload"]
        client.post(
            f"/api/jobs/{job['id']}/fail",
            json={"worker_id": worker_id, "lease_id": retry["lease_id"], "error": "failed again"},
        )
        action(client, project, "generate-world", instruction="別の発想で")
        assert client.post(f"/api/jobs/{job['id']}/retry").status_code == 409
        new, _ = run_one(client, worker_id, "m2_world")
        assert new["id"] != job["id"]
        action(client, project, "confirm-world")
        for _ in range(3):
            run_one(client, worker_id)
        assert action(client, project, "approve")["draft"]["approved"]


@pytest.mark.parametrize("bad", ["missing-field", "empty-result", "extra-file", "path-traversal"])
def test_invalid_world_bundles_never_become_results(tmp_path, bad):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = create(client), worker(client)
        action(client, project, "generate-world")
        job = claim(client, worker_id)
        result = copy.deepcopy(WORLD)
        if bad == "missing-field":
            del result["setting"]
        if bad == "empty-result":
            result["setting"] = ""
        extra = (
            "script.exe"
            if bad == "extra-file"
            else "../result.json"
            if bad == "path-traversal"
            else None
        )
        assert (
            complete(client, job, worker_id, output(job, result=result, extra=extra)).status_code
            == 422
        )
        assert detail(client, project)["draft"]["worldResult"] is None


@pytest.mark.parametrize(
    "kind,media",
    [
        ("m2_image", png()[:-4]),
        ("m2_image", b"not PNG"),
        ("m2_voice", wav()[:-2]),
        ("m2_voice", b"not WAV"),
    ],
)
def test_invalid_media_cannot_replace_adopted_asset(tmp_path, kind, media):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        before = project["draft"]["characters"][0]
        scope = "image-retake" if kind == "m2_image" else "voice-retake"
        action(client, project, "retake", character_id="character-1", scope=scope)
        job = claim(client, worker_id)
        assert complete(client, job, worker_id, media=media).status_code == 422
        after = detail(client, project)["draft"]["characters"][0]
        assert after["imageArtifactId"] == before["imageArtifactId"]
        assert after["voiceArtifactId"] == before["voiceArtifactId"]


def test_save_characters_optional_input_and_fixed_changes_are_guarded(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        original = project["draft"]["characters"][0]
        briefs = [
            original["input"],
            CharacterBrief(id="character-2", freeform="人間以外の相棒").model_dump(),
        ]
        saved = action(client, project, "save-characters", characters=briefs)
        assert saved["draft"]["characters"][0]["resultArtifactId"] == original["resultArtifactId"]
        action(client, project, "approve", status=409)
        action(client, project, "generate-characters")
        while claimable := claim(client, worker_id):
            assert complete(client, claimable, worker_id).status_code == 200
        action(client, project, "toggle-lock", character_id="character-1", scope="settings")
        current = detail(client, project)
        locked_character = current["draft"]["characters"][0]
        artifact = client.get(
            f"/api/artifacts/{locked_character['resultArtifactId']}/content"
        ).json()
        assert artifact["result"] == locked_character["result"]
        briefs = [copy.deepcopy(value["input"]) for value in current["draft"]["characters"]]
        briefs[0]["name"] = "固定を回避する名前"
        action(client, project, "save-characters", characters=briefs, status=409)
        action(client, project, "save-characters", characters=briefs[1:], status=409)
        assert action(client, project, "approve")["draft"]["approved"]


def test_partial_edit_cannot_drop_unapplied_character_input(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker_id = ready(client)
        brief = {**project["draft"]["characters"][0]["input"], "freeform": "根本的に別の種族に変更"}
        action(client, project, "save-characters", characters=[brief])
        action(
            client,
            project,
            "edit-character",
            character_id="character-1",
            patch={"settings": "小さな変更"},
            status=409,
        )
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="voice",
            instruction="声だけ変更",
            status=409,
        )
        action(
            client, project, "retake", character_id="character-1", scope="voice-retake", status=409
        )
        action(client, project, "toggle-lock", character_id="character-1", scope="appearance")
        assert detail(client, project)["draft"]["characters"][0]["pendingChanges"]
        action(client, project, "approve", status=409)
        action(
            client,
            project,
            "revise-character",
            character_id="character-1",
            scope="all",
            instruction="保存した入力を反映",
        )
        run_one(client, worker_id, "m2_character")
        assert action(client, project, "approve")["draft"]["approved"]
