"""Real MP3 adoption, automatic publication, immutable music adjustments."""
import copy
import io
import json
import shutil
import subprocess
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient

from packages.contracts.m3 import M3_KINDS
from services.coordinator.app import create_app
from tests.integration import test_adjustments
from tests.integration.planning_fixtures import complete_and_approve_plan
from tests.integration.test_adjustments import apply, start
from tests.integration.test_m2 import action, claim, ready, report_voice_inventory
from tests.integration.test_m3 import complete, production
from tests.integration.test_project_history import history, restore

story = test_adjustments.story  # Register the reusable pytest fixture in this module.


@pytest.fixture(scope="module")
def mp3s(tmp_path_factory):
    directory = tmp_path_factory.mktemp("music-fixture")
    ffmpeg = shutil.which("ffmpeg") or "C:/ffmpeg/bin/ffmpeg.exe"
    values = {}
    for name, duration in (("source.mp3", 120), ("music.mp3", 70)):
        path = directory / name
        subprocess.run([ffmpeg, "-nostdin", "-v", "error", "-f", "lavfi", "-i", f"sine=frequency=440:duration={duration}",
                        "-ac", "2", "-ar", "44100", "-codec:a", "libmp3lame", "-b:a", "192k", str(path)],
                       check=True, capture_output=True)
        values[name] = path.read_bytes()
    return values


def music_output(job, files, *, patch=None):
    payload = job["payload"]
    if job["kind"] == "m3_music_plan":
        result = {"scenes": [{"scene_id": row["id"], "interpretation": "場面の緊張感", "prompt":
            "Genre: Mystery soundtrack. Instruments: bass clarinet, marimba and muted strings. "
            "Mood: curious and alert. BPM: 90. TrackType: Music, VocalType: Instrumental."}
            for row in payload["context"]["narrative"]["scenes"]]}
        if payload.get("planning_version") == 2:
            result["planning_version"] = 2
            for row in result["scenes"]:
                row.update(transition={"visual": "fade", "duration_ms": 500, "music_fade_out_ms": 1000,
                                       "music_fade_in_ms": 1000}, reason="場面の役割が変わるため、新しい曲を使います。")
        if payload.get("planning_version") == 2:
            result["planning_version"] = 2
            for row in result["scenes"]:
                row.update(transition={"visual": "fade", "duration_ms": 500, "music_fade_out_ms": 1000,
                                       "music_fade_in_ms": 1000}, reason="場面の役割が変わるため、新しい曲を使います。")
        assets = {}
    else:
        result = {"scene_id": payload["scene_id"], "prompt": payload["prompt"], "loop_start_seconds": 10.0,
                  "loop_end_seconds": 70.0, "duration_seconds": 70.0, "source_duration_seconds": 120.0,
                  "sample_rate": 44100, "quality": {"needs_review": False, "near_silence_seconds": 0}}
        assets = files
    result.update(patch or {})
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("result.json", json.dumps({"schema_version": 1, "kind": job["kind"], "result": result,
                         "provenance": {"backend": "stable_audio3", "seed": payload["seed"]}, "trace": []}))
        for name, data in assets.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def grouped_output(job, files, actions):
    with ZipFile(io.BytesIO(music_output(job, files))) as archive:
        envelope = json.loads(archive.read("result.json"))
    for row, selected in zip(envelope["result"]["scenes"], actions, strict=True):
        row["action"] = selected
        row["reason"] = "一続きの会話なので同じ曲を維持します。" if selected == "continue" else "場面の役割に合わせて音楽を切り替えます。"
        if selected != "play":
            row.pop("prompt", None)
        if selected == "continue":
            row["transition"].update(visual="dissolve", music_fade_out_ms=0, music_fade_in_ms=0)
    return music_output(job, files, patch=envelope["result"])


def finish_music(client, worker, files, *, actions=None):
    jobs = []
    for _ in range(150):
        job = claim(client, worker)
        if not job:
            return jobs
        data = (grouped_output(job, files, actions) if actions and job["kind"] == "m3_music_plan"
                else music_output(job, files) if job["kind"].startswith("m3_music") else None)
        response = complete(client, worker, job, data)
        assert response.status_code == 200, response.text
        jobs.append(job)
    raise AssertionError("Production did not settle.")


@pytest.fixture
def musical_story(tmp_path, mp3s):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        action(client, project, "approve")
        complete_and_approve_plan(client, project, music_enabled=True)
        worker = client.post("/api/workers", json={"name": "Music fixture", "capabilities": [*M3_KINDS, "tts_download"]}).json()["id"]
        report_voice_inventory(client, worker)
        project_id = project["project"]["id"]
        jobs = finish_music(client, worker, mp3s)
        yield client, project_id, worker, jobs, f"/api/m3/projects/{project_id}/adjustments"


@pytest.fixture
def grouped_story(tmp_path, mp3s):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        action(client, project, "approve")
        complete_and_approve_plan(client, project, music_enabled=True)
        worker = client.post("/api/workers", json={"name": "Continuity fixture", "capabilities": [*M3_KINDS, "tts_download"]}).json()["id"]
        report_voice_inventory(client, worker)
        project_id = project["project"]["id"]
        jobs = finish_music(client, worker, mp3s, actions=("play", "continue"))
        yield client, project_id, worker, jobs, f"/api/m3/projects/{project_id}/adjustments"


def script_for(client, chapter):
    with ZipFile(io.BytesIO(client.get(chapter["export_url"]).content)) as archive:
        return json.loads(archive.read("script.json")), archive.namelist()


def test_normal_production_has_music_before_any_adjustment(musical_story):
    client, project, _worker, jobs, endpoint = musical_story
    value = production(client, project)
    assert all(row["status"] == "published" for row in value["chapters"])
    plan_jobs = [job for job in jobs if job["kind"] == "m3_music_plan"]
    assert len(plan_jobs) == len(value["chapters"])
    assert all(job["payload"]["context"]["overall_plot"]["ending"] for job in plan_jobs)
    for chapter in value["chapters"]:
        script, names = script_for(client, chapter)
        assert len(script["music_cues"]) == 2
        assert all(cue["action"] == "play" and cue["loop_start_seconds"] == 10 for cue in script["music_cues"])
        assert len([name for name in names if name.startswith("data/bgm/") and name.endswith(".mp3")]) == 2
        assert all(".wav" not in name for name in names if name.startswith("data/bgm/"))
    view = client.get(endpoint).json()
    assert view["draft"] is None and len(view["music_candidates"]) == 2 * len(value["chapters"])


def test_contiguous_group_generates_only_one_track_and_persists_decisions(grouped_story):
    client, project, _worker, jobs, endpoint = grouped_story
    chapters = production(client, project)["chapters"]
    assert len([job for job in jobs if job["kind"] == "m3_music"]) == len(chapters)
    for chapter in chapters:
        script, names = script_for(client, chapter)
        assert [row["action"] for row in script["music_cues"]] == ["play", "continue"]
        assert len([name for name in names if name.startswith("data/bgm/") and name.endswith(".mp3")]) == 1
        assert len(script["scene_transitions"]) == 2
        assert script["scene_transitions"][1]["music_fade_out_ms"] == 0
        assert [row["music_plan"]["action"] for row in chapter["scenes"]] == ["play", "continue"]
    value = start(client, endpoint)
    settings = value["draft"]["scene_music"]
    assert [row["action"] for row in settings[:2]] == ["play", "continue"]
    assert settings[1]["candidate_id"] is None
    assert value["scenes"][1]["music_plan"]["source_scene_id"] == settings[0]["scene_id"]


def test_replan_reuses_inherited_audio_with_target_scope_and_preserves_narrative(grouped_story, mp3s, monkeypatch):
    from services.coordinator.adjustment_service import _AdjustmentPublisher

    client, project, worker, _jobs, endpoint = grouped_story
    before_chapters = production(client, project)["chapters"]
    original_bytes = client.get(before_chapters[0]["export_url"]).content
    with ZipFile(io.BytesIO(original_bytes)) as archive:
        original_narrative = archive.read("narrative.json")
    value = start(client, endpoint)
    before_history = history(client, project)["current_revision_id"]
    first, second = value["draft"]["scene_music"][:2]
    response = client.post(endpoint + "/music/replan", json={"expected_revision": value["draft"]["revision"],
                                                           "production_id": first["production_id"]})
    assert response.status_code == 200, response.text
    plan = claim(client, worker)
    assert plan["kind"] == "m3_music_plan" and plan["payload"]["planning_version"] == 2
    assert plan["payload"]["adjustment"]["replan"] is True
    assert complete(client, worker, plan, grouped_output(plan, mp3s, ("stop", "play"))).status_code == 200
    assert claim(client, worker) is None, "replan must not generate fresh audio or text"
    after = client.get(endpoint).json()
    assert after["draft"]["scene_music"][0]["action"] == "stop"
    selected = after["draft"]["scene_music"][1]
    assert selected["action"] == "play" and selected["candidate_id"] != first["candidate_id"]
    picked = next(row for row in after["music_candidates"] if row["id"] == selected["candidate_id"])
    original = next(row for row in after["music_candidates"] if row["id"] == first["candidate_id"])
    assert picked["scene_id"] == second["scene_id"] and picked["source"] == "reused"
    assert client.get(picked["music_url"]).content == client.get(original["music_url"]).content
    assert after["jobs"][-1]["adoption_status"] == "adopted"
    after_history = history(client, project)["current_revision_id"]
    compiled = []
    publish = _AdjustmentPublisher._publish

    def record_publish(self, connection, chapter, requirements, **options):
        compiled.append(chapter["id"])
        return publish(self, connection, chapter, requirements, **options)

    monkeypatch.setattr(_AdjustmentPublisher, "_publish", record_publish)
    applied = apply(client, endpoint, after)
    assert applied["draft"]["status"] == "applied" and compiled == [first["production_id"]]
    applied_history = history(client, project)["current_revision_id"]
    current = production(client, project)["chapters"]
    with ZipFile(io.BytesIO(client.get(current[0]["export_url"]).content)) as archive:
        assert archive.read("narrative.json") == original_narrative
        script = json.loads(archive.read("script.json"))
        assert [row["action"] for row in script["music_cues"]] == ["stop", "play"]
    assert client.get(before_chapters[0]["export_url"]).content == original_bytes
    restore(client, project, before_history)
    assert client.get(endpoint).json()["draft"]["scene_music"][1]["action"] == "continue"
    restore(client, project, after_history)
    restored = client.get(endpoint).json()
    assert restored["draft"]["scene_music"][1]["candidate_id"] == picked["id"]
    assert restored["scenes"][1]["music_plan"]["action"] == "play"
    restore(client, project, applied_history)
    assert client.get(endpoint).json()["draft"]["status"] == "applied"


def test_late_replan_does_not_overwrite_manual_edit(musical_story, mp3s):
    client, _project, worker, _jobs, endpoint = musical_story
    value = start(client, endpoint)
    response = client.post(endpoint + "/music/replan", json={"expected_revision": value["draft"]["revision"],
                        "production_id": value["scenes"][0]["production_id"]})
    assert response.status_code == 200, response.text
    value = response.json()
    plan = claim(client, worker)
    value["draft"]["scene_music"][0]["volume"] = .13
    edited = client.put(endpoint, json={"expected_revision": value["draft"]["revision"],
        "characters": value["draft"]["characters"], "scene_music": value["draft"]["scene_music"]})
    assert edited.status_code == 200, edited.text
    revision = edited.json()["draft"]["revision"]
    assert complete(client, worker, plan, grouped_output(plan, mp3s, ("play", "continue"))).status_code == 200
    after = client.get(endpoint).json()
    assert after["draft"]["revision"] == revision
    assert after["draft"]["scene_music"][0]["volume"] == .13
    assert after["draft"]["scene_music"][1]["action"] == "play"
    assert after["jobs"][-1]["adoption_status"] == "stale"
    assert "上書きしていません" in after["jobs"][-1]["adoption_reason"]


def test_replan_does_not_pick_unrelated_music_when_target_has_no_adopted_source(grouped_story, mp3s):
    client, _project, worker, _jobs, endpoint = grouped_story
    value = start(client, endpoint)
    rows = value["draft"]["scene_music"]
    for row in rows[:2]:
        row.update(action="stop", candidate_id=None)
    saved = client.put(endpoint, json={"expected_revision": value["draft"]["revision"],
        "characters": value["draft"]["characters"], "scene_music": rows})
    assert saved.status_code == 200, saved.text
    value = saved.json()
    response = client.post(endpoint + "/music/replan", json={"expected_revision": value["draft"]["revision"],
                        "production_id": rows[0]["production_id"]})
    assert response.status_code == 200, response.text
    job = claim(client, worker)
    assert complete(client, worker, job, grouped_output(job, mp3s, ("stop", "play"))).status_code == 200
    after = client.get(endpoint).json()
    second = after["draft"]["scene_music"][1]
    assert second["action"] == "stop" and second["candidate_id"] is None
    assert "採用済みの曲がない" in second["reason"]
    assert claim(client, worker) is None


@pytest.mark.parametrize("stop_first", [False, True])
def test_manual_continue_requires_active_source_in_same_chapter(musical_story, stop_first):
    client, _project, _worker, _jobs, endpoint = musical_story
    value = start(client, endpoint)
    rows = value["draft"]["scene_music"]
    index = 1 if stop_first else 0
    if stop_first:
        rows[0].update(action="stop", candidate_id=None)
    rows[index].update(action="continue", candidate_id=None)
    response = client.put(endpoint, json={"expected_revision": value["draft"]["revision"],
        "characters": value["draft"]["characters"], "scene_music": rows})
    assert response.status_code == 422 and "継続" in response.text


def test_music_only_apply_preserves_original_and_does_not_generate_speech(musical_story):
    client, project, _worker, _jobs, endpoint = musical_story
    original = production(client, project)["chapters"][0]
    old_bytes = client.get(original["export_url"]).content
    value = start(client, endpoint)
    initial = history(client, project)
    bindings = copy.deepcopy(value["draft"]["scene_music"])
    bindings[0].update(action="stop", candidate_id=None)
    bindings[1]["volume"] = 0.18
    response = client.put(endpoint, json={"expected_revision": value["draft"]["revision"],
        "characters": value["draft"]["characters"], "scene_music": bindings})
    assert response.status_code == 200, response.text
    changed = history(client, project)
    value = apply(client, endpoint, response.json())
    assert value["draft"]["status"] == "applied" and not value["jobs"]
    current = production(client, project)["chapters"][0]
    script, _ = script_for(client, current)
    assert script["music_cues"][0]["action"] == "stop"
    assert script["music_cues"][1]["volume"] == 0.18
    assert client.get(original["export_url"]).content == old_bytes
    restore(client, project, initial["current_revision_id"])
    assert client.get(endpoint).json()["draft"]["scene_music"][0]["action"] == "play"
    restore(client, project, changed["current_revision_id"])
    assert client.get(endpoint).json()["draft"]["scene_music"][0]["action"] == "stop"


def test_candidate_generation_two_stages_and_invalid_loop_not_adopted(musical_story, mp3s):
    client, project, worker, _jobs, endpoint = musical_story
    value = start(client, endpoint)
    scene = value["scenes"][0]
    response = client.post(endpoint + "/music/generate", json={"expected_revision": value["draft"]["revision"],
        "production_id": scene["production_id"], "scene_id": scene["scene_id"], "instruction": "緊張感を強く"})
    assert response.status_code == 200, response.text
    plan = claim(client, worker)
    assert plan["kind"] == "m3_music_plan" and len(plan["payload"]["context"]["narrative"]["scenes"]) == 1
    assert complete(client, worker, plan, music_output(plan, mp3s)).status_code == 200
    music = claim(client, worker)
    assert music["kind"] == "m3_music" and music["payload"]["duration_seconds"] == 120
    invalid = music_output(music, mp3s, patch={"loop_end_seconds": 130})
    assert complete(client, worker, music, invalid).status_code == 422
    assert complete(client, worker, music, music_output(music, mp3s)).status_code == 200
    after = client.get(endpoint).json()
    picked = next(row for row in after["music_candidates"] if row["job_id"] == music["id"])
    assert picked["music_url"] and picked["status"] == "completed"
    assert after["draft"]["scene_music"] == value["draft"]["scene_music"]
    bindings = copy.deepcopy(after["draft"]["scene_music"])
    bindings[1]["candidate_id"] = picked["id"]
    assert client.put(endpoint, json={"expected_revision": after["draft"]["revision"],
        "characters": after["draft"]["characters"], "scene_music": bindings}).status_code == 422
    assert production(client, project)["chapters"][0]["status"] == "published"


def test_music_only_apply_compiles_changed_chapters_and_clones_untouched_builds(musical_story, monkeypatch):
    from services.coordinator.adjustment_service import _AdjustmentPublisher

    client, project, _worker, _jobs, endpoint = musical_story
    original = production(client, project)["chapters"]
    assert len(original) > 1
    value = start(client, endpoint)
    bindings = copy.deepcopy(value["draft"]["scene_music"])
    changed_production = bindings[0]["production_id"]
    bindings[0].update(action="stop", candidate_id=None)
    response = client.put(endpoint, json={"expected_revision": value["draft"]["revision"],
        "characters": value["draft"]["characters"], "scene_music": bindings})
    assert response.status_code == 200, response.text
    compiled = []
    publish = _AdjustmentPublisher._publish

    def record_publish(self, connection, chapter, requirements, **options):
        compiled.append(chapter["id"])
        return publish(self, connection, chapter, requirements, **options)

    monkeypatch.setattr(_AdjustmentPublisher, "_publish", record_publish)
    value = apply(client, endpoint, response.json())
    assert value["draft"]["status"] == "applied"
    assert compiled == [changed_production], "BGM-only edits must not rebuild untouched chapters"
    current = production(client, project)["chapters"]
    for before, after in zip(original, current, strict=True):
        assert before["build"]["id"] != after["build"]["id"], "edition links need a distinct build identity"
        if before["production_id"] != changed_production:
            assert before["build"]["export_artifact_id"] == after["build"]["export_artifact_id"]
            assert before["build"]["script_artifact_id"] == after["build"]["script_artifact_id"]


@pytest.mark.parametrize("invalid", ["production_id", "scene_id", "expected_revision"])
def test_music_upload_validates_scope_and_revision_before_cpu_conversion(musical_story, monkeypatch, invalid):
    from services.coordinator import music_import

    client, _project, _worker, _jobs, endpoint = musical_story
    value = start(client, endpoint)
    scene = value["scenes"][0]
    fields = {"expected_revision": value["draft"]["revision"],
              "production_id": scene["production_id"], "scene_id": scene["scene_id"]}
    fields[invalid] = value["draft"]["revision"] + 1 if invalid == "expected_revision" else "wrong-scope"
    conversions = []

    def conversion(data, scene_id):
        conversions.append(scene_id)
        raise ValueError("A scope rejection must happen before conversion.")

    monkeypatch.setattr(music_import, "normalize_music", conversion)
    response = client.post(endpoint + "/music/upload", data=fields,
                           files={"file": ("candidate.mp3", b"mock audio bytes", "audio/mpeg")})
    assert response.status_code in (409, 422), response.text
    assert not conversions, "invalid uploads must not launch the 180-second CPU normalizer"


def test_mp3_validation_restricts_ffprobe_protocols(mp3s, monkeypatch):
    from services.coordinator import music_service

    run = music_service.subprocess.run
    commands = []

    def record(command, **options):
        commands.append(command)
        return run(command, **options)

    monkeypatch.setattr(music_service.subprocess, "run", record)
    assert music_service.mp3_duration(mp3s["music.mp3"]) == pytest.approx(70, abs=.1)
    assert len(commands) == 1
    command = commands[0]
    assert "-protocol_whitelist" in command, "a playlist disguised as MP3 must not open network URLs"
    assert command[command.index("-protocol_whitelist") + 1] == "file,pipe"


@pytest.mark.parametrize("reuse_draft", [True, False])
def test_existing_editions_without_music_settings_can_edit_a_single_scene(story, monkeypatch, reuse_draft):
    from services.coordinator.adjustment_service import AdjustmentService

    client, coordinator, project, _worker, endpoint = story
    insert = AdjustmentService._insert_edition

    def legacy_edition(self, connection, identifier, root, source, state, builds):
        state.pop("scene_music")
        return insert(self, connection, identifier, root, source, state, builds)

    # Reproduce a version created before BGM support, retaining immutable DB triggers.
    with monkeypatch.context() as patch:
        patch.setattr(AdjustmentService, "_insert_edition", legacy_edition)
        value = start(client, endpoint)
    edition_id, draft_id = value["edition"]["id"], value["draft"]["id"]
    original_revision = value["draft"]["revision"]
    with coordinator.db.transaction() as connection:
        if not reuse_draft:
            connection.execute("UPDATE project_history_state SET adjustment_draft_id=NULL WHERE project_id=?", (project,))
    response = client.post(endpoint + "/start", json={"expected_edition_id": edition_id})
    assert response.status_code == 200, response.text
    value = response.json()
    settings = value["draft"]["scene_music"]
    assert {(row["production_id"], row["scene_id"]) for row in settings} == {
        (row["production_id"], row["scene_id"]) for row in value["scenes"]}
    assert all(row["action"] == "stop" and row["candidate_id"] is None for row in settings)
    if reuse_draft:
        assert value["draft"]["id"] == draft_id
        assert value["draft"]["revision"] == original_revision + 1
    else:
        assert value["draft"]["id"] != draft_id
    settings[0]["volume"] = .17
    response = client.put(endpoint, json={"expected_revision": value["draft"]["revision"],
        "characters": value["draft"]["characters"], "scene_music": settings})
    assert response.status_code == 200, response.text
    assert response.json()["draft"]["scene_music"][0]["volume"] == .17
    with coordinator.db.transaction() as connection:
        edition = json.loads(connection.execute("SELECT state FROM publication_edition WHERE id=?", (edition_id,)).fetchone()[0])
        assert "scene_music" not in edition, "the legacy edition remains immutable"
    revision = response.json()["draft"]["revision"]
    assert client.post(endpoint + "/start", json={"expected_edition_id": edition_id}).json()["draft"]["revision"] == revision


def test_already_open_legacy_draft_get_has_all_music_rows_without_mutation_and_can_save(story, monkeypatch):
    from services.coordinator.adjustment_service import AdjustmentService

    client, coordinator, _project, _worker, endpoint = story
    insert = AdjustmentService._insert_edition

    def legacy_edition(self, connection, identifier, root, source, state, builds):
        state.pop("scene_music")
        return insert(self, connection, identifier, root, source, state, builds)

    with monkeypatch.context() as patch:
        patch.setattr(AdjustmentService, "_insert_edition", legacy_edition)
        before = start(client, endpoint)
    value = client.get(endpoint).json()
    assert value["draft"]["revision"] == before["draft"]["revision"]
    assert {(row["production_id"], row["scene_id"]) for row in value["draft"]["scene_music"]} == {
        (row["production_id"], row["scene_id"]) for row in value["scenes"]}
    with coordinator.db.transaction() as connection:
        stored = json.loads(connection.execute("SELECT state FROM adjustment_draft WHERE id=?", (value["draft"]["id"],)).fetchone()[0])
        assert "scene_music" not in stored, "reading a legacy draft must remain read-only"
    value["draft"]["scene_music"][0]["volume"] = .21
    response = client.put(endpoint, json={"expected_revision": value["draft"]["revision"],
        "characters": value["draft"]["characters"], "scene_music": value["draft"]["scene_music"]})
    assert response.status_code == 200, response.text
    assert response.json()["draft"]["scene_music"][0]["volume"] == .21
    with coordinator.db.transaction() as connection:
        edition = json.loads(connection.execute("SELECT state FROM publication_edition WHERE id=?", (value["edition"]["id"],)).fetchone()[0])
        assert "scene_music" not in edition
