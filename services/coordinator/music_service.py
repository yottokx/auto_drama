"""Immutable scene music adoption and publication helpers."""
from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from uuid import uuid4

from packages.contracts import Script
from packages.contracts.music import MusicPlan, MusicResult, SceneMusicSetting
from packages.contracts.script import SceneTransitionSpec

from .service import ServiceError, required


def enabled_for_new_series(plan_approval_id):
    # Legacy workflows and saved publications remain reproducible. The normal
    # approved-plan workflow opts into music at series creation, then pins it.
    from .m2_service import ROOT

    config = json.loads((ROOT / "config/m2-generation.json").read_text(encoding="utf-8"))
    return bool(plan_approval_id and config.get("music", {}).get("enabled", True))


def context(m3, connection, production, narrative=None):
    narrative = narrative or m3._load_narrative(connection, production)
    snapshot = m3._snapshot(connection, production)
    return {"approval_snapshot": snapshot, "world": snapshot["world"]["result"],
            "overall_plot": narrative.outline.model_dump(mode="json"),
            "chapter_number": production["chapter_number"],
            "narrative": narrative.model_dump(mode="json")}


def validate_plan(result, payload):
    plan = MusicPlan.model_validate(result)
    expected = [row["id"] for row in payload["context"]["narrative"]["scenes"]]
    if [row.scene_id for row in plan.scenes] != expected:
        raise ValueError("Music plan differs from the frozen scenes.")
    if plan.planning_version != payload.get("planning_version", 1):
        raise ValueError("Music plan differs from its requested version.")
    if payload.get("planning_scope") == "single_scene" and any(row.action != "play" for row in plan.scenes):
        raise ValueError("A manual music candidate must request a new track.")
    return plan


def mp3_duration(data):
    if not data or len(data) > 32 * 1024 * 1024:
        raise ValueError("Music must be an MP3 within 32MiB.")
    ffprobe = shutil.which("ffprobe") or ("C:/ffmpeg/bin/ffprobe.exe" if Path("C:/ffmpeg/bin/ffprobe.exe").is_file() else None)
    if not ffprobe:
        raise ValueError("BGMの検証にはffprobeが必要です。")
    with tempfile.TemporaryDirectory(prefix="auto-drama-music-") as directory:
        path = Path(directory) / "music.mp3"
        path.write_bytes(data)
        process = subprocess.run([ffprobe, "-v", "error", "-protocol_whitelist", "file,pipe", "-f", "mp3",
                                  "-show_format", "-show_streams", "-of", "json", str(path)],
                                 capture_output=True, timeout=30, check=False,
                                 creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if process.returncode:
        raise ValueError("MP3を読み取れません。")
    value = json.loads(process.stdout)
    streams = value.get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_name") != "mp3" or streams[0].get("channels") not in (1, 2):
        raise ValueError("BGMにはMP3音声を指定してください。")
    duration = float(value["format"]["duration"])
    if not 5 <= duration <= 381:
        raise ValueError("BGMの長さは5〜380秒にしてください。")
    return duration


def validate_result(result, payload, files):
    music = MusicResult.model_validate(result)
    if music.scene_id != payload["scene_id"] or music.prompt != payload["prompt"]:
        raise ValueError("Music differs from the fixed scene or prompt.")
    if abs(music.source_duration_seconds - payload.get("duration_seconds", 120)) > 0.1:
        raise ValueError("Music source duration differs from its request.")
    if abs(mp3_duration(files["music.mp3"]) - music.duration_seconds) > 0.15:
        raise ValueError("Music playback duration differs from metadata.")
    if abs(mp3_duration(files["source.mp3"]) - music.source_duration_seconds) > 0.15:
        raise ValueError("Music source duration differs from metadata.")
    return music


def store_candidate(m3, connection, production, result, files, *, source="generated", job=None, attempt=None,
                    candidate_id=None, provenance=None):
    result = MusicResult.model_validate(result).model_dump(mode="json")
    identifier = candidate_id or uuid4().hex
    metadata = {**(provenance or {}), **result}
    music = m3._artifact(connection, production, "music-" + identifier, "music", "music.mp3", files["music.mp3"],
                         provenance=metadata, job=job, attempt=attempt)
    original = m3._artifact(connection, production, "music-source-" + identifier, "music_source", "source.mp3", files["source.mp3"],
                            provenance=metadata, job=job, attempt=attempt)
    if candidate_id:
        connection.execute("UPDATE music_candidate SET artifact_id=?,source_artifact_id=?,prompt=?,metadata=? WHERE id=?",
                           (music["id"], original["id"], result["prompt"], json.dumps(metadata), identifier))
    else:
        connection.execute("INSERT INTO music_candidate VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                           (identifier, production["project_id"], production["storyline_id"], production["id"],
                            result["scene_id"], music["id"], original["id"], source, result["prompt"],
                            json.dumps(metadata), job["id"] if job else None, m3.clock()))
    return music, identifier


def fallback_transition(narrative, index):
    same_place = index > 0 and narrative.scenes[index - 1].plan.location_id == narrative.scenes[index].plan.location_id
    return SceneTransitionSpec(visual="dissolve" if same_place else "fade",
                               music_fade_out_ms=0 if index == 0 else 1000).model_dump(mode="json")


def chapter_plan(connection, chapter, m3, presentation=None):
    identifier = (presentation or {}).get("music_plans", {}).get(chapter["id"])
    if identifier:
        record = required(connection, "artifact", identifier)
        provenance = json.loads(record["provenance"])
        if (record["project_id"] != chapter["project_id"] or record["kind"] != "music_plan"
                or provenance.get("production_id") != chapter["id"]):
            raise ServiceError(422, "この章のBGM計画を指定してください。")
    else:
        row = connection.execute("SELECT artifact_id FROM m3_requirement WHERE production_id=? "
                                 "AND kind='m3_music_plan' AND target_id='chapter-music'", (chapter["id"],)).fetchone()
        if not row or not row["artifact_id"]:
            return None
        record = required(connection, "artifact", row["artifact_id"])
    plan = MusicPlan.model_validate_json(m3.store.read(record))
    expected = [scene.id for scene in m3._load_narrative(connection, chapter).scenes]
    if [row.scene_id for row in plan.scenes] != expected:
        raise ServiceError(422, "BGM計画と章の場面が一致しません。")
    return plan


def automatic_settings(connection, chapter, m3, presentation=None):
    narrative = m3._load_narrative(connection, chapter)
    plan = chapter_plan(connection, chapter, m3, presentation)
    decisions = {row.scene_id: row for row in plan.scenes} if plan else {}
    # Replanned rows pin their adopted candidate as part of the immutable plan.
    override = (presentation or {}).get("music_plans", {}).get(chapter["id"])
    bindings = json.loads(required(connection, "artifact", override)["provenance"]).get("bindings", []) if override else []
    selected = {row["scene_id"]: row for row in bindings}
    settings = []
    for index, scene in enumerate(narrative.scenes):
        row = connection.execute("SELECT c.id FROM m3_requirement r JOIN music_candidate c ON c.artifact_id=r.artifact_id "
                                 "WHERE r.production_id=? AND r.kind='m3_music' AND r.target_id=? ORDER BY c.created_at,c.id LIMIT 1",
                                 (chapter["id"], scene.id)).fetchone()
        decision = decisions.get(scene.id)
        binding = selected.get(scene.id)
        action = decision.action if decision else "play" if row else "stop"
        identifier = binding.get("candidate_id") if binding else row["id"] if row else None
        if action == "play" and not identifier:
            action = "stop"
        settings.append({"production_id": chapter["id"], "scene_id": scene.id,
                         "candidate_id": identifier if action == "play" else None,
                         "action": action, "volume": 0.35,
                         "transition": decision.transition.model_dump(mode="json") if decision and decision.transition
                         else fallback_transition(narrative, index),
                         "reason": decision.reason if decision and decision.reason else "既存の採用曲と場面の切り替わりに合わせた設定。"})
    return settings


def original_settings(connection, chapters, m3, presentation=None):
    return [row for chapter in chapters for row in automatic_settings(connection, chapter, m3, presentation)]


def effective_settings(connection, chapters, m3, presentation=None):
    selected = {(row["production_id"], row["scene_id"]): row for row in (presentation or {}).get("scene_music", [])}
    return [base | selected.get((base["production_id"], base["scene_id"]), {})
            for base in original_settings(connection, chapters, m3, presentation)]


def plan_metadata(connection, chapter, m3, presentation=None):
    plan = chapter_plan(connection, chapter, m3, presentation)
    prompts = {row.scene_id: row.prompt for row in plan.scenes} if plan else {}
    source, result = None, {}
    for row in automatic_settings(connection, chapter, m3, presentation):
        source = row["scene_id"] if row["action"] == "play" else None if row["action"] == "stop" else source
        result[row["scene_id"]] = {key: row[key] for key in ("action", "candidate_id", "reason", "transition")}
        result[row["scene_id"]].update(source_scene_id=source, prompt=prompts.get(source, ""))
    return result


def validate_continuity(settings, chapters, m3, connection):
    by_scene = {(row["production_id"], row["scene_id"]): row for row in settings}
    for chapter in chapters:
        active = False
        for scene in m3._load_narrative(connection, chapter).scenes:
            action = by_scene[(chapter["id"], scene.id)]["action"]
            if action == "continue" and not active:
                raise ServiceError(422, "章の先頭・BGM停止後には継続を指定できません。")
            if action != "continue":
                active = action == "play"


def candidate(connection, production_id, scene_id, identifier, project_id=None):
    row = required(connection, "music_candidate", identifier)
    if (row["production_id"] != production_id or row["scene_id"] != scene_id or not row["artifact_id"]
            or (project_id and row["project_id"] != project_id)):
        raise ServiceError(422, "この場面の完成済みBGM候補を指定してください。")
    return row


def add_to_script(m3, connection, production, narrative, script, content, requirements, presentation=None):
    settings = effective_settings(connection, [production], m3, presentation)
    validate_continuity(settings, [production], m3, connection)
    by_scene = {row["scene_id"]: row for row in settings}
    has_music = any(row["kind"] in {"m3_music", "m3_music_plan"} for row in requirements) or bool(
        (presentation or {}).get("scene_music"))
    assets, cues, transitions = list(script.model_dump(mode="json")["assets"]), [], []
    for scene in narrative.scenes:
        setting = by_scene[scene.id]
        transition = setting["transition"]
        if setting["action"] == "continue":
            transition = {**transition, "music_fade_out_ms": 0, "music_fade_in_ms": 0}
        transitions.append({"id": "scene-transition-" + scene.id, "utterance_id": scene.utterances[0].id,
                            **transition})
        if not has_music:
            continue
        record = None
        SceneMusicSetting.model_validate(setting)
        if setting["action"] == "play":
            picked = candidate(connection, production["id"], scene.id, setting["candidate_id"], production["project_id"])
            record = required(connection, "artifact", picked["artifact_id"])
        action = setting["action"]
        cue = {"id": "music-cue-" + scene.id, "utterance_id": scene.utterances[0].id,
               "action": action, "volume": setting["volume"] if setting else 0.35}
        if record:
            if record["kind"] != "music" or record["project_id"] != production["project_id"]:
                raise ValueError("Music asset belongs to a different publication.")
            metadata = json.loads(record["provenance"])
            identifier = "music-" + scene.id
            assets.append({"id": identifier, "kind": "music", "artifact_id": record["id"],
                           "filename": identifier + ".mp3", "sha256": record["sha256"]})
            content[identifier] = m3.store.read(record)
            cue.update(asset_id=identifier, loop_start_seconds=metadata["loop_start_seconds"],
                       loop_end_seconds=metadata["loop_end_seconds"])
        cues.append(cue)
    additions = {"assets": assets, "scene_transitions": transitions}
    if has_music:
        additions["music_cues"] = cues
    return Script.model_validate(script.model_dump(mode="json") | additions)
