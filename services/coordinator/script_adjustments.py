"""A published chapter's script, line by line, for the script and staging tab of the adjustments.

The tab shows the script as the current edition publishes it: its lines in playing order,
grouped by scene, with the directions anchored to each line and the stage each line shows.
The stage comes from the compiler's own pass over the script, so the preview follows the
same rules as the player.

Edits are kept in the draft as the values a line or direction should have. Every apply builds
each chapter's script again from its narrative, so the edits are layered on after each build
(``apply_script_edits``); the published script alone would lose them on the next apply.
"""
from __future__ import annotations

import json

from pydantic import TypeAdapter

from packages.contracts import Script
from packages.contracts.script import Direction
from packages.narrative.script_conversion import stable_id
from packages.narrative.staging import normalize_directions
from packages.narrative.validation import script_character_id
from packages.tyrano_export.compiler import event_cg_frames, line_stages, remote_frames
from packages.tyrano_export.presentation import STAGE

from .adjustment_service import AdjustmentService
from .service import ServiceError, required

# The directions the tab lets the user change; the others follow from the cast and the
# scene plan, and change every line after them.
EDITABLE = {"expression", "pause", "blackout"}
# What the staging step itself keeps to; a longer wait reads as the line having stalled.
PAUSE_BUDGET_MS = 1500


def _url(artifact_id: str) -> str:
    return f"/api/artifacts/{artifact_id}/content"


def _voice(row) -> tuple:
    return row["spoken_text"], row["voice_emotion"], row.get("delivery") or ""


def _edited_directions(script: Script, edits: dict | None, production_id: str) -> list[dict]:
    """The chapter's directions after the edits, before normalization caps anything.

    A direction disabled by an earlier edition is missing from that edition's script; its edit
    keeps the direction as it was, so turning it back on restores it.
    """
    changes = {row["direction_id"]: row for row in (edits or {}).get("directions", [])
               if row["production_id"] == production_id}

    def changed(direction, edit):
        if edit.get("expression") and direction["kind"] == "expression":
            direction["expression"] = edit["expression"]
        if edit.get("duration_ms") and direction["kind"] in {"pause", "blackout"}:
            direction["duration_ms"] = edit["duration_ms"]
        return direction

    rows = script.model_dump(mode="json")["directions"]
    present = {row["id"] for row in rows}
    result = [changed(row, changes[row["id"]]) if row["id"] in changes else row
              for row in rows if not changes.get(row["id"], {}).get("disabled")]
    return result + [changed(dict(edit["original"]), edit) for edit in changes.values()
                     if not edit["disabled"] and edit["direction_id"] not in present and edit.get("original")]


def apply_script_edits(script: Script, edits: dict | None, production_id: str) -> Script:
    """The chapter's script with the draft's edits for it; the same script when it has none."""
    lines = {row["utterance_id"]: row for row in (edits or {}).get("lines", []) if row["production_id"] == production_id}
    if not lines and not any(row["production_id"] == production_id for row in (edits or {}).get("directions", [])):
        return script
    data = script.model_dump(mode="json")
    for utterance in data["utterances"]:
        edit = lines.get(utterance["id"])
        if edit:
            utterance.update(display_text=edit["display_text"], spoken_text=edit["spoken_text"],
                             voice_emotion=edit["voice_emotion"])
            if edit.get("delivery"):
                utterance["delivery"] = edit["delivery"]
            else:
                utterance.pop("delivery", None)
    data["directions"], _report = normalize_directions(_edited_directions(script, edits, production_id))
    return Script.model_validate(data)


class ScriptAdjustmentService(AdjustmentService):
    def _chapter(self, connection, project_id, production_id):
        root, chapters = self._context(connection, project_id)
        chapter = next((row for row in chapters if row["id"] == production_id), None)
        if chapter is None:
            raise ServiceError(404, "完成した作品の章を指定してください。")
        return root, chapters, chapter

    def _script(self, connection, chapter) -> Script:
        record = required(connection, "artifact", chapter["build"]["script_artifact_id"])
        return Script.model_validate_json(self.store.read(record))

    def _edition_edits(self, connection, project_id) -> dict:
        selection = self.history.selection(connection, project_id)
        if not selection or not selection["edition_id"]:
            return {}
        return json.loads(required(connection, "publication_edition", selection["edition_id"])["state"]).get("script") or {}

    def _view_of(self, connection, root, chapter, script, edits=None):
        pictures = {asset.id: required(connection, "artifact", asset.artifact_id)
                    for asset in script.assets if asset.kind == "character"}
        narrative = self.m3._load_narrative(connection, chapter)
        snapshot = self.m3._snapshot(connection, root)
        published = self._edition_edits(connection, root["project_id"])
        disabled = [row["original"] for row in published.get("directions", [])
                    if row["production_id"] == chapter["id"] and row["disabled"] and row.get("original")]
        if edits is not None:
            script = apply_script_edits(script, edits, chapter["id"])
            disabled = []
        # Published files never change, so the portraits are read after the transaction.
        return lambda: _chapter_view(chapter, script, {key: self.store.read(row) for key, row in pictures.items()},
                                     narrative, snapshot, disabled)

    def chapter(self, project_id: str, production_id: str) -> dict:
        with self.db.transaction() as connection:
            root, _chapters, chapter = self._chapter(connection, project_id, production_id)
            view = self._view_of(connection, root, chapter, self._script(connection, chapter))
        return view()

    def preview(self, project_id: str, production_id: str, request) -> dict:
        """The chapter with unsaved edits, staged by the same pass as the player."""
        with self.db.transaction() as connection:
            root, _chapters, chapter = self._chapter(connection, project_id, production_id)
            script = self._script(connection, chapter)
            draft = self._draft(connection, project_id)
            known = (json.loads(draft["state"]).get("script") or {}) if draft else {}
            edits = request.model_dump(mode="json")
            edits["directions"] = _with_originals(edits["directions"], script, [
                *self._edition_edits(connection, project_id).get("directions", []), *known.get("directions", [])])
            view = self._view_of(connection, root, chapter, script, edits)
        return view()

    def voice(self, project_id: str, request) -> dict:
        """Record one line again in the voice the draft has chosen for its speaker."""
        with self.db.transaction() as connection:
            draft, root, chapters = self._editable(connection, project_id, request.expected_revision)
            chapter = next((row for row in chapters if row["id"] == request.production_id), None)
            if chapter is None:
                raise ServiceError(404, "完成した作品の章を指定してください。")
            utterance = next((row for row in self._script(connection, chapter).utterances
                              if row.id == request.utterance_id), None)
            state = json.loads(draft["state"])
            requirement = next((row for row in state["requirements"].get(chapter["id"], [])
                                if row["kind"] == "m3_voice_clone" and row["target_id"] == request.utterance_id), None)
            if utterance is None or utterance.speaker_id is None or requirement is None:
                raise ServiceError(422, "音声のある台詞を指定してください。")
            descriptor = json.loads(requirement["descriptor"])
            speaker = descriptor["character_id"]
            setting = next((row for row in state["characters"] if row["character_id"] == speaker), None)
            if setting is None or not setting["voice_candidate_id"]:
                raise ServiceError(422, "この人物の基準音声を選んでください。")
            candidate = self._candidate(connection, draft, setting["voice_candidate_id"], speaker, "voice")
            record = required(connection, "artifact", candidate["artifact_id"])
            self.history.begin(connection, project_id, "台詞の音声を作り直す", "m3-adjustment-line-voice", concurrent=True)
            draft = {**draft, "revision": draft["revision"] + 1}
            self._enqueue(connection, draft, root, "m3_voice_clone", "dialogue", speaker, production=chapter,
                          target_id=request.utterance_id, material={
                              **descriptor, "dialogue_text": request.spoken_text,
                              "voice_emotion": request.voice_emotion, "delivery": request.delivery,
                              "reference_voice": {"artifact_id": record["id"], "sha256": record["sha256"],
                                                  "text": candidate["reference_text"]}})
            connection.execute("UPDATE adjustment_draft SET revision=revision+1 WHERE id=?", (draft["id"],))
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)


def _with_originals(edits: list[dict], script: Script, known: list[dict]) -> list[dict]:
    """Each direction edit with the direction as it was, from the script or an earlier edit."""
    directions = {row.id: row.model_dump(mode="json") for row in script.directions}
    earlier = {(row["production_id"], row["direction_id"]): row.get("original") for row in known}
    return [{**edit, "original": directions.get(edit["direction_id"])
             or earlier.get((edit["production_id"], edit["direction_id"]))} for edit in edits]


def script_voices(connection, draft_id: str) -> list[dict]:
    """The lines recorded again in this draft, oldest first, with the words each was made for."""
    rows = connection.execute(
        "SELECT a.job_id,a.production_id,a.target_id,a.artifact_id,j.status,j.error,"
        "json_extract(j.payload,'$.dialogue_text') AS spoken_text,"
        "json_extract(j.payload,'$.voice_emotion') AS voice_emotion,"
        "json_extract(j.payload,'$.delivery') AS delivery "
        "FROM adjustment_job a JOIN job j ON j.id=a.job_id "
        "WHERE a.draft_id=? AND a.purpose='dialogue' AND a.apply_id IS NULL ORDER BY j.created_at,j.id",
        (draft_id,)).fetchall()
    return [{"job_id": row["job_id"], "production_id": row["production_id"], "utterance_id": row["target_id"],
             "spoken_text": row["spoken_text"], "voice_emotion": row["voice_emotion"], "delivery": row["delivery"] or "",
             "status": row["status"], "error": row["error"],
             "url": _url(row["artifact_id"]) if row["artifact_id"] else None} for row in rows]


def store_script_edits(service, connection, draft, chapters, edits) -> dict:
    """Check the edits against the published chapters and keep each direction as it was."""
    by_id = {row["id"]: row for row in chapters}
    scripts: dict[str, Script] = {}

    def script(production_id):
        if production_id not in by_id:
            raise ServiceError(422, "完成した作品の章を指定してください。")
        if production_id not in scripts:
            record = required(connection, "artifact", by_id[production_id]["build"]["script_artifact_id"])
            scripts[production_id] = Script.model_validate_json(service.store.read(record))
        return scripts[production_id]

    known = (json.loads(draft["state"]).get("script") or {}).get("directions", [])
    voices = {row["job_id"]: row for row in script_voices(connection, draft["id"])}
    lines = []
    for edit in edits.lines:
        value = edit.model_dump(mode="json")
        utterance = next((row for row in script(edit.production_id).utterances if row.id == edit.utterance_id), None)
        if utterance is None:
            raise ServiceError(422, "台本にない行は編集できません。")
        if utterance.speaker_id is None:
            # Narration has no voice: what it reads is what it shows.
            value.update(spoken_text=value["display_text"], voice_emotion=utterance.voice_emotion,
                         delivery=utterance.delivery or "", audio=None, voice_job_id=None)
        if value["voice_job_id"]:
            voice = voices.get(value["voice_job_id"])
            if (voice is None or voice["utterance_id"] != edit.utterance_id or voice["production_id"] != edit.production_id
                    or voice["status"] != "completed" or not voice["url"]):
                raise ServiceError(422, "この行の完成した音声候補を選んでください。")
            if _voice(voice) != _voice(value):
                raise ServiceError(422, "音声候補が今の読み・感情・話し方と一致しません。")
        if value["audio"] == "old" and _voice(value) == _voice(utterance.model_dump(mode="json")):
            value["audio"] = None
        lines.append(value)
    directions = [row.model_dump(mode="json") for row in edits.directions]
    for production_id in {row["production_id"] for row in directions}:
        script(production_id)
    by_chapter: dict[str, list[dict]] = {}
    for row in directions:
        by_chapter.setdefault(row["production_id"], []).append(row)
    directions = [value for production_id, rows in by_chapter.items()
                  for value in _with_originals(rows, script(production_id), known)]
    for value in directions:
        original = value["original"]
        if original is None or original["kind"] not in EDITABLE:
            raise ServiceError(422, "変更できる演出を指定してください。")
        if (value["expression"] and original["kind"] != "expression") or (
                value["duration_ms"] and original["kind"] not in {"pause", "blackout"}):
            raise ServiceError(422, "演出の種類に合う値を指定してください。")
    edited = {"lines": lines, "directions": directions}
    if any(pauses_overrun(script(production_id), edited, production_id) for production_id in by_chapter):
        raise ServiceError(422, "間は1行で合計1.5秒までです。")
    return edited


def pauses_overrun(script: Script, edits: dict, production_id: str) -> bool:
    """Whether the edits make some line wait longer than the staging step ever would."""
    pauses: dict[str, int] = {}
    for row in _edited_directions(script, edits, production_id):
        if row["kind"] == "pause":
            pauses[row["utterance_id"]] = pauses.get(row["utterance_id"], 0) + row["duration_ms"]
    return any(total > PAUSE_BUDGET_MS for total in pauses.values())


def pending_voices(service, connection, chapters, edits: dict | None) -> list[dict]:
    """Lines whose reading, emotion or delivery changed with no voice chosen for them yet."""
    by_id = {row["id"]: row for row in chapters}
    pending, scripts = [], {}
    for row in (edits or {}).get("lines", []):
        if row.get("audio") or row.get("voice_job_id") or row["production_id"] not in by_id:
            continue
        if row["production_id"] not in scripts:
            record = required(connection, "artifact", by_id[row["production_id"]]["build"]["script_artifact_id"])
            scripts[row["production_id"]] = {value.id: value for value in
                                             Script.model_validate_json(service.store.read(record)).utterances}
        utterance = scripts[row["production_id"]].get(row["utterance_id"])
        if utterance and utterance.speaker_id and _voice(row) != _voice(utterance.model_dump(mode="json")):
            pending.append(row)
    return pending


def _chapter_view(chapter, script: Script, images, narrative, snapshot, disabled=()) -> dict:
    """``disabled`` lists directions an earlier edition left out; they are shown, turned off."""
    assets = {value.id: value for value in script.assets}
    files = {value.filename: _url(value.artifact_id) for value in script.assets}
    names = {value.id: value.name for value in script.characters}
    outfits = {script_character_id(row["id"]): {outfit["id"]: outfit["name"] for outfit in row.get("outfits", [])}
               for row in snapshot["characters"]}
    places = {stable_id("background", value.id): value.name for value in narrative.locations}
    stages = line_stages(script, images)
    unseen = set(stages["unseen"])
    frames = remote_frames(script, images)["lines"] if any(
        value.remote is not None for value in script.utterances) else {}
    cg_frames = event_cg_frames(script)
    cg_starts = {value.start_utterance_id for value in script.event_cg_segments}
    cues = {value.utterance_id: value for value in script.music_cues}
    transitions = {value.utterance_id: value for value in script.scene_transitions}
    directions: dict[str, list] = {}
    off = [TypeAdapter(Direction).validate_python(row) for row in disabled]
    for value in [*script.directions, *off]:
        if value.kind != "focus":
            directions.setdefault(value.utterance_id, []).append(value)
    off_ids = {value.id for value in off}

    scene_of, scenes = {}, []
    locations = {value.id: value.name for value in narrative.locations}
    for number, scene in enumerate(narrative.scenes, 1):
        for utterance in scene.utterances:
            scene_of[utterance.id] = scene.id
        scenes.append({"scene_id": scene.id, "title": f"シーン{number}・{locations[scene.plan.location_id]}",
                       "first_utterance_id": scene.utterances[0].id})
    titles = {row["scene_id"]: row["title"] for row in scenes}

    def stage(utterance_id):
        shown = stages["lines"][utterance_id]
        frame = frames.get(utterance_id)
        return {
            "background_url": files[shown["background"]["storage"]] if shown["background"] else None,
            "characters": [{"id": row["id"], "image_url": files[row["storage"]],
                            **{key: row[key] for key in ("left", "top", "width", "height")}}
                           for row in shown["characters"]],
            "lit": shown["lit"],
            "event_cg_url": files[shown["event_cg"]["storage"]] if shown["event_cg"] else None,
            "remote_frame": {"image_url": files[frame["storage"]], "crop": frame["crop"],
                             "width": frame["width"], "height": frame["height"]} if frame else None,
        }

    def direction(value):
        row = {"id": value.id, "kind": value.kind, "timing": value.timing,
               "editable": value.kind in EDITABLE, "shown": value.id not in unseen and value.id not in off_ids,
               "disabled": value.id in off_ids}
        character = getattr(value, "character_id", None)
        if character:
            row.update(character_id=character, character_name=names.get(character, ""))
        if value.kind == "background":
            row["place"] = places.get(value.asset_id, "")
        if value.kind in {"enter", "position"}:
            row["position"] = value.position
        outfit = value.outfit_id if value.kind in {"enter", "outfit"} else None
        if outfit:
            row["outfit_name"] = outfits.get(character, {}).get(outfit) or ("通常" if outfit == "default" else outfit)
        if value.kind == "expression":
            row["expression"] = value.expression
        if value.kind in {"pause", "blackout"}:
            row["duration_ms"] = value.duration_ms
        return row

    lines, playing = [], None
    for utterance in script.utterances:
        scene_id = scene_of.get(utterance.id, lines[-1]["scene_id"] if lines else scenes[0]["scene_id"])
        cue, music = cues.get(utterance.id), None
        if cue and cue.action == "play":
            playing = f"{titles[scene_id]}の曲"
            music = {"action": "play", "title": playing}
        elif cue and cue.action == "stop":
            music = {"action": "stop"} if playing else None
            playing = None
        transition = transitions.get(utterance.id)
        lines.append({
            "id": utterance.id, "scene_id": scene_id,
            "speaker_id": utterance.speaker_id, "speaker_name": names.get(utterance.speaker_id, ""),
            "display_text": utterance.display_text, "spoken_text": utterance.spoken_text,
            "voice_emotion": utterance.voice_emotion, "delivery": utterance.delivery or "",
            "audio_url": _url(assets[utterance.audio_asset_id].artifact_id) if utterance.audio_asset_id else None,
            "remote": utterance.remote.model_dump(mode="json") if utterance.remote else None,
            "music": music, "music_title": playing,
            "event_cg": {"start": utterance.id in cg_starts,
                         "image_url": files[assets[cg_frames[utterance.id]["asset_id"]].filename]}
            if utterance.id in cg_frames else None,
            "transition": {"visual": transition.visual, "duration_ms": transition.duration_ms} if transition else None,
            "directions": [direction(value) for value in directions.get(utterance.id, [])],
            "stage": stage(utterance.id),
        })
    first = {row["id"]: row for row in lines}
    for scene in scenes:
        line = first.get(scene["first_utterance_id"])
        scene["background_url"] = line["stage"]["background_url"] if line else None
        cue = cues.get(scene["first_utterance_id"])
        scene["music"] = cue.action if cue else "continue" if line and line["music_title"] else "stop"
        scene["transition"] = line["transition"] if line else None
    return {"production_id": chapter["id"], "chapter_number": chapter["chapter_number"],
            "stage": dict(STAGE), "scenes": scenes, "lines": lines}
