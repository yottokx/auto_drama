"""A published chapter's script, line by line, for the script and staging tab of the adjustments.

The tab shows the script as the current edition publishes it: its lines in playing order,
grouped by scene, with the directions anchored to each line and the stage each line shows.
The stage comes from the compiler's own pass over the script, so the preview follows the
same rules as the player.
"""
from __future__ import annotations

from packages.contracts import Script
from packages.narrative.script_conversion import stable_id
from packages.narrative.validation import script_character_id
from packages.tyrano_export.compiler import event_cg_frames, line_stages, remote_frames
from packages.tyrano_export.presentation import STAGE

from .adjustment_service import AdjustmentService
from .service import ServiceError, required

# The directions the tab will let the user change; the others follow from the cast and the
# scene plan, and change every line after them.
EDITABLE = {"expression", "pause", "blackout"}


def _url(artifact_id: str) -> str:
    return f"/api/artifacts/{artifact_id}/content"


class ScriptAdjustmentService(AdjustmentService):
    def chapter(self, project_id: str, production_id: str) -> dict:
        with self.db.transaction() as connection:
            root, chapters = self._context(connection, project_id)
            chapter = next((row for row in chapters if row["id"] == production_id), None)
            if chapter is None:
                raise ServiceError(404, "完成した作品の章を指定してください。")
            record = required(connection, "artifact", chapter["build"]["script_artifact_id"])
            script = Script.model_validate_json(self.store.read(record))
            pictures = {asset.id: required(connection, "artifact", asset.artifact_id)
                        for asset in script.assets if asset.kind == "character"}
            narrative = self.m3._load_narrative(connection, chapter)
            snapshot = self.m3._snapshot(connection, root)
        # Published files never change, so the portraits are read after the transaction.
        images = {identifier: self.store.read(row) for identifier, row in pictures.items()}
        return _chapter_view(chapter, script, images, narrative, snapshot)


def _chapter_view(chapter, script: Script, images, narrative, snapshot) -> dict:
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
    for value in script.directions:
        if value.kind != "focus":
            directions.setdefault(value.utterance_id, []).append(value)

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
               "editable": value.kind in EDITABLE, "shown": value.id not in unseen}
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
