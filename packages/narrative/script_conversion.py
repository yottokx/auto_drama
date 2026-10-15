"""Pure conversion shared by publication and media-free script experiments."""

from __future__ import annotations

import hashlib

from packages.contracts import Script
from packages.contracts.m3 import NarrativeDirection, NarrativeResult, PortraitSetting
from packages.contracts.outfits import DEFAULT_OUTFIT, image_target, snapshot_outfits

from .staging import normalize_directions
from .validation import script_character_id


def stable_id(prefix: str, value: str) -> str:
    return prefix + "-" + hashlib.sha256(value.encode()).hexdigest()[:24]


def narrative_to_script(narrative: NarrativeResult, snapshot: dict, references: dict, *,
                        script_id: str, portrait_settings: dict | None = None,
                        allow_missing_audio: bool = False,
                        omitted_portraits: set[str] | None = None,
                        normalization_reports: list[dict] | None = None,
                        expression_pictures: dict | None = None,
                        portrait_faces: dict | None = None) -> Script:
    """Use already-resolved assets without DB access, generation, or prose rewriting.

    Reference keys are the production requirement's ``(kind, target_id)`` pairs.
    Production requires every speaking line's audio; debug exports explicitly opt
    into the Script contract's existing nullable audio reference.

    ``expression_pictures`` maps ``(character ID, look ID or None)`` to the asset of each
    expression that look has a picture of; their assets are among ``references``.

    ``portrait_faces`` maps a portrait's image target to where the face is in it, for the frame
    of a remote voice; a portrait without one is framed from its silhouette.
    """
    portrait_faces = portrait_faces or {}
    narrative = NarrativeResult.model_validate(narrative)
    cast = {value.get("id", value["result"]["id"]): value["result"]
            for value in snapshot["characters"]}
    cast.update({value.id: value.model_dump(mode="json")
                 for value in narrative.supporting_characters})
    portrait_settings = portrait_settings or {}
    omitted_portraits = omitted_portraits or set()
    if not omitted_portraits <= {value.id for value in narrative.supporting_characters}:
        raise ValueError("Only supporting portraits may be omitted.")

    def presentation(cid, value):
        if cid in omitted_portraits:
            return {}
        if cid in portrait_settings:
            setting = PortraitSetting.model_validate(portrait_settings[cid])
            result = setting.model_dump(exclude={"character_id", "image_artifact_id"})
            if setting.image_artifact_id != references[("m3_image", cid)]["artifact_id"]:
                result["body_bounds"] = None
            return result
        body = value.get("body_type", "unknown")
        return {"framing": {"humanoid": "upper_body", "nonhumanoid": "full_body"}.get(
            body, "auto"), "height_cm": value.get("height_cm") if body == "humanoid" else None}

    # An alternate look is used only where its own portrait was resolved.
    looks = {cid: {row.id: references[("m3_image", image_target(cid, row.id))]["id"] for row in rows
                   if row.id != DEFAULT_OUTFIT and ("m3_image", image_target(cid, row.id)) in references}
             for cid, rows in snapshot_outfits(snapshot).items()}

    def faces(cid):
        # Only for a look this chapter shows; a picture of a look it does not show is left out.
        return [{"expression": expression, "image_asset_id": asset, **({"outfit_id": oid} if oid else {})}
                for (owner, oid), pictures in sorted((expression_pictures or {}).items(), key=lambda row: row[0][1] or "")
                if owner == cid and (oid is None or oid in looks.get(cid, {}))
                for expression, asset in sorted(pictures.items())]

    def look(cid, oid, asset):
        value = {"id": oid, "image_asset_id": asset}
        target = image_target(cid, oid)
        if target in portrait_faces:
            value["face"] = portrait_faces[target]
        if target in portrait_settings:
            setting = PortraitSetting.model_validate(portrait_settings[target])
            if (setting.body_bounds is not None
                    and setting.image_artifact_id == references[("m3_image", target)]["artifact_id"]):
                value["body_bounds"] = setting.body_bounds.model_dump(mode="json")
        return value

    characters = [{"id": script_character_id(cid), "name": value["name"],
                   "image_asset_id": references[("m3_image", cid)]["id"] if cid not in omitted_portraits else None,
                   **presentation(cid, value),
                   **({"outfits": [look(cid, oid, asset) for oid, asset in looks[cid].items()]}
                      if looks.get(cid) else {}),
                   **({"expressions": faces(cid)} if cid not in omitted_portraits and faces(cid) else {}),
                   **({"face": portrait_faces[cid]} if cid in portrait_faces and cid not in omitted_portraits
                      else {})}
                  for cid, value in cast.items() if ("m3_image", cid) in references or cid in omitted_portraits]
    utterances, directions = [], []
    for scene in narrative.scenes:
        first = scene.utterances[0].id
        worn = {row.character_id: row.outfit_id for row in scene.plan.outfits
                if row.outfit_id in looks.get(row.character_id, {})}
        directions.append({"id": stable_id("background", scene.id), "kind": "background",
                           "utterance_id": first, "timing": "before", "duration_ms": 0,
                           "asset_id": references[("m3_background", scene.plan.location_id)]["id"]})
        for index, character in enumerate(characters):
            if character["image_asset_id"] is None:
                continue
            directions.append({"id": stable_id("exit", scene.id + str(index)), "kind": "exit",
                               "utterance_id": first, "timing": "before", "duration_ms": 0,
                               "character_id": character["id"]})
        for utterance in scene.utterances:
            audio = None
            if utterance.speaker_id:
                key = ("m3_voice_clone", utterance.id)
                audio = references.get(key) if allow_missing_audio else references[key]
            utterances.append({"id": utterance.id,
                               "speaker_id": script_character_id(utterance.speaker_id)
                               if utterance.speaker_id else None,
                               "display_text": utterance.display_text,
                               "spoken_text": utterance.spoken_text,
                               "voice_emotion": utterance.voice_emotion,
                               "delivery": utterance.delivery,
                               "audio_asset_id": audio["id"] if audio else None,
                               **({"remote": utterance.remote.model_dump(mode="json")} if utterance.remote else {})})
        normalized, report = normalize_directions([
            row.model_dump(mode="json") for row in scene.directions])
        if report["changes"] and normalization_reports is not None:
            normalization_reports.append({"scene_id": scene.id, **report})
        for raw_direction in normalized:
            direction = NarrativeDirection.model_validate(raw_direction)
            if direction.character_id in omitted_portraits:
                continue
            if direction.kind == "outfit" and direction.outfit_id not in {
                    DEFAULT_OUTFIT, *looks.get(direction.character_id, {})}:
                continue
            value = {"id": direction.id, "kind": direction.kind,
                     "utterance_id": direction.utterance_id, "timing": direction.timing}
            if direction.kind in ("enter", "exit", "position", "focus", "outfit", "expression"):
                value["character_id"] = script_character_id(direction.character_id)
            if direction.kind == "outfit":
                value["outfit_id"] = direction.outfit_id
            if direction.kind == "expression":
                # The face the script asks for, whether or not a picture of it exists yet.
                value["expression"] = direction.expression
            if direction.kind in ("enter", "position"):
                value["position"] = direction.position
            if direction.kind == "enter" and direction.character_id in worn:
                value["outfit_id"] = worn[direction.character_id]
            if direction.kind != "focus":
                value["duration_ms"] = direction.duration_ms
            directions.append(value)
        # A person only heard in this scene never comes on, yet their frame shows the scene's look.
        entered = {row.character_id for row in scene.directions if row.kind == "enter"}
        for cid, outfit_id in sorted(worn.items()):
            if cid not in entered and cid not in omitted_portraits:
                directions.append({"id": stable_id("look", scene.id + cid), "kind": "outfit",
                                   "utterance_id": first, "timing": "before", "duration_ms": 0,
                                   "character_id": script_character_id(cid), "outfit_id": outfit_id})
    return Script.model_validate({"schema_version": 1, "id": script_id,
                                  "title": narrative.title, "characters": characters,
                                  "utterances": utterances, "directions": directions,
                                  "assets": list(references.values())})
