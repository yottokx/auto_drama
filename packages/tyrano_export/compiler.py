"""Reproducible compilation with package-owned player assets and no model calls."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile

from packages.contracts import Script
from packages.contracts.script import portrait_key

from .player import player_files
from .portrait import _silhouette
from .presentation import BACKGROUND, PORTRAIT_Z_INDEX, script_portrait_layouts

# Entering, leaving and changing an event CG: the previous picture fades over the next one.
# Only those lines carry it. The player hides the message window for any anchor with an
# effect, whether or not the picture changes, so the other lines inside a CG must cut.
EVENT_CG_TRANSITION = {"visual": "dissolve", "duration_ms": 500}
ASSET_FOLDERS = {"background": "bgimage", "character": "fgimage", "audio": "sound", "music": "bgm",
                 "event_cg": "cgimage"}


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def event_cg_frames(script: Script) -> dict[str, dict]:
    """Resolve adopted half-open intervals without changing narrative directions."""
    positions = {value.id: index for index, value in enumerate(script.utterances)}
    frames = {}
    for segment in script.event_cg_segments:
        start = positions[segment.start_utterance_id]
        end = positions[segment.end_utterance_id] if segment.end_utterance_id else len(positions)
        variants = {value.utterance_id: value for value in segment.variants}
        asset_id, variant_id = segment.base_asset_id, None
        for utterance in script.utterances[start:end]:
            if utterance.id in variants:
                variant = variants[utterance.id]
                asset_id, variant_id = variant.asset_id, variant.id
            frames[utterance.id] = {"segment_id": segment.id, "variant_id": variant_id,
                                    "asset_id": asset_id}
    return frames


def stage_anchors(script: Script) -> dict[str, dict]:
    """Include CG lines so failed images and restored saves use current stage state."""
    anchors = {value.utterance_id: value.model_dump(mode="json") for value in script.scene_transitions}
    if script.event_cg_segments:
        lines = set(event_cg_frames(script))
        lines.update(value.end_utterance_id for value in script.event_cg_segments if value.end_utterance_id)
        for utterance in script.utterances:
            if utterance.id in lines and utterance.id not in anchors:
                anchors[utterance.id] = {
                    "id": "cg:" + utterance.id, "utterance_id": utterance.id,
                    "visual": "cut", "duration_ms": 0,
                    "music_fade_out_ms": 0, "music_fade_in_ms": 0,
                }
        # Where the picture changes it crossfades. A CG that starts or ends on a scene entry
        # shares that entry's anchor: its own effect is kept, an instant one crossfades.
        changes = {value.start_utterance_id for value in script.event_cg_segments}
        changes.update(value.end_utterance_id for value in script.event_cg_segments
                       if value.end_utterance_id)
        changes.update(variant.utterance_id for value in script.event_cg_segments
                       for variant in value.variants)
        for utterance_id in changes:
            anchor = anchors[utterance_id]
            if anchor["visual"] not in {"fade", "dissolve"} or not anchor["duration_ms"]:
                anchors[utterance_id] = {**anchor, **EVENT_CG_TRANSITION}
    return anchors


def event_cg_after_anchors(script: Script) -> set[str]:
    frames = event_cg_frames(script)
    return {value.utterance_id for value in script.directions
            if value.utterance_id in frames and value.timing == "after"
            and value.kind in {"background", "enter", "exit", "position", "outfit", "expression"}}


class _Wardrobe:
    """Which portrait each character currently shows: the look an entrance decides, and the face.

    A face has a picture only where one was made for that look. Without one the look's own
    portrait is shown, so a script may ask for any expression whatever pictures exist.
    """

    def __init__(self, script: Script, portraits: dict):
        assets = {value.id: value for value in script.assets}
        self.portraits = portraits
        self.files = {portrait_key(character.id, outfit_id): assets[asset_id].filename
                      for character in script.characters if character.image_asset_id
                      for outfit_id, asset_id in ((None, character.image_asset_id),
                                                  *((value.id, value.image_asset_id) for value in character.outfits))}
        self.face_files = {(character.id, value.outfit_id, value.expression): assets[value.image_asset_id].filename
                           for character in script.characters for value in character.expressions}
        self.worn, self.slots, self.faces = {}, {}, {}

    def enter(self, direction) -> None:
        if direction.kind in {"enter", "outfit"}:
            self.worn[direction.character_id] = direction.outfit_id
        if direction.kind == "enter":
            # Everyone comes on in their ordinary face; the scene's annotations start from there.
            self.faces.pop(direction.character_id, None)
        elif direction.kind == "expression":
            self.faces[direction.character_id] = direction.expression
        if direction.kind in {"enter", "position"}:
            self.slots[direction.character_id] = direction.position
        elif direction.kind == "exit":
            self.slots.pop(direction.character_id, None)

    def look(self, character_id: str):
        worn = self.worn.get(character_id)
        key = portrait_key(character_id, worn)
        face = (character_id, None if worn == "default" else worn, self.faces.get(character_id))
        # The face replaces only the picture: it stands where its look stands.
        return self.face_files.get(face, self.files[key]), self.portraits[key]

    def storage_with(self, character_id: str, expression: str | None) -> str:
        """The picture the current look would show with another face."""
        worn = self.worn.get(character_id)
        face = (character_id, None if worn == "default" else worn, expression)
        return self.face_files.get(face, self.files[portrait_key(character_id, worn)])

    def shown(self, character_id: str, z_index: int) -> dict:
        """The stage record of a visible character in its current look and slot."""
        storage, portrait = self.look(character_id)
        return {"id": character_id, "storage": storage,
                "left": portrait.left(self.slots[character_id]), "top": portrait.top,
                "width": portrait.width, "height": portrait.height, "z_index": z_index}


def scene_stage_data(script: Script, images: dict[str, bytes]) -> dict:
    """Compile typed scene-entry directions into complete, immutable stage targets.

    The narrative converter resets all portraits before each scene. Folding the
    entry directions first lets the player reconcile the final cast instead of
    briefly hiding and recreating characters who remain in the same position.
    Mid-scene directions still update the compiler's state and run normally.
    """
    assets = {value.id: value for value in script.assets}
    wardrobe = _Wardrobe(script, script_portrait_layouts(script, images))
    specs = stage_anchors(script)
    cg_frames = event_cg_frames(script)
    cg_after = event_cg_after_anchors(script)
    music = {value.utterance_id: value.id for value in script.music_cues}
    directions = {}
    for value in script.directions:
        directions.setdefault((value.utterance_id, value.timing), []).append(value)
    background, cast, scenes = None, {}, []

    def apply(direction):
        nonlocal background
        if direction.kind == "background":
            background = {"storage": assets[direction.asset_id].filename,
                          "position": BACKGROUND["position"] if script.portrait_baseline else ""}
        elif direction.kind == "exit":
            wardrobe.enter(direction)
            cast.pop(direction.character_id, None)
        elif direction.kind in {"enter", "position", "outfit", "expression"}:
            if direction.kind != "enter" and direction.character_id not in cast:
                if direction.kind in {"outfit", "expression"}:
                    wardrobe.enter(direction)
                return
            wardrobe.enter(direction)
            cast[direction.character_id] = wardrobe.shown(
                direction.character_id, PORTRAIT_Z_INDEX if script.portrait_baseline else 1)

    def snapshot(utterance_id, spec):
        target = {**spec, "music_cue_id": music.get(utterance_id) if spec.get("timing") != "after" else None,
                  "background": dict(background) if background else None,
                  "characters": [dict(cast[cid]) for cid in sorted(cast)]}
        if script.event_cg_segments:
            frame = cg_frames.get(utterance_id)
            target["event_cg"] = ({**frame, "storage": assets[frame["asset_id"]].filename}
                                  if frame else None)
        scenes.append(target)

    for utterance in script.utterances:
        for timing in ("before", "start"):
            for direction in directions.get((utterance.id, timing), []):
                apply(direction)
        if utterance.id in specs:
            snapshot(utterance.id, specs[utterance.id])
        for direction in directions.get((utterance.id, "after"), []):
            apply(direction)
        if utterance.id in cg_after:
            snapshot(utterance.id, {"id": "cg-after:" + utterance.id, "utterance_id": utterance.id,
                                    "timing": "after", "visual": "cut", "duration_ms": 0,
                                    "music_fade_out_ms": 0, "music_fade_in_ms": 0})
    return {"schema_version": 1, "scenes": scenes}


def compile_scenario(script: Script, images: dict[str, bytes]) -> str:
    """Stage tags only. Names and prose are looked up in script.json by utterance ID,

    so no author text enters the scenario or a tag parameter.
    """
    script = Script.model_validate(script.model_dump(mode="json"))
    assets = {value.id: value for value in script.assets}
    characters = {value.id: value for value in script.characters}
    wardrobe = _Wardrobe(script, script_portrait_layouts(script, images))
    music_by_anchor = {cue.utterance_id: cue for cue in script.music_cues}
    transitions_by_anchor = stage_anchors(script)
    scene_entries = {cue.utterance_id for cue in script.scene_transitions}
    cg_frames = event_cg_frames(script)
    cg_after = event_cg_after_anchors(script)
    directions_by_anchor: dict[tuple[str, str], list] = {}
    for direction in script.directions:
        directions_by_anchor.setdefault((direction.utterance_id, direction.timing), []).append(direction)

    def direction_lines(utterance_id: str, timing: str) -> list[str]:
        result = []
        for direction in directions_by_anchor.get((utterance_id, timing), []):
            visible = getattr(direction, "character_id", None) in wardrobe.slots
            before = wardrobe.look(direction.character_id)[1] if direction.kind == "outfit" else None
            shown = wardrobe.look(direction.character_id)[0] if direction.kind == "expression" else None
            wardrobe.enter(direction)
            if direction.kind in {"background", "enter", "exit", "position", "outfit", "expression"} and (
                    utterance_id in cg_frames or
                    (utterance_id in transitions_by_anchor and timing in {"before", "start"})):
                continue
            if (utterance_id in scene_entries and timing in {"before", "start"}
                    and direction.kind == "blackout"):
                continue
            match direction.kind:
                case "background":
                    filename = assets[direction.asset_id].filename
                    position = f' position="{BACKGROUND["position"]}"' if script.portrait_baseline is not None else ""
                    result.append(f'[bg storage="{filename}" time="{direction.duration_ms}"{position}]')
                case "enter":
                    storage, portrait = wardrobe.look(direction.character_id)
                    z_index = f' zindex="{PORTRAIT_Z_INDEX}"' if script.portrait_baseline is not None else ""
                    if characters[direction.character_id].outfits or characters[direction.character_id].expressions:
                        # A hidden character only records the image its next entrance shows.
                        result.append(f'[chara_mod name="ad_{direction.character_id}" storage="{storage}" time="0"]')
                    result.append(
                        f'[chara_show name="ad_{direction.character_id}" '
                        f'left="{portrait.left(direction.position)}" top="{portrait.top}" '
                        f'width="{portrait.width}" height="{portrait.height}" '
                        f'time="{direction.duration_ms}"' + z_index + ']'
                    )
                case "exit":
                    result.append(f'[chara_hide name="ad_{direction.character_id}" time="{direction.duration_ms}"]')
                case "position":
                    _, portrait = wardrobe.look(direction.character_id)
                    result.append(
                        f'[chara_move name="ad_{direction.character_id}" '
                        f'left="{portrait.left(direction.position)}" time="{direction.duration_ms}" anim="true"]'
                    )
                case "outfit":
                    storage, portrait = wardrobe.look(direction.character_id)
                    name = f"ad_{direction.character_id}"
                    if not visible:
                        result.append(f'[chara_mod name="{name}" storage="{storage}" time="0"]')
                        continue
                    result.append(f'[chara_mod name="{name}" storage="{storage}" time="{direction.duration_ms}"]')
                    if portrait != before:
                        # Another look has its own silhouette; keep it standing on the same spot.
                        result.append(
                            f'[chara_move name="{name}" left="{portrait.left(wardrobe.slots[direction.character_id])}" '
                            f'top="{portrait.top}" width="{portrait.width}" height="{portrait.height}" '
                            'time="0" anim="true"]')
                case "expression":
                    storage, _ = wardrobe.look(direction.character_id)
                    # A face without a picture of its own changes nothing on screen.
                    if storage != shown:
                        # A hidden character only records the image its next appearance shows; a
                        # visible one changes without holding back the line the face belongs to.
                        # The new face fades in over the old picture: fading both at once would
                        # let the background show through the whole body, which has not changed.
                        result.append(
                            f'[chara_mod name="ad_{direction.character_id}" storage="{storage}" '
                            + (f'time="{direction.duration_ms}" cross="false" wait="false"]' if visible
                               else 'time="0"]'))
                case "focus":
                    result.append(f'[chara_ptext name="ad_{direction.character_id}"]')
                case "blackout":
                    result.extend([f'[mask color="0x000000" time="{direction.duration_ms}"]',
                                   f'[mask_off time="{direction.duration_ms}"]'])
                case "pause":
                    # The engine's [wait] ignores skip and cannot be cancelled by a jump.
                    result.append(f'[ad_pause time="{direction.duration_ms}"]')
        if timing == "after" and utterance_id in cg_after:
            result.append(f'[ad_transition cue="cg-after:{utterance_id}"]')
        return result

    lines = [
        "; Auto Drama: source-only deterministic export.",
        # A browser may still hold an earlier screen's index.html and scripts for this URL,
        # cached as immutable, while the engine always fetches this scenario fresh. That
        # screen cannot run these tags, so make it reload once to pick up the current one.
        ("[eval exp=\"window.AutoDramaPlayer||sessionStorage.getItem('ad_reload'+location.pathname)||"
         "(sessionStorage.setItem('ad_reload'+location.pathname,1),location.reload())\"]"),
        '[chara_config pos_mode="false" talk_focus="brightness"]',
        '[layopt layer="message0" visible="false"]',
    ]
    for character in sorted(script.characters, key=lambda value: value.id):
        if character.image_asset_id:
            lines.append(f'[chara_new name="ad_{character.id}" storage="{assets[character.image_asset_id].filename}"]')
    lines.append("[ad_gate]")
    for utterance in script.utterances:
        lines.extend([f"*utterance_{utterance.id}", f'[ad_line id="{utterance.id}"]'])
        if utterance.id in transitions_by_anchor:
            lines.append(f'[ad_transition cue="{transitions_by_anchor[utterance.id]["id"]}"]')
        elif utterance.id in music_by_anchor:
            lines.append(f'[ad_music cue="{music_by_anchor[utterance.id].id}"]')
        lines.extend(direction_lines(utterance.id, "before"))
        character = characters.get(utterance.speaker_id)
        # A voice from elsewhere lights up no one on stage; the player dims its speaker if present.
        lines.append(f'[chara_ptext name="ad_{character.id}"]'
                     if character and character.image_asset_id and utterance.remote is None
                     else '[chara_ptext name=""]')
        lines.extend(direction_lines(utterance.id, "start"))
        lines.append(f'[ad_say id="{utterance.id}"]')
        lines.extend(direction_lines(utterance.id, "after"))
        lines.append("[ad_wait]")
    lines.extend(["*auto_drama_chapter_end", "[ad_end]", "[s]"])
    return "\n".join(lines) + "\n"


def _stage_walk(script: Script, images: dict[str, bytes]) -> dict:
    """Follow the stage through every line once, as the player shows it.

    Each line is seen twice: as it begins, before its own directions, and while its text
    shows, after its ``before`` and ``start`` directions. Jumps restore the first; the
    adjustment preview shows the second. Both come from this one pass so they cannot
    drift apart, nor from what the scenario does.

    A face counts as shown once some moment of a line shows it on stage: a face set
    while its character is away, or under an event CG and changed again before the CG
    ends, or that has no picture of its own, never reaches the screen.
    """
    assets = {value.id: value for value in script.assets}
    characters = {value.id: value for value in script.characters}
    wardrobe = _Wardrobe(script, script_portrait_layouts(script, images))
    cg_frames = event_cg_frames(script)
    cues = {value.utterance_id: value for value in script.music_cues}
    directions: dict[str, list] = {}
    for value in script.directions:
        directions.setdefault(value.utterance_id, []).append(value)
    background, cast, music, frame = None, {}, None, None
    entries, shown, faces_seen, unseen_faces = {}, {}, set(), {}

    def stage(cg):
        return {
            "background": dict(background) if background else None,
            "characters": [dict(cast[key]) for key in sorted(cast)],
            "event_cg": {**cg, "storage": assets[cg["asset_id"]].filename} if cg else None,
            "music_cue_id": music,
        }

    def look_faces(utterance_id):
        if utterance_id in cg_frames:
            return
        for character_id, (identifier, before) in list(unseen_faces.items()):
            if character_id in cast and (wardrobe.look(character_id)[0]
                                         != wardrobe.storage_with(character_id, before)):
                faces_seen.add(identifier)
                del unseen_faces[character_id]

    for utterance in script.utterances:
        entries[utterance.id] = stage(frame)
        cue = cues.get(utterance.id)
        if cue and cue.action in {"play", "stop"}:
            music = cue.id if cue.action == "play" else None
        lit = None
        for timing in ("before", "start", "after"):
            for direction in (value for value in directions.get(utterance.id, []) if value.timing == timing):
                if direction.kind == "expression":
                    unseen_faces[direction.character_id] = (direction.id,
                                                            wardrobe.faces.get(direction.character_id))
                elif direction.kind == "enter":
                    # Everyone comes on in their ordinary face, so a face set before is gone.
                    unseen_faces.pop(direction.character_id, None)
                elif direction.kind == "focus" and timing == "start":
                    lit = direction.character_id
                if direction.kind == "background":
                    background = {"storage": assets[direction.asset_id].filename,
                                  "position": BACKGROUND["position"] if script.portrait_baseline else ""}
                elif direction.kind == "exit":
                    wardrobe.enter(direction)
                    cast.pop(direction.character_id, None)
                elif direction.kind in {"outfit", "expression"} and direction.character_id not in cast:
                    wardrobe.enter(direction)
                elif direction.kind in {"enter", "outfit", "expression"} or (
                        direction.kind == "position" and direction.character_id in cast):
                    wardrobe.enter(direction)
                    cast[direction.character_id] = wardrobe.shown(
                        direction.character_id, PORTRAIT_Z_INDEX if script.portrait_baseline else 1)
            if timing == "before":
                # The scenario lights the speaker here; a voice from elsewhere lights no one.
                speaker = characters.get(utterance.speaker_id)
                lit = speaker.id if speaker and speaker.image_asset_id and utterance.remote is None else None
            elif timing == "start":
                shown[utterance.id] = {**stage(cg_frames.get(utterance.id)),
                                       "lit": lit if lit in cast else None}
            if timing != "before":
                look_faces(utterance.id)
        frame = cg_frames.get(utterance.id)
    faces = {value.id for value in script.directions if value.kind == "expression"}
    return {"entries": entries, "shown": shown, "unseen_faces": faces - faces_seen}


def entry_states(script: Script, images: dict[str, bytes]) -> dict:
    """The stage and music in effect when each line begins, before its own directions.

    A jump restores this and then replays the line from its label, so entrances
    and scene transitions that belong to the line still play as written.
    """
    return {"schema_version": 1, "entries": _stage_walk(script, images)["entries"]}


def line_stages(script: Script, images: dict[str, bytes]) -> dict:
    """What each line looks like while its text shows, for the adjustment preview.

    ``lit`` is the one person on stage left bright; everyone else is dimmed. Directions
    listed in ``unseen`` change nothing on screen: faces that are never shown, and
    blackouts the scene transition of their line replaces.
    """
    walk = _stage_walk(script, images)
    entries = {cue.utterance_id for cue in script.scene_transitions}
    blackouts = {value.id for value in script.directions if value.kind == "blackout"
                 and value.utterance_id in entries and value.timing in {"before", "start"}}
    return {"lines": walk["shown"], "unseen": sorted(walk["unseen_faces"] | blackouts)}


# The frame of a remote voice shows the speaker from the shoulders up, 3:4: from 0.45 face heights
# above the face to one below the chin. Without a known face, the top of the silhouette is used.
BUST_ABOVE, BUST_BELOW, BUST_ASPECT = 0.45, 1.0, 3 / 4
SILHOUETTE_BUST_WIDTH = 0.55


def _bust(face, image: bytes) -> list[int]:
    if face is not None:
        face_height = face.bottom - face.top
        top = face.top - BUST_ABOVE * face_height
        crop_height = face_height * (1 + BUST_ABOVE + BUST_BELOW)
        center = (face.left + face.right) / 2
    else:
        subject = _silhouette(image, "auto")
        crop_height = subject.visible_width * SILHOUETTE_BUST_WIDTH / BUST_ASPECT
        top = subject.top - 0.03 * crop_height
        center = (subject.left + subject.right) / 2
    crop_width = crop_height * BUST_ASPECT
    return [round(center - crop_width / 2), round(top), round(crop_width), round(crop_height)]


def remote_frames(script: Script, images: dict[str, bytes]) -> dict:
    """What each remote voice's frame shows: its speaker's current look, with the line's face.

    The look follows the speaker's entrances and changes even while they are off stage, so a
    person only heard in a scene is framed in the look that scene gives them.
    """
    assets = {value.id: value for value in script.assets}
    characters = {value.id: value for value in script.characters}
    wardrobe = _Wardrobe(script, {})
    directions: dict[str, list] = {}
    for value in script.directions:
        directions.setdefault(value.utterance_id, []).append(value)
    lines = {}
    for utterance in script.utterances:
        for timing in ("before", "start", "after"):
            if timing == "after" and utterance.remote is not None:
                character = characters[utterance.speaker_id]
                worn = wardrobe.worn.get(character.id)
                look = None if worn in (None, "default") else next(
                    (value for value in character.outfits if value.id == worn), None)
                asset_id = look.image_asset_id if look else character.image_asset_id
                if asset_id is not None:
                    image = images[asset_id]
                    width, height = _png_size(image)
                    face_file = wardrobe.face_files.get((character.id, look.id if look else None,
                                                         utterance.remote.expression))
                    lines[utterance.id] = {
                        "speaker": character.id,
                        "source": utterance.remote.source,
                        "channel": utterance.remote.channel,
                        "storage": face_file or assets[asset_id].filename,
                        "width": width, "height": height,
                        "crop": _bust(look.face if look else character.face, image),
                        "on_stage": character.id in wardrobe.slots,
                    }
            for direction in directions.get(utterance.id, []):
                if direction.timing == timing and direction.kind in {"enter", "exit", "outfit", "expression"}:
                    wardrobe.enter(direction)
    return {"schema_version": 1, "lines": lines}


def _png_size(image: bytes) -> tuple[int, int]:
    if image[:8] != b"\x89PNG\r\n\x1a\n" or image[12:16] != b"IHDR":
        raise ValueError("a portrait must be a PNG")
    return int.from_bytes(image[16:20], "big"), int.from_bytes(image[20:24], "big")


def player_overlay(script: Script, images: dict[str, bytes], script_bytes: bytes) -> dict[str, bytes]:
    """Everything the viewing screen needs besides script.json, assets and documents.

    The live route regenerates these from a published ZIP's own script and images,
    so chapters published by an earlier player still open in the current one.
    """
    files = {
        "data/scenario/first.ks": compile_scenario(script, images).encode("utf-8"),
        # Installed projects may call make.ks during engine initialization.
        "data/scenario/make.ks": b"[return]\n",
        "data/others/auto_drama_states.json": canonical_json(entry_states(script, images)),
    }
    if script.scene_transitions or script.event_cg_segments:
        files["data/others/auto_drama_stages.json"] = canonical_json(scene_stage_data(script, images))
    if any(utterance.remote is not None for utterance in script.utterances):
        files["data/others/auto_drama_remote.json"] = canonical_json(remote_frames(script, images))
    files.update(player_files(script_bytes))
    return files


def _bundle_files(
    script: Script, assets: dict[str, bytes], documents: dict[str, bytes] | None = None
) -> dict[str, bytes]:
    script = Script.model_validate(script.model_dump(mode="json"))
    if set(assets) != {asset.id for asset in script.assets}:
        raise ValueError("provided asset IDs must exactly match the script manifest")
    files = {"script.json": canonical_json(script.model_dump(mode="json"))}
    files.update(player_overlay(script, assets, files["script.json"]))
    for path, content in (documents or {}).items():
        if path not in {"approval.json", "narrative.json", "chapter-manifest.json",
                        "staging-normalization.json"} and not re.fullmatch(
            r"sources/[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}\.txt", path
        ):
            raise ValueError("unsupported public document path")
        if not isinstance(content, bytes) or not content:
            raise ValueError("public documents must contain nonempty bytes")
        # Callers provide public, sanitized records. Credentials and runtime
        # configuration must never be supplied to this export boundary.
        content.decode("utf-8")
        if path.endswith(".json"):
            json.loads(content)
        files[path] = content
    for asset in script.assets:
        content = assets[asset.id]
        if not isinstance(content, bytes) or not content:
            raise ValueError(f"asset must contain complete nonempty bytes: {asset.id}")
        if hashlib.sha256(content).hexdigest() != asset.sha256:
            raise ValueError(f"asset hash mismatch: {asset.id}")
        files[f"data/{ASSET_FOLDERS[asset.kind]}/{asset.filename}"] = content
    manifest = {
        "schema_version": 1,
        "format": "auto-drama.tyrano-source",
        "script_id": script.id,
        "engine_included": False,
        "entrypoint": "data/scenario/first.ks",
        "assets": [asset.model_dump(mode="json") for asset in script.assets],
        "files": {
            path: {"bytes": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            for path, content in sorted(files.items())
        },
    }
    files["manifest.json"] = canonical_json(manifest)
    return files


def compile_bundle(
    script: Script, assets: dict[str, bytes], *, documents: dict[str, bytes] | None = None
) -> bytes:
    """Produce a reproducible ZIP. Assets are keyed by logical ID, not artifact ID."""
    stream = io.BytesIO()
    # STORED avoids varying output across zlib implementations/releases.
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for path, content in sorted(_bundle_files(script, assets, documents).items()):
            info = zipfile.ZipInfo(path, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = 0o100644 << 16
            archive.writestr(info, content)
    return stream.getvalue()


def validate_bundle(
    data: bytes,
    expected_script: Script,
    assets: dict[str, bytes],
    *,
    documents: dict[str, bytes] | None = None,
) -> dict:
    """Accept only the exact expected output, including archive paths and metadata.

    No untrusted ZIP entry is extracted or decompressed. Trailing bytes, duplicate
    paths, extra files, truncated streams and mismatched manifests all fail.
    """
    expected = compile_bundle(expected_script, assets, documents=documents)
    if data != expected:
        raise ValueError("result is not the deterministic bundle for the adopted inputs")
    return json.loads(_bundle_files(expected_script, assets, documents)["manifest.json"])
