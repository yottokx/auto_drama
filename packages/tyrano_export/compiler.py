"""Reproducible compilation with package-owned player assets and no model calls."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile

from packages.contracts import Script

from .player import player_files
from .presentation import BACKGROUND, PORTRAIT_Z_INDEX, script_portrait_layouts

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
    return anchors


def event_cg_after_anchors(script: Script) -> set[str]:
    frames = event_cg_frames(script)
    return {value.utterance_id for value in script.directions
            if value.utterance_id in frames and value.timing == "after"
            and value.kind in {"background", "enter", "exit", "position"}}


def scene_stage_data(script: Script, images: dict[str, bytes]) -> dict:
    """Compile typed scene-entry directions into complete, immutable stage targets.

    The narrative converter resets all portraits before each scene. Folding the
    entry directions first lets the player reconcile the final cast instead of
    briefly hiding and recreating characters who remain in the same position.
    Mid-scene directions still update the compiler's state and run normally.
    """
    assets = {value.id: value for value in script.assets}
    characters = {value.id: value for value in script.characters}
    portraits = script_portrait_layouts(script, images)
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
            cast.pop(direction.character_id, None)
        elif direction.kind in {"enter", "position"}:
            if direction.kind == "position" and direction.character_id not in cast:
                return
            portrait = portraits[direction.character_id]
            character = characters[direction.character_id]
            cast[direction.character_id] = {
                "id": direction.character_id, "storage": assets[character.image_asset_id].filename,
                "left": portrait.left(direction.position), "top": portrait.top,
                "width": portrait.width, "height": portrait.height,
                "z_index": PORTRAIT_Z_INDEX if script.portrait_baseline else 1,
            }

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
    portraits = script_portrait_layouts(script, images)
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
            if direction.kind in {"background", "enter", "exit", "position"} and (
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
                    portrait = portraits[direction.character_id]
                    z_index = f' zindex="{PORTRAIT_Z_INDEX}"' if script.portrait_baseline is not None else ""
                    result.append(
                        f'[chara_show name="ad_{direction.character_id}" '
                        f'left="{portrait.left(direction.position)}" top="{portrait.top}" '
                        f'width="{portrait.width}" height="{portrait.height}" '
                        f'time="{direction.duration_ms}"' + z_index + ']'
                    )
                case "exit":
                    result.append(f'[chara_hide name="ad_{direction.character_id}" time="{direction.duration_ms}"]')
                case "position":
                    portrait = portraits[direction.character_id]
                    result.append(
                        f'[chara_move name="ad_{direction.character_id}" '
                        f'left="{portrait.left(direction.position)}" time="{direction.duration_ms}" anim="true"]'
                    )
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
        lines.append(f'[chara_ptext name="ad_{character.id}"]' if character and character.image_asset_id
                     else '[chara_ptext name=""]')
        lines.extend(direction_lines(utterance.id, "start"))
        lines.append(f'[ad_say id="{utterance.id}"]')
        lines.extend(direction_lines(utterance.id, "after"))
        lines.append("[ad_wait]")
    lines.extend(["*auto_drama_chapter_end", "[ad_end]", "[s]"])
    return "\n".join(lines) + "\n"


def entry_states(script: Script, images: dict[str, bytes]) -> dict:
    """The stage and music in effect when each line begins, before its own directions.

    A jump restores this and then replays the line from its label, so entrances
    and scene transitions that belong to the line still play as written.
    """
    assets = {value.id: value for value in script.assets}
    characters = {value.id: value for value in script.characters}
    portraits = script_portrait_layouts(script, images)
    cg_frames = event_cg_frames(script)
    cues = {value.utterance_id: value for value in script.music_cues}
    directions: dict[str, list] = {}
    for value in script.directions:
        directions.setdefault(value.utterance_id, []).append(value)
    background, cast, music, frame, entries = None, {}, None, None, {}
    for utterance in script.utterances:
        entries[utterance.id] = {
            "background": dict(background) if background else None,
            "characters": [dict(cast[key]) for key in sorted(cast)],
            "event_cg": {**frame, "storage": assets[frame["asset_id"]].filename} if frame else None,
            "music_cue_id": music,
        }
        cue = cues.get(utterance.id)
        if cue and cue.action in {"play", "stop"}:
            music = cue.id if cue.action == "play" else None
        for timing in ("before", "start", "after"):
            for direction in (value for value in directions.get(utterance.id, []) if value.timing == timing):
                if direction.kind == "background":
                    background = {"storage": assets[direction.asset_id].filename,
                                  "position": BACKGROUND["position"] if script.portrait_baseline else ""}
                elif direction.kind == "exit":
                    cast.pop(direction.character_id, None)
                elif direction.kind == "enter" or (direction.kind == "position" and direction.character_id in cast):
                    portrait = portraits[direction.character_id]
                    cast[direction.character_id] = {
                        "id": direction.character_id,
                        "storage": assets[characters[direction.character_id].image_asset_id].filename,
                        "left": portrait.left(direction.position), "top": portrait.top,
                        "width": portrait.width, "height": portrait.height,
                        "z_index": PORTRAIT_Z_INDEX if script.portrait_baseline else 1,
                    }
        frame = cg_frames.get(utterance.id)
    return {"schema_version": 1, "entries": entries}


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
