"""Reproducible compilation with package-owned player assets and no model calls."""

from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile

from packages.contracts import Script

from .player import player_files
from .presentation import BACKGROUND, MESSAGE_WINDOW, PORTRAIT_Z_INDEX, script_portrait_layouts

ASSET_FOLDERS = {"background": "bgimage", "character": "fgimage", "audio": "sound", "music": "bgm"}

# Compiler-owned code only. Tyrano's story text uses a text sink, but its backlog
# uses HTML; escaping solely the .ks parser would leave an HTML injection there.
BACKLOG_SAFETY = b""""use strict";
(function () {
  const tag = tyrano.plugin.kag.tag.text;
  if (tag.autoDramaSafeBacklog) return;
  const original = tag.pushTextToBackLog;
  const escape = value => String(value).replace(/[&<>"']/g, character =>
    ({"&":"&amp;", "<":"&lt;", ">":"&gt;", '"':"&quot;", "'":"&#39;"}[character]));
  tag.pushTextToBackLog = function (_name, text) {
    // Display names already occur as literal prose. Internal character IDs are
    // used only for focus, and must not become extra visible backlog labels.
    return original.call(this, "", escape(text));
  };
  tag.autoDramaSafeBacklog = true;
})();
"""


def canonical_json(value: dict) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


def literal_text(value: str, *, next_tag: str = "[r]") -> str:
    """Protect all line-prefix syntax and brackets without changing visible text.

    Tyrano trims each source line before parsing, so an owned tag is placed at
    each line's end to preserve trailing spaces. All author content is text,
    including titles and speaker names; none enters a tag parameter.
    """
    lines = value.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    encoded = ["_" + line.replace("\\", "\\\\").replace("[", "\\[") for line in lines]
    return "\n".join(
        line + ("[r]" if i < len(lines) - 1 else next_tag) for i, line in enumerate(encoded)
    )


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
    specs = {value.utterance_id: value for value in script.scene_transitions}
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

    for utterance in script.utterances:
        for timing in ("before", "start"):
            for direction in directions.get((utterance.id, timing), []):
                apply(direction)
        if utterance.id in specs:
            scenes.append({**specs[utterance.id].model_dump(mode="json"),
                           "music_cue_id": music.get(utterance.id),
                           "background": dict(background) if background else None,
                           "characters": [dict(cast[cid]) for cid in sorted(cast)]})
        for direction in directions.get((utterance.id, "after"), []):
            apply(direction)
    return {"schema_version": 1, "scenes": scenes}


def compile_scenario(script: Script, images: dict[str, bytes]) -> str:
    script = Script.model_validate(script.model_dump(mode="json"))
    characters = {value.id: value for value in script.characters}
    assets = {value.id: value for value in script.assets}
    portraits = script_portrait_layouts(script, images)
    window, padding = MESSAGE_WINDOW, MESSAGE_WINDOW["padding"]
    lines = [
        "; Auto Drama M1: source-only deterministic export.",
        '[loadjs storage="auto_drama_backlog.js"]',
        '[chara_config pos_mode="false" talk_focus="brightness"]',
        (
            f'[position layer="message0" left="{window["left"]}" top="{window["top"]}" '
            f'width="{window["width"]}" height="{window["height"]}" '
            f'marginl="{padding["left"]}" margint="{padding["top"]}" '
            f'marginr="{padding["right"]}" marginb="{padding["bottom"]}"]'
        ),
        '[layopt layer="message0" visible="true"]',
        f'[deffont size="{window["font_size"]}"]',
    ]
    for character in sorted(script.characters, key=lambda value: value.id):
        if character.image_asset_id:
            filename = assets[character.image_asset_id].filename
            lines.append(f'[chara_new name="ad_{character.id}" storage="{filename}"]')
    lines.extend([literal_text(script.title, next_tag="[p]"), "[cm]"])
    directions_by_anchor = {}
    music_by_anchor = {cue.utterance_id: cue for cue in script.music_cues}
    transitions_by_anchor = {cue.utterance_id: cue for cue in script.scene_transitions}
    for direction in script.directions:
        directions_by_anchor.setdefault((direction.utterance_id, direction.timing), []).append(
            direction
        )

    def direction_lines(utterance_id: str, timing: str) -> list[str]:
        result = []
        for direction in directions_by_anchor.get((utterance_id, timing), []):
            if (utterance_id in transitions_by_anchor and timing in {"before", "start"}
                    and direction.kind in {"background", "enter", "exit", "position", "blackout"}):
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
                    result.append(
                        f'[chara_hide name="ad_{direction.character_id}" '
                        f'time="{direction.duration_ms}"]'
                    )
                case "position":
                    portrait = portraits[direction.character_id]
                    result.append(
                        f'[chara_move name="ad_{direction.character_id}" '
                        f'left="{portrait.left(direction.position)}" '
                        f'time="{direction.duration_ms}" anim="true"]'
                    )
                case "focus":
                    result.append(f'[chara_ptext name="ad_{direction.character_id}"]')
                case "blackout":
                    result.extend(
                        [
                            f'[mask color="0x000000" time="{direction.duration_ms}"]',
                            f'[mask_off time="{direction.duration_ms}"]',
                        ]
                    )
                case "pause":
                    result.append(f'[wait time="{direction.duration_ms}"]')
        return result

    for utterance in script.utterances:
        lines.extend([f"*utterance_{utterance.id}", "[cm]"])
        if utterance.id in transitions_by_anchor:
            lines.append(f'[ad_transition cue="{transitions_by_anchor[utterance.id].id}"]')
        elif utterance.id in music_by_anchor:
            lines.append(f'[ad_music cue="{music_by_anchor[utterance.id].id}"]')
        lines.extend(direction_lines(utterance.id, "before"))
        character = characters.get(utterance.speaker_id)
        if character:
            if character.image_asset_id:
                lines.append(f'[chara_ptext name="ad_{character.id}"]')
            else:
                lines.append('[chara_ptext name=""]')
            lines.append(literal_text(character.name))
        else:
            lines.append('[chara_ptext name=""]')
        lines.extend(direction_lines(utterance.id, "start"))
        if utterance.audio_asset_id:
            filename = assets[utterance.audio_asset_id].filename
            lines.append(f'[playse storage="{filename}" buf="1"]')
        # [r] is also a terminator that preserves final-line trailing spaces.
        lines.append(literal_text(utterance.display_text))
        if utterance.audio_asset_id:
            lines.append("[wse]")
        lines.extend(direction_lines(utterance.id, "after"))
        lines.append("[p]")
    lines.extend(["*auto_drama_chapter_end", "[cm]", literal_text("この章はここまでです。")])
    # Retain the last scene's loop while the reader waits at the boundary.
    # The owned Next button fades it only when leaving this chapter.
    lines.append("[s]")
    return "\n".join(lines) + "\n"


def _bundle_files(
    script: Script, assets: dict[str, bytes], documents: dict[str, bytes] | None = None
) -> dict[str, bytes]:
    script = Script.model_validate(script.model_dump(mode="json"))
    if set(assets) != {asset.id for asset in script.assets}:
        raise ValueError("provided asset IDs must exactly match the script manifest")
    files = {
        "script.json": canonical_json(script.model_dump(mode="json")),
        "data/scenario/first.ks": compile_scenario(script, assets).encode("utf-8"),
        # Installed projects may call make.ks during engine initialization.
        "data/scenario/make.ks": b"[return]\n",
        "data/others/auto_drama_backlog.js": BACKLOG_SAFETY,
    }
    if script.scene_transitions:
        files["data/others/auto_drama_stages.json"] = canonical_json(scene_stage_data(script, assets))
    files.update(player_files(files["script.json"]))
    for path, content in (documents or {}).items():
        if path not in {"approval.json", "narrative.json", "chapter-manifest.json"} and not re.fullmatch(
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
