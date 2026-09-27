"""Export the real Script/Tyrano formats with fixed, visibly synthetic media."""

from __future__ import annotations

import hashlib
import io
import zipfile
from pathlib import Path

from packages.contracts.m3 import NarrativeResult
from packages.narrative.script_conversion import narrative_to_script, stable_id
from packages.tyrano_export import compile_bundle, demo_content, validate_bundle
from packages.tyrano_export.compiler import canonical_json


def _hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _safe_path(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Debug export path leaves its output directory.")
    return path


def _write(root: Path, relative: str, data: bytes) -> None:
    path = _safe_path(root, relative)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file() and path.read_bytes() == data:
        return
    temporary = _safe_path(root, relative + ".tmp")
    temporary.write_bytes(data)
    temporary.replace(path)


def export_debug_chapter(narrative: NarrativeResult, snapshot: dict, output: Path) -> dict:
    """Save dialogue, staging and a validated source bundle without media inference.

    ``player_directory`` contains the same compiler-owned launcher as production.
    Serve its ``tyrano/`` URLs from the separately installed engine when playing;
    the engine and any production database/build are untouched here.
    Returned paths are relative to ``output`` and every written file has a hash.
    """
    narrative = NarrativeResult.model_validate(narrative)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cast = {row.get("id", row["result"]["id"]): row["result"]
            for row in snapshot["characters"]}
    cast.update({row.id: row.model_dump(mode="json") for row in narrative.supporting_characters})
    used = {cid for scene in narrative.scenes for cid in scene.plan.character_ids}
    _, fixtures = demo_content()
    references, assets, requirements = {}, {}, []

    def image(kind, target, data, descriptor):
        requirement = "m3_image" if kind == "character" else "m3_background"
        aid = stable_id(kind, target)
        references[requirement, target] = {
            "id": aid, "kind": kind, "artifact_id": "debug-placeholder-" + aid,
            "filename": aid + ".png", "sha256": _hash(data)}
        assets[aid] = data
        requirements.append({"kind": requirement, "target_id": target, "descriptor": descriptor,
                             "resolution": "fixed_test_image", "debug_asset_id": aid})

    for index, (cid, character) in enumerate(cast.items()):
        if cid not in used:
            continue
        image("character", cid, fixtures["aki_sprite" if index % 2 == 0 else "ren_sprite"],
              {"character_id": cid, "character_result": character})
        requirements.append({"kind": "m3_voice", "target_id": cid,
                             "descriptor": {"character_id": cid, "character_result": character,
                                            "reference_text": character.get("selfIntroduction")},
                             "resolution": "not_generated"})
    for location in narrative.locations:
        image("background", location.id, fixtures["station"],
              {"location": location.model_dump(mode="json")})
    for scene in narrative.scenes:
        for utterance in scene.utterances:
            if utterance.speaker_id:
                requirements.append({"kind": "m3_voice_clone", "target_id": utterance.id,
                                     "descriptor": {"character_id": utterance.speaker_id,
                                                    "character_result": cast[utterance.speaker_id],
                                                    "dialogue_text": utterance.spoken_text,
                                                    "voice_emotion": utterance.voice_emotion,
                                                    "delivery": utterance.delivery},
                                     "resolution": "not_generated"})
    narrative_bytes = canonical_json(narrative.model_dump(mode="json"))
    script_id = f"debug-chapter-{narrative.chapter_number:03d}-{_hash(narrative_bytes)[:24]}"
    script = narrative_to_script(narrative, snapshot, references, script_id=script_id,
                                 allow_missing_audio=True)
    public_approval = {"schema_version": 1, "approval_id": snapshot.get("id"),
                       "world": snapshot["world"]["result"],
                       "characters": [row["result"] for row in snapshot["characters"]],
                       "relationships": snapshot.get("relationships", {}).get("result")}
    documents = {"approval.json": canonical_json(public_approval), "narrative.json": narrative_bytes,
                 **{f"sources/{scene.id}.txt": scene.raw_text.encode("utf-8")
                    for scene in narrative.scenes}}
    bundle = compile_bundle(script, assets, documents=documents)
    manifest = validate_bundle(bundle, script, assets, documents=documents)
    bundle_hash = _hash(bundle)
    # Different revisions get independent directories, so stale files never join a player.
    player_directory = f"player/{bundle_hash[:24]}"
    files = {"script.json": canonical_json(script.model_dump(mode="json")),
             "tyrano-source.zip": bundle, **documents,
             "asset-requirements.json": canonical_json({
                 "schema_version": 1, "execution_mode": "text_only",
                 "placeholder_assets_are_production_assets": False, "requirements": requirements})}
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        for entry in archive.infolist():
            files[f"{player_directory}/{entry.filename}"] = archive.read(entry)
    for relative, data in files.items():
        _write(output, relative, data)
    return {"schema_version": 1, "execution_mode": "text_only",
            "chapter_number": narrative.chapter_number, "script_id": script.id,
            "quality_acceptance": "not_evaluated", "engine_included": False,
            "script_path": "script.json", "bundle_path": "tyrano-source.zip",
            "player_directory": player_directory,
            "player_entrypoint": f"{player_directory}/index.html",
            "bundle_sha256": bundle_hash, "bundle_manifest": manifest,
            "files": {path: {"bytes": len(data), "sha256": _hash(data)}
                      for path, data in sorted(files.items())}}
