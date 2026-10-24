"""Persistent M2 wizard and atomic adoption of local generation jobs."""

from __future__ import annotations

import copy
import json
import secrets
import sqlite3
import wave
import zipfile
import zlib
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from uuid import uuid4

from pydantic import ValidationError

from packages.contracts.m2 import (
    CharacterBrief,
    CharacterResult,
    LegacyCharacterResult,
    M2ProjectInput,
    RelationshipsResult,
    WorldResult,
)
from packages.contracts.outfits import DEFAULT_OUTFIT

from .m2_bundle import validate_bundle
from .service import Coordinator, ServiceError, encode_json, public, required

ROOT = Path(__file__).resolve().parents[2]
GROUPS = {
    "settings": (
        "name",
        "age",
        "gender",
        "role",
        "freeform",
        "settings",
        "selfIntroduction",
        "sampleLines",
    ),
    "appearance": ("appearance", "height_cm", "body_type"),
    "voice": ("voice",),
}


def iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, UTC).isoformat()


def entry(brief: dict) -> dict:
    return {
        "id": brief["id"],
        "input": brief,
        "result": None,
        "locked": {name: False for name in GROUPS},
        "imageArtifactId": None,
        "voiceArtifactId": None,
        "imageUrl": None,
        "voiceUrl": None,
        "voiceReferenceText": None,
        "voiceTests": [],
        "imagePendingChanges": True,
        "voicePendingChanges": True,
        "resultArtifactId": None,
        "resultVersion": None,
        "pendingChanges": True,
    }


def task(kind: str, character_id: str | None = None, scope: str = "all", instruction: str = ""):
    return {
        "kind": kind,
        "character_id": character_id,
        "scope": scope,
        "instruction": instruction,
        "seed": secrets.randbelow(2**31),
    }


def pair_ids(characters: list[dict]) -> set[tuple[str, str]]:
    return set(combinations(sorted(value["id"] for value in characters), 2))


def relationship_state(characters: list[dict]) -> dict:
    return {
        "result": None,
        "artifactId": None,
        "version": None,
        "pendingChanges": len(characters) >= 2,
    }


def normalize_relationship_inputs(values: list[dict], characters: list[dict]) -> list[dict]:
    expected, seen, normalized = pair_ids(characters), set(), []
    for value in values:
        pair = tuple(sorted(value["characterIds"]))
        if pair not in expected or pair in seen:
            raise ServiceError(422, "関係性には作品内の異なる2人を重複なく指定してください。")
        seen.add(pair)
        normalized.append({"characterIds": list(pair), "instruction": value["instruction"]})
    return sorted(normalized, key=lambda value: value["characterIds"])


def validate_relationship_coverage(result: dict, characters: list[dict]) -> dict:
    value = RelationshipsResult.model_validate(result).model_dump(mode="json")
    pairs = [tuple(pair["characterIds"]) for pair in value["pairs"]]
    if len(pairs) != len(set(pairs)) or set(pairs) != pair_ids(characters):
        raise ValueError("relationships must cover exactly every main-character pair")
    return value


def validate_character_result(result: dict) -> dict:
    # Saved pre-extension result artifacts stay verbatim until an explicit regeneration.
    contract = (
        CharacterResult
        if "selfIntroduction" in result or "sampleLines" in result
        else LegacyCharacterResult
    )
    return contract.model_validate(result).model_dump()


class M2Service:
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.db, self.store, self.clock = coordinator.db, coordinator.store, coordinator.clock

    @staticmethod
    def _load(connection: sqlite3.Connection, project_id: str) -> dict:
        row = connection.execute(
            "SELECT state FROM m2_draft WHERE project_id=?", (project_id,)
        ).fetchone()
        if row is None:
            raise ServiceError(404, "M2の作品が見つかりません。")
        state = json.loads(row["state"])
        state.setdefault("appliedInstructions", [])
        draft = state["draft"]
        if draft["step"] == "character-input":
            draft["step"] = "world-input"
        if state.get("wizardStepVersion", 1) < 2:
            # Older approvals ended on character-review. Record the navigation
            # format on the next save so an explicit return to settings sticks.
            if draft["step"] == "character-review" and (
                draft["approved"] or M2Service._has_production(connection, project_id)
            ):
                draft["step"] = "production"
            state["wizardStepVersion"] = 2
        draft.setdefault("relationshipInputs", [])
        # Missing flags identify saved pre-split projects, whose start rules stay intact.
        draft.setdefault("planningRequired", False)
        draft.setdefault("planApproved", False)
        draft.setdefault("relationships", relationship_state(draft["characters"]))
        for character in draft["characters"]:
            character.setdefault("voiceTests", [])
            character.setdefault("outfits", [])
        return state

    @staticmethod
    def _has_production(connection: sqlite3.Connection, project_id: str) -> bool:
        from .project_history import HistoryService

        selected = HistoryService.selection(connection, project_id)
        if selected is not None:
            return selected["production_id"] is not None
        return connection.execute(
            "SELECT 1 FROM m3_production WHERE project_id=? LIMIT 1", (project_id,),
        ).fetchone() is not None

    def _save(self, connection: sqlite3.Connection, project_id: str, state: dict) -> None:
        draft = state["draft"]
        connection.execute(
            "UPDATE m2_draft SET state=?,updated_at=? WHERE project_id=?",
            (json.dumps(state, ensure_ascii=False), self.clock(), project_id),
        )
        world = draft["worldResult"] or draft["worldInput"]
        connection.execute(
            "UPDATE project SET title=?,instructions=?,chapter_count=?,settings_version=? WHERE id=?",
            (
                world["title"] or "新しい物語",
                draft["worldInput"]["prompt"],
                world["chapterCount"],
                draft["revision"],
                project_id,
            ),
        )
        from .project_history import HistoryService

        HistoryService(self.coordinator).finish(connection, project_id)

    @staticmethod
    def _summary(connection: sqlite3.Connection, project_id: str, state: dict) -> dict:
        project = public(required(connection, "project", project_id))
        active = connection.execute(
            "SELECT status FROM job WHERE id=? AND project_id=?", (state["activeJobId"], project_id)
        ).fetchone()
        project["status"] = (
            "ready" if state["draft"]["approved"] else active["status"] if active else "draft"
        )
        project["updated_at"] = iso(
            connection.execute(
                "SELECT updated_at FROM m2_draft WHERE project_id=?", (project_id,)
            ).fetchone()[0]
        )
        return project

    def _view(self, connection: sqlite3.Connection, project_id: str, state: dict) -> dict:
        jobs = [
            public(dict(row))
            for row in connection.execute(
                "SELECT * FROM job WHERE project_id=? ORDER BY created_at,id", (project_id,)
            )
        ]
        project = self._summary(connection, project_id, state)
        active = next((job for job in jobs if job["id"] == state["activeJobId"]), None)
        history = {}
        draft = copy.deepcopy(state["draft"])
        draft["hasProduction"] = self._has_production(connection, project_id)
        draft["activeJobId"] = state["activeJobId"]
        draft["remainingJobCount"] = len(state["queue"]) + (1 if active else 0)
        for character in draft["characters"]:
            for prefix in ("image", "voice"):
                artifact = character[f"{prefix}ArtifactId"]
                character[f"{prefix}Url"] = (
                    f"/api/artifacts/{artifact}/content" if artifact else None
                )
            character["voiceReferenceText"] = self._voice_reference_text(connection, character)
            for trial in character["voiceTests"]:
                trial["url"] = f"/api/artifacts/{trial['artifactId']}/content"
            for outfit in character.setdefault("outfits", []):
                artifact = outfit.get("imageArtifactId")
                outfit["imageUrl"] = f"/api/artifacts/{artifact}/content" if artifact else None
            for picture in character.get("expressions", []):
                picture["imageUrl"] = f"/api/artifacts/{picture['imageArtifactId']}/content"
                # Edited from a portrait that has since been replaced: kept, but no longer shown.
                picture["stale"] = picture["baseImageArtifactId"] != self._look_image(character, picture["outfitId"])
            # Every picture made of this character, rejected ones included, newest first. It is a
            # record of what exists, not part of the draft: restoring an older draft keeps it.
            history[character["id"]] = [
                {"artifactId": row["id"], "url": f"/api/artifacts/{row['id']}/content",
                 "outfitId": DEFAULT_OUTFIT if row["logical_id"] == f"image-{character['id']}"
                 else row["logical_id"][len(f"outfit-{character['id']}-"):]}
                for row in connection.execute(
                    "SELECT id,logical_id FROM artifact WHERE project_id=? AND media_type='image/png' "
                    "AND (logical_id=? OR logical_id LIKE ? ESCAPE '!') ORDER BY created_at DESC,rowid DESC LIMIT 24",
                    (project_id, f"image-{character['id']}", self._history_pattern(character["id"])))]
        from .llm_settings import cast_limit_choice

        return {"project": project, "draft": draft, "jobs": jobs, "imageHistory": history,
                "castLimit": cast_limit_choice(self.coordinator, connection)}

    @staticmethod
    def _history_pattern(character_id: str) -> str:
        escaped = character_id.replace("!", "!!").replace("_", "!_").replace("%", "!%")
        return f"outfit-{escaped}-%"

    def _own_image(self, connection, project_id: str, character: dict, artifact_id: str) -> dict:
        """A picture of this character that an edit may start from or a look may go back to."""
        record = required(connection, "artifact", artifact_id)
        logical = record["logical_id"]
        if (record["project_id"] != project_id or record["media_type"] != "image/png"
                or not (logical == f"image-{character['id']}" or logical.startswith(f"outfit-{character['id']}-"))):
            raise ServiceError(422, "この人物の絵を選んでください。")
        return record

    def projects(self) -> list[dict]:
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            ids = [
                row[0]
                for row in connection.execute(
                    "SELECT project_id FROM m2_draft ORDER BY updated_at DESC,project_id"
                )
            ]
            # Every open editor polls this list, so it reads only what a library entry shows.
            return [
                self._summary(connection, identifier, self._load(connection, identifier))
                for identifier in ids
            ]

    def project(self, project_id: str) -> dict:
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            from .project_history import HistoryService

            HistoryService(self.coordinator).finish(connection, project_id)
            return self._view(connection, project_id, self._load(connection, project_id))

    def create(self, body: M2ProjectInput) -> dict:
        world = body.world.model_dump()
        characters = body.characters or [CharacterBrief(id="character-1")]
        if any(any(character.locked.model_dump().values()) for character in characters):
            raise ServiceError(422, "生成結果を確認してから固定してください。")
        with self.db.transaction() as connection:
            project_id = self.coordinator._create_project(
                connection, world["title"] or "新しい物語", world["prompt"], world["chapterCount"]
            )
            state = {
                "wizardStepVersion": 3,
                "appliedInstructions": [],
                "draft": {
                    "revision": 1,
                    "step": "world-input",
                    "worldInput": world,
                    "worldResult": None,
                    "worldConfirmed": False,
                    "worldPendingChanges": True,
                    "worldArtifactId": None,
                    "worldVersion": None,
                    "worldConfirmationArtifactId": None,
                    "characters": [entry(character.model_dump()) for character in characters],
                    "relationshipInputs": normalize_relationship_inputs(
                        [value.model_dump() for value in body.relationshipInputs],
                        [value.model_dump() for value in characters],
                    ),
                    "relationships": relationship_state(characters),
                    "approved": False,
                    "approval": None,
                    "planningRequired": True,
                    "planApproved": False,
                    "requests": [],
                },
                "activeJobId": None,
                "queue": [],
            }
            connection.execute(
                "INSERT INTO m2_draft VALUES (?,?,?)",
                (
                    project_id,
                    json.dumps(state, ensure_ascii=False),
                    self.clock(),
                ),
            )
            from .project_history import HistoryService

            HistoryService(self.coordinator).ensure(connection, project_id)
            return self._view(connection, project_id, state)

    @staticmethod
    def _character(draft: dict, identifier: str) -> dict:
        character = next(
            (value for value in draft["characters"] if value["id"] == identifier), None
        )
        if character is None:
            raise ServiceError(404, "キャラクターが見つかりません。")
        return character

    @staticmethod
    def _world_confirmed(draft: dict) -> None:
        if not draft["worldConfirmed"] or draft["worldPendingChanges"]:
            raise ServiceError(409, "先に生成された世界観を確認して確定してください。")

    @staticmethod
    def _not_locked(character: dict, fields: dict, *, source: str) -> None:
        for scope, names in GROUPS.items():
            if character["locked"][scope] and any(
                name in fields
                and fields[name] != character[source].get(name, {
                    "sampleLines": [], "height_cm": None, "body_type": "unknown",
                }.get(name, ""))
                for name in names
            ):
                raise ServiceError(409, "固定した項目の変更には、先に固定解除が必要です。")
        if (
            character["locked"]["voice"]
            and "selfIntroduction" in fields
            and fields["selfIntroduction"] != character[source].get("selfIntroduction", "")
        ):
            raise ServiceError(409, "自己紹介の変更には、サンプル音声の固定解除が必要です。")

    @staticmethod
    def _voice_reference_text(connection, character: dict) -> str | None:
        identifier = character["voiceArtifactId"]
        if not identifier:
            return None
        provenance = json.loads(required(connection, "artifact", identifier)["provenance"])
        voice = provenance.get("voice", {})
        if not isinstance(voice, dict):
            return None
        text = voice.get("reference_text") or voice.get("text")
        return text if isinstance(text, str) and text.strip() else None

    @staticmethod
    def _invalidate_relationships(draft: dict) -> None:
        draft["relationships"]["pendingChanges"] = len(draft["characters"]) >= 2

    @staticmethod
    def _can_generate_character(character: dict, scope: str = "all") -> None:
        result = character["result"]
        legacy = result and (not result.get("selfIntroduction") or not result.get("sampleLines"))
        if legacy and scope != "all":
            raise ServiceError(
                409, "自己紹介・代表台詞が未生成です。先にキャラクター全体を生成してください。"
            )
        if legacy and (character["locked"]["settings"] or character["locked"]["voice"]):
            raise ServiceError(
                409, "自己紹介・代表台詞を新しく生成するには、設定と声の固定を解除してください。"
            )

    @staticmethod
    def _assert_mutable(connection: sqlite3.Connection, project_id: str) -> None:
        if connection.execute(
            # M3 owns an immutable approval snapshot, so subsequent draft edits
            # and auxiliary voice auditions cannot change its inputs.
            "SELECT 1 FROM job WHERE project_id=? AND kind NOT LIKE 'm3_%' "
            "AND status IN ('pending','running') LIMIT 1",
            (project_id,),
        ).fetchone():
            raise ServiceError(
                409, "生成処理中です。完了または失敗を確認してから変更してください。"
            )

    def _invalidate(self, state: dict) -> None:
        draft = state["draft"]
        draft["approved"] = False
        draft["approval"] = None
        draft["planApproved"] = False
        if draft["step"] in {"production", "planning-review"}:
            draft["step"] = (
                "character-review" if draft["worldConfirmed"] and not draft["worldPendingChanges"]
                else "world-review" if draft["worldResult"] else "world-input"
            )
        state["activeJobId"] = None
        state["queue"] = []
        state.pop("portraitPromptBatch", None)

    def _discard_cast(self, state: dict) -> None:
        """Leave only what the brief asks of the cast: no results, portraits, voices or locks.

        A work forked before its cast starts from here, so that nothing of the cast it came
        from is handed to the generator or mistaken for still being current.
        """
        draft = state["draft"]
        for index, character in enumerate(draft["characters"]):
            fresh = entry({**character["input"], "locked": {name: False for name in GROUPS}})
            # What each look is meant to be stays with the brief; its picture does not.
            fresh["outfits"] = [
                {"id": outfit["id"], "name": outfit["name"], "description": outfit["description"],
                 "occasion": outfit["occasion"], "imageArtifactId": None, "baseImageArtifactId": None,
                 "imagePendingChanges": outfit["id"] != DEFAULT_OUTFIT,
                 **({"materialBrief": outfit["materialBrief"]} if outfit.get("materialBrief") else {})}
                for outfit in character.get("outfits", [])]
            draft["characters"][index] = fresh
        draft["relationships"].update(result=None, artifactId=None, version=None)
        self._invalidate_relationships(draft)
        state["appliedInstructions"] = [item for item in state.get("appliedInstructions", [])
                                        if item["kind"] == "m2_world"]
        draft["requests"] = [item for item in draft.get("requests", []) if item.get("scope") == "world"]

    @staticmethod
    def _invalidate_world(draft: dict) -> None:
        draft["worldConfirmed"] = False
        draft["worldConfirmationArtifactId"] = None

    def _request(self, draft: dict, descriptor: dict, character: dict | None = None) -> None:
        if not descriptor["instruction"].strip():
            return
        request = {
            "id": uuid4().hex,
            "scope": descriptor["scope"],
            "instruction": descriptor["instruction"],
            "createdAt": iso(self.clock()),
            "protectedScopes": [key for key, value in character["locked"].items() if value]
            if character
            else [],
        }
        if character:
            request["characterId"] = character["id"]
        draft["requests"].append(request)

    @staticmethod
    def _remember_instruction(state: dict, kind: str, *, character_id: str | None = None,
                              scope: str = "all", instruction: str = "",
                              changes: dict | None = None) -> None:
        """Retain only accepted user edits, never inferred generated facts."""
        if not instruction.strip() and not changes:
            return
        entry = {"kind": kind, "character_id": character_id, "scope": scope}
        if instruction.strip():
            entry["instruction"] = instruction.strip()
        if changes:
            entry["changes"] = copy.deepcopy(changes)
        state.setdefault("appliedInstructions", []).append(entry)

    def _artifact(
        self,
        connection,
        project_id,
        logical_id,
        kind,
        filename,
        media_type,
        data,
        *,
        provenance=None,
        job=None,
        attempt=None,
    ) -> dict:
        record = self.coordinator._register_artifact(
            connection,
            project_id,
            logical_id,
            kind,
            filename,
            media_type,
            self.store.put(data),
            source_job_id=job["id"] if job else None,
            source_attempt_id=attempt["id"] if attempt else None,
        )
        connection.execute(
            "UPDATE artifact SET provenance=? WHERE id=?",
            (
                json.dumps(provenance or {"producer": "m2-user/1"}, ensure_ascii=False),
                record["id"],
            ),
        )
        return required(connection, "artifact", record["id"])

    def _result(
        self,
        connection,
        project_id,
        draft,
        character,
        result,
        *,
        envelope=None,
        job=None,
        attempt=None,
    ):
        kind = "m2_character" if character else "m2_world"
        logical_id = f"character-{character['id']}" if character else "world"
        value = dict(
            envelope
            or {
                "schema_version": 1,
                "kind": kind,
                "provenance": {"producer": "m2-user/1"},
                "trace": [],
            }
        )
        value["result"] = result
        record = self._artifact(
            connection,
            project_id,
            logical_id,
            kind,
            f"{logical_id}.json",
            "application/json",
            encode_json(value),
            provenance=value["provenance"],
            job=job,
            attempt=attempt,
        )
        if character:
            previous = character["result"]
            for scope, prefix in (("appearance", "image"), ("voice", "voice")):
                changed = previous and (
                    previous[scope] != result[scope]
                    or (
                        scope == "voice"
                        and previous.get("selfIntroduction") != result.get("selfIntroduction")
                    )
                )
                if changed and not character["locked"][scope]:
                    character[f"{prefix}ArtifactId"] = None
                    character[f"{prefix}Url"] = None
                    character[f"{prefix}PendingChanges"] = True
                    if prefix == "image":
                        self._stale_outfits(character)
            character["result"] = result
            character["resultArtifactId"], character["resultVersion"] = (
                record["id"],
                record["version"],
            )
            character["pendingChanges"] = False
            if not previous or any(
                previous.get(key) != result.get(key) for key in GROUPS["settings"]
            ):
                self._invalidate_relationships(draft)
        else:
            draft["worldResult"] = result
            draft["worldArtifactId"], draft["worldVersion"] = record["id"], record["version"]
            draft["worldPendingChanges"] = False
            self._invalidate_world(draft)
            for value in draft["characters"]:
                value["pendingChanges"] = True
            self._invalidate_relationships(draft)

    @staticmethod
    def _alternates(character: dict) -> list[dict]:
        return [value for value in character.get("outfits", []) if value["id"] != DEFAULT_OUTFIT]

    @classmethod
    def _stale_outfits(cls, character: dict) -> None:
        """An alternate look is an edit of the base portrait and follows its changes."""
        for outfit in cls._alternates(character):
            # A picture the user edited or chose by hand is theirs to keep.
            if not outfit.get("handPicked"):
                outfit["imagePendingChanges"] = True

    def _qwen_ready(self, connection, draft: dict | None = None) -> bool:
        from .image_editing import enabled

        # A work that forgoes the editing model behaves as if no worker offered it.
        if draft is not None and not enabled(draft):
            return False
        from .event_cg_settings import EventCgSettings

        return EventCgSettings(self.coordinator).portraits_ready(connection)

    @staticmethod
    def _approved_outfits(character: dict, reference) -> list[dict]:
        """The looks an approval hands to the story: what each shows, not how it was requested."""
        rows = list(character["outfits"])
        if not any(row["id"] == DEFAULT_OUTFIT for row in rows):
            rows.insert(0, {"id": DEFAULT_OUTFIT, "name": "通常の姿", "description": "", "occasion": ""})
        result = []
        for outfit in rows:
            base = outfit["id"] == DEFAULT_OUTFIT
            # The base look is told as the character's appearance, so the story can tell its
            # clothes from those of the other looks.
            description = (outfit.get("generatedDescription") or outfit["description"]
                           or (character["result"]["appearance"][:400] if base else ""))
            result.append({
                "id": outfit["id"], "name": outfit["name"], "description": description,
                "occasion": outfit["occasion"] or outfit.get("generatedOccasion") or "",
                **({} if base else {"image": reference(outfit["imageArtifactId"])})})
        return result

    @staticmethod
    def _uses_looks(draft: dict) -> bool:
        """Alternate looks exist only through the editing model; without it the story has none."""
        from .image_editing import enabled

        return enabled(draft)

    @classmethod
    def _stale_looks(cls, character: dict) -> list[dict]:
        """Alternate looks whose image is missing or was made from another portrait or wish."""
        return [outfit for outfit in cls._alternates(character)
                if not outfit.get("imageArtifactId") or outfit.get("imagePendingChanges")
                or (outfit.get("baseImageArtifactId") != character["imageArtifactId"]
                    and not outfit.get("handPicked"))]

    def _outfit_descriptor(self, connection, project_id: str, character: dict, outfit: dict,
                           *, source: str | None = None) -> dict:
        reference = required(connection, "artifact", source or character["imageArtifactId"])
        if reference["project_id"] != project_id or reference["kind"] != "m2_image":
            raise ServiceError(409, "立ち絵とキャラクターが一致しません。")
        self.store.read(reference)
        from .event_cg_settings import EventCgSettings

        descriptor = task("m2_outfit_image", character["id"], "outfit-image")
        descriptor["cg_profile"] = EventCgSettings(self.coordinator).settings(connection)["profile"]
        descriptor["outfit"] = self._look_wish(outfit)
        descriptor["reference_image"] = {"artifact_id": reference["id"], "sha256": reference["sha256"]}
        if outfit.get("materialImageId") or outfit.get("materialBrief"):
            material = self._material(connection, project_id, outfit)
            if material is None:
                raise ServiceError(409, "着せる共通の素材の絵がまだありません。")
            descriptor["material_image"] = material
        return descriptor

    @staticmethod
    def _look_wish(outfit: dict) -> dict:
        wish = {key: outfit[key] for key in ("id", "name", "description", "occasion")}
        if outfit.get("descriptionEdited") and outfit.get("generatedDescription"):
            # A directly edited look is drawn as written rather than designed again from the wish.
            wish["appearance"] = outfit["generatedDescription"]
            wish["occasion"] = outfit["occasion"] or outfit.get("generatedOccasion") or ""
        return wish

    def _looks_to_design(self, connection, project_id: str, draft: dict, drawing: str) -> list[dict]:
        """The looks that will be drawn once the cast's portraits are, as their jobs will ask for them.

        They go with a portrait job: the text model has just written the cast and is still
        loaded, so it puts the looks into words before the image models take the GPU in turn.
        """
        if not self._uses_looks(draft):
            return []
        rows = []
        for character in draft["characters"]:
            if not character.get("result"):
                continue
            # The portrait being drawn now replaces the one this character's looks were made from.
            looks = self._alternates(character) if character["id"] == drawing else self._stale_looks(character)
            for outfit in looks:
                if outfit.get("handPicked"):
                    continue
                worn = {}
                if outfit.get("materialImageId") or outfit.get("materialBrief"):
                    # A look that wears a shared material is described by its sheet: one that is
                    # not drawn yet leaves the look to be put into words when it is.
                    material = self._material(connection, project_id, outfit)
                    if material is None:
                        continue
                    worn = {"material_image": material}
                rows.append({"character_id": character["id"], "character_result": copy.deepcopy(character["result"]),
                             "outfit": self._look_wish(outfit), **worn,
                             **{key: value for key, value in self._look_context(draft, character, outfit).items()
                                if key == "current_look"}})
        return rows[:50]

    @staticmethod
    def _look_context(draft: dict, character: dict, outfit: dict) -> dict:
        """What the work already wears, so a look is designed apart from it rather than alike."""
        others = []
        for row in draft["characters"]:
            name = (row.get("result") or row["input"]).get("name") or "キャラクター"
            for look in row.get("outfits", []):
                if look.get("generatedDescription") and look["id"] != DEFAULT_OUTFIT:
                    mine = row["id"] == character["id"] and look["id"] == outfit["id"]
                    others.append({"worn_by": name, "look": look["name"], "outfit": look["generatedDescription"],
                                   **({"note": "the look being designed again"} if mine else {})})
        context = {"other_looks": others} if others else {}
        if outfit.get("generatedDescription"):
            context["current_look"] = outfit["generatedDescription"]
        return context

    def _material(self, connection, project_id: str, outfit: dict) -> dict | None:
        """The sheet a look wears: the one chosen in review, else the one its brief names."""
        from .material_service import MaterialService

        materials = MaterialService(self.coordinator)
        if outfit.get("materialImageId"):
            return materials.reference(connection, project_id, outfit["materialImageId"])
        return materials.brief_reference(connection, project_id, outfit["materialBrief"])

    def _sheet_redrawn(self, connection, project_id: str, outfit: dict) -> bool:
        """Whether the sheet this look wears through the brief was drawn again since it was put on."""
        worn = outfit.get("materialArtifactId")
        if not outfit.get("materialBrief") or not worn or outfit.get("materialImageId") or outfit.get("handPicked"):
            return False
        current = self._material(connection, project_id, outfit)
        return current is not None and current["artifact_id"] != worn

    def material_ready(self, connection, project_id: str) -> None:
        """A shared sheet was finished; draw the looks of an unapproved cast that waited for it."""
        row = connection.execute("SELECT 1 FROM m2_draft WHERE project_id=?", (project_id,)).fetchone()
        if row is None:
            return
        state = self._load(connection, project_id)
        if state["activeJobId"] or state["queue"] or state["draft"]["approved"]:
            return
        for character in state["draft"]["characters"]:
            state["queue"].extend(self._outfit_tasks(connection, project_id, state, character))
        if state["queue"]:
            state["draft"]["revision"] += 1
            self._queue_next(connection, project_id, state)
            self._save(connection, project_id, state)

    def _outfit_tasks(self, connection, project_id: str, state: dict, character: dict) -> list[dict]:
        """Looks wished for in the brief are drawn as soon as their base portrait is ready."""
        if (not character["imageArtifactId"] or character["imagePendingChanges"] or character["pendingChanges"]
                or not self._qwen_ready(connection, state["draft"])):
            return []
        queued = {(item["character_id"], item["outfit"]["id"]) for item in state["queue"]
                  if item["kind"] == "m2_outfit_image"}
        base = next((item for item in character.get("outfits", []) if item["id"] == DEFAULT_OUTFIT), None)
        if (base and base.get("materialBrief") and not character["locked"]["appearance"]
                and (base.get("materialApplied") != base["materialBrief"]
                     or self._sheet_redrawn(connection, project_id, base))):
            # The base look puts its shared design on first; every other look is derived from the result.
            if (character["id"], DEFAULT_OUTFIT) in queued or self._material(connection, project_id, base) is None:
                return []
            # Dressed again, it starts from the portrait as it was drawn, not from the earlier dressing.
            plain = base.get("plainArtifactId")
            if plain and connection.execute("SELECT 1 FROM artifact WHERE id=? AND project_id=? AND kind='m2_image'",
                                            (plain, project_id)).fetchone() is None:
                plain = None
            return [self._outfit_descriptor(connection, project_id, character, base, source=plain)]
        tasks = []
        stale = self._stale_looks(character)
        # A look follows the sheet it wears through the brief when that sheet is drawn again.
        stale += [outfit for outfit in self._alternates(character)
                  if outfit not in stale and self._sheet_redrawn(connection, project_id, outfit)]
        for outfit in stale:
            if (character["id"], outfit["id"]) in queued:
                continue
            if outfit.get("materialBrief") and not outfit.get("materialImageId") \
                    and self._material(connection, project_id, outfit) is None:
                continue
            outfit["imagePendingChanges"] = True
            tasks.append({**self._outfit_descriptor(connection, project_id, character, outfit),
                          **self._look_context(state["draft"], character, outfit)})
        return tasks

    def _sync_materials(self, connection, project_id: str, draft: dict, wishes) -> bool:
        """Keep the shared materials declared in the brief; True if the declaration changed."""
        if wishes is None:
            return False
        from .material_service import MaterialService

        value = [wish.model_dump() for wish in wishes]
        changed = value != draft.get("materials", [])
        draft["materials"] = value
        service = MaterialService(self.coordinator)
        service.sync_brief(connection, project_id, value)
        if draft.get("worldResult"):
            service.ensure_sheets(connection, project_id, world=draft["worldResult"])
        return changed

    def _sync_outfits(self, draft: dict, wishes: dict | None) -> bool:
        """Make each named character's alternate looks match the brief; True if anything changed."""
        changed = False
        for character in draft["characters"]:
            if wishes is None or character["id"] not in wishes:
                continue
            outfits = character.setdefault("outfits", [])
            current = {item["id"]: item for item in self._alternates(character)}
            kept = [item for item in outfits if item["id"] == DEFAULT_OUTFIT]
            used = set(current) | {DEFAULT_OUTFIT}
            declared = {row["id"] for row in draft.get("materials", [])}
            rows = [wish.model_dump() if hasattr(wish, "model_dump") else dict(wish) for wish in wishes[character["id"]]]
            # The base look only says which shared material it wears.
            worn = next((row.get("material") for row in rows if row["id"] == DEFAULT_OUTFIT), None)
            worn = worn if worn in declared else None
            if worn != (kept[0].get("materialBrief") if kept else None):
                if not kept:
                    kept.append({"id": DEFAULT_OUTFIT, "name": "通常の姿", "description": "", "occasion": "",
                                 "imageArtifactId": None, "baseImageArtifactId": None, "imagePendingChanges": False})
                kept[0].pop("materialApplied", None)
                kept[0].pop("materialBrief", None)
                if worn:
                    kept[0]["materialBrief"] = worn
                changed = True
            for value in rows:
                if value["id"] == DEFAULT_OUTFIT:
                    continue
                look = current.pop(value["id"], None) if value["id"] else None
                if look is None:
                    identifier = next(f"look{n}" for n in range(1, 100) if f"look{n}" not in used)
                    used.add(identifier)
                    look = {"id": identifier, "imageArtifactId": None, "baseImageArtifactId": None,
                            "imagePendingChanges": True}
                    changed = True
                if (look.get("name"), look.get("description")) != (value["name"], value["description"]):
                    look["imagePendingChanges"] = True
                    if look.get("description") != value["description"]:
                        look.pop("descriptionEdited", None)
                material = value.get("material") if value.get("material") in declared else None
                if look.get("materialBrief") != material:
                    look.pop("materialBrief", None)
                    look.pop("materialImageId", None)
                    if material:
                        look["materialBrief"] = material
                    look["imagePendingChanges"] = changed = True
                fields = {key: value[key] for key in ("name", "description", "occasion")}
                changed = changed or any(look.get(key) != item for key, item in fields.items())
                look.update(fields)
                kept.append(look)
            changed = changed or bool(current)
            character["outfits"] = kept
        return changed

    def _image_action(self, connection, project_id: str, state: dict, character: dict, body) -> None:
        outfits = character.setdefault("outfits", [])
        base = body.outfit_id == DEFAULT_OUTFIT
        outfit = next((item for item in outfits if item["id"] == body.outfit_id), None)
        if base and character["locked"]["appearance"]:
            raise ServiceError(409, "外見の固定を解除してから操作してください。")
        if base and outfit is None:
            outfit = {"id": DEFAULT_OUTFIT, "name": "通常の姿", "description": "", "occasion": "",
                      "imageArtifactId": None, "baseImageArtifactId": None, "imagePendingChanges": False}
            outfits.insert(0, outfit)
        if outfit is None:
            raise ServiceError(404, "別バージョンが見つかりません。")
        if not character["imageArtifactId"] or character["imagePendingChanges"] or character["pendingChanges"]:
            raise ServiceError(409, "現在の設定を反映した立ち絵を先に生成してください。")
        if body.action == "adopt-image":
            record = self._own_image(connection, project_id, character, body.artifact_id)
            if base:
                if record["kind"] != "m2_image":
                    raise ServiceError(422, "通常の姿には、通常の姿として作った絵を選んでください。")
                if record["id"] != character["imageArtifactId"]:
                    character.update(imageArtifactId=record["id"], imagePendingChanges=False)
                    # A picture chosen by hand is final; it is not dressed again on its own.
                    if outfit.get("materialBrief"):
                        outfit["materialApplied"] = outfit["materialBrief"]
                    self._stale_outfits(character)
                    state["queue"].extend(self._outfit_tasks(connection, project_id, state, character))
            else:
                # A picture chosen by hand stays, whatever base portrait it was once derived from.
                outfit.update(imageArtifactId=record["id"], imagePendingChanges=False, handPicked=True,
                              baseImageArtifactId=character["imageArtifactId"])
            return
        if not self._qwen_ready(connection, state["draft"]):
            raise ServiceError(409, "絵をもとにした修正には、画像編集の準備ができたWorkerが必要です。")
        current = character["imageArtifactId"] if base else outfit.get("imageArtifactId")
        source = body.base_artifact_id or current
        if not source:
            raise ServiceError(409, "もとにする絵がまだありません。")
        reference = self._own_image(connection, project_id, character, source)
        from .event_cg_settings import EventCgSettings

        descriptor = task("m2_outfit_image", character["id"], "outfit-image")
        descriptor["cg_profile"] = EventCgSettings(self.coordinator).settings(connection)["profile"]
        descriptor["outfit"] = {key: outfit[key] for key in ("id", "name", "description", "occasion")}
        descriptor["reference_image"] = {"artifact_id": reference["id"], "sha256": reference["sha256"]}
        descriptor["edit"] = {"instruction": body.instruction}
        descriptor.update(self._look_context(state["draft"], character, outfit))
        if body.material_image_id:
            from .material_service import MaterialService

            material = MaterialService(self.coordinator).reference(connection, project_id, body.material_image_id)
            if material is None:
                raise ServiceError(409, "着せる共通の素材の絵がまだありません。")
            descriptor["material_image"] = material
        state["queue"] = [descriptor]

    def _outfit_action(self, connection, project_id: str, state: dict, character: dict, body) -> None:
        outfits = character.setdefault("outfits", [])
        if body.action == "save-outfit":
            value = body.outfit.model_dump()
            current = next((item for item in outfits if item["id"] == value["id"]), None)
            if current is None:
                if value["id"] not in (None, DEFAULT_OUTFIT):
                    raise ServiceError(404, "別バージョンが見つかりません。")
                if value["id"] is None:
                    if len(self._alternates(character)) >= 4:
                        raise ServiceError(422, "別バージョンは1人につき4つまでです。")
                    used = {item["id"] for item in outfits}
                    value["id"] = next(f"look{n}" for n in range(1, 100) if f"look{n}" not in used)
                current = {"id": value["id"], "imageArtifactId": None, "baseImageArtifactId": None,
                           "imagePendingChanges": value["id"] != DEFAULT_OUTFIT}
                outfits.append(current)
            if value["id"] == DEFAULT_OUTFIT:
                pass
            elif value["appearance"] is not None:
                # A direct edit states what the look shows; only that text redraws it.
                if value["appearance"] != current.get("generatedDescription"):
                    current.update(generatedDescription=value["appearance"], descriptionEdited=True,
                                   imagePendingChanges=True)
            elif (current.get("name"), current.get("description")) != (value["name"], value["description"]):
                # The look is designed from its name and the user's wish; either one changes it.
                current["imagePendingChanges"] = True
                if current.get("description") != value["description"]:
                    current.pop("descriptionEdited", None)
            current.update(name=value["name"], description=value["description"], occasion=value["occasion"])
            outfits.sort(key=lambda item: item["id"] != DEFAULT_OUTFIT)
            return
        base = body.action == "generate-outfit" and body.outfit_id == DEFAULT_OUTFIT
        outfit = next((item for item in (outfits if base else self._alternates(character))
                       if item["id"] == body.outfit_id), None)
        if base:
            # The base look is only ever redrawn by putting a shared design on it.
            if not body.material_image_id:
                raise ServiceError(422, "着せる共通の素材を選んでください。")
            if character["locked"]["appearance"]:
                raise ServiceError(409, "外見の固定を解除してから操作してください。")
            if outfit is None:
                outfit = {"id": DEFAULT_OUTFIT, "name": "通常の姿", "description": "", "occasion": "",
                          "imageArtifactId": None, "baseImageArtifactId": None, "imagePendingChanges": False}
                outfits.insert(0, outfit)
        if outfit is None:
            raise ServiceError(404, "別バージョンが見つかりません。")
        if body.action == "delete-outfit":
            outfits.remove(outfit)
            return
        if body.material_image_id:
            outfit["materialImageId"] = body.material_image_id
        elif body.clear_material:
            outfit.pop("materialImageId", None)
        if not character["imageArtifactId"] or character["imagePendingChanges"] or character["pendingChanges"]:
            raise ServiceError(409, "現在の設定を反映した立ち絵を先に生成してください。")
        if not self._qwen_ready(connection, state["draft"]):
            raise ServiceError(409, "別バージョンの立ち絵には、画像編集の準備ができたWorkerが必要です。")
        if base:
            descriptor = self._outfit_descriptor(connection, project_id, character,
                                                 outfit)
            # One redraw; the base look does not stay tied to the design afterwards.
            outfit.pop("materialImageId", None)
            state["queue"] = [descriptor]
            return
        outfit["imagePendingChanges"] = True
        state["queue"] = [{**self._outfit_descriptor(connection, project_id, character, outfit),
                           **self._look_context(state["draft"], character, outfit)}]

    @classmethod
    def _look_image(cls, character: dict, outfit_id: str) -> str | None:
        """The portrait a look currently shows, if it has one that reflects its settings."""
        if outfit_id == DEFAULT_OUTFIT:
            return None if character["imagePendingChanges"] else character["imageArtifactId"]
        outfit = next((item for item in cls._alternates(character) if item["id"] == outfit_id), None)
        if outfit is None or outfit in cls._stale_looks(character):
            return None
        return outfit["imageArtifactId"]

    def _expression_action(self, connection, project_id: str, state: dict, character: dict, body) -> None:
        """Expression pictures are extras: they neither enter nor end the approval of the cast."""
        pictures = character.setdefault("expressions", [])
        if body.action == "delete-expression":
            kept = [row for row in pictures
                    if (row["outfitId"], row["expression"]) != (body.outfit_id, body.expression)]
            if len(kept) == len(pictures):
                raise ServiceError(404, "その表情の絵が見つかりません。")
            character["expressions"] = kept
            return
        if not self._qwen_ready(connection, state["draft"]):
            raise ServiceError(409, "表情の作成には、画像編集の準備ができたWorkerが必要です。")
        known = [DEFAULT_OUTFIT, *(item["id"] for item in self._alternates(character))]
        looks = body.outfit_ids or known
        if any(look not in known for look in looks):
            raise ServiceError(404, "別バージョンが見つかりません。")
        from .event_cg_settings import EventCgSettings

        profile = EventCgSettings(self.coordinator).settings(connection)["profile"]
        names = {item["id"]: item["name"] for item in character.get("outfits", [])}
        queued = {(item["character_id"], item["outfit"]["id"], item["expression"]["id"])
                  for item in state["queue"] if item.get("expression")}
        tasks = []
        for look in looks:
            image = self._look_image(character, look)
            if image is None:
                if body.outfit_ids:
                    raise ServiceError(409, "現在の設定を反映した立ち絵を先に生成してください。")
                # Asked for every look: one still waiting for its portrait is simply left out.
                continue
            reference = required(connection, "artifact", image)
            self.store.read(reference)
            for expression in body.expressions:
                current = next((row for row in pictures if (row["outfitId"], row["expression"]) == (look, expression)
                                and row["baseImageArtifactId"] == image), None)
                if (current and not body.redraw) or (character["id"], look, expression) in queued:
                    continue
                descriptor = task("m2_outfit_image", character["id"], "expression-image")
                descriptor["cg_profile"] = profile
                descriptor["outfit"] = {"id": look, "name": names.get(look) or "通常の姿", "description": "",
                                        "occasion": ""}
                descriptor["reference_image"] = {"artifact_id": reference["id"], "sha256": reference["sha256"]}
                descriptor["expression"] = {"id": expression, **({"face_box": list(body.face_box)}
                                                                if body.face_box else {})}
                tasks.append(descriptor)
        if not tasks:
            raise ServiceError(409, "作成する表情がありません。すでに作成済みです。")
        state["queue"].extend(tasks)

    def _profile(self, connection=None) -> dict:
        from .llm_settings import generation_profile

        return generation_profile(self.coordinator, connection)

    def _queue_next(self, connection: sqlite3.Connection, project_id: str, state: dict) -> None:
        if not state["queue"]:
            state["activeJobId"] = None
            state.pop("portraitPromptBatch", None)
            return
        # Keep all text generation ahead of media, including pending queues saved
        # by older versions. Stable ordering preserves cast order within a stage.
        stages = {"m2_character": 0, "m2_relationships": 1, "m2_image": 2, "m2_voice": 3,
                  "m2_outfit_image": 4}
        state["queue"].sort(key=lambda item: stages.get(item["kind"], 0))
        descriptor = state["queue"].pop(0)
        draft = state["draft"]
        character = (
            self._character(draft, descriptor["character_id"])
            if descriptor["character_id"]
            else None
        )
        profile = self._profile(connection)
        from .tts_settings import VOICE_PURPOSES, TTSService

        tts_profile = (TTSService(self.coordinator).profile(connection)
                       if descriptor["kind"] in VOICE_PURPOSES else None)
        payload = {
            "schema_version": 1,
            "seed": descriptor["seed"],
            "base_revision": draft["revision"],
            "world_input": draft["worldInput"],
            "world_result": draft["worldResult"],
            "character_id": descriptor["character_id"],
            "character_input": character["input"] if character else None,
            "character_result": character["result"] if character else None,
            "scope": descriptor["scope"],
            "instruction": descriptor["instruction"],
            "applied_instructions": copy.deepcopy(state.get("appliedInstructions", [])),
            "locked": character["locked"] if character else {name: False for name in GROUPS},
            "profile": profile,
            **({"tts_profile": tts_profile} if tts_profile is not None else {}),
            "character_contract_version": 2
            if descriptor["kind"] == "m2_character"
            or not character
            or (
                character["result"]
                and character["result"].get("selfIntroduction")
                and character["result"].get("sampleLines")
            )
            else 1,
            "cast_inputs": [value["input"] for value in draft["characters"]],
            "cast_results": [value["result"] for value in draft["characters"] if value["result"]],
            "relationship_inputs": draft["relationshipInputs"],
            "relationships_result": draft["relationships"]["result"],
        }
        from .llm_settings import execution_settings

        execution = execution_settings(self.coordinator, connection, payload,
                                       kind=descriptor["kind"], profile=profile)
        if descriptor["kind"] == "m2_image":
            # Freeze the whole portrait stage on its first job. Every following
            # image (and retries/restarts) shares the same converter cache, so
            # translation finishes before the image model is loaded.
            if "portraitPromptBatch" not in state:
                state["portraitPromptBatch"] = [
                    {
                        "character_id": item["character_id"],
                        "character_result": copy.deepcopy(
                            self._character(draft, item["character_id"])["result"]
                        ),
                    }
                    for item in [descriptor, *state["queue"]]
                    if item["kind"] == "m2_image"
                ]
            payload["portrait_prompt_batch"] = copy.deepcopy(state["portraitPromptBatch"])
            waiting = self._looks_to_design(connection, project_id, draft, descriptor["character_id"])
            if waiting:
                payload["look_design_batch"] = waiting
        else:
            state.pop("portraitPromptBatch", None)
        if descriptor["kind"] == "m2_world":
            from .material_service import MaterialService

            sheets = MaterialService(self.coordinator).sheets_to_design(
                connection, project_id, redraw=not (descriptor.get("instruction") or "").strip())
            if sheets:
                payload["material_design_batch"] = sheets
        for key in ("reference_voice", "dialogue_text", "reference_image", "outfit", "cg_profile", "material_image", "edit",
                    "other_looks", "current_look", "expression"):
            if key in descriptor:
                payload[key] = descriptor[key]
        if descriptor["kind"] == "m2_outfit_image" and "outfit" in descriptor and "expression" not in descriptor:
            # The looks that wait behind this one are put into words in the same text-model
            # session, so the editing model then stays loaded from one picture to the next.
            waiting = [
                {"character_id": item["character_id"],
                 "character_result": copy.deepcopy(self._character(draft, item["character_id"])["result"]),
                 **{key: copy.deepcopy(item[key])
                    for key in ("outfit", "edit", "other_looks", "current_look", "material_image") if key in item}}
                for item in state["queue"]
                if item["kind"] == "m2_outfit_image" and "outfit" in item and "expression" not in item
            ][:50]
            if waiting:
                payload["look_design_batch"] = waiting
        identifier, run_id, now = uuid4().hex, uuid4().hex, self.clock()
        connection.execute(
            "INSERT INTO generation_run (id,project_id,settings_version,story_revision_id,policy,status,created_at) "
            "VALUES (?,?,?,?,?,'pending',?)",
            (
                run_id,
                project_id,
                draft["revision"],
                draft["worldArtifactId"] or "m2-draft",
                json.dumps({"mode": "m2-local", "max_attempts": 3}),
                now,
            ),
        )
        connection.execute(
            "INSERT INTO job (id,project_id,run_id,kind,payload,settings_snapshot,status,max_attempts,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,'pending',3,?,?)",
            (
                identifier,
                project_id,
                run_id,
                descriptor["kind"],
                json.dumps(payload, ensure_ascii=False),
                json.dumps(
                    {
                        "schema_version": 1,
                        **execution,
                        **({"tts_profile": tts_profile} if tts_profile is not None else {}),
                        "base_revision": draft["revision"],
                        "seed": descriptor["seed"],
                    }
                ),
                now,
                now,
            ),
        )
        state["activeJobId"] = identifier

    def action(self, project_id: str, body) -> dict:
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            state = self._load(connection, project_id)
            draft = state["draft"]
            if body.expected_revision != draft["revision"]:
                raise ServiceError(
                    409, "別の操作で更新されました。最新の内容を読み込み直してください。"
                )
            name = body.action
            reconfirm = getattr(body, "reconfirm", False) and name in {"confirm-world", "approve"}
            if name == "go-to":
                step = "world-input" if body.step == "character-input" else body.step
                if step == "character-review":
                    self._world_confirmed(draft)
                if step == "production" and not (
                    draft["approved"] or self._has_production(connection, project_id)
                ):
                    raise ServiceError(409, "キャラクターと素材を確認・承認してから制作へ進んでください。")
                if step == "planning-review" and not draft["approved"]:
                    raise ServiceError(409, "メインキャラの承認後に構成を確認できます。")
                if (step == "production" and draft.get("planningRequired")
                        and not draft.get("planApproved") and not self._has_production(connection, project_id)):
                    raise ServiceError(409, "先に全体構成を確認して承認してください。")
                draft["step"] = step
                # Navigation does not change generation inputs or invalidate approvals/retries.
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if (
                name == "confirm-world"
                and not reconfirm
                and draft["worldConfirmed"]
                and not draft["worldPendingChanges"]
                and (state["activeJobId"] or not self._needs_cast_generation(draft))
            ):
                # Revisiting an already confirmed world must not restart generation
                # or detach the immutable approval used by an existing chapter.
                if draft["step"] != "character-review":
                    draft["step"] = "character-review"
                    self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            self._assert_mutable(connection, project_id)
            if name == "approve" and draft["approved"] and not reconfirm:
                self._approve(connection, project_id, state)
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if name == "edit-character":
                character = self._character(draft, body.character_id)
                if (character["result"] and not character["pendingChanges"]
                        and all(character["result"].get(key) == value for key, value in body.patch.items())):
                    return self._view(connection, project_id, state)
            from .project_history import HistoryService, operation_label

            history = HistoryService(self.coordinator)
            # A direct edit may already have cleared the approval flag. Every
            # real confirmation detaches later steps; duplicate navigation has
            # returned above without changing the selected version.
            restart_downstream = name in {"generate-world", "confirm-world", "approve"}
            if restart_downstream:
                history.assert_reconfirmation_allowed(connection, project_id)
            history.begin(
                connection, project_id, operation_label(body, state), name,
            )
            if restart_downstream:
                history.clear_downstream(connection, project_id, planning=True)
            if name == "save-brief":
                previous = (
                    draft["worldInput"],
                    [copy.deepcopy(value["input"]) for value in draft["characters"]],
                    draft["relationshipInputs"],
                )
                self._save_characters(draft, body)
                materials_changed = self._sync_materials(connection, project_id, draft, body.materials)
                looks_changed = self._sync_outfits(draft, body.outfits) or materials_changed
                draft["worldInput"] = body.world.model_dump()
                current = (
                    draft["worldInput"],
                    [value["input"] for value in draft["characters"]],
                    draft["relationshipInputs"],
                )
                if current != previous:
                    state["appliedInstructions"] = []
                    self._invalidate(state)
                    draft["revision"] += 1
                    draft["worldPendingChanges"] = True
                    self._invalidate_world(draft)
                    for character in draft["characters"]:
                        character["pendingChanges"] = True
                    self._invalidate_relationships(draft)
                elif looks_changed:
                    # A look alone leaves the world and cast as they are; only the approval ends.
                    self._invalidate(state)
                    draft["revision"] += 1
                draft["step"] = "world-input"
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if name in {"generate-expressions", "delete-expression"}:
                character = self._character(draft, body.character_id)
                if not character["result"]:
                    raise ServiceError(409, "先にキャラクターを生成してください。")
                # Like a voice audition, this leaves the cast and its approval as they are.
                self._expression_action(connection, project_id, state, character, body)
                self._queue_next(connection, project_id, state)
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if name == "clone-voice":
                self._clone_voice(connection, project_id, state, body)
                self._queue_next(connection, project_id, state)
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if name == "approve":
                self._approve(connection, project_id, state, reconfirm=reconfirm,
                              cast_limit=getattr(body, "cast_limit", None))
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            # Failed previous work is superseded by an explicit edit/new generation.
            self._invalidate(state)
            draft["revision"] += 1
            if name == "save-world":
                value = body.world.model_dump()
                if value != draft["worldInput"]:
                    state["appliedInstructions"] = [
                        item for item in state["appliedInstructions"] if item["kind"] != "m2_world"
                    ]
                    draft["worldInput"] = value
                    draft["worldPendingChanges"] = True
                    self._invalidate_world(draft)
                    self._invalidate_relationships(draft)
                draft["step"] = "world-input"
            elif name == "edit-world":
                try:
                    result = WorldResult.model_validate(body.world.model_dump()).model_dump()
                except ValidationError as exc:
                    raise ServiceError(
                        422, "世界観のタイトル・ジャンル・雰囲気・設定を入力してください。"
                    ) from exc
                changes = {key: value for key, value in result.items()
                           if (draft["worldResult"] or {}).get(key) != value}
                self._result(connection, project_id, draft, None, result)
                self._remember_instruction(state, "m2_world", scope="world", changes=changes)
                draft["step"] = "world-review"
            elif name == "generate-world":
                self._invalidate_world(draft)
                draft["worldPendingChanges"] = True
                descriptor = task("m2_world", scope="world", instruction=body.instruction.strip())
                self._request(draft, descriptor)
                state["queue"] = [descriptor]
                draft["step"] = "world-review"
            elif name == "confirm-world":
                if not draft["worldResult"] or draft["worldPendingChanges"]:
                    raise ServiceError(
                        409, "入力を反映した完全な世界観を生成してから確定してください。"
                    )
                if not draft["worldConfirmed"] or reconfirm:
                    snapshot = {
                        "schema_version": 1,
                        "revision": draft["revision"],
                        "confirmedAt": iso(self.clock()),
                        "worldArtifactId": draft["worldArtifactId"],
                        "worldVersion": draft["worldVersion"],
                        "world": draft["worldResult"],
                    }
                    record = self._artifact(
                        connection,
                        project_id,
                        "world-confirmation",
                        "m2_confirmation",
                        "world-confirmation.json",
                        "application/json",
                        encode_json(snapshot),
                    )
                    draft["worldConfirmationArtifactId"] = record["id"]
                draft["worldConfirmed"], draft["step"] = True, "character-review"
                self._queue_characters(state, force=reconfirm, connection=connection, project_id=project_id)
            elif name == "set-image-editing":
                # A choice about the whole work, open from the first brief on.
                draft["imageEditing"] = body.enabled
                if body.enabled:
                    # Looks that waited while editing was off are drawn now.
                    for character in draft["characters"]:
                        state["queue"].extend(self._outfit_tasks(connection, project_id, state, character))
            else:
                self._world_confirmed(draft)
                self._character_action(connection, project_id, state, body)
            self._queue_next(connection, project_id, state)
            self._save(connection, project_id, state)
            return self._view(connection, project_id, state)

    def _save_characters(self, draft: dict, body) -> bool:
        if len(body.characters) > 3:
            raise ServiceError(422, "メインキャラクターは3人以下にしてください。")
        previous = {value["id"]: value for value in draft["characters"]}
        previous_inputs = [copy.deepcopy(value["input"]) for value in draft["characters"]]
        previous_relationships = copy.deepcopy(draft["relationshipInputs"])

        def identity_inputs(characters):
            return {
                value["id"]: {
                    key: value["input"].get(key, [] if key == "sampleLines" else "")
                    for key in GROUPS["settings"]
                }
                for value in characters
            }

        old_inputs = identity_inputs(draft["characters"])
        replacements = []
        for value in body.characters:
            brief = value.model_dump()
            character = previous.pop(brief["id"], None)
            if character:
                if brief["locked"] != character["locked"]:
                    raise ServiceError(409, "固定の変更は専用の操作で行ってください。")
                self._not_locked(character, brief, source="input")
                if brief != character["input"]:
                    character["input"] = brief
                    character["pendingChanges"] = True
                replacements.append(character)
            else:
                if any(brief["locked"].values()):
                    raise ServiceError(422, "生成結果を確認してから固定してください。")
                replacements.append(entry(brief))
        if any(any(character["locked"].values()) for character in previous.values()):
            raise ServiceError(
                409, "固定したキャラクターを削除するには、先に固定を解除してください。"
            )
        draft["characters"] = replacements
        valid_pairs = pair_ids(replacements)
        instructions = (
            [value.model_dump() for value in body.relationshipInputs]
            if body.relationshipInputs is not None
            else [
                value
                for value in draft["relationshipInputs"]
                if tuple(value["characterIds"]) in valid_pairs
            ]
        )
        instructions = normalize_relationship_inputs(instructions, replacements)
        if instructions != draft["relationshipInputs"] or old_inputs != identity_inputs(
            replacements
        ):
            self._invalidate_relationships(draft)
        draft["relationshipInputs"] = instructions
        if len(replacements) < 2:
            draft["relationships"] = relationship_state(replacements)
        retained = {value["id"] for value in replacements}
        draft["requests"] = [
            value
            for value in draft["requests"]
            if not value.get("characterId") or value["characterId"] in retained
        ]
        draft["step"] = "world-input"
        return (previous_inputs != [value["input"] for value in replacements]
                or previous_relationships != draft["relationshipInputs"])

    def _queue_characters(self, state: dict, *, force: bool = False, connection=None, project_id=None) -> None:
        draft = state["draft"]
        if len(draft["characters"]) > 3:
            raise ServiceError(409, "メインキャラクターを3人以下にしてから生成してください。")
        for character in draft["characters"]:
            needs_result = force or not character["result"] or character["pendingChanges"]
            if needs_result and (not character["result"] or not all(character["locked"].values())):
                self._can_generate_character(character)
                character["pendingChanges"] = True
                state["queue"].append(task("m2_character", character["id"]))
            else:
                character["pendingChanges"] = False
                state["queue"].extend(self._missing_media(character))
                if connection is not None:
                    state["queue"].extend(self._outfit_tasks(connection, project_id, state, character))
        if len(draft["characters"]) >= 2 and (
            force
            or draft["relationships"]["pendingChanges"]
            or not draft["relationships"]["result"]
            or not draft["relationships"]["artifactId"]
        ):
            draft["relationships"]["pendingChanges"] = True
            state["queue"].append(task("m2_relationships", scope="relationships"))

    def _needs_cast_generation(self, draft: dict) -> bool:
        return any(
            not character["result"] or character["pendingChanges"] or self._missing_media(character)
            or (self._uses_looks(draft) and self._stale_looks(character))
            for character in draft["characters"]
        ) or (
            len(draft["characters"]) >= 2
            and (
                draft["relationships"]["pendingChanges"]
                or not draft["relationships"]["result"]
                or not draft["relationships"]["artifactId"]
            )
        )

    def _character_action(self, connection, project_id: str, state: dict, body) -> None:
        draft, name = state["draft"], body.action
        if name == "save-characters":
            previous_inputs = {person["id"]: copy.deepcopy(person["input"])
                               for person in draft["characters"]}
            previous_relationships = copy.deepcopy(draft["relationshipInputs"])
            saved = self._save_characters(draft, body)
            self._sync_outfits(draft, body.outfits)
            if saved:
                current_inputs = {person["id"]: person["input"] for person in draft["characters"]}
                changed_ids = {identifier for identifier in previous_inputs.keys() | current_inputs.keys()
                               if previous_inputs.get(identifier) != current_inputs.get(identifier)}
                reset_relationships = (
                    previous_relationships != draft["relationshipInputs"]
                    or bool(previous_inputs.keys() - current_inputs.keys())
                )
                state["appliedInstructions"] = [
                    item for item in state["appliedInstructions"]
                    if not (item["kind"] == "m2_character" and item["character_id"] in changed_ids)
                    and not (item["kind"] == "m2_relationships" and reset_relationships)
                ]
            return
        draft["step"] = "character-review"
        if len(draft["characters"]) > 3 and (
            name == "generate-relationships"
            or (name == "revise-character" and body.scope in ("all", "settings"))
        ):
            raise ServiceError(
                409, "メインキャラクターを3人以下にしてから関係性を生成してください。"
            )
        if name == "save-relationships":
            instructions = normalize_relationship_inputs(
                [value.model_dump() for value in body.relationshipInputs], draft["characters"]
            )
            if instructions != draft["relationshipInputs"]:
                state["appliedInstructions"] = [
                    item for item in state["appliedInstructions"]
                    if item["kind"] != "m2_relationships"
                ]
                draft["relationshipInputs"] = instructions
                self._invalidate_relationships(draft)
            return
        if name == "generate-relationships":
            if len(draft["characters"]) < 2:
                raise ServiceError(409, "関係性の生成には2人以上のキャラクターが必要です。")
            if any(not value["result"] or value["pendingChanges"] for value in draft["characters"]):
                raise ServiceError(409, "先に全キャラクターの設定を生成してください。")
            draft["relationships"]["pendingChanges"] = True
            descriptor = task(
                "m2_relationships", scope="relationships", instruction=body.instruction.strip()
            )
            self._request(draft, descriptor)
            state["queue"] = [descriptor]
            return
        if name == "generate-characters":
            self._queue_characters(state, force=True, connection=connection, project_id=project_id)
            return
        character = self._character(draft, body.character_id)
        if not character["result"]:
            raise ServiceError(409, "先にキャラクターを生成してください。")
        if name in {"edit-image", "adopt-image"}:
            self._image_action(connection, project_id, state, character, body)
        elif name in {"save-outfit", "delete-outfit", "generate-outfit"}:
            self._outfit_action(connection, project_id, state, character, body)
        elif name == "toggle-lock":
            scope = body.scope
            character["locked"][scope] = not character["locked"][scope]
            character["input"]["locked"] = dict(character["locked"])
            # Locks are part of CharacterBrief: version the exact displayed result,
            # so an approval's result remains identical to its artifact/hash.
            pending = character["pendingChanges"]
            result = {**character["result"], "locked": dict(character["locked"])}
            self._result(connection, project_id, draft, character, result)
            character["pendingChanges"] = pending
        elif name == "edit-character":
            if character["pendingChanges"]:
                raise ServiceError(
                    409,
                    "未反映の入力や修正指示があります。先にキャラクター全体を生成してください。",
                )
            self._not_locked(character, body.patch, source="result")
            result = {**character["result"], **body.patch}
            try:
                result = validate_character_result(result)
            except ValidationError as exc:
                raise ServiceError(
                    422, "キャラクターの生成結果に必要な設定を入力してください。"
                ) from exc
            changes = {key: result[key] for key in body.patch
                       if character["result"].get(key) != result[key]}
            self._result(connection, project_id, draft, character, result)
            self._remember_instruction(state, "m2_character", character_id=character["id"],
                                       changes=changes)
        elif name == "revise-character":
            self._can_generate_character(character, body.scope)
            if character["pendingChanges"] and body.scope != "all":
                raise ServiceError(
                    409,
                    "未反映の入力や修正指示があります。先にキャラクター全体を生成してください。",
                )
            if (body.scope == "all" and all(character["locked"].values())) or character[
                "locked"
            ].get(body.scope):
                raise ServiceError(
                    409, "対象の項目は固定されています。先に固定を解除してください。"
                )
            descriptor = task("m2_character", character["id"], body.scope, body.instruction.strip())
            character["pendingChanges"] = True
            self._request(draft, descriptor, character)
            state["queue"] = [descriptor]
            if body.scope in ("all", "settings") and len(draft["characters"]) >= 2:
                draft["relationships"]["pendingChanges"] = True
                state["queue"].append(task("m2_relationships", scope="relationships"))
        elif name == "retake":
            scope = "appearance" if body.scope == "image-retake" else "voice"
            if character["locked"][scope]:
                raise ServiceError(
                    409, "対象の素材は固定されています。先に固定を解除してください。"
                )
            if character["pendingChanges"]:
                raise ServiceError(409, "保存した入力を先にキャラクター生成に反映してください。")
            descriptor = task(
                "m2_image" if scope == "appearance" else "m2_voice",
                character["id"],
                body.scope,
                body.instruction.strip(),
            )
            character["imagePendingChanges" if scope == "appearance" else "voicePendingChanges"] = (
                True
            )
            self._request(draft, descriptor, character)
            state["queue"] = [descriptor]

    def _clone_voice(self, connection, project_id: str, state: dict, body) -> None:
        draft = state["draft"]
        if state["activeJobId"]:
            active = required(connection, "job", state["activeJobId"])
            if active["status"] == "failed" and active["kind"] != "m2_voice_clone":
                raise ServiceError(
                    409,
                    "未完了の生成処理があります。失敗した生成を再試行してから、ボイスクローンを生成してください。",
                )
        character = self._character(draft, body.character_id)
        if (
            not character["result"]
            or character["pendingChanges"]
            or character["voicePendingChanges"]
            or not character["voiceArtifactId"]
        ):
            raise ServiceError(409, "現在の設定を反映したサンプル音声を先に生成してください。")
        reference = required(connection, "artifact", character["voiceArtifactId"])
        if (
            reference["project_id"] != project_id
            or reference["logical_id"] != f"voice-{character['id']}"
            or reference["kind"] != "m2_voice"
        ):
            raise ServiceError(409, "サンプル音声とキャラクターが一致しません。")
        transcript = self._voice_reference_text(connection, character)
        if not transcript:
            raise ServiceError(
                409, "サンプル音声の台詞が記録されていません。先に音声をリテイクしてください。"
            )
        self.store.read(reference)
        descriptor = task("m2_voice_clone", character["id"], "voice-clone")
        descriptor["reference_voice"] = {
            "artifact_id": reference["id"],
            "sha256": reference["sha256"],
            "text": transcript,
        }
        descriptor["dialogue_text"] = body.text
        # Trials are auxiliary output: keep the approved cast and its immutable snapshot.
        state["activeJobId"] = None
        state["queue"] = [descriptor]
        draft["revision"] += 1

    @staticmethod
    def _missing_media(character: dict) -> list[dict]:
        pending = []
        if not character["imageArtifactId"] or (
            character["imagePendingChanges"] and not character["locked"]["appearance"]
        ):
            pending.append(task("m2_image", character["id"], "image-retake"))
        if not character["voiceArtifactId"] or (
            character["voicePendingChanges"] and not character["locked"]["voice"]
        ):
            pending.append(task("m2_voice", character["id"], "voice-retake"))
        return pending

    def _approve(self, connection, project_id: str, state: dict, *, reconfirm=False, cast_limit=None) -> None:
        draft = state["draft"]
        if cast_limit is not None and cast_limit < len(draft["characters"]):
            raise ServiceError(400, "登場キャラの上限は、メインキャラの人数以上にしてください。")
        from services.worker.generation.script_cast import work_limits

        from .llm_settings import script_profile

        limits = work_limits(script_profile(self.coordinator, connection))
        if cast_limit is not None and cast_limit > limits.characters:
            raise ServiceError(400, f"登場キャラの上限は{limits.characters}人までです。")
        if limits.looks is not None and sum(
                len(self._alternates(value)) for value in draft["characters"]) > limits.looks:
            raise ServiceError(409, f"別バージョンの立ち絵は、全員で{limits.looks}着までです。")
        self._world_confirmed(draft)
        if state["activeJobId"] or state["queue"]:
            raise ServiceError(409, "失敗または未完了の生成処理を解決してから承認してください。")
        if not draft["characters"] or any(
            not value["result"]
            or value["pendingChanges"]
            or not value["imageArtifactId"]
            or not value["voiceArtifactId"]
            or value["imagePendingChanges"]
            or value["voicePendingChanges"]
            for value in draft["characters"]
        ):
            raise ServiceError(
                409, "全キャラクターの設定・立ち絵・音声を生成して確認してください。"
            )
        if self._uses_looks(draft) and any(self._stale_looks(value) for value in draft["characters"]):
            raise ServiceError(409, "別バージョンの立ち絵を現在の立ち絵から生成するか、削除してください。")
        relationships = draft["relationships"]
        if len(draft["characters"]) >= 2:
            if (
                relationships["pendingChanges"]
                or not relationships["artifactId"]
                or not relationships["result"]
            ):
                raise ServiceError(409, "全キャラクター間の関係性を生成して確認してください。")
            try:
                validate_relationship_coverage(relationships["result"], draft["characters"])
            except (ValidationError, ValueError) as exc:
                raise ServiceError(409, "関係性が現在のキャラクター構成と一致しません。") from exc
        if draft["approved"] and not reconfirm:
            draft["step"] = ("planning-review" if draft.get("planningRequired")
                             and not draft.get("planApproved") else "production")
            return
        draft.update(step="planning-review", planningRequired=True, planApproved=False)
        draft["revision"] += 1
        identifier, now = uuid4().hex, self.clock()

        def reference(artifact_id):
            value = required(connection, "artifact", artifact_id)
            if value["project_id"] != project_id:
                raise ServiceError(409, "承認対象の素材が作品と一致しません。")
            self.store.read(value)
            return {
                "artifactId": value["id"],
                "version": value["version"],
                "sha256": value["sha256"],
            }

        snapshot = {
            "schema_version": 1,
            "id": identifier,
            "projectId": project_id,
            "revision": draft["revision"],
            "approvedAt": iso(now),
            "status": "ready",
            "world": {"result": draft["worldResult"], **reference(draft["worldArtifactId"])},
            "worldConfirmation": reference(draft["worldConfirmationArtifactId"]),
            "characters": [
                {
                    "id": value["id"],
                    "result": value["result"],
                    "locked": value["locked"],
                    **reference(value["resultArtifactId"]),
                    "image": reference(value["imageArtifactId"]),
                    "voice": reference(value["voiceArtifactId"]),
                    # Present only for a character with an alternate look.
                    **({"outfits": self._approved_outfits(value, reference)}
                       if self._alternates(value) and self._uses_looks(draft) else {}),
                }
                for value in draft["characters"]
            ],
        }
        if len(draft["characters"]) >= 2:
            snapshot["relationships"] = {
                "result": relationships["result"],
                **reference(relationships["artifactId"]),
            }
        record = self._artifact(
            connection,
            project_id,
            "approval",
            "m2_approval",
            "approval.json",
            "application/json",
            encode_json(snapshot),
        )
        connection.execute(
            "INSERT INTO m2_approval VALUES (?,?,?,?,?)",
            (
                identifier,
                project_id,
                draft["revision"],
                record["id"],
                now,
            ),
        )
        draft["approval"] = {**snapshot, "artifactId": record["id"], "version": record["version"]}
        draft["approved"] = True
        from .planning_service import PlanningService

        PlanningService(self.coordinator).start_approved(connection, project_id, identifier, cast_limit=cast_limit)

    def validate_retry(self, connection, job: dict) -> None:
        state = self._load(connection, job["project_id"])
        if (
            state["activeJobId"] != job["id"]
            or json.loads(job["payload"])["base_revision"] != state["draft"]["revision"]
        ):
            raise ServiceError(
                409, "設定が変更済みのため、以前の生成は再試行できません。新しく生成してください。"
            )
        self._assert_mutable(connection, job["project_id"])

    def _adopt(self, connection, job, attempt, state, envelope, files) -> None:
        draft = state["draft"]
        payload = json.loads(job["payload"])
        character = (
            self._character(draft, payload["character_id"]) if payload["character_id"] else None
        )
        if job["kind"] == "m2_world":
            from .material_service import MaterialService

            self._result(
                connection,
                job["project_id"],
                draft,
                None,
                envelope["result"],
                envelope=envelope,
                job=job,
                attempt=attempt,
            )
            draft["step"] = "world-review"
            # The declared shared materials are drawn now, to be reviewed together with the world.
            MaterialService(self.coordinator).ensure_sheets(
                connection, job["project_id"], world=draft["worldResult"],
                # Made again from the brief, the world gets its materials again; a revision by
                # instruction keeps the sheets the user has already looked at.
                redraw=not (payload.get("instruction") or "").strip())
        elif job["kind"] == "m2_character":
            result = envelope["result"]
            previous = character["result"]
            if previous:
                for scope, fields in GROUPS.items():
                    if character["locked"][scope] or payload["scope"] not in ("all", scope):
                        for field in fields:
                            if field in previous:
                                result[field] = previous[field]
                            else:
                                result.pop(field, None)
                if character["locked"]["voice"]:
                    if "selfIntroduction" in previous:
                        result["selfIntroduction"] = previous["selfIntroduction"]
                    else:
                        result.pop("selfIntroduction", None)
            result["locked"] = dict(character["locked"])
            self._result(
                connection,
                job["project_id"],
                draft,
                character,
                result,
                envelope=envelope,
                job=job,
                attempt=attempt,
            )
            state["queue"].extend(self._missing_media(character))
            # Settings that left the portrait as it was still owe it a shared sheet drawn since.
            state["queue"].extend(self._outfit_tasks(connection, job["project_id"], state, character))
        elif job["kind"] == "m2_relationships":
            result = validate_relationship_coverage(envelope["result"], draft["characters"])
            record = self._artifact(
                connection,
                job["project_id"],
                "relationships",
                "m2_relationships",
                "relationships.json",
                "application/json",
                encode_json({**envelope, "result": result}),
                provenance=envelope["provenance"],
                job=job,
                attempt=attempt,
            )
            draft["relationships"] = {
                "result": result,
                "artifactId": record["id"],
                "version": record["version"],
                "pendingChanges": False,
            }
        elif job["kind"] == "m2_outfit_image" and payload.get("expression"):
            # One expression of one look. It records the portrait it was edited from, and is
            # shown only while that portrait is the look's own.
            look, expression = payload["outfit"]["id"], payload["expression"]["id"]
            record = self._artifact(
                connection, job["project_id"], f"expression-{character['id']}-{look}-{expression}",
                "m2_expression_image", "image.png", "image/png", files["image.png"],
                provenance=envelope["provenance"], job=job, attempt=attempt)
            character["expressions"] = [
                *(row for row in character.get("expressions", [])
                  if (row["outfitId"], row["expression"]) != (look, expression)),
                {"outfitId": look, "expression": expression, "imageArtifactId": record["id"],
                 "baseImageArtifactId": payload["reference_image"]["artifact_id"]}]
            self._artifact(
                connection, job["project_id"], f"result-{job['id']}", "m2_metadata", "result.json",
                "application/json", files["result.json"], provenance=envelope["provenance"],
                job=job, attempt=attempt)
        elif job["kind"] == "m2_outfit_image" and payload["outfit"]["id"] == DEFAULT_OUTFIT:
            # The base portrait wearing a shared design replaces it; every look is derived again.
            record = self._artifact(
                connection, job["project_id"], f"image-{character['id']}", "m2_image", "image.png",
                "image/png", files["image.png"], provenance=envelope["provenance"], job=job, attempt=attempt)
            character.update(imageArtifactId=record["id"], imagePendingChanges=False)
            base = next(item for item in character["outfits"] if item["id"] == DEFAULT_OUTFIT)
            base["generatedDescription"] = (envelope["provenance"].get("outfit", {}).get("summary")
                                            or base.get("generatedDescription"))
            if base.get("materialBrief"):
                base["materialApplied"] = base["materialBrief"]
            worn = payload.get("material_image")
            if payload.get("edit") or not worn:
                # Edited by hand, the portrait as it stands is the one to keep working from.
                base.pop("plainArtifactId", None)
                base.pop("materialArtifactId", None)
            else:
                base["materialArtifactId"] = worn["artifact_id"]
                source = required(connection, "artifact", payload["reference_image"]["artifact_id"])
                drawn = required(connection, "job", source["source_job_id"])["kind"] if source["source_job_id"] else None
                if drawn == "m2_image":
                    base["plainArtifactId"] = source["id"]
            self._stale_outfits(character)
            state["queue"].extend(self._outfit_tasks(connection, job["project_id"], state, character))
            self._artifact(
                connection, job["project_id"], f"result-{job['id']}", "m2_metadata", "result.json",
                "application/json", files["result.json"], provenance=envelope["provenance"],
                job=job, attempt=attempt)
        elif job["kind"] == "m2_outfit_image":
            outfit = next(item for item in self._alternates(character)
                          if item["id"] == payload["outfit"]["id"])
            record = self._artifact(
                connection, job["project_id"], f"outfit-{character['id']}-{outfit['id']}",
                job["kind"], "image.png", "image/png", files["image.png"],
                provenance=envelope["provenance"], job=job, attempt=attempt)
            made = envelope["provenance"].get("outfit", {})
            if payload.get("edit"):
                # A hand edit of some picture of this person: it stays through later base portraits.
                outfit.update(imageArtifactId=record["id"], imagePendingChanges=False, handPicked=True,
                              baseImageArtifactId=character["imageArtifactId"],
                              generatedDescription=made.get("summary") or outfit.get("generatedDescription"))
            else:
                outfit.pop("handPicked", None)
                outfit["materialArtifactId"] = (payload.get("material_image") or {}).get("artifact_id")
                outfit.update(imageArtifactId=record["id"], imagePendingChanges=False,
                              baseImageArtifactId=payload["reference_image"]["artifact_id"],
                              generatedDescription=made.get("summary"), generatedOccasion=made.get("occasion"))
            self._artifact(
                connection, job["project_id"], f"result-{job['id']}", "m2_metadata", "result.json",
                "application/json", files["result.json"], provenance=envelope["provenance"],
                job=job, attempt=attempt)
        else:
            is_image = job["kind"] == "m2_image"
            prefix, filename, media_type = (
                ("image", "image.png", "image/png")
                if is_image
                else ("voice", "voice.wav", "audio/wav")
            )
            is_trial = job["kind"] == "m2_voice_clone"
            record = self._artifact(
                connection,
                job["project_id"],
                f"voice-trial-{job['id']}" if is_trial else f"{prefix}-{character['id']}",
                job["kind"],
                filename,
                media_type,
                files[filename],
                provenance=envelope["provenance"],
                job=job,
                attempt=attempt,
            )
            if is_trial:
                character["voiceTests"].append(
                    {
                        "id": job["id"],
                        "text": payload["dialogue_text"],
                        "artifactId": record["id"],
                        "url": f"/api/artifacts/{record['id']}/content",
                        "sourceVoiceArtifactId": payload["reference_voice"]["artifact_id"],
                        "createdAt": iso(self.clock()),
                    }
                )
            else:
                character[f"{prefix}ArtifactId"] = record["id"]
                character[f"{prefix}PendingChanges"] = False
                if is_image:
                    # A newly drawn portrait wears what its settings say, not the shared design yet.
                    for outfit in character.get("outfits", []):
                        if outfit["id"] == DEFAULT_OUTFIT:
                            for key in ("materialApplied", "generatedDescription", "materialArtifactId",
                                        "plainArtifactId"):
                                outfit.pop(key, None)
                    self._stale_outfits(character)
                    state["queue"].extend(self._outfit_tasks(connection, job["project_id"], state, character))
            self._artifact(
                connection,
                job["project_id"],
                f"result-{job['id']}",
                "m2_metadata",
                "result.json",
                "application/json",
                files["result.json"],
                provenance=envelope["provenance"],
                job=job,
                attempt=attempt,
            )
        if job["kind"] in ("m2_world", "m2_character", "m2_relationships"):
            self._remember_instruction(
                state, job["kind"], character_id=payload["character_id"],
                scope=payload["scope"], instruction=payload.get("instruction", ""),
            )
        draft["revision"] += 1
        state["activeJobId"] = None

    def complete(self, job_id: str, worker_id: str, lease_id: str, data: bytes) -> dict:
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(
                connection, job_id, worker_id, lease_id, completed_ok=True
            )
            if job["kind"] not in (
                "m2_world",
                "m2_character",
                "m2_image",
                "m2_voice",
                "m2_relationships",
                "m2_voice_clone",
                "m2_outfit_image",
            ):
                raise ServiceError(422, "M2の生成ジョブを指定してください。")
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            payload = json.loads(job["payload"])
        try:
            envelope, files = validate_bundle(
                data,
                job["kind"],
                payload["character_id"],
                payload.get("character_contract_version", 1),
            )
            if job["kind"] == "m2_relationships":
                validate_relationship_coverage(envelope["result"], payload["cast_inputs"])
            if job["kind"] == "m2_voice":
                introduction = (payload.get("character_result") or {}).get("selfIntroduction")
                voice_metadata = envelope["provenance"].get("voice", {})
                if not isinstance(voice_metadata, dict):
                    raise ValueError("voice provenance must be an object")
                actual = voice_metadata.get("reference_text")
                if introduction and actual != introduction:
                    raise ValueError("reference speech must use the character self-introduction")
            elif job["kind"] == "m2_outfit_image":
                source = envelope["provenance"].get("outfit", {})
                if (not isinstance(source, dict) or source.get("id") != payload["outfit"]["id"]
                        or source.get("reference_sha256") != payload["reference_image"]["sha256"]):
                    raise ValueError("outfit image provenance must match its base portrait")
                wanted = (payload.get("expression") or {}).get("id")
                if (source.get("expression") or {}).get("id") != wanted:
                    raise ValueError("expression picture differs from the expression asked for")
            elif job["kind"] == "m2_voice_clone":
                voice_metadata = envelope["provenance"].get("voice", {})
                if (
                    not isinstance(voice_metadata, dict)
                    or voice_metadata.get("text") != payload["dialogue_text"]
                ):
                    raise ValueError("voice clone output metadata must match requested speech")
                reference = payload["reference_voice"]
                if any(
                    voice_metadata.get(key) != reference[source]
                    for key, source in (
                        ("reference_artifact_id", "artifact_id"),
                        ("reference_sha256", "sha256"),
                        ("reference_text", "text"),
                    )
                ):
                    raise ValueError("voice clone provenance must match its immutable reference")
        except (
            ValueError,
            TypeError,
            KeyError,
            zipfile.BadZipFile,
            EOFError,
            wave.Error,
            zlib.error,
            RuntimeError,
            NotImplementedError,
        ) as exc:
            raise ServiceError(
                422, "生成結果が完全なJSON・透過PNG・PCM音声の契約を満たしていません。"
            ) from exc
        stored = self.store.put(data)
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(
                connection, job_id, worker_id, lease_id, completed_ok=True
            )
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            state = self._load(connection, job["project_id"])
            if (
                state["activeJobId"] != job_id
                or state["draft"]["revision"] != payload["base_revision"]
            ):
                raise ServiceError(409, "古い設定に対する生成結果は採用できません。")
            artifact = self.coordinator._register_artifact(
                connection,
                job["project_id"],
                f"m2-job-{job_id}",
                "m2_bundle",
                "m2-result.zip",
                "application/zip",
                stored,
                source_job_id=job_id,
                source_attempt_id=attempt["id"],
            )
            connection.execute(
                "UPDATE artifact SET provenance=? WHERE id=?",
                (
                    json.dumps(envelope["provenance"], ensure_ascii=False),
                    artifact["id"],
                ),
            )
            self._adopt(connection, job, attempt, state, envelope, files)
            now = self.clock()
            connection.execute(
                "UPDATE job_attempt SET status='completed',ended_at=? WHERE id=?",
                (now, attempt["id"]),
            )
            connection.execute(
                "UPDATE job SET status='completed',result_artifact_id=?,updated_at=?,error=NULL WHERE id=?",
                (artifact["id"], now, job_id),
            )
            connection.execute(
                "UPDATE generation_run SET status='completed' WHERE id=?", (job["run_id"],)
            )
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            self._queue_next(connection, job["project_id"], state)
            self._save(connection, job["project_id"], state)
            return {
                "job": public(required(connection, "job", job_id)),
                "artifact": public(required(connection, "artifact", artifact["id"])),
            }
