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
        draft.setdefault("relationships", relationship_state(draft["characters"]))
        for character in draft["characters"]:
            character.setdefault("voiceTests", [])
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

    def _view(self, connection: sqlite3.Connection, project_id: str, state: dict) -> dict:
        jobs = [
            public(dict(row))
            for row in connection.execute(
                "SELECT * FROM job WHERE project_id=? ORDER BY created_at,id", (project_id,)
            )
        ]
        project = public(required(connection, "project", project_id))
        active = next((job for job in jobs if job["id"] == state["activeJobId"]), None)
        project["status"] = (
            "ready" if state["draft"]["approved"] else active["status"] if active else "draft"
        )
        project["updated_at"] = iso(
            connection.execute(
                "SELECT updated_at FROM m2_draft WHERE project_id=?", (project_id,)
            ).fetchone()[0]
        )
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
        return {"project": project, "draft": draft, "jobs": jobs}

    def projects(self) -> list[dict]:
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            ids = [
                row[0]
                for row in connection.execute(
                    "SELECT project_id FROM m2_draft ORDER BY updated_at DESC,project_id"
                )
            ]
            return [
                self._view(connection, identifier, self._load(connection, identifier))["project"]
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
                "wizardStepVersion": 2,
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
        if draft["step"] == "production":
            draft["step"] = (
                "character-review" if draft["worldConfirmed"] and not draft["worldPendingChanges"]
                else "world-review" if draft["worldResult"] else "world-input"
            )
        state["activeJobId"] = None
        state["queue"] = []

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

    def _profile(self) -> dict:
        from .llm_settings import saved_settings
        common = saved_settings(self.coordinator)
        if common is not None:
            return {**common.profile(), "max_tokens": 3072, "prompt_version": 4}
        configuration = json.loads((ROOT / "config/m2-generation.json").read_text(encoding="utf-8"))
        llm = configuration["llm"]
        return {
            "provider": "local",
            "model_id": llm["model_id"],
            "temperature": llm["temperature"],
            "max_tokens": llm["max_tokens"],
            "reasoning_level": llm["reasoning_level"],
            "prompt_version": 4,
        }

    def _queue_next(self, connection: sqlite3.Connection, project_id: str, state: dict) -> None:
        if not state["queue"]:
            state["activeJobId"] = None
            return
        descriptor = state["queue"].pop(0)
        draft = state["draft"]
        character = (
            self._character(draft, descriptor["character_id"])
            if descriptor["character_id"]
            else None
        )
        profile = self._profile()
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
        for key in ("reference_voice", "dialogue_text"):
            if key in descriptor:
                payload[key] = descriptor[key]
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
                        "profile": profile,
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
            if name == "go-to":
                step = "world-input" if body.step == "character-input" else body.step
                if step == "character-review":
                    self._world_confirmed(draft)
                if step == "production" and not (
                    draft["approved"] or self._has_production(connection, project_id)
                ):
                    raise ServiceError(409, "キャラクターと素材を確認・承認してから制作へ進んでください。")
                draft["step"] = step
                # Navigation does not change generation inputs or invalidate approvals/retries.
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if (
                name == "confirm-world"
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
            if name == "approve" and draft["approved"]:
                self._approve(connection, project_id, state)
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if name == "edit-character":
                character = self._character(draft, body.character_id)
                if (character["result"] and not character["pendingChanges"]
                        and all(character["result"].get(key) == value for key, value in body.patch.items())):
                    return self._view(connection, project_id, state)
            from .project_history import HistoryService, operation_label

            HistoryService(self.coordinator).begin(
                connection, project_id, operation_label(body, state), name,
            )
            if name == "save-brief":
                previous = (
                    draft["worldInput"],
                    [copy.deepcopy(value["input"]) for value in draft["characters"]],
                    draft["relationshipInputs"],
                )
                self._save_characters(draft, body)
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
                draft["step"] = "world-input"
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if name == "clone-voice":
                self._clone_voice(connection, project_id, state, body)
                self._queue_next(connection, project_id, state)
                self._save(connection, project_id, state)
                return self._view(connection, project_id, state)
            if name == "approve":
                self._approve(connection, project_id, state)
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
                if not draft["worldConfirmed"]:
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
                self._queue_characters(state)
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

    def _queue_characters(self, state: dict, *, force: bool = False) -> None:
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
            if self._save_characters(draft, body):
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
            self._queue_characters(state, force=True)
            return
        character = self._character(draft, body.character_id)
        if not character["result"]:
            raise ServiceError(409, "先にキャラクターを生成してください。")
        if name == "toggle-lock":
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

    def _approve(self, connection, project_id: str, state: dict) -> None:
        draft = state["draft"]
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
        draft["step"] = "production"
        if draft["approved"]:
            return
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
        from .m3_service import M3Service

        M3Service(self.coordinator).start_approved(connection, project_id, identifier)

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
            state["queue"] = self._missing_media(character) + state["queue"]
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
