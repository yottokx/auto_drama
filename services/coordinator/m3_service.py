"""Approval-frozen, durable first-chapter generation and atomic publication."""

from __future__ import annotations

import hashlib
import json
import wave
import zipfile
import zlib
from uuid import uuid4

from packages.contracts import Script
from packages.contracts.m3 import (
    EMOTION_TAGS,
    M3_KINDS,
    PortraitSetting,
    PortraitSettingsUpdate,
    ProductionPlot,
    ProductionPortraits,
)
from packages.contracts.portrait_recovery import PortraitOmission
from packages.narrative.continuity import narrative_hash
from packages.narrative.script_conversion import narrative_to_script
from packages.narrative.validation import (
    approved_characters,
    script_character_id,
    validate_narrative,
)
from packages.tyrano_export import compile_bundle
from services.worker.generation.workflow_version import generator_protocol

from .m2_bundle import validate_png, validate_wav
from .m2_service import ROOT as GENERATION_ROOT
from .m2_service import M2Service
from .m3_bundle import validate_background, validate_bundle
from .m4_service import ChapterProduction
from .project_history import HistoryService
from .service import Coordinator, ServiceError, encode_json, public, required


def stable_id(prefix: str, value: str) -> str:
    return prefix + "-" + hashlib.sha256(value.encode()).hexdigest()[:24]


def _checkpoint_digest(value: dict) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _validate_script_checkpoint(payload, narrative, provenance):
    checkpoint = provenance.get("script_checkpoint")
    if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get("state"), dict):
        raise TypeError("Script result lacks a portable continuation checkpoint.")
    expected = {
        "schema_version": 1,
        "generator_protocol": payload["generator_protocol"],
        "storyline_id": payload["storyline_id"],
        "approval_sha256": _checkpoint_digest(payload["approval_snapshot"]),
        "chapter_number": narrative.chapter_number,
        "narrative_hash": narrative_hash(narrative),
    }
    if any(checkpoint.get(key) != value for key, value in expected.items()):
        raise ValueError("Script checkpoint identity differs from its adopted chapter.")
    if payload.get("approved_plan"):
        from packages.contracts.planning import validate_plan_content

        plan = payload["approved_plan"]
        if (checkpoint.get("plan_approval_id") != plan["approval_id"]
                or checkpoint.get("plan_sha256") != plan["sha256"]):
            raise ValueError("Script checkpoint differs from the approved common plan.")
        approved = validate_plan_content(plan["content"], payload["approval_snapshot"])
        if narrative.outline != approved.plot.as_outline():
            raise ValueError("Narrative outline differs from the approved common plan.")
        planned_cast = {row.id: row for row in approved.cast_plan.supporting_characters}
        if any(row.id in planned_cast and row != planned_cast[row.id]
               for row in narrative.supporting_characters):
            raise ValueError("Previously planned character settings cannot be replaced by chapter writing.")
    if (not isinstance(checkpoint.get("generation_identity"), str)
            or len(checkpoint["generation_identity"]) != 64
            or checkpoint.get("sha256") != _checkpoint_digest(
                {key: value for key, value in checkpoint.items() if key != "sha256"})):
        raise ValueError("Script checkpoint identity or content hash is invalid.")


class M3Service(ChapterProduction):
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.db, self.store, self.clock = coordinator.db, coordinator.store, coordinator.clock
        self.history = HistoryService(coordinator)

    def _selected_production(self, connection, project_id):
        selection = self.history.selection(connection, project_id)
        if selection is None:
            return connection.execute(
                "SELECT * FROM m3_production WHERE project_id=? AND chapter_number=1 "
                "ORDER BY created_at DESC,id DESC LIMIT 1",
                (project_id,),
            ).fetchone()
        if selection["production_id"] is None:
            return None
        production = required(connection, "m3_production", selection["production_id"])
        if production["project_id"] != project_id:
            raise ServiceError(422, "選択された制作版が作品と一致しません。")
        if selection["production_frozen"] and selection["production_snapshot"]:
            frozen = json.loads(selection["production_snapshot"])
            production = {**production, "narrative_artifact_id": frozen["narrative_artifact_id"],
                          "error": frozen["error"]}
        return production

    def _selected_build(self, connection, production):
        selection = self.history.selection(connection, production["project_id"])
        if selection and selection.get("edition_id"):
            build = connection.execute(
                "SELECT b.* FROM edition_build eb JOIN chapter_build b ON b.id=eb.build_id "
                "JOIN publication_edition e ON e.id=eb.edition_id "
                "WHERE eb.edition_id=? AND eb.production_id=? AND e.project_id=?",
                (selection["edition_id"], production["id"], production["project_id"]),
            ).fetchone()
            if build is not None:
                return build
            raise ServiceError(422, "調整版の章が見つかりません。")
        if production["chapter_number"] > 1:
            if selection and selection["production_frozen"]:
                frozen = json.loads(selection["production_snapshot"] or "{}")
                saved = next((item for item in frozen.get("chapters", [])
                              if item["id"] == production["id"]), None)
                return (required(connection, "chapter_build", saved["build_id"])
                        if saved and saved.get("build_id") else None)
            return self._latest_build(connection, production["id"])
        if selection is None:
            return self._latest_build(connection, production["id"])
        if selection["production_id"] != production["id"] or selection["build_id"] is None:
            return None
        build = required(connection, "chapter_build", selection["build_id"])
        if (build["production_id"] != production["id"]
                or build["project_id"] != production["project_id"] or build["status"] != "published"):
            raise ServiceError(422, "選択された公開版が制作版と一致しません。")
        return build

    def _is_selected(self, connection, production):
        selected = self._selected_production(connection, production["project_id"])
        return selected is not None and selected["id"] == production["storyline_id"]

    def _has_current_plan(self, connection, production):
        root = self._root(connection, production)
        if not root["plan_approval_id"]:
            return True
        return connection.execute(
            "SELECT 1 FROM project_history_state h JOIN planning_draft p ON p.id=h.planning_id "
            "WHERE h.project_id=? AND p.approved_plan_id=? AND p.main_approval_id=?",
            (root["project_id"], root["plan_approval_id"], root["approval_id"]),
        ).fetchone() is not None

    def _assert_current_plan(self, connection, production):
        if not self._has_current_plan(connection, production):
            raise ServiceError(409, "構成を変更しています。STEP4の新しい構成を承認してから制作してください。")

    def _assert_resumable(self, connection, production):
        production = self._root(connection, production)
        selection = self.history.selection(connection, production["project_id"])
        if (selection is None or not selection["production_frozen"]
                or not selection["production_snapshot"]):
            return
        frozen = json.loads(selection["production_snapshot"])
        live = self.history.production_state(connection, production["id"])
        if (live["narrative_artifact_id"] != frozen["narrative_artifact_id"]
                or live["requirements"] != frozen["requirements"]
                or live.get("chapters", []) != frozen.get("chapters", [])):
            raise ServiceError(
                409, "この制作版には後から生成された内容があります。再開するには新しい履歴を選んでください。"
            )

    def _snapshot(self, connection, production: dict) -> dict:
        record = required(connection, "artifact", production["approval_artifact_id"])
        if record["project_id"] != production["project_id"] or record["kind"] != "m2_approval":
            raise ServiceError(422, "承認版が作品と一致しません。")
        return json.loads(self.store.read(record))

    def _approved_plan(self, connection, production):
        identifier = production["plan_approval_id"]
        if not identifier:
            return None
        approval = required(connection, "planning_approval", identifier)
        artifact = required(connection, "artifact", approval["artifact_id"])
        value = json.loads(self.store.read(artifact))
        if (approval["project_id"] != production["project_id"] or artifact["kind"] != "planning_approval"
                or value["main_approval_id"] != production["approval_id"]
                or value["approval_id"] != identifier or value["sha256"] != approval["sha256"]
                or _checkpoint_digest(value["content"]) != approval["sha256"]):
            raise ServiceError(422, "承認した構成版が制作系列と一致しません。")
        return value

    def _artifact(
        self,
        connection,
        production,
        logical_id,
        kind,
        filename,
        data,
        *,
        provenance=None,
        job=None,
        attempt=None,
    ):
        media_type = (
            "image/png"
            if filename.endswith(".png")
            else "audio/wav"
            if filename.endswith(".wav")
            else "audio/mpeg"
            if filename.endswith(".mp3")
            else "application/zip"
            if filename.endswith(".zip")
            else "text/plain; charset=utf-8"
            if filename.endswith(".txt")
            else "application/json"
        )
        record = self.coordinator._register_artifact(
            connection,
            production["project_id"],
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
                json.dumps(
                    provenance or {"producer": "m3/1", "approval_id": production["approval_id"]},
                    ensure_ascii=False,
                ),
                record["id"],
            ),
        )
        return required(connection, "artifact", record["id"])

    def _enqueue(self, connection, production, kind, descriptor):
        snapshot = self._snapshot(connection, production)
        identifier, run_id, now = uuid4().hex, uuid4().hex, self.clock()
        narrative_job = connection.execute(
            "SELECT job.* FROM job JOIN m3_production_job p ON p.job_id=job.id "
            "WHERE p.production_id=? AND job.kind='m3_narrative'",
            (production["id"],),
        ).fetchone()
        profile_job = narrative_job or connection.execute(
            "SELECT job.* FROM job JOIN m3_production_job p ON p.job_id=job.id "
            "WHERE p.production_id=? AND job.kind='m3_narrative'",
            (production["storyline_id"],),
        ).fetchone()
        approved_plan = self._approved_plan(connection, production)
        plan_generation = approved_plan.get("generation", {}) if approved_plan else {}
        profile = (
            json.loads(profile_job["settings_snapshot"])["profile"]
            if profile_job
            else plan_generation.get("profile", M2Service(self.coordinator)._profile())
        )
        from .tts_settings import TTSService

        # The first narrative fixes both voice models for the entire production,
        # including asset jobs and every later chapter.
        if profile_job:
            tts_profile = json.loads(profile_job["settings_snapshot"]).get("tts_profile")
        else:
            tts_profile = plan_generation.get("tts_profile", TTSService(self.coordinator).profile(connection))
        workflow = {key: plan_generation[key] for key in (
            "story_workflow_version", "workflow_policy", "generator_protocol", "script_options",
            "profiles", "workflow_limits", "seed",
        ) if key in plan_generation} if kind == "m3_narrative" else {}
        if kind == "m3_narrative" and profile_job:
            pinned = json.loads(profile_job["payload"])
            workflow = {key: pinned[key] for key in (
                "story_workflow_version", "workflow_policy", "generator_protocol", "script_options",
                "profiles", "workflow_limits",
            ) if key in pinned}
            if pinned.get("workflow_policy") == "script_continuation_v1":
                workflow["seed"] = pinned["seed"]
        if kind == "m3_narrative" and descriptor.get("workflow_policy") == "script_continuation_v1":
            # The generation route owns its per-purpose output budget. M2's
            # uniform response limit would otherwise replace every script budget.
            profile = {key: value for key, value in profile.items()
                       if key not in {"max_tokens", "prompt_version"}}
            settings = json.loads((GENERATION_ROOT / "config/m2-generation.json").read_text(
                encoding="utf-8"))["llm"]
            if not profile.get("common_settings_version"):
                profile.update(reasoning_level="none", context_size=settings.get("context_size", 16384))
        payload = {
            "schema_version": 1,
            "production_id": production["id"],
            "chapter_number": production["chapter_number"],
            "storyline_id": production["storyline_id"],
            "m4": bool(production["m4_enabled"]),
            "approval_snapshot": snapshot,
            "profile": profile,
            **({"tts_profile": tts_profile} if tts_profile is not None else {}),
            "seed": int(identifier[:8], 16) & 0x7FFFFFFF,
            **workflow,
            **descriptor,
        }
        if approved_plan:
            payload["approved_plan"] = {key: approved_plan[key] for key in ("content", "approval_id", "sha256")}
            payload["approved_plan"]["planning_protocol"] = plan_generation["planning_protocol"]
        if kind == "m3_image":
            # Translate all required portraits before loading the image model.
            # Approved main images have no character_result and are reused.
            batch = []
            for row in connection.execute(
                "SELECT target_id,descriptor FROM m3_requirement "
                "WHERE production_id=? AND kind='m3_image' ORDER BY target_id", (production["id"],)
            ):
                source = json.loads(row["descriptor"])
                if source.get("character_result"):
                    batch.append({"character_id": row["target_id"], "character_result": source["character_result"]})
            if batch:
                payload["portrait_prompt_batch"] = batch
        connection.execute(
            "INSERT INTO generation_run (id,project_id,settings_version,story_revision_id,"
            "policy,status,created_at) VALUES (?,?,?,?,?,'pending',?)",
            (
                run_id,
                production["project_id"],
                snapshot["revision"],
                production["storyline_id"],
                json.dumps({"mode": "m4-chapters", "approval_id": production["approval_id"]}),
                now,
            ),
        )
        connection.execute(
            "INSERT INTO job (id,project_id,run_id,kind,payload,settings_snapshot,priority,status,"
            "max_attempts,created_at,updated_at) VALUES (?,?,?,?,?,?,?,'pending',3,?,?)",
            (
                identifier,
                production["project_id"],
                run_id,
                kind,
                json.dumps(payload, ensure_ascii=False),
                json.dumps(
                    {
                        "schema_version": 1,
                        "profile": profile,
                        **({"tts_profile": tts_profile} if tts_profile is not None else {}),
                        "approval_artifact_id": production["approval_artifact_id"],
                        "seed": payload["seed"],
                    }
                ),
                101 - production["chapter_number"],
                now,
                now,
            ),
        )
        connection.execute(
            "INSERT INTO m3_production_job VALUES (?,?)", (production["id"], identifier)
        )
        dependencies = {narrative_job["id"]} if narrative_job else set()
        if descriptor.get("previous_narrative_artifact_id"):
            prior = required(connection, "artifact", descriptor["previous_narrative_artifact_id"])
            if prior["source_job_id"]:
                dependencies.add(prior["source_job_id"])
        if descriptor.get("reference_voice"):
            reference = required(
                connection, "artifact", descriptor["reference_voice"]["artifact_id"]
            )
            if reference["source_job_id"]:
                dependencies.add(reference["source_job_id"])
        for dependency in dependencies:
            connection.execute("INSERT INTO job_dependency VALUES (?,?)", (identifier, dependency))
        return identifier

    def start_approved(self, connection, project_id: str, approval_id: str, plan_approval_id=None) -> dict:
        """Start a pinned series atomically; saved legacy series omit the plan."""
        self.history.ensure(connection, project_id)
        approval = required(connection, "m2_approval", approval_id)
        if approval["project_id"] != project_id:
            raise ServiceError(409, "承認版が作品と一致しません。")
        existing = connection.execute(
            "SELECT * FROM m3_production WHERE approval_id=? AND chapter_number=1 "
            "AND plan_approval_id IS ?", (approval_id, plan_approval_id)
        ).fetchone()
        if existing:
            if not self._is_selected(connection, existing):
                raise ServiceError(409, "以前の制作版は再開できません。キャストを再承認してください。")
            return dict(existing)
        identifier = uuid4().hex
        if plan_approval_id:
            plan = required(connection, "planning_approval", plan_approval_id)
            draft = required(connection, "planning_draft", plan["planning_id"])
            if (plan["project_id"] != project_id or draft["main_approval_id"] != approval_id
                    or draft["approved_plan_id"] != plan_approval_id):
                raise ServiceError(409, "承認した構成とメイン設定が一致しません。")
        from .music_service import enabled_for_new_series

        connection.execute(
            "INSERT INTO m3_production (id,project_id,approval_id,approval_artifact_id,created_at,"
            "storyline_id,m4_enabled,plan_approval_id,sequential_publication,music_enabled) VALUES (?,?,?,?,?,?,1,?,?,?)",
            (identifier, project_id, approval_id, approval["artifact_id"], self.clock(), identifier,
             plan_approval_id, int(bool(plan_approval_id)), int(enabled_for_new_series(plan_approval_id))),
        )
        production = required(connection, "m3_production", identifier)
        self.history.set_selection(connection, project_id, identifier, None, None)
        descriptor = {} if plan_approval_id else {
            "story_workflow_version": 2,
            "workflow_policy": "script_continuation_v1",
            "generator_protocol": generator_protocol("causal", "script_continuation_v1"),
        }
        self._enqueue(connection, production, "m3_narrative", descriptor)
        return production

    def start(self, project_id: str) -> dict:
        with self.db.transaction() as connection:
            state = M2Service(self.coordinator)._load(connection, project_id)
            if not state["draft"]["approved"] or not state["draft"]["approval"]:
                raise ServiceError(
                    409, "世界観・キャスト・素材の最終承認後に本編制作を開始できます。"
                )
            plan_approval_id = None
            if state["draft"].get("planningRequired"):
                from .planning_service import PlanningService
                plan = PlanningService.selected(connection, project_id)
                if (not plan or not plan["approved_plan_id"] or not state["draft"].get("planApproved")
                        or plan["main_approval_id"] != state["draft"]["approval"]["id"]):
                    raise ServiceError(409, "先にSTEP4の全体構成を承認してください。")
                plan_approval_id = plan["approved_plan_id"]
            selected = self._selected_production(connection, project_id)
            selection = self.history.selection(connection, project_id)
            needs_start = (selected is None or selected["error"]
                           or selection is not None and selection["production_frozen"])
            if needs_start:
                if selected is not None:
                    self._assert_resumable(connection, selected)
                self.history.begin(connection, project_id, "本編制作を再開", "m3-start")
            production = self.start_approved(
                connection, project_id, state["draft"]["approval"]["id"], plan_approval_id
            )
            if needs_start:
                selection = self.history.selection(connection, project_id)
                self.history.set_selection(
                    connection, project_id, production["id"], selection["build_id"],
                    selection["portrait_settings_id"],
                )
            connection.execute("UPDATE m3_production SET error=NULL WHERE storyline_id=?",
                               (production["id"],))
            connection.execute("UPDATE m3_production SET control_state='running',m4_enabled=1 WHERE id=?",
                               (production["id"],))
        self.advance(production["id"])
        return self.project(project_id)

    def _jobs(self, connection, production_id):
        production = required(connection, "m3_production", production_id)
        selected = self.history.selection(connection, production["project_id"])
        frozen_jobs = None
        if (selected is not None and selected["production_id"] == production["storyline_id"]
                and selected["production_frozen"] and selected["production_snapshot"]
                and production["chapter_number"] > 1):
            frozen = json.loads(selected["production_snapshot"])
            chapter = next((item for item in frozen.get("chapters", [])
                            if item["id"] == production_id), {})
            frozen_jobs = {job["id"]: job for job in chapter.get("jobs", [])}
        if (selected is not None and selected["production_id"] == production_id
                and selected["production_frozen"] and selected["production_snapshot"]):
            frozen_jobs = {job["id"]: job for job in
                           json.loads(selected["production_snapshot"])["jobs"]}
        return [
            public({**dict(row), **(frozen_jobs[row["id"]] if frozen_jobs is not None else {}),
                    **({"progress": frozen_jobs[row["id"]].get("progress"), "progress_active": False}
                       if frozen_jobs is not None else {}),
                    **({"result_artifact_id": None} if frozen_jobs is not None
                       and frozen_jobs[row["id"]]["status"] != "completed" else {})})
            for row in connection.execute(
                "SELECT job.* FROM job JOIN m3_production_job p ON p.job_id=job.id "
                "WHERE p.production_id=? ORDER BY job.created_at,job.id",
                (production_id,),
            )
            if frozen_jobs is None or row["id"] in frozen_jobs
        ]

    @staticmethod
    def _latest_build(connection, production_id):
        return connection.execute(
            "SELECT * FROM chapter_build WHERE production_id=? ORDER BY revision DESC LIMIT 1",
            (production_id,),
        ).fetchone()

    def rebuild(self, project_id: str) -> dict:
        """Recompile adopted inputs atomically; never enqueue or regenerate anything."""
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            selection = self.history.selection(connection, project_id)
            if selection and selection.get("edition_id"):
                raise ServiceError(409, "調整版は「表示と素材を調整」から全章へ反映してください。")
            production = self._selected_production(connection, project_id)
            if not production or not self._selected_build(connection, production):
                raise ServiceError(409, "公開済みの章だけを組み立て直せます。")
            self.history.begin(connection, project_id, "章を組み立て直す", "m3-rebuild", concurrent=True)
            requirements = [
                dict(row)
                for row in connection.execute(
                    "SELECT * FROM m3_requirement WHERE production_id=? ORDER BY rowid",
                    (production["id"],),
                )
            ]
            if not production["narrative_artifact_id"] or not requirements or not all(
                item["artifact_id"] for item in requirements
            ):
                raise ServiceError(409, "採用済みの本文と素材が揃っていません。")
            try:
                self._publish(connection, production, requirements)
            except (OSError, ValueError, ServiceError, wave.Error, zlib.error) as exc:
                # The transaction rolls back all new records; the old build stays playable.
                raise ServiceError(
                    422, "章の再組み立てに失敗しました。公開済みの章はそのまま鑑賞できます。"
                ) from exc
            self.history.finish(connection, project_id)
        return self.project(project_id)

    def _portrait_settings(self, connection, production):
        production = self._root(connection, production)
        selection = self.history.selection(connection, production["project_id"])
        if selection is None:
            record = connection.execute(
                "SELECT * FROM artifact WHERE project_id=? AND logical_id=? "
                "AND kind='m3_portrait_settings' ORDER BY version DESC LIMIT 1",
                (production["project_id"], "portrait-settings-" + production["id"]),
            ).fetchone()
        else:
            record = (required(connection, "artifact", selection["portrait_settings_id"])
                      if selection["production_id"] == production["id"]
                      and selection["portrait_settings_id"] else None)
            if record and (record["project_id"] != production["project_id"]
                           or record["kind"] != "m3_portrait_settings"
                           or record["logical_id"] != "portrait-settings-" + production["id"]):
                raise ServiceError(422, "立ち絵の表示設定が制作版と一致しません。")
        values = json.loads(self.store.read(dict(record))) if record else []
        settings = [PortraitSetting.model_validate(value) for value in values]
        return record, {value.character_id: value for value in settings}

    def _published_portraits(self, connection, production, build):
        """Project the immutable build, even if newer inputs await publication."""
        record = required(connection, "artifact", build["script_artifact_id"])
        if (
            build["project_id"] != production["project_id"] or build["status"] != "published"
            or record["project_id"] != production["project_id"] or record["kind"] != "script"
            or record["logical_id"] != "chapter-" + production["id"]
        ):
            raise ServiceError(422, "公開版の人物情報が作品・制作版と一致しません。")
        script = Script.model_validate_json(self.store.read(record))
        originals = {
            script_character_id(row["target_id"]): row["target_id"]
            for row in connection.execute(
                "SELECT target_id FROM m3_requirement WHERE production_id=? AND kind='m3_image'",
                (production["id"],),
            )
        }
        assets = {value.id: value for value in script.assets}
        characters = []
        for character in script.characters:
            if character.image_asset_id is None:
                continue
            reference = assets[character.image_asset_id]
            image = required(connection, "artifact", reference.artifact_id)
            if (
                reference.kind != "character" or image["kind"] != "character"
                or image["project_id"] != production["project_id"]
                or image["sha256"] != reference.sha256
            ):
                raise ServiceError(422, "公開版の立ち絵が作品・種類・ハッシュと一致しません。")
            self.store.read(image)
            characters.append({
                "character_id": originals[character.id], "name": character.name,
                "image_artifact_id": reference.artifact_id,
                "image_url": f"/api/artifacts/{reference.artifact_id}/content",
                "framing": character.framing, "height_cm": character.height_cm,
                "body_bounds": character.body_bounds,
            })
        return characters

    def portraits(self, project_id: str) -> ProductionPortraits:
        """Inspect published presentation without recovery, adoption or generation."""
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            production = self._selected_production(connection, project_id)
            build = self._selected_build(connection, production) if production else None
            try:
                return ProductionPortraits.model_validate({
                    "project_id": project_id,
                    "production_id": production["id"] if production else None,
                    "build_id": build["id"] if build else None,
                    "characters": self._published_portraits(connection, production, build)
                    if build else [],
                })
            except (ValueError, KeyError, TypeError) as exc:
                raise ServiceError(422, "公開版の人物の表示設定を確認できません。") from exc

    def update_portraits(self, project_id: str, request: PortraitSettingsUpdate) -> dict:
        """Save presentation separately from approval and publish it atomically."""
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            selection = self.history.selection(connection, project_id)
            if selection and selection.get("edition_id"):
                raise ServiceError(409, "調整版は「表示と素材を調整」から全章へ反映してください。")
            production = self._selected_production(connection, project_id)
            latest = self._selected_build(connection, production) if production else None
            if latest is None or latest["id"] != request.expected_build_id:
                raise ServiceError(409, "最新の公開版を確認してから表示設定を変更してください。")
            self.history.begin(connection, project_id, "立ち絵の表示を調整", "m3-portraits", concurrent=True)
            snapshot = self._snapshot(connection, production)
            narrative = self._load_narrative(connection, production)
            cast_ids = {value["id"] for value in snapshot["characters"]}
            cast_ids.update(value.id for value in narrative.supporting_characters)
            if any(value.character_id not in cast_ids for value in request.characters):
                raise ServiceError(422, "表示設定に未登録の人物が含まれています。")
            requirements = [dict(row) for row in connection.execute(
                "SELECT * FROM m3_requirement WHERE production_id=? ORDER BY rowid",
                (production["id"],))]
            images = {value["target_id"]: value["artifact_id"] for value in requirements
                      if value["kind"] == "m3_image"}
            if any(value.image_artifact_id is not None
                   and value.image_artifact_id != images.get(value.character_id)
                   for value in request.characters):
                raise ServiceError(422, "基準範囲の立ち絵が現在の人物画像と一致しません。")
            current, _ = self._portrait_settings(connection, production)
            data = encode_json([value.model_dump(mode="json") for value in sorted(
                request.characters, key=lambda value: value.character_id)])
            try:
                if current is None or self.store.read(dict(current)) != data:
                    current = self._artifact(
                        connection, production, "portrait-settings-" + production["id"],
                        "m3_portrait_settings", "portrait-settings.json", data,
                        provenance={"producer": "portrait-settings/1",
                                    "original_build_id": latest["id"]})
                self.history.set_selection(
                    connection, project_id, production["id"], latest["id"], current["id"],
                )
                self._publish(connection, production, requirements)
            except (OSError, ValueError, ServiceError, wave.Error, zlib.error) as exc:
                raise ServiceError(422, "表示設定を反映できませんでした。元の公開版を保持します。") from exc
            self.history.finish(connection, project_id)
        return self.project(project_id)

    @staticmethod
    def _public_build(record):
        result = public(dict(record))
        result["player_url"] = f"/player/{record['id']}/"
        result["export_url"] = f"/api/artifacts/{record['export_artifact_id']}/content"
        for name in ("manifest", "validation"):
            result[name] = json.loads(result[name])
        return result

    def build(self, build_id: str) -> dict:
        with self.db.transaction() as connection:
            record = required(connection, "chapter_build", build_id)
            if record["status"] != "published":
                raise ServiceError(404, "公開済みの章が見つかりません。")
            return self._public_build(record)

    def project(self, project_id: str) -> dict:
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            self.coordinator._recover(connection)
            row = self._selected_production(connection, project_id)
            if not row:
                return {"project_id": project_id, "production": None}
            production_id = row["id"]
        # Repair the crash window between committed result adoption and next-job/build creation.
        self.advance(production_id)
        with self.db.transaction() as connection:
            row = self._selected_production(connection, project_id)
            if row is None:
                return {"project_id": project_id, "production": None}
            production = public(dict(row))
            production["music_enabled"] = bool(row["music_enabled"])
            production_id = production["id"]
            selected = self.history.selection(connection, project_id)
            production["history_frozen"] = bool(selected and selected["production_frozen"])
            chapters = self._chapters(connection, row)
            from .music_service import plan_metadata

            presentation = json.loads(required(connection, "publication_edition", selected["edition_id"])["state"]) if (
                selected and selected.get("edition_id")) else None
            for chapter in chapters:
                chapter["music_enabled"] = production["music_enabled"]
                chapter["scenes"] = []
                if not chapter["narrative_artifact_id"]:
                    continue
                adopted = required(connection, "m3_production", chapter["production_id"])
                adopted["narrative_artifact_id"] = chapter["narrative_artifact_id"]
                metadata = plan_metadata(connection, adopted, self, presentation)
                chapter["scenes"] = [{"scene_id": scene.id, "music_plan": metadata[scene.id]}
                                     for scene in self._load_narrative(connection, adopted).scenes]
            production["chapters"] = chapters
            production["chapter_count"] = len(chapters)
            self._settle_control(connection, row)
            production["control_state"] = required(connection, "m3_production", production_id)["control_state"]
            jobs = [job for chapter in chapters for job in chapter["jobs"]]
            build = self._selected_build(connection, production)
            if production["history_frozen"] or (not production["m4_enabled"] and build):
                production["control_state"] = "paused"
            production["jobs"] = jobs
            production["chapter_number"] = 1
            production["completed_jobs"] = sum(job["status"] == "completed" for job in jobs)
            production["total_jobs"] = len(jobs)
            production["build"] = self._public_build(build) if build else None
            failed = next((job for job in jobs if job["status"] == "failed"), None)
            production["status"] = (
                "published"
                if all(chapter["build"] for chapter in chapters)
                else "failed"
                if failed or any(chapter["error"] for chapter in chapters)
                else "running"
                if any(j["status"] == "running" for j in jobs)
                else "pending"
            )
            production["stage"] = (
                "published"
                if production["status"] == "published"
                else "assets"
                if production["narrative_artifact_id"]
                else "narrative"
            )
            production["error"] = next((c["error"] for c in chapters if c["error"]), None)
            production["player_url"] = f"/player/{build['id']}/" if build else None
            production["export_url"] = (
                f"/api/artifacts/{build['export_artifact_id']}/content" if build else None
            )
            production["chapters_export_url"] = (
                f"/api/m3/builds/{build['id']}/export" if build else None)
            return {"project_id": project_id, "production": production}

    def plot(self, project_id: str) -> ProductionPlot:
        """Read adopted plot only; inspecting it must never advance generation."""
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            production = self._selected_production(connection, project_id)
            result = {
                "project_id": project_id,
                "production_id": production["id"] if production else None,
                "narrative_artifact_id": production["narrative_artifact_id"] if production else None,
                "plot": None,
            }
            if not production or not production["narrative_artifact_id"]:
                return ProductionPlot.model_validate(result)
            record = required(connection, "artifact", production["narrative_artifact_id"])
            if (
                record["project_id"] != project_id or record["kind"] != "m3_narrative"
                or record["logical_id"] != "narrative-" + production["id"]
            ):
                raise ServiceError(422, "採用済みの物語が作品・制作版と一致しません。")
            approval = required(connection, "m2_approval", production["approval_id"])
            if (
                approval["project_id"] != project_id
                or approval["artifact_id"] != production["approval_artifact_id"]
            ):
                raise ServiceError(422, "承認版が作品・制作版と一致しません。")
            try:
                snapshot = self._snapshot(connection, production)
                if snapshot["projectId"] != project_id or snapshot["id"] != production["approval_id"]:
                    raise ValueError("Approval identity differs from its production.")
                narrative = self._load_narrative(connection, production)
                result["plot"] = {
                    "outline": narrative.outline.model_dump(),
                    "characters": [
                        {"id": value["id"], "name": value["name"]}
                        for value in [*approved_characters(snapshot),
                                      *(c.model_dump() for c in narrative.supporting_characters)]
                    ],
                }
                return ProductionPlot.model_validate(result)
            except (ValueError, KeyError, TypeError) as exc:
                raise ServiceError(422, "採用済みのプロットと人物情報を確認できません。") from exc

    def _reference(self, connection, production, reference, expected_kind):
        record = required(connection, "artifact", reference["artifactId"])
        if (
            record["project_id"] != production["project_id"]
            or record["sha256"] != reference["sha256"]
            or record["kind"] != expected_kind
        ):
            raise ServiceError(422, "承認版の素材参照が作品・種類・ハッシュと一致しません。")
        return record, self.store.read(record)

    def _requirement(self, connection, production, kind, target_id, descriptor, artifact_id=None):
        identifier = uuid4().hex
        connection.execute(
            "INSERT INTO m3_requirement (id,production_id,kind,target_id,descriptor,artifact_id) "
            "VALUES (?,?,?,?,?,?)",
            (
                identifier,
                production["id"],
                kind,
                target_id,
                json.dumps(descriptor, ensure_ascii=False),
                artifact_id,
            ),
        )

    def _plan(self, connection, production, narrative):
        snapshot = self._snapshot(connection, production)
        used_characters = {cid for scene in narrative.scenes for cid in scene.plan.character_ids}
        for character in snapshot["characters"]:
            if character["id"] not in used_characters:
                continue
            image, data = self._reference(connection, production, character["image"], "m2_image")
            validate_png(data)
            alias = connection.execute(
                "SELECT * FROM artifact WHERE project_id=? AND logical_id=? AND kind='character' "
                "AND sha256=? ORDER BY version DESC LIMIT 1",
                (production["project_id"], "m3-image-" + image["id"], image["sha256"]),
            ).fetchone()
            if alias is None:
                alias = self._artifact(
                    connection, production, "m3-image-" + image["id"], "character",
                    script_character_id(character["id"]) + ".png", data,
                    provenance={"approved_artifact_id": image["id"]},
                )
            self._requirement(connection, production, "m3_image", character["id"], {}, alias["id"])
            voice, data = self._reference(connection, production, character["voice"], "m2_voice")
            validate_wav(data)
            metadata = json.loads(voice["provenance"]).get("voice", {})
            transcript = metadata.get("reference_text") or metadata.get("text")
            if not isinstance(transcript, str) or not transcript.strip():
                raise ServiceError(422, "承認済み基準音声に読み上げ本文の記録がありません。")
            self._requirement(
                connection,
                production,
                "m3_voice",
                character["id"],
                {"reference_text": transcript},
                voice["id"],
            )
        for character in narrative.supporting_characters:
            if character.id not in used_characters:
                continue
            descriptor = {
                "character_id": character.id,
                "character_result": character.model_dump(mode="json"),
            }
            self._requirement(connection, production, "m3_image", character.id, descriptor)
            self._requirement(
                connection,
                production,
                "m3_voice",
                character.id,
                {**descriptor, "reference_text": character.selfIntroduction},
            )
        for location in narrative.locations:
            self._requirement(
                connection,
                production,
                "m3_background",
                location.id,
                {"location": location.model_dump(mode="json")},
            )
        cast = {value["id"]: value["result"] for value in snapshot["characters"]}
        cast.update(
            {value.id: value.model_dump(mode="json") for value in narrative.supporting_characters}
        )
        for scene in narrative.scenes:
            for utterance in scene.utterances:
                if utterance.speaker_id is not None:
                    self._requirement(
                        connection,
                        production,
                        "m3_voice_clone",
                        utterance.id,
                        {
                            "character_id": utterance.speaker_id,
                            "character_result": cast[utterance.speaker_id],
                            "dialogue_text": utterance.spoken_text,
                            "voice_emotion": utterance.voice_emotion,
                            "delivery": utterance.delivery,
                        },
                    )
        if self._root(connection, production)["music_enabled"]:
            from .music_service import context

            self._requirement(connection, production, "m3_music_plan", "chapter-music",
                              {"context": context(self, connection, production, narrative), "planning_version": 2})

    def advance(self, production_id: str) -> None:
        with self.db.transaction() as connection:
            production = required(connection, "m3_production", production_id)
            root = self._root(connection, production)
            self._settle_control(connection, root)
            identifiers = [row[0] for row in connection.execute(
                "SELECT id FROM m3_production WHERE storyline_id=? ORDER BY chapter_number",
                (root["id"],))]
        for identifier in identifiers:
            self._advance_chapter(identifier)
        with self.db.transaction() as connection:
            self.history.finish(connection, root["project_id"])

    def _advance_chapter(self, production_id: str) -> None:
        try:
            with self.db.transaction() as connection:
                production = required(connection, "m3_production", production_id)
                selection = self.history.selection(connection, production["project_id"])
                if not self._is_selected(connection, production) or (
                    selection is not None and selection["production_frozen"]
                ):
                    return
                if not self._has_current_plan(connection, production):
                    return
                if (
                    production["error"]
                    or not production["narrative_artifact_id"]
                ):
                    return
                root = self._root(connection, production)
                if not root["sequential_publication"]:
                    self._next_chapter(connection, production)
                if self._selected_build(connection, production):
                    if root["sequential_publication"]:
                        self._next_chapter(connection, production)
                    return
                requirements = [
                    dict(row)
                    for row in connection.execute(
                        "SELECT * FROM m3_requirement WHERE production_id=? ORDER BY rowid",
                        (production_id,),
                    )
                ]
                references = {
                    row["target_id"]: row for row in requirements if row["kind"] == "m3_voice"
                }
                for requirement in requirements:
                    if requirement["artifact_id"] or requirement["job_id"]:
                        continue
                    if root["control_state"] != "running":
                        continue
                    # Reuse identical shared character/background requirements. Wait for
                    # an earlier chapter's in-flight copy instead of generating it twice.
                    if requirement["kind"] in {"m3_image", "m3_voice", "m3_background"}:
                        shared = connection.execute(
                            "SELECT r.* FROM m3_requirement r JOIN m3_production p "
                            "ON p.id=r.production_id WHERE p.storyline_id=? AND p.chapter_number<? "
                            "AND r.kind=? AND r.target_id=? AND r.descriptor=? "
                            "ORDER BY p.chapter_number LIMIT 1",
                            (root["id"], production["chapter_number"], requirement["kind"],
                             requirement["target_id"], requirement["descriptor"]),
                        ).fetchone()
                        if shared:
                            if shared["artifact_id"]:
                                connection.execute("UPDATE m3_requirement SET artifact_id=? WHERE id=?",
                                                   (shared["artifact_id"], requirement["id"]))
                                requirement["artifact_id"] = shared["artifact_id"]
                            continue
                    descriptor = json.loads(requirement["descriptor"])
                    if requirement["kind"] == "m3_voice_clone":
                        reference = references[descriptor["character_id"]]
                        if not reference["artifact_id"]:
                            continue
                        artifact = required(connection, "artifact", reference["artifact_id"])
                        descriptor["reference_voice"] = {
                            "artifact_id": artifact["id"],
                            "sha256": artifact["sha256"],
                            "text": json.loads(reference["descriptor"])["reference_text"],
                        }
                    job_id = self._enqueue(
                        connection,
                        production,
                        requirement["kind"],
                        {**descriptor, "requirement_id": requirement["id"]},
                    )
                    connection.execute(
                        "UPDATE m3_requirement SET job_id=? WHERE id=?", (job_id, requirement["id"])
                    )
                if requirements and all(item["artifact_id"] for item in requirements):
                    previous = self._previous(connection, production)
                    if previous is None or self._selected_build(connection, previous):
                        self._publish(connection, production, requirements)
                        if root["sequential_publication"]:
                            self._next_chapter(connection, production)
        except (OSError, ValueError, ServiceError, wave.Error, zlib.error) as exc:
            # Adopted media remain ready; a build retry only rechecks/assembles those files.
            with self.db.transaction() as connection:
                connection.execute(
                    "UPDATE m3_production SET error=? WHERE id=?",
                    (
                        (
                            "章の組み立てに失敗しました。素材の整合性を確認して制作を再開してください。"
                            f" ({type(exc).__name__})"
                        ),
                        production_id,
                    ),
                )
                production = required(connection, "m3_production", production_id)

    def validate_retry(self, connection, job):
        production = required(
            connection, "m3_production", json.loads(job["payload"])["production_id"]
        )
        if not self._is_selected(connection, production):
            raise ServiceError(409, "以前の制作版のジョブは再試行できません。")
        self._assert_current_plan(connection, production)
        self._assert_resumable(connection, production)
        self.history.ensure(connection, job["project_id"])
        selection = self.history.selection(connection, job["project_id"])
        if not self.history.has_operation(connection, job["project_id"], domain="m3"):
            self.history.begin(connection, job["project_id"], "本編制作を再試行", "m3-retry")
        self.history.set_selection(
            connection, job["project_id"], production["storyline_id"], selection["build_id"],
            selection["portrait_settings_id"],
        )
        connection.execute("UPDATE m3_production SET error=NULL WHERE id=?", (production["id"],))

    def recover(self) -> None:
        with self.db.transaction() as connection:
            identifiers = [
                row[0]
                for row in connection.execute(
                    "SELECT DISTINCT storyline_id FROM m3_production "
                    "WHERE error IS NULL AND narrative_artifact_id IS NOT NULL"
                )
            ]
        for identifier in identifiers:
            self.advance(identifier)

    def _publish(self, connection, production, requirements, *, presentation=None):
        snapshot = self._snapshot(connection, production)
        narrative = self._load_narrative(connection, production)
        references, content = {}, {}
        omitted_portraits = set()
        for item in requirements:
            if item["kind"] in {"m3_music_plan", "m3_music"}:
                continue
            record = required(connection, "artifact", item["artifact_id"])
            data = self.store.read(record)
            if record["project_id"] != production["project_id"]:
                raise ValueError("asset belongs to another project")
            if item["kind"] == "m3_image" and record["kind"] == "portrait_omission":
                omission = PortraitOmission.model_validate_json(data)
                source = next(value for value in narrative.supporting_characters
                              if value.id == item["target_id"])
                omission.validate_source(source.model_dump(mode="json"))
                omitted_portraits.add(item["target_id"])
                continue
            if item["kind"] == "m3_voice":
                validate_wav(data)
                continue
            expected_kind = {
                "m3_image": "character",
                "m3_background": "background",
                "m3_voice_clone": "audio",
            }[item["kind"]]
            if record["kind"] != expected_kind:
                raise ValueError("asset kind differs from requirement")
            if expected_kind == "character":
                validate_png(data)
            elif expected_kind == "background":
                validate_background(data)
            else:
                validate_wav(data)
            identifier = stable_id(expected_kind, item["target_id"])
            reference = {
                "id": identifier,
                "kind": expected_kind,
                "artifact_id": record["id"],
                "filename": identifier + (".wav" if expected_kind == "audio" else ".png"),
                "sha256": record["sha256"],
            }
            references[(item["kind"], item["target_id"])] = reference
            content[identifier] = data
        portrait_record, portrait_settings = self._portrait_settings(connection, production)
        script = narrative_to_script(
            narrative, snapshot, references, script_id="chapter-" + production["id"],
            portrait_settings=portrait_settings,
            omitted_portraits=omitted_portraits,
        )
        from .music_service import add_to_script

        script = add_to_script(self, connection, production, narrative, script, content, requirements, presentation)
        if presentation is not None:
            from packages.narrative.validation import script_character_id
            from packages.tyrano_export.presentation import apply_portrait_presentation

            baseline = {**presentation["baseline"], "layouts": {
                script_character_id(cid): layout for cid, layout in presentation["baseline"]["layouts"].items()
            }}
            adjustments = {
                script_character_id(value["character_id"]): {
                    "offset_y": value.get("offset_y", 0), "scale": value.get("scale", 1),
                } for value in presentation.get("characters", [])
            }
            script = apply_portrait_presentation(script, baseline, adjustments)
        self.coordinator._script_assets(connection, production["project_id"], script)
        public_approval = {
            "schema_version": 1,
            "approval_id": snapshot["id"],
            "world": snapshot["world"]["result"],
            "characters": [value["result"] for value in snapshot["characters"]],
            "relationships": snapshot.get("relationships", {}).get("result"),
        }
        documents = {
            "approval.json": encode_json(public_approval),
            "narrative.json": encode_json(narrative.model_dump(mode="json")),
        }
        from packages.narrative.validation import story_state_hash

        continuity = {
            "storyline_id": production["storyline_id"],
            "chapter_number": production["chapter_number"],
            "narrative_artifact_id": production["narrative_artifact_id"],
            "previous_narrative_artifact_id": production["previous_narrative_artifact_id"],
            "previous_state_hash": production["previous_state_hash"],
            "end_state_hash": story_state_hash(narrative.end_state) if narrative.end_state else None,
        }
        previous_chapter = self._previous(connection, production)
        if previous_chapter:
            previous_build = self._selected_build(connection, previous_chapter)
            if (not previous_build or self._build_lineage(connection, previous_build)["narrative_artifact_id"]
                    != production["previous_narrative_artifact_id"]):
                raise ValueError("Predecessor publication belongs to a different narrative revision")
            continuity["previous_script_artifact_id"] = previous_build["script_artifact_id"]
        if production["m4_enabled"]:
            documents["chapter-manifest.json"] = encode_json({**continuity, "assets": list(references.values())})
        documents.update(
            {
                f"sources/{scene.id}.txt": scene.raw_text.encode("utf-8")
                for scene in narrative.scenes
            }
        )
        bundle = compile_bundle(script, content, documents=documents)
        previous = self._selected_build(connection, production)
        if previous:
            previous_export = required(connection, "artifact", previous["export_artifact_id"])
            if previous_export["sha256"] == hashlib.sha256(bundle).hexdigest():
                return
        script_record = self._artifact(
            connection,
            production,
            script.id,
            "script",
            "script.json",
            encode_json(script.model_dump(mode="json")),
        )
        export = self._artifact(
            connection,
            production,
            "export-" + production["id"],
            "tyrano_export",
            "tyrano-source.zip",
            bundle,
        )
        validation = {
            "schema_version": 1,
            "raw_text_mapping": True,
            "references": True,
            "media": True,
            "content_review": narrative.workflow_policy != "script_continuation_v1",
            "required_assets": len(references),
            **continuity,
        }
        if narrative.workflow_policy == "script_continuation_v1":
            validation["content_review_status"] = "not_evaluated"
        if portrait_record:
            validation["portrait_settings_artifact_id"] = portrait_record["id"]
        identifier = uuid4().hex
        revision = connection.execute(
            "SELECT COALESCE(MAX(revision),0)+1 FROM chapter_build WHERE production_id=?",
            (production["id"],),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO chapter_build (id,production_id,project_id,chapter_number,script_artifact_id,"
            "export_artifact_id,manifest,validation,status,created_at,revision) "
            "VALUES (?,?,?,?,?,?,?,?, 'published',?,?)",
            (
                identifier,
                production["id"],
                production["project_id"],
                production["chapter_number"],
                script_record["id"],
                export["id"],
                json.dumps(script.model_dump(mode="json")["assets"], ensure_ascii=False),
                json.dumps(validation),
                self.clock(),
                revision,
            ),
        )
        if production["chapter_number"] == 1:
            self.history.set_selection(
                connection, production["project_id"], production["id"], identifier,
                portrait_record["id"] if portrait_record else None,
            )

    def complete(self, job_id: str, worker_id: str, lease_id: str, data: bytes) -> dict:
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(
                connection, job_id, worker_id, lease_id, completed_ok=True
            )
            if job["kind"] not in M3_KINDS:
                raise ServiceError(422, "M3の生成ジョブを指定してください。")
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            payload = json.loads(job["payload"])
        try:
            envelope, files = validate_bundle(data, job["kind"])
            from .music_service import validate_plan, validate_result

            music_plan = validate_plan(envelope["result"], payload) if job["kind"] == "m3_music_plan" else None
            music = validate_result(envelope["result"], payload, files) if job["kind"] == "m3_music" else None
            omission = None
            if job["kind"] == "m3_image" and envelope["result"]:
                omission = PortraitOmission.model_validate(envelope["result"]["portrait"])
                omission.validate_source(payload["character_result"])
                if omission.character_id != payload["character_id"]:
                    raise ValueError("Portrait omission belongs to another character")
            narrative = (
                validate_narrative(envelope["result"], payload["approval_snapshot"],
                                   payload.get("previous_narrative"),
                                   require_state=payload.get("m4", False),
                                   expected_previous_artifact_id=payload.get("previous_narrative_artifact_id"))
                if job["kind"] == "m3_narrative"
                else None
            )
            if narrative:
                script_policy = payload.get("workflow_policy") == "script_continuation_v1"
                if script_policy:
                    if (narrative.workflow_policy != "script_continuation_v1"
                            or narrative.workflow_version != payload.get("story_workflow_version")
                            or envelope["provenance"].get("generator_protocol") != payload.get("generator_protocol")):
                        raise ValueError("Script result differs from the pinned workflow or lacks its checkpoint.")
                    _validate_script_checkpoint(payload, narrative, envelope["provenance"])
                elif narrative.workflow_policy == "script_continuation_v1":
                    raise ValueError("An existing production cannot silently change its narrative workflow.")
                if narrative.chapter_number != payload.get("chapter_number", 1):
                    raise ValueError("Chapter differs from claimed input")
                if payload.get("m4"):
                    if narrative.storyline_id != payload["storyline_id"]:
                        raise ValueError("Storyline differs from claimed input")
                    if narrative.previous_narrative_artifact_id != payload.get("previous_narrative_artifact_id"):
                        raise ValueError("Predecessor differs from claimed input")
                    if (payload.get("previous_state_hash") is not None
                            and narrative.previous_state_hash != payload["previous_state_hash"]):
                        raise ValueError("Predecessor state differs from claimed input")
            voice = envelope["provenance"].get("voice", {})
            if job["kind"] == "m3_voice":
                if voice.get("reference_text") != payload["reference_text"]:
                    raise ValueError("reference speech changed")
            elif job["kind"] == "m3_voice_clone":
                reference = payload["reference_voice"]
                if (
                    voice.get("spoken_text") != payload["dialogue_text"]
                    or voice.get("text")
                    != EMOTION_TAGS[payload["voice_emotion"]] + payload["dialogue_text"]
                    or voice.get("voice_emotion") != payload["voice_emotion"]
                    or any(
                        voice.get(key) != reference[source]
                        for key, source in (
                            ("reference_artifact_id", "artifact_id"),
                            ("reference_sha256", "sha256"),
                            ("reference_text", "text"),
                        )
                    )
                ):
                    raise ValueError("speech or immutable voice reference changed")
        except (
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            zipfile.BadZipFile,
            EOFError,
            wave.Error,
            zlib.error,
            RuntimeError,
            NotImplementedError,
        ) as exc:
            raise ServiceError(
                422, "台本・章の参照元・画像・音声の生成契約を満たしていません。"
            ) from exc
        stored = self.store.put(data)
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(
                connection, job_id, worker_id, lease_id, completed_ok=True
            )
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            production = required(connection, "m3_production", payload["production_id"])
            if production["project_id"] != job["project_id"]:
                raise ServiceError(409, "生成結果が制作対象と一致しません。")
            selection = self.history.selection(connection, job["project_id"])
            if not self._is_selected(connection, production) or (
                selection is not None and selection["production_frozen"]
            ):
                raise ServiceError(409, "以前の制作版の生成結果は採用できません。")
            self._assert_current_plan(connection, production)
            artifact = self.coordinator._register_artifact(
                connection,
                job["project_id"],
                "m3-job-" + job_id,
                "m3_bundle",
                "m3-result.zip",
                "application/zip",
                stored,
                source_job_id=job_id,
                source_attempt_id=attempt["id"],
            )
            if narrative:
                if production["narrative_artifact_id"]:
                    raise ServiceError(409, "本文は既に採用されています。")
                previous = self._previous(connection, production)
                if previous and previous["narrative_artifact_id"] != production["previous_narrative_artifact_id"]:
                    raise ServiceError(409, "前章の採用本文が変更されたため、この結果は採用できません。")
                record = self._artifact(
                    connection,
                    production,
                    "narrative-" + production["id"],
                    "m3_narrative",
                    "narrative.json",
                    encode_json(narrative.model_dump(mode="json")),
                    provenance=envelope["provenance"],
                    job=job,
                    attempt=attempt,
                )
                for scene in narrative.scenes:
                    self._artifact(
                        connection,
                        production,
                        "source-" + production["id"] + "-" + scene.id,
                        "scene_text",
                        scene.id + ".txt",
                        scene.raw_text.encode("utf-8"),
                        provenance=envelope["provenance"],
                        job=job,
                        attempt=attempt,
                    )
                self._plan(connection, production, narrative)
                connection.execute(
                    "UPDATE m3_production SET narrative_artifact_id=?,previous_state_hash=? WHERE id=?",
                    (record["id"], narrative.previous_state_hash, production["id"]),
                )
            else:
                requirement = required(connection, "m3_requirement", payload["requirement_id"])
                if (
                    requirement["job_id"] != job_id
                    or requirement["production_id"] != production["id"]
                ):
                    raise ServiceError(409, "素材の生成対象が一致しません。")
                kind = {
                    "m3_image": "character",
                    "m3_background": "background",
                    "m3_voice": "audio",
                    "m3_voice_clone": "audio",
                    "m3_music_plan": "music_plan",
                    "m3_music": "music",
                }[job["kind"]]
                filename = "voice.wav" if kind == "audio" else "image.png"
                if music_plan is not None:
                    filename = "music-plan.json"
                    files[filename] = encode_json(music_plan.model_dump(mode="json"))
                    for scene_prompt in music_plan.scenes:
                        if scene_prompt.action != "play":
                            continue
                        self._requirement(connection, production, "m3_music", scene_prompt.scene_id,
                            {"scene_id": scene_prompt.scene_id, "prompt": scene_prompt.prompt,
                             "context": payload["context"], "duration_seconds": 120,
                             "backend": "stable_audio3", "model": "medium"})
                if music is not None:
                    from .music_service import store_candidate

                    record, _candidate = store_candidate(self, connection, production, music.model_dump(mode="json"),
                        files, job=job, attempt=attempt, provenance=envelope["provenance"])
                if omission is not None:
                    kind, filename = "portrait_omission", "portrait-omission.json"
                    files[filename] = encode_json(omission.model_dump())
                record = record if music is not None else self._artifact(
                    connection,
                    production,
                    "media-" + requirement["id"],
                    kind,
                    filename,
                    files[filename],
                    provenance=envelope["provenance"],
                    job=job,
                    attempt=attempt,
                )
                connection.execute(
                    "UPDATE m3_requirement SET artifact_id=? WHERE id=?",
                    (record["id"], requirement["id"]),
                )
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
            result = {
                "job": public(required(connection, "job", job_id)),
                "artifact": public(artifact),
            }
        self.advance(production["id"])
        return result
