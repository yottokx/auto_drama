"""Editable common plans and immutable approvals before chapter production."""
from __future__ import annotations

import copy
import hashlib
import json
import zipfile
import zlib
from uuid import uuid4

from packages.contracts.planning import plan_snapshot, planning_protocol, validate_plan_content

from .m2_service import M2Service
from .project_history import HistoryService
from .service import Coordinator, ServiceError, encode_json, public, required


def plan_digest(content):
    from .m3_service import _checkpoint_digest
    return _checkpoint_digest(content)


ASSET_JOBS = {"image": "m3_image", "voice": "m3_voice"}
# Only these settings shape the asset; a change elsewhere in the plan keeps it.
ASSET_FIELDS = {"image": ("id", "name", "age", "gender", "appearance", "height_cm", "body_type"),
                "voice": ("id", "name", "age", "gender", "voice", "selfIntroduction")}


def asset_source(character: dict, kind: str) -> str:
    source = {key: character.get(key) for key in ASSET_FIELDS[kind]}
    return hashlib.sha256(json.dumps(source, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


class PlanningService:
    def __init__(self, coordinator: Coordinator):
        self.coordinator = coordinator
        self.db, self.store, self.clock = coordinator.db, coordinator.store, coordinator.clock
        self.history = HistoryService(coordinator)

    @staticmethod
    def selected(connection, project_id):
        selection = HistoryService.selection(connection, project_id)
        identifier = selection.get("planning_id") if selection else None
        if not identifier:
            return None
        row = required(connection, "planning_draft", identifier)
        if row["project_id"] != project_id:
            raise ServiceError(422, "構成の選択版が作品と一致しません。")
        return row

    def _snapshot(self, connection, row):
        approval = required(connection, "m2_approval", row["main_approval_id"])
        artifact = required(connection, "artifact", approval["artifact_id"])
        if approval["project_id"] != row["project_id"] or artifact["kind"] != "m2_approval":
            raise ServiceError(422, "構成の元となる承認版が一致しません。")
        return json.loads(self.store.read(artifact))

    def _valid_content(self, content, snapshot):
        try:
            return validate_plan_content(content, snapshot).model_dump(mode="json")
        except (ValueError, KeyError, TypeError) as exc:
            raise ServiceError(422, "人物・章数・プロットの参照を確認してください。") from exc

    def _current(self, connection, project_id):
        row = self.selected(connection, project_id)
        state = M2Service._load(connection, project_id)
        approval = state["draft"].get("approval")
        if (row is None or not state["draft"]["approved"] or not approval
                or row["main_approval_id"] != approval["id"]):
            raise ServiceError(409, "先に世界観とメインキャラを承認してください。")
        return row, state

    def _view(self, connection, project_id):
        row = self.selected(connection, project_id)
        selected = self.history.selection(connection, project_id)
        production = required(connection, "m3_production", selected["production_id"]) if (
            selected and selected["production_id"]) else None
        result = {"project_id": project_id, "planning": None,
                  "legacy_production": bool(production and not production["plan_approval_id"])}
        if row is None:
            return result
        jobs = [public(dict(job)) for job in connection.execute(
            "SELECT j.* FROM job j JOIN planning_job p ON p.job_id=j.id "
            "WHERE p.planning_id=? ORDER BY j.created_at,j.id", (row["id"],))]
        active = next((job for job in jobs if job["id"] == row["active_job_id"]), None)
        status = "failed" if active and active["status"] == "failed" else row["status"]
        result["planning"] = {"id": row["id"], "revision": row["revision"], "status": status,
            "content": json.loads(row["content"]) if row["content"] else None,
            "approval_id": row["approved_plan_id"], "active_job_id": row["active_job_id"],
            "jobs": jobs, "error": active["error"] if active else row["error"],
            "assets": self._asset_view(connection, row),
            # How many chapters the current settings allow a long-form plan.
            "chapter_limit": self._chapter_limit(connection)}
        return result

    def _chapter_limit(self, connection):
        from services.worker.generation.script_cast import work_limits

        from .llm_settings import script_profile

        return work_limits(script_profile(self.coordinator, connection)).chapters

    def _assets(self, connection, row, content=None):
        """The newest asset of each planned supporting character that matches its settings."""
        content = content if content is not None else json.loads(row["content"]) if row["content"] else None
        result = {}
        for character in (content or {}).get("cast_plan", {}).get("supporting_characters", []):
            for kind in ASSET_JOBS:
                asset = connection.execute(
                    "SELECT * FROM planning_asset WHERE planning_id=? AND character_id=? AND kind=? "
                    "AND source_sha256=? ORDER BY created_at DESC,rowid DESC LIMIT 1",
                    (row["id"], character["id"], kind, asset_source(character, kind))).fetchone()
                result[(character["id"], kind)] = (character, dict(asset) if asset else None)
        return result

    def _asset_view(self, connection, row):
        view = {}
        for (cid, kind), (_character, asset) in self._assets(connection, row).items():
            job = required(connection, "job", asset["job_id"]) if asset and asset["job_id"] else None
            artifact = required(connection, "artifact", asset["artifact_id"]) if asset and asset["artifact_id"] else None
            omitted = bool(artifact and artifact["kind"] == "portrait_omission")
            previous = None if artifact else connection.execute(
                "SELECT a.* FROM planning_asset p JOIN artifact a ON a.id=p.artifact_id WHERE p.planning_id=? "
                "AND p.character_id=? AND p.kind=? AND p.source_sha256=? AND a.kind!='portrait_omission' "
                "ORDER BY p.created_at DESC,p.rowid DESC LIMIT 1",
                (row["id"], cid, kind, asset["source_sha256"] if asset else "")).fetchone()
            shown = artifact if artifact and not omitted else previous
            view.setdefault(cid, {})[kind] = {
                "status": ("omitted" if omitted else "ready" if artifact else job["status"] if job else "missing"),
                "url": f"/api/artifacts/{shown['id']}/content" if shown else None,
                "job_id": job["id"] if job else None, "error": job["error"] if job else None,
                # The portrait itself, by which its expression pictures are made and found.
                **({"artifact_id": artifact["id"]} if kind == "image" and artifact and not omitted else {})}
        return view

    def _sync_assets(self, connection, row, content):
        """Supporting characters are drawn and voiced as soon as the plan names them."""
        for (_cid, kind), (character, asset) in self._assets(connection, row, content).items():
            if asset is None:
                self._enqueue_asset(connection, row, character, kind)

    def _enqueue_asset(self, connection, row, character, kind, *, instruction=""):
        from .llm_settings import execution_settings
        from .tts_settings import TTSService

        first = connection.execute(
            "SELECT j.* FROM job j JOIN planning_job p ON p.job_id=j.id "
            "WHERE p.planning_id=? ORDER BY j.created_at,j.id LIMIT 1", (row["id"],)).fetchone()
        frozen = json.loads(first["payload"]) if first else {}
        asset_id, identifier, run_id, now = uuid4().hex, uuid4().hex, uuid4().hex, self.clock()
        profile = M2Service(self.coordinator)._profile(connection)
        # Voice models stay the ones the plan was made with, as they do for the chapters.
        tts_profile = frozen.get("tts_profile") or TTSService(self.coordinator).profile(connection)
        payload = {"schema_version": 1, "approval_snapshot": self._snapshot(connection, row),
                   "character_id": character["id"], "character_result": character, "profile": profile,
                   **({"tts_profile": tts_profile} if tts_profile is not None else {}),
                   "seed": int(identifier[:8], 16) & 0x7FFFFFFF, "instruction": instruction,
                   "planning_asset": {"planning_id": row["id"], "asset_id": asset_id},
                   **({"reference_text": character["selfIntroduction"]} if kind == "voice" else {})}
        settings = {"schema_version": 1, "seed": payload["seed"],
                    **({"tts_profile": tts_profile} if tts_profile is not None else {}),
                    **execution_settings(self.coordinator, connection, payload, kind=ASSET_JOBS[kind], profile=profile)}
        connection.execute(
            "INSERT INTO generation_run(id,project_id,settings_version,story_revision_id,policy,status,created_at) "
            "VALUES (?,?,?,?,?,'pending',?)", (run_id, row["project_id"], row["revision"], row["id"],
                json.dumps({"mode": "planning-asset"}), now))
        connection.execute(
            "INSERT INTO job(id,project_id,run_id,kind,payload,settings_snapshot,priority,status,max_attempts,created_at,updated_at) "
            "VALUES (?,?,?,?,?,?,?,'pending',3,?,?)",
            # Every portrait before the first voice, so each model is loaded once.
            (identifier, row["project_id"], run_id, ASSET_JOBS[kind], json.dumps(payload, ensure_ascii=False),
             json.dumps(settings, ensure_ascii=False), 100 if kind == "image" else 99, now, now))
        connection.execute("INSERT INTO planning_asset VALUES (?,?,?,?,?,?,NULL,?)", (
            asset_id, row["id"], character["id"], kind, asset_source(character, kind), identifier, now))

    def approved_assets(self, connection, row, content):
        """References frozen into an approval; a character whose asset is absent is made in production."""
        result = {}
        for (cid, kind), (character, asset) in self._assets(connection, row, content).items():
            if not asset or not asset["artifact_id"]:
                job = required(connection, "job", asset["job_id"]) if asset and asset["job_id"] else None
                if job and job["status"] in {"pending", "running"}:
                    raise ServiceError(409, "サブキャラの立ち絵と声を生成中です。完了してから承認してください。")
                continue
            artifact = required(connection, "artifact", asset["artifact_id"])
            result.setdefault(cid, {})[kind] = {
                "artifact_id": artifact["id"], "sha256": artifact["sha256"],
                **({"reference_text": character["selfIntroduction"]} if kind == "voice" else {})}
        return result

    def retake_asset(self, project_id, body):
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            row, _state = self._current(connection, project_id)
            if row["revision"] != body.expected_revision:
                raise ServiceError(409, "構成が更新されました。最新の版を確認してください。")
            if row["approved_plan_id"]:
                raise ServiceError(409, "承認済みの構成の素材は、完成後の調整で変更できます。")
            found = self._assets(connection, row).get((body.character_id, body.kind))
            if found is None:
                raise ServiceError(422, "登場予定のサブキャラを指定してください。")
            character, asset = found
            job = required(connection, "job", asset["job_id"]) if asset and asset["job_id"] else None
            if job and job["status"] in {"pending", "running"}:
                raise ServiceError(409, "この素材は生成中です。")
            self._enqueue_asset(connection, row, character, body.kind, instruction=body.instruction or "")
            return self._view(connection, project_id)

    def dress_asset(self, project_id, body):
        """Redraw a planned supporting character's portrait wearing a shared design."""
        from .event_cg_settings import EventCgSettings
        from .llm_settings import execution_settings
        from .material_service import MaterialService

        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            row, _state = self._current(connection, project_id)
            if row["revision"] != body.expected_revision:
                raise ServiceError(409, "構成が更新されました。最新の版を確認してください。")
            if row["approved_plan_id"]:
                raise ServiceError(409, "承認済みの構成の素材は、完成後の調整で変更できます。")
            found = self._assets(connection, row).get((body.character_id, "image"))
            if found is None:
                raise ServiceError(422, "登場予定のサブキャラを指定してください。")
            character, asset = found
            current = required(connection, "artifact", asset["artifact_id"]) if asset and asset["artifact_id"] else None
            if current is None or current["kind"] != "character":
                raise ServiceError(409, "この人物の立ち絵ができてから着せてください。")
            material = MaterialService(self.coordinator).reference(connection, project_id, body.material_image_id)
            if material is None:
                raise ServiceError(409, "着せる共通の素材の絵がまだありません。")
            if not M2Service(self.coordinator)._qwen_ready(connection, _state["draft"]):
                raise ServiceError(409, "着せ替えには、画像編集の準備ができたWorkerが必要です。")
            asset_id, identifier, run_id, now = uuid4().hex, uuid4().hex, uuid4().hex, self.clock()
            profile = M2Service(self.coordinator)._profile(connection)
            snapshot = self._snapshot(connection, row)
            payload = {"schema_version": 1, "seed": int(identifier[:8], 16) & 0x7FFFFFFF,
                       "character_id": character["id"], "character_result": character,
                       "world_result": snapshot["world"]["result"], "profile": profile,
                       "cg_profile": EventCgSettings(self.coordinator).settings(connection)["profile"],
                       "outfit": {"id": "default", "name": "通常の姿", "description": "", "occasion": ""},
                       "reference_image": {"artifact_id": current["id"], "sha256": current["sha256"]},
                       "material_image": material,
                       "planning_asset": {"planning_id": row["id"], "asset_id": asset_id}}
            settings = {"schema_version": 1, "seed": payload["seed"], **execution_settings(
                self.coordinator, connection, payload, kind="m2_outfit_image", profile=profile)}
            connection.execute(
                "INSERT INTO generation_run(id,project_id,settings_version,story_revision_id,policy,status,created_at) "
                "VALUES (?,?,?,?,?,'pending',?)", (run_id, row["project_id"], row["revision"], row["id"],
                    json.dumps({"mode": "planning-asset"}), now))
            connection.execute(
                "INSERT INTO job(id,project_id,run_id,kind,payload,settings_snapshot,priority,status,max_attempts,created_at,updated_at) "
                "VALUES (?,?,?,'m2_outfit_image',?,?,100,'pending',3,?,?)",
                (identifier, row["project_id"], run_id, json.dumps(payload, ensure_ascii=False),
                 json.dumps(settings, ensure_ascii=False), now, now))
            connection.execute("INSERT INTO planning_asset VALUES (?,?,?,?,?,?,NULL,?)", (
                asset_id, row["id"], character["id"], "image", asset_source(character, "image"), identifier, now))
            return self._view(connection, project_id)

    def validate_asset_retry(self, connection, job):
        link = json.loads(job["payload"])["planning_asset"]
        row = self.selected(connection, job["project_id"])
        asset = required(connection, "planning_asset", link["asset_id"])
        current = self._assets(connection, row).get((asset["character_id"], asset["kind"])) if row else None
        if (not row or row["id"] != link["planning_id"] or row["approved_plan_id"]
                or not current or not current[1] or current[1]["id"] != asset["id"]):
            raise ServiceError(409, "以前の構成版の素材は再試行できません。")

    def complete_asset(self, job_id, worker_id, lease_id, data):
        import wave

        from packages.contracts.portrait_recovery import PortraitOmission

        from .m3_bundle import validate_bundle
        with self.db.transaction() as connection:
            job, _ = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            payload = json.loads(job["payload"])
        try:
            if job["kind"] == "m2_outfit_image":
                from .m2_bundle import validate_bundle as validate_edit

                envelope, files = validate_edit(data, job["kind"], payload["character_id"])
                if envelope["provenance"].get("outfit", {}).get("reference_sha256") != payload["reference_image"]["sha256"]:
                    raise ValueError("the portrait was edited from another picture")
            else:
                envelope, files = validate_bundle(data, job["kind"])
            omission = None
            if job["kind"] == "m3_image" and envelope["result"]:
                omission = PortraitOmission.model_validate(envelope["result"]["portrait"])
                omission.validate_source(payload["character_result"])
                if omission.character_id != payload["character_id"]:
                    raise ValueError("Portrait omission belongs to another character")
            if (job["kind"] == "m3_voice"
                    and envelope["provenance"].get("voice", {}).get("reference_text") != payload["reference_text"]):
                raise ValueError("reference speech changed")
            filename = "voice.wav" if job["kind"] == "m3_voice" else "image.png"
            if omission is None and filename not in files:
                raise ValueError("the asset itself is missing")
        except (ValueError, KeyError, TypeError, AttributeError, zipfile.BadZipFile, EOFError, zlib.error,
                wave.Error, RuntimeError) as exc:
            raise ServiceError(422, "生成結果の素材・音声情報が入力と一致しません。") from exc
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            asset = required(connection, "planning_asset", payload["planning_asset"]["asset_id"])
            store = M2Service(self.coordinator)._artifact
            if omission is not None:
                record = store(connection, job["project_id"], "planning-asset-" + asset["id"], "portrait_omission",
                               "portrait-omission.json", "application/json", encode_json(omission.model_dump()),
                               provenance=envelope["provenance"], job=job, attempt=attempt)
            else:
                record = store(connection, job["project_id"], "planning-asset-" + asset["id"],
                               "audio" if filename == "voice.wav" else "character", filename,
                               "audio/wav" if filename == "voice.wav" else "image/png", files[filename],
                               provenance=envelope["provenance"], job=job, attempt=attempt)
            bundle = self.coordinator._register_artifact(connection, job["project_id"],
                "planning-asset-job-" + job_id, "m3_bundle", "result.zip", "application/zip", self.store.put(data),
                source_job_id=job_id, source_attempt_id=attempt["id"])
            now = self.clock()
            connection.execute("UPDATE planning_asset SET artifact_id=? WHERE id=?", (record["id"], asset["id"]))
            connection.execute("UPDATE job_attempt SET status='completed',ended_at=? WHERE id=?", (now, attempt["id"]))
            connection.execute("UPDATE job SET status='completed',result_artifact_id=?,updated_at=?,error=NULL WHERE id=?",
                               (bundle["id"], now, job_id))
            connection.execute("UPDATE generation_run SET status='completed' WHERE id=?", (job["run_id"],))
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            return {"job": public(required(connection, "job", job_id)), "artifact": public(bundle)}

    def project(self, project_id):
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            self.coordinator._recover(connection)
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)

    def start_approved(self, connection, project_id, approval_id, *, cast_limit=None):
        self.history.ensure(connection, project_id)
        existing = connection.execute(
            "SELECT * FROM planning_draft WHERE main_approval_id=?", (approval_id,)).fetchone()
        if existing:
            return dict(existing)
        identifier, now = uuid4().hex, self.clock()
        connection.execute(
            "INSERT INTO planning_draft(id,project_id,main_approval_id,created_at,updated_at) "
            "VALUES (?,?,?,?,?)", (identifier, project_id, approval_id, now, now))
        self.history.set_selection(connection, project_id, None, None, None)
        connection.execute("UPDATE project_history_state SET planning_id=?,version=version+1 WHERE project_id=?",
                           (identifier, project_id))
        row = required(connection, "planning_draft", identifier)
        self._enqueue(connection, row, cast_limit=cast_limit)
        return required(connection, "planning_draft", identifier)

    def _enqueue(self, connection, row, *, body=None, cast_limit=None, judge=False):
        from services.worker.generation.workflow_version import generator_protocol

        from .llm_settings import cast_limit_choice, execution_settings
        from .tts_settings import TTSService

        first = connection.execute(
            "SELECT j.* FROM job j JOIN planning_job p ON p.job_id=j.id "
            "WHERE p.planning_id=? ORDER BY j.created_at,j.id LIMIT 1", (row["id"],)).fetchone()
        if first:
            frozen = json.loads(first["payload"])
            generation = {key: frozen[key] for key in (
                "profile", "tts_profile", "seed", "script_options", "profiles", "workflow_limits",
                "story_workflow_version", "workflow_policy", "generator_protocol", "planning_protocol",
            ) if key in frozen}
        else:
            generation = {"tts_profile": TTSService(self.coordinator).profile(connection),
                "seed": int(uuid4().hex[:8], 16) & 0x7FFFFFFF, "story_workflow_version": 2,
                "workflow_policy": "script_continuation_v1",
                "generator_protocol": generator_protocol("causal", "script_continuation_v1"),
                "planning_protocol": planning_protocol(),
                # A planned supporting character is drawn and voiced in this step, so the plot uses each
                # one. The user chose how many characters the work may have (the guideline if not), and a
                # scene may hold more than three. The plan first holds its cast and how many chapters it
                # needs; the plot follows at the chapter count the user chose (long_form). A chapter plan sees
                # later chapters in brief and is told what the latest chapter already showed, so it neither
                # runs ahead nor plays a step twice (chapter_scope_guard). Plans started earlier carry none
                # of these options and keep the wording and limits they were made with.
                "script_options": {"planned_cast_appears": True, "open_scene_cast": True,
                                   "cast_limit": cast_limit or cast_limit_choice(self.coordinator, connection)["guide"],
                                   "long_form": True, "chapter_scope_guard": True}}
        content = json.loads(row["content"]) if row["content"] else None
        chosen = {"chapter_count": body.chapter_count} if body and body.action == "plot" else content
        payload = {"schema_version": 1, "phase": "planning", "planning_id": row["id"],
            "planning_revision": row["revision"],
            "approval_snapshot": plan_snapshot(self._snapshot(connection, row), chosen), **generation}
        generation.update(execution_settings(self.coordinator, connection, payload,
                          kind="m3_plan", profile=M2Service(self.coordinator)._profile(connection)))
        if body and body.action == "revise":
            payload.update(plan_content=content, target=body.target,
                           character_id=body.character_id, chapter_number=body.chapter_number,
                           instruction=body.instruction)
        elif body and body.action == "plot":
            payload.update(plan_content=content, chapter_count=body.chapter_count)
        elif judge:
            payload.update(plan_content=content, judge_only=True)
        identifier, run_id, now = uuid4().hex, uuid4().hex, self.clock()
        connection.execute(
            "INSERT INTO generation_run(id,project_id,settings_version,story_revision_id,policy,status,created_at) "
            "VALUES (?,?,?,?,?,'pending',?)", (run_id, row["project_id"], row["revision"], row["id"],
                json.dumps({"mode": "common-planning"}), now))
        connection.execute(
            "INSERT INTO job(id,project_id,run_id,kind,payload,settings_snapshot,priority,status,max_attempts,created_at,updated_at) "
            "VALUES (?,?,?,'m3_plan',?,?,100,'pending',3,?,?)",
            (identifier, row["project_id"], run_id, json.dumps(payload, ensure_ascii=False),
             json.dumps(generation, ensure_ascii=False), now, now))
        connection.execute("INSERT INTO planning_job VALUES (?,?,?)", (row["id"], identifier, row["revision"]))
        connection.execute("UPDATE planning_draft SET status='generating',active_job_id=?,error=NULL,updated_at=? WHERE id=?",
                           (identifier, now, row["id"]))

    @staticmethod
    def assert_selected_job(connection, job):
        payload = json.loads(job["payload"])
        row = PlanningService.selected(connection, job["project_id"])
        state = M2Service._load(connection, job["project_id"])
        approval = state["draft"].get("approval")
        if (not row or row["id"] != payload["planning_id"]
                or row["revision"] != payload["planning_revision"]
                or row["active_job_id"] != job["id"] or not state["draft"]["approved"]
                or not approval or approval["id"] != row["main_approval_id"]):
            raise ServiceError(409, "以前の構成版の生成結果は採用・再試行できません。")
        return row

    def validate_retry(self, connection, job):
        self.assert_selected_job(connection, job)
        if not self.history.has_operation(connection, job["project_id"], "m3"):
            self.history.begin(connection, job["project_id"], "構成生成を再試行", "m3-plan-retry")

    def action(self, project_id, body):
        production_id = None
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            if (body.action == "generate" and body.expected_revision == 0
                    and self.selected(connection, project_id) is None):
                state = M2Service._load(connection, project_id)
                approval = state["draft"].get("approval")
                if not state["draft"]["approved"] or not approval:
                    raise ServiceError(409, "先に世界観とメインキャラを承認してください。")
                if connection.execute("SELECT 1 FROM m3_production WHERE project_id=? LIMIT 1",
                                      (project_id,)).fetchone():
                    raise ServiceError(409, "既存作品は保存した制作方式で再開できます。新しい構成にはメイン設定を再承認してください。")
                self.history.begin(connection, project_id, "全体構成を生成", "m3-plan-generate")
                existing = connection.execute("SELECT id FROM planning_draft WHERE main_approval_id=?",
                                              (approval["id"],)).fetchone()
                row = self.start_approved(connection, project_id, approval["id"])
                if existing:
                    connection.execute("UPDATE project_history_state SET planning_id=?,version=version+1 WHERE project_id=?",
                                       (row["id"], project_id))
                    connection.execute("UPDATE planning_draft SET revision=revision+1,approved_plan_id=NULL,active_job_id=NULL WHERE id=?",
                                       (row["id"],))
                    self._enqueue(connection, required(connection, "planning_draft", row["id"]))
                state["draft"].update(planningRequired=True, planApproved=False, step="planning-review")
                state["draft"]["revision"] += 1
                M2Service(self.coordinator)._save(connection, project_id, state)
                return self._view(connection, project_id)
            row, state = self._current(connection, project_id)
            if row["revision"] != body.expected_revision:
                raise ServiceError(409, "構成が更新されました。最新の版を確認してください。")
            reconfirm = body.action == "approve" and body.reconfirm
            if reconfirm:
                self.history.assert_reconfirmation_allowed(connection, project_id)
            if body.action == "approve" and (not row["approved_plan_id"] or reconfirm):
                from .event_cg_settings import EventCgSettings

                EventCgSettings(self.coordinator).freeze(
                    connection, project_id,
                    expected_revision=body.expected_event_cg_policy_revision)
            active = required(connection, "job", row["active_job_id"]) if row["active_job_id"] else None
            if active and active["status"] in {"pending", "running"}:
                raise ServiceError(409, "構成を生成中です。完了または失敗を確認してください。")
            if body.action == "approve" and row["approved_plan_id"] and not reconfirm:
                from .m3_service import M3Service
                production = M3Service(self.coordinator).start_approved(
                    connection, project_id, row["main_approval_id"], row["approved_plan_id"])
                production_id = production["id"]
            else:
                self.history.begin(connection, project_id, {
                    "generate": "全体構成を生成", "revise": "全体構成を修正",
                    "save": "全体構成を編集", "approve": "構成を承認して本編制作",
                    "plot": "章数を決めて全体構成を作成",
                }[body.action], "m3-plan-" + body.action)
                if reconfirm:
                    self.history.clear_downstream(connection, project_id)
                    connection.execute(
                        "UPDATE planning_draft SET revision=revision+1,approved_plan_id=NULL,updated_at=? WHERE id=?",
                        (self.clock(), row["id"]),
                    )
                    row = required(connection, "planning_draft", row["id"])
                previous = json.loads(row["content"]) if row["content"] else None
                if body.action == "save":
                    content = self._valid_content(body.content, self._snapshot(connection, row))
                else:
                    content = previous
                if body.action in {"revise", "approve", "plot"} and content is None:
                    raise ServiceError(409, "全体構成の生成完了後に操作してください。")
                if body.action == "plot":
                    if "chapter_judgment" not in content:
                        raise ServiceError(409, "この構成は章数を選んで作り直せません。")
                    limit = self._chapter_limit(connection)
                    if body.chapter_count is None or body.chapter_count > limit:
                        raise ServiceError(400, f"章数は1〜{limit}話で選んでください。")
                if body.action == "approve" and "plot" not in content:
                    raise ServiceError(409, "章数を選んで全体構成を作ってから承認してください。")
                if body.action == "revise" and "plot" not in content and body.target not in {
                        "character", "relationships"}:
                    raise ServiceError(422, "章数を選ぶ前は、サブキャラと関係だけを修正できます。")
                if body.action == "revise":
                    if not body.instruction or not body.instruction.strip():
                        raise ServiceError(422, "修正指示を入力してください。")
                    if body.target == "character" and body.character_id not in {
                        value["id"] for value in content["cast_plan"]["supporting_characters"]}:
                        raise ServiceError(422, "修正対象のサブキャラを選択してください。")
                    if body.chapter_number is not None and body.chapter_number not in {
                        value["number"] for value in content["plot"]["chapters"]}:
                        raise ServiceError(422, "修正対象の章がありません。")
                if body.action == "approve":
                    content = self._valid_content(content, self._snapshot(connection, row))
                    identifier, now = uuid4().hex, self.clock()
                    first = connection.execute(
                        "SELECT j.payload FROM job j JOIN planning_job p ON p.job_id=j.id "
                        "WHERE p.planning_id=? ORDER BY j.created_at,j.id LIMIT 1", (row["id"],)).fetchone()
                    payload = json.loads(first["payload"]) if first else {}
                    generation = {key: payload[key] for key in (
                        "profile", "tts_profile", "seed", "profiles", "script_options", "workflow_limits",
                        "story_workflow_version", "workflow_policy", "generator_protocol", "planning_protocol",
                    ) if key in payload}
                    options = generation.get("script_options", {})
                    if options.get("long_form") and not options.get("chapter_scope_guard"):
                        # A long-form plan begun before chapter_scope_guard gets it here: the plan never
                        # used it, and the production this approval starts writes every chapter with it.
                        generation["script_options"] = {**options, "chapter_scope_guard": True}
                    assets = self.approved_assets(connection, row, content)
                    approved = {"schema_version": 1, "approval_id": identifier,
                        "main_approval_id": row["main_approval_id"], "content": content,
                        "sha256": plan_digest(content), "generation": generation,
                        # Outside the content, so the plan's identity does not depend on its media.
                        **({"assets": assets} if assets else {})}
                    artifact = M2Service(self.coordinator)._artifact(connection, project_id,
                        "planning-approval-" + identifier, "planning_approval", "plan.json", "application/json",
                        encode_json(approved), provenance={"planning_protocol": planning_protocol()})
                    connection.execute("INSERT INTO planning_approval VALUES (?,?,?,?,?,?,?)", (
                        identifier, project_id, row["id"], row["revision"], artifact["id"], approved["sha256"], now))
                    connection.execute("UPDATE planning_draft SET approved_plan_id=?,status='approved',active_job_id=NULL,error=NULL WHERE id=?",
                                       (identifier, row["id"]))
                    from .m3_service import M3Service
                    production_id = M3Service(self.coordinator).start_approved(
                        connection, project_id, row["main_approval_id"], identifier)["id"]
                    state["draft"].update(planApproved=True, step="production")
                else:
                    connection.execute(
                        "UPDATE planning_draft SET revision=revision+1,content=?,status='ready',"
                        "active_job_id=NULL,approved_plan_id=NULL,error=NULL,updated_at=? WHERE id=?",
                        (json.dumps(content, ensure_ascii=False) if content is not None else None, self.clock(), row["id"]))
                    state["draft"].update(planApproved=False, step="planning-review")
                    if body.action != "save":
                        self._enqueue(connection, required(connection, "planning_draft", row["id"]), body=body)
                    else:
                        self._sync_assets(connection, required(connection, "planning_draft", row["id"]), content)
                        if "plot" not in content and content["cast_plan"] != (previous or {}).get("cast_plan"):
                            # The cast changed before its count was chosen: judge the count again.
                            self._enqueue(connection, required(connection, "planning_draft", row["id"]), judge=True)
                state["draft"]["revision"] += 1
                M2Service(self.coordinator)._save(connection, project_id, state)
            result = self._view(connection, project_id)
        if production_id:
            from .m3_service import M3Service
            M3Service(self.coordinator).advance(production_id)
        return result

    def complete(self, job_id, worker_id, lease_id, data):
        from .m3_bundle import validate_bundle
        with self.db.transaction() as connection:
            job, _ = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            row = self.assert_selected_job(connection, job)
            snapshot = self._snapshot(connection, row)
            payload = json.loads(job["payload"])
        try:
            envelope, _ = validate_bundle(data, "m3_plan")
            content = self._valid_content(envelope["result"], snapshot)
            if envelope["provenance"].get("planning_protocol") != payload["planning_protocol"]:
                raise ValueError("Planning protocol differs from the claimed job.")
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile, EOFError, zlib.error, RuntimeError) as exc:
            raise ServiceError(422, "構成生成の結果が現在の契約と一致しません。") from exc
        stored = self.store.put(data)
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            row = self.assert_selected_job(connection, job)
            artifact = self.coordinator._register_artifact(connection, job["project_id"],
                "planning-job-" + job_id, "m3_bundle", "plan-result.zip", "application/zip", stored,
                source_job_id=job_id, source_attempt_id=attempt["id"])
            now = self.clock()
            connection.execute("UPDATE planning_draft SET content=?,revision=revision+1,status='ready',"
                "active_job_id=NULL,error=NULL,updated_at=? WHERE id=?",
                (json.dumps(content, ensure_ascii=False), now, row["id"]))
            self._sync_assets(connection, row, content)
            connection.execute("UPDATE job_attempt SET status='completed',ended_at=? WHERE id=?", (now, attempt["id"]))
            connection.execute("UPDATE job SET status='completed',result_artifact_id=?,updated_at=?,error=NULL WHERE id=?",
                               (artifact["id"], now, job_id))
            connection.execute("UPDATE generation_run SET status='completed' WHERE id=?", (job["run_id"],))
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            self.history.finish(connection, job["project_id"])
            return {"job": public(required(connection, "job", job_id)), "artifact": public(artifact)}

    @staticmethod
    def restore(connection, project_id, snapshot):
        """Restore content, never old jobs; increase revision to invalidate callbacks."""
        identifier = snapshot["selection"].get("planning_id")
        planning = snapshot.get("planning_state")
        connection.execute("UPDATE project_history_state SET planning_id=? WHERE project_id=?", (identifier, project_id))
        if not identifier or not planning:
            return
        row = required(connection, "planning_draft", identifier)
        connection.execute("UPDATE planning_draft SET revision=?,content=?,status=?,active_job_id=NULL,"
            "approved_plan_id=?,error=NULL WHERE id=?", (max(row["revision"], planning["revision"]) + 1,
                planning["content"], "approved" if planning["approved_plan_id"] else "ready" if planning["content"] else "draft",
                planning["approved_plan_id"], identifier))

    @staticmethod
    def history_state(connection, project_id):
        row = PlanningService.selected(connection, project_id)
        return copy.deepcopy(row) if row else None
