"""Editable common plans and immutable approvals before chapter production."""
from __future__ import annotations

import copy
import json
import zipfile
import zlib
from uuid import uuid4

from packages.contracts.planning import planning_protocol, validate_plan_content

from .m2_service import M2Service
from .project_history import HistoryService
from .service import Coordinator, ServiceError, encode_json, public, required


def plan_digest(content):
    from .m3_service import _checkpoint_digest
    return _checkpoint_digest(content)


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
            "jobs": jobs, "error": active["error"] if active else row["error"]}
        return result

    def project(self, project_id):
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            self.coordinator._recover(connection)
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)

    def start_approved(self, connection, project_id, approval_id):
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
        self._enqueue(connection, row)
        return required(connection, "planning_draft", identifier)

    def _enqueue(self, connection, row, *, body=None):
        from services.worker.generation.workflow_version import generator_protocol

        from .m3_service import GENERATION_ROOT
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
            profile = {key: value for key, value in M2Service(self.coordinator)._profile().items()
                       if key not in {"max_tokens", "prompt_version"}}
            settings = json.loads((GENERATION_ROOT / "config/m2-generation.json").read_text(encoding="utf-8"))["llm"]
            if not profile.get("common_settings_version"):
                profile.update(reasoning_level="none", context_size=settings.get("context_size", 16384))
            generation = {"profile": profile, "tts_profile": TTSService(self.coordinator).profile(connection),
                "seed": int(uuid4().hex[:8], 16) & 0x7FFFFFFF, "story_workflow_version": 2,
                "workflow_policy": "script_continuation_v1",
                "generator_protocol": generator_protocol("causal", "script_continuation_v1"),
                "planning_protocol": planning_protocol()}
        payload = {"schema_version": 1, "phase": "planning", "planning_id": row["id"],
            "planning_revision": row["revision"], "approval_snapshot": self._snapshot(connection, row),
            **generation}
        if body and body.action == "revise":
            payload.update(plan_content=json.loads(row["content"]), target=body.target,
                           character_id=body.character_id, chapter_number=body.chapter_number,
                           instruction=body.instruction)
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
                }[body.action], "m3-plan-" + body.action)
                if reconfirm:
                    self.history.clear_downstream(connection, project_id)
                    connection.execute(
                        "UPDATE planning_draft SET revision=revision+1,approved_plan_id=NULL,updated_at=? WHERE id=?",
                        (self.clock(), row["id"]),
                    )
                    row = required(connection, "planning_draft", row["id"])
                if body.action == "save":
                    content = self._valid_content(body.content, self._snapshot(connection, row))
                else:
                    content = json.loads(row["content"]) if row["content"] else None
                if body.action in {"revise", "approve"} and content is None:
                    raise ServiceError(409, "全体構成の生成完了後に操作してください。")
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
                    approved = {"schema_version": 1, "approval_id": identifier,
                        "main_approval_id": row["main_approval_id"], "content": content,
                        "sha256": plan_digest(content), "generation": generation}
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
