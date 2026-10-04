"""Operation checkpoints: immutable metadata, shared media, explicit adopted versions."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .service import ServiceError, required


class RestoreRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    expected_version: int = Field(ge=1)
    expected_revision: int = Field(ge=1)


LABELS = {
    "save-brief": "世界観・メインキャラの指示を保存",
    "save-world": "世界観の指示を保存", "edit-world": "世界観を編集",
    "generate-world": "世界観を生成・修正", "confirm-world": "世界観を確定・メインキャラを生成",
    "save-characters": "メインキャラの指示を保存", "generate-characters": "メインキャラを生成",
    "revise-character": "キャラクターを修正", "edit-character": "キャラクターを編集",
    "toggle-lock": "キャラクターの固定を変更", "retake": "素材をリテイク",
    "save-relationships": "関係性の指示を保存", "generate-relationships": "関係性を生成・修正",
    "clone-voice": "台詞の音声を試す", "approve": "メインキャラを確定・全体構成を生成",
}


def operation_label(action, state):
    label = LABELS.get(action.action, action.action)
    cid = getattr(action, "character_id", None)
    if cid:
        character = next((item for item in state["draft"]["characters"] if item["id"] == cid), None)
        if character:
            name = (character.get("result") or character["input"]).get("name") or "キャラクター"
            label = f"{name}：{label}"
    return label


class HistoryService:
    def __init__(self, coordinator):
        self.coordinator = coordinator
        self.db, self.store, self.clock = coordinator.db, coordinator.store, coordinator.clock

    @staticmethod
    def selection(connection, project_id):
        row = connection.execute(
            "SELECT * FROM project_history_state WHERE project_id=?", (project_id,),
        ).fetchone()
        return dict(row) if row else None

    @staticmethod
    def busy(connection, project_id):
        return connection.execute(
            "SELECT 1 FROM job WHERE project_id=? AND (status='running' OR "
            "(status='pending' AND NOT EXISTS (SELECT 1 FROM planning_job pj "
            "JOIN planning_draft p ON p.id=pj.planning_id "
            "LEFT JOIN project_history_state h ON h.project_id=p.project_id "
            "JOIN m2_draft md ON md.project_id=p.project_id "
            "WHERE pj.job_id=job.id AND (h.planning_id IS NULL OR h.planning_id!=p.id "
            "OR p.active_job_id IS NULL OR p.active_job_id!=job.id OR p.revision!=pj.revision "
            "OR json_extract(md.state,'$.draft.approved')!=1 "
            "OR json_extract(md.state,'$.draft.approval.id')!=p.main_approval_id)) "
            # Later media remain queued after a terminal stage failure. They
            # cannot execute until retry, so they must not keep history busy.
            "AND NOT EXISTS (SELECT 1 FROM m3_production_job pj "
            "JOIN m3_requirement r ON r.production_id=pj.production_id "
            "JOIN m3_production owner ON owner.id=pj.production_id "
            "LEFT JOIN job blocker ON blocker.id=r.job_id "
            "WHERE pj.job_id=job.id AND r.artifact_id IS NULL "
            "AND (blocker.status='failed' OR EXISTS ("
            "SELECT 1 FROM m3_requirement prior "
            "JOIN m3_production prior_chapter ON prior_chapter.id=prior.production_id "
            "JOIN job shared_blocker ON shared_blocker.id=prior.job_id "
            "WHERE prior_chapter.storyline_id=owner.storyline_id "
            "AND prior_chapter.chapter_number<owner.chapter_number "
            "AND prior.kind=r.kind AND prior.target_id=r.target_id AND prior.descriptor=r.descriptor "
            "AND prior.artifact_id IS NULL AND shared_blocker.status='failed')) AND ("
            "(job.kind='m3_background' AND r.kind='m3_image') OR "
            "(job.kind='m3_voice' AND r.kind IN ('m3_image','m3_background')) OR "
            "(job.kind='m3_voice_clone' AND r.kind IN ('m3_image','m3_background','m3_voice')))) "
            "AND NOT EXISTS (SELECT 1 FROM m3_production_job pj "
            "JOIN m3_production p ON p.id=pj.production_id "
            "JOIN m3_production root ON root.id=p.storyline_id "
            "LEFT JOIN project_history_state h ON h.project_id=p.project_id "
            "WHERE pj.job_id=job.id AND (root.control_state!='running' "
            "OR h.production_frozen=1 OR h.production_id!=root.id "
            "OR (root.plan_approval_id IS NOT NULL AND NOT EXISTS (SELECT 1 FROM planning_draft dp "
            "WHERE dp.id=h.planning_id AND dp.approved_plan_id=root.plan_approval_id)))))) LIMIT 1",
            (project_id,),
        ).fetchone() is not None

    @staticmethod
    def _operations(selection):
        return json.loads(selection["operation"]) if selection and selection["operation"] else []

    def has_operation(self, connection, project_id, domain):
        return any(item["domain"] == domain for item in self._operations(self.selection(connection, project_id)))

    @staticmethod
    def production_state(connection, production_id, *, children=True):
        if not production_id:
            return None
        production = required(connection, "m3_production", production_id)
        result = {
            "narrative_artifact_id": production["narrative_artifact_id"], "error": production["error"],
            "jobs": [dict(row) for row in connection.execute(
                "SELECT j.id,j.status,j.error,j.attempt_count,j.max_attempts,j.result_artifact_id,j.updated_at,j.progress FROM job j "
                "JOIN m3_production_job p ON p.job_id=j.id WHERE p.production_id=? ORDER BY j.created_at,j.id",
                (production_id,),
            )],
            "requirements": [dict(row) for row in connection.execute(
                "SELECT * FROM m3_requirement "
                "WHERE production_id=? ORDER BY rowid", (production_id,),
            )],
        }
        if children:
            chapters = []
            for row in connection.execute(
                "SELECT * FROM m3_production WHERE storyline_id=? AND chapter_number>1 "
                "ORDER BY chapter_number", (production["storyline_id"],),
            ):
                build = connection.execute(
                    "SELECT id FROM chapter_build WHERE production_id=? ORDER BY revision DESC LIMIT 1",
                    (row["id"],),
                ).fetchone()
                chapters.append({"id": row["id"], "chapter_number": row["chapter_number"],
                                 "build_id": build["id"] if build else None,
                                 **HistoryService.production_state(connection, row["id"], children=False)})
            if chapters:
                result["chapters"] = chapters
        return result

    def ensure(self, connection, project_id):
        existing = self.selection(connection, project_id)
        if existing:
            return existing
        required(connection, "project", project_id)
        if not connection.execute("SELECT 1 FROM m2_draft WHERE project_id=?", (project_id,)).fetchone():
            raise ServiceError(404, "履歴を保存できる作品が見つかりません。")
        production = connection.execute(
            "SELECT * FROM m3_production WHERE project_id=? AND chapter_number=1 "
            "ORDER BY created_at DESC,id DESC LIMIT 1",
            (project_id,),
        ).fetchone()
        build = connection.execute(
            "SELECT * FROM chapter_build WHERE production_id=? ORDER BY revision DESC LIMIT 1",
            (production["id"],),
        ).fetchone() if production else None
        portrait_id = json.loads(build["validation"]).get("portrait_settings_artifact_id") if build else None
        connection.execute(
            "INSERT INTO project_history_state(project_id,production_id,build_id,portrait_settings_id) "
            "VALUES (?,?,?,?)",
            (project_id, production["id"] if production else None, build["id"] if build else None, portrait_id),
        )
        self._append(connection, project_id, "履歴の保存を開始", "baseline", self._snapshot(connection, project_id))
        return self.selection(connection, project_id)

    def set_selection(self, connection, project_id, production_id, build_id, portrait_settings_id, *, frozen=False):
        self.ensure(connection, project_id)
        values = (production_id, build_id, portrait_settings_id, int(frozen))
        current = self.selection(connection, project_id)
        if values != tuple(current[key] for key in ("production_id", "build_id", "portrait_settings_id", "production_frozen")):
            if production_id != current["production_id"] or build_id != current["build_id"]:
                connection.execute("UPDATE project_history_state SET edition_id=NULL,adjustment_draft_id=NULL WHERE project_id=?", (project_id,))
            connection.execute(
                "UPDATE project_history_state SET production_id=?,build_id=?,portrait_settings_id=?,"
                "production_frozen=?,production_snapshot=NULL,version=version+1 WHERE project_id=?", (*values, project_id),
            )

    def _snapshot(self, connection, project_id):
        from .m2_service import M2Service

        state = M2Service._load(connection, project_id)
        # Jobs are audit records, never restorable work. Pending content flags stay
        # intact so a partial/failed result cannot become falsely approved.
        state["queue"] = []
        state["activeJobId"] = None
        selection = self.selection(connection, project_id)
        production_state = (json.loads(selection["production_snapshot"])
                            if selection["production_frozen"] and selection["production_snapshot"]
                            else self.production_state(connection, selection["production_id"]))
        from .planning_service import PlanningService

        result = {"schema_version": 1, "state": state, "production_state": production_state,
                "planning_state": PlanningService.history_state(connection, project_id),
                "adjustment_state": required(connection, "adjustment_draft", selection["adjustment_draft_id"])
                    if selection.get("adjustment_draft_id") else None, "selection": {
            key: selection.get(key) for key in ("production_id", "build_id", "portrait_settings_id", "planning_id", "edition_id", "adjustment_draft_id")
        }}
        if selection.get("adjustment_draft_id"):
            # Candidate-only actions do not change selected character settings.
            # Retain their audit state so generation/upload/sample operations
            # still receive an explicit history entry. Restoring a selection
            # never removes candidates or rewinds completed worker attempts.
            draft = result["adjustment_state"]
            result["adjustment_candidates"] = [dict(row) for row in connection.execute(
                "SELECT id,character_id,kind,artifact_id,original_artifact_id,reference_text,source,prompt,job_id,sample_artifact_id "
                "FROM adjustment_candidate WHERE production_id=? ORDER BY created_at,id", (draft["production_id"],))]
            result["adjustment_jobs"] = [dict(row) for row in connection.execute(
                "SELECT j.id,j.status,j.attempt_count,j.result_artifact_id FROM job j JOIN adjustment_job a ON a.job_id=j.id "
                "WHERE a.draft_id=? ORDER BY j.created_at,j.id", (draft["id"],))]
        return result

    @staticmethod
    def _content(snapshot):
        result = copy.deepcopy(snapshot)
        draft = result["state"]["draft"]
        draft.pop("revision", None)
        draft.pop("step", None)
        if result.get("adjustment_state"):
            result["adjustment_state"].pop("revision", None)
        return result

    def _append(self, connection, project_id, label, action, snapshot, restored_from_id=None):
        number = connection.execute(
            "SELECT COALESCE(MAX(number),0)+1 FROM project_revision WHERE project_id=?", (project_id,),
        ).fetchone()[0]
        identifier = uuid4().hex
        connection.execute(
            "INSERT INTO project_revision VALUES (?,?,?,?,?,?,?,?)",
            (identifier, project_id, number, label, action,
             json.dumps(snapshot, ensure_ascii=False, separators=(",", ":")), restored_from_id, self.clock()),
        )
        connection.execute(
            "UPDATE project_history_state SET current_revision_id=?,version=version+1 WHERE project_id=?",
            (identifier, project_id),
        )
        return identifier

    def begin(self, connection, project_id, label, action, *, concurrent=False):
        self.ensure(connection, project_id)
        domain = "m3" if action == "approve" or action.startswith("m3-") else "m2"
        if not concurrent and connection.execute(
            "SELECT 1 FROM job WHERE project_id=? AND kind LIKE ? AND status IN ('pending','running') LIMIT 1",
            (project_id, domain + "_%"),
        ).fetchone() and self.busy(connection, project_id):
            raise ServiceError(409, "生成処理中です。完了または失敗を確認してから変更してください。")
        self.finish(connection, project_id)
        selection = self.selection(connection, project_id)
        operations = self._operations(selection)
        previous = required(connection, "project_revision", selection["current_revision_id"])
        snapshot = self._snapshot(connection, project_id)
        if not operations and self._content(json.loads(previous["snapshot"])) != self._content(snapshot):
            self._append(connection, project_id, "変更前の状態を保存", "checkpoint", snapshot)
        operations.append({"id": uuid4().hex, "label": label, "action": action, "domain": domain})
        connection.execute(
            "UPDATE project_history_state SET operation=?,version=version+1 WHERE project_id=?",
            (json.dumps(operations, ensure_ascii=False), project_id),
        )

    def finish(self, connection, project_id):
        selection = self.selection(connection, project_id)
        operations = self._operations(selection)
        if not operations:
            return
        remaining = []
        for operation in operations:
            if not self._finish_operation(connection, project_id, operation):
                remaining.append(operation)
        if remaining != operations:
            connection.execute(
                "UPDATE project_history_state SET operation=?,version=version+1 WHERE project_id=?",
                (json.dumps(remaining, ensure_ascii=False) if remaining else None, project_id),
            )

    def _finish_operation(self, connection, project_id, operation):
        selection = self.selection(connection, project_id)
        if connection.execute(
            "SELECT 1 FROM job WHERE project_id=? AND kind LIKE ? AND status IN ('pending','running') LIMIT 1",
            (project_id, operation["domain"] + "_%"),
        ).fetchone() and self.busy(connection, project_id):
            return False
        # M3 commits a result before advancing its durable pipeline. Do not split
        # one operation at that recovery window between two generation jobs.
        pid = selection["production_id"]
        failed = False
        if operation["domain"] == "m3" and pid and not selection["build_id"] and not selection["production_frozen"]:
            production = required(connection, "m3_production", pid)
            failed = bool(production["error"] or connection.execute(
                "SELECT 1 FROM m3_production_job p JOIN job j ON j.id=p.job_id "
                "WHERE p.production_id=? AND j.status='failed' LIMIT 1", (pid,),
            ).fetchone())
            if not failed and production["control_state"] == "running":
                return False
        raw_state = json.loads(connection.execute(
            "SELECT state FROM m2_draft WHERE project_id=?", (project_id,),
        ).fetchone()[0])
        active = raw_state.get("activeJobId")
        if active and operation["domain"] == "m2":
            job = required(connection, "job", active)
            failed = failed or job["status"] == "failed"
        if operation["domain"] == "m2" and raw_state.get("queue") and not failed:
            return False
        snapshot = self._snapshot(connection, project_id)
        previous = required(connection, "project_revision", selection["current_revision_id"])
        if self._content(json.loads(previous["snapshot"])) != self._content(snapshot):
            label = operation["label"] + ("（途中で失敗）" if failed else "")
            self._append(connection, project_id, label, operation["action"], snapshot)
        return True

    def list(self, project_id):
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            self.ensure(connection, project_id)
            self.finish(connection, project_id)
            selected = self.selection(connection, project_id)
            busy = self.busy(connection, project_id)
            operations = self._operations(selected)
            operation = ({"label": " / ".join(item["label"] for item in operations),
                          "status": "running" if busy else "pending"} if operations else None)
            return {
                "project_id": project_id, "version": selected["version"],
                "current_revision_id": selected["current_revision_id"], "busy": busy,
                "pending_operation": operation,
                "entries": [{
                    "id": row["id"], "number": row["number"], "label": row["label"],
                    "created_at": datetime.fromtimestamp(row["created_at"], UTC).isoformat(),
                    "restored_from_id": row["restored_from_id"],
                    "current": row["id"] == selected["current_revision_id"],
                } for row in connection.execute(
                    "SELECT id,number,label,created_at,restored_from_id FROM project_revision "
                    "WHERE project_id=? ORDER BY number DESC", (project_id,),
                )],
            }

    def _validate_snapshot(self, connection, project_id, snapshot):
        references = set()

        def collect(value):
            if isinstance(value, dict):
                for key, child in value.items():
                    if (key == "artifactId" or key.endswith("ArtifactId")) and isinstance(child, str):
                        references.add(child)
                    else:
                        collect(child)
            elif isinstance(value, list):
                for child in value:
                    collect(child)

        collect(snapshot["state"]["draft"])
        planning_id = snapshot["selection"].get("planning_id")
        planning_state = snapshot.get("planning_state")
        if planning_id:
            planning = required(connection, "planning_draft", planning_id)
            if planning["project_id"] != project_id or not planning_state or planning_state["id"] != planning_id:
                raise ValueError("planning belongs to another project")
            if planning_state["approved_plan_id"]:
                approved = required(connection, "planning_approval", planning_state["approved_plan_id"])
                if approved["project_id"] != project_id or approved["planning_id"] != planning_id:
                    raise ValueError("plan approval belongs to another planning draft")
                references.add(approved["artifact_id"])
        selected = snapshot["selection"]
        if selected.get("edition_id"):
            edition = required(connection, "publication_edition", selected["edition_id"])
            if edition["project_id"] != project_id or edition["production_id"] != selected["production_id"]:
                raise ValueError("publication edition belongs to another story")
            for row in connection.execute("SELECT b.* FROM edition_build e JOIN chapter_build b ON b.id=e.build_id WHERE e.edition_id=?", (edition["id"],)):
                references.update((row["script_artifact_id"], row["export_artifact_id"]))
                references.update(item["artifact_id"] for item in json.loads(row["manifest"]))
            if not connection.execute("SELECT 1 FROM edition_build WHERE edition_id=? AND production_id=? AND build_id=?",
                (edition["id"], selected["production_id"], selected["build_id"])).fetchone():
                raise ValueError("selected first chapter differs from its publication edition")
        adjustment = snapshot.get("adjustment_state")
        if selected.get("adjustment_draft_id"):
            from packages.contracts.adjustments import AdjustmentCharacter

            row = required(connection, "adjustment_draft", selected["adjustment_draft_id"])
            if (row["project_id"] != project_id or not adjustment or adjustment["id"] != row["id"]
                    or adjustment["production_id"] != selected["production_id"]):
                raise ValueError("adjustment draft belongs to another story")
            source = required(connection, "publication_edition", adjustment["base_edition_id"])
            if source["project_id"] != project_id or source["production_id"] != selected["production_id"]:
                raise ValueError("adjustment source belongs to another story")
            adjustment_content = json.loads(adjustment["state"])
            for value in adjustment_content["characters"]:
                person = AdjustmentCharacter.model_validate(value)
                for kind in ("image", "voice"):
                    identifier = getattr(person, kind + "_candidate_id")
                    if not identifier:
                        continue
                    candidate = required(connection, "adjustment_candidate", identifier)
                    if (candidate["project_id"] != project_id or candidate["production_id"] != selected["production_id"]
                            or candidate["character_id"] != person.character_id or candidate["kind"] != kind):
                        raise ValueError("adjustment candidate belongs to another story or character")
                    references.add(candidate["artifact_id"])
            for rows in adjustment_content["requirements"].values():
                references.update(item["artifact_id"] for item in rows if item["artifact_id"])
            for candidate in snapshot.get("adjustment_candidates", []):
                original = required(connection, "adjustment_candidate", candidate["id"])
                if original["project_id"] != project_id or original["production_id"] != selected["production_id"]:
                    raise ValueError("candidate audit belongs to another story")
                references.update(candidate[key] for key in ("artifact_id", "original_artifact_id", "sample_artifact_id") if candidate[key])
            for job in snapshot.get("adjustment_jobs", []):
                original = required(connection, "job", job["id"])
                if original["project_id"] != project_id:
                    raise ValueError("adjustment job audit belongs to another story")
                if job["result_artifact_id"]:
                    references.add(job["result_artifact_id"])
        production_id, build_id = selected["production_id"], selected["build_id"]
        if production_id:
            production = required(connection, "m3_production", production_id)
            if production["project_id"] != project_id:
                raise ValueError("production belongs to another project")
        if build_id:
            build = required(connection, "chapter_build", build_id)
            if build["project_id"] != project_id or build["production_id"] != production_id:
                raise ValueError("build belongs to another project or production")
            references.update((build["script_artifact_id"], build["export_artifact_id"]))
            references.update(item["artifact_id"] for item in json.loads(build["manifest"]))
        if selected["portrait_settings_id"]:
            references.add(selected["portrait_settings_id"])
        production_state = snapshot.get("production_state")
        if production_state:
            if production_state["narrative_artifact_id"]:
                references.add(production_state["narrative_artifact_id"])
            references.update(item["artifact_id"] for item in production_state["requirements"] if item["artifact_id"])
            for chapter in production_state.get("chapters", []):
                row = required(connection, "m3_production", chapter["id"])
                if row["storyline_id"] != production_id or row["project_id"] != project_id:
                    raise ValueError("chapter belongs to another storyline")
                if chapter["narrative_artifact_id"]:
                    references.add(chapter["narrative_artifact_id"])
                references.update(item["artifact_id"] for item in chapter["requirements"] if item["artifact_id"])
                if chapter.get("build_id"):
                    build = required(connection, "chapter_build", chapter["build_id"])
                    if build["production_id"] != chapter["id"]:
                        raise ValueError("chapter build belongs to another production")
                    references.update((build["script_artifact_id"], build["export_artifact_id"]))
        for identifier in references:
            record = required(connection, "artifact", identifier)
            if record["project_id"] != project_id:
                raise ValueError("artifact belongs to another project")
            self.store.read(record)

    def restore(self, project_id, revision_id, request: RestoreRequest):
        from .m2_service import M2Service

        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            self.ensure(connection, project_id)
            self.finish(connection, project_id)
            selection = self.selection(connection, project_id)
            current = M2Service._load(connection, project_id)
            if selection["version"] != request.expected_version or current["draft"]["revision"] != request.expected_revision:
                raise ServiceError(409, "作品または履歴が更新されました。最新の履歴を確認してから復元してください。")
            if self.busy(connection, project_id):
                raise ServiceError(409, "生成処理中は復元できません。完了または失敗を確認してください。")
            target = connection.execute(
                "SELECT * FROM project_revision WHERE id=? AND project_id=?", (revision_id, project_id),
            ).fetchone()
            if target is None:
                raise ServiceError(404, "この作品の履歴が見つかりません。")
            snapshot = json.loads(target["snapshot"])
            try:
                self._validate_snapshot(connection, project_id, snapshot)
            except (OSError, ValueError, KeyError, ServiceError) as exc:
                raise ServiceError(422, "復元に必要な設定・素材を確認できません。現在の状態を保持しました。") from exc
            # Preserve the latest state too, if this is a failed/legacy operation
            # whose last callback never ran. Restoring never discards a later edit.
            present = self._snapshot(connection, project_id)
            head = required(connection, "project_revision", selection["current_revision_id"])
            if self._content(present) != self._content(json.loads(head["snapshot"])):
                self._append(connection, project_id, "復元前の状態を保存", "checkpoint", present)
            if revision_id == selection["current_revision_id"] and self._content(present) == self._content(snapshot):
                return M2Service(self.coordinator)._view(connection, project_id, current)
            restored = copy.deepcopy(snapshot["state"])
            restored["draft"]["revision"] = current["draft"]["revision"] + 1
            restored["queue"], restored["activeJobId"] = [], None
            selected = snapshot["selection"]
            self.set_selection(connection, project_id, selected["production_id"], selected["build_id"], selected["portrait_settings_id"], frozen=True)
            from .planning_service import PlanningService
            PlanningService.restore(connection, project_id, snapshot)
            adjustment = snapshot.get("adjustment_state")
            if adjustment:
                current_adjustment = required(connection, "adjustment_draft", adjustment["id"])
                if current_adjustment["project_id"] != project_id:
                    raise ServiceError(422, "調整版が別の作品に属しています。")
                connection.execute("UPDATE adjustment_draft SET revision=?,state=?,status=?,base_edition_id=?,"
                    "active_apply_id=NULL,error=NULL WHERE id=?", (max(current_adjustment["revision"], adjustment["revision"]) + 1,
                    adjustment["state"], "draft" if adjustment["status"] != "applied" else "applied",
                    adjustment["base_edition_id"], adjustment["id"]))
            connection.execute("UPDATE project_history_state SET edition_id=?,adjustment_draft_id=? WHERE project_id=?",
                (selected.get("edition_id"), selected.get("adjustment_draft_id"), project_id))
            connection.execute(
                "UPDATE project_history_state SET operation=NULL,production_snapshot=? WHERE project_id=?",
                (json.dumps(snapshot.get("production_state"), ensure_ascii=False), project_id),
            )
            service = M2Service(self.coordinator)
            service._save(connection, project_id, restored)
            self._append(connection, project_id, f"版{target['number']}の状態に復元", "restore",
                         self._snapshot(connection, project_id), restored_from_id=revision_id)
            return service._view(connection, project_id, restored)
