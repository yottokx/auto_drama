"""Music candidates for a complete story, scoped by chapter and scene."""
from __future__ import annotations

import json
import zipfile
from uuid import uuid4

from .adjustment_service import AdjustmentService
from .m3_bundle import validate_bundle
from .music_service import (
    candidate,
    context,
    effective_settings,
    original_settings,
    store_candidate,
    validate_continuity,
    validate_plan,
    validate_result,
)
from .service import ServiceError, public, required


def inspect_music(service, connection, root, draft):
    if not root:
        return [], []
    rows = [dict(row) for row in connection.execute("SELECT c.*,j.status AS job_status,j.error AS job_error FROM music_candidate c "
        "LEFT JOIN job j ON j.id=c.job_id WHERE c.storyline_id=? ORDER BY c.created_at,c.id", (root["id"],))]
    for row in rows:
        metadata = json.loads(row["metadata"])
        row.update({key: metadata.get(key) for key in ("loop_start_seconds", "loop_end_seconds", "duration_seconds",
                                                       "source_duration_seconds", "quality")})
        row["music_url"] = f"/api/artifacts/{row['artifact_id']}/content" if row["artifact_id"] else None
        row["source_url"] = f"/api/artifacts/{row['source_artifact_id']}/content" if row["source_artifact_id"] else None
        row["status"] = "completed" if row["artifact_id"] else row.pop("job_status") or "pending"
        row["error"] = row.pop("job_error")
        row.pop("job_status", None)
    jobs = [public(dict(row)) | {"kind": "music", "generation_kind": row["kind"]} for row in connection.execute(
        "SELECT j.* FROM job j WHERE j.id IN (SELECT job_id FROM music_adjustment_job WHERE draft_id=? "
        "UNION SELECT job_id FROM music_replan_job WHERE draft_id=?) ORDER BY j.created_at,j.id",
        (draft["id"], draft["id"]))] if draft else []
    for job in jobs:
        if job["payload"].get("adjustment", {}).get("replan"):
            job["purpose"] = "music_replan"
            if job["result_artifact_id"]:
                metadata = json.loads(required(connection, "artifact", job["result_artifact_id"])["provenance"])
                job["adoption_status"] = "adopted" if metadata.get("replan_adopted") else "stale"
                job["adoption_reason"] = metadata.get("replan_adoption_reason", "")
    return rows, jobs


def validate_settings(service, connection, draft, state, settings):
    from packages.contracts.music import SceneMusicSetting

    chapters = [dict(row) for row in connection.execute("SELECT * FROM m3_production WHERE storyline_id=? ORDER BY chapter_number", (draft["production_id"],))]
    expected = {(row["production_id"], row["scene_id"]) for row in original_settings(connection, chapters, service.m3)}
    values = [SceneMusicSetting.model_validate(row).model_dump(mode="json") for row in settings]
    if {(row["production_id"], row["scene_id"]) for row in values} != expected or len(values) != len(expected):
        raise ServiceError(422, "作品の全場面のBGM設定を指定してください。")
    for value in values:
        if value["candidate_id"]:
            candidate(connection, value["production_id"], value["scene_id"], value["candidate_id"], draft["project_id"])
    validate_continuity(values, chapters, service.m3, connection)
    state["scene_music"] = values


class MusicAdjustmentService(AdjustmentService):
    def _scene(self, connection, root, chapters, production_id, scene_id):
        chapter = next((row for row in chapters if row["id"] == production_id), None)
        if not chapter:
            raise ServiceError(422, "この作品の章を指定してください。")
        frozen = context(self.m3, connection, chapter)
        scenes = [row for row in frozen["narrative"]["scenes"] if row["id"] == scene_id]
        if not scenes:
            raise ServiceError(422, "この章の場面を指定してください。")
        frozen["narrative"]["scenes"] = scenes
        return chapter, frozen

    def _enqueue_music(self, connection, draft, root, chapter, candidate_id, kind, material):
        identifier, run_id = uuid4().hex, uuid4().hex
        frozen = connection.execute("SELECT j.settings_snapshot FROM job j JOIN m3_production_job p ON p.job_id=j.id "
                                    "WHERE p.production_id=? AND j.kind='m3_narrative' LIMIT 1", (root["id"],)).fetchone()
        settings = json.loads(frozen[0])
        payload = {"schema_version": 1, "approval_snapshot": self.m3._snapshot(connection, root),
                   "production_id": chapter["id"], "chapter_number": chapter["chapter_number"],
                   "profile": settings.get("profile", {}), "seed": int(identifier[:8], 16) & 0x7fffffff,
                   "adjustment": {"music": True, "draft_id": draft["id"], "revision": draft["revision"]}, **material}
        project = required(connection, "project", root["project_id"])
        connection.execute("INSERT INTO generation_run(id,project_id,settings_version,story_revision_id,policy,status,created_at) "
                           "VALUES (?,?,?,?,?,'pending',?)", (run_id, root["project_id"], project["settings_version"], draft["id"],
                           json.dumps({"mode": "music-adjustment"}), self.clock()))
        connection.execute("INSERT INTO job(id,project_id,run_id,kind,payload,settings_snapshot,status,max_attempts,created_at,updated_at) "
                           "VALUES (?,?,?,?,?,?,'pending',3,?,?)", (identifier, root["project_id"], run_id, kind, json.dumps(payload),
                           json.dumps(settings), self.clock(), self.clock()))
        scene_id = material.get("scene_id") or material["context"]["narrative"]["scenes"][0]["id"]
        if candidate_id is None:
            payload["adjustment"]["replan"] = True
            connection.execute("UPDATE job SET payload=? WHERE id=?", (json.dumps(payload), identifier))
            connection.execute("INSERT INTO music_replan_job VALUES (?,?,?,?)",
                               (identifier, draft["id"], chapter["id"], draft["revision"]))
        else:
            connection.execute("INSERT INTO music_adjustment_job VALUES (?,?,?,?,?,?)", (identifier, draft["id"], candidate_id,
                               chapter["id"], scene_id, draft["revision"]))
            connection.execute("UPDATE music_candidate SET job_id=? WHERE id=?", (identifier, candidate_id))
        return identifier

    def generate(self, project_id, request):
        with self.db.transaction() as connection:
            draft, root, chapters = self._editable(connection, project_id, request.expected_revision)
            chapter, frozen = self._scene(connection, root, chapters, request.production_id, request.scene_id)
            self.history.begin(connection, project_id, "場面のBGM候補を生成", "m3-music-generate", concurrent=True)
            identifier = uuid4().hex
            connection.execute("INSERT INTO music_candidate VALUES (?,?,?,?,?,NULL,NULL,'generated',?,'{}',NULL,?)",
                               (identifier, project_id, root["id"], chapter["id"], request.scene_id,
                                request.source_prompt or "", self.clock()))
            updated = {**draft, "revision": draft["revision"] + 1}
            self._enqueue_music(connection, updated, root, chapter, identifier, "m3_music_plan",
                                {"context": frozen, "instruction": request.instruction,
                                 "planning_version": 2, "planning_scope": "single_scene",
                                 **({"source_prompt": request.source_prompt} if request.source_prompt else {})})
            connection.execute("UPDATE adjustment_draft SET revision=revision+1,status='draft',error=NULL WHERE id=?", (draft["id"],))
            return self._view(connection, project_id)

    def replan(self, project_id, request):
        with self.db.transaction() as connection:
            draft, root, chapters = self._editable(connection, project_id, request.expected_revision)
            chapter = next((row for row in chapters if row["id"] == request.production_id), None)
            if not chapter:
                raise ServiceError(422, "この作品の章を指定してください。")
            pending = connection.execute("SELECT 1 FROM music_replan_job a JOIN job j ON j.id=a.job_id "
                "WHERE a.draft_id=? AND a.production_id=? AND j.status IN ('pending','running')",
                (draft["id"], chapter["id"])).fetchone()
            if pending:
                raise ServiceError(409, "この章のBGM切り替えを再計画中です。")
            self.history.begin(connection, project_id, "章のBGM・場面転換を再計画", "m3-music-replan", concurrent=True)
            updated = {**draft, "revision": draft["revision"] + 1}
            self._enqueue_music(connection, updated, root, chapter, None, "m3_music_plan",
                                {"context": context(self.m3, connection, chapter), "planning_version": 2})
            connection.execute("UPDATE adjustment_draft SET revision=revision+1,status='draft',error=NULL WHERE id=?", (draft["id"],))
            return self._view(connection, project_id)

    def _adopt_replan(self, connection, draft, chapter, plan, job, attempt):
        from packages.contracts.music import MusicPlan

        state = json.loads(draft["state"])
        previous = effective_settings(connection, [chapter], self.m3, state)
        by_scene = {row["scene_id"]: row for row in previous}
        active = None
        inherited = {}
        for row in previous:
            if row["action"] == "play":
                active = candidate(connection, chapter["id"], row["scene_id"], row["candidate_id"], chapter["project_id"])
            elif row["action"] == "stop":
                active = None
            inherited[row["scene_id"]] = active
        resolved, settings = [], []
        active = None
        for decision in plan.scenes:
            row = decision.model_dump(mode="json")
            picked = None
            if decision.action == "play":
                own = by_scene[decision.scene_id]
                picked = candidate(connection, chapter["id"], decision.scene_id, own["candidate_id"], chapter["project_id"]) if (
                    own["action"] == "play") else None
                if picked is None:
                    original = connection.execute("SELECT c.id FROM m3_requirement r JOIN music_candidate c ON c.artifact_id=r.artifact_id "
                        "WHERE r.production_id=? AND r.kind='m3_music' AND r.target_id=? ORDER BY c.created_at,c.id LIMIT 1",
                        (chapter["id"], decision.scene_id)).fetchone()
                    picked = candidate(connection, chapter["id"], decision.scene_id, original["id"], chapter["project_id"]) if original else inherited.get(decision.scene_id)
                if picked:
                    if picked["scene_id"] != decision.scene_id:
                        metadata = json.loads(picked["metadata"])
                        result = {**metadata, "scene_id": decision.scene_id, "prompt": picked["prompt"]}
                        from packages.contracts.music import MusicResult

                        result = MusicResult.model_validate({key: result[key] for key in MusicResult.model_fields})
                        files = {"music.mp3": self.store.read(required(connection, "artifact", picked["artifact_id"])),
                                 "source.mp3": self.store.read(required(connection, "artifact", picked["source_artifact_id"]))}
                        _record, identifier = store_candidate(self.m3, connection, chapter, result.model_dump(mode="json"),
                            files, source="reused", job=job, attempt=attempt,
                            provenance={**metadata, "reuse_source_candidate_id": picked["id"]})
                        picked = candidate(connection, chapter["id"], decision.scene_id, identifier, chapter["project_id"])
                    row["prompt"] = picked["prompt"]
                    row["reason"] = decision.reason[:1600] + " 採用済み音源を再利用します（新しい曲は生成しません）。"
                else:
                    row.update(action="stop", prompt="", reason=decision.reason[:1600] + " 採用済みの曲がないため、音楽生成は行わず停止にしました。")
                active = picked
            elif decision.action == "continue" and active is None:
                row.update(action="stop", reason=decision.reason[:1600] + " 継続元の採用曲がないため停止にしました。")
            elif decision.action == "stop":
                active = None
            resolved.append(row)
            settings.append({"production_id": chapter["id"], "scene_id": decision.scene_id,
                "action": row.get("action", "play"), "candidate_id": picked["id"] if picked else None,
                "volume": by_scene[decision.scene_id]["volume"], "transition": row["transition"], "reason": row["reason"]})
        result = MusicPlan(planning_version=2, scenes=resolved)
        record = self.m3._artifact(connection, chapter, "music-replan-" + job["id"], "music_plan", "music-plan.json",
            json.dumps(result.model_dump(mode="json"), ensure_ascii=False).encode(),
            provenance={"production_id": chapter["id"], "replan_job_id": job["id"], "bindings": settings}, job=job, attempt=attempt)
        state.setdefault("music_plans", {})[chapter["id"]] = record["id"]
        chapters = [dict(row) for row in connection.execute("SELECT * FROM m3_production WHERE storyline_id=? ORDER BY chapter_number",
                                                          (chapter["storyline_id"],))]
        previous_all = effective_settings(connection, chapters, self.m3, json.loads(draft["state"]))
        by_key = {(row["production_id"], row["scene_id"]): row for row in settings}
        state["scene_music"] = [by_key.get((row["production_id"], row["scene_id"]), row) for row in previous_all]
        connection.execute("UPDATE adjustment_draft SET state=?,revision=revision+1 WHERE id=?", (json.dumps(state), draft["id"]))

    def validate_upload(self, project_id, revision, production_id, scene_id):
        with self.db.transaction() as connection:
            _draft, root, chapters = self._editable(connection, project_id, revision)
            self._scene(connection, root, chapters, production_id, scene_id)

    def upload(self, project_id, revision, production_id, scene_id, files, result):
        with self.db.transaction() as connection:
            draft, root, chapters = self._editable(connection, project_id, revision)
            chapter, _ = self._scene(connection, root, chapters, production_id, scene_id)
            self.history.begin(connection, project_id, "場面のBGMを取り込み", "m3-music-upload", concurrent=True)
            store_candidate(self.m3, connection, chapter, result, files, source="upload")
            connection.execute("UPDATE adjustment_draft SET revision=revision+1 WHERE id=?", (draft["id"],))
            self.history.finish(connection, project_id)
            return self._view(connection, project_id)

    def complete(self, job_id, worker_id, lease_id, data):
        with self.db.transaction() as connection:
            job, _ = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            payload = json.loads(job["payload"])
        try:
            envelope, files = validate_bundle(data, job["kind"])
            plan = validate_plan(envelope["result"], payload) if job["kind"] == "m3_music_plan" else None
            result = validate_result(envelope["result"], payload, files) if job["kind"] == "m3_music" else None
            if plan is None and result is None:
                raise ValueError("Not a music job.")
        except (ValueError, KeyError, TypeError, zipfile.BadZipFile) as exc:
            raise ServiceError(422, "BGMの生成結果・ループ区間が入力と一致しません。") from exc
        with self.db.transaction() as connection:
            job, attempt = self.coordinator._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self.coordinator._duplicate(connection, job, data)
            replan = bool(payload.get("adjustment", {}).get("replan"))
            table = "music_replan_job" if replan else "music_adjustment_job"
            link = dict(connection.execute(f"SELECT * FROM {table} WHERE job_id=?", (job_id,)).fetchone())
            draft = required(connection, "adjustment_draft", link["draft_id"])
            root = required(connection, "m3_production", draft["production_id"])
            chapter = required(connection, "m3_production", link["production_id"])
            selection = self.history.selection(connection, job["project_id"])
            current = selection and selection["adjustment_draft_id"] == draft["id"] and selection["edition_id"] == draft["base_edition_id"]
            adopt_replan = bool(replan and current and draft["revision"] == link["adoption_revision"] and draft["status"] == "draft")
            adoption_reason = "採用済みの曲を使い、章のBGM・場面転換を再計画しました。" if adopt_replan else (
                "調整版が更新・反映されたため、この再計画結果は保持し、設定には上書きしていません。")
            bundle = self.m3._artifact(connection, chapter, "music-result-" + job_id, "m3_bundle", "result.zip", data,
                provenance={**envelope["provenance"], **({"replan_adopted": adopt_replan,
                    "replan_adoption_reason": adoption_reason} if replan else {})}, job=job, attempt=attempt)
            if replan:
                if adopt_replan:
                    self._adopt_replan(connection, draft, chapter, plan, job, attempt)
            elif plan:
                prompt = plan.scenes[0].prompt
                connection.execute("UPDATE music_candidate SET prompt=? WHERE id=?", (prompt, link["candidate_id"]))
                if current:
                    self._enqueue_music(connection, draft, root, chapter, link["candidate_id"], "m3_music",
                        {"scene_id": link["scene_id"], "prompt": prompt, "context": payload["context"],
                         "backend": "stable_audio3", "model": "medium", "duration_seconds": 120})
            else:
                store_candidate(self.m3, connection, chapter, result.model_dump(mode="json"), files,
                    candidate_id=link["candidate_id"], job=job, attempt=attempt, provenance=envelope["provenance"])
            now = self.clock()
            connection.execute("UPDATE job_attempt SET status='completed',ended_at=? WHERE id=?", (now, attempt["id"]))
            connection.execute("UPDATE job SET status='completed',result_artifact_id=?,updated_at=?,error=NULL WHERE id=?", (bundle["id"], now, job_id))
            connection.execute("UPDATE generation_run SET status='completed' WHERE id=?", (job["run_id"],))
            self.history.finish(connection, job["project_id"])
            return {"job": public(required(connection, "job", job_id)), "artifact": public(bundle)}

    def validate_retry(self, connection, job):
        replan = json.loads(job["payload"]).get("adjustment", {}).get("replan")
        table = "music_replan_job" if replan else "music_adjustment_job"
        link = dict(connection.execute(f"SELECT * FROM {table} WHERE job_id=?", (job["id"],)).fetchone())
        draft = self._draft(connection, job["project_id"])
        selection = self.history.selection(connection, job["project_id"])
        if not draft or draft["id"] != link["draft_id"] or selection["edition_id"] != draft["base_edition_id"]:
            raise ServiceError(409, "以前の調整BGMは再試行できません。")
        if replan and (draft["revision"] != link["adoption_revision"] or draft["status"] != "draft"):
            raise ServiceError(409, "調整版が変わったため、章のBGM再計画を新しく開始してください。")
        self.history.begin(connection, job["project_id"], "BGM生成を再試行", "m3-music-retry", concurrent=True)

    def retry(self, project_id, request):
        with self.db.transaction() as connection:
            draft, _, _ = self._editable(connection, project_id, request.expected_revision)
            jobs = [row[0] for row in connection.execute("SELECT j.id FROM job j WHERE j.status='failed' AND j.id IN "
                "(SELECT job_id FROM music_adjustment_job WHERE draft_id=? UNION SELECT job_id FROM music_replan_job WHERE draft_id=?)",
                (draft["id"], draft["id"]))]
        if not jobs:
            raise ServiceError(409, "失敗したBGM生成がありません。")
        for identifier in jobs:
            self.coordinator.retry(identifier)
        return self.project(project_id)
