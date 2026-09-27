"""Chapter continuity, durable production controls and immutable viewing boundaries."""

from __future__ import annotations

import io
import json
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

from .service import ServiceError, encode_json, public, required


class ChapterProduction:
    def _root(self, connection, production):
        return required(connection, "m3_production", production["storyline_id"])

    def _previous(self, connection, production):
        row = connection.execute(
            "SELECT * FROM m3_production WHERE storyline_id=? AND chapter_number=?",
            (production["storyline_id"], production["chapter_number"] - 1),
        ).fetchone()
        return dict(row) if row else None

    def _load_narrative(self, connection, production):
        from packages.narrative.validation import validate_narrative

        value = json.loads(self.store.read(required(
            connection, "artifact", production["narrative_artifact_id"])))
        previous = None
        if production["previous_narrative_artifact_id"]:
            previous = json.loads(self.store.read(required(
                connection, "artifact", production["previous_narrative_artifact_id"])))
        return validate_narrative(value, self._snapshot(connection, production), previous,
                                  expected_previous_artifact_id=production["previous_narrative_artifact_id"])

    def _next_chapter(self, connection, production):
        """Commit the next writer once its exact predecessor has been adopted."""
        from packages.narrative.validation import story_state_hash

        root = self._root(connection, production)
        if not root["m4_enabled"] or root["control_state"] != "running":
            return
        count = self._snapshot(connection, root)["world"]["result"]["chapterCount"]
        number = production["chapter_number"] + 1
        if not production["narrative_artifact_id"] or number > count:
            return
        if connection.execute(
            "SELECT 1 FROM m3_production WHERE storyline_id=? AND chapter_number=?",
            (root["id"], number),
        ).fetchone():
            return
        narrative = self._load_narrative(connection, production)
        state_hash = story_state_hash(narrative.end_state) if narrative.end_state else None
        continuation = {}
        if narrative.workflow_policy == "script_continuation_v1":
            record = required(connection, "artifact", production["narrative_artifact_id"])
            checkpoint = json.loads(record["provenance"]).get("script_checkpoint")
            if not isinstance(checkpoint, dict) or not checkpoint:
                raise ValueError("The adopted script chapter has no continuation checkpoint.")
            continuation["script_checkpoint"] = checkpoint
        identifier = uuid4().hex
        connection.execute(
            "INSERT INTO m3_production (id,project_id,approval_id,approval_artifact_id,created_at,"
            "storyline_id,chapter_number,previous_narrative_artifact_id,previous_state_hash,m4_enabled) "
            "VALUES (?,?,?,?,?,?,?,?,?,1)",
            (identifier, root["project_id"], root["approval_id"], root["approval_artifact_id"],
             self.clock(), root["id"], number, production["narrative_artifact_id"], state_hash),
        )
        child = required(connection, "m3_production", identifier)
        self._enqueue(connection, child, "m3_narrative", {
            "previous_narrative": narrative.model_dump(mode="json"),
            "previous_narrative_artifact_id": production["narrative_artifact_id"],
            "previous_state_hash": state_hash,
            **continuation,
        })

    def _settle_control(self, connection, root):
        if root["control_state"] == "stopping" and not connection.execute(
            "SELECT 1 FROM job j JOIN m3_production_job pj ON pj.job_id=j.id "
            "JOIN m3_production p ON p.id=pj.production_id "
            "WHERE p.storyline_id=? AND j.status='running' LIMIT 1", (root["id"],),
        ).fetchone():
            connection.execute("UPDATE m3_production SET control_state='paused' WHERE id=?",
                               (root["id"],))

    def stop(self, project_id, mode):
        with self.db.transaction() as connection:
            self.coordinator._recover(connection)
            root = self._selected_production(connection, project_id)
            if root is None:
                raise ServiceError(409, "制作が開始されていません。")
            state = "interrupted" if mode == "immediate" else "stopping"
            connection.execute("UPDATE m3_production SET control_state=? WHERE id=?",
                               (state, root["id"]))
            if mode == "immediate":
                jobs = connection.execute(
                    "SELECT j.* FROM job j JOIN m3_production_job pj ON pj.job_id=j.id "
                    "JOIN m3_production p ON p.id=pj.production_id "
                    "WHERE p.storyline_id=? AND j.status='running'", (root["id"],),
                ).fetchall()
                for job in jobs:
                    connection.execute(
                        "UPDATE job_attempt SET status='expired',ended_at=?,error_kind='interrupted',"
                        "error='制作を即時中断しました。' WHERE job_id=? AND status='running'",
                        (self.clock(), job["id"]),
                    )
                    # A user interruption does not consume the automatic retry allowance.
                    connection.execute(
                        "UPDATE job SET status='pending',max_attempts=max_attempts+1,"
                        "error=NULL,updated_at=? WHERE id=?", (self.clock(), job["id"]),
                    )
                    connection.execute("UPDATE generation_run SET status='pending' WHERE id=?",
                                       (job["run_id"],))
            self._settle_control(connection, {**dict(root), "control_state": state})
        return self.project(project_id)

    def resume(self, project_id):
        with self.db.transaction() as connection:
            root = self._selected_production(connection, project_id)
            if root is None:
                raise ServiceError(409, "制作が開始されていません。")
            self._assert_resumable(connection, root)
            if not self.history.has_operation(connection, project_id, "m3"):
                self.history.begin(connection, project_id, "本編制作を再開", "m3-resume")
            selected = self.history.selection(connection, project_id)
            self.history.set_selection(connection, project_id, root["id"],
                                       selected["build_id"], selected["portrait_settings_id"])
            connection.execute(
                "UPDATE m3_production SET control_state='running',m4_enabled=1 WHERE id=?",
                (root["id"],),
            )
            connection.execute("UPDATE m3_production SET error=NULL WHERE storyline_id=?",
                               (root["id"],))
        self.advance(root["id"])
        return self.project(project_id)

    def _chapters(self, connection, root):
        selection = self.history.selection(connection, root["project_id"])
        frozen = (json.loads(selection["production_snapshot"])
                  if selection and selection["production_frozen"]
                  and selection["production_snapshot"] else None)
        saved = {item["id"]: item for item in frozen.get("chapters", [])} if frozen else None
        result = []
        for row in connection.execute(
            "SELECT * FROM m3_production WHERE storyline_id=? ORDER BY chapter_number", (root["id"],)
        ):
            if frozen and row["id"] != root["id"] and row["id"] not in saved:
                continue
            chapter = dict(row)
            if saved and chapter["id"] in saved:
                chapter.update({key: saved[chapter["id"]][key]
                                for key in ("narrative_artifact_id", "error")})
            elif frozen and chapter["id"] == root["id"]:
                chapter.update({key: frozen[key] for key in ("narrative_artifact_id", "error")})
            jobs = self._jobs(connection, chapter["id"])
            build = self._selected_build(connection, chapter)
            failed = next((job for job in jobs if job["status"] == "failed"), None)
            requirements = [public(dict(r)) for r in connection.execute(
                "SELECT * FROM m3_requirement WHERE production_id=? ORDER BY rowid", (chapter["id"],))]
            if frozen:
                state = saved.get(chapter["id"], frozen if chapter["id"] == root["id"] else {})
                requirements = state.get("requirements", [])
            result.append({
                "chapter_number": chapter["chapter_number"], "production_id": chapter["id"],
                "narrative_artifact_id": chapter["narrative_artifact_id"],
                "status": "published" if build else "failed" if failed or chapter["error"] else
                    "generating_assets" if chapter["narrative_artifact_id"] else "writing",
                "error": chapter["error"] or (failed["error"] if failed else None),
                "jobs": jobs, "requirements": requirements,
                "build": self._public_build(build) if build else None,
                "player_url": f"/player/{build['id']}/" if build else None,
                "export_url": f"/api/artifacts/{build['export_artifact_id']}/content" if build else None,
            })
        count = self._snapshot(connection, root)["world"]["result"]["chapterCount"]
        existing = {item["chapter_number"] for item in result}
        result.extend({"chapter_number": n, "production_id": None, "narrative_artifact_id": None,
                       "status": "waiting", "error": None, "jobs": [], "requirements": [],
                       "build": None, "player_url": None, "export_url": None}
                      for n in range(1, count + 1) if n not in existing)
        return sorted(result, key=lambda item: item["chapter_number"])

    def _build_lineage(self, connection, build):
        """Published validation pins narrative identity; never consult mutable draft text."""
        validation = json.loads(build["validation"])
        if validation.get("narrative_artifact_id"):
            return validation
        # Legacy builds predate explicit lineage. Their ZIP contains the adopted text.
        from packages.contracts.m3 import NarrativeResult
        record = required(connection, "artifact", build["export_artifact_id"])
        with ZipFile(io.BytesIO(self.store.read(record))) as archive:
            narrative = NarrativeResult.model_validate_json(archive.read("narrative.json"))
        for item in connection.execute(
            "SELECT * FROM artifact WHERE project_id=? AND logical_id=? ORDER BY version",
            (build["project_id"], "narrative-" + build["production_id"]),
        ):
            candidate = NarrativeResult.model_validate_json(self.store.read(dict(item)))
            if candidate == narrative:
                return {"narrative_artifact_id": item["id"], "end_state_hash": None}
        raise ServiceError(422, "公開版の本文の参照を確認できません。")

    def _next_build(self, connection, build):
        production = required(connection, "m3_production", build["production_id"])
        lineage = self._build_lineage(connection, build)
        for candidate in connection.execute(
            "SELECT b.* FROM chapter_build b JOIN m3_production p ON p.id=b.production_id "
            "WHERE p.storyline_id=? AND b.chapter_number=? ORDER BY b.revision DESC",
            (production["storyline_id"], build["chapter_number"] + 1),
        ):
            child = json.loads(candidate["validation"])
            if child.get("previous_narrative_artifact_id") != lineage["narrative_artifact_id"]:
                continue
            if (lineage.get("end_state_hash") is not None
                    and child.get("previous_state_hash") != lineage["end_state_hash"]):
                continue
            return candidate
        return None

    def next_build(self, build_id):
        with self.db.transaction() as connection:
            build = required(connection, "chapter_build", build_id)
            production = required(connection, "m3_production", build["production_id"])
            count = self._snapshot(connection, production)["world"]["result"]["chapterCount"]
            next_build = self._next_build(connection, build) if build["chapter_number"] < count else None
            # Only chapter number and URLs: never reveal unread titles, plot or traces.
            return {"build_id": build_id, "status": "ready" if next_build else
                    "complete" if build["chapter_number"] >= count else "waiting",
                    "next_build": {"id": next_build["id"],
                                   "chapter_number": next_build["chapter_number"],
                                   "player_url": f"/player/{next_build['id']}/",
                                   "export_url": f"/api/artifacts/{next_build['export_artifact_id']}/content"}
                    if next_build else None}

    def export_chapters(self, build_id):
        """Snapshot a consistent published prefix as independent playable chapter sources."""
        with self.db.transaction() as connection:
            build = required(connection, "chapter_build", build_id)
            builds = [build]
            while following := self._next_build(connection, builds[-1]):
                builds.append(following)
            output = io.BytesIO()
            with ZipFile(output, "w", ZIP_DEFLATED) as archive:
                entries = []
                for chapter in builds:
                    record = required(connection, "artifact", chapter["export_artifact_id"])
                    name = f"chapter-{chapter['chapter_number']:03d}.zip"
                    archive.writestr(name, self.store.read(record))
                    entries.append({"chapter_number": chapter["chapter_number"],
                                    "build_id": chapter["id"], "filename": name,
                                    "sha256": record["sha256"]})
                archive.writestr("chapters.json", encode_json({
                    "schema_version": 1, "chapters": entries,
                    "terminal": "書き出し時点で揃った章はここまでです。",
                }))
                archive.writestr("README.txt", "各chapter ZIPは章ごとのティラノ用ソースです。\n"
                                 "chapters.jsonの順番で各章を配置・再生してください。\n"
                                 "この書き出しに含まれる章は一覧の末尾までです。\n")
            return output.getvalue()
