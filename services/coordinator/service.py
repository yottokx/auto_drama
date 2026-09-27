"""Durable M1 projects, inputs, leases and atomic result adoption."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from packages.contracts import Script
from packages.tyrano_export import demo_content, validate_bundle

from .database import Database
from .storage import ArtifactStore, StoredObject


def encode_json(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode("utf-8")


class ServiceError(Exception):
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(detail)


def required(connection: sqlite3.Connection, table: str, identifier: str) -> dict:
    # table is exclusively supplied by our own call sites, never request data.
    row = connection.execute(f"SELECT * FROM {table} WHERE id = ?", (identifier,)).fetchone()
    if row is None:
        raise ServiceError(404, f"{table} が見つかりません。")
    return dict(row)


def public(record: dict) -> dict:
    result = dict(record)
    result.pop("storage_key", None)
    for key in ("payload", "settings_snapshot", "capabilities", "provenance", "policy"):
        if key in result:
            result[key] = json.loads(result[key])
    for key in (
        "created_at",
        "updated_at",
        "last_seen_at",
        "started_at",
        "ended_at",
        "lease_expires_at",
    ):
        if key in result and result[key] is not None:
            result[key] = datetime.fromtimestamp(result[key], UTC).isoformat()
    if "storage_key" in record:
        result["download_url"] = f"/api/artifacts/{record['id']}/content"
    return result


class Coordinator:
    def __init__(
        self, data_dir: Path, *, lease_seconds: float = 60, clock: Callable[[], float] = time.time
    ):
        if lease_seconds <= 0:
            raise ValueError("lease_seconds must be positive")
        self.db = Database(data_dir / "auto_drama.db")
        self.store = ArtifactStore(data_dir / "artifacts")
        self.lease_seconds = lease_seconds
        self.clock = clock

    def initialize(self) -> None:
        self.db.migrate()
        with self.db.transaction() as connection:
            self._recover(connection)
        from .m3_service import M3Service

        M3Service(self).recover()

    def _recover(self, connection: sqlite3.Connection) -> None:
        now = self.clock()
        attempts = connection.execute(
            "SELECT * FROM job_attempt WHERE status='running' AND lease_expires_at <= ?", (now,)
        ).fetchall()
        for attempt in attempts:
            error = "ワーカーの応答期限が切れました。"
            connection.execute(
                "UPDATE job_attempt SET status='expired', ended_at=?, error_kind='lease_expired', "
                "error=? WHERE id=?",
                (now, error, attempt["id"]),
            )
            self._release(connection, attempt["job_id"], error, now)

    @staticmethod
    def _release(connection: sqlite3.Connection, job_id: str, error: str, now: float) -> None:
        job = required(connection, "job", job_id)
        status = "failed" if job["attempt_count"] >= job["max_attempts"] else "pending"
        connection.execute(
            "UPDATE job SET status=?, error=?, updated_at=? WHERE id=?",
            (status, error, now, job_id),
        )
        connection.execute("UPDATE generation_run SET status=? WHERE id=?", (status, job["run_id"]))

    def projects(self) -> list[dict]:
        with self.db.transaction() as connection:
            return [
                public(dict(row))
                for row in connection.execute("SELECT * FROM project ORDER BY created_at DESC, id")
            ]

    def project(self, project_id: str) -> dict:
        with self.db.transaction() as connection:
            self._recover(connection)
            project = public(required(connection, "project", project_id))
            jobs = [
                public(dict(row))
                for row in connection.execute(
                    "SELECT * FROM job WHERE project_id=? ORDER BY created_at, id", (project_id,)
                )
            ]
            artifacts = [
                public(dict(row))
                for row in connection.execute(
                    "SELECT * FROM artifact WHERE project_id=? ORDER BY created_at, id",
                    (project_id,),
                )
            ]
        return {"project": project, "jobs": jobs, "artifacts": artifacts}

    def _create_project(
        self, connection: sqlite3.Connection, title: str, instructions: str, chapter_count: int
    ) -> str:
        identifier = uuid4().hex
        connection.execute(
            "INSERT INTO project (id,title,instructions,chapter_count,created_at) VALUES (?,?,?,?,?)",
            (identifier, title, instructions, chapter_count, self.clock()),
        )
        return identifier

    def create_project(self, title: str, instructions: str = "", chapter_count: int = 1) -> dict:
        with self.db.transaction() as connection:
            identifier = self._create_project(connection, title, instructions, chapter_count)
            return public(required(connection, "project", identifier))

    def _register_artifact(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        logical_id: str,
        kind: str,
        filename: str,
        media_type: str,
        stored: StoredObject,
        *,
        source_job_id: str | None = None,
        source_attempt_id: str | None = None,
    ) -> dict:
        identifier = uuid4().hex
        version = connection.execute(
            "SELECT COALESCE(MAX(version),0)+1 FROM artifact WHERE project_id=? AND logical_id=?",
            (project_id, logical_id),
        ).fetchone()[0]
        connection.execute(
            "INSERT INTO artifact (id,project_id,logical_id,version,kind,filename,media_type,"
            "sha256,size_bytes,storage_key,source_job_id,source_attempt_id,provenance,created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (
                identifier,
                project_id,
                logical_id,
                version,
                kind,
                filename,
                media_type,
                stored.sha256,
                stored.size_bytes,
                stored.storage_key,
                source_job_id,
                source_attempt_id,
                json.dumps({"producer": "tyrano_export/1" if source_job_id else "m1_input/1"}),
                self.clock(),
            ),
        )
        return required(connection, "artifact", identifier)

    def create_demo(self, title: str) -> dict:
        script, assets = demo_content()
        document = script.model_dump(mode="json")
        document["title"] = title
        stored_assets = {key: self.store.put(value) for key, value in assets.items()}
        with self.db.transaction() as connection:
            project_id = self._create_project(connection, title, "M1 固定サンプル", 1)
            for asset in document["assets"]:
                media_type = "audio/wav" if asset["kind"] == "audio" else "image/png"
                record = self._register_artifact(
                    connection,
                    project_id,
                    asset["id"],
                    asset["kind"],
                    asset["filename"],
                    media_type,
                    stored_assets[asset["id"]],
                )
                asset["artifact_id"] = record["id"]
            script = Script.model_validate(document)
            stored = self.store.put(encode_json(script.model_dump(mode="json")))
            source = self._register_artifact(
                connection,
                project_id,
                script.id,
                "script",
                "script.json",
                "application/json",
                stored,
            )
            job = self._enqueue(connection, project_id, source["id"], [], 0, 3)
            return {"project": public(required(connection, "project", project_id)), "job": job}

    def _script_assets(
        self, connection: sqlite3.Connection, project_id: str, script: Script
    ) -> dict[str, bytes]:
        assets = {}
        for reference in script.assets:
            record = required(connection, "artifact", reference.artifact_id)
            if (
                record["project_id"] != project_id
                or record["sha256"] != reference.sha256
                or record["kind"] != reference.kind
            ):
                raise ServiceError(422, "脚本の素材参照が作品・種類・ハッシュと一致しません。")
            assets[reference.id] = self.store.read(record)
        return assets

    def save_script(self, project_id: str, script: Script) -> dict:
        with self.db.transaction() as connection:
            required(connection, "project", project_id)
            self._script_assets(connection, project_id, script)
            stored = self.store.put(encode_json(script.model_dump(mode="json")))
            record = self._register_artifact(
                connection,
                project_id,
                script.id,
                "script",
                "script.json",
                "application/json",
                stored,
            )
            return public(record)

    def _load_input(self, connection: sqlite3.Connection, job: dict) -> tuple[Script, dict]:
        identifier = json.loads(job["payload"])["script_artifact_id"]
        record = required(connection, "artifact", identifier)
        if record["kind"] != "script" or record["project_id"] != job["project_id"]:
            raise ServiceError(422, "対象作品の脚本を指定してください。")
        script = Script.model_validate_json(self.store.read(record))
        return script, self._script_assets(connection, job["project_id"], script)

    def _enqueue(
        self,
        connection: sqlite3.Connection,
        project_id: str,
        script_artifact_id: str,
        dependencies: list[str],
        priority: int,
        max_attempts: int,
    ) -> dict:
        project = required(connection, "project", project_id)
        source = required(connection, "artifact", script_artifact_id)
        if source["project_id"] != project_id or source["kind"] != "script":
            raise ServiceError(422, "対象作品の脚本を指定してください。")
        for dependency in dependencies:
            if required(connection, "job", dependency)["project_id"] != project_id:
                raise ServiceError(422, "依存ジョブは同じ作品を指定してください。")
        identifier, run_id = uuid4().hex, uuid4().hex
        connection.execute(
            "INSERT INTO generation_run (id,project_id,settings_version,story_revision_id,"
            "policy,status,created_at) VALUES (?,?,?,?,?,'pending',?)",
            (
                run_id,
                project_id,
                project["settings_version"],
                script_artifact_id,
                json.dumps({"mode": "fixed_script", "max_attempts": max_attempts}),
                self.clock(),
            ),
        )
        connection.execute(
            "INSERT INTO job (id,project_id,run_id,kind,payload,settings_snapshot,priority,status,"
            "max_attempts,created_at,updated_at) VALUES (?,?,?,'tyrano_export',?,?,?,'pending',?,?,?)",
            (
                identifier,
                project_id,
                run_id,
                json.dumps({"script_artifact_id": script_artifact_id}),
                json.dumps(
                    {
                        "schema_version": 1,
                        "settings_version": project["settings_version"],
                        "exporter_version": 1,
                    }
                ),
                priority,
                max_attempts,
                self.clock(),
                self.clock(),
            ),
        )
        for dependency in dependencies:
            connection.execute("INSERT INTO job_dependency VALUES (?,?)", (identifier, dependency))
        return public(required(connection, "job", identifier))

    def enqueue(
        self,
        project_id: str,
        script_artifact_id: str,
        dependencies: list[str],
        priority: int,
        max_attempts: int,
    ) -> dict:
        with self.db.transaction() as connection:
            return self._enqueue(
                connection, project_id, script_artifact_id, dependencies, priority, max_attempts
            )

    def job(self, identifier: str) -> dict:
        with self.db.transaction() as connection:
            self._recover(connection)
            job = public(required(connection, "job", identifier))
            attempts = [
                public(dict(row))
                for row in connection.execute(
                    "SELECT * FROM job_attempt WHERE job_id=? ORDER BY attempt", (identifier,)
                )
            ]
            dependencies = [
                row[0]
                for row in connection.execute(
                    "SELECT depends_on_id FROM job_dependency WHERE job_id=? ORDER BY depends_on_id",
                    (identifier,),
                )
            ]
        return {"job": job, "attempts": attempts, "dependencies": dependencies}

    def retry(self, identifier: str) -> dict:
        with self.db.transaction() as connection:
            self._recover(connection)
            job = required(connection, "job", identifier)
            if job["status"] != "failed":
                raise ServiceError(409, "失敗したジョブだけ再試行できます。")
            if job["kind"].startswith("m2_"):
                from .m2_service import M2Service

                M2Service(self).validate_retry(connection, job)
                from .project_history import HistoryService

                history = HistoryService(self)
                if not history.has_operation(connection, job["project_id"], "m2"):
                    history.begin(connection, job["project_id"], "生成を再試行", "m2-retry")
            elif job["kind"].startswith("m3_"):
                from .m3_service import M3Service

                M3Service(self).validate_retry(connection, job)
            # An explicit user retry grants one additional attempt and preserves history.
            connection.execute(
                "UPDATE job SET status='pending',max_attempts=attempt_count+1,error=NULL,"
                "updated_at=? WHERE id=?",
                (self.clock(), identifier),
            )
            connection.execute(
                "UPDATE generation_run SET status='pending' WHERE id=?", (job["run_id"],)
            )
            return public(required(connection, "job", identifier))

    def register_worker(self, name: str, capabilities: list[str]) -> dict:
        with self.db.transaction() as connection:
            identifier = uuid4().hex
            connection.execute(
                "INSERT INTO worker (id,name,capabilities,last_seen_at) VALUES (?,?,?,?)",
                (identifier, name, json.dumps(sorted(set(capabilities))), self.clock()),
            )
            return public(required(connection, "worker", identifier))

    def workers(self) -> list[dict]:
        with self.db.transaction() as connection:
            return [
                public(dict(row))
                for row in connection.execute("SELECT * FROM worker ORDER BY last_seen_at DESC, id")
            ]

    def claim(self, worker_id: str) -> dict | None:
        with self.db.transaction() as connection:
            self._recover(connection)
            worker = required(connection, "worker", worker_id)
            now = self.clock()
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            # Repeating a lost claim waits for expiry, avoiding two active jobs
            # with one worker identity.
            if connection.execute(
                "SELECT 1 FROM job_attempt WHERE worker_id=? AND status='running'", (worker_id,)
            ).fetchone():
                return None
            capabilities = json.loads(worker["capabilities"])
            candidates = connection.execute(
                "SELECT * FROM job WHERE status='pending' AND NOT EXISTS "
                "(SELECT 1 FROM job_dependency d JOIN job parent ON parent.id=d.depends_on_id "
                "WHERE d.job_id=job.id AND parent.status!='completed') "
                "AND NOT EXISTS (SELECT 1 FROM m3_production_job pj "
                "JOIN m3_production p ON p.id=pj.production_id "
                "JOIN m3_production root ON root.id=p.storyline_id "
                "LEFT JOIN project_history_state h ON h.project_id=p.project_id "
                "WHERE pj.job_id=job.id AND (root.control_state!='running' "
                "OR h.production_frozen=1 OR h.production_id!=root.id)) "
                "ORDER BY priority DESC, created_at, id"
            )
            job = next((dict(row) for row in candidates if row["kind"] in capabilities), None)
            if job is None:
                return None
            attempt = job["attempt_count"] + 1
            lease_id = uuid4().hex
            expires = now + self.lease_seconds
            connection.execute(
                "INSERT INTO job_attempt (id,job_id,worker_id,attempt,lease_id,lease_expires_at,"
                "status,started_at) VALUES (?,?,?,?,?,?,'running',?)",
                (uuid4().hex, job["id"], worker_id, attempt, lease_id, expires, now),
            )
            connection.execute(
                "UPDATE job SET status='running',attempt_count=?,error=NULL,updated_at=? WHERE id=?",
                (attempt, now, job["id"]),
            )
            connection.execute(
                "UPDATE generation_run SET status='running' WHERE id=?", (job["run_id"],)
            )
            result = public(required(connection, "job", job["id"]))
            result.update(
                public({"lease_id": lease_id, "lease_expires_at": expires, "attempt": attempt})
            )
            return result

    def _lease(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        worker_id: str,
        lease_id: str,
        *,
        completed_ok: bool = False,
    ) -> tuple[dict, dict]:
        job = required(connection, "job", job_id)
        row = connection.execute(
            "SELECT * FROM job_attempt WHERE job_id=? AND worker_id=? AND lease_id=?",
            (job_id, worker_id, lease_id),
        ).fetchone()
        if row is None:
            raise ServiceError(409, "この試行は有効ではありません。")
        attempt = dict(row)
        if completed_ok and attempt["status"] == job["status"] == "completed":
            return job, attempt
        if (
            job["status"] != "running"
            or attempt["status"] != "running"
            or attempt["lease_expires_at"] <= self.clock()
        ):
            raise ServiceError(409, "リースが期限切れか、すでに終了した試行です。")
        return job, attempt

    def heartbeat(self, job_id: str, worker_id: str, lease_id: str) -> dict:
        with self.db.transaction() as connection:
            _, attempt = self._lease(connection, job_id, worker_id, lease_id)
            now = self.clock()
            expires = now + self.lease_seconds
            connection.execute(
                "UPDATE job_attempt SET lease_expires_at=? WHERE id=?", (expires, attempt["id"])
            )
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            return public({"lease_expires_at": expires})

    def fail(self, job_id: str, worker_id: str, lease_id: str, error: str) -> dict:
        with self.db.transaction() as connection:
            _, attempt = self._lease(connection, job_id, worker_id, lease_id)
            now = self.clock()
            connection.execute(
                "UPDATE job_attempt SET status='failed',ended_at=?,error_kind='worker_error',"
                "error=? WHERE id=?",
                (now, error, attempt["id"]),
            )
            self._release(connection, job_id, error, now)
            from .project_history import HistoryService

            job = required(connection, "job", job_id)
            HistoryService(self).finish(connection, job["project_id"])
            return public(required(connection, "job", job_id))

    def complete(self, job_id: str, worker_id: str, lease_id: str, data: bytes) -> dict:
        # Validate outside the write transaction. Adoption rechecks the lease
        # after file work so expiry during compilation cannot publish a stale result.
        with self.db.transaction() as connection:
            job, attempt = self._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["kind"] != "tyrano_export":
                raise ServiceError(422, "この完了APIはTyrano出力ジョブ用です。M2生成はM2の完了APIを使用してください。")
            if job["status"] == "completed":
                return self._duplicate(connection, job, data)
            script, assets = self._load_input(connection, job)
        try:
            validate_bundle(data, script, assets)
        except ValueError as exc:
            raise ServiceError(
                422, "出力が入力脚本・素材から再現される完成ZIPと一致しません。"
            ) from exc
        stored = self.store.put(data)
        with self.db.transaction() as connection:
            job, attempt = self._lease(connection, job_id, worker_id, lease_id, completed_ok=True)
            if job["status"] == "completed":
                return self._duplicate(connection, job, data)
            artifact = self._register_artifact(
                connection,
                job["project_id"],
                f"export-{job_id}",
                "tyrano_export",
                "tyrano-source.zip",
                "application/zip",
                stored,
                source_job_id=job_id,
                source_attempt_id=attempt["id"],
            )
            now = self.clock()
            connection.execute(
                "UPDATE job_attempt SET status='completed',ended_at=? WHERE id=?",
                (now, attempt["id"]),
            )
            connection.execute(
                "UPDATE job SET status='completed',result_artifact_id=?,updated_at=?,error=NULL "
                "WHERE id=?",
                (artifact["id"], now, job_id),
            )
            connection.execute(
                "UPDATE generation_run SET status='completed' WHERE id=?", (job["run_id"],)
            )
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            return {
                "job": public(required(connection, "job", job_id)),
                "artifact": public(artifact),
            }

    def _duplicate(self, connection: sqlite3.Connection, job: dict, data: bytes) -> dict:
        artifact = required(connection, "artifact", job["result_artifact_id"])
        if artifact["sha256"] != hashlib.sha256(data).hexdigest():
            raise ServiceError(409, "採用済みの結果と異なる内容は受け付けません。")
        self.store.read(artifact)
        return {"job": public(job), "artifact": public(artifact)}

    def artifact(self, identifier: str) -> tuple[dict, bytes]:
        with self.db.transaction() as connection:
            record = required(connection, "artifact", identifier)
        return public(record), self.store.read(record)
