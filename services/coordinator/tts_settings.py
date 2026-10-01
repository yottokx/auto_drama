"""Durable TTS settings, immutable generation selections and leased downloads."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from uuid import uuid4

from fastapi import FastAPI

from packages.contracts.tts_settings import (
    DownloadInput,
    DownloadProgress,
    DownloadResume,
    TTSInventory,
    TTSSettings,
)

from .service import ServiceError, required

ACTIVE = ("downloading", "verifying", "cancelling")
RESUMABLE = ("interrupted", "failed", "cancelled")
VOICE_PURPOSES = {
    "m2_voice": "voice_design", "m3_voice": "voice_design",
    "m2_voice_clone": "voice_clone", "m3_voice_clone": "voice_clone",
}


def generation_purposes(capabilities: list[str]) -> list[str]:
    return sorted({VOICE_PURPOSES[kind] for kind in capabilities if kind in VOICE_PURPOSES})


def _catalog():
    from packages.contracts.tts_catalog import catalog

    return catalog()


def _bundle(model_id: str, precision: str) -> dict:
    from packages.contracts.tts_catalog import resolve_bundle

    try:
        return resolve_bundle(model_id, precision)
    except (KeyError, ValueError) as exc:
        raise ServiceError(422, "指定したTTSモデルまたは精度は取得できません。") from exc


def _public_operation(row: dict) -> dict:
    result = {key: row[key] for key in (
        "id", "worker_id", "model_id", "precision", "manifest_id", "status", "phase",
        "done_bytes", "total_bytes", "current_file", "error", "cancel_requested",
        "created_at", "updated_at",
    )}
    result["cancel_requested"] = bool(result["cancel_requested"])
    result["file_download_ready"] = row["status"] == "completed"
    result["generation_ready"] = False
    for key in ("created_at", "updated_at"):
        result[key] = datetime.fromtimestamp(result[key], UTC).isoformat()
    return result


def _safe_file(value: str | None, manifest: dict) -> str | None:
    # Reports must never reveal the Worker's absolute storage path.
    if not value:
        return None
    name = value.replace("\\", "/").rsplit("/", 1)[-1]
    known = {item["path"].rsplit("/", 1)[-1] for item in manifest["files"]}
    return name if name in known else None


def _safe_error(value: str | None) -> str | None:
    if not value:
        return None
    # Exception messages may contain token-bearing URLs or private paths.
    lower = value.lower()
    if "network access blocked" in lower or "winerror 10013" in lower:
        return "このPCの実行環境で外部通信が拒否されています。通常のターミナルからアプリの実行処理を起動してください。"
    if "local storage permission denied" in lower:
        return "モデル保存先への書き込み権限がありません。保存先のフォルダーを確認してください。"
    if "tls verification failed" in lower:
        return "モデル配布元の証明書を検証できません。このPCの証明書設定を確認してください。"
    if "timed out" in lower:
        return "モデル配布元との通信がタイムアウトしました。再開すると部分ファイルから続行します。"
    if "network connection failed" in lower:
        return "モデル配布元に接続できません。このPCの通信環境を確認してください。"
    if "http 404" in lower:
        return "指定されたモデルファイルが配布元にありません。モデル一覧の固定版を確認してください。"
    if any(term in lower for term in ("disk", "space", "enospc", "空き容量")):
        return "保存先の空き容量が不足しています。"
    if any(term in lower for term in ("hash", "checksum", "sha256", "integrity", "検証")):
        return "ダウンロードしたファイルの整合性検査に失敗しました。"
    if any(term in lower for term in ("401", "403", "permission", "gated", "認証")):
        return "モデル配布元へのアクセス権を確認してください。"
    return "ダウンロードに失敗しました。アプリの実行ログを確認してください。"


class TTSService:
    def __init__(self, coordinator):
        self.coordinator = coordinator

    @property
    def db(self):
        return self.coordinator.db

    @staticmethod
    def settings(connection) -> TTSSettings:
        row = connection.execute("SELECT value FROM tts_settings WHERE id=1").fetchone()
        if not row:
            return TTSSettings()
        # Existing draft choices become active without changing either selection.
        value = json.loads(row["value"])
        value["generation_active"] = True
        return TTSSettings.model_validate(value)

    def read_settings(self) -> dict:
        connection = self.db.connect()
        try:
            settings = self.settings(connection)
        finally:
            connection.close()
        return {"settings": settings.model_dump(), "stage": "active", "generation_active": True}

    def profile(self, connection) -> dict:
        from packages.contracts.tts_profile import build_tts_profile

        settings = self.settings(connection)
        return build_tts_profile(settings.model_dump())

    @staticmethod
    def supports(connection, worker: dict, kind: str, profile: dict | None) -> bool:
        purpose = VOICE_PURPOSES.get(kind)
        if purpose is None or profile is None:
            return True
        selection = profile.get(purpose, {})
        row = connection.execute(
            "SELECT 1 FROM tts_worker_inventory WHERE worker_id=? AND model_id=? "
            "AND precision=? AND manifest_id=? AND file_download_ready=1 AND generation_ready=1 "
            "AND EXISTS(SELECT 1 FROM json_each(generation_purposes) WHERE value=?)",
            (worker["id"], selection.get("model_id"), selection.get("precision"),
             selection.get("manifest_id"), purpose),
        ).fetchone()
        return bool(row)

    def save_settings(self, settings: TTSSettings) -> dict:
        with self.db.transaction() as connection:
            for purpose, kind in (("voice_design", "m2_voice"), ("voice_clone", "m2_voice_clone")):
                selection = getattr(settings, purpose)
                profile = {purpose: {**selection.model_dump(), "manifest_id": _bundle(
                    selection.model_id, selection.precision,
                )["manifest_id"]}}
                if not any(
                    purpose in generation_purposes(json.loads(row["capabilities"]))
                    and self.supports(connection, row, kind, profile)
                    for row in connection.execute("SELECT * FROM worker WHERE last_seen_at>=?",
                                                  (self.coordinator.clock() - 60,))
                ):
                    label = "ボイスデザイン" if purpose == "voice_design" else "ボイスクローン"
                    raise ServiceError(422, f"{label}のモデルが接続中の実行環境に取得されていません。")
            connection.execute(
                "INSERT INTO tts_settings(id,value) VALUES(1,?) "
                "ON CONFLICT(id) DO UPDATE SET value=excluded.value",
                (settings.model_dump_json(),),
            )
        return self.read_settings()

    def models(self) -> dict:
        return {"models": _catalog(), "generation_active": True}

    def _recover(self, connection) -> None:
        now = self.coordinator.clock()
        connection.execute(
            "UPDATE tts_download SET status=CASE WHEN cancel_requested=1 THEN 'cancelled' "
            "ELSE 'interrupted' END,phase=NULL,lease_id=NULL,lease_expires_at=NULL,updated_at=?, "
            "error=CASE WHEN cancel_requested=1 THEN NULL ELSE ? END "
            "WHERE status IN ('downloading','verifying','cancelling') AND lease_expires_at<=?",
            (now, "Workerの応答期限が切れました。再開してください。", now),
        )

    def _worker(self, connection, worker_id: str, *, require_online: bool = True) -> dict:
        worker = required(connection, "worker", worker_id)
        if "tts_download" not in json.loads(worker["capabilities"]):
            raise ServiceError(409, "このWorkerはTTSモデルの取得に対応していません。")
        if require_online and worker["last_seen_at"] < self.coordinator.clock() - 60:
            raise ServiceError(409, "取得先Workerが接続されていません。")
        return worker

    def downloads(self) -> dict:
        with self.db.transaction() as connection:
            self._recover(connection)
            workers = []
            for row in connection.execute("SELECT * FROM worker ORDER BY last_seen_at DESC,id"):
                purposes = generation_purposes(json.loads(row["capabilities"]))
                inventory = [
                    {
                        "model_id": item["model_id"], "precision": item["precision"],
                        "manifest_id": item["manifest_id"],
                        "file_download_ready": bool(item["file_download_ready"]),
                        "generation_ready": bool(item["file_download_ready"])
                            and bool(item["generation_ready"])
                            and bool(set(json.loads(item["generation_purposes"])) & set(purposes)),
                        "generation_purposes": sorted(set(json.loads(item["generation_purposes"])) & set(purposes)),
                    }
                    for item in connection.execute(
                        "SELECT * FROM tts_worker_inventory WHERE worker_id=? "
                        "ORDER BY model_id,precision,manifest_id", (row["id"],),
                    )
                ]
                workers.append({
                    "id": row["id"], "name": row["name"],
                    "online": row["last_seen_at"] >= self.coordinator.clock() - 60,
                    "download_capable": "tts_download" in json.loads(row["capabilities"]),
                    "generation_purposes": purposes,
                    "inventory": inventory,
                })
            operations = [
                _public_operation(dict(row)) for row in connection.execute(
                    "SELECT * FROM tts_download ORDER BY created_at DESC,id"
                )
            ]
        return {"workers": workers, "operations": operations, "generation_active": True}

    def enqueue(self, worker_id: str, body: DownloadInput) -> dict:
        manifest = _bundle(body.model_id, body.precision)
        with self.db.transaction() as connection:
            self._recover(connection)
            self._worker(connection, worker_id)
            existing = connection.execute(
                "SELECT * FROM tts_download WHERE worker_id=? AND manifest_id=? "
                "AND status IN ('queued','downloading','verifying','cancelling') "
                "ORDER BY created_at LIMIT 1", (worker_id, manifest["manifest_id"]),
            ).fetchone()
            if existing:
                return _public_operation(dict(existing))
            now = self.coordinator.clock()
            identifier = uuid4().hex
            connection.execute(
                "INSERT INTO tts_download(id,worker_id,model_id,precision,manifest_id,manifest,"
                "status,total_bytes,created_at,updated_at) VALUES(?,?,?,?,?,?,'queued',?,?,?)",
                (identifier, worker_id, body.model_id, body.precision, manifest["manifest_id"],
                 json.dumps(manifest, ensure_ascii=False, sort_keys=True),
                 manifest["total_bytes"], now, now),
            )
            return _public_operation(required(connection, "tts_download", identifier))

    def cancel(self, operation_id: str) -> dict:
        with self.db.transaction() as connection:
            self._recover(connection)
            operation = required(connection, "tts_download", operation_id)
            if operation["status"] == "completed":
                raise ServiceError(409, "完了済みの取得は中断できません。")
            if operation["status"] != "cancelled":
                status = "cancelling" if operation["status"] in ACTIVE else "cancelled"
                connection.execute(
                    "UPDATE tts_download SET status=?,cancel_requested=1,updated_at=? WHERE id=?",
                    (status, self.coordinator.clock(), operation_id),
                )
            return _public_operation(required(connection, "tts_download", operation_id))

    def resume(self, operation_id: str, worker_id: str | None = None) -> dict:
        with self.db.transaction() as connection:
            self._recover(connection)
            operation = required(connection, "tts_download", operation_id)
            if operation["status"] not in RESUMABLE:
                raise ServiceError(409, "この取得操作は再開できません。")
            target = worker_id or operation["worker_id"]
            self._worker(connection, target)
            duplicate = connection.execute(
                "SELECT id FROM tts_download WHERE worker_id=? AND manifest_id=? AND id!=? "
                "AND status IN ('queued','downloading','verifying','cancelling')",
                (target, operation["manifest_id"], operation_id),
            ).fetchone()
            if duplicate:
                raise ServiceError(409, "同じファイルの取得操作がこのWorkerで実行中です。")
            # Retargeting never re-resolves the model: the original pinned manifest stays intact.
            connection.execute(
                "UPDATE tts_download SET worker_id=?,status='queued',phase=NULL,error=NULL,"
                "cancel_requested=0,lease_id=NULL,lease_expires_at=NULL,updated_at=? WHERE id=?",
                (target, self.coordinator.clock(), operation_id),
            )
            return _public_operation(required(connection, "tts_download", operation_id))

    def claim(self, worker_id: str) -> dict | None:
        with self.db.transaction() as connection:
            self._recover(connection)
            self._worker(connection, worker_id, require_online=False)
            now = self.coordinator.clock()
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
            if connection.execute(
                "SELECT 1 FROM tts_download WHERE worker_id=? "
                "AND status IN ('downloading','verifying','cancelling')", (worker_id,),
            ).fetchone():
                return None
            operation = connection.execute(
                "SELECT * FROM tts_download WHERE worker_id=? AND status='queued' "
                "ORDER BY created_at,id LIMIT 1", (worker_id,),
            ).fetchone()
            if operation is None:
                return None
            lease_id = uuid4().hex
            connection.execute(
                "UPDATE tts_download SET status='downloading',phase='downloading',lease_id=?,"
                "lease_expires_at=?,updated_at=? WHERE id=?",
                (lease_id, now + self.coordinator.lease_seconds, now, operation["id"]),
            )
            claimed = required(connection, "tts_download", operation["id"])
            return {**_public_operation(claimed), "lease_id": lease_id,
                    "manifest": json.loads(claimed["manifest"])}

    def progress(self, operation_id: str, body: DownloadProgress) -> dict:
        with self.db.transaction() as connection:
            self._recover(connection)
            operation = required(connection, "tts_download", operation_id)
            if (operation["worker_id"] != body.worker_id
                    or operation["lease_id"] != body.lease_id
                    or operation["status"] not in ACTIVE):
                raise ServiceError(409, "この取得操作のリースは有効ではありません。")
            # A stale completion after cancellation may never advertise readiness.
            if operation["cancel_requested"]:
                status = "cancelled" if body.status in ("cancelled", "failed", "completed") else "cancelling"
            else:
                status = body.status
            if body.done_bytes > operation["total_bytes"]:
                raise ServiceError(422, "取得済みサイズがカタログの容量を超えています。")
            if body.total_bytes not in (0, operation["total_bytes"]):
                raise ServiceError(422, "取得容量が固定済みカタログと一致しません。")
            now = self.coordinator.clock()
            terminal = status in ("completed", "cancelled", "failed")
            done = operation["total_bytes"] if status == "completed" else body.done_bytes
            connection.execute(
                "UPDATE tts_download SET status=?,phase=?,done_bytes=?,current_file=?,error=?,"
                "lease_id=?,lease_expires_at=?,updated_at=? WHERE id=?",
                (status, status if not terminal else None, done,
                 _safe_file(body.current_file, json.loads(operation["manifest"])),
                 _safe_error(body.error) if status == "failed" else None,
                 None if terminal else body.lease_id,
                 None if terminal else now + self.coordinator.lease_seconds, now, operation_id),
            )
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, body.worker_id))
            if status == "completed":
                self._mark_ready(connection, operation, now)
            return _public_operation(required(connection, "tts_download", operation_id))

    @staticmethod
    def _mark_ready(connection, operation: dict, now: float) -> None:
        manifest = json.loads(operation["manifest"])
        ready = {(operation["precision"], operation["manifest_id"])}
        # Always preserve the selected frozen manifest. New catalog versions may
        # advertise aliases only if their actual file identities still match it.
        for precision in ("fp32", "bf16", "int8", "int4"):
            candidate = _bundle(operation["model_id"], precision)
            if (candidate.get("weight_id") == manifest.get("weight_id")
                    and candidate["files"] == manifest["files"]):
                ready.add((precision, candidate["manifest_id"]))
        for precision, manifest_id in ready:
            connection.execute(
                "INSERT INTO tts_worker_inventory "
                "(worker_id,model_id,precision,manifest_id,file_download_ready,updated_at) VALUES(?,?,?,?,1,?) "
                "ON CONFLICT(worker_id,model_id,precision,manifest_id) "
                "DO UPDATE SET file_download_ready=1,updated_at=excluded.updated_at",
                (operation["worker_id"], operation["model_id"], precision, manifest_id, now),
            )

    def inventory(self, worker_id: str, body: TTSInventory) -> dict:
        with self.db.transaction() as connection:
            worker = self._worker(connection, worker_id, require_online=False)
            purposes = set(generation_purposes(json.loads(worker["capabilities"])))
            now = self.coordinator.clock()
            seen = set()
            for item in body.models:
                key = (item.model_id, item.precision, item.manifest_id)
                if key in seen:
                    raise ServiceError(422, "モデルの取得記録が重複しています。")
                seen.add(key)
                if len(set(item.generation_purposes)) != len(item.generation_purposes):
                    raise ServiceError(422, "モデルの生成用途が重複しています。")
                if (not set(item.generation_purposes).issubset(purposes)
                        or item.generation_ready and not item.file_download_ready
                        or item.generation_ready and not item.generation_purposes
                        or not item.generation_ready and item.generation_purposes):
                    raise ServiceError(422, "モデルの生成対応情報がWorkerの機能と一致しません。")
                current = _bundle(item.model_id, item.precision)["manifest_id"]
                pinned = connection.execute(
                    "SELECT 1 FROM tts_download WHERE worker_id=? AND model_id=? AND manifest_id=?",
                    (worker_id, item.model_id, item.manifest_id),
                ).fetchone()
                if item.manifest_id != current and not pinned:
                    raise ServiceError(422, "モデルの取得記録がカタログと一致しません。")
            connection.execute("DELETE FROM tts_worker_inventory WHERE worker_id=?", (worker_id,))
            for item in body.models:
                connection.execute(
                    "INSERT INTO tts_worker_inventory "
                    "(worker_id,model_id,precision,manifest_id,file_download_ready,updated_at,"
                    "generation_ready,generation_purposes) VALUES(?,?,?,?,?,?,?,?)",
                    (worker_id, item.model_id, item.precision, item.manifest_id,
                     int(item.file_download_ready), now, int(item.generation_ready),
                     json.dumps(item.generation_purposes)),
                )
            connection.execute("UPDATE worker SET last_seen_at=? WHERE id=?", (now, worker_id))
        return {"worker_id": worker_id, "generation_active": True}


def register_routes(app: FastAPI, service) -> None:
    """Register independent management routes without expanding Coordinator."""
    def tts() -> TTSService:
        return TTSService(service())

    @app.get("/api/settings/tts")
    def read_tts_settings():
        return tts().read_settings()

    @app.post("/api/settings/tts")
    def save_tts_settings(body: TTSSettings):
        return tts().save_settings(body)

    @app.get("/api/tts/models")
    def tts_models():
        return tts().models()

    @app.get("/api/tts-downloads")
    def tts_downloads():
        return tts().downloads()

    @app.post("/api/workers/{worker_id}/tts-downloads", status_code=201)
    def create_tts_download(worker_id: str, body: DownloadInput):
        return tts().enqueue(worker_id, body)

    @app.post("/api/tts-downloads/{operation_id}/cancel")
    def cancel_tts_download(operation_id: str):
        return tts().cancel(operation_id)

    @app.post("/api/tts-downloads/{operation_id}/resume")
    def resume_tts_download(operation_id: str, body: DownloadResume | None = None):
        return tts().resume(operation_id, body.worker_id if body else None)

    @app.post("/api/workers/{worker_id}/tts-downloads/claim")
    def claim_tts_download(worker_id: str):
        return {"operation": tts().claim(worker_id)}

    @app.post("/api/tts-downloads/{operation_id}/progress")
    def report_tts_download(operation_id: str, body: DownloadProgress):
        return tts().progress(operation_id, body)

    @app.post("/api/workers/{worker_id}/tts-models")
    def report_tts_inventory(worker_id: str, body: TTSInventory):
        return tts().inventory(worker_id, body)
