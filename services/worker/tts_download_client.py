"""Poll model-management requests independently of long-running generation jobs."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)
TERMINAL = frozenset({"completed", "cancelled", "failed", "interrupted"})


class TTSDownloadAgent:
    def __init__(self, client: httpx.Client, worker_id: str, root: Path, *, manager=None):
        if manager is None:
            from services.worker.tts_downloads import DownloadManager

            manager = DownloadManager(root)
        self.client = client
        self.worker_id = worker_id
        self.root = Path(root)
        self.manager = manager
        self._leases: dict[str, str] = {}
        self._stopped = threading.Event()
        self._thread: threading.Thread | None = None
        self._next_inventory = 0.0

    def start(self):
        self._thread = threading.Thread(target=self._run, name="tts-download-control", daemon=True)
        self._thread.start()
        return self

    def _run(self):
        while not self._stopped.is_set():
            try:
                self.tick()
            except (httpx.HTTPError, OSError, ValueError):
                logger.exception("TTS model management unavailable; retrying")
            self._stopped.wait(2)

    def _progress(self, identifier: str, snapshot: dict):
        status = snapshot.get("status", "downloading")
        if status == "queued":
            status = "downloading"
        response = self.client.post(f"/api/tts-downloads/{identifier}/progress", json={
            "worker_id": self.worker_id,
            "lease_id": self._leases[identifier],
            "status": status,
            "done_bytes": snapshot.get("completed_bytes", snapshot.get("downloaded_bytes", 0)),
            "total_bytes": snapshot.get("total_bytes", 0),
            "phase": snapshot.get("phase", status),
            "current_file": snapshot.get("current_file"),
            "error": snapshot.get("error"),
        })
        if response.status_code == 409:
            self.manager.cancel(identifier)
            self._leases.pop(identifier, None)
            return
        response.raise_for_status()
        result = response.json()
        operation = result.get("operation", result)
        if result.get("cancel_requested") or operation.get("cancel_requested"):
            self.manager.cancel(identifier)
        if status in TERMINAL:
            self._leases.pop(identifier, None)
            self._next_inventory = 0

    def tick(self):
        """Renew active leases even when a file stream has made no new progress."""
        snapshots = {
            item.get("operation_id", item.get("id")): item
            for item in self.manager.snapshots()
        }
        for identifier in list(self._leases):
            self._progress(identifier, snapshots.get(identifier, {}))
        if time.monotonic() >= self._next_inventory:
            from services.worker.tts_inventory import generation_inventory

            models = [{key: item[key] for key in (
                "model_id", "precision", "manifest_id", "file_download_ready"
            ) if key in item} for item in self.manager.inventory()]
            models = generation_inventory(self.root, models)
            response = self.client.post(
                f"/api/workers/{self.worker_id}/tts-models", json={"models": models}
            )
            response.raise_for_status()
            self._next_inventory = time.monotonic() + 15
        if self._leases:
            return
        response = self.client.post(f"/api/workers/{self.worker_id}/tts-downloads/claim", json={})
        response.raise_for_status()
        operation = response.json().get("operation")
        if operation is None:
            return
        identifier = operation["id"]
        self._leases[identifier] = operation["lease_id"]
        try:
            self.manager.start({**operation, "operation_id": identifier})
        except (OSError, ValueError):
            logger.exception("Could not start TTS download %s", identifier)
            self._progress(identifier, {
                "status": "failed", "error": "モデルの取得を開始できませんでした。Workerのログを確認してください。"
            })

    def close(self):
        self._stopped.set()
        for identifier in list(self._leases):
            self.manager.cancel(identifier)
        if self._thread is not None:
            self._thread.join(timeout=10)
        self.manager.close(wait=False)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()
