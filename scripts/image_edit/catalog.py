"""Download immutable character references and scene context without starting jobs."""

from __future__ import annotations

import hashlib
import io
import threading
import uuid
from pathlib import Path
from urllib.parse import quote, unquote, urlparse

from PIL import Image, ImageOps, UnidentifiedImageError

from scripts.audio.catalog import CatalogError, CoordinatorCatalog, fetch_bytes, fetch_json


class CoordinatorImageCatalog(CoordinatorCatalog):
    """One instance pins a project snapshot across portrait and scene requests."""

    def __init__(self, base_url: str = "http://127.0.0.1:8000", timeout: float = 20):
        super().__init__(base_url, timeout)
        self._snapshots: dict[str, dict] = {}
        self._snapshot_lock = threading.Lock()

    def _snapshot(self, project_id: str) -> dict:
        with self._snapshot_lock:
            if project_id not in self._snapshots:
                url = self.base_url + "/api/image-experiments/projects/" + quote(project_id, safe="")
                value = fetch_json(url, self.timeout)
                if value.get("project_id") != project_id:
                    raise CatalogError("画像実験用の取得結果が選択した作品と一致しません。")
                self._snapshots[project_id] = value
            return self._snapshots[project_id]

    def _json(self, path: str) -> dict:
        prefix = "/api/m3/projects/"
        if path.startswith(prefix):
            return self._snapshot(unquote(path[len(prefix):]))
        return fetch_json(self.base_url + path, self.timeout)

    def projects(self) -> list[dict]:
        values = super().projects()
        return [{"id": value.id, "title": value.title} for value in values]

    def portraits(self, project_id: str) -> list[dict]:
        values = self._snapshot(project_id).get("portraits", [])
        if not isinstance(values, list) or any(not isinstance(row, dict) for row in values):
            raise CatalogError("立ち絵一覧の形式が正しくありません。")
        return values

    def chapters(self, project_id: str) -> list[dict]:
        return [{"number": value.number, "title": value.title,
                 "scenes": [{"id": scene.id, "label": scene.label,
                             "preview": scene.preview, "context": scene.context}
                            for scene in value.scenes]}
                for value in super().chapters(project_id)]

    def download_reference(self, portrait: dict, directory: Path) -> dict:
        artifact_id = str(portrait.get("artifact_id") or portrait.get("image_artifact_id") or "")
        if not artifact_id:
            raise CatalogError("立ち絵の素材IDがありません。")
        # Build a same-coordinator URL from the ID rather than trusting a returned URL.
        content = fetch_bytes(self.base_url + "/api/artifacts/"
                              + quote(artifact_id, safe="") + "/content", self.timeout)
        original_hash = hashlib.sha256(content).hexdigest()
        source = dict(portrait.get("source") or {})
        expected = source.get("sha256")
        if expected and expected != original_hash:
            raise CatalogError("立ち絵のハッシュが選択した公開版と一致しません。")
        try:
            with Image.open(io.BytesIO(content)) as image:
                if image.format not in {"PNG", "JPEG", "WEBP"} or getattr(image, "n_frames", 1) != 1:
                    raise CatalogError("参照画像は静止画のPNG・JPEG・WebPを選んでください。")
                if max(image.size) > 4096 or min(image.size) < 1:
                    raise CatalogError("参照画像の各辺は4096px以内にしてください。")
                normalized = ImageOps.exif_transpose(image).convert("RGBA")
        except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
            raise CatalogError("立ち絵を画像として読み込めません。") from exc
        directory = Path(directory).expanduser().resolve()
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"reference-{uuid.uuid4().hex}.png"
        with destination.open("xb") as stream:
            normalized.save(stream, format="PNG")
        source.update(artifact_id=artifact_id, original_sha256=original_hash,
                      coordinator_origin=urlparse(self.base_url).netloc)
        return {"path": str(destination), "name": str(portrait.get("name") or "参照人物"),
                "character_id": str(portrait.get("character_id") or artifact_id), "source": source}
