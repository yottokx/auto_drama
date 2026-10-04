"""Pin the material selected together and verify immutable portrait downloads."""

import hashlib
import io

import pytest
from PIL import Image

from scripts.audio.catalog import CatalogError
from scripts.image_edit import catalog


def test_one_readonly_snapshot_pins_portraits_and_scene_text(monkeypatch):
    responses = {
        "/api/projects": {"projects": [{"id": "p", "title": "作品"}]},
        "/api/image-experiments/projects/p": {
            "project_id": "p", "portraits": [{"name": "エマ", "artifact_id": "image-old"}],
            "production": {"id": "story", "chapters": [{"chapter_number": 1,
                "production_id": "story", "narrative_artifact_id": "unpublished-new",
                "build": {"id": "old-build", "validation": {"narrative_artifact_id": "text-old"}}}]}},
        "/api/artifacts/text-old/content": {"chapter_number": 1, "title": "公開本文",
            "storyline_id": "story", "scenes": [{"id": "s", "raw_text": "公開された本文。",
                "plan": {"objectives": "再会"}}]},
    }
    calls = []

    def fetch(url, timeout):
        path = url.removeprefix("http://localhost:8000")
        calls.append(path)
        return responses[path]

    monkeypatch.setattr(catalog, "fetch_json", fetch)
    client = catalog.CoordinatorImageCatalog("http://localhost:8000")
    assert client.projects() == [{"id": "p", "title": "作品"}]
    assert client.portraits("p")[0]["artifact_id"] == "image-old"
    chapters = client.chapters("p")
    assert chapters[0]["scenes"][0]["context"]["raw_text"] == "公開された本文。"
    assert calls.count("/api/image-experiments/projects/p") == 1
    assert "/api/m3/projects/p" not in calls
    assert "/api/artifacts/unpublished-new/content" not in calls


def test_download_keeps_transparency_checks_hash_and_ignores_foreign_url(monkeypatch, tmp_path):
    stream = io.BytesIO()
    Image.new("RGBA", (32, 64), (40, 80, 120, 0)).save(stream, "PNG")
    content = stream.getvalue()
    calls = []

    def fetch(url, timeout):
        calls.append(url)
        return content

    monkeypatch.setattr(catalog, "fetch_bytes", fetch)
    client = catalog.CoordinatorImageCatalog("http://localhost:8000")
    portrait = {"artifact_id": "portrait", "name": "エマ", "character_id": "emma",
                "image_url": "https://foreign.example/secret",
                "source": {"sha256": hashlib.sha256(content).hexdigest()}}
    value = client.download_reference(portrait, tmp_path)
    with Image.open(value["path"]) as image:
        assert image.mode == "RGBA" and image.getchannel("A").getextrema() == (0, 0)
    assert calls == ["http://localhost:8000/api/artifacts/portrait/content"]
    assert value["source"]["original_sha256"] == portrait["source"]["sha256"]
    portrait["source"]["sha256"] = "0" * 64
    with pytest.raises(CatalogError, match="ハッシュ"):
        client.download_reference(portrait, tmp_path)


def test_snapshot_rejects_wrong_project(monkeypatch):
    monkeypatch.setattr(catalog, "fetch_json", lambda *_: {"project_id": "other"})
    with pytest.raises(CatalogError, match="作品"):
        catalog.CoordinatorImageCatalog().portraits("p")
