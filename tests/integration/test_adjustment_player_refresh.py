"""Applying an unchanged presentation can refresh a newer player runtime."""

import copy
import io
import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from tests.integration import test_adjustments
from tests.integration.test_adjustments import apply, ids, save, start, zip_chapters
from tests.integration.test_m3 import production
from tests.integration.test_m3_rebuild import revise_player
from tests.integration.test_project_history import history, restore

story = test_adjustments.story


def test_unchanged_apply_refreshes_all_chapters_without_regenerating_assets(story, monkeypatch):
    client, coordinator, project, _worker, endpoint = story
    before = production(client, project)["chapters"]
    archives = {chapter["production_id"]: client.get(chapter["export_url"]).content for chapter in before}
    with coordinator.db.transaction() as connection:
        jobs = connection.execute("SELECT COUNT(*) FROM job").fetchone()[0]
    value = start(client, endpoint)
    revise_player(monkeypatch)
    after = apply(client, endpoint, value)
    assert after["draft"]["status"] == "applied"
    chapters = production(client, project)["chapters"]
    with coordinator.db.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM job").fetchone()[0] == jobs
    for old, new in zip(before, chapters, strict=True):
        old_data = archives[old["production_id"]]
        assert client.get(old["export_url"]).content == old_data
        with ZipFile(io.BytesIO(old_data)) as left, ZipFile(io.BytesIO(client.get(new["export_url"]).content)) as right:
            assert {name for name in left.namelist() if left.read(name) != right.read(name)} == {
                "manifest.json", "data/others/auto_drama_player.css",
            }
            assert b"Revised rendering" in right.read("data/others/auto_drama_player.css")
            for name in left.namelist():
                if name == "narrative.json" or name.startswith("sources/") or Path(name).suffix in {".png", ".wav", ".mp3"}:
                    assert left.read(name) == right.read(name), name
            old_script, new_script = (json.loads(archive.read("script.json")) for archive in (left, right))
            for field in ("utterances", "assets", "music_cues"):
                assert old_script.get(field) == new_script.get(field), field
        assert new["build"]["id"] != old["build"]["id"]


@pytest.mark.parametrize("runtime_changed", [False, True])
def test_unchanged_apply_refreshes_selected_history_and_pins_new_edition(
    story, monkeypatch, runtime_changed,
):
    client, coordinator, project, _worker, endpoint = story
    value = start(client, endpoint)
    selected = production(client, project)
    selected_history = history(client, project)["current_revision_id"]
    selected_archives = {
        chapter["production_id"]: client.get(chapter["export_url"]).content
        for chapter in selected["chapters"]
    }
    selected_navigation = zip_chapters(client, ids(selected)[0])
    changes = copy.deepcopy(value["draft"]["characters"])
    changes[0].update(offset_y=24, scale=1.15)
    changed = apply(client, endpoint, save(client, endpoint, value, changes))
    assert changed["draft"]["status"] == "applied"
    future = production(client, project)
    future_archives = {
        chapter["production_id"]: client.get(chapter["export_url"]).content
        for chapter in future["chapters"]
    }
    future_navigation = zip_chapters(client, ids(future)[0])
    assert any(
        selected_archives[key] != future_archives[key] for key in selected_archives
    )
    restore(client, project, selected_history)
    assert ids(production(client, project)) == ids(selected)
    restored = client.get(endpoint).json()
    with coordinator.db.transaction() as connection:
        jobs = connection.execute("SELECT COUNT(*) FROM job").fetchone()[0]
        artifacts = connection.execute("SELECT COUNT(*) FROM artifact").fetchone()[0]
    if runtime_changed:
        revise_player(monkeypatch)
    refreshed = apply(client, endpoint, restored)
    assert refreshed["draft"]["status"] == "applied"
    assert refreshed["edition"]["source_edition_id"] == restored["edition"]["id"]
    chapters = production(client, project)["chapters"]
    with coordinator.db.transaction() as connection:
        assert connection.execute("SELECT COUNT(*) FROM job").fetchone()[0] == jobs
        assert connection.execute("SELECT COUNT(*) FROM artifact").fetchone()[0] == (
            artifacts + (len(chapters) if runtime_changed else 0)
        )
    for old, newer, new in zip(selected["chapters"], future["chapters"], chapters, strict=True):
        assert new["build"]["id"] not in {old["build"]["id"], newer["build"]["id"]}
        assert new["build"]["revision"] > newer["build"]["revision"]
        for field in ("script_artifact_id", "manifest", "validation"):
            assert new["build"][field] == old["build"][field], field
        old_data = selected_archives[old["production_id"]]
        assert client.get(old["export_url"]).content == old_data
        assert client.get(newer["export_url"]).content == future_archives[old["production_id"]]
        new_data = client.get(new["export_url"]).content
        with ZipFile(io.BytesIO(old_data)) as left, ZipFile(io.BytesIO(new_data)) as right:
            assert left.namelist() == right.namelist()
            assert {name for name in left.namelist() if left.read(name) != right.read(name)} == (
                {"manifest.json", "data/others/auto_drama_player.css"}
                if runtime_changed else set()
            )
        if not runtime_changed:
            assert new["build"]["export_artifact_id"] == old["build"]["export_artifact_id"]
    assert zip_chapters(client, ids(selected)[0]) == selected_navigation
    assert zip_chapters(client, ids(future)[0]) == future_navigation
    assert [row["build_id"] for row in zip_chapters(client, chapters[0]["build"]["id"])] == [
        chapter["build"]["id"] for chapter in chapters
    ]
