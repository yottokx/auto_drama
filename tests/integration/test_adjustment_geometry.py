"""A non-saving preview and every published chapter use identical portrait rectangles."""
from __future__ import annotations

import io
import json
import re
from copy import deepcopy
from zipfile import ZipFile

from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from packages.narrative.validation import script_character_id
from services.coordinator.adjustment_service import _AdjustmentPublisher
from services.coordinator.app import create_app
from tests.integration.test_m3 import approved, finish, production


def members(data):
    with ZipFile(io.BytesIO(data)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def test_preview_does_not_change_original_build_and_all_chapters_publish_its_geometry(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        finish(client, worker)
        original = production(client, project)
        archives = {chapter["chapter_number"]: client.get(chapter["export_url"]).content
                    for chapter in original["chapters"]}
        endpoint = f"/api/m3/projects/{project}/adjustments"
        read = client.get(endpoint)
        assert read.status_code == 200, read.text
        assert read.json()["complete"]
        assert read.json()["draft"] is None
        opened = client.post(endpoint + "/start", json={})
        assert opened.status_code == 200, opened.text
        opened = opened.json()
        base = opened["draft"]
        assert [row["build"]["id"] for row in production(client, project)["chapters"]] == [
            row["build"]["id"] for row in original["chapters"]]
        edits = deepcopy(base["characters"])
        changed = edits[0]["character_id"]
        edits[0].update(offset_y=-28, scale=1.2)
        payload = {"expected_revision": base["revision"], "characters": edits}
        response = client.post(endpoint + "/preview", json=payload)
        assert response.status_code == 200, response.text
        preview = response.json()["draft"]["geometry"]
        assert preview["baseline"] == base["baseline"]
        for cid in preview["characters"]:
            if cid != changed:
                assert preview["characters"][cid] == base["geometry"]["characters"][cid]
        # Opening and moving preview controls cannot mutate published geometry.
        assert client.get(endpoint).json()["draft"]["characters"] == base["characters"]
        for chapter in original["chapters"]:
            assert client.get(chapter["export_url"]).content == archives[chapter["chapter_number"]]
        saved = client.put(endpoint, json=payload)
        assert saved.status_code == 200, saved.text
        applied = client.post(endpoint + "/apply", json={"expected_revision": saved.json()["draft"]["revision"]})
        assert applied.status_code == 200, applied.text
        assert applied.json()["edition"]["id"] != opened["edition"]["id"], applied.text
        assert applied.json()["draft"]["status"] == "applied", applied.text
        assert not applied.json()["busy"], applied.text
        current = production(client, project)
        lookup = {script_character_id(cid): cid for cid in preview["characters"]}
        fixed_baseline = None
        for chapter in current["chapters"]:
            old = members(archives[chapter["chapter_number"]])
            new = members(client.get(chapter["export_url"]).content)
            assert new["narrative.json"] == old["narrative.json"]
            assert new["approval.json"] == old["approval.json"]
            script = json.loads(new["script.json"])
            if fixed_baseline is None:
                fixed_baseline = script["portrait_baseline"]
            assert script["portrait_baseline"] == fixed_baseline
            served = client.get(f"/player/{chapter['build']['id']}/data/scenario/first.ks")
            assert served.status_code == 200
            assert served.content == new["data/scenario/first.ks"]
            shows = [dict(re.findall(r'(\w+)="([^"]*)"', line))
                     for line in served.text.splitlines() if line.startswith("[chara_show ")]
            entries = [direction for direction in script["directions"] if direction["kind"] == "enter"]
            assert len(shows) == len(entries)
            for tag, entry in zip(shows, entries, strict=True):
                cid = lookup[entry["character_id"]]
                expected = preview["characters"][cid]["positions"][entry["position"]]
                assert {key: int(tag[key]) for key in ("left", "top", "width", "height")} == expected
        for chapter in original["chapters"]:
            assert client.get(chapter["export_url"]).content == archives[chapter["chapter_number"]]


def test_second_chapter_publish_failure_keeps_every_original_build_selected(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        project, worker = approved(client)
        finish(client, worker)
        original = production(client, project)
        archives = {chapter["build"]["id"]: client.get(chapter["export_url"]).content
                    for chapter in original["chapters"]}
        endpoint = f"/api/m3/projects/{project}/adjustments"
        opened = client.post(endpoint + "/start", json={}).json()
        edits = deepcopy(opened["draft"]["characters"])
        edits[0].update(offset_y=-32, scale=1.3)
        saved = client.put(endpoint, json={"expected_revision": opened["draft"]["revision"], "characters": edits})
        assert saved.status_code == 200, saved.text
        coordinator = app.state.coordinator

        def counts():
            with coordinator.db.transaction() as connection:
                return {table: connection.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
                        for table in ("chapter_build", "artifact", "publication_edition", "edition_build")}

        before = counts()
        publish = _AdjustmentPublisher._publish
        attempted = []

        def fail_second(self, connection, chapter, requirements, *, presentation=None):
            attempted.append(chapter["chapter_number"])
            if chapter["chapter_number"] == 2:
                raise ValueError("Second chapter export failed after first chapter was assembled.")
            return publish(self, connection, chapter, requirements, presentation=presentation)

        monkeypatch.setattr(_AdjustmentPublisher, "_publish", fail_second)
        applied = client.post(endpoint + "/apply", json={"expected_revision": saved.json()["draft"]["revision"]})
        assert applied.status_code == 200, applied.text
        assert attempted == [1, 2]
        assert applied.json()["draft"]["status"] == "failed"
        assert applied.json()["edition"]["id"] == opened["edition"]["id"]
        assert counts() == before  # The first chapter's new records roll back too.
        current = production(client, project)
        assert [row["build"]["id"] for row in current["chapters"]] == [
            row["build"]["id"] for row in original["chapters"]]
        for chapter in original["chapters"]:
            assert client.get(chapter["export_url"]).content == archives[chapter["build"]["id"]]


def test_new_portrait_candidate_recomputes_only_its_layout_and_matches_published_rectangles(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = approved(client)
        finish(client, worker)
        endpoint = f"/api/m3/projects/{project}/adjustments"
        opened = client.post(endpoint + "/start", json={}).json()
        base = opened["draft"]
        cid = base["characters"][0]["character_id"]
        image = Image.new("RGBA", (600, 1200))
        ImageDraw.Draw(image).rectangle((160, 80, 430, 1110), fill="teal")
        output = io.BytesIO()
        image.save(output, "PNG")
        uploaded = client.post(endpoint + "/upload",
            data={"expected_revision": str(base["revision"]), "character_id": cid, "kind": "image"},
            files={"file": ("replacement.png", output.getvalue(), "image/png")})
        assert uploaded.status_code == 200, uploaded.text
        candidate = next(row for row in uploaded.json()["candidates"]
                         if row["character_id"] == cid and row["source"] == "upload")
        edits = deepcopy(base["characters"])
        edits[0].update(image_candidate_id=candidate["id"], offset_y=-19, scale=1.15)
        payload = {"expected_revision": uploaded.json()["draft"]["revision"], "characters": edits}
        response = client.post(endpoint + "/preview", json=payload)
        assert response.status_code == 200, response.text
        preview = response.json()["draft"]["geometry"]
        assert preview["baseline"]["layouts"][cid] != base["baseline"]["layouts"][cid]
        assert preview["baseline"]["pixels_per_cm"] == base["baseline"]["pixels_per_cm"]
        assert preview["baseline"]["ground_y"] == base["baseline"]["ground_y"]
        for other in preview["characters"]:
            if other != cid:
                assert preview["characters"][other] == base["geometry"]["characters"][other]
        # The replacement's automatic rectangle remains an unsaved candidate.
        reread = client.get(endpoint).json()["draft"]
        assert reread["characters"] == base["characters"]
        assert reread["baseline"] == base["baseline"]
        saved = client.put(endpoint, json=payload)
        assert saved.status_code == 200, saved.text
        applied = client.post(endpoint + "/apply", json={"expected_revision": saved.json()["draft"]["revision"]})
        assert applied.status_code == 200, applied.text
        assert applied.json()["draft"]["status"] == "applied", applied.text
        observed = 0
        for chapter in production(client, project)["chapters"]:
            bundle = members(client.get(chapter["export_url"]).content)
            script = json.loads(bundle["script.json"])
            entries = [direction for direction in script["directions"] if direction["kind"] == "enter"]
            shows = [dict(re.findall(r'(\w+)="([^"]*)"', line))
                     for line in bundle["data/scenario/first.ks"].decode().splitlines()
                     if line.startswith("[chara_show ")]
            for tag, entry in zip(shows, entries, strict=True):
                if entry["character_id"] == script_character_id(cid):
                    expected = preview["characters"][cid]["positions"][entry["position"]]
                    assert {key: int(tag[key]) for key in ("left", "top", "width", "height")} == expected
                    observed += 1
        assert observed > 0
