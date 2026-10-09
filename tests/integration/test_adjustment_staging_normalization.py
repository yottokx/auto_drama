"""Unchanged adjustments repair legacy pauses without replacing adopted content."""

import copy
import io
import json
from pathlib import Path
from zipfile import ZipFile

from fastapi.testclient import TestClient

from packages.narrative import script_conversion
from services.coordinator.app import create_app
from services.coordinator.m3_service import M3Service
from services.coordinator.service import Coordinator
from tests.integration import test_m3
from tests.integration.test_adjustments import apply, start
from tests.integration.test_m3 import approved, finish, production
from tests.integration.test_m3_rebuild import counts


def test_unchanged_apply_normalizes_legacy_pauses_and_retains_audit(tmp_path, monkeypatch):
    coordinator = Coordinator(tmp_path)
    original_narrative = test_m3.narrative

    def legacy_narrative(snapshot):
        value = original_narrative(snapshot)
        scene = value["scenes"][0]
        utterance_id = scene["utterances"][1]["id"]
        scene["directions"].extend([
            {"id": f"legacy-pause-{index}", "utterance_id": utterance_id,
             "kind": "pause", "timing": "after", "character_id": None,
             "position": None, "duration_ms": 500}
            for index in range(49)
        ])
        return value

    with TestClient(create_app(coordinator=coordinator)) as client:
        # Model a publication made before normalization was introduced. The
        # resulting Script and ZIP are genuine immutable adopted artifacts.
        with monkeypatch.context() as patch:
            patch.setattr(test_m3, "narrative", legacy_narrative)
            patch.setattr(script_conversion, "normalize_directions",
                          lambda directions: (copy.deepcopy(directions), {"changes": []}))
            project, worker = approved(client)
            finish(client, worker)
        endpoint = f"/api/m3/projects/{project}/adjustments"
        before = production(client, project)["chapters"]
        originals = {chapter["build"]["id"]: client.get(chapter["export_url"]).content
                     for chapter in before}
        original_builds = {chapter["build"]["id"]: copy.deepcopy(chapter["build"])
                           for chapter in before}
        baseline = counts(coordinator)
        value = apply(client, endpoint, start(client, endpoint))
        assert value["draft"]["status"] == "applied"
        after = production(client, project)["chapters"]
        assert counts(coordinator) == {
            **baseline, "artifact": baseline["artifact"] + 2 * len(after),
            "chapter_build": baseline["chapter_build"] + len(after),
        }
        audits = {}
        for old, new in zip(before, after, strict=True):
            old_build, new_build = old["build"], new["build"]
            assert new_build["id"] != old_build["id"]
            assert new_build["script_artifact_id"] != old_build["script_artifact_id"]
            assert new_build["export_artifact_id"] != old_build["export_artifact_id"]
            assert new_build["manifest"] == old_build["manifest"]
            assert new["narrative_artifact_id"] == old["narrative_artifact_id"]
            old_data = originals[old_build["id"]]
            assert client.get(old["export_url"]).content == old_data
            assert M3Service(coordinator).build(old_build["id"]) == original_builds[old_build["id"]]
            with ZipFile(io.BytesIO(old_data)) as left, ZipFile(io.BytesIO(
                    client.get(new["export_url"]).content)) as right:
                original_script = json.loads(left.read("script.json"))
                script = json.loads(right.read("script.json"))
                assert {key: value for key, value in script.items() if key != "directions"} == {
                    key: value for key, value in original_script.items() if key != "directions"}
                old_pauses = [row for row in original_script["directions"] if row["kind"] == "pause"]
                new_pauses = [row for row in script["directions"] if row["kind"] == "pause"]
                assert len(old_pauses) == 49
                assert new_pauses == [old_pauses[0]]
                scenario = right.read("data/scenario/first.ks")
                assert left.read("data/scenario/first.ks").count(b'[ad_pause time="500"]') == 49
                assert scenario.count(b'[ad_pause time="500"]') == 1
                for name in left.namelist():
                    if (name in {"approval.json", "narrative.json", "chapter-manifest.json",
                                 "data/others/auto_drama_stages.json"}
                            or name.startswith("sources/")
                            or Path(name).suffix in {".png", ".wav", ".mp3"}):
                        assert right.read(name) == left.read(name), name
                audit_bytes = right.read("staging-normalization.json")
                audit = json.loads(audit_bytes)
                assert audit["schema_version"] == 1
                assert len(audit["reports"]) == 1
                report = audit["reports"][0]
                assert report["removed_count"] == 48
                assert report["original_directions"] == original_script["directions"]
                assert new_build["validation"] == {
                    **old_build["validation"], "staging_normalization": audit,
                }
                audits[new["production_id"]] = audit_bytes

        # A subsequent unchanged apply must retain the original audit and reuse
        # the normalized Script and ZIP, rather than creating empty reports.
        normalized_counts = counts(coordinator)
        restarted = client.post(endpoint + "/start", json={"expected_edition_id": value["edition"]["id"]})
        assert restarted.status_code == 200, restarted.text
        again = apply(client, endpoint, restarted.json())
        assert again["draft"]["status"] == "applied"
        final = production(client, project)["chapters"]
        assert counts(coordinator) == {
            **normalized_counts,
            "chapter_build": normalized_counts["chapter_build"] + len(final),
        }
        for previous, current in zip(after, final, strict=True):
            for field in ("script_artifact_id", "export_artifact_id", "manifest", "validation"):
                assert current["build"][field] == previous["build"][field], field
            with ZipFile(io.BytesIO(client.get(current["export_url"]).content)) as archive:
                assert archive.read("staging-normalization.json") == audits[current["production_id"]]
        for old in before:
            assert client.get(old["export_url"]).content == originals[old["build"]["id"]]
