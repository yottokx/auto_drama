"""The script and staging tab reads each chapter as the current edition publishes it."""
from __future__ import annotations

from tests.integration.test_adjustments import start, story  # noqa: F401


def test_each_chapter_lists_its_lines_by_scene_with_the_stage_they_show(story):  # noqa: F811
    client, _coordinator, _project, _worker, endpoint = story
    view = start(client, endpoint)
    chapters = {row["production_id"]: row["chapter_number"] for row in view["scenes"]}
    for production_id, number in chapters.items():
        response = client.get(f"{endpoint}/script/{production_id}")
        assert response.status_code == 200, response.text
        chapter = response.json()
        assert chapter["chapter_number"] == number and chapter["stage"] == {"width": 960, "height": 640, "max_portraits": 3}
        scenes = [row["scene_id"] for row in view["scenes"] if row["production_id"] == production_id]
        assert [row["scene_id"] for row in chapter["scenes"]] == scenes
        assert {line["scene_id"] for line in chapter["lines"]} == set(scenes)
        first = chapter["lines"][0]
        assert first["stage"]["background_url"].startswith("/api/artifacts/")
        # The scene opens with its background and cast; the tab never lets those change.
        assert {row["kind"] for row in first["directions"] if not row["editable"]} >= {"background"}
        assert all(row["kind"] != "focus" for line in chapter["lines"] for row in line["directions"])
        spoken = [line for line in chapter["lines"] if line["speaker_id"]]
        assert spoken and all(line["audio_url"] for line in spoken)
        for line in chapter["lines"]:
            for row in line["stage"]["characters"]:
                assert client.get(row["image_url"]).status_code == 200


def test_a_chapter_outside_the_work_is_refused(story):  # noqa: F811
    client, _coordinator, _project, _worker, endpoint = story
    start(client, endpoint)
    assert client.get(f"{endpoint}/script/unknown").status_code == 404
