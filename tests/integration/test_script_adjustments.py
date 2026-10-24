"""The script and staging tab reads each chapter as the current edition publishes it."""
from __future__ import annotations

import copy

from tests.integration.test_adjustments import apply, start, story, upload  # noqa: F401
from tests.integration.test_m2 import claim
from tests.integration.test_m3 import complete


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


def edit(line, **changes):
    return {"production_id": line["production_id"], "utterance_id": line["id"], "display_text": line["display_text"],
            "spoken_text": line["spoken_text"], "voice_emotion": line["voice_emotion"], "delivery": line["delivery"], **changes}


def lines_of(client, endpoint, value):
    production_id = value["scenes"][0]["production_id"]
    chapter = client.get(f"{endpoint}/script/{production_id}").json()
    return {line["id"]: {**line, "production_id": production_id} for line in chapter["lines"]}


def save_script(client, endpoint, value, lines, status=200):
    response = client.put(endpoint, json={"expected_revision": value["draft"]["revision"],
                                          "characters": value["draft"]["characters"],
                                          "script": {"lines": lines, "directions": []}})
    assert response.status_code == status, response.text
    return response.json()


def test_changed_words_wait_for_a_voice_choice_and_reach_the_published_script(story):  # noqa: F811
    client, _coordinator, _project, _worker, endpoint = story
    value = start(client, endpoint)
    lines = lines_of(client, endpoint, value)
    narration, spoken = lines["scene-1-u2"], lines["scene-1-u3"]
    changes = [edit(narration, display_text="エマは扉の前で足を止めた。"), edit(spoken, voice_emotion="happy")]
    value = save_script(client, endpoint, value, changes)
    # A changed feeling leaves the old recording out of date until a voice is chosen for it.
    assert client.post(endpoint + "/apply", json={"expected_revision": value["draft"]["revision"]}).status_code == 409
    value = save_script(client, endpoint, value, [changes[0], {**changes[1], "audio": "old"}])
    stored = value["draft"]["script"]["lines"]
    # Narration has no voice: what it reads is what it shows.
    assert stored[0]["spoken_text"] == "エマは扉の前で足を止めた。"
    value = apply(client, endpoint, value)
    assert value["draft"]["status"] == "applied"
    published = lines_of(client, endpoint, value)
    assert published["scene-1-u2"]["display_text"] == "エマは扉の前で足を止めた。"
    assert published["scene-1-u3"]["voice_emotion"] == "happy"
    assert published["scene-1-u3"]["audio_url"] == spoken["audio_url"]
    # The next adjustment starts from what is published; nothing waits on its voice any more.
    value = client.post(endpoint + "/start", json={"expected_edition_id": value["edition"]["id"]}).json()
    assert [row["audio"] for row in value["draft"]["script"]["lines"]] == [None, None]
    assert apply(client, endpoint, value)["draft"]["status"] == "applied"
    assert lines_of(client, endpoint, value)["scene-1-u2"]["display_text"] == "エマは扉の前で足を止めた。"


def test_a_line_recorded_again_is_used_only_for_the_words_it_was_made_for(story):  # noqa: F811
    client, _coordinator, _project, worker, endpoint = story
    value = start(client, endpoint)
    spoken = lines_of(client, endpoint, value)["scene-1-u3"]
    request = {"production_id": spoken["production_id"], "utterance_id": spoken["id"],
               "spoken_text": "新しい読みです。", "voice_emotion": "happy", "delivery": "明るく"}
    value = client.post(endpoint + "/script-voice", json={"expected_revision": value["draft"]["revision"], **request}).json()
    job = claim(client, worker)
    payload = job["payload"]
    assert (payload["dialogue_text"], payload["voice_emotion"], payload["delivery"]) == ("新しい読みです。", "happy", "明るく")
    voice = next(row for row in value["cast"] if row["character_id"] == payload["character_id"])
    chosen = next(row for row in value["candidates"] if row["id"] == voice["voice_candidate_id"])
    assert payload["reference_voice"]["artifact_id"] == chosen["artifact_id"]
    assert complete(client, worker, job).status_code == 200
    value = client.get(endpoint).json()
    [candidate] = value["draft"]["script_voices"]
    assert candidate["status"] == "completed" and candidate["url"] and candidate["utterance_id"] == spoken["id"]
    line = edit(spoken, spoken_text="新しい読みです。", voice_emotion="happy", delivery="明るく", voice_job_id=candidate["job_id"])
    save_script(client, endpoint, value, [{**line, "voice_emotion": "sad"}], status=422)
    value = save_script(client, endpoint, value, [line])
    value = apply(client, endpoint, value)
    assert lines_of(client, endpoint, value)["scene-1-u3"]["audio_url"] == candidate["url"]


def test_a_new_reference_voice_records_the_changed_words(story):  # noqa: F811
    client, _coordinator, _project, worker, endpoint = story
    value = start(client, endpoint)
    spoken = lines_of(client, endpoint, value)["scene-1-u3"]
    value = upload(client, endpoint, value, kind="voice")
    assert complete(client, worker, claim(client, worker)).status_code == 200
    candidate = next(row for row in value["candidates"] if row["source"] == "upload")
    characters = copy.deepcopy(value["draft"]["characters"])
    next(row for row in characters if row["character_id"] == "support-1")["voice_candidate_id"] = candidate["id"]
    value = client.put(endpoint, json={"expected_revision": client.get(endpoint).json()["draft"]["revision"],
                                       "characters": characters, "script": {"lines": [
                                           edit(spoken, spoken_text="別の読みです。", audio="old")], "directions": []}}).json()
    value = apply(client, endpoint, value)
    texts = []
    while job := claim(client, worker):
        texts.append(job["payload"]["dialogue_text"])
        assert complete(client, worker, job).status_code == 200
    # Other lines saying the same words keep them; only the edited line reads anew.
    assert texts.count("別の読みです。") == 1 and spoken["spoken_text"] in texts
    assert lines_of(client, endpoint, client.get(endpoint).json())["scene-1-u3"]["spoken_text"] == "別の読みです。"
