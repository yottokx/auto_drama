"""Expression pictures are extras of a look: made on request, never required, shown where they exist."""

import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from packages.narrative.validation import script_character_id
from services.coordinator.app import create_app
from tests.integration.test_m2 import action, claim, detail, png, ready
from tests.integration.test_m2 import complete as complete_m2
from tests.integration.test_m2_outfits import outfit_bundle, outfit_worker
from tests.integration.test_m3 import complete, finish, narrative, output, production
from tests.integration.test_m3_outfits import approved_with_look


def expression_bundle(job, *, expression=None, media=None):
    payload = job["payload"]
    envelope = {"schema_version": 1, "kind": "m2_outfit_image", "result": {}, "trace": [],
                "provenance": {"seed": payload["seed"], "outfit": {
                    "id": payload["outfit"]["id"], "reference_sha256": payload["reference_image"]["sha256"],
                    "expression": {"id": expression or payload["expression"]["id"], "face_box": [10, 10, 60, 60]}}}}
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr("result.json", json.dumps(envelope, ensure_ascii=False))
        archive.writestr("image.png", media if media is not None else png(120))
    return data.getvalue()


def draw(client, project, worker, expected):
    """Run the queued expression jobs to the end; what each one was asked for, in order."""
    asked = []
    for _ in range(expected):
        job = claim(client, worker)
        assert job["kind"] == "m2_outfit_image" and "edit" not in job["payload"]
        asked.append((job["payload"]["outfit"]["id"], job["payload"]["expression"]["id"],
                      job["payload"]["reference_image"]["artifact_id"]))
        assert complete_m2(client, job, worker, expression_bundle(job)).status_code == 200
    assert claim(client, worker) is None
    return asked


def test_expressions_are_drawn_for_each_look_without_touching_the_approval(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        qwen = outfit_worker(client)
        action(client, project, "save-outfit", character_id="character-1",
               outfit={"name": "浴衣", "description": "藍色の浴衣。", "occasion": "夏祭り。"})
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        job = claim(client, qwen)
        assert complete_m2(client, job, qwen, outfit_bundle(job)).status_code == 200
        approved = action(client, project, "approve")
        approval = approved["draft"]["approval"]["id"]
        character = approved["draft"]["characters"][0]
        base, look = character["imageArtifactId"], character["outfits"][-1]["imageArtifactId"]

        action(client, project, "generate-expressions", character_id="character-1", expressions=["smile", "angry"])
        asked = draw(client, project, qwen, 4)
        # Each look's pictures are edited from that look's own portrait.
        assert asked == [("default", "smile", base), ("default", "angry", base),
                         ("look1", "smile", look), ("look1", "angry", look)]
        state = detail(client, project)["draft"]
        # The cast stays approved and unchanged: an expression is not something to approve.
        assert state["approved"] and state["approval"]["id"] == approval
        assert "expressions" not in json.dumps(state["approval"])
        pictures = state["characters"][0]["expressions"]
        assert [(row["outfitId"], row["expression"], row["stale"]) for row in pictures] == [
            ("default", "smile", False), ("default", "angry", False),
            ("look1", "smile", False), ("look1", "angry", False)]
        assert client.get(pictures[0]["imageUrl"]).content == png(120)
        # They are not candidates for the portrait itself.
        history = detail(client, project)["imageHistory"]["character-1"]
        assert pictures[0]["imageArtifactId"] not in {row["artifactId"] for row in history}

        # What exists and still matches its portrait is not drawn twice, unless asked.
        action(client, project, "generate-expressions", character_id="character-1", expressions=["smile"], status=409)
        action(client, project, "generate-expressions", character_id="character-1", outfit_ids=["look1"],
               expressions=["smile", "cry"])
        assert draw(client, project, qwen, 1) == [("look1", "cry", look)]
        action(client, project, "generate-expressions", character_id="character-1", outfit_ids=["default"],
               expressions=["smile"], redraw=True, face_box=[0.3, 0.1, 0.6, 0.3])
        job = claim(client, qwen)
        assert job["payload"]["expression"] == {"id": "smile", "face_box": [0.3, 0.1, 0.6, 0.3]}
        # A picture of another expression than the one asked for is refused.
        assert complete_m2(client, job, qwen, expression_bundle(job, expression="angry")).status_code == 422
        assert complete_m2(client, job, qwen, expression_bundle(job, media=png(121))).status_code == 200
        pictures = detail(client, project)["draft"]["characters"][0]["expressions"]
        assert len(pictures) == 5 and client.get(pictures[-1]["imageUrl"]).content == png(121)

        action(client, project, "delete-expression", character_id="character-1", outfit_id="look1", expression="cry")
        action(client, project, "delete-expression", character_id="character-1", outfit_id="look1", expression="cry",
               status=404)
        assert len(detail(client, project)["draft"]["characters"][0]["expressions"]) == 4
        assert detail(client, project)["draft"]["approved"]


def test_requests_that_cannot_be_drawn_are_refused(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        # Without a worker for the editing model in use there is nothing to draw with.
        action(client, project, "generate-expressions", character_id="character-1", expressions=["smile"], status=409)
        outfit_worker(client)
        for body in ({"expressions": ["neutral"]}, {"expressions": ["smile", "smile"]}, {"expressions": []},
                     {"expressions": ["wink"]}, {"expressions": ["smile"], "face_box": [0.1, 0.1, 0.5, 0.5]},
                     {"expressions": ["smile"], "outfit_ids": ["default"], "face_box": [0.5, 0.1, 0.5, 0.5]}):
            action(client, project, "generate-expressions", character_id="character-1", status=422, **body)
        action(client, project, "generate-expressions", character_id="character-1", outfit_ids=["nowhere"],
               expressions=["smile"], status=404)


def test_a_replaced_portrait_leaves_its_expressions_stale_until_drawn_again(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = ready(client)
        qwen = outfit_worker(client)
        action(client, project, "generate-expressions", character_id="character-1", expressions=["smile"])
        draw(client, project, qwen, 1)
        old = detail(client, project)["draft"]["characters"][0]["imageArtifactId"]
        action(client, project, "retake", character_id="character-1", scope="image-retake")
        job = claim(client, worker)
        assert job["kind"] == "m2_image" and complete_m2(client, job, worker, media=png(77)).status_code == 200
        character = detail(client, project)["draft"]["characters"][0]
        assert character["imageArtifactId"] != old
        (picture,) = character["expressions"]
        assert picture["stale"] and picture["baseImageArtifactId"] == old
        # Asking again draws it from the new portrait, and the old picture gives way to it.
        action(client, project, "generate-expressions", character_id="character-1", expressions=["smile"])
        assert draw(client, project, qwen, 1) == [("default", "smile", character["imageArtifactId"])]
        (picture,) = detail(client, project)["draft"]["characters"][0]["expressions"]
        assert not picture["stale"]


def test_a_chapter_shows_the_pictures_its_portraits_have_and_asks_for_the_rest(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, _ = approved_with_look(client)
        qwen = client.post("/api/workers", json={"name": "qwen", "capabilities": ["m2_outfit_image"]}).json()["id"]
        action(client, project, "generate-expressions", character_id="character-1", expressions=["smile"])
        draw(client, project, qwen, 2)
        job = claim(client, worker)
        assert job["kind"] == "m3_narrative"
        result = narrative(job["payload"]["approval_snapshot"])
        first, second = result["scenes"]
        first["plan"]["outfits"] = [{"character_id": "character-1", "outfit_id": "default", "change_to": "look1"}]
        lines = [row["id"] for row in first["utterances"]]

        def face(index, line, expression):
            return {"id": f"scene-1-x{index}", "utterance_id": line, "kind": "expression", "timing": "before",
                    "character_id": "character-1", "position": None, "duration_ms": 150, "expression": expression}

        first["directions"].extend([
            face(1, lines[1], "smile"), face(2, lines[2], "angry"),
            {"id": "scene-1-o1", "utterance_id": lines[3], "kind": "outfit", "timing": "before",
             "character_id": "character-1", "position": None, "duration_ms": 400, "outfit_id": "look1"},
            face(3, lines[3], "smile")])
        second["plan"]["outfits"] = [{"character_id": "character-1", "outfit_id": "look1"}]
        assert complete(client, worker, job, output(job, result)).status_code == 200
        finish(client, worker)
        chapter = production(client, project)["chapters"][0]
        assert chapter["status"] == "published", chapter
        with ZipFile(io.BytesIO(client.get(chapter["export_url"]).content)) as archive:
            script = json.loads(archive.read("script.json"))
            names = archive.namelist()
            scenario = archive.read("data/scenario/first.ks").decode("utf-8")
        main = next(row for row in script["characters"] if row["id"] == script_character_id("character-1"))
        files = {row["id"]: row["filename"] for row in script["assets"]}
        pictures = {(row.get("outfit_id"), row["expression"]): files[row["image_asset_id"]] for row in main["expressions"]}
        assert set(pictures) == {(None, "smile"), ("look1", "smile")}
        assert all(f"data/fgimage/{name}" in names for name in pictures.values())
        # The script keeps every face it was staged with, pictured or not.
        asked = [row["expression"] for row in script["directions"] if row["kind"] == "expression"]
        assert asked == ["smile", "angry", "smile"]
        ordinary = files[main["image_asset_id"]]
        second_line, third, fourth = (scenario.index(f"*utterance_{line}") for line in lines[1:4])
        assert f'storage="{pictures[(None, "smile")]}" time="150" cross="false" wait="false"]' in scenario[second_line:third]
        # No picture of anger exists: the ordinary portrait comes back.
        assert f'storage="{ordinary}" time="150" cross="false" wait="false"]' in scenario[third:fourth]
        assert f'storage="{pictures[("look1", "smile")]}"' in scenario[fourth:]
        # The finished work says how often it asks for each face, pictured or not.
        state = production(client, project)
        if all(row["status"] == "published" for row in state["chapters"]) and len(state["chapters"]) == 3:
            usage = client.get(f"/api/expressions/projects/{project}/usage").json()["characters"]
            assert usage["character-1"]["smile"] >= 2 and usage["character-1"]["angry"] >= 1
        # Asked by the name the chapter shows the portrait under, the cast's own pictures are found.
        shown = next(row["artifact_id"] for row in script["assets"] if row["id"] == main["image_asset_id"])
        listed = client.get(f"/api/expressions/projects/{project}", params={"base": shown}).json()["portraits"][shown]
        assert [(row["expression"], row["status"], row.get("from_cast")) for row in listed] == [("smile", "ready", True)]
        again = client.post(f"/api/expressions/projects/{project}", json={
            "base_artifact_id": shown, "character_id": "character-1", "expressions": ["smile"]})
        assert again.status_code == 409


def test_any_portrait_of_a_work_can_be_given_expressions_by_itself(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        identifier = project["project"]["id"]
        base = detail(client, project)["draft"]["characters"][0]["imageArtifactId"]
        endpoint = f"/api/expressions/projects/{identifier}"
        wanted = {"base_artifact_id": base, "character_id": "character-1", "expressions": ["smile", "cry"]}
        # Nothing to draw with until a worker holds the editing model.
        assert client.post(endpoint, json=wanted).status_code == 409
        qwen = outfit_worker(client)
        assert client.post(endpoint, json={**wanted, "base_artifact_id": "0" * 32}).status_code == 404
        assert client.post(endpoint, json={**wanted, "expressions": ["neutral"]}).status_code == 422
        started = client.post(endpoint, json=wanted).json()["portraits"][base]
        assert [(row["expression"], row["status"], row["url"]) for row in started] == [
            ("smile", "pending", None), ("cry", "pending", None)]
        # What is already being made is not asked for twice.
        assert client.post(endpoint, json=wanted).status_code == 409
        asked = set()
        for _ in range(2):
            job = claim(client, qwen)
            assert job["kind"] == "m2_outfit_image" and job["payload"]["portrait_expression"]
            asked.add(job["payload"]["expression"]["id"])
            assert job["payload"]["reference_image"]["artifact_id"] == base
            assert complete_m2(client, job, qwen, expression_bundle(job, expression="angry")).status_code == 422
            assert complete_m2(client, job, qwen, expression_bundle(job)).status_code == 200
        assert asked == {"smile", "cry"}
        shown = client.get(endpoint, params={"base": base}).json()
        assert shown["enabled"] is True
        smile, cry = shown["portraits"][base]
        assert (smile["status"], cry["status"]) == ("ready", "ready")
        assert client.get(smile["url"]).content == png(120)
        # Drawn again, the earlier picture stays in use until the new one is finished.
        again = client.post(endpoint, json={**wanted, "expressions": ["smile"], "redraw": True}).json()["portraits"][base]
        assert (again[0]["status"], again[0]["url"]) == ("pending", smile["url"])
        assert client.post(f"{endpoint}/remove", json={"base_artifact_id": base, "expression": "smile"}).status_code == 409
        job = claim(client, qwen)
        assert complete_m2(client, job, qwen, expression_bundle(job, media=png(121))).status_code == 200
        smile = client.get(endpoint, params={"base": base}).json()["portraits"][base][0]
        assert client.get(smile["url"]).content == png(121)
        removed = client.post(f"{endpoint}/remove", json={"base_artifact_id": base, "expression": "cry"}).json()
        assert [row["expression"] for row in removed["portraits"][base]] == ["smile"]
        assert client.post(f"{endpoint}/remove", json={"base_artifact_id": base, "expression": "cry"}).status_code == 404
        # The cast is untouched by any of it.
        assert "expressions" not in detail(client, project)["draft"]["characters"][0]


def test_a_chapter_finds_the_expressions_kept_by_the_portrait_it_shows(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, _ = approved_with_look(client)
        qwen = client.post("/api/workers", json={"name": "qwen", "capabilities": ["m2_outfit_image"]}).json()["id"]
        base = detail(client, project)["draft"]["characters"][0]["imageArtifactId"]
        assert client.post(f"/api/expressions/projects/{project}", json={
            "base_artifact_id": base, "character_id": "character-1", "expressions": ["serious"]}).status_code == 200
        job = claim(client, qwen)
        assert complete_m2(client, job, qwen, expression_bundle(job)).status_code == 200
        job = claim(client, worker)
        result = narrative(job["payload"]["approval_snapshot"])
        assert complete(client, worker, job, output(job, result)).status_code == 200
        finish(client, worker)
        chapter = production(client, project)["chapters"][0]
        assert chapter["status"] == "published", chapter
        with ZipFile(io.BytesIO(client.get(chapter["export_url"]).content)) as archive:
            script = json.loads(archive.read("script.json"))
        main = next(row for row in script["characters"] if row["id"] == script_character_id("character-1"))
        assert [(row.get("outfit_id"), row["expression"]) for row in main["expressions"]] == [(None, "serious")]


def test_expressions_made_for_a_finished_work_reach_it_when_the_adjustment_is_applied(tmp_path):
    from services.coordinator.service import Coordinator
    from tests.integration.test_adjustments import apply, start
    from tests.integration.test_m3 import approved

    def faces(chapter):
        with ZipFile(io.BytesIO(client.get(chapter["export_url"]).content)) as archive:
            script = json.loads(archive.read("script.json"))
        return {(row["id"], face["expression"]) for row in script["characters"] for face in row.get("expressions", [])}

    with TestClient(create_app(coordinator=Coordinator(tmp_path))) as client:
        project, worker = approved(client)
        finish(client, worker)
        endpoint = f"/api/m3/projects/{project}/adjustments"
        usage = client.get(f"/api/expressions/projects/{project}/usage")
        assert usage.status_code == 200 and isinstance(usage.json()["characters"], dict)
        value = start(client, endpoint)
        person = next(row for row in value["cast"] if row["role"] == "main")
        chosen = next(row for row in value["candidates"] if row["id"] == person["image_candidate_id"])
        qwen = client.post("/api/workers", json={"name": "qwen", "capabilities": ["m2_outfit_image"]}).json()["id"]
        made = client.post(f"/api/expressions/projects/{project}", json={
            "base_artifact_id": chosen["artifact_id"], "character_id": person["character_id"], "expressions": ["angry"]})
        assert made.status_code == 200, made.text
        job = claim(client, qwen)
        assert complete_m2(client, job, qwen, expression_bundle(job)).status_code == 200
        before = production(client, project)["chapters"]
        assert all(not faces(chapter) for chapter in before)
        # Nothing in the draft changed, yet the chapters are built again to take the new picture.
        value = apply(client, endpoint, client.get(endpoint).json())
        assert value["draft"]["status"] == "applied"
        after = production(client, project)["chapters"]
        shown = [chapter for chapter in after if faces(chapter)]
        assert shown and all(faces(chapter) == {(script_character_id(person["character_id"]), "angry")} for chapter in shown)
