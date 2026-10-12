"""Shared materials: design sheets of the work, and the looks and portraits dressed in them."""
import io
import json
from zipfile import ZipFile

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from tests.integration.test_m2 import action, claim, complete, detail, png, ready
from tests.integration.test_m2_outfits import LOOK, outfit_bundle
from tests.integration.test_m3 import finish
from tests.integration.test_planning import endpoint, plan_action, planning
from tests.integration.test_planning_assets import assets, reviewed_with_support


def workers(client):
    return [client.post("/api/workers", json={"name": name, "capabilities": kinds}).json()["id"]
            for name, kinds in (("anima", ["m2_material_image"]), ("qwen", ["m2_outfit_image"]))]


def materials(client, project_id, **body):
    path = f"/api/materials/projects/{project_id}"
    response = client.post(path + "/actions", json=body) if body else client.get(path)
    assert response.status_code == 200, response.text
    return response.json()["materials"]


def sheet_bundle(job, summary="紺のブレザーとチェックのスカートの制服。"):
    payload = job["payload"]
    material = {"mode": payload["material_sheet"]["mode"], "category": "outfit",
                "outfit_en": "a navy blazer, a plaid skirt", "summary": summary}
    if "reference_image" in payload:
        material["reference_sha256"] = payload["reference_image"]["sha256"]
    data = io.BytesIO()
    with ZipFile(data, "w") as archive:
        archive.writestr("result.json", json.dumps({
            "schema_version": 1, "kind": job["kind"], "result": {}, "trace": [],
            "provenance": {"seed": payload["seed"], "material": material}}, ensure_ascii=False))
        archive.writestr("image.png", png(210))
    return data.getvalue()


def made_material(client, project_id, anima):
    materials(client, project_id, action="add-material", name="制服", description="紺のブレザー")
    job = claim(client, anima)
    assert job["kind"] == "m2_material_image" and job["payload"]["material_sheet"] == {
        "mode": "text", "name": "制服", "label": "", "wish": "紺のブレザー"}
    assert complete(client, job, anima, sheet_bundle(job)).status_code == 200
    (material,) = materials(client, project_id)
    return material


def test_sheets_are_drawn_varied_and_taken_from_a_portrait(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        project_id = project["project"]["id"]
        anima, qwen = workers(client)
        material = made_material(client, project_id, anima)
        (sheet,) = material["images"]
        assert sheet["status"] == "ready" and sheet["summary"].startswith("紺のブレザー")
        materials(client, project_id, action="add-image", material_id=material["id"], label="男子用",
                  wish="下はズボン", mode="derive", source={"image_id": sheet["id"]})
        job = claim(client, qwen)
        assert job["kind"] == "m2_outfit_image" and job["payload"]["material_sheet"] == {
            "mode": "derive", "name": "制服", "label": "男子用", "wish": "下はズボン",
            "base_outfit_en": "a navy blazer, a plaid skirt"}
        assert complete(client, job, qwen, sheet_bundle(job)).status_code == 200
        materials(client, project_id, action="add-image", material_id=material["id"], label="リオの服",
                  mode="extract", source={"character_id": "character-1"})
        job = claim(client, qwen)
        portrait = detail(client, project)["draft"]["characters"][0]["imageArtifactId"]
        assert job["payload"]["reference_image"]["artifact_id"] == portrait
        assert job["payload"]["material_sheet"]["appearance"]
        assert complete(client, job, qwen, sheet_bundle(job)).status_code == 200
        (material,) = materials(client, project_id)
        assert [image["status"] for image in material["images"]] == ["ready"] * 3
        materials(client, project_id, action="delete-image", image_id=material["images"][1]["id"])
        (material,) = materials(client, project_id)
        assert [image["label"] for image in material["images"]] == ["", "リオの服"]
        assert materials(client, project_id, action="delete-material", material_id=material["id"]) == []


def test_a_look_and_the_base_portrait_wear_a_chosen_sheet(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        project_id = project["project"]["id"]
        anima, qwen = workers(client)
        sheet = made_material(client, project_id, anima)["images"][0]
        action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1",
               material_image_id=sheet["id"])
        job = claim(client, qwen)
        assert job["payload"]["material_image"]["outfit_en"] == "a navy blazer, a plaid skirt"
        assert job["payload"]["outfit"]["id"] == "look1"
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        before = detail(client, project)["draft"]["characters"][0]
        # Without a sheet there is nothing to redraw the base look with.
        action(client, project, "generate-outfit", status=422, character_id="character-1", outfit_id="default")
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="default",
               material_image_id=sheet["id"])
        job = claim(client, qwen)
        assert job["payload"]["outfit"]["id"] == "default"
        assert job["payload"]["reference_image"]["artifact_id"] == before["imageArtifactId"]
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        after = detail(client, project)["draft"]["characters"][0]
        assert after["imageArtifactId"] != before["imageArtifactId"] and not after["imagePendingChanges"]
        # The look was derived from the old base, so it is drawn again, still wearing its sheet.
        job = claim(client, qwen)
        assert job["payload"]["outfit"]["id"] == "look1" and job["payload"]["material_image"]["image_id"] == sheet["id"]
        assert job["payload"]["reference_image"]["artifact_id"] == after["imageArtifactId"]
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        action(client, project, "approve")


def test_a_planned_supporting_character_is_dressed_before_approval(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, media = reviewed_with_support(client)
        project_id = project["project"]["id"]
        finish(client, media)
        anima, qwen = workers(client)
        sheet = made_material(client, project_id, anima)["images"][0]
        before = assets(client, project)["image"]
        response = client.post(endpoint(project) + "/assets/dress", json={
            "expected_revision": planning(client, project)["revision"], "character_id": "support-1",
            "material_image_id": sheet["id"]})
        assert response.status_code == 200, response.text
        # The finished portrait stays on show while it is redrawn; approval waits.
        pending = assets(client, project)["image"]
        assert pending["status"] == "pending" and pending["url"] == before["url"]
        plan_action(client, project, "approve", status=409)
        job = claim(client, qwen)
        assert job["payload"]["reference_image"]["artifact_id"] in before["url"]
        assert job["payload"]["material_image"]["image_id"] == sheet["id"]
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        after = assets(client, project)["image"]
        assert after["status"] == "ready" and after["url"] != before["url"]
        plan_action(client, project, "approve")


def test_material_declared_in_the_brief_is_drawn_with_the_world_and_worn_without_asking(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        from tests.integration.test_m2 import create, run_one, worker

        project, main = create(client, prompt="異星の物語を自由に考える。"), worker(client)
        project_id = project["project"]["id"]
        anima, qwen = workers(client)
        current = detail(client, project)
        saved = action(client, project, "save-brief", world=current["draft"]["worldInput"],
                       characters=[row["input"] for row in current["draft"]["characters"]],
                       materials=[{"id": "mat-a", "name": "制服", "description": "紺のブレザー"}],
                       outfits={"character-1": [{"id": "default", "name": "通常の姿", "material": "mat-a"},
                                                {"name": "私服"}]})
        assert saved["draft"]["materials"] == [{"id": "mat-a", "name": "制服", "description": "紺のブレザー"}]
        # Nothing is drawn before the world it belongs to exists.
        assert materials(client, project_id)[0]["images"] == [] and claim(client, anima) is None
        action(client, project, "generate-world")
        run_one(client, main, "m2_world")
        job = claim(client, anima)
        assert job["payload"]["material_sheet"]["name"] == "制服" and job["payload"]["world_result"]["title"]
        assert complete(client, job, anima, sheet_bundle(job)).status_code == 200
        action(client, project, "confirm-world")
        for kind in ("m2_character", "m2_image", "m2_voice"):
            run_one(client, main, kind)
        # The base look puts the shared design on first; the other look is derived from the result.
        job = claim(client, qwen)
        drawn = detail(client, project)["draft"]["characters"][0]["imageArtifactId"]
        assert job["payload"]["outfit"]["id"] == "default" and job["payload"]["material_image"]["outfit_en"]
        assert job["payload"]["reference_image"]["artifact_id"] == drawn
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        dressed = detail(client, project)["draft"]["characters"][0]["imageArtifactId"]
        job = claim(client, qwen)
        assert job["payload"]["outfit"]["name"] == "私服" and "material_image" not in job["payload"]
        assert job["payload"]["reference_image"]["artifact_id"] == dressed != drawn
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        assert claim(client, qwen) is None
        # Going back to the undressed portrait is a choice; it is not dressed again behind the user's back.
        action(client, project, "adopt-image", character_id="character-1", outfit_id="default", artifact_id=drawn)
        job = claim(client, qwen)
        assert job["payload"]["outfit"]["name"] == "私服" and job["payload"]["reference_image"]["artifact_id"] == drawn
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        assert claim(client, qwen) is None
        action(client, project, "approve")


def test_a_sheet_added_later_dresses_the_waiting_look(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        from tests.integration.test_m2 import create, run_one, worker

        project, main = create(client, prompt="異星の物語を自由に考える。"), worker(client)
        project_id = project["project"]["id"]
        qwen = client.post("/api/workers", json={"name": "qwen", "capabilities": ["m2_outfit_image"]}).json()["id"]
        current = detail(client, project)
        action(client, project, "save-brief", world=current["draft"]["worldInput"],
               characters=[row["input"] for row in current["draft"]["characters"]],
               materials=[{"id": "mat-a", "name": "制服"}],
               outfits={"character-1": [{"name": "体操服", "material": "mat-a"}]})
        action(client, project, "generate-world")
        run_one(client, main, "m2_world")
        action(client, project, "confirm-world")
        for kind in ("m2_character", "m2_image", "m2_voice"):
            run_one(client, main, kind)
        # No worker could draw the sheet with the world, so the look waits instead of being drawn bare.
        (material,) = materials(client, project_id)
        assert material["images"] == [] and claim(client, qwen) is None
        anima = client.post("/api/workers", json={"name": "anima", "capabilities": ["m2_material_image"]}).json()["id"]
        materials(client, project_id, action="add-image", material_id=material["id"], mode="text")
        job = claim(client, anima)
        assert complete(client, job, anima, sheet_bundle(job)).status_code == 200
        job = claim(client, qwen)
        assert job["payload"]["outfit"]["name"] == "体操服" and job["payload"]["material_image"]["outfit_en"]


def test_a_picture_is_edited_in_words_and_an_earlier_one_is_taken_back(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        _anima, qwen = workers(client)
        first = detail(client, project)["draft"]["characters"][0]["imageArtifactId"]
        action(client, project, "edit-image", status=422, character_id="character-1", outfit_id="default")
        action(client, project, "edit-image", character_id="character-1", outfit_id="default", instruction="髪を短く")
        job = claim(client, qwen)
        assert job["payload"]["edit"] == {"instruction": "髪を短く"} and job["payload"]["outfit"]["id"] == "default"
        assert job["payload"]["reference_image"]["artifact_id"] == first
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        character = detail(client, project)["draft"]["characters"][0]
        edited = character["imageArtifactId"]
        assert edited != first
        history = detail(client, project)["imageHistory"]["character-1"]
        assert [row["artifactId"] for row in history] == [edited, first]
        # A look edited by hand from a rejected picture stays when the base portrait changes.
        action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        job = claim(client, qwen)
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        action(client, project, "edit-image", character_id="character-1", outfit_id="look1",
               instruction="帯を赤に", base_artifact_id=first)
        job = claim(client, qwen)
        assert job["payload"]["reference_image"]["artifact_id"] == first
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        look = detail(client, project)["draft"]["characters"][0]["outfits"][-1]
        assert look["handPicked"] and not look["imagePendingChanges"]
        action(client, project, "adopt-image", character_id="character-1", outfit_id="default", artifact_id=first)
        character = detail(client, project)["draft"]["characters"][0]
        assert character["imageArtifactId"] == first and claim(client, qwen) is None
        assert not character["outfits"][-1]["imagePendingChanges"]
        action(client, project, "approve")
        action(client, project, "adopt-image", status=404, character_id="character-1", outfit_id="default",
               artifact_id="0" * 32)


def editing(client, project_id):
    response = client.get(f"/api/image-editing/projects/{project_id}")
    assert response.status_code == 200, response.text
    return response.json()


def test_edited_pictures_are_listed_and_a_work_can_forgo_the_editing_model(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        project_id = project["project"]["id"]
        _anima, qwen = workers(client)
        assert editing(client, project_id) == {
            "project_id": project_id, "enabled": True, "model": "Qwen-Image-2.1", "caution": True,
            "portraits": [], "event_cgs": 0}
        action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
        action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
        job = claim(client, qwen)
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        action(client, project, "edit-image", character_id="character-1", outfit_id="default", instruction="髪を短く")
        job = claim(client, qwen)
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        job = claim(client, qwen)
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        name = detail(client, project)["draft"]["characters"][0]["result"]["name"]
        assert editing(client, project_id)["portraits"] == [
            {"name": name, "look": "通常の姿"}, {"name": name, "look": "浴衣"}]
        # Without the editing model the look is neither required nor part of the approval.
        off = action(client, project, "set-image-editing", enabled=False)
        assert off["draft"]["imageEditing"] is False and not editing(client, project_id)["enabled"]
        action(client, project, "generate-outfit", status=409, character_id="character-1", outfit_id="look1")
        action(client, project, "edit-image", status=409, character_id="character-1", outfit_id="default",
               instruction="髪を長く")
        approved = action(client, project, "approve")
        assert "outfits" not in approved["draft"]["approval"]["characters"][0]
        # The edited base portrait is still in use, and the list keeps saying so.
        assert editing(client, project_id)["portraits"][0] == {"name": name, "look": "通常の姿"}


def test_the_editing_model_can_be_forgone_before_the_world_exists(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        from tests.integration.test_m2 import create

        project = create(client, prompt="異星の物語を自由に考える。")
        off = action(client, project, "set-image-editing", enabled=False)
        assert off["draft"]["imageEditing"] is False and off["draft"]["step"] == "world-input"


def test_a_work_without_the_editing_model_has_no_event_cgs_even_when_some_are_allowed(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        from tests.integration.planning_fixtures import complete_and_approve_plan

        project, _ = ready(client)
        project_id = project["project"]["id"]
        saved = client.put(f"/api/event-cg/projects/{project_id}/policy",
                           json={"max_cgs": 2, "max_variants_per_cg": 0, "expected_revision": 0})
        assert saved.status_code == 200, saved.text
        action(client, project, "set-image-editing", enabled=False)
        action(client, project, "approve")
        # With CGs allowed and no editing worker, approval would stop; forgoing the model lifts that.
        complete_and_approve_plan(client, project)
        with client.app.state.coordinator.db.transaction() as connection:
            assert connection.execute("SELECT COUNT(*) FROM event_cg_production").fetchone()[0] == 0
            kinds = {row[0] for row in connection.execute("SELECT kind FROM job WHERE project_id=?", (project_id,))}
        assert not {kind for kind in kinds if "event_cg" in kind}


def test_drawing_a_sheet_again_keeps_its_design_until_the_wish_changes(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        project_id = project["project"]["id"]
        anima, _qwen = workers(client)

        def draw(**body):
            materials(client, project_id, **body)
            job = claim(client, anima)
            data = sheet_bundle(job)
            # A worker of the current design wording marks what it decided.
            with ZipFile(io.BytesIO(data)) as archive:
                envelope = json.loads(archive.read("result.json"))
            envelope["provenance"]["material"].update(version=2, change_en="")
            packed = io.BytesIO()
            with ZipFile(packed, "w") as archive:
                archive.writestr("result.json", json.dumps(envelope, ensure_ascii=False))
                archive.writestr("image.png", png(210))
            assert complete(client, job, anima, packed.getvalue()).status_code == 200
            return job["payload"]["material_sheet"]

        first = draw(action="add-material", name="制服", description="紺のブレザー")
        assert "design" not in first
        (material,) = materials(client, project_id)
        again = draw(action="retake-image", image_id=material["images"][0]["id"])
        assert again["design"]["outfit_en"] == "a navy blazer, a plaid skirt" and again["design"]["version"] == 2
        changed = draw(action="retake-image", image_id=material["images"][0]["id"], wish="緑のブレザーに")
        assert changed["design"]["version"] == 0 and changed["wish"] == "緑のブレザーに"
        materials(client, project_id, action="save-material", material_id=material["id"], name="制服", description="紺のブレザー")
        materials(client, project_id, action="add-image", material_id=material["id"], label="夏服", mode="text")
        claim(client, anima)


def test_a_world_made_again_redraws_its_materials_and_dresses_the_cast_in_them(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        from tests.integration.test_m2 import create, run_one, worker

        project, main = create(client, prompt="異星の物語を自由に考える。"), worker(client)
        project_id = project["project"]["id"]
        anima, qwen = workers(client)
        current = detail(client, project)
        action(client, project, "save-brief", world=current["draft"]["worldInput"],
               characters=[row["input"] for row in current["draft"]["characters"]],
               materials=[{"id": "mat-a", "name": "制服"}],
               outfits={"character-1": [{"id": "default", "name": "通常の姿", "material": "mat-a"}]})
        action(client, project, "generate-world")
        run_one(client, main, "m2_world")
        job = claim(client, anima)
        assert complete(client, job, anima, sheet_bundle(job)).status_code == 200
        first_sheet = materials(client, project_id)[0]["images"][0]["url"]
        action(client, project, "confirm-world")
        for kind in ("m2_character", "m2_image", "m2_voice"):
            run_one(client, main, kind)
        plain = detail(client, project)["draft"]["characters"][0]["imageArtifactId"]
        job = claim(client, qwen)
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        # Revising the world in words keeps the sheet that was already looked at.
        action(client, project, "generate-world", instruction="もう少し明るい雰囲気に。")
        run_one(client, main, "m2_world")
        assert claim(client, anima) is None
        # Making the world again from the brief makes its materials again, without being asked.
        action(client, project, "generate-world")
        run_one(client, main, "m2_world")
        job = claim(client, anima)
        assert job["payload"]["material_sheet"]["name"] == "制服"
        assert complete(client, job, anima, sheet_bundle(job)).status_code == 200
        assert materials(client, project_id)[0]["images"][0]["url"] != first_sheet
        # A sheet drawn again by hand is put on the portrait again, from the portrait as it was drawn.
        materials(client, project_id, action="retake-image", image_id=materials(client, project_id)[0]["images"][0]["id"])
        job = claim(client, anima)
        assert complete(client, job, anima, sheet_bundle(job)).status_code == 200
        action(client, project, "confirm-world")
        while job := claim(client, main):
            assert complete(client, job, main).status_code == 200
        job = claim(client, qwen)
        assert job["payload"]["outfit"]["id"] == "default" and job["payload"]["material_image"]["artifact_id"] in \
            materials(client, project_id)[0]["images"][0]["url"]
        assert complete(client, job, qwen, outfit_bundle(job)).status_code == 200
        materials(client, project_id, action="retake-image", image_id=materials(client, project_id)[0]["images"][0]["id"])
        job = claim(client, anima)
        assert complete(client, job, anima, sheet_bundle(job)).status_code == 200
        job = claim(client, qwen)
        assert job["payload"]["outfit"]["id"] == "default"
        assert job["payload"]["reference_image"]["artifact_id"] == plain
