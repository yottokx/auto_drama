"""An event CG shows each person in one look and references that look's portrait."""

from unittest.mock import patch

from fastapi.testclient import TestClient

from packages.contracts.event_cg import CG_KINDS
from packages.contracts.m3 import M3_KINDS
from services.coordinator.app import create_app
from tests.integration.planning_fixtures import complete_and_approve_plan
from tests.integration.test_event_cg import cg_output, cg_result, published_script
from tests.integration.test_m2 import action, claim, png, ready, report_voice_inventory
from tests.integration.test_m2 import complete as complete_m2
from tests.integration.test_m2_outfits import LOOK, outfit_bundle, outfit_worker
from tests.integration.test_m3 import complete, narrative, output, production


def story_with_look(client):
    from services.coordinator.event_cg_settings import EventCgSettings

    project, _ = ready(client)
    identifier = project["project"]["id"]
    qwen = outfit_worker(client)
    action(client, project, "save-outfit", character_id="character-1", outfit=LOOK)
    action(client, project, "generate-outfit", character_id="character-1", outfit_id="look1")
    job = claim(client, qwen)
    assert complete_m2(client, job, qwen, outfit_bundle(job)).status_code == 200
    worker = client.post("/api/workers", json={
        "name": "CG fixture", "capabilities": [*sorted(set(M3_KINDS) | set(CG_KINDS)), "tts_download"]}).json()["id"]
    report_voice_inventory(client, worker)
    assert client.put(f"/api/event-cg/projects/{identifier}/policy", json={
        "expected_revision": 0, "max_cgs": 1, "max_variants_per_cg": 0}).status_code == 200
    action(client, project, "approve")
    freeze = EventCgSettings.freeze
    with patch.object(EventCgSettings, "freeze",
                      lambda *args, **kwargs: freeze(*args, **kwargs).model_copy(update={"planning_version": 1})):
        complete_and_approve_plan(client, project)
    return identifier, worker


def cg_plan(job, scene_index, first, end):
    scene = job["payload"]["context"]["narrative"]["scenes"][scene_index]
    lines = scene["utterances"]
    return {"source_sha256": job["payload"]["context"]["source_sha256"], "cgs": [{
        "id": "cg_one", "scene_id": scene["id"], "start_utterance_id": lines[first]["id"],
        "end_utterance_id": lines[end]["id"], "character_ids": scene["plan"]["character_ids"],
        "interpretation": "二人が記録を確かめる。", "prompt": "Two characters study an old record.", "variants": []}]}


def test_cg_cannot_cross_a_change_of_look_and_references_the_look_it_shows(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = story_with_look(client)
        image = None
        for _ in range(200):
            job = claim(client, worker)
            if not job:
                break
            data = None
            if job["kind"] == "m3_narrative":
                result = narrative(job["payload"]["approval_snapshot"])
                first, second = result["scenes"]
                first["plan"]["outfits"] = [
                    {"character_id": "character-1", "outfit_id": "default", "change_to": "look1"}]
                first["directions"].append({
                    "id": "scene-1-o1", "utterance_id": first["utterances"][3]["id"], "kind": "outfit",
                    "timing": "before", "character_id": "character-1", "position": None, "duration_ms": 400,
                    "outfit_id": "look1"})
                second["plan"]["outfits"] = [{"character_id": "character-1", "outfit_id": "look1"}]
                data = output(job, result)
            elif job["kind"] == "m3_event_cg_plan":
                crossing = complete(client, worker, job, cg_output(job, result=cg_plan(job, 0, 1, 5)))
                assert crossing.status_code == 422, crossing.text
                data = cg_output(job, result=cg_plan(job, 1, 0, 4))
            elif job["kind"] == "m3_event_cg":
                image = job
                data = cg_output(job)
            elif job["kind"] in CG_KINDS:
                data = cg_output(job, result=cg_result(job))
            response = complete(client, worker, job, data)
            assert response.status_code == 200, response.text
        assert image is not None
        main, support = image["payload"]["references"]
        assert (main["character_id"], main["outfit_id"]) == ("character-1", "look1")
        assert client.get(f"/api/artifacts/{main['artifact_id']}/content").content == png(200)
        assert support["outfit_id"] == "default"
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        script, _, _ = published_script(client, state["chapters"][0])
        assert len(script["event_cg_segments"]) == 1
