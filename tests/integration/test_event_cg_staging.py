"""New productions freeze staging v2; omission never extends an unsafe picture."""

import copy
import json

import pytest
from fastapi.testclient import TestClient

from packages.contracts.event_cg import CG_KINDS, EventCgPolicy, validate_plan
from packages.narrative.validation import parse_scene_text
from services.coordinator.app import create_app
from tests.integration.test_event_cg import cg_output, cg_result, published_script, setup_story
from tests.integration.test_m2 import claim
from tests.integration.test_m3 import complete, narrative, output, production


def long_narrative(snapshot):
    value = narrative(snapshot)
    for scene in value["scenes"]:
        scene["raw_text"] += "\n" + "\n".join(
            f"NARRATOR: 二人は記録の第{index}項を机に置いたまま、静かに読み続けた。"
            for index in range(1, 11))
        scene["utterances"] = [line.model_dump() for line in parse_scene_text(
            scene["raw_text"], scene["id"], set(scene["plan"]["character_ids"]))]
    return value


def staged_plan(job, *, base_safe=5, first_safe=10):
    payload = job["payload"]
    scene = payload["context"]["narrative"]["scenes"][0]
    lines = scene["utterances"]

    def staging(start, end, change=""):
        return {"visual_state": "二人は机を挟んで座り、記録を読んでいる。",
                "evidence_utterance_ids": [lines[start]["id"]],
                "safe_end_utterance_id": lines[end]["id"],
                "reason": "二人の受け止め方が変わる区間を見せる。", "change": change}

    return {"planning_version": 2, "source_sha256": payload["context"]["source_sha256"], "cgs": [{
        "id": "cg_one", "scene_id": scene["id"], "start_utterance_id": lines[0]["id"],
        "end_utterance_id": lines[15]["id"], "character_ids": scene["plan"]["character_ids"],
        "interpretation": "二人が記録を確かめる。", "prompt": "Two people seated at a library table.",
        "composition": "机を挟んだ二人の横顔。", "end_reason": "記録を置いて席を立つ。",
        "end_evidence_utterance_id": lines[15]["id"], "staging": staging(0, base_safe),
        "variants": [
            {"id": "variant_one", "start_utterance_id": lines[5]["id"],
             "interpretation": "記録を受け入れる。", "prompt": "Their tense expressions soften.",
             "staging": staging(5, first_safe, "拒絶から受容への変化。")},
            {"id": "variant_two", "start_utterance_id": lines[10]["id"],
             "interpretation": "記録を読んで涙を流す。", "prompt": "They quietly cry while reading.",
             "staging": staging(10, 15, "堪えていた涙がこぼれる。")},
        ],
    }], "planning_notes": [{"cg_id": "cg_one", "variant_id": "discarded_reaction",
                              "reason": "一言だけの反応は差分にしない。"}]}


def finish_staged(client, worker, *, stop_at=None, omit=()):
    for _ in range(300):
        job = claim(client, worker)
        if not job or (stop_at and stop_at(job)):
            return job
        if job["kind"] == "m3_narrative":
            data = output(job, result=long_narrative(job["payload"]["approval_snapshot"]))
        elif job["kind"] in CG_KINDS:
            result = staged_plan(job) if job["kind"] == "m3_event_cg_plan" else cg_result(job)
            data = cg_output(job, result=result, omitted=(
                job["kind"] == "m3_event_cg" and job["payload"].get("variant_id") in omit))
        else:
            data = None
        response = complete(client, worker, job, data)
        assert response.status_code == 200, response.text
    raise AssertionError("Staging production did not settle")


def test_new_production_freezes_v2_and_rejects_short_images_before_publication(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client, legacy=False, max_variants_per_cg=2)
        plan_job = finish_staged(client, worker, stop_at=lambda job: job["kind"] == "m3_event_cg_plan")
        assert plan_job["payload"]["policy"]["planning_version"] == 2
        result = staged_plan(plan_job)
        short = copy.deepcopy(result)
        lines = plan_job["payload"]["context"]["narrative"]["scenes"][0]["utterances"]
        short["cgs"][0]["variants"][0]["start_utterance_id"] = lines[2]["id"]
        with pytest.raises(ValueError, match="at least 5 utterances"):
            validate_plan(short, plan_job["payload"])
        rejected = complete(client, worker, plan_job, cg_output(plan_job, result=short))
        assert rejected.status_code == 422
        legacy = copy.deepcopy(result)
        legacy.pop("planning_version")
        rejected = complete(client, worker, plan_job, cg_output(plan_job, result=legacy))
        assert rejected.status_code == 422
        assert complete(client, worker, plan_job, cg_output(plan_job, result=result)).status_code == 200
        finish_staged(client, worker)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        expected = result["cgs"][0]
        for summary in (state["event_cg"], state["chapters"][0]["event_cg"]):
            detail, = summary["plans"]
            assert detail["chapter_number"] == 1 and detail["planning_version"] == 2
            assert detail["start_reason"] == expected["staging"]["reason"]
            assert detail["end_reason"] == expected["end_reason"]
            assert detail["composition"] == expected["composition"]
            assert [image["utterance_count"] for image in detail["images"]] == [5, 5, 5]
            assert [image["character_count"] for image in detail["images"]] == [
                sum(len(row["display_text"]) for row in lines[start:start + 5]) for start in (0, 5, 10)]
            assert detail["images"][1]["visual_change"] == expected["variants"][0]["staging"]["change"]
            assert summary["planning_notes"] == [{"chapter_number": 1, **result["planning_notes"][0]}]
        script, _, _ = published_script(client, state["chapters"][0])
        assert len(script["event_cg_segments"][0]["variants"]) == 2


@pytest.mark.parametrize("base_safe,first_safe,omitted,end,variants", [
    (5, 10, ("variant_one",), 5, []),  # Cannot jump across an unsafe gap to the successful second variant.
    (10, 10, ("variant_one",), 15, ["variant_two"]),  # Safe boundary exactly reaches the later anchor.
    (5, 10, ("variant_two",), 10, ["variant_one"]),  # The first successful variant has its own safe end.
    (15, 10, ("variant_one", "variant_two"), 15, []),  # Base state can safely cover the complete CG.
])
def test_missing_variants_stop_or_bridge_only_within_safe_state(
        tmp_path, base_safe, first_safe, omitted, end, variants):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client, legacy=False, max_variants_per_cg=2)
        job = finish_staged(client, worker, stop_at=lambda job: job["kind"] == "m3_event_cg_plan")
        result = staged_plan(job, base_safe=base_safe, first_safe=first_safe)
        response = complete(client, worker, job, cg_output(job, result=result))
        assert response.status_code == 200, response.text
        finish_staged(client, worker, omit=omitted)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        script, names, _ = published_script(client, state["chapters"][0])
        segment, = script["event_cg_segments"]
        lines = job["payload"]["context"]["narrative"]["scenes"][0]["utterances"]
        assert segment["end_utterance_id"] == lines[end]["id"]
        assert [variant["id"] for variant in segment["variants"]] == variants
        assert len([name for name in names if name.startswith("data/cgimage/")]) == 1 + len(variants)
        assert state["event_cg"]["omitted"] == len(omitted)


def test_old_frozen_policy_without_version_stays_legacy_when_resumed(tmp_path, monkeypatch):
    with TestClient(create_app(tmp_path)) as client:
        serialize = EventCgPolicy.model_dump_json

        def old_policy_json(self, **kwargs):
            value = json.loads(serialize(self, **kwargs))
            value.pop("planning_version")
            return json.dumps(value)

        with monkeypatch.context() as legacy:
            # Reproduce the historical serialized policy at creation time;
            # production rows remain immutable throughout the test.
            legacy.setattr(EventCgPolicy, "model_dump_json", old_policy_json)
            project, worker = setup_story(client, legacy=True)
        with client.app.state.coordinator.db.transaction() as connection:
            row = connection.execute("SELECT * FROM event_cg_production").fetchone()
            assert "planning_version" not in json.loads(row["policy"])
        assert client.post(f"/api/m3/projects/{project}/stop", json={"mode": "immediate"}).status_code == 200
        assert client.post(f"/api/m3/projects/{project}/resume").status_code == 200
        job = finish_staged(client, worker, stop_at=lambda job: job["kind"] == "m3_event_cg_plan")
        assert job["payload"]["policy"]["planning_version"] == 1
        old_plan = cg_result(job)
        assert complete(client, worker, job, cg_output(job, result=old_plan)).status_code == 200
