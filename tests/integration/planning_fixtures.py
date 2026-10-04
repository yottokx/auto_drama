"""Complete the real STEP4 approval boundary with deterministic planning data."""

import io
import json
from zipfile import ZipFile

from packages.contracts.planning import planning_protocol, validate_plan_content


def plan_content(snapshot):
    characters = [row["result"] for row in snapshot["characters"]]
    count = snapshot["world"]["result"]["chapterCount"]
    content = {
        "cast_plan": {"supporting_characters": [], "everyday_context": [], "connections": []},
        "plot": {
            "core": {
                "central_question": "記録を誰と守るか。",
                "external_resolution": "家族の記録を守る。",
                "relationship_resolution": "互いの判断を尊重する。",
                "characters": [{
                    "character_id": row["id"], "initial_behavior": "独力で記録を調べる。",
                    "enduring_value": "記録を尊重する。", "turning_experience": "協力を頼む。",
                    "final_behavior": "仲間と記録を守る。",
                } for row in characters],
                "foreshadowing": [],
            },
            "chapters": [{
                "number": number, "title": f"第{number}章", "role": "記録を確かめる。",
                "conversation_topics": [], "events": [],
                "route": {
                    "start_condition": "記録を調べる。", "attempt": "協力を頼む。",
                    "consequence": "相手の事情を知る。", "choice": "一緒に確かめる。",
                    "next_state": "次の記録を調べる。", "core_progress": "協力の経験を得る。",
                },
            } for number in range(1, count + 1)],
        },
    }
    return validate_plan_content(content, snapshot).model_dump(mode="json")


def complete_and_approve_plan(client, project, *, llm_models=None):
    project_id = project if isinstance(project, str) else project["project"]["id"]
    endpoint = f"/api/planning/projects/{project_id}"
    state = client.get(endpoint)
    assert state.status_code == 200, state.text
    planning = state.json()["planning"]
    if planning["content"] is None:
        worker = client.post("/api/workers", json={
            "name": "STEP4 planning fixture", "capabilities": ["m3_plan"],
            **({"llm_models": llm_models} if llm_models is not None else {}),
        }).json()["id"]
        response = client.post(f"/api/workers/{worker}/claim")
        assert response.status_code == 200, response.text
        job = response.json()["job"]
        assert job and job["kind"] == "m3_plan"
        content = plan_content(job["payload"]["approval_snapshot"])
        data = io.BytesIO()
        with ZipFile(data, "w") as archive:
            archive.writestr("result.json", json.dumps({
                "schema_version": 1, "kind": "m3_plan", "result": content,
                "provenance": {"seed": job["payload"]["seed"],
                               "planning_protocol": planning_protocol(),
                               "generator_protocol": planning_protocol()},
                "trace": [],
            }, ensure_ascii=False))
        response = client.post(f"/api/m3/jobs/{job['id']}/complete",
            params={"worker_id": worker, "lease_id": job["lease_id"]},
            content=data.getvalue(), headers={"Content-Type": "application/zip"})
        assert response.status_code == 200, response.text
        planning = client.get(endpoint).json()["planning"]
    response = client.post(endpoint + "/actions", json={
        "action": "approve", "expected_revision": planning["revision"],
    })
    assert response.status_code == 200, response.text
    return response.json()
