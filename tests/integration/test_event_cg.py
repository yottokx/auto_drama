"""Optional CG production uses immutable plans and adopted image dependencies."""

import copy
import io
import json
from unittest.mock import patch
from zipfile import ZipFile

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from packages.contracts.event_cg import CG_KINDS, EventCgProfile
from packages.contracts.m3 import M3_KINDS
from services.coordinator.app import create_app
from tests.integration.planning_fixtures import complete_and_approve_plan
from tests.integration.test_m2 import action, claim, create, ready, report_voice_inventory
from tests.integration.test_m3 import complete, production
from tests.integration.test_project_history import history, restore


def setup_story(client, *, enabled=True, max_cgs=1, max_variants_per_cg=1, legacy=True):
    """Existing short-CG cases replay a policy frozen before staging version 2."""
    project, _ = ready(client)
    identifier = project["project"]["id"]
    kinds = (set(M3_KINDS) | set(CG_KINDS)) if enabled else (set(M3_KINDS) - set(CG_KINDS))
    response = client.post("/api/workers", json={"name": "CG fixture", "capabilities": [*sorted(kinds), "tts_download"]})
    assert response.status_code == 201, response.text
    worker = response.json()["id"]
    report_voice_inventory(client, worker)
    if enabled:
        response = client.put(f"/api/event-cg/projects/{identifier}/policy",
                              json={"expected_revision": 0, "max_cgs": max_cgs,
                                    "max_variants_per_cg": max_variants_per_cg})
        assert response.status_code == 200, response.text
    action(client, project, "approve")
    if legacy:
        from services.coordinator.event_cg_settings import EventCgSettings

        freeze = EventCgSettings.freeze

        def legacy_freeze(*args, **kwargs):
            return freeze(*args, **kwargs).model_copy(update={"planning_version": 1})

        with patch.object(EventCgSettings, "freeze", legacy_freeze):
            complete_and_approve_plan(client, project)
    else:
        complete_and_approve_plan(client, project)
    return identifier, worker


def cg_result(job):
    payload = job["payload"]
    if job["kind"] == "m3_event_cg_budget":
        return {"source_sha256": payload["context"]["source_sha256"], "chapters": [
            {"chapter_number": row["number"], "limit": payload["policy"]["max_cgs"] if index == 0 else 0,
             "reason": "最初の選択を描く。"}
            for index, row in enumerate(payload["context"]["overall_plot"]["chapters"])]}
    if job["kind"] == "m3_event_cg_plan":
        scene = payload["context"]["narrative"]["scenes"][0]
        lines = scene["utterances"]
        return {"source_sha256": payload["context"]["source_sha256"], "cgs": [{
            "id": "cg_one", "scene_id": scene["id"], "start_utterance_id": lines[0]["id"],
            "end_utterance_id": lines[4]["id"], "character_ids": scene["plan"]["character_ids"],
            "interpretation": "二人が記録を確かめる。", "prompt": "Two characters study an old record in a library.",
            "variants": [{"id": "cg_one_variant", "start_utterance_id": lines[2]["id"],
                          "interpretation": "表情が柔らかくなる。", "prompt": "Keep the composition. Both characters smile."}],
        }]}
    return {"input_sha256": payload["input_sha256"], "cg_id": payload["cg_id"],
            "variant_id": payload.get("variant_id"), "status": "complete"}


def cg_output(job, *, result=None, omitted=False):
    result = copy.deepcopy(result if result is not None else cg_result(job))
    if omitted:
        result.update(status="omitted", reason="fixture: out of memory")
    buffer = io.BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("result.json", json.dumps({"schema_version": 1, "kind": job["kind"],
            "result": result, "provenance": {"backend": "fixture", "private_runtime_path": "C:/private/qwen"},
            "trace": []}))
        if job["kind"] == "m3_event_cg" and not omitted:
            profile = job["payload"]["cg_profile"]
            image = io.BytesIO()
            Image.new("RGB", (profile["width"], profile["height"]), (90, 50, 160)).save(image, format="PNG")
            archive.writestr("image.png", image.getvalue())
            archive.writestr("original.png", image.getvalue())
    return buffer.getvalue()


def finish(client, worker, *, omit=None, stop_at=None):
    jobs = []
    for _ in range(200):
        job = claim(client, worker)
        if not job:
            return jobs, None
        if stop_at and stop_at(job):
            return jobs, job
        data = cg_output(job, omitted=bool(omit and omit(job))) if job["kind"] in CG_KINDS else None
        response = complete(client, worker, job, data)
        assert response.status_code == 200, response.text
        jobs.append(job)
    raise AssertionError("CG production did not settle")


def published_script(client, chapter):
    response = client.get(chapter["export_url"])
    assert response.status_code == 200, response.text
    with ZipFile(io.BytesIO(response.content)) as archive:
        return json.loads(archive.read("script.json")), archive.namelist(), response.content


def test_settings_defaults_readiness_revision_and_bounds(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project = create(client)["project"]["id"]
        settings = client.get("/api/event-cg/settings").json()
        assert settings["ready"] is False and settings["profile"] == EventCgProfile().model_dump(mode="json")
        endpoint = f"/api/event-cg/projects/{project}/policy"
        assert client.get(endpoint).json() == {"max_cgs": 0, "max_variants_per_cg": 0, "revision": 0}
        request = {"expected_revision": 0, "max_cgs": 2, "max_variants_per_cg": 1}
        assert client.put(endpoint, json=request).json()["revision"] == 1
        assert client.put(endpoint, json=request).status_code == 409
        assert client.put(endpoint, json={**request, "expected_revision": 1, "max_cgs": -1}).status_code == 422
        assert client.put(endpoint, json={**request, "expected_revision": 1, "max_variants_per_cg": 11}).status_code == 422
        profile = {"expected_revision": 0, "profile": {**settings["profile"], "steps": 30}}
        assert client.put("/api/event-cg/settings", json=profile).json()["revision"] == 1
        assert client.put("/api/event-cg/settings", json=profile).status_code == 409


def test_settings_hide_disconnected_history_and_keep_live_same_name_workers(tmp_path, monkeypatch):
    app = create_app(tmp_path)
    with TestClient(app) as client:
        now = [1000.0]
        monkeypatch.setattr(app.state.coordinator, "clock", lambda: now[0])

        def register(kinds):
            response = client.post("/api/workers", json={"name": "local", "capabilities": sorted(kinds)})
            assert response.status_code == 201, response.text
            return response.json()["id"]

        old_ids = [register(CG_KINDS), register({"tyrano_export"})]
        now[0] += 100
        unavailable = register({"tyrano_export"})
        available = register(CG_KINDS)
        endpoint = "/api/event-cg/settings"
        settings = client.get(endpoint).json()
        assert settings["ready"] is True
        assert settings["workers"] == [
            {"id": available, "name": "local", "ready": True},
            {"id": unavailable, "name": "local", "ready": False},
        ]
        now[0] += 90
        assert len(client.get(endpoint).json()["workers"]) == 2
        now[0] += 0.01
        settings = client.get(endpoint).json()
        assert settings["workers"] == [] and settings["ready"] is False
        with app.state.coordinator.db.transaction() as connection:
            assert {row["id"] for row in connection.execute("SELECT id FROM worker")} == {
                *old_ids, unavailable, available,
            }


def test_disabled_production_never_checks_qwen_or_adds_jobs(tmp_path, monkeypatch):
    from services.coordinator.event_cg_settings import EventCgSettings

    def forbidden(*_args, **_kwargs):
        raise AssertionError("disabled CG must not inspect Qwen availability")

    monkeypatch.setattr(EventCgSettings, "settings", forbidden)
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client, enabled=False)
        jobs, _ = finish(client, worker)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        assert not any(job["kind"] in CG_KINDS for job in jobs)
        for chapter in state["chapters"]:
            script, names, _ = published_script(client, chapter)
            assert "event_cg_segments" not in script
            assert not any(name.startswith("data/cgimage/") for name in names)


def test_approved_production_pins_policy_and_generates_base_then_variant_once(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client)
        _, base = finish(client, worker, stop_at=lambda job: job["kind"] == "m3_event_cg")
        assert base and base["payload"]["variant_id"] is None
        assert all(row["role"] == "character" for row in base["payload"]["references"])
        assert len(base["payload"]["references"]) == 2
        data = cg_output(base)
        first = complete(client, worker, base, data)
        assert first.status_code == 200, first.text
        assert complete(client, worker, base, data).status_code == 200
        assert complete(client, worker, base, data + b"changed").status_code == 409
        client.put(f"/api/event-cg/projects/{project}/policy",
                   json={"expected_revision": 1, "max_cgs": 0, "max_variants_per_cg": 0})
        settings = client.get("/api/event-cg/settings").json()
        client.put("/api/event-cg/settings", json={"expected_revision": settings["revision"],
                   "profile": {**settings["profile"], "steps": 20}})
        remaining, _ = finish(client, worker)
        variants = [job for job in remaining if job["kind"] == "m3_event_cg"]
        assert len(variants) == 1 and variants[0]["payload"]["variant_id"] == "cg_one_variant"
        assert variants[0]["payload"]["cg_profile"]["steps"] == 40
        assert variants[0]["payload"]["references"][0]["role"] == "base_cg"
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        script, names, bundle = published_script(client, state["chapters"][0])
        assert len(script["event_cg_segments"]) == 1
        assert len(script["event_cg_segments"][0]["variants"]) == 1
        assert len([name for name in names if name.startswith("data/cgimage/")]) == 2
        assert b"C:/private/qwen" not in bundle
        assert claim(client, worker) is None


@pytest.mark.parametrize("last_chapter_limit", [0, 1])
def test_budget_precedes_narrative_and_unwritten_chapters_show_their_allocation(tmp_path, last_chapter_limit):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client, max_cgs=2)
        before = production(client, project)
        assert before["narrative_artifact_id"] is None
        assert before["event_cg"]["budget_completed"] is False
        assert before["event_cg"]["plan_completed"] is False
        assert before["event_cg"]["chapter_budgets"] == []
        budget = claim(client, worker)
        assert budget and budget["kind"] == "m3_event_cg_budget"
        assert "narrative" not in budget["payload"]["context"]
        assert budget["payload"]["context"]["overall_plot"] == budget["payload"]["approved_plan"]["content"]["plot"]
        with client.app.state.coordinator.db.transaction() as connection:
            assert not connection.execute("SELECT 1 FROM job_dependency WHERE job_id=?", (budget["id"],)).fetchone()
            writing = connection.execute("SELECT status FROM job WHERE project_id=? AND kind='m3_narrative'",
                                         (project,)).fetchall()
            assert [row["status"] for row in writing] == ["pending"]

        result = cg_result(budget)
        limits = [2 - last_chapter_limit, 0, last_chapter_limit]
        for row, limit in zip(result["chapters"], limits, strict=True):
            row["limit"] = limit
        response = complete(client, worker, budget, cg_output(budget, result=result))
        assert response.status_code == 200, response.text
        allocated = production(client, project)
        assert allocated["narrative_artifact_id"] is None
        assert allocated["event_cg"]["budget_completed"] is True
        assert allocated["event_cg"]["chapter_budgets"] == [
            {"chapter_number": number, "limit": limit} for number, limit in enumerate(limits, 1)]
        assert [chapter["event_cg"]["chapter_budget"] for chapter in allocated["chapters"]] == limits
        assert [chapter["event_cg"]["plan_completed"] for chapter in allocated["chapters"]] == [
            False, True, last_chapter_limit == 0]
        assert allocated["event_cg"]["plan_completed"] is False

        narrative = claim(client, worker)
        assert narrative and narrative["kind"] == "m3_narrative"
        assert complete(client, worker, narrative).status_code == 200
        _, plan = finish(client, worker, stop_at=lambda job: job["kind"] == "m3_event_cg_plan")
        assert plan and plan["payload"]["chapter_number"] == 1
        assert complete(client, worker, plan, cg_output(plan)).status_code == 200
        planned = production(client, project)
        assert planned["chapters"][0]["event_cg"]["plan_completed"] is True
        assert planned["event_cg"]["plan_completed"] is (last_chapter_limit == 0)
        if last_chapter_limit:
            _, later_plan = finish(client, worker, stop_at=lambda job: job["kind"] == "m3_event_cg_plan")
            assert later_plan and later_plan["payload"]["chapter_number"] == 3
            assert complete(client, worker, later_plan, cg_output(later_plan)).status_code == 200
            assert production(client, project)["event_cg"]["plan_completed"] is True


@pytest.mark.parametrize("interrupted", [False, True])
def test_resume_upgrades_preallocation_production_without_changing_frozen_inputs(tmp_path, monkeypatch, interrupted):
    from services.coordinator import event_cg_service

    with TestClient(create_app(tmp_path)) as client:
        # Reproduce a production approved by the version that waited for the
        # narrative before creating its CG budget requirement.
        with monkeypatch.context() as old_version:
            old_version.setattr(event_cg_service, "ensure_budget", lambda *_args: None)
            project, worker = setup_story(client)
            if interrupted:
                response = client.post(f"/api/m3/projects/{project}/stop", json={"mode": "immediate"})
                assert response.status_code == 200, response.text
            with client.app.state.coordinator.db.transaction() as connection:
                writing = dict(connection.execute("SELECT * FROM job WHERE project_id=? AND kind='m3_narrative'",
                                                  (project,)).fetchone())
                frozen = dict(connection.execute("SELECT * FROM event_cg_production").fetchone())
                assert not connection.execute("SELECT 1 FROM m3_requirement WHERE kind='m3_event_cg_budget'").fetchone()

        if interrupted:
            # Inspection and worker polling cannot start the migration or resume
            # a production the user explicitly interrupted.
            paused = production(client, project)
            assert paused["control_state"] == "interrupted"
            assert paused["event_cg"]["budget_completed"] is False
            assert claim(client, worker) is None
            with client.app.state.coordinator.db.transaction() as connection:
                assert not connection.execute("SELECT 1 FROM m3_requirement WHERE kind='m3_event_cg_budget'").fetchone()

        assert client.put(f"/api/event-cg/projects/{project}/policy", json={
            "expected_revision": 1, "max_cgs": 0, "max_variants_per_cg": 0}).status_code == 200
        settings = client.get("/api/event-cg/settings").json()
        assert client.put("/api/event-cg/settings", json={"expected_revision": settings["revision"],
            "profile": {**settings["profile"], "steps": 20}}).status_code == 200
        response = client.post(f"/api/m3/projects/{project}/resume")
        assert response.status_code == 200, response.text
        budget = claim(client, worker)
        assert budget and budget["kind"] == "m3_event_cg_budget"
        assert budget["payload"]["policy"] == json.loads(frozen["policy"])
        assert budget["payload"]["policy"]["generation_profile"]["steps"] == 40
        with client.app.state.coordinator.db.transaction() as connection:
            unchanged = dict(connection.execute("SELECT * FROM job WHERE id=?", (writing["id"],)).fetchone())
            assert unchanged["payload"] == writing["payload"]
            assert unchanged["settings_snapshot"] == writing["settings_snapshot"]
            assert unchanged["status"] == "pending"
            assert not connection.execute("SELECT 1 FROM job_dependency WHERE job_id=?", (budget["id"],)).fetchone()
        assert complete(client, worker, budget, cg_output(budget)).status_code == 200
        next_job = claim(client, worker)
        assert next_job and next_job["id"] == writing["id"]


@pytest.mark.parametrize("scope", ["base", "variant"])
def test_omission_finishes_publication_without_regenerating_successful_images(tmp_path, scope):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client)
        jobs, _ = finish(client, worker, omit=lambda job: job["kind"] == "m3_event_cg" and
                         ((job["payload"].get("variant_id") is None) == (scope == "base")))
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        images = [job for job in jobs if job["kind"] == "m3_event_cg"]
        script, _, _ = published_script(client, state["chapters"][0])
        if scope == "base":
            assert len(images) == 1 and not script.get("event_cg_segments")
        else:
            assert len(images) == 2 and script["event_cg_segments"][0]["variants"] == []
        assert state["event_cg"]["omitted"] >= 1
        assert claim(client, worker) is None


def test_plan_budget_and_unknown_anchor_rejected_before_images(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client)
        _, job = finish(client, worker, stop_at=lambda value: value["kind"] == "m3_event_cg_plan")
        assert job
        result = cg_result(job)
        invalid = copy.deepcopy(result)
        invalid["cgs"][0]["end_utterance_id"] = "unknown_line"
        assert complete(client, worker, job, cg_output(job, result=invalid)).status_code == 422
        invalid = copy.deepcopy(result)
        invalid["cgs"].append({**copy.deepcopy(invalid["cgs"][0]), "id": "over_budget"})
        assert complete(client, worker, job, cg_output(job, result=invalid)).status_code == 422
        state = production(client, project)
        assert not any(row["kind"] == "m3_event_cg" for row in state["jobs"])
        assert complete(client, worker, job, cg_output(job)).status_code == 200
        finish(client, worker)
        assert production(client, project)["status"] == "published"


def test_graceful_stop_and_resume_keep_adopted_base_and_history_does_not_generate(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client)
        _, base = finish(client, worker, stop_at=lambda job: job["kind"] == "m3_event_cg")
        assert complete(client, worker, base, cg_output(base)).status_code == 200
        assert client.post(f"/api/m3/projects/{project}/stop", json={"mode": "graceful"}).status_code == 200
        assert claim(client, worker) is None
        paused = history(client, project)["current_revision_id"]
        assert client.post(f"/api/m3/projects/{project}/resume").status_code == 200
        jobs, _ = finish(client, worker)
        assert not any(job["kind"] == "m3_event_cg" and job["payload"].get("variant_id") is None for job in jobs)
        completed = production(client, project)
        assert completed["status"] == "published"
        done_history = history(client, project)["current_revision_id"]
        restore(client, project, paused)
        assert production(client, project)["history_frozen"] and claim(client, worker) is None
        restore(client, project, done_history)
        assert production(client, project)["build"] == completed["build"] and claim(client, worker) is None


@pytest.mark.parametrize("scope", ["budget", "plan"])
def test_zero_budget_or_plan_publishes_with_recorded_reason_and_no_images(tmp_path, scope):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client)
        kind = "m3_event_cg_" + scope
        _, job = finish(client, worker, stop_at=lambda value: value["kind"] == kind)
        result = cg_result(job)
        result["omission_reason"] = "この章には適切なCG区間がありません。"
        if scope == "budget":
            for row in result["chapters"]:
                row["limit"] = 0
        else:
            result["cgs"] = []
        response = complete(client, worker, job, cg_output(job, result=result))
        assert response.status_code == 200, response.text
        if scope == "budget":
            # A failed/unusable allocation is adopted as explicit zero budgets;
            # the story still starts, with a settled zero-image CG plan.
            state = production(client, project)
            assert state["narrative_artifact_id"] is None
            assert state["event_cg"]["budget_completed"] is True
            assert state["event_cg"]["plan_completed"] is True
            assert state["event_cg"]["planned"] == 0
            assert state["event_cg"]["budget_omission_reason"] == result["omission_reason"]
            assert all(chapter["event_cg"]["budget_omission_reason"] == result["omission_reason"]
                       for chapter in state["chapters"])
            assert [chapter["event_cg"]["chapter_budget"] for chapter in state["chapters"]] == [0, 0, 0]
            assert state["event_cg"]["omissions"][0]["reason"] == result["omission_reason"]
            narrative = claim(client, worker)
            assert narrative and narrative["kind"] == "m3_narrative"
            assert complete(client, worker, narrative).status_code == 200
        jobs, _ = finish(client, worker)
        assert not any(row["kind"] == "m3_event_cg" for row in jobs)
        if scope == "budget":
            assert not any(row["kind"] == "m3_event_cg_plan" for row in jobs)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        script, _, _ = published_script(client, state["chapters"][0])
        assert not script.get("event_cg_segments")


@pytest.mark.parametrize("all_prompts_fail", [False, True])
def test_prompt_omissions_preserve_other_cgs_and_do_not_count_as_image_failures(tmp_path, all_prompts_fail):
    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client, max_cgs=2)
        _, job = finish(client, worker, stop_at=lambda value: value["kind"] == "m3_event_cg_plan")
        assert job["payload"]["chapter_budget"] == 2
        result = cg_result(job)
        omitted = [{"cg_id": "cg_two", "reason": "二つ目のCGの描画指示を作成できませんでした。"}]
        if all_prompts_fail:
            result["cgs"] = []
            omitted.insert(0, {"cg_id": "cg_one", "reason": "一つ目のCGの描画指示を作成できませんでした。"})
            result["omission_reason"] = "; ".join(row["reason"] for row in omitted)
        result["prompt_omissions"] = omitted
        response = complete(client, worker, job, cg_output(job, result=result))
        assert response.status_code == 200, response.text
        jobs, _ = finish(client, worker)
        state = production(client, project)
        assert state["status"] == "published", state["error"]
        expected_images = 0 if all_prompts_fail else 2  # One remaining CG plus its variant.
        image_jobs = [row for row in jobs if row["kind"] == "m3_event_cg"]
        assert len(image_jobs) == expected_images
        assert all(row["payload"]["cg_id"] == "cg_one" for row in image_jobs)
        for summary in (state["event_cg"], state["chapters"][0]["event_cg"]):
            assert summary["planned"] == summary["generated"] == expected_images
            assert summary["omitted"] == 0
            assert summary["planning_omitted"] == len(omitted)
            assert [{key: row[key] for key in ("cg_id", "reason")} for row in summary["omissions"]] == omitted
        assert state["chapters"][0]["event_cg"]["plan_completed"]
        script, names, _ = published_script(client, state["chapters"][0])
        assert len([name for name in names if name.startswith("data/cgimage/")]) == expected_images
        assert [row["id"] for row in script.get("event_cg_segments", [])] == ([] if all_prompts_fail else ["cg_one"])
        assert claim(client, worker) is None


def test_character_adjustment_republishes_adopted_cg_without_generation(tmp_path):
    from tests.integration.test_adjustments import apply, save, start

    with TestClient(create_app(tmp_path)) as client:
        project, worker = setup_story(client)
        finish(client, worker)
        before = production(client, project)
        original, _, bundle = published_script(client, before["chapters"][0])
        endpoint = f"/api/m3/projects/{project}/adjustments"
        view = start(client, endpoint)
        characters = copy.deepcopy(view["draft"]["characters"])
        characters[0]["offset_y"] = 20
        view = save(client, endpoint, view, characters)
        view = apply(client, endpoint, view)
        assert view["draft"]["status"] == "applied", view
        assert claim(client, worker) is None
        after = production(client, project)
        changed, _, _ = published_script(client, after["chapters"][0])
        assert changed["event_cg_segments"] == original["event_cg_segments"]
        assert [row for row in changed["assets"] if row["kind"] == "event_cg"] == [
            row for row in original["assets"] if row["kind"] == "event_cg"]
        assert client.get(before["chapters"][0]["export_url"]).content == bundle


def test_approval_requires_current_policy_and_ready_worker_before_starting(tmp_path):
    from packages.contracts.planning import planning_protocol
    from tests.integration.planning_fixtures import plan_content

    with TestClient(create_app(tmp_path)) as client:
        project, _ = ready(client)
        identifier = project["project"]["id"]
        action(client, project, "approve")
        worker = client.post("/api/workers", json={"name": "plan only", "capabilities": ["m3_plan"]}).json()["id"]
        job = claim(client, worker)
        content = plan_content(job["payload"]["approval_snapshot"])
        buffer = io.BytesIO()
        with ZipFile(buffer, "w") as archive:
            archive.writestr("result.json", json.dumps({"schema_version": 1, "kind": "m3_plan", "result": content,
                "provenance": {"seed": job["payload"]["seed"], "planning_protocol": planning_protocol(),
                               "generator_protocol": planning_protocol()}, "trace": []}))
        assert complete(client, worker, job, buffer.getvalue()).status_code == 200
        endpoint = f"/api/planning/projects/{identifier}"
        planning = client.get(endpoint).json()["planning"]
        policy = client.put(f"/api/event-cg/projects/{identifier}/policy",
                            json={"expected_revision": 0, "max_cgs": 1, "max_variants_per_cg": 0}).json()
        approve = {"action": "approve", "expected_revision": planning["revision"],
                   "expected_event_cg_policy_revision": policy["revision"]}
        response = client.post(endpoint + "/actions", json=approve)
        assert response.status_code == 409 and "準備済みWorker" in response.text
        assert production(client, identifier) is None
        client.post("/api/workers", json={"name": "CG ready", "capabilities": sorted(CG_KINDS)})
        assert client.get("/api/event-cg/settings").json()["ready"]
        assert client.post(endpoint + "/actions", json={**approve, "expected_event_cg_policy_revision": 0}).status_code == 409
        response = client.post(endpoint + "/actions", json=approve)
        assert response.status_code == 200, response.text
        assert production(client, identifier) is not None


def test_settings_offer_memory_configurations_and_accept_only_supported_sizes(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        settings = client.get("/api/event-cg/settings").json()
        assert settings["configuration"] == "vram32"
        assert [row["id"] for row in settings["configurations"]][:2] == ["vram32", "vram24_fast"]
        assert all({"label", "peak_vram_gib", "time_ratio", "quality"} <= set(row)
                   for row in settings["configurations"])
        assert settings["sizes"] == [{"width": 960, "height": 640}, {"width": 1536, "height": 1024}]
        chosen = next(row for row in settings["configurations"] if row["id"] == "vram16_8bit")
        profile = {**settings["profile"], "width": 1536, "height": 1024, "text_encoder_offload": "layers",
                   **{key: chosen[key] for key in ("reference_resolution", "use_kv_cache",
                                                   "transformer_storage", "vae_tiling")}}
        saved = client.put("/api/event-cg/settings", json={"expected_revision": 0, "profile": profile}).json()
        assert saved["configuration"] == "vram16_8bit" and saved["profile"] == profile
        assert client.put("/api/event-cg/settings", json={
            "expected_revision": 1, "profile": {**profile, "height": 640}}).status_code == 422
        assert client.get("/api/event-cg/settings").json()["profile"]["transformer_storage"] == "fp8"
