"""A long-form plan in STEP4: first the cast and the judged chapter count, then the plot at the
count the user chose, and production at that count."""
import copy
import json

from fastapi.testclient import TestClient

from services.coordinator.app import create_app
from tests.integration.planning_fixtures import plan_content
from tests.integration.test_m2 import action, claim, ready
from tests.integration.test_m3 import M3_KINDS
from tests.integration.test_planning import complete_plan, plan_action, plan_worker, planning
from tests.integration.test_planning_assets import SUPPORT, with_support

JUDGMENT = {"endurance": "記録をめぐる話が続く。", "structure": "二人が交互に場面を持つ。",
            "distinct_characters": ["二人"], "chapters": 6, "reason": "二人の変化に要る。"}


def cast_stage(client):
    project, _ = ready(client)
    project = action(client, project, "approve")
    worker = plan_worker(client)
    job = claim(client, worker)
    assert job["payload"]["script_options"]["long_form"] is True
    assert job["payload"]["script_options"]["chapter_scope_guard"] is True
    assert job["payload"]["script_options"]["opening_split"] is True
    snapshot = job["payload"]["approval_snapshot"]
    content = {"cast_plan": plan_content(snapshot)["cast_plan"], "chapter_judgment": JUDGMENT}
    complete_plan(client, job, worker, content=content)
    return project, worker, content, snapshot


def test_the_plot_is_planned_at_the_chosen_count_and_production_follows_it(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, cast, _ = cast_stage(client)
        state = planning(client, project)
        # The default settings run at 16k, whose tier holds three chapters.
        assert state["content"] == cast and state["chapter_limit"] == 3
        plan_action(client, project, "approve", status=409)
        plan_action(client, project, "revise", status=422, target="plot", instruction="章を変える。")
        plan_action(client, project, "plot", status=400, chapter_count=4)
        plan_action(client, project, "plot", chapter_count=2)
        job = claim(client, worker)
        assert job["payload"]["chapter_count"] == 2 and job["payload"]["plan_content"] == cast
        snapshot = job["payload"]["approval_snapshot"]
        assert snapshot["world"]["result"]["chapterCount"] == 2
        content = {**plan_content(snapshot), "chapter_count": 2, "chapter_judgment": JUDGMENT}
        complete_plan(client, job, worker, content=content)
        assert planning(client, project)["content"]["chapter_count"] == 2
        # A revision of the plot keeps the count it was planned at.
        plan_action(client, project, "revise", target="plot", instruction="第1章を穏やかに。")
        revised = claim(client, worker)
        assert revised["payload"]["approval_snapshot"]["world"]["result"]["chapterCount"] == 2
        complete_plan(client, revised, worker, content=content)
        plan_action(client, project, "approve")
        media = client.post("/api/workers", json={"name": "media", "capabilities": M3_KINDS}).json()["id"]
        chapter = claim(client, media)
        assert chapter["kind"] == "m3_narrative"
        assert chapter["payload"]["approval_snapshot"]["world"]["result"]["chapterCount"] == 2
        assert chapter["payload"]["approved_plan"]["content"]["chapter_count"] == 2
        production = client.get(f"/api/m3/projects/{project['project']['id']}").json()["production"]
        assert production["chapter_count"] == 2


def test_a_plan_begun_before_the_guard_writes_its_chapters_with_it(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, _, _ = cast_stage(client)
        # A long-form plan begun before chapter_scope_guard and opening_split: its planning jobs carry neither.
        with client.app.state.coordinator.db.transaction() as connection:
            for row in connection.execute("SELECT id,payload FROM job WHERE kind='m3_plan'").fetchall():
                payload = json.loads(row["payload"])
                payload["script_options"].pop("chapter_scope_guard")
                payload["script_options"].pop("opening_split")
                connection.execute("UPDATE job SET payload=? WHERE id=?",
                                   (json.dumps(payload, ensure_ascii=False), row["id"]))
        plan_action(client, project, "plot", chapter_count=2)
        job = claim(client, worker)
        assert not {"chapter_scope_guard", "opening_split"} & set(job["payload"]["script_options"])
        content = {**plan_content(job["payload"]["approval_snapshot"]), "chapter_count": 2,
                   "chapter_judgment": JUDGMENT}
        complete_plan(client, job, worker, content=content)
        plan_action(client, project, "approve")
        media = client.post("/api/workers", json={"name": "media", "capabilities": M3_KINDS}).json()["id"]
        chapter = claim(client, media)
        assert chapter["kind"] == "m3_narrative"
        assert chapter["payload"]["script_options"]["chapter_scope_guard"] is True
        assert chapter["payload"]["script_options"]["opening_split"] is True
        assert chapter["payload"]["script_options"]["long_form"] is True


def test_an_earlier_plot_starts_each_chapter_at_the_previous_end_when_approved(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, _, _ = cast_stage(client)
        plan_action(client, project, "plot", chapter_count=2)
        job = claim(client, worker)
        content = {**plan_content(job["payload"]["approval_snapshot"]), "chapter_count": 2,
                   "chapter_judgment": JUDGMENT}
        cid = content["plot"]["core"]["characters"][0]["character_id"]
        steps = [{"character_id": cid, "action": f"行動{n}。", "result": f"結果{n}。"} for n in (1, 2)]
        # As a long-form plot expanded before the fix: chapter 2 starts after the skeleton's step 1,
        # before step 2 added after it.
        content["plot"]["chapters"][0]["events"] = [{"start_condition": "冒頭。", "steps": steps}]
        content["plot"]["chapters"][1]["events"] = [{"start_condition": "結果1。", "steps": [
            {"character_id": cid, "action": "行動3。", "result": "結果3。"}]}]
        complete_plan(client, job, worker, content=content)
        assert planning(client, project)["content"]["plot"]["chapters"][1]["events"][0]["start_condition"] == "結果1。"
        plan_action(client, project, "approve")
        media = client.post("/api/workers", json={"name": "media", "capabilities": M3_KINDS}).json()["id"]
        chapter = claim(client, media)
        plot = chapter["payload"]["approved_plan"]["content"]["plot"]
        assert [row["events"][0]["start_condition"] for row in plot["chapters"]] == ["冒頭。", "結果2。"]
        assert plot["chapters"][1]["route"]["start_condition"] == "結果2。"
        # The plan shown in STEP4 is the one approved.
        assert planning(client, project)["content"]["plot"] == plot


def test_a_cast_edited_by_hand_before_the_count_is_judged_again(tmp_path):
    with TestClient(create_app(tmp_path)) as client:
        project, worker, cast, snapshot = cast_stage(client)
        edited = {**copy.deepcopy(cast), "cast_plan": with_support(snapshot)["cast_plan"]}
        plan_action(client, project, "save", content=edited)
        job = claim(client, worker)
        assert job["kind"] == "m3_plan" and job["payload"]["judge_only"] is True
        assert job["payload"]["plan_content"]["cast_plan"]["supporting_characters"][0]["id"] == SUPPORT["id"]
        complete_plan(client, job, worker, content={**edited, "chapter_judgment": {**JUDGMENT, "chapters": 4}})
        assert planning(client, project)["content"]["chapter_judgment"]["chapters"] == 4
