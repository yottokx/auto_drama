"""STEP4 stops at semantic plans; STEP5 consumes exactly the accepted plan."""
import copy
import io
import json
import zipfile

import pytest

from packages.contracts.planning import planning_protocol, validate_plan_content
from services.worker.generation import m3_pipeline
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.causal_runtime import digest
from services.worker.generation.draft_story import DraftExecutionError
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    allocation,
    chapter_plan,
    empty_cast,
    outline,
    read_json,
    snapshot,
    staging,
    story_chain_draft,
)
from tests.unit.test_script_continuation_run import runtime as _runtime
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import following, job

runtime = _runtime
app_runtime = _app_runtime


def plan_job(count=2):
    request = job(snapshot())
    request["kind"] = "m3_plan"
    request["payload"].pop("chapter_number")
    request["payload"].update(planning_id="plan-1", planning_revision=0,
                              planning_protocol=planning_protocol())
    request["payload"]["approval_snapshot"]["world"]["result"]["chapterCount"] = count
    return request


def generate(request, path):
    with zipfile.ZipFile(io.BytesIO(m3_pipeline.generate_job(request, path))) as archive:
        assert archive.namelist() == ["result.json"]
        return json.loads(archive.read("result.json"))


def semantic_plan(count=2):
    planned = outline()
    if count != 2:
        planned["chapters"] = [{**copy.deepcopy(planned["chapters"][0]), "number": n}
                               for n in range(1, count + 1)]
    return {"cast_plan": empty_cast(), "plot": planned}


def supporting_cast():
    character = copy.deepcopy(snapshot()["characters"][0]["result"])
    character.update(id="helper", name="手伝いの友人")
    return {"supporting_characters": [character], "everyday_context": [{"character_id": "helper",
            "personal_concern": "展示を見たい。", "contact": "会場で会う。", "initial_knowledge": "会場を知る。"}],
            "connections": [{"character_ids": ["aoi", "helper"], "relationship": "学校の友人。"}]}


@pytest.mark.parametrize("count", [1, 2, 3, 9])
def test_common_plan_generates_cast_for_every_count_and_never_starts_body(app_runtime, tmp_path, count):
    calls, control = app_runtime
    request = plan_job(count)
    planned = semantic_plan(count)["plot"]
    if count == 3:
        responses = [story_chain_draft(), allocation()]
    elif count <= 8:
        responses = [planned]
    else:
        responses = [planned["core"], {"chapters": planned["chapters"][:8]},
                     {"chapters": planned["chapters"][8:]}]
    control["responses"] = [supporting_cast(), *responses]
    result = generate(request, tmp_path)
    assert result["kind"] == "m3_plan"
    content = validate_plan_content(result["result"], request["payload"]["approval_snapshot"])
    assert [row.number for row in content.plot.chapters] == list(range(1, count + 1))
    assert content.cast_plan.supporting_characters[0].id == "helper"
    assert result["provenance"]["generator_protocol"] == planning_protocol()
    assert all(call["purpose"] in {"script-cast", "script-outline", "script-allocation"} for call in calls)
    state = read_json(tmp_path / "script/draft-state.json")
    assert state["chapters"] == [] and state["partial"] is None
    before = len(calls)
    assert generate(request, tmp_path) == result and len(calls) == before


@pytest.mark.parametrize("count", [1, 2, 3, 9])
def test_approved_plan_starts_chapter_body_without_new_cast_or_plot(app_runtime, tmp_path, count):
    calls, control = app_runtime
    request = job()
    request["payload"]["approval_snapshot"]["world"]["result"]["chapterCount"] = count
    content = semantic_plan(count)
    request["payload"]["approved_plan"] = {"content": content, "approval_id": "accepted-1", "sha256": digest(content)}
    control["responses"] = [chapter_plan("s1"), FIRST, staging(FIRST)]
    result = generate(request, tmp_path / "first")
    assert [call["purpose"] for call in calls] == ["script-plan", "script-scene", "script-staging"]
    checkpoint = result["provenance"]["script_checkpoint"]
    assert checkpoint["plan_approval_id"] == "accepted-1" and checkpoint["plan_sha256"] == digest(content)
    assert checkpoint["state"]["plot"] == content["plot"]
    assert "story_chain" not in checkpoint["state"] and "chapter_allocation" not in checkpoint["state"]
    if count > 1:
        control["responses"] = [HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND)]
        continuation = generate(following(request, result), tmp_path / "second")
        assert continuation["provenance"]["script_checkpoint"]["plan_sha256"] == digest(content)
        assert len(calls) == 7


@pytest.mark.parametrize("changed", ["content", "approval", "checkpoint"])
def test_accepted_plan_identity_cannot_change_in_chapter_continuation(app_runtime, tmp_path, changed):
    calls, control = app_runtime
    request = job()
    content = semantic_plan()
    request["payload"]["approved_plan"] = {"content": content, "approval_id": "accepted-1", "sha256": digest(content)}
    control["responses"] = [chapter_plan("s1"), FIRST, staging(FIRST)]
    next_job = following(request, generate(request, tmp_path / "first"))
    if changed == "content":
        next_job["payload"]["approved_plan"]["content"]["plot"]["core"]["central_question"] = "違う課題"
        next_job["payload"]["approved_plan"]["sha256"] = digest(next_job["payload"]["approved_plan"]["content"])
    elif changed == "approval":
        next_job["payload"]["approved_plan"]["approval_id"] = "accepted-2"
    else:
        next_job["payload"]["script_checkpoint"]["plan_sha256"] = "bad"
    with pytest.raises(ValueError, match="checkpoint"):
        generate(next_job, tmp_path / "second")
    assert len(calls) == 3


@pytest.mark.parametrize("target", ["plot", "chapter", "character", "relationships", "all"])
def test_revision_changes_only_the_requested_semantic_part(app_runtime, tmp_path, target):
    calls, control = app_runtime
    request = plan_job()
    original = semantic_plan()
    original["cast_plan"] = supporting_cast()
    request["payload"].update(plan_content=original, instruction="指定部分を修正。",
                              target="plot" if target == "chapter" else target)
    expected = copy.deepcopy(original)
    if target == "plot":
        expected["plot"]["core"]["central_question"] = "修正した課題。"
        response = expected["plot"]
    elif target == "chapter":
        request["payload"]["chapter_number"] = 2
        expected["plot"]["chapters"][1]["title"] = "修正した章。"
        response = expected["plot"]["chapters"][1]
    elif target == "character":
        request["payload"]["character_id"] = "helper"
        expected["cast_plan"]["supporting_characters"][0]["settings"] = "修正した設定。"
        response = expected["cast_plan"]["supporting_characters"][0]
    elif target == "relationships":
        expected["cast_plan"]["connections"][0]["relationship"] = "同じ展示を準備する友人。"
        response = {"connections": expected["cast_plan"]["connections"]}
    else:
        expected["plot"]["core"]["central_question"] = "全体修正した課題。"
        response = expected
    control["responses"] = [response]
    result = generate(request, tmp_path)
    assert result["result"] == expected and len(calls) == 1
    assert read_json(tmp_path / "script/draft-state.json")["chapters"] == []


def test_plan_rejects_unknown_people_and_main_replacement_without_changing_snapshot():
    original = snapshot()
    bad = semantic_plan()
    bad["cast_plan"] = supporting_cast()
    bad["cast_plan"]["supporting_characters"][0]["id"] = "aoi"
    with pytest.raises(ValueError, match="unique unused"):
        validate_plan_content(bad, original)
    bad = semantic_plan()
    bad["plot"]["chapters"][0]["conversation_topics"] = [{"character_ids": ["aoi", "unknown"],
        "topic": "声を掛ける。", "exchange": "応える。"}]
    with pytest.raises(ValueError, match="registered"):
        validate_plan_content(bad, original)
    assert original == snapshot()


def test_event_edits_normalize_derived_route_and_reject_unregistered_steps():
    content = semantic_plan()
    chapter = content["plot"]["chapters"][0]
    chapter["events"] = [{"start_condition": "会場を訪れる。", "steps": [
        {"character_id": "aoi", "action": "写真を並べる。", "result": "配置を確かめる。"},
        {"character_id": "ren", "action": "写真を離す。", "result": "見やすくなる。"}]}]
    result = validate_plan_content(content, snapshot())
    assert result.plot.chapters[0].route.next_state == "見やすくなる。"
    assert "写真を離す。" in result.plot.as_outline().chapters[0].summary
    chapter["events"][0]["steps"][0]["character_id"] = "unknown"
    with pytest.raises(ValueError, match="registered"):
        validate_plan_content(content, snapshot())


def test_cancelled_plan_resumes_accepted_cast_without_regeneration(app_runtime, tmp_path):
    calls, control = app_runtime
    request = plan_job()
    control["responses"] = [empty_cast(), GenerationCancelled("stop")]
    with pytest.raises(GenerationCancelled):
        generate(request, tmp_path)
    assert not (tmp_path / "result.zip").exists()
    assert read_json(tmp_path / "script/draft-state.json")["chapters"] == []
    control["responses"] = [outline()]
    assert generate(request, tmp_path)["result"] == semantic_plan()
    assert [call["purpose"] for call in calls] == ["script-cast", "script-outline", "script-outline"]


def test_explicit_plan_retry_keeps_cast_and_charges_failed_replies(app_runtime, tmp_path):
    calls, control = app_runtime
    request = plan_job()
    control["responses"] = [empty_cast(), "{bad", "{bad"]
    with pytest.raises(DraftExecutionError):
        generate(request, tmp_path)
    assert len(calls) == 3
    control["responses"] = [outline()]
    result = generate({**request, "retry_generation": 1}, tmp_path)
    assert result["result"] == semantic_plan() and len(calls) == 4
    state = read_json(tmp_path / "script/draft-state.json")
    assert len(state["steps"]["outline"]["attempts"]) == 3
    assert state["request_ordinal"] == 4


def test_ai_character_revision_cannot_change_selected_id_or_approved_main(app_runtime, tmp_path):
    calls, control = app_runtime
    request = plan_job()
    content = semantic_plan()
    content["cast_plan"] = supporting_cast()
    request["payload"].update(plan_content=content, target="character", character_id="helper",
                              instruction="性格を変更。")
    wrong = copy.deepcopy(content["cast_plan"]["supporting_characters"][0])
    wrong["id"] = "aoi"
    control["responses"] = [wrong, wrong]
    with pytest.raises(DraftExecutionError, match="preserve the selected"):
        generate(request, tmp_path)
    state = read_json(tmp_path / "script/draft-state.json")
    assert state["cast_plan"]["plan"] == content["cast_plan"]
    assert state["chapters"] == [] and len(calls) == 2
    assert not (tmp_path / "result.zip").exists()


def test_planning_capability_uses_only_the_existing_text_runtime():
    assert "m3_plan" in m3_pipeline.generation_kinds(["m2_world"])
    assert "m3_plan" not in m3_pipeline.generation_kinds(["m2_image", "m2_voice"])
