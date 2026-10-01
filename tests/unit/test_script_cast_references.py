"""Repair cast UUID assignment failures without regenerating creative material."""
import copy
import json

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.draft_story import DraftExecutionError
from services.worker.generation.script_cast import (
    CastConnectionError,
    CastPlan,
    CastReferenceRepair,
    apply_connection_repair,
    approved_main_assignments,
    check_cast,
    connection_repair_material,
)
from tests.unit.test_script_cast import ensemble
from tests.unit.test_script_continuation_run import (
    allocation,
    read_json,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import generate, job

runtime = _runtime
app_runtime = _app_runtime


def broken_cast():
    value = ensemble()
    value["supporting_characters"] = value["supporting_characters"][:1]
    value["everyday_context"] = value["everyday_context"][:1]
    value["connections"] = [
        {"character_ids": ["neighbor", "ren"], "relationship": "葵はこの友人の料理を楽しみにしている。"},
        {"character_ids": ["neighbor", "ren"], "relationship": "この友人は葵に試作品を食べてもらいたい。"},
        {"character_ids": ["neighbor", "ren"], "relationship": "蓮とは普段から仕事の相談をする間柄。"},
    ]
    return value


def assignments():
    return {"assignments": {"R1": "P1", "R2": "P1", "R3": "P2"}}


def test_wrong_uuid_reassignment_preserves_both_views_and_all_people():
    plan = CastPlan.model_validate(broken_cast())
    main = [row["result"] for row in three_chapter_snapshot()["characters"]]
    # Long IDs are copied only by code, never by the repair model.
    mapping = {"aoi": "character-11111111-1111-1111-1111-111111111111",
               "ren": "character-22222222-2222-2222-2222-222222222222"}
    for row in main:
        row["id"] = mapping[row["id"]]
    source = plan.model_dump(mode="json")
    for row in source["connections"]:
        row["character_ids"] = [mapping.get(cid, cid) for cid in row["character_ids"]]
    plan = CastPlan.model_validate(source)
    material, schema, pairs = connection_repair_material(plan, main)
    assert schema["properties"]["assignments"]["required"] == ["R1", "R2", "R3"]
    assert material["available_pairs"][0]["people"][0]["name"] == main[0]["name"]
    result = apply_connection_repair(plan, CastReferenceRepair.model_validate(assignments()), pairs, set(mapping.values()))
    assert result.supporting_characters == plan.supporting_characters
    assert result.everyday_context == plan.everyday_context
    assert result.connections[0].character_ids == [mapping["aoi"], "neighbor"]
    assert result.connections[0].relationship == "\n".join(row.relationship for row in plan.connections[:2])
    assert result.connections[1].character_ids == [mapping["ren"], "neighbor"]
    assert result.connections[1].relationship == plan.connections[2].relationship
    assert plan.model_dump(mode="json") == source


@pytest.mark.parametrize("damage", ["missing", "extra", "unknown", "unresolved", "discard"])
def test_repair_cannot_drop_invent_or_leave_unknown_references(damage):
    plan = CastPlan.model_validate(broken_cast())
    main = [row["result"] for row in three_chapter_snapshot()["characters"]]
    _, _, pairs = connection_repair_material(plan, main)
    fixed = assignments()
    if damage == "missing":
        fixed["assignments"].pop("R3")
    elif damage == "extra":
        fixed["assignments"]["R4"] = "P1"
    else:
        fixed["assignments"]["R1"] = {"unknown": "P999", "unresolved": "unresolved",
                                         "discard": "approved_main_pair"}[damage]
    with pytest.raises(ValueError):
        apply_connection_repair(plan, CastReferenceRepair.model_validate(fixed), pairs, {"aoi", "ren"})


def test_approved_main_relationships_remain_authoritative():
    value = broken_cast()
    value["connections"] = [{"character_ids": ["aoi", "ren"], "relationship": "承認情報と違う関係"}]
    plan = CastPlan.model_validate(value)
    main = [row["result"] for row in three_chapter_snapshot()["characters"]]
    _, _, pairs = connection_repair_material(plan, main)
    result = apply_connection_repair(plan, CastReferenceRepair(assignments={"R1": "approved_main_pair"}),
                                    pairs, {"aoi", "ren"})
    assert result.connections == []
    assert result.supporting_characters == plan.supporting_characters


def test_new_reference_failure_repairs_without_regenerating_cast(runtime, tmp_path):
    calls, control = runtime
    original = broken_cast()
    control["responses"] = [original, assignments(), story_chain_draft(), allocation()]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True)
    assert report["status"] == "plot_complete"
    assert len(calls) == 4
    assert report["cast_connection_repair"]["assignments"] == assignments()["assignments"]
    assert report["metrics"]["charged_tokens"] == 4 * 530
    state = read_json(tmp_path / "draft-state.json")
    assert len(state["steps"]["cast-plan"]["attempts"]) == 1
    assert json.loads(state["steps"]["cast-plan"]["attempts"][0]["reply"]["content"]) == original
    assert state["cast_plan"]["plan"]["supporting_characters"] == original["supporting_characters"]
    assert state["cast_plan"]["plan"]["everyday_context"] == original["everyday_context"]
    assert state["cast_connection_repair"]["source_request"] == 1
    saved = (tmp_path / "cast-plan.json").read_bytes()
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
    assert len(calls) == 4 and (tmp_path / "cast-plan.json").read_bytes() == saved


def fail_legacy(runtime, monkeypatch, output):
    calls, control = runtime
    first, latest = broken_cast(), broken_cast()
    first["supporting_characters"][0]["name"] = "旧案"
    control["responses"] = [first, latest]

    def legacy_failure(*args):
        raise ValueError("Cast relationships must reference distinct known people without replacing approved pairs.")

    with monkeypatch.context() as patch:
        patch.setattr(runner.ScriptRun, "repair_cast_connections", legacy_failure)
        with pytest.raises(DraftExecutionError, match="Cast relationships"):
            runner.run_script_debug(three_chapter_snapshot(), output, plot_only=True)
    assert len(calls) == 2
    return read_json(output / "draft-state.json")


def test_legacy_failure_resumes_latest_cast_preserving_requests_and_usage(runtime, tmp_path, monkeypatch):
    calls, control = runtime
    old = fail_legacy(runtime, monkeypatch, tmp_path)
    files = {path: path.read_bytes() for path in (tmp_path / "requests").glob("*")}
    control["responses"] = [assignments(), story_chain_draft(), allocation()]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
    assert report["status"] == "plot_complete" and len(calls) == 5
    state = read_json(tmp_path / "draft-state.json")
    assert state["steps"]["cast-plan"]["attempts"] == old["steps"]["cast-plan"]["attempts"]
    assert all(path.read_bytes() == content for path, content in files.items())
    assert state["cast_connection_repair"]["source_request"] == 2
    assert state["cast_plan"]["plan"]["supporting_characters"] == broken_cast()["supporting_characters"]
    assert report["metrics"]["charged_tokens"] == 5 * 530


def test_failed_reference_repair_has_fixed_attempt_budget_across_resume(runtime, tmp_path, monkeypatch):
    calls, control = runtime
    fail_legacy(runtime, monkeypatch, tmp_path)
    bad = assignments()
    bad["assignments"]["R1"] = "unresolved"
    control["responses"] = [bad, copy.deepcopy(bad)]
    for _ in range(3):
        with pytest.raises(DraftExecutionError, match="unresolved"):
            runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
        assert len(calls) == 4
    state = read_json(tmp_path / "draft-state.json")
    assert len(state["steps"]["cast-plan-connections-v1"]["attempts"]) == 2


def test_completed_source_is_not_confused_with_later_transport_failure(runtime, tmp_path, monkeypatch):
    calls, control = runtime
    state = fail_legacy(runtime, monkeypatch, tmp_path)
    failed = state["steps"]["cast-plan"]["attempts"][-1]
    failed.update(status="failed", error="HTTP failure")
    failed.pop("reply")
    (tmp_path / "draft-state.json").write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")
    control["responses"] = [assignments(), story_chain_draft(), allocation()]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
    result = read_json(tmp_path / "draft-state.json")
    assert result["cast_connection_repair"]["source_request"] == 1
    assert result["cast_plan"]["plan"]["supporting_characters"][0]["name"] == "旧案"
    assert len(calls) == 5


def test_app_entrypoint_adopts_repaired_cast_and_retains_approved_relationships(app_runtime, tmp_path):
    from tests.unit.test_script_continuation_run import FIRST, chapter_plan, staging

    calls, control = app_runtime
    source = three_chapter_snapshot()
    original = copy.deepcopy(source)
    control["responses"] = [broken_cast(), assignments(), story_chain_draft(), allocation(),
                            chapter_plan("s1"), FIRST, staging(FIRST)]
    envelope = generate(job(source), tmp_path)
    state = envelope["provenance"]["script_checkpoint"]["state"]
    assert state["cast_plan"]["plan"]["connections"][0]["character_ids"] == ["aoi", "neighbor"]
    assert source == original
    assert len(calls) == 7
    assert any(row["type"] == "cast_connection_repair" for row in envelope["trace"])
    assert generate(job(source), tmp_path) == envelope
    assert len(calls) == 7


def test_connection_errors_identify_the_actual_problem():
    with pytest.raises(CastConnectionError, match="duplicate pair"):
        check_cast(CastPlan.model_validate(broken_cast()), {"aoi", "ren"})


def main_pair_cast(*, mixed=False):
    value = broken_cast() if mixed else {"supporting_characters": [], "everyday_context": [], "connections": []}
    value["connections"].insert(0, {"character_ids": ["aoi", "ren"], "relationship": "承認済み関係の書き直し"})
    return value


def mixed_repair():
    return {"assignments": {"R1": "unresolved", "R2": "P1", "R3": "P1", "R4": "P2"}}


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("legacy_failure", [False, True])
def test_exact_main_pairs_resolve_without_model_judgment_and_keep_saved_evidence(
        runtime, tmp_path, monkeypatch, mixed, legacy_failure):
    calls, control = runtime
    source = three_chapter_snapshot()
    original = copy.deepcopy(source)
    cast = main_pair_cast(mixed=mixed)
    repair = mixed_repair() if mixed else {"assignments": {"R1": "unresolved"}}
    old = files = None
    if legacy_failure:
        control["responses"] = [cast, repair, copy.deepcopy(repair)]
        with monkeypatch.context() as patch:
            patch.setattr(runner, "approved_main_assignments", lambda *_: {})
            with pytest.raises(DraftExecutionError, match="R1: relationship reference remains unresolved"):
                runner.run_script_debug(source, tmp_path, plot_only=True)
        old = read_json(tmp_path / "draft-state.json")
        files = {path: path.read_bytes() for path in (tmp_path / "requests").glob("*")}
        control["responses"] = [story_chain_draft(), allocation()]
    else:
        control["responses"] = [cast, *([repair] if mixed else []), story_chain_draft(), allocation()]
    report = runner.run_script_debug(source, tmp_path, plot_only=True, resume=legacy_failure)
    assert report["status"] == "plot_complete"
    state = read_json(tmp_path / "draft-state.json")
    result = state["cast_plan"]["plan"]
    assert source == original
    assert result["supporting_characters"] == cast["supporting_characters"]
    assert result["everyday_context"] == cast["everyday_context"]
    assert state["cast_connection_repair"]["deterministic_assignments"] == {"R1": "approved_main_pair"}
    assert state["cast_connection_repair"]["assignments"]["R1"] == "approved_main_pair"
    if mixed:
        assert result["connections"][0]["relationship"] == "\n".join(row["relationship"] for row in cast["connections"][1:3])
        assert result["connections"][1]["relationship"] == cast["connections"][3]["relationship"]
    else:
        assert result["connections"] == []
    if legacy_failure:
        assert len(calls) == 5  # No additional relationship-repair calls.
        assert all(path.read_bytes() == content for path, content in files.items())
        for key in old["steps"]:
            assert state["steps"][key]["attempts"] == old["steps"][key]["attempts"]
    else:
        assert len(calls) == (4 if mixed else 3)
        assert [call["purpose"] for call in calls].count("script-cast") == (2 if mixed else 1)
    assert report["metrics"]["charged_tokens"] == len(calls) * 530
    before = len(calls)
    runner.run_script_debug(source, tmp_path, plot_only=True, resume=True)
    assert len(calls) == before


def test_deterministic_main_pair_rule_requires_two_distinct_exact_main_ids():
    value = main_pair_cast(mixed=True)
    value["connections"] = [
        {"character_ids": pair, "relationship": "説明"} for pair in (
            ["aoi", "ren"], ["ren", "aoi"], ["aoi", "aoi"], ["aoi", "unknown"],
            ["aoi", "neighbor"], ["aoi", "ren_"])]
    plan = CastPlan.model_validate(value)
    assert approved_main_assignments(plan, {"aoi", "ren"}) == {
        "R1": "approved_main_pair", "R2": "approved_main_pair"}
    with pytest.raises(CastConnectionError, match="self-reference"):
        check_cast(plan.model_copy(update={"connections": plan.connections[2:]}), {"aoi", "ren"})


@pytest.mark.parametrize("damage", ["unresolved", "missing", "extra"])
def test_deterministic_main_pair_handling_does_not_hide_other_repair_errors(runtime, tmp_path, damage):
    calls, control = runtime
    repair = mixed_repair()
    if damage == "missing":
        repair["assignments"].pop("R2")
    elif damage == "extra":
        repair["assignments"]["R5"] = "P1"
    else:
        repair["assignments"]["R2"] = "unresolved"
    control["responses"] = [main_pair_cast(mixed=True), repair, copy.deepcopy(repair)]
    with pytest.raises(DraftExecutionError):
        runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True)
    assert len(calls) == 3
    assert "cast_plan" not in read_json(tmp_path / "draft-state.json")
