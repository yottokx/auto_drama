"""A product chapter job is written again with other script options from the product's own checkpoint."""
import copy
import json
import sys

import pytest

from packages.contracts.planning import validate_plan_content
from scripts.story import rerun_product_chapters as rerun
from services.worker.generation import m3_pipeline
from services.worker.generation.causal_runtime import digest
from services.worker.generation.script_production import ProductionScriptRun
from tests.unit.test_common_plan_worker import semantic_plan
from tests.unit.test_live_script_settings import configured as _configured
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    allocation,
    chapter_plan,
    empty_cast,
    staging,
    story_chain,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import following, generate, job

runtime = _runtime
app_runtime = _app_runtime
configured = _configured


@pytest.fixture
def history(monkeypatch):
    # The rerun replaces the static method; put the original back after each test.
    monkeypatch.setattr(ProductionScriptRun, "_history", ProductionScriptRun.__dict__["_history"])


@pytest.fixture(params=["plain", "execution_settings"])
def second_chapter(request, configured, tmp_path):
    first = job()
    if request.param == "execution_settings":
        first["payload"].update(execution_settings_version=1, execution_settings_revision=0)
    return following(first, generate(first, tmp_path / "first"))


def test_an_option_runs_from_the_checkpoint_bound_again(tmp_path, second_chapter, history):
    source = copy.deepcopy(second_chapter)
    payload = rerun.chapter_payload(second_chapter, ["chapter_scope_guard", "skip_staging"], None)
    config = m3_pipeline.pipeline.load_config()
    with pytest.raises(ValueError, match="checkpoint"):
        ProductionScriptRun(tmp_path / "plain", config, payload)

    rerun.rebind_first_checkpoint(2, -1)
    run = ProductionScriptRun(tmp_path / "bound", config, payload)
    checkpoint = source["payload"]["script_checkpoint"]
    assert run.state["request_ordinal"] == checkpoint["state"]["request_ordinal"] - 1
    assert run.options.chapter_scope_guard and run.options.skip_staging
    assert [row["text"] for row in run.state["chapters"]] == [row["text"] for row in checkpoint["state"]["chapters"]]
    # Only the run's copy is bound again; the product's request stays as it was.
    assert second_chapter == source


def test_unchanged_options_keep_the_checkpoint(tmp_path, second_chapter, history):
    payload = rerun.chapter_payload(second_chapter, [], None)
    checkpoint = payload["script_checkpoint"]
    assert rerun.rebound(checkpoint, checkpoint["generation_identity"], 0) == checkpoint
    rerun.rebind_first_checkpoint(2, 0)
    run = ProductionScriptRun(tmp_path / "same", m3_pipeline.pipeline.load_config(), payload)
    assert run.state["request_ordinal"] == checkpoint["state"]["request_ordinal"]


def test_the_product_job_is_written_again_with_the_guard(configured, tmp_path, history, monkeypatch, capsys):
    calls, control = configured
    # A plot of events with steps, as a long-form plot has, so the guard has steps to check.
    control["responses"] = [empty_cast(), story_chain_draft(), allocation(), chapter_plan("s1"), FIRST,
                            staging(FIRST)]
    first = job(three_chapter_snapshot())
    first["payload"].update(execution_settings_version=1, execution_settings_revision=0)
    second_chapter = following(first, generate(first, tmp_path / "first"))
    job_dir = tmp_path / "product-job"
    job_dir.mkdir()
    (job_dir / "job-request.json").write_text(json.dumps(second_chapter, ensure_ascii=False), encoding="utf-8")
    # The previous chapter's note, the guard's check of what it already showed, the plan and the text;
    # skip_staging asks for no staging.
    control["responses"] = [HANDOFF, {"done": []}, chapter_plan("s1", continued=True), SECOND]
    output = tmp_path / "rerun"
    monkeypatch.setattr(sys, "argv", ["rerun", str(job_dir), "--output-dir", str(output),
                                      "--option", "chapter_scope_guard", "--option", "skip_staging"])
    before = second_chapter["payload"]["script_checkpoint"]["state"]["request_ordinal"]
    assert rerun.main() == 0
    assert [call["purpose"] for call in calls[-4:]] == ["script-handoff", "script-handoff", "script-plan",
                                                        "script-scene"]
    row = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert row["chapter"] == 2 and row["scenes"][0]["id"] == "s1" and row["characters"] > 0
    assert row["route_progress"] == [{"type": "route_progress", "chapter": 2, "done": []}]
    assert (output / "chapter-002.md").read_text(encoding="utf-8").strip()
    trace = json.loads((output / "chapter-002" / "generation-trace.json").read_text(encoding="utf-8"))
    requests = [row["request"] for row in trace if row.get("type") == "llm_generation"]
    # The previous chapter's note is asked with the product's number; the guard's check comes after it.
    assert requests[:3] == [before + 1, before + 2, before + 3]


def test_chained_starts_begin_each_chapter_after_the_previous_one(app_runtime, tmp_path, history):
    calls, control = app_runtime
    first = job()
    content = semantic_plan()
    chain = story_chain()["events"]
    # As a long-form plot is saved: chapter 2 starts from the skeleton, before chapter 1's added last step.
    content["plot"]["chapters"][0]["events"] = [{"start_condition": "作業中。", "steps": [
        *chain[0]["steps"], {"character_id": "ren", "action": "ADDED_ACTION", "result": "ADDED_RESULT"}]}]
    content["plot"]["chapters"][1]["events"] = [{**chain[1], "start_condition": chain[0]["steps"][-1]["result"]}]
    content = validate_plan_content(content, first["payload"]["approval_snapshot"]).model_dump(mode="json")
    first["payload"]["approved_plan"] = {"content": content, "approval_id": "accepted-1", "sha256": digest(content)}
    control["responses"] = [chapter_plan("s1"), FIRST, staging(FIRST)]
    second_chapter = following(first, generate(first, tmp_path / "first"))

    rerun.chain_starts(second_chapter["payload"])
    plot = second_chapter["payload"]["approved_plan"]["content"]["plot"]
    assert plot["chapters"][1]["events"][0]["start_condition"] == "ADDED_RESULT"
    assert plot["chapters"][0] == content["plot"]["chapters"][0]
    payload = rerun.chapter_payload(second_chapter, ["skip_staging"], None)
    rerun.rebind_first_checkpoint(2, 0)
    run = ProductionScriptRun(tmp_path / "chained", m3_pipeline.pipeline.load_config(), payload)
    run._verify_narratives()
    assert run.state["plot"] == plot and run.state["accepted_plan"]["approval_id"] == "accepted-1"


def test_a_later_chapter_continues_from_this_run(second_chapter):
    previous = {"narrative": {"title": "書き直した章"}, "checkpoint": {"sha256": "own"}}
    payload = rerun.chapter_payload(second_chapter, ["skip_staging"], previous)
    assert payload["previous_narrative"] == previous["narrative"]
    assert payload["script_checkpoint"] == previous["checkpoint"]
    assert payload["previous_narrative_artifact_id"].startswith("rerun-")


def test_only_chapter_jobs_are_written_again(second_chapter):
    with pytest.raises(ValueError, match="m3_narrative"):
        rerun.chapter_payload({**second_chapter, "kind": "m3_image"}, [], None)
