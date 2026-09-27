"""Regression coverage for compact planning, phase changes and request accounting."""

import copy
import json

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.llm import ContextBudgetError
from services.worker.generation.script_plot import (
    PRESENTATION_INSTRUCTION,
    ChapterAllocation,
    ChapterPresentation,
    FixedChapterAllocation,
)
from services.worker.generation.script_realization import ChapterRealizationResponse
from tests.unit.test_script_context import TokenCounter, cost, material, run
from tests.unit.test_script_continuation_run import (
    FIRST,
    allocation,
    chapter_plan,
    empty_cast,
    outline,
    prompt_text,
    read_json,
    realization_plan,
    snapshot,
    staging,
    story_chain,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


def test_plot_phase_can_resume_then_switch_to_body_without_regenerating(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [empty_cast(), story_chain_draft(), allocation()]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True)
    assert report["status"] == "plot_complete" and len(calls) == 3
    before = {name: (tmp_path / name).read_bytes() for name in
              ("experiment.json", "cast-plan.json", "plot.json", "story-chain.json")}
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
    assert report["status"] == "plot_complete" and len(calls) == 3
    control["responses"] = [chapter_plan("s1"), FIRST, staging(FIRST)]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert report["exported_chapter_count"] == 1 and len(calls) == 6
    assert all((tmp_path / name).read_bytes() == value for name, value in before.items())
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True, resume=True)
    assert report["status"] == "plot_complete" and len(calls) == 6
    sessions = read_json(tmp_path / "draft-state.json")["sessions"]
    assert [row["requested_phase"] for row in sessions] == ["plot", "plot", "chapters", "plot"]


@pytest.mark.parametrize("event_count", [3, 4])
def test_allocation_paths_share_compact_schema_and_do_not_ask_for_body_length(runtime, tmp_path, event_count):
    calls, control = runtime
    chain = story_chain()
    chain["events"] = chain["events"][:event_count]
    topics = [{"character_ids": ["aoi", "ren"], "topic": "昼食を分けたいが、相手は午後の展示準備を終えたい。",
               "exchange": "葵が弁当を広げると蓮は作業中だと断る。葵が紙を押さえる役を買って出て、蓮が好物を選ぶ。"}]
    response = allocation()
    for row in response["chapters"]:
        row["conversation_topics"] = copy.deepcopy(topics)
    if event_count == 3:
        response = {**{f"chapter_{n}": {k: v for k, v in row.items() if k not in {"number", "last_event"}}
                     for n, row in enumerate(response["chapters"], 1)}, "foreshadowing": []}
    control["responses"] = [empty_cast(), story_chain_draft(chain), response]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, plot_only=True)
    call = calls[-1]
    assert PRESENTATION_INSTRUCTION in prompt_text(call)
    assert "4000" not in prompt_text(call) and "2000" not in prompt_text(call)
    assert call["profile"]["max_tokens"] == 2048
    schema = call["extra"]["response_format"]["json_schema"]["schema"]
    (FixedChapterAllocation if event_count == 3 else ChapterAllocation).model_validate(response)
    definition = schema["$defs"]["ConversationTopic"]["properties"]
    assert definition["topic"]["maxLength"] == definition["exchange"]["maxLength"] == 120
    assert "relationship_aspect" not in definition
    assert definition["character_ids"]["items"]["enum"] == ["aoi", "ren"]
    assert read_json(tmp_path / "plot.json")["chapters"][0]["conversation_topics"] == topics
    assert topics[0]["topic"] in (tmp_path / "plot.md").read_text(encoding="utf-8")
    assert topics[0]["exchange"] in (tmp_path / "plot.md").read_text(encoding="utf-8")


def test_compact_plan_limits_reject_oversize_without_slicing():
    good = {"title": "章", "role": "役割", "conversation_topics": []}
    assert ChapterPresentation.model_validate(good).conversation_topics == []
    with pytest.raises(ValueError):
        ChapterPresentation.model_validate({**good, "role": "長" * 201})
    topic = {"character_ids": ["aoi", "ren"], "topic": "長" * 121, "exchange": "声をかけると笑って振り向く。"}
    with pytest.raises(ValueError):
        ChapterPresentation.model_validate({**good, "conversation_topics": [topic]})
    with pytest.raises(ValueError):
        ChapterPresentation.model_validate({**good, "conversation_topics": [{**topic, "topic": "話しかける。", "exchange": "長" * 121}]})
    with pytest.raises(ValueError, match="Extra inputs"):
        ChapterPresentation.model_validate({**good, "conversation_space": "old unbounded text"})


def test_placement_schema_rejects_original_failure_and_projects_deterministic_values():
    data = realization_plan(3)
    schema = ChapterRealizationResponse.model_json_schema()
    assert len(schema["properties"]["placements"]["items"]["anyOf"]) == 3
    for kind in ("StagedStep", "CompletedStep"):
        assert "adjustment" not in schema["$defs"][kind]["properties"]
        assert schema["$defs"][kind]["additionalProperties"] is False
    bad = copy.deepcopy(data)
    bad["placements"][0]["adjustment"] = {"character_id": "ren", "action": "違う行動", "result": "違う結果"}
    with pytest.raises(ValueError):
        ChapterRealizationResponse.model_validate(bad)
    data["placements"][1] = {"step_number": 2, "handling": "already_done", "reason_from_source": "本文で完了。"}
    data["placements"][2] = {"step_number": 3, "handling": "adjust", "scene_number": 1,
                             "reason_from_source": "本文で状況が変化。", "adjustment": bad["placements"][0]["adjustment"]}
    projected = ChapterRealizationResponse.model_validate(data).as_realization()
    assert projected.placements[0].adjustment is None
    assert projected.placements[1].scene_number == 0
    assert projected.placements[2].adjustment.action == "違う行動"


def test_context_overflow_never_crops_current_conversation_material():
    future = {"core_version": "plot", "core": {}, "current_chapter": 1,
              "routes": [{"number": 1, "version": "plot", "conversation_topics": [
                  {"character_ids": ["aoi", "ren"], "topic": "消してはいけない会話。", "exchange": "話しかけると返事が来る。"}]}]}
    data = material(future=future)
    counter = TokenCounter(cost(data) - 1)
    with pytest.raises(ContextBudgetError):
        run(counter, data)
    assert all("消してはいけない会話。" in row["messages"][-1]["content"] for row in counter.checked)


def test_preflight_failure_does_not_spend_repair_and_exact_validation_is_saved(runtime, tmp_path, monkeypatch):
    calls, control = runtime
    fit = runner.fit_context
    def blocked(llm, stage, *args, **kwargs):
        if stage == "script-plan":
            error = ContextBudgetError("input 9000 + output 8192 + margin 512 > 16384")
            error.selection = {"budget": {"prompt_tokens": 9000, "output_tokens": 8192,
                                         "margin_tokens": 512, "context_size": 16384, "fits": False}}
            raise error
        return fit(llm, stage, *args, **kwargs)
    control["responses"] = [outline()]
    monkeypatch.setattr(runner, "fit_context", blocked)
    with pytest.raises(ContextBudgetError):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    step = read_json(tmp_path / "draft-state.json")["steps"]["plan-001"]
    assert step["attempts"] == [] and not step["preflight_failures"][0]["dispatched"]
    monkeypatch.setattr(runner, "fit_context", fit)
    invalid = chapter_plan("s1")
    invalid["scenes"][0]["character_ids"] = ["unknown"]
    control["responses"] = [invalid, chapter_plan("s1"), FIRST, staging(FIRST)]
    report = runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert report["exported_chapter_count"] == 1
    step = read_json(tmp_path / "draft-state.json")["steps"]["plan-001"]
    assert len(step["attempts"]) == 2
    error = read_json(tmp_path / "requests/plan-001-1.validation.json")
    assert "unknown characters" in error["error"] and json.loads(error["response"]["content"]) == invalid
    assert len(calls) == 5


def test_failed_structural_repair_remains_exhausted_on_resume(runtime, tmp_path):
    calls, control = runtime
    invalid = chapter_plan("s1")
    invalid["scenes"][0]["character_ids"] = ["unknown"]
    control["responses"] = [outline(), invalid, invalid]
    with pytest.raises(RuntimeError, match="structure/reference") as original:
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    error = read_json(tmp_path / "requests/plan-001-2.validation.json")["error"]
    assert error.split(": ", 1)[-1] in str(original.value)
    with pytest.raises(RuntimeError, match="structure/reference"):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert len(calls) == 3
    assert len(read_json(tmp_path / "draft-state.json")["steps"]["plan-001"]["attempts"]) == 2


def test_truncated_plan_stops_and_keeps_response_without_body_generation(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [outline(), {"content": '{"current_facts":', "_finish_reason": "length"}]
    with pytest.raises(RuntimeError):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    step = read_json(tmp_path / "draft-state.json")["steps"]["plan-001"]
    assert len(step["attempts"]) == 1 and step["attempts"][0]["reply"]["_finish_reason"] == "length"
    assert len(calls) == 2
    with pytest.raises(RuntimeError, match="truncated"):
        runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert len(calls) == 2
