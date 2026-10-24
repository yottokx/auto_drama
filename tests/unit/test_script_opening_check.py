"""A later chapter opens after the saved script: without its planned start, or planned again when
its opening plays what the previous chapter already showed."""

import copy

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation import script_opening_check as opening
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    chapter_plan,
    outline,
    prompt_text,
    snapshot,
    staging,
)
from tests.unit.test_script_continuation_run import runtime as runtime  # noqa: PLC0414

START = "展示の札を並べ終えた。"
STEP = {"character_id": "aoi", "action": "札を貼り直す。", "result": "配置が決まる。"}


def two_chapters():
    plot = outline()
    plot["chapters"][1]["events"] = [{"start_condition": START, "steps": [STEP]}]
    return plot


def source(**options):
    return {"approval_snapshot": snapshot(), "script_options": options}


def test_a_later_chapter_is_planned_and_written_without_its_planned_start(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [two_chapters(), chapter_plan("s1"), FIRST, staging(FIRST), HANDOFF,
                            chapter_plan("s1", continued=True), SECOND, staging(SECOND)]
    report = runner.run_script_debug(copy.deepcopy(source(opening_from_script=True)), tmp_path)
    assert report["status"] == "script_complete"
    purposes = [call["purpose"] for call in calls]
    first_plan, second_plan = (prompt_text(calls[index]) for index, purpose in enumerate(purposes)
                               if purpose == "script-plan")
    second_scene = prompt_text(calls[len(purposes) - 1 - purposes[::-1].index("script-scene")])
    # The first chapter still sees the next chapter's start; the chapter itself does not.
    assert START in first_plan
    assert START not in second_plan and START not in second_scene
    assert STEP["action"] in second_plan


@pytest.mark.parametrize("replayed", [True, False])
def test_an_opening_that_plays_the_saved_script_is_planned_again(runtime, tmp_path, replayed):
    calls, control = runtime
    answer = {"done": [{"step": "opening", "evidence": "前章の最後の台詞"}] if replayed else []}
    control["responses"] = [two_chapters(), chapter_plan("s1"), FIRST, staging(FIRST), HANDOFF,
                            chapter_plan("s1", continued=True), answer,
                            *([chapter_plan("s1", continued=True)] if replayed else []),
                            SECOND, staging(SECOND)]
    report = runner.run_script_debug(copy.deepcopy(source(opening_replay_check=True)), tmp_path)
    assert report["status"] == "script_complete"
    purposes = [call["purpose"] for call in calls]
    # The previous chapter's note, the plan, then the check of its opening against the saved script.
    assert purposes[4:7] == ["script-handoff", "script-plan", "script-handoff"]
    assert purposes.count("script-plan") == (3 if replayed else 2)
    check = prompt_text(calls[6])
    assert FIRST in check and "opening（当章の最初の新行動）" in check
    if replayed:
        again = prompt_text(calls[7])
        assert opening.REPLAN_NOTE in again and "前章の最後の台詞" in again
        assert opening.REPLAN_NOTE not in prompt_text(calls[5])


def test_a_split_opening_gives_the_text_the_end_reached_apart_from_the_new_action(runtime, tmp_path):
    calls, control = runtime
    split = {**chapter_plan("s1", continued=True), "reached": "REACHED_END"}
    control["responses"] = [two_chapters(), chapter_plan("s1"), FIRST, staging(FIRST), HANDOFF,
                            split, SECOND, staging(SECOND)]
    report = runner.run_script_debug(copy.deepcopy(source(opening_split=True)), tmp_path)
    assert report["status"] == "script_complete"
    plan_call = calls[5]
    assert plan_call["purpose"] == "script-plan"
    schema = plan_call["extra"]["response_format"]["json_schema"]["schema"]
    assert list(schema["properties"])[:2] == ["reached", "continuation"]
    assert "reachedとcontinuationを先に書きます" in prompt_text(plan_call)
    scene = prompt_text(calls[6])
    # The scene gets its plan as JSON, so the line break between the two is escaped.
    assert "【直前の到達点（前章で上演済み）】REACHED_END\\n【最初の新行動】" + split["continuation"] in scene


def test_the_first_event_loses_only_its_planned_start():
    future = {"routes": [{"number": 2, "events": [{"planned_start_condition": "a", "steps": []},
                                                  {"planned_start_condition": "b", "steps": []}]},
                         {"number": 3, "events": [{"planned_start_condition": "c", "steps": []}]}]}
    result = opening.without_planned_start(future, 2)
    assert result["routes"][0]["events"] == [{"steps": []}, {"planned_start_condition": "b", "steps": []}]
    assert result["routes"][1] == future["routes"][1]
    assert future["routes"][0]["events"][0]["planned_start_condition"] == "a"
