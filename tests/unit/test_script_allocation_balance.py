"""Experimental chapter allocation by size: event-end options and cuts between steps."""

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.script_plot import (
    EQUAL_SHARE_NOT_NEEDED,
    SIZED_ALLOCATION_RULE,
    ChapterAllocation,
    StepChapterAllocation,
    StoryChain,
    allocate_chain,
    allocate_chain_by_steps,
    boundary_options,
    numbered_chain_steps,
    split_chain,
)
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    allocation,
    chapter_plan,
    empty_cast,
    prompt_text,
    staging,
    story_chain,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as runtime  # noqa: PLC0414


def chain():
    return StoryChain.model_validate(story_chain())


def step_allocation(*ends):
    return StepChapterAllocation.model_validate({
        "chapters": [{"number": n, "title": f"章{n}", "role": "選択の結果を実行する。", "last_step": end}
                     for n, end in enumerate(ends, 1)], "foreshadowing": []})


def test_every_event_end_cut_is_offered_with_each_chapters_steps():
    options = boundary_options(chain())
    assert options == [{"last_events": [1, 2, 4], "steps_per_chapter": [2, 2, 4]},
                       {"last_events": [1, 3, 4], "steps_per_chapter": [2, 4, 2]},
                       {"last_events": [2, 3, 4], "steps_per_chapter": [4, 2, 2]}]


def test_steps_are_numbered_through_the_whole_chain():
    rows = numbered_chain_steps(chain())
    assert [[step["step_number"] for step in row["steps"]] for row in rows] == [[1, 2], [3, 4], [5, 6], [7, 8]]
    assert rows[1]["steps"][0]["action"] == "EVENT_2_ATTEMPT"


def test_a_cut_inside_an_event_continues_from_the_last_kept_result():
    chapters = split_chain(chain(), [3, 5, 8])
    assert [[len(event.steps) for event in events] for events in chapters] == [[2, 1], [1, 1], [1, 2]]
    second = chapters[1]
    assert second[0].start_condition == "EVENT_2_EXPERIENCE"
    assert second[0].steps[0].action == "EVENT_2_CHOICE"
    assert second[1].start_condition == "作業中。"
    assert chapters[2][0].start_condition == "EVENT_3_EXPERIENCE"
    for ends in ([3, 3, 8], [0, 4, 8], [2, 4, 7], [2, 4]):
        with pytest.raises(ValueError, match="partition all 8 steps"):
            split_chain(chain(), ends)


def test_step_allocation_keeps_every_step_once_and_matches_event_allocation_at_event_ends():
    plot = allocate_chain_by_steps(chain(), step_allocation(3, 5, 8))
    actions = [step.action for chapter in plot.chapters for event in chapter.events for step in event.steps]
    assert actions == [step.action for event in chain().events for step in event.steps]
    assert plot.chapters[1].route.start_condition == "EVENT_2_EXPERIENCE"
    by_event = allocate_chain(chain(), ChapterAllocation.model_validate(allocation()))
    by_step = allocate_chain_by_steps(chain(), step_allocation(2, 6, 8))
    assert by_step.model_dump() == by_event.model_dump()


def responses():
    return [empty_cast(), story_chain_draft(), allocation(), chapter_plan("s1"), FIRST, staging(FIRST),
            HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND),
            HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND)]


@pytest.mark.parametrize("sized", [True, False])
def test_a_sized_allocation_sees_each_cut_and_is_asked_for_similar_chapters(runtime, tmp_path, sized):
    calls, control = runtime
    control["responses"] = responses()
    source = {"approval_snapshot": three_chapter_snapshot(),
              "script_options": {"sized_allocation": True} if sized else {}}
    report = runner.run_script_debug(source, tmp_path)
    assert report["status"] == "script_complete"
    request = prompt_text(next(call for call in calls if call["purpose"] == "script-allocation"))
    assert ('"boundary_options"' in request) is sized and ('"total_steps": 8' in request) is sized
    assert (SIZED_ALLOCATION_RULE in request) is sized and (EQUAL_SHARE_NOT_NEEDED in request) is not sized
