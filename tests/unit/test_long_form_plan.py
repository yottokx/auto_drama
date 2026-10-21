"""A long-form plan: a chain allocated to any count at the most even cuts, then each chapter's
own steps added around it, the last chapter's closing step kept last."""

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.script_expansion import FINAL_END_RULE, FINAL_EXPANSION_RULE
from services.worker.generation.script_plot import balanced_boundary_options
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    chapter_plan,
    empty_cast,
    prompt_text,
    read_json,
    snapshot,
    staging,
    story_chain,
    story_chain_draft,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


def long_input(count, **options):
    source = snapshot()
    source["world"]["result"]["chapterCount"] = count
    return {"approval_snapshot": source, "script_options": {"long_form": True, **options}}


def chain_of(events):
    chain = story_chain()
    chain["events"] = [{"start_condition": "作業中。", "steps": [
        {"character_id": "ren", "action": f"EVENT_{n}_STEP_{s}", "result": f"EVENT_{n}_RESULT_{s}"}
        for s in range(1, 3)]} for n in range(1, events + 1)]
    return chain


def presentation(number):
    return {"number": number, "title": f"章{number}", "role": "結果を受けて進む。", "conversation_topics": []}


def added(after, label):
    return {"after_step": after, "character_id": "aoi", "action": f"ADDED_{label}", "result": f"ADDED_{label}_RESULT"}


def test_four_chapters_are_cut_evenly_and_each_chapter_is_expanded(runtime, tmp_path):
    calls, control = runtime
    chain = chain_of(6)
    ends = balanced_boundary_options([2] * 6, 4)[0]["last_events"]
    allocation = {"chapters": [{**presentation(n), "last_event": end} for n, end in enumerate(ends, 1)],
                  "foreshadowing": []}
    expansions = [{"added_steps": [added(0, f"{n}_FIRST")]} for n in range(1, 4)]
    expansions.append({"added_steps": [added(1, "4_MIDDLE")]})
    control["responses"] = [empty_cast(), story_chain_draft(chain), allocation, *expansions]
    report = runner.run_script_debug(long_input(4), tmp_path, plot_only=True)
    assert report["status"] == "plot_complete"
    chain_call, allocation_call, *expansion_calls = calls[1:]
    assert "全4章の分量" in prompt_text(chain_call) and "4〜12個の出来事" in prompt_text(chain_call)
    # The cuts offered are the most even ones; an uneven cut is refused by the schema check.
    assert '"boundary_options"' in prompt_text(allocation_call) and "全4章へ配分" in prompt_text(allocation_call)
    assert [f"全4章の第{n}章を肉付けします" in prompt_text(call) for n, call in enumerate(expansion_calls, 1)] == [True] * 4
    assert [FINAL_EXPANSION_RULE in prompt_text(call) for call in expansion_calls] == [False, False, False, True]
    schemas = [call["extra"]["response_format"]["json_schema"]["schema"] for call in expansion_calls]
    skeleton = read_json(tmp_path / "skeleton-plot.json")
    sizes = [sum(len(event["steps"]) for event in row["events"]) for row in skeleton["chapters"]]
    # Nothing can be placed after the last chapter's closing step.
    assert [schema["$defs"]["AddedStep"]["properties"]["after_step"]["maximum"] for schema in schemas] == [
        *sizes[:3], sizes[3] - 1]
    assert [schema["properties"]["added_steps"]["maxItems"] for schema in schemas] == [6 - size for size in sizes]
    plot = read_json(tmp_path / "plot.json")
    steps = [[step["action"] for event in row["events"] for step in event["steps"]] for row in plot["chapters"]]
    assert steps[0][0] == "ADDED_1_FIRST" and steps[3][1] == "ADDED_4_MIDDLE"
    assert steps[3][-1] == skeleton["chapters"][3]["events"][-1]["steps"][-1]["action"]
    # The skeleton steps stay as they were, in order.
    assert [action for row in steps for action in row if not action.startswith("ADDED")] == [
        step["action"] for event in chain["events"] for step in event["steps"]]
    count = len(calls)
    runner.run_script_debug(long_input(4), tmp_path, plot_only=True, resume=True)
    assert len(calls) == count


def test_an_expansion_resumes_from_the_saved_skeleton(runtime, tmp_path):
    calls, control = runtime
    chain = chain_of(2)
    fixed = {f"chapter_{n}": {key: value for key, value in presentation(n).items() if key != "number"}
             for n in (1, 2)}
    fixed["foreshadowing"] = []
    control["responses"] = [empty_cast(), story_chain_draft(chain), fixed, {"added_steps": []},
                            GenerationCancelled("expansion interrupted")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(long_input(2), tmp_path, plot_only=True)
    assert not (tmp_path / "plot.json").exists()
    count = len(calls)
    control["responses"] = [GenerationCancelled("only the second chapter is asked again")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(long_input(2), tmp_path, plot_only=True, resume=True)
    # The chain, the allocation and the first chapter's expansion are not asked again.
    assert len(calls) == count + 1 and "全2章の第2章を肉付けします" in prompt_text(calls[-1])


def test_the_last_scene_plan_ends_on_the_closing_step(runtime, tmp_path):
    calls, control = runtime
    chain = chain_of(2)
    fixed = {f"chapter_{n}": {key: value for key, value in presentation(n).items() if key != "number"}
             for n in (1, 2)}
    fixed["foreshadowing"] = []
    control["responses"] = [empty_cast(), story_chain_draft(chain), fixed, {"added_steps": []}, {"added_steps": []},
                            chapter_plan("s1"), FIRST, staging(FIRST),
                            HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND)]
    report = runner.run_script_debug(long_input(2), tmp_path)
    assert report["status"] == "script_complete"
    plans = [call for call in calls if call["purpose"] == "script-plan"]
    assert [FINAL_END_RULE.strip() in prompt_text(call) for call in plans] == [False, True]


def test_a_long_form_cast_plan_names_no_chapter_count(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [{"surroundings": ["担任"], **empty_cast()}, GenerationCancelled("only the cast")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(long_input(5, cast_limit=4), tmp_path, plot_only=True)
    text = prompt_text(calls[0])
    assert "人数の目安はありません。物語でメイン" in text and "全5章" not in text
