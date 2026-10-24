"""A long-form plan: a chain allocated to any count at the most even cuts, then each chapter's
own steps added around it, the last chapter's closing step kept last."""

import re

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.script_expansion import FINAL_END_RULE, FINAL_EXPANSION_RULE, chained_starts
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
from tests.unit.test_script_production import app_runtime as _app_runtime

runtime = _runtime
app_runtime = _app_runtime


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


def test_a_chapter_starts_after_the_steps_added_to_the_previous_one(runtime, tmp_path):
    calls, control = runtime
    chain = chain_of(2)
    fixed = {f"chapter_{n}": {key: value for key, value in presentation(n).items() if key != "number"}
             for n in (1, 2)}
    fixed["foreshadowing"] = []
    control["responses"] = [empty_cast(), story_chain_draft(chain), fixed,
                            {"added_steps": [added(2, "1_LAST")]}, {"added_steps": []}]
    runner.run_script_debug(long_input(2), tmp_path, plot_only=True)
    skeleton = read_json(tmp_path / "skeleton-plot.json")
    plot = read_json(tmp_path / "plot.json")
    # The skeleton starts chapter 2 before the step added at the end of chapter 1.
    assert skeleton["chapters"][1]["events"][0]["start_condition"] == "EVENT_1_RESULT_2"
    assert [row["events"][0]["start_condition"] for row in plot["chapters"]] == ["作業中。", "ADDED_1_LAST_RESULT"]


def test_an_earlier_plot_is_chained_where_it_starts_partway_through():
    def chapter(start, *results):
        return {"events": [{"start_condition": start, "steps": [
            {"character_id": "ren", "action": "行動。", "result": result} for result in results]}]}

    plot = {"chapters": [chapter("冒頭。", "A", "B"), chapter("A", "C", "D"), chapter("書き直した開始。", "E"),
                         chapter("E", "F")]}
    chained = chained_starts(plot)
    # Partway through the previous chapter: moved to its end. A start written by hand, or one
    # already at the end, stays.
    assert [row["events"][0]["start_condition"] for row in chained["chapters"]] == [
        "冒頭。", "B", "書き直した開始。", "E"]
    assert plot["chapters"][1]["events"][0]["start_condition"] == "A"


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


# The STEP4 plan of a long-form work, in two stages.

def judgment(chapters=4):
    return {"endurance": "日常が積み重なる。", "structure": "二人が交互に場面を持つ。",
            "distinct_characters": ["葵", "蓮"], "chapters": chapters, "reason": "二人の変化に要る。"}


def long_plan_job(count=3, **payload):
    from tests.unit.test_common_plan_worker import plan_job

    request = plan_job(count)
    request["payload"]["script_options"] = {"long_form": True}
    request["payload"].update(payload)
    return request


def test_a_long_form_plan_first_holds_its_cast_and_the_judged_count(app_runtime, tmp_path):
    from tests.unit.test_common_plan_worker import generate, supporting_cast

    calls, control = app_runtime
    request = long_plan_job()
    control["responses"] = [supporting_cast(), judgment()]
    content = generate(request, tmp_path)["result"]
    assert content == {"cast_plan": supporting_cast(), "chapter_judgment": judgment()}
    assert [call["purpose"] for call in calls] == ["script-cast", "script-cast"]
    # Neither the cast nor the judgment sees the approval's chapter count.
    assert all("chapterCount:" not in prompt_text(call) for call in calls)
    assert "この物語に適した話数（章数）を判断してください" in prompt_text(calls[1])
    assert "helper" in prompt_text(calls[1])


def test_a_supporting_character_revised_before_the_count_is_judged_again(app_runtime, tmp_path):
    from tests.unit.test_common_plan_worker import generate, supporting_cast

    calls, control = app_runtime
    source = {"cast_plan": supporting_cast(), "chapter_judgment": judgment()}
    revised = {**supporting_cast()["supporting_characters"][0], "name": "新しい友人"}
    request = long_plan_job(plan_content=source, target="character", character_id="helper",
                            instruction="名前を変える。")
    control["responses"] = [revised, judgment(5)]
    content = generate(request, tmp_path)["result"]
    assert content["cast_plan"]["supporting_characters"][0]["name"] == "新しい友人"
    assert content["chapter_judgment"]["chapters"] == 5 and "plot" not in content
    assert "新しい友人" in prompt_text(calls[1])
    plot_request = long_plan_job(plan_content=source, target="plot", instruction="章を変える。")
    with pytest.raises(ValueError, match="only its supporting characters"):
        generate(plot_request, tmp_path / "plot")


def test_the_chosen_count_plans_the_plot_from_the_kept_cast(app_runtime, tmp_path):
    from tests.unit.test_common_plan_worker import generate, supporting_cast

    calls, control = app_runtime
    source = {"cast_plan": supporting_cast(), "chapter_judgment": judgment()}
    chain = chain_of(4)
    fixed = {f"chapter_{n}": {key: value for key, value in presentation(n).items() if key != "number"}
             for n in range(1, 5)}
    fixed["foreshadowing"] = []
    request = long_plan_job(4, plan_content=source, chapter_count=4)
    control["responses"] = [story_chain_draft(chain), fixed, *[{"added_steps": []}] * 4]
    content = generate(request, tmp_path)["result"]
    assert content["chapter_count"] == 4 and content["chapter_judgment"] == judgment()
    assert content["cast_plan"] == supporting_cast()
    assert [row["number"] for row in content["plot"]["chapters"]] == [1, 2, 3, 4]
    # No cast is planned again, and the chain sees the chosen count.
    assert [call["purpose"] for call in calls] == ["script-outline", "script-allocation", *["script-outline"] * 4]
    assert re.search(r"chapterCount:\n +4\n", prompt_text(calls[0])) and "helper" in prompt_text(calls[0])
    mismatch = long_plan_job(3, plan_content=source, chapter_count=4)
    with pytest.raises(ValueError, match="chosen chapter count"):
        generate(mismatch, tmp_path / "mismatch")
