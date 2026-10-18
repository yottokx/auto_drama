"""A chapter plan keeps to its chapter: later routes in brief, steps already shown marked done."""

import copy

from services.worker.generation import script_continuation as runner
from services.worker.generation import script_route_progress as progress
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

SHOWN = {"character_id": "ren", "action": "写真の間に紙を置く。", "result": "写真と説明が分かれる。"}
LEFT = {"character_id": "aoi", "action": "札を貼り直す。", "result": "配置が決まる。"}


def future():
    return {"core": {}, "core_version": "plot", "current_chapter": 2, "routes": [
        {"number": 2, "title": "現在章", "role": "決着", "version": "plot", "conversation_topics": [],
         "events": [{"planned_start_condition": "準備中。", "steps": [SHOWN, LEFT]},
                    {"planned_start_condition": "配置済み。", "steps": [dict(SHOWN, action="紙を外す。")]}]},
        {"number": 3, "title": "最終章", "role": "余韻", "version": "plot", "conversation_topics": [],
         "events": [{"planned_start_condition": "配置済み。", "steps": [LEFT]}]}]}


def test_later_chapters_keep_only_purpose_and_end_state():
    original = future()
    brief = progress.later_in_brief(original, 2)
    assert brief["routes"][0] == original["routes"][0]
    assert brief["routes"][1] == {"number": 3, "version": "plot", "title": "最終章", "role": "余韻",
                                  "planned_end_state": "配置が決まる。"}
    assert original["routes"][1]["events"]


def test_shown_steps_leave_the_route_and_are_named_as_done():
    context = {"future": future(), "brief": "依頼"}
    steps = progress.route_steps(context["future"], 2)
    assert list(steps) == ["E1-1", "E1-2", "E2-1"]
    schema = progress.progress_schema(steps)
    assert schema["$defs"]["DoneStep"]["properties"]["step"]["enum"] == ["E1-1", "E1-2", "E2-1"]
    answer = progress.RouteProgress.model_validate(
        {"done": [{"step": "E1-1", "evidence": "紙を一枚置いてみた"}, {"step": "E2-1", "evidence": "外した"}]})
    marked = progress.mark_done(context, 2, answer)
    route = marked["future"]["routes"][0]
    assert route["events"] == [{"planned_start_condition": "準備中。", "steps": [LEFT]}]
    assert route["already_done_in_saved_script"] == [
        "ren: 写真の間に紙を置く。 → 写真と説明が分かれる。", "ren: 紙を外す。 → 写真と説明が分かれる。"]
    assert progress.DONE_NOTE in marked["brief"]
    assert context["future"]["routes"][0]["events"][0]["steps"] == [SHOWN, LEFT]
    assert progress.mark_done(context, 2, progress.RouteProgress(done=[])) is context


def test_a_guarded_run_hides_later_steps_and_tells_the_next_plan_what_was_shown(runtime, tmp_path):
    calls, control = runtime
    plot = outline()
    plot["chapters"][1]["events"] = [{"start_condition": "展示を準備する。", "steps": [SHOWN, LEFT]}]
    control["responses"] = [plot, chapter_plan("s1"), FIRST, staging(FIRST), HANDOFF,
                            {"done": [{"step": "E1-1", "evidence": "紙を一枚置いてみた"}]},
                            chapter_plan("s1", continued=True), SECOND, staging(SECOND)]
    source = {"approval_snapshot": snapshot(), "script_options": {"chapter_scope_guard": True}}
    report = runner.run_script_debug(copy.deepcopy(source), tmp_path)
    assert report["status"] == "script_complete"
    assert [call["purpose"] for call in calls] == [
        "script-outline", "script-plan", "script-scene", "script-staging",
        "script-handoff", "script-handoff", "script-plan", "script-scene", "script-staging"]
    first_plan, check, second_plan = (prompt_text(calls[index]) for index in (1, 5, 6))
    assert SHOWN["action"] not in first_plan and LEFT["result"] in first_plan
    assert FIRST in check and "E1-1" in check and SHOWN["action"] in check
    assert "already_done_in_saved_script" in second_plan and progress.DONE_NOTE in second_plan
    assert SHOWN["action"] in second_plan and LEFT["action"] in second_plan
    # Writing keeps every later chapter in full; only planning sees them in brief.
    assert SHOWN["action"] in prompt_text(calls[2])


def test_an_unguarded_run_makes_no_progress_call(runtime, tmp_path):
    calls, control = runtime
    plot = outline()
    plot["chapters"][1]["events"] = [{"start_condition": "展示を準備する。", "steps": [SHOWN, LEFT]}]
    control["responses"][0] = plot
    report = runner.run_script_debug(snapshot(), tmp_path)
    assert report["status"] == "script_complete"
    assert [call["purpose"] for call in calls].count("script-handoff") == 1
    assert SHOWN["action"] in prompt_text(calls[1])
