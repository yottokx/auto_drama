"""A plan with forces (ScriptOptions.story_forces): the chain writes each person's forces first and
each step its aim and who does not learn its result; the allocation and the expansions read the
forces; chapter planning and scene writing read them by the approved method (script_drama)."""

import copy
import json

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation import script_drama
from services.worker.generation.script_expansion import expansion_schema
from services.worker.generation.script_plot import (
    FORCES_BRIEF,
    FORCES_EXPANSION_RULE,
    FORCES_RULE,
    ChainStep,
    balanced_boundary_options,
    chapter_models,
)
from tests.unit.test_long_form_plan import added, chain_of, long_input, presentation
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    chapter_plan,
    empty_cast,
    prompt_text,
    read_json,
    staging,
    story_chain_draft,
)
from tests.unit.test_script_continuation_run import runtime as _runtime
from tests.unit.test_script_production import app_runtime as _app_runtime

runtime = _runtime
app_runtime = _app_runtime
FORCES = [
    {"character_id": "aoi", "want": "展示を年表どおりに仕上げたい。", "means": "写真を年順に並べる。",
     "last_resort": "自分の写真を外す。", "unknown": "蓮が年を知らないこと。", "at_odds": "蓮と並べ方で食い違う。"},
    {"character_id": "ren", "want": "見る人が迷わない展示にしたい。", "means": "写真の間を空ける。",
     "last_resort": "", "unknown": "", "at_odds": "葵と並べ方で食い違う。"},
]


def with_aims(chain):
    result = copy.deepcopy(chain)
    for event in result["events"]:
        event["steps"] = [{"character_id": step["character_id"], "aim": "AIM_" + step["action"],
                           "action": step["action"], "result": step["result"], "hidden_from": ["aoi"]}
                          for step in event["steps"]]
    return result


def forces_draft(chain):
    return {"forces": FORCES, **story_chain_draft(chain)}


def schema_of(call):
    return call["extra"]["response_format"]["json_schema"]["schema"]


def forces_plot(count):
    """Responses for a plot of ``count`` chapters, one two-step event each."""
    fixed = {f"chapter_{n}": {key: value for key, value in presentation(n).items() if key != "number"}
             for n in range(1, count + 1)}
    fixed["foreshadowing"] = []
    return [empty_cast(), forces_draft(with_aims(chain_of(count))), fixed, *[{"added_steps": []}] * count]


def test_the_chain_writes_the_forces_first_and_the_later_calls_read_them(runtime, tmp_path):
    calls, control = runtime
    chain = with_aims(chain_of(6))
    ends = balanced_boundary_options([2] * 6, 4)[0]["last_events"]
    allocation = {"chapters": [{"number": n, "last_event": end, **{key: value for key, value in presentation(n).items()
                                                                     if key != "number"}}
                               for n, end in enumerate(ends, 1)], "foreshadowing": []}
    expansions = [{"added_steps": [{**added(0, f"{n}_FIRST"), "aim": "AIM_ADDED", "hidden_from": []}]}
                  for n in range(1, 5)]
    control["responses"] = [empty_cast(), forces_draft(chain), allocation, *expansions]
    report = runner.run_script_debug(long_input(4, story_forces=True), tmp_path, plot_only=True)
    assert report["status"] == "plot_complete"
    chain_call, allocation_call, *expansion_calls = calls[1:]
    assert FORCES_RULE in prompt_text(chain_call) and "aim、action、具体的なresult、hidden_fromの順" in prompt_text(chain_call)
    chain_schema = schema_of(chain_call)
    assert list(chain_schema["properties"]) == ["forces", "core", "resolution_basis", "opening_condition", "events"]
    step = chain_schema["$defs"]["ChainStep"]
    assert list(step["properties"]) == ["character_id", "aim", "action", "result", "hidden_from"]
    assert {"aim", "hidden_from"} <= set(step["required"])
    assert step["properties"]["hidden_from"]["items"]["enum"] == ["aoi", "ren"]
    assert read_json(tmp_path / "story-forces.json") == FORCES
    # The allocation and every expansion read the forces; each chapter's cut comes before its title.
    forces = FORCES_BRIEF + json.dumps(FORCES, ensure_ascii=False)
    assert forces in prompt_text(allocation_call)
    assert list(schema_of(allocation_call)["$defs"]["LongChapterBoundary"]["properties"])[:2] == ["number", "last_event"]
    assert all(forces in prompt_text(call) and FORCES_EXPANSION_RULE in prompt_text(call) for call in expansion_calls)
    assert all("aim" in schema_of(call)["$defs"]["AddedStep"]["properties"] for call in expansion_calls)
    plot = read_json(tmp_path / "plot.json")
    steps = [step for row in plot["chapters"] for event in row["events"] for step in event["steps"]]
    assert steps[0] == {"character_id": "aoi", "aim": "AIM_ADDED", "action": "ADDED_1_FIRST",
                        "result": "ADDED_1_FIRST_RESULT", "hidden_from": []}
    assert steps[1]["aim"] == "AIM_EVENT_1_STEP_1" and steps[1]["hidden_from"] == ["aoi"]


def test_the_chain_without_forces_holds_every_main(runtime, tmp_path):
    calls, control = runtime
    draft = forces_draft(with_aims(chain_of(2)))
    draft["forces"] = FORCES[:1]
    control["responses"] = [empty_cast(), draft, draft]
    with pytest.raises(runner.DraftExecutionError, match="every approved main character"):
        runner.run_script_debug(long_input(2, story_forces=True), tmp_path, plot_only=True)


def test_a_plan_without_forces_is_asked_as_before():
    # Earlier plans keep every answer format: the step fields are absent unless asked for.
    step = ChainStep(character_id="aoi", action="行動。", result="結果。")
    assert step.model_dump() == {"character_id": "aoi", "action": "行動。", "result": "結果。"}
    assert list(ChainStep.model_json_schema()["properties"]) == ["character_id", "action", "result"]
    assert "forces" not in chapter_models(4)[1].model_json_schema()["properties"]
    schema, _ = expansion_schema({"aoi"}, 2, False)
    assert list(schema["$defs"]["AddedStep"]["properties"]) == ["after_step", "character_id", "action", "result"]


def drama_run(runtime, tmp_path, method, unspoken=""):
    calls, control = runtime
    first, second = chapter_plan("s1"), chapter_plan("s1", continued=True)
    if method == "forces_unspoken":
        first["scenes"][0]["unspoken"] = unspoken
        second["scenes"][0]["unspoken"] = ""
    control["responses"] = [*forces_plot(2), first, FIRST, staging(FIRST), HANDOFF, second, SECOND, staging(SECOND)]
    report = runner.run_script_debug(long_input(2, story_forces=True, script_drama=method), tmp_path)
    assert report["status"] == "script_complete"
    return ([call for call in calls if call["purpose"] == "script-plan"],
            [call for call in calls if call["purpose"] == "script-scene"])


def test_unspoken_aims_reach_the_scene_writer_but_not_the_saved_plan(runtime, tmp_path):
    plans, scenes = drama_run(runtime, tmp_path, "forces_unspoken", "葵は自分の写真を外すつもりでいる。\n\nlength_weight: 3")
    forces = script_drama.FORCES_BRIEF + json.dumps(FORCES, ensure_ascii=False)
    for call in plans:
        assert script_drama.UNSPOKEN_PLAN + script_drama.FORCES_PLAN in prompt_text(call) and forces in prompt_text(call)
        scene = schema_of(call)["$defs"]["WeightedScene"]
        keys = list(scene["properties"])
        assert keys[keys.index("objectives") + 1] == "unspoken" and "unspoken" in scene["required"]
    for call in scenes:
        text = prompt_text(call)
        assert script_drama.UNSPOKEN_NARRATOR in text and script_drama.FORCES_SCENE in text
        assert script_drama.NARRATOR_RULE not in text and forces in text
    # Only what the plan wrote, without the next field run into it; a scene with none gets no note.
    note = script_drama.UNSPOKEN_BRIEF + "葵は自分の写真を外すつもりでいる。"
    assert [note in prompt_text(call) for call in scenes] == [True, False]
    assert "length_weight: 3" not in prompt_text(scenes[0])
    saved = read_json(tmp_path / "chapters" / "chapter-001.plan.json")
    assert "unspoken" not in json.dumps(saved, ensure_ascii=False)


def test_forces_reach_planning_and_writing_without_unspoken_aims(runtime, tmp_path):
    plans, scenes = drama_run(runtime, tmp_path, "forces")
    assert all(script_drama.FORCES_PLAN in prompt_text(call) and script_drama.UNSPOKEN_PLAN not in prompt_text(call)
               for call in plans)
    assert all("unspoken" not in schema_of(call)["$defs"]["WeightedScene"]["properties"] for call in plans)
    assert all(script_drama.FORCES_SCENE in prompt_text(call) and script_drama.NARRATOR_RULE in prompt_text(call)
               and script_drama.UNSPOKEN_BRIEF not in prompt_text(call) for call in scenes)


def test_a_method_with_forces_needs_a_plan_with_them(runtime, tmp_path):
    calls, control = runtime
    fixed = {f"chapter_{n}": {key: value for key, value in presentation(n).items() if key != "number"}
             for n in (1, 2)}
    fixed["foreshadowing"] = []
    control["responses"] = [empty_cast(), story_chain_draft(chain_of(2)), fixed, {"added_steps": []},
                            {"added_steps": []}, chapter_plan("s1")]
    with pytest.raises(ValueError, match="needs a plan with forces"):
        runner.run_script_debug(long_input(2, script_drama="forces"), tmp_path)


def test_the_plan_holds_its_forces_and_a_whole_revision_keeps_them(app_runtime, tmp_path):
    from tests.unit.test_common_plan_worker import generate, supporting_cast
    from tests.unit.test_long_form_plan import judgment, long_plan_job

    calls, control = app_runtime
    source = {"cast_plan": supporting_cast(), "chapter_judgment": judgment()}
    request = long_plan_job(2, plan_content=source, chapter_count=2)
    request["payload"]["script_options"]["story_forces"] = True
    control["responses"] = forces_plot(2)[1:]
    content = generate(request, tmp_path / "plot")["result"]
    assert content["story_forces"] == FORCES and "aim" in content["plot"]["chapters"][0]["events"][0]["steps"][0]
    revision = long_plan_job(2, plan_content=content, target="all", instruction="全体を穏やかに。")
    control["responses"] = [{"cast_plan": content["cast_plan"], "plot": content["plot"]}]
    revised = generate(revision, tmp_path / "revision")["result"]
    assert revised["story_forces"] == FORCES
    assert "StoryForce" not in json.dumps(schema_of(calls[-1]), ensure_ascii=False)
