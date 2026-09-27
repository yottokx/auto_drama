"""Direct scene planning retains VN scope and volume without plot-index gates."""

from services.worker.generation.script_cast import ScriptOptions
from services.worker.generation.script_chapter_plan import (
    ContinuedChapterPlan,
    FirstChapterPlan,
    project_plan,
)
from tests.unit.test_script_continuation_run import chapter_plan


def test_initial_and_continued_schemas_have_no_mapping_or_fact_classification():
    forbidden = {"placements", "step_number", "handling", "current_facts",
                 "character_actions", "future_route_changes", "unresolved_core_gaps"}
    for model in (FirstChapterPlan, ContinuedChapterPlan):
        schema = model.model_json_schema()
        for definition in [schema, *schema["$defs"].values()]:
            assert not forbidden.intersection(definition.get("properties", {}))
    assert "continuation" not in FirstChapterPlan.model_json_schema()["properties"]
    assert "continuation" in ContinuedChapterPlan.model_json_schema()["required"]
    assert next(iter(ContinuedChapterPlan.model_fields)) == "continuation"


def test_direct_daily_scene_keeps_its_contents_and_receives_weighted_volume():
    data = chapter_plan("s1", "s2")
    first, second = data["scenes"]
    first.update(length_weight=1, start_state="放課後に二人が集まる。",
                 required_events=[{"id": "chat", "description": "夕飯の好みで軽口を交わす。"}],
                 end_state="笑いながら会話を終える。")
    second.update(length_weight=3, start_state=first["end_state"])
    plan = project_plan(FirstChapterPlan.model_validate(data), ScriptOptions())
    assert plan.continuation == ""
    for original, projected in zip(data["scenes"], plan.scenes, strict=True):
        assert projected.model_dump(exclude={"objectives"}) == {
            k: v for k, v in original.items() if k not in {"objectives", "length_weight"}}
    assert "本文約1000文字、うち台詞約500文字" in plan.scenes[0].objectives
    assert "本文約3000文字、うち台詞約1500文字" in plan.scenes[1].objectives
    assert "length_weight" not in plan.scenes[0].model_dump()
