"""Craft material stays separate from the sole plot and the recorded script history."""

import copy

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation import script_examples
from tests.unit.test_script_continuation_run import (
    FIRST,
    allocation,
    chapter_plan,
    empty_cast,
    prompt_text,
    read_json,
    snapshot,
    staging,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime


def test_draft_basis_is_saved_but_not_in_canonical_plot_or_later_instructions(runtime, tmp_path):
    calls, control = runtime
    draft = story_chain_draft()
    draft["resolution_basis"] = [{"earlier_experience": "DRAFT_ONLY_SETUP",
                                  "later_application": "DRAFT_ONLY_METHOD"}]
    control["responses"] = [empty_cast(), draft, allocation(), chapter_plan("s1"), FIRST, staging(FIRST)]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    state = read_json(tmp_path / "draft-state.json")
    assert "DRAFT_ONLY_SETUP" in state["steps"]["story-chain"]["attempts"][0]["reply"]["content"]
    for filename in ("story-chain.json", "plot.json", "outline.json", "chapters/chapter-001.plan.json"):
        assert "DRAFT_ONLY" not in (tmp_path / filename).read_text(encoding="utf-8")
    assert not any("DRAFT_ONLY" in prompt_text(call) for call in calls[2:])
    assert len(calls) == 6


def test_exchange_is_planned_once_and_scene_instruction_keeps_the_concrete_exchange(runtime, tmp_path):
    calls, control = runtime
    chapters = allocation()
    chapters["chapters"][0]["conversation_topics"] = [{"character_ids": ["aoi", "ren"],
        "topic": "休憩に誘いたい。", "exchange": "PLAN_ONLY_EXCHANGE"}]
    plan = chapter_plan("s1")
    plan["scenes"][0]["objectives"] = "葵は休憩したいが、蓮は展示を仕上げたい。"
    plan["scenes"][0]["required_events"][0]["description"] = "葵が休憩に誘い、蓮は区切りを付けてからと返す。"
    control["responses"] = [empty_cast(), story_chain_draft(), chapters, plan, FIRST, staging(FIRST)]
    runner.run_script_debug(three_chapter_snapshot(), tmp_path, chapter_limit=1)
    assert "PLAN_ONLY_EXCHANGE" in prompt_text(calls[3])
    writer = prompt_text(calls[4])
    assert "PLAN_ONLY_EXCHANGE" not in writer
    assert plan["scenes"][0]["objectives"] in writer
    assert plan["scenes"][0]["required_events"][0]["description"] in writer


def test_examples_do_not_enter_handoff_or_staging_and_are_recorded_in_requests(runtime, tmp_path):
    calls, _ = runtime
    report = runner.run_script_debug(snapshot(), tmp_path)
    assert report["script_examples"] == script_examples.example_catalog_identity()
    state = read_json(tmp_path / "draft-state.json")
    for key, step in state["steps"].items():
        for row in step["attempts"]:
            candidate = script_examples.example_for(row["purpose"])
            record = read_json(tmp_path / "requests" / f"{key}-{row['attempt']}.json")
            if candidate:
                assert record["selection"]["example"]["sha256"] == candidate["sha256"]
                assert record["selection"]["example"]["included"] is True
            else:
                assert record["selection"]["example"] is None
    for call in calls:
        if call["purpose"] in {"script-handoff", "script-staging", "script-speech"}:
            assert not any(example["text"] in prompt_text(call) for example in script_examples.EXAMPLES.values())
    assert "sample-a:" not in state["chapters"][0]["text"]


def test_changed_example_catalog_cannot_resume_saved_work(runtime, tmp_path, monkeypatch):
    calls, _ = runtime
    runner.run_script_debug(snapshot(), tmp_path, chapter_limit=1)
    count = len(calls)
    changed = copy.deepcopy(script_examples.EXAMPLES["script-exchange"])
    changed["text"] += "\n改訂された作例。"
    monkeypatch.setitem(script_examples.EXAMPLES, "script-exchange", changed)
    with pytest.raises(ValueError, match="protocol or resource limits changed"):
        runner.run_script_debug(snapshot(), tmp_path, resume=True)
    assert len(calls) == count
