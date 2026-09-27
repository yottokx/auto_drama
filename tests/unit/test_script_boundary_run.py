"""Chapter overlap correction must propagate to playback, history and resume."""

import copy

import pytest

from services.worker.generation import script_continuation as runner
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.script_cast import ScriptOptions, script_metrics
from services.worker.generation.script_chapter_plan import ContinuedChapterPlan, project_plan
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    allocation,
    chapter_plan,
    empty_cast,
    outline,
    prompt_text,
    read_json,
    snapshot,
    staging,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as _runtime

runtime = _runtime
LONG_END = "\n".join(
    f"{'aoi' if n % 2 else 'ren'}: {n}枚目はこの向きに置こう。"
    "写真の裏側に記された説明を読んでから、お客さんが隣の写真と間違えないか確かめたい。"
    "札を固定する前に少し離れて見てみよう。" for n in range(6)) + "\n"


def test_opening_is_projected_once_without_replacing_events_or_volume_targets():
    source = chapter_plan("s1", "s2", continued=True)
    source["scenes"][0]["required_events"] = [
        {"id": f"e{n}", "description": f"応答と動作{n}。"} for n in range(8)]
    draft = ContinuedChapterPlan.model_validate(source)
    before = copy.deepcopy(draft)
    plan = project_plan(draft, ScriptOptions())
    events = plan.scenes[0].required_events
    assert len(events) == 8 and events[0].id == "e0"
    assert events[0].description.startswith("【最初の新行動】" + source["continuation"])
    assert events[0].description.endswith(source["scenes"][0]["required_events"][0]["description"])
    assert [row.model_dump() for row in events[1:]] == source["scenes"][0]["required_events"][1:]
    assert plan.scenes[1].required_events == draft.scenes[1].required_events
    assert sum(row.body_characters for row in plan.scene_sizes.values()) == 4000
    assert sum(row.dialogue_characters for row in plan.scene_sizes.values()) == 2000
    assert draft == before


def test_corrected_boundary_is_used_by_history_metrics_and_export(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [empty_cast(), story_chain_draft(), allocation(),
        chapter_plan("s1"), LONG_END, staging(LONG_END),
        HANDOFF, chapter_plan("s1", continued=True), LONG_END + SECOND, staging(SECOND),
        HANDOFF, chapter_plan("s1", continued=True), FIRST, staging(FIRST)]
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path)
    assert report["status"] == "script_complete" and len(calls) == 14
    assert (tmp_path / "sources/c002-s1.raw.txt").read_text(encoding="utf-8") == LONG_END + SECOND
    for filename in ("sources/c002-s1.effective.txt", "sources/c002-s1.normalized.txt",
                     "chapters/chapter-002.md", "exports/chapter-002/sources/s1.txt"):
        assert (tmp_path / filename).read_text(encoding="utf-8").strip() == SECOND.strip()
    correction = read_json(tmp_path / "sources/c002-s1.overlap.json")
    assert correction == report["chapter_boundary_corrections"]["c002-s1"]
    assert correction["changed"] and correction["removed_utterances"] == 6
    assert correction["removed_body_characters"] == script_metrics(LONG_END)["body_characters"]
    assert report["content_metrics"]["chapters"][1]["body_characters"] == script_metrics(SECOND)["body_characters"]
    exported = read_json(tmp_path / "exports/chapter-002/script.json")
    assert len(exported["utterances"]) == 3
    assert LONG_END not in prompt_text(calls[9])  # staging only sees the accepted scene
    assert LONG_END not in prompt_text(calls[10])  # next handoff uses chapter 2 only
    state = read_json(tmp_path / "draft-state.json")
    assert state["chapters"][1]["text"].strip() == SECOND.strip()
    for index in (11, 12):
        text = prompt_text(calls[index])
        assert text.count(LONG_END.strip()) == 1  # chapter 1 history only
        assert SECOND.strip() in text
    before = len(calls)
    resumed = runner.run_script_debug(three_chapter_snapshot(), tmp_path, resume=True)
    assert len(calls) == before and resumed["chapter_boundary_corrections"] == report["chapter_boundary_corrections"]


def test_all_copy_stops_with_evidence_and_no_generation_on_resume(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [outline(), chapter_plan("s1"), LONG_END, staging(LONG_END),
                            HANDOFF, chapter_plan("s1", continued=True), LONG_END]
    for resume in (False, True):
        with pytest.raises(RuntimeError, match="No new script after removing"):
            runner.run_script_debug(snapshot(), tmp_path, resume=resume)
        assert len(calls) == 7
        report = read_json(tmp_path / "report.json")
        assert report["status"] == "failed" and report["exported_chapter_count"] == 1
        assert report["content_metrics"]["partial"]["body_characters"] == 0
        assert report["chapter_boundary_corrections"]["c002-s1"]["remaining_utterances"] == 0
        assert (tmp_path / "sources/c002-s1.raw.txt").read_text(encoding="utf-8") == LONG_END
        assert (tmp_path / "sources/c002-s1.effective.txt").read_text(encoding="utf-8") == ""


def test_resume_after_boundary_correction_reuses_source_before_staging(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [outline(), chapter_plan("s1"), LONG_END, staging(LONG_END),
        HANDOFF, chapter_plan("s1", continued=True), LONG_END + SECOND,
        GenerationCancelled("cancel staging")]
    with pytest.raises(GenerationCancelled):
        runner.run_script_debug(snapshot(), tmp_path)
    assert read_json(tmp_path / "draft-state.json")["partial"]["text"] == SECOND
    control["responses"] = [staging(SECOND)]
    report = runner.run_script_debug(snapshot(), tmp_path, resume=True)
    assert report["status"] == "script_complete" and len(calls) == 9
    assert calls[-1]["purpose"] == "script-staging" and LONG_END not in prompt_text(calls[-1])
    assert report["content_metrics"]["chapters"][1]["body_characters"] == script_metrics(SECOND)["body_characters"]


def test_length_continuation_finishes_original_anchor_before_boundary_correction(runtime, tmp_path):
    calls, control = runtime
    combined = LONG_END + SECOND
    partial = combined[:-22]
    anchor = partial[-160:]
    control["responses"] = [outline(), chapter_plan("s1"), LONG_END, staging(LONG_END),
        HANDOFF, chapter_plan("s1", continued=True),
        {"content": partial, "_finish_reason": "length"}, anchor + combined[len(partial):], staging(SECOND)]
    report = runner.run_script_debug(snapshot(), tmp_path)
    assert report["status"] == "script_complete" and len(calls) == 9
    assert partial in prompt_text(calls[7])
    assert (tmp_path / "sources/c002-s1.raw.txt").read_text(encoding="utf-8") == combined
    assert (tmp_path / "sources/c002-s1.effective.txt").read_text(encoding="utf-8") == SECOND
    assert LONG_END not in prompt_text(calls[8])


def test_chapter_opening_is_not_reissued_to_later_scene(runtime, tmp_path):
    calls, control = runtime
    second_plan = chapter_plan("s1", "s2", continued=True)
    opening = second_plan["continuation"]
    control["responses"] = [outline(), chapter_plan("s1"), FIRST, staging(FIRST),
        HANDOFF, second_plan, SECOND, staging(SECOND), FIRST, staging(FIRST, "s2")]
    runner.run_script_debug(snapshot(), tmp_path)
    writers = [call for call in calls if call["purpose"] == "script-scene"]
    assert prompt_text(writers[1]).count(opening) == 1
    assert opening not in prompt_text(writers[2])
