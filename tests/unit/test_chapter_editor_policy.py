"""The chapter editor defers prose checks honestly and preserves uncertain state."""

from types import SimpleNamespace

import pytest

from packages.contracts.m3 import MappedUtterance, SceneReview
from packages.contracts.story_workflow import ChapterExtraction, StateEntry
from packages.contracts.story_workflow import ReviewReport
from packages.narrative.story_ledger import (
    apply_chapter_memory, apply_scene_memory, empty_memory, project_story_state,
    source_catalog,
)
from packages.narrative.validation import validate_narrative
from services.worker.generation import causal_narrative as causal
from services.worker.generation.causal_context import build_context
from tests.unit.test_causal_narrative import Harness, NoServerLLM
from tests.unit.test_causal_validation import bind_reviews, causal_fixture
from tests.unit.test_story_ledger import scene
from services.worker.generation.causal_runtime import CausalRun
from services.worker.generation.causal_runtime import RepairLimitError
from services.worker.generation.model_routing import resolve_purpose_profile
from services.worker.generation.pipeline import load_config
from services.worker.generation.text_debug import _draft_report
from services.worker.generation.narrative import StructuredGenerationError, _structured
from services.worker.generation.llm import ContextBudgetError
from services.worker.generation.causal_runtime import digest


def test_editor_adoption_requires_real_chapter_review_and_deferred_scene_marker():
    result, snapshot, _ = causal_fixture()
    deferred = SceneReview(policy="deferred_to_chapter", passed=False, issues=[], events=[])
    result = bind_reviews(result.model_copy(update={
        "workflow_policy": "chapter_editor_v1",
        "scenes": [item.model_copy(update={"review": deferred}) for item in result.scenes],
    }))
    assert validate_narrative(result, snapshot) == result
    assert "chapter-editor-review" in {report.scope for report in result.workflow_reviews}
    assert not any(report.scope.startswith("scene-continuity-") for report in result.workflow_reviews)
    with pytest.raises(ValueError, match="required stage"):
        validate_narrative(result.model_copy(update={"workflow_reviews":
            [report for report in result.workflow_reviews
             if report.scope != "chapter-editor-review"]}), snapshot)


def test_editor_profiles_keep_qwen_on_both_sides_of_the_gemma_writer():
    payload = {"story_workflow_version": 2, "workflow_policy": "chapter_editor_v1"}
    config = load_config()
    purposes = ["chapter-intent", "scene-sequence", "scene_text", "speech_separation",
                "extract-s1", "information-s1", "chapter-editor-review"]
    selected = [resolve_purpose_profile(config, payload, purpose) for purpose in purposes]
    assert [category for category, _ in selected] == [
        "editor_planning", "editor_planning", "editor_writer", "editor_writer",
        "editor_extraction", "editor_extraction", "editor_review"]
    assert [profile["model_id"].startswith("gemma-") for _, profile in selected] == [
        False, False, True, True, False, False, False]
    assert all(profile["context_size"] == 32768 for _, profile in selected[:2] + selected[4:])
    assert all(profile["context_size"] == 16384 for _, profile in selected[2:4])
    assert selected[-1][1]["reasoning_level"] == "none"


def test_editor_repair_budget_is_persistently_split_between_semantics_and_record():
    payload = {"chapter_number": 1, "storyline_id": "story",
               "workflow_policy": "chapter_editor_v1"}
    with CausalRun(NoServerLLM(), payload) as run:
        run.repair("chapter_semantic", "one actual prose defect")
        with pytest.raises(RepairLimitError, match="semantic"):
            run.repair("blueprint", "another semantic cycle")
        run.repair("record", "unsupported state delta")
        with pytest.raises(RepairLimitError, match="record"):
            run.repair("record", "another candidate")


def test_truncated_editor_answer_does_not_repeat_the_same_long_request():
    calls = []

    def truncated(*args, **kwargs):
        calls.append(args)
        raise ValueError("chapter-editor-review: LLM output incomplete or truncated.")

    llm = SimpleNamespace(payload={"story_workflow_version": 2,
                                   "workflow_policy": "chapter_editor_v1"},
                          config=load_config(), select_purpose=lambda purpose: None,
                          trace=[], requests=0, chat=truncated)
    with pytest.raises(StructuredGenerationError, match="source draft is retained"):
        _structured(llm, "chapter-editor-review", "原文", causal.RawDraft)
    assert len(calls) == 1


def test_chapter_editor_issue_can_cite_exact_prior_chapter_source(monkeypatch):
    first, _, _ = causal_fixture()
    second, _, _ = causal_fixture(first)
    source = next(item for item in first.story_memory.sources
                  if item.ref.scene_id == first.scenes[-1].id)
    alias = (f"prior-c{source.ref.chapter_number}-{source.ref.scene_id}-"
             f"{source.ref.utterance_id}")

    def decision(run, stage, inputs, instruction, model, validate=None):
        assert {entry["id"] for entry in inputs["previous_source_aliases"]} >= {alias}
        value = model(verdict="fail", checked_categories=["causality"],
            rationale="前章の行動と今章の開始がつながらない", issues=[{
                "code": "entry-reset", "description": "前章末の行動を巻き戻した",
                "repair_scope": "scene", "evidence_ids": [alias]}])
        validate(value)
        return value

    monkeypatch.setattr(causal, "_structured", decision)
    report = causal._gate(SimpleNamespace(payload={"chapter_number": 2,
        "storyline_id": "story"}), "chapter-editor-review",
        {"previous_chapter_end": causal._body(first.scenes[-1:])},
        "章境界を確認", ["causality"], second.scenes,
        memory=second.story_memory, subject={"test": True}, allow_failure=True)
    assert report.issues[0].evidence == [source.ref]


def test_oversized_chapter_review_splits_sources_before_compact_synthesis(monkeypatch):
    result, snapshot, _ = causal_fixture()
    calls = []

    def gate(run, stage, inputs, instruction, categories, scenes=(), **kwargs):
        calls.append(stage)
        if stage == "chapter-editor-review" and "current_sources" in inputs:
            raise ContextBudgetError("full source exceeds configured context")
        return ReviewReport(scope=stage, subject_hash=digest(kwargs["subject"]),
            verdict="pass", checked_categories=categories,
            checked_scene_ids=[item.id for item in scenes], rationale="原文を確認した")

    monkeypatch.setattr(causal, "_gate", gate)
    reports, mode = causal._editor_review(
        SimpleNamespace(payload={"chapter_number": 1}), result.blueprint.chapters[0],
        result.chapter_intent, result.blueprint, result.start_memory,
        result.story_memory, None, result.scenes, result.scene_extractions, [],
        {"Hero", "keeper"})
    assert mode == "partitioned"
    assert calls == ["chapter-editor-review", "chapter-editor-source-review-s1",
                     "chapter-editor-review"]
    assert [report.checked_scene_ids for report in reports] == [["s1"], ["s1"]]
    deferred = SceneReview(policy="deferred_to_chapter", passed=False, issues=[], events=[])
    partitioned = bind_reviews(result.model_copy(update={
        "workflow_policy": "chapter_editor_v1", "chapter_review_mode": "partitioned",
        "scenes": [item.model_copy(update={"review": deferred}) for item in result.scenes]}))
    assert "chapter-editor-source-review-s1" in {r.scope for r in partitioned.workflow_reviews}
    assert validate_narrative(partitioned, snapshot) == partitioned


def test_failed_editor_chapter_exposes_saved_source_draft(tmp_path):
    import json

    stages = tmp_path / "causal-stages"
    stages.mkdir()
    (stages / "editor-raw-s1-a.json").write_text(json.dumps({
        "value": {"text": "Hero: 原稿の言葉。"}}, ensure_ascii=False), encoding="utf-8")
    (stages / "editor-draft-s1-b.json").write_text(json.dumps({
        "value": {"id": "s1", "raw_text": "Hero: 表示用の言葉。"}},
        ensure_ascii=False), encoding="utf-8")
    _draft_report(tmp_path)
    readable = (tmp_path / "draft.md").read_text(encoding="utf-8")
    assert "表示用の言葉。" in readable and "原稿の言葉。" in readable
    assert "未採用" in readable


def test_editor_writes_all_drafts_before_extracting_and_passes_actual_previous_source(monkeypatch):
    harness = Harness(monkeypatch)
    order, preceding = [], []

    def writer(run, setting, memory, proposal, names, cast, direct_previous):
        order.append("write-" + proposal.plan.id)
        preceding.append(direct_previous)
        original = harness.write(run, setting, memory, proposal, names, cast)
        return causal.EditorDraft(**original.model_dump(exclude={"review"}))

    original_extract = causal._extract

    def extract(*args):
        order.append("extract-" + args[2].id)
        return original_extract(*args)

    monkeypatch.setattr(causal, "_write_editor_draft", writer)
    monkeypatch.setattr(causal, "_extract", extract)
    monkeypatch.setattr(causal, "_editor_adjust_records",
                        lambda run, memory, scenes, items, feedback="":
                        (items, _apply_all(memory, scenes, items)))
    payload = {"chapter_number": 1, "storyline_id": "story",
               "workflow_policy": "chapter_editor_v1", "execution_mode": "text_only"}
    with CausalRun(NoServerLLM(), payload) as run:
        result = causal._chapter(run, harness.setting, harness.blueprint,
                                 harness.initial, None, harness.cast, "")
    assert order == ["write-s1", "write-s2", "extract-s1", "extract-s2"]
    assert preceding[1]["scene"]["raw_text"] == result.scenes[0].raw_text
    assert all(scene.review.policy == "deferred_to_chapter" and not scene.review.passed
               for scene in result.scenes)
    assert not any(stage.startswith("adapt-plan-") for stage, _ in harness.calls)


def _apply_all(memory, scenes, extractions):
    for narrative, extraction in zip(scenes, extractions, strict=True):
        memory = apply_scene_memory(memory, extraction, scenes=[narrative],
                                    storyline_id="story", chapter_number=1)
    return memory


def test_record_correction_keeps_last_known_value_pending_until_later_source(monkeypatch):
    raw = "Hero: 写真を仮止めした。\nNARRATOR: まだ全枚数は確認していない。\n"
    source = scene(raw, "s1")
    first_scene = SimpleNamespace(id=source["id"], raw_text=source["raw_text"],
        utterances=[MappedUtterance.model_validate(item) for item in source["utterances"]])
    refs = source_catalog([first_scene], storyline_id="story", chapter_number=1)
    memory = empty_memory("story", state=[StateEntry(
        scope="character", entity_id="Hero", key="location", value="作業前の場所")])
    extraction = ChapterExtraction(chapter_number=1, summary="仮止めに着手した。",
        events=[{"id": "c1-s1-work", "description": "写真を仮止めした。",
                 "character_ids": ["Hero"], "location_ids": ["room"],
                 "story_time": "第1章", "evidence": refs}],
        state_deltas=[{"id": "c1-s1-location", "scope": "character",
                       "entity_id": "Hero", "key": "location",
                       "before": "作業前の場所", "after": "全作業完了地点",
                       "event_id": "c1-s1-work", "evidence": refs}])

    def decision(run, stage, inputs, instruction, model, validate=None):
        value = model(drop_state_delta_ids=["c1-s1-location"], pending_findings=[{
            "scene_id": "s1", "scope": "character", "entity_id": "Hero",
            "key": "location", "claim": "作業完了地点へ移ったか不明",
            "reason": "仮止めだけ確認できる", "evidence_ids": [refs[1].utterance_id]}],
            rationale="完了は本文にない")
        validate(value)
        return value

    monkeypatch.setattr(causal, "_structured", decision)
    corrected, current = causal._editor_adjust_records(
        SimpleNamespace(payload={"chapter_number": 1}), memory, [first_scene], [extraction])
    assert not corrected[0].state_deltas
    assert current.pending_findings[0].last_known_value == "作業前の場所"
    packet = build_context(current, scope="scene", character_ids=["Hero"])
    assert not packet["state"]
    assert packet["pending_findings"][0]["claim"] == "作業完了地点へ移ったか不明"
    assert project_story_state(current, ["Hero"]).characters[0].location == "未提示"

    second_scene = scene("Hero: 実際に移動した。", "s1")
    second_refs = source_catalog([second_scene], storyline_id="story", chapter_number=2)
    second_extraction = ChapterExtraction(chapter_number=2,
        summary="移動した。", events=[{"id": "c2-s1-move", "description": "実際に移動した。",
            "character_ids": ["Hero"], "location_ids": ["room"],
            "story_time": "第2章", "evidence": second_refs}],
        state_deltas=[{"id": "c2-s1-location", "scope": "character",
            "entity_id": "Hero", "key": "location", "before": None,
            "after": "別室", "event_id": "c2-s1-move", "evidence": second_refs}])
    stale = second_extraction.model_copy(update={"state_deltas": [
        second_extraction.state_deltas[0].model_copy(update={"before": "作業前の場所"})]})
    with pytest.raises(ValueError, match="before value"):
        apply_chapter_memory(current, stale, scenes=[second_scene],
            storyline_id="story", chapter_number=2)
    resolved = apply_chapter_memory(current, second_extraction,
        scenes=[second_scene], storyline_id="story", chapter_number=2)
    assert resolved.pending_findings[0].resolved_by == second_refs
    assert project_story_state(resolved, ["Hero"]).characters[0].location == "別室"
