"""Causal adoption refuses forged memory, stale reviews and partial coverage."""

import copy

import pytest

from packages.contracts.m2 import CharacterResult
from packages.contracts.m3 import NarrativeResult, NarrativeScene
from packages.contracts.m4 import ContinuityReview
from packages.contracts.story_workflow import (
    ChapterExtraction,
    ChapterIntent,
    LocationIdentity,
    ReviewIssue,
    ReviewReport,
    SceneIntent,
    StoryBlueprint,
)
from packages.narrative.causal_validation import causal_review_bindings, validate_causal_continuity
from packages.narrative.story_ledger import (
    apply_scene_memory,
    content_hash,
    empty_memory,
    memory_hash,
    project_story_state,
    source_catalog,
)
from packages.narrative.validation import parse_scene_text, validate_narrative
from tests.unit.test_m3_narrative import narrative_fixture

CAST = {"Hero", "keeper"}
PLACES = {
    "gate": {"id": "gate", "name": "城門", "structural_description": "街の入口にある石の門"},
    "hallway": {"id": "hallway", "name": "城内の廊下", "structural_description": "門の内側から薬室へ続く廊下"},
}


def bind_reviews(result):
    reports = [ReviewReport(scope=scope, subject_hash=content_hash(subject),
        input_hash=content_hash({"test_input": subject}), verdict="pass",
        checked_categories=sorted(categories), checked_scene_ids=[s.id for s in scenes],
        rationale="対象本文と前章の原文根拠を照合し、必要な変化と接続を確認した。")
        for scope, (subject, categories, scenes) in causal_review_bindings(result).items()]
    return result.model_copy(update={"workflow_reviews": reports})


def supporting_character(character_id="support-c1-1"):
    return CharacterResult(id=character_id, name="案内人", age="成人", gender="未指定",
        role="道を教える", freeform="", settings="町の歴史に詳しい", appearance="青い服",
        voice="落ち着いた声", locked={}, selfIntroduction="私は案内人です。",
        sampleLines=["ここは古い道です。", "気を付けて。", "また会いましょう。"])


def causal_fixture(previous=None, *, add_support=False, location_id="gate"):
    document, snapshot = narrative_fixture()
    number = previous.chapter_number + 1 if previous else 1
    document["outline"]["foreshadowing"] = []
    if previous:
        document["outline"] = previous.outline.model_dump()
    raw_scene = document["scenes"][0]
    raw_scene["plan"]["location_id"] = location_id
    place = PLACES[location_id]
    document["locations"] = [{"id": location_id, "name": place["name"],
        "description": place["structural_description"], "time_of_day": "夕方",
        "atmosphere": "落ち着いた", "image_prompt": "stone architecture, no people"}]
    registry = list(previous.location_registry) if previous else []
    if location_id not in {item.id for item in registry}:
        registry.append(LocationIdentity(**place, introduced_chapter=number))
    if previous:
        raw_scene["raw_text"] = raw_scene["raw_text"].replace("扉を開けてくれ。", "薬を届けに進もう。")
        raw_scene["utterances"] = [u.model_dump() for u in parse_scene_text(
            raw_scene["raw_text"], "s1", CAST)]
    scene = NarrativeScene.model_validate(raw_scene)
    supporting = list(previous.supporting_characters) if previous else []
    if add_support:
        supporting.append(supporting_character(f"support-c{number}-1"))
    cast = CAST | {c.id for c in supporting}
    start = previous.story_memory if previous else empty_memory("story")
    blueprint = previous.blueprint if previous else StoryBlueprint(
        central_question="薬を届けられるか", ending_conditions=["薬を届ける"],
        chapters=[{"number": n, "question": f"{n}番目の障害をどう越えるか",
                   "unique_progress": [f"第{n}段階を越える"], "ending_conditions": [f"第{n}地点へ進む"]}
                  for n in range(1, 4)])
    intent = ChapterIntent(chapter_number=number, predecessor_memory_hash=memory_hash(start),
        question=blueprint.chapters[number - 1].question, entry_bridge="前章の結果から行動を続ける。",
        unique_progress=["信頼を得て薬を運ぶ"], desired_end=["二人で行動する"],
        character_ids=sorted(CAST), inherited_event_ids=[e.id for e in start.events][-1:])
    scene_intent = SceneIntent(scene_id="s1", chapter_number=number,
        character_ids=scene.plan.character_ids, location_id=location_id, story_time=f"第{number}日の夕方",
        entry_bridge="城門へ到着した。", motive="薬を届けたい", obstacle="門番が通さない",
        planned_action="薬を見せて頼む", expected_change="門番が協力する")
    refs = source_catalog([scene], storyline_id="story", chapter_number=number)
    extraction = ChapterExtraction(chapter_number=number, summary="門番が薬を届けるため同行した。",
        events=[{"id": f"c{number}_persuasion", "description": "薬を見せ、門番が同行に同意した。",
                 "character_ids": sorted(CAST), "location_ids": [location_id],
                 "story_time": scene_intent.story_time, "evidence": refs[3:6],
                 "causes": [e.id for e in start.events][-1:]}])
    end = apply_scene_memory(start, extraction, scenes=[scene], storyline_id="story", chapter_number=number)
    first_state = previous.end_state if previous else project_story_state(start, CAST)
    document.update(chapter_number=number, workflow_version=2, storyline_id="story",
        previous_narrative_artifact_id=f"artifact-{number - 1}" if previous else None,
        previous_state_hash=content_hash(first_state) if previous else None,
        supporting_characters=[c.model_dump() for c in supporting],
        location_registry=[item.model_dump() for item in registry],
        start_state=first_state.model_dump(), end_state=project_story_state(end, cast).model_dump(),
        continuity_review=ContinuityReview(passed=True, issues=[],
            checked_character_ids=sorted(cast), checked_foreshadowing_indices=[]).model_dump(),
        blueprint=blueprint.model_dump(), chapter_intent=intent.model_dump(),
        scene_intents=[scene_intent.model_dump()], start_memory=start.model_dump(),
        story_memory=end.model_dump(), scene_extractions=[extraction.model_dump()],
        workflow_reviews=[ReviewReport(scope="placeholder", subject_hash="a" * 64,
                                       verdict="pass").model_dump()])
    result = bind_reviews(NarrativeResult.model_validate(document))
    return result, snapshot, cast


def test_new_story_and_continuation_pass_source_replay_and_full_adoption():
    first, snapshot, cast = causal_fixture()
    assert validate_narrative(first, snapshot) == first
    second, _, second_cast = causal_fixture(first)
    validate_causal_continuity(second, second_cast, previous_narrative=first)
    assert validate_narrative(second, snapshot, previous_narrative=first) == second
    assert second.start_memory == first.story_memory
    assert second.story_memory.previous_memory_hash == memory_hash(first.story_memory)
    assert cast == second_cast


def test_revisit_retains_place_identity_across_an_absent_chapter():
    first, snapshot, cast = causal_fixture()
    second, _, _ = causal_fixture(first, location_id="hallway")
    third, _, _ = causal_fixture(second, location_id="gate")
    for current, previous in ((first, None), (second, first), (third, second)):
        validate_narrative(current, snapshot, previous_narrative=previous)
    assert [location.id for location in second.locations] == ["hallway"]
    assert [location.id for location in third.locations] == ["gate"]
    assert third.location_registry == second.location_registry
    assert third.location_registry[0] == first.location_registry[0]
    assert [location.introduced_chapter for location in third.location_registry] == [1, 2]
    assert cast == CAST


@pytest.mark.parametrize("field,value", [
    ("name", "病室"), ("structural_description", "窓のない病室"), ("introduced_chapter", 2),
])
def test_previously_registered_place_cannot_change_even_with_fresh_reviews(field, value):
    first, _, _ = causal_fixture()
    second, _, cast = causal_fixture(first, location_id="hallway")
    registry = [second.location_registry[0].model_copy(update={field: value}),
                second.location_registry[1]]
    changed = bind_reviews(second.model_copy(update={"location_registry": registry}))
    with pytest.raises(ValueError, match="location identities must be preserved"):
        validate_causal_continuity(changed, cast, previous_narrative=first)


def test_absent_place_cannot_be_removed_from_the_registry():
    first, _, _ = causal_fixture()
    second, _, cast = causal_fixture(first, location_id="hallway")
    changed = bind_reviews(second.model_copy(update={"location_registry": second.location_registry[1:]}))
    with pytest.raises(ValueError, match="unregistered place|location identities must be preserved"):
        validate_causal_continuity(changed, cast, previous_narrative=first)


@pytest.mark.parametrize("field,value", [
    ("name", "病室"), ("description", "全く別の部屋"), ("time_of_day", "深夜"),
    ("atmosphere", "荒廃した"), ("image_prompt", "hospital bed"),
])
def test_scene_review_is_bound_to_current_place_details(field, value):
    first, _, _ = causal_fixture()
    second, _, cast = causal_fixture(first)
    changed = second.model_copy(update={"locations": [
        second.locations[0].model_copy(update={field: value})]})
    with pytest.raises(ValueError, match="identity/name|different or stale subject"):
        validate_causal_continuity(changed, cast, previous_narrative=first)


def test_reviewed_time_and_condition_change_keeps_the_same_place_identity():
    first, _, _ = causal_fixture()
    second, _, cast = causal_fixture(first)
    changed = bind_reviews(second.model_copy(update={"locations": [second.locations[0].model_copy(
        update={"time_of_day": "夜", "description": "石の門の横にランタンが灯る"})]}))
    validate_causal_continuity(changed, cast, previous_narrative=first)
    assert changed.location_registry == first.location_registry


def test_location_registry_rejects_duplicate_future_and_backdated_new_places():
    first, _, cast = causal_fixture()
    duplicate = first.model_copy(update={"location_registry": first.location_registry * 2})
    with pytest.raises(ValueError, match="Duplicate location registry"):
        validate_causal_continuity(duplicate, cast)
    future = first.model_copy(update={"location_registry": [
        first.location_registry[0].model_copy(update={"introduced_chapter": 2})]})
    with pytest.raises(ValueError, match="future chapter"):
        validate_causal_continuity(future, cast)
    second, _, _ = causal_fixture(first, location_id="hallway")
    backdated = second.model_copy(update={"location_registry": [second.location_registry[0],
        second.location_registry[1].model_copy(update={"introduced_chapter": 1})]})
    with pytest.raises(ValueError, match="introduced in the current chapter"):
        validate_causal_continuity(backdated, cast, previous_narrative=first)


@pytest.mark.parametrize("target", ["start_memory", "story_memory", "extraction"])
def test_location_scoped_state_cannot_reference_an_unregistered_place(target):
    result, _, cast = causal_fixture()
    document = result.model_dump()
    if target == "extraction":
        event = document["scene_extractions"][0]["events"][0]
        document["scene_extractions"][0]["state_deltas"].append({
            "id": "unknown-place-state", "scope": "location", "entity_id": "unknown-place",
            "key": "condition", "before": None, "after": "damaged", "event_id": event["id"],
            "evidence": event["evidence"]})
    else:
        document[target]["state"].append({"scope": "location", "entity_id": "unknown-place",
            "key": "condition", "value": "damaged", "changed_chapter": 0, "evidence": []})
        if target == "start_memory":
            document["story_memory"]["previous_memory_hash"] = memory_hash(document[target])
            document["chapter_intent"]["predecessor_memory_hash"] = memory_hash(document[target])
    with pytest.raises(ValueError, match="[Ll]ocation state.*registered place"):
        validate_causal_continuity(document, cast)


def test_new_supporting_cast_can_join_without_rewriting_start_state_or_old_settings():
    first, _, cast = causal_fixture(add_support=True)
    validate_causal_continuity(first, cast)
    assert {c.character_id for c in first.start_state.characters} == CAST
    assert len(first.end_state.characters) == 3
    second, _, second_cast = causal_fixture(first, add_support=True)
    validate_causal_continuity(second, second_cast, previous_narrative=first)
    assert len(second.end_state.characters) == 4
    changed = second.model_copy(update={"supporting_characters": [
        second.supporting_characters[0].model_copy(update={"settings": "過去の設定を変更"}),
        second.supporting_characters[1]]})
    with pytest.raises(ValueError, match="identities/settings"):
        validate_causal_continuity(changed, second_cast, previous_narrative=first)


@pytest.mark.parametrize("change", [
    lambda v: v["story_memory"].update(summary="予定の要約に差し替えた"),
    lambda v: v["story_memory"]["events"][0].update(description="原文では起きていない出来事"),
    lambda v: v["story_memory"]["state"].append({"scope": "character", "entity_id": "Hero",
        "key": "physical_condition", "value": "治癒した", "changed_chapter": 1, "evidence": []}),
])
def test_adopted_memory_must_be_exact_replay_not_an_arbitrary_summary(change):
    result, _, cast = causal_fixture()
    document = result.model_dump()
    change(document)
    with pytest.raises(ValueError, match="deterministic replay"):
        validate_causal_continuity(document, cast)


def test_start_memory_chain_and_predecessor_state_are_both_immutable():
    first, _, _ = causal_fixture()
    second, _, cast = causal_fixture(first)
    altered = second.model_dump()
    altered["start_memory"]["summary"] = "前章で別のことが起きた"
    with pytest.raises(ValueError, match="predecessor memory hash"):
        validate_causal_continuity(altered, cast, previous_narrative=first)
    altered = second.model_dump()
    altered["previous_state_hash"] = "a" * 64
    with pytest.raises(ValueError, match="predecessor and state hash"):
        validate_causal_continuity(altered, cast, previous_narrative=first)


def test_legacy_predecessor_cannot_be_implicitly_reconstructed():
    first, _, _ = causal_fixture()
    second, _, cast = causal_fixture(first)
    legacy, _ = narrative_fixture()
    with pytest.raises(ValueError, match="Legacy stories"):
        validate_causal_continuity(second, cast, previous_narrative=legacy)


def test_missing_scene_extraction_and_mismatched_scene_intent_are_rejected():
    result, _, cast = causal_fixture()
    second_extraction = result.scene_extractions[0].model_copy(deep=True)
    damaged = result.model_copy(update={"scene_extractions": [*result.scene_extractions, second_extraction]})
    with pytest.raises(ValueError, match="exactly one intent and extraction"):
        validate_causal_continuity(damaged, cast)
    damaged = result.model_copy(update={"scene_intents": [
        result.scene_intents[0].model_copy(update={"location_id": "hallway"})]})
    with pytest.raises(ValueError, match="exact scene, cast and location"):
        validate_causal_continuity(damaged, cast)


@pytest.mark.parametrize("scope", ["blueprint-review", "chapter-intent-review",
    "scene-sequence-review", "scene-continuity-s1", "extraction-review-s1", "chapter-progress-review"])
def test_a_passing_review_for_an_old_subject_is_not_accepted(scope):
    result, _, cast = causal_fixture()
    reviews = [r.model_copy(update={"subject_hash": "a" * 64}) if r.scope == scope else r
               for r in result.workflow_reviews]
    with pytest.raises(ValueError, match="different or stale subject"):
        validate_causal_continuity(result.model_copy(update={"workflow_reviews": reviews}), cast)


def test_a_missing_review_or_category_does_not_claim_whole_chapter_coverage():
    result, _, cast = causal_fixture()
    without_scene = [r for r in result.workflow_reviews if r.scope != "scene-continuity-s1"]
    with pytest.raises(ValueError, match="every required stage"):
        validate_causal_continuity(result.model_copy(update={"workflow_reviews": without_scene}), cast)
    reviews = copy.deepcopy(result.workflow_reviews)
    index = next(i for i, r in enumerate(reviews) if r.scope == "scene-continuity-s1")
    reviews[index] = reviews[index].model_copy(update={"checked_categories": ["causality"]})
    with pytest.raises(ValueError, match="required categories"):
        validate_causal_continuity(result.model_copy(update={"workflow_reviews": reviews}), cast)


def test_source_coverage_cannot_claim_an_unchecked_scene():
    result, _, cast = causal_fixture()
    reviews = [r.model_copy(update={"checked_scene_ids": []})
               if r.scope == "scene-continuity-s1" else r for r in result.workflow_reviews]
    with pytest.raises(ValueError, match="exact original source scope"):
        validate_causal_continuity(result.model_copy(update={"workflow_reviews": reviews}), cast)


def test_insufficient_evidence_is_never_publishable():
    result, _, cast = causal_fixture()
    reviews = [r.model_copy(update={"verdict": "insufficient_evidence",
               "missing_information": ["前章の原文が不足"]})
               if r.scope == "scene-continuity-s1" else r for r in result.workflow_reviews]
    with pytest.raises(ValueError, match="failed or incomplete review"):
        validate_causal_continuity(result.model_copy(update={"workflow_reviews": reviews}), cast)


def test_compatibility_state_is_only_a_projection_and_cannot_override_the_ledger():
    result, _, cast = causal_fixture()
    end = result.end_state.model_copy(update={"summary": "予定を達成したという別の要約"})
    with pytest.raises(ValueError, match="exact projection"):
        validate_causal_continuity(result.model_copy(update={"end_state": end}), cast)


def test_review_evidence_hash_must_resolve_to_the_original_prose():
    result, _, cast = causal_fixture()
    ref = source_catalog(result.scenes, storyline_id="story", chapter_number=1)[0]
    invalid = ref.model_copy(update={"text_hash": "a" * 64})
    issue = ReviewIssue(code="minor_style", severity="warning", description="改善余地",
                        repair_scope="scene", evidence=[invalid])
    reviews = [r.model_copy(update={"issues": [issue]})
               if r.scope == "scene-continuity-s1" else r for r in result.workflow_reviews]
    with pytest.raises(ValueError, match="text hash"):
        validate_causal_continuity(result.model_copy(update={"workflow_reviews": reviews}), cast)


def test_blueprint_cannot_rewrite_the_already_adopted_chapter_plan():
    first, _, _ = causal_fixture()
    second, _, cast = causal_fixture(first)
    blueprint = second.blueprint.model_dump()
    blueprint.update(revision=2, revision_reason="構成の調整")
    blueprint["chapters"][0]["unique_progress"] = ["前章を別の出来事に変更"]
    changed = second.model_copy(update={"blueprint": StoryBlueprint.model_validate(blueprint),
        "chapter_intent": second.chapter_intent.model_copy(update={"blueprint_revision": 2})})
    with pytest.raises(ValueError, match="never adopted history"):
        validate_causal_continuity(changed, cast, previous_narrative=first)


def test_revised_blueprint_keeps_fixed_ending_even_with_fresh_passing_reviews():
    first, _, _ = causal_fixture()
    second, _, cast = causal_fixture(first)

    def revised(ending):
        blueprint = second.blueprint.model_copy(update={
            "revision": 2, "revision_reason": "残りの行動手順を調整", "ending_conditions": ending})
        result = bind_reviews(second.model_copy(update={"blueprint": blueprint,
            "chapter_intent": second.chapter_intent.model_copy(update={"blueprint_revision": 2})}))
        review = ReviewReport(scope="blueprint-review", subject_hash=content_hash(blueprint),
            verdict="pass", rationale="採用済みの章と結末条件を保ち残りの進行を確認した。",
            checked_categories=["causality", "progression", "ending_preparation", "fixed_conditions"])
        return result.model_copy(update={"workflow_reviews": [review, *result.workflow_reviews]})

    valid = revised(first.blueprint.ending_conditions)
    validate_causal_continuity(valid, cast, previous_narrative=first)
    with pytest.raises(ValueError, match="fixed ending conditions"):
        validate_causal_continuity(revised(["薬を届けず計画を放棄する"]), cast, previous_narrative=first)
