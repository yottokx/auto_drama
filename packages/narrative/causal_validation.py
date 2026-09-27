"""Adoption checks for causal narratives, independent of worker implementation.

The adopted memory must be reproducible from exact source and recorded deltas.
Semantic reviews are bound to the artifacts they checked and must cover every
scene; a label saying 'passed' without that coverage is insufficient.
"""

from __future__ import annotations

from packages.contracts.m3 import NarrativeResult

from .story_ledger import (
    apply_scene_memory,
    content_hash,
    empty_memory,
    memory_hash,
    project_story_state,
    validate_evidence,
)


def _unique(values, label):
    values = list(values)
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate {label} in causal narrative.")


def causal_review_bindings(result: NarrativeResult) -> dict[str, tuple[dict | list, set, list]]:
    """Return canonical review subjects, categories and original-source coverage.

    The generation layer hashes the same subjects when it reviews them. The
    full prompt input gets a separate input_hash for audit, not a replacement
    for this verifiable binding to the adopted artifact.
    """
    scenes = result.scenes
    result_bindings = {
        "chapter-intent-review": (
            result.chapter_intent.model_dump(mode="json"),
            {"causality", "progression", "repetition", "entry_state", "remaining_story"}, []),
        "scene-sequence-review": (
            {"scenes": [{"scene_id": s.scene_id, "location_id": s.location_id,
                         "character_ids": s.character_ids} for s in result.scene_intents],
             "locations": [location.model_dump(mode="json") for location in result.locations],
             "location_registry": [location.model_dump(mode="json")
                                   for location in result.location_registry]},
            {"causality", "progression", "repetition", "state", "dramatization"}, []),
        "chapter-progress-review": (
            {"intent": result.chapter_intent.model_dump(mode="json"),
             "memory": result.story_memory.model_dump(mode="json"),
             "scenes": [scene.model_dump(mode="json") for scene in scenes]},
            {"progression", "causality", "open_threads", "ending"}, scenes),
    }
    if result.workflow_policy == "chapter_editor_v1":
        result_bindings.pop("chapter-progress-review")
        if result.chapter_review_mode == "partitioned":
            for scene in scenes:
                result_bindings["chapter-editor-source-review-" + scene.id] = (
                    scene.model_dump(mode="json"),
                    {"causality", "knowledge", "repetition", "record_support"}, [scene])
        result_bindings["chapter-editor-review"] = (
            {"intent": result.chapter_intent.model_dump(mode="json"),
             "memory": result.story_memory.model_dump(mode="json"),
             "scenes": [scene.model_dump(mode="json") for scene in scenes]},
            {"causality", "progression", "repetition", "knowledge", "ending", "record_support"},
            scenes)
        if result.chapter_number == 1:
            result_bindings["blueprint-review"] = (
                result.blueprint.model_dump(mode="json"),
                {"causality", "progression", "ending_preparation", "fixed_conditions"}, [])
        return result_bindings
    predecessor = result.start_memory
    for scene, extraction in zip(scenes, result.scene_extractions, strict=True):
        result_bindings["fact-comparison-" + scene.id] = (
            {"memory_hash": memory_hash(predecessor), "scene": scene.model_dump(mode="json")},
            {"knowledge_preservation"}, [scene])
        result_bindings["scene-continuity-" + scene.id] = (
            {"scene": scene.model_dump(mode="json")},
            {"causality", "progression", "knowledge", "state", "repetition"}, [scene])
        result_bindings["extraction-review-" + scene.id] = (
            {"scene": scene.model_dump(mode="json"), "extraction": extraction.model_dump(mode="json")},
            {"support", "completeness", "knowledge", "state", "promises", "introductions"}, [scene])
        result_bindings["state-comparison-" + scene.id] = (
            {"scene": scene.model_dump(mode="json"), "extraction": extraction.model_dump(mode="json")},
            {"state_support", "action_stage", "object_identity"}, [scene])
        predecessor = apply_scene_memory(predecessor, extraction, scenes=[scene],
            storyline_id=result.storyline_id, chapter_number=result.chapter_number)
    if result.chapter_number > 1:
        result_bindings["chapter-boundary-review"] = (
            {"previous_memory_hash": memory_hash(result.start_memory),
             "scene": scenes[0].model_dump(mode="json"),
             "entry_bridge": result.chapter_intent.entry_bridge},
            {"causality", "state", "knowledge", "repetition"}, scenes[:1])
    else:
        result_bindings["blueprint-review"] = (
            result.blueprint.model_dump(mode="json"),
            {"causality", "progression", "ending_preparation", "fixed_conditions"}, [])
    return result_bindings


def _validate_reviews(result: NarrativeResult, *, changed_blueprint: bool) -> None:
    _unique((report.scope for report in result.workflow_reviews), "review scopes")
    reports = {report.scope: report for report in result.workflow_reviews}
    expected = causal_review_bindings(result)
    if changed_blueprint:
        expected["blueprint-review"] = (
            result.blueprint.model_dump(mode="json"),
            {"causality", "progression", "ending_preparation", "fixed_conditions"}, [])
    if set(reports) != set(expected):
        raise ValueError("Causal reviews must cover every required stage exactly once.")
    event_ids = {event.id for event in result.story_memory.events}
    for scope, (subject, categories, scenes) in expected.items():
        report = reports[scope]
        if report.verdict != "pass" or report.missing_information or any(
                issue.severity == "error" or issue.missing_information for issue in report.issues):
            raise ValueError("Causal chapter has a failed or incomplete review.")
        if report.subject_hash != content_hash(subject):
            raise ValueError(f"Review {scope} refers to a different or stale subject.")
        if not report.rationale:
            raise ValueError("Causal review must record its evidence-based rationale.")
        _unique(report.checked_categories, "review categories")
        if set(report.checked_categories) != categories:
            raise ValueError("Causal review omitted or changed its required categories.")
        if report.checked_scene_ids != [scene.id for scene in scenes]:
            raise ValueError("Causal review does not cover its exact original source scope.")
        _unique(report.checked_event_ids, "review event IDs")
        if not set(report.checked_event_ids).issubset(event_ids):
            raise ValueError("Causal review cites an unknown observed event.")
        for issue in report.issues:
            validate_evidence(issue.evidence, narratives=[result], sources=result.story_memory.sources)


def validate_causal_continuity(value: dict | NarrativeResult, character_ids: set[str], *,
                               previous_narrative: dict | NarrativeResult | None = None) -> None:
    result = NarrativeResult.model_validate(value)
    if result.workflow_version != 2:
        raise ValueError("Causal validation cannot automatically convert a legacy story.")
    if not result.storyline_id or result.start_memory is None or result.story_memory is None:
        raise ValueError("Causal narrative requires a storyline and start/end memory.")
    start, end = result.start_memory, result.story_memory
    number = result.chapter_number
    if (start.storyline_id != result.storyline_id or end.storyline_id != result.storyline_id
            or start.chapter_number != number - 1 or end.chapter_number != number):
        raise ValueError("Causal memory has incorrect chapter or storyline lineage.")
    if result.predecessor_state_inferred:
        raise ValueError("Causal stories cannot use inferred legacy predecessor state.")
    if end.previous_memory_hash != memory_hash(start):
        raise ValueError("Causal chapter must name the exact predecessor memory hash.")
    if result.start_state is None or result.end_state is None or result.continuity_review is None:
        raise ValueError("Causal narrative requires its compatibility state projection.")
    if result.chapter_intent is None or result.blueprint is None:
        raise ValueError("Causal narrative requires its blueprint and chapter intent.")
    intent, blueprint = result.chapter_intent, result.blueprint
    if (intent.chapter_number != number or intent.blueprint_revision != blueprint.revision
            or intent.predecessor_memory_hash != memory_hash(start)):
        raise ValueError("Chapter intent must use the adopted blueprint and exact preceding memory.")
    if (len(blueprint.chapters) != len(result.outline.chapters)
            or number > len(blueprint.chapters)
            or [c.number for c in result.outline.chapters] != list(range(1, len(blueprint.chapters) + 1))):
        raise ValueError("Causal blueprint and outline must preserve the chapter count/order.")
    if not set(intent.inherited_event_ids).issubset({e.id for e in start.events}):
        raise ValueError("Chapter intent inherited a planned or nonexistent event.")
    if not set(intent.inherited_thread_ids).issubset({t.id for t in start.threads if t.status == "open"}):
        raise ValueError("Chapter intent inherited a nonexistent or already closed thread.")
    if not set(intent.character_ids).issubset(character_ids):
        raise ValueError("Chapter intent references an unknown character.")
    _unique((c.id for c in result.supporting_characters), "supporting character IDs")
    supporting_ids = {c.id for c in result.supporting_characters}
    if not supporting_ids.issubset(character_ids):
        raise ValueError("Supporting character registry is inconsistent with the cast.")
    main_ids = character_ids - supporting_ids
    _unique((location.id for location in result.location_registry), "location registry IDs")
    registry = {location.id: location for location in result.location_registry}
    for location in result.location_registry:
        if location.introduced_chapter > number:
            raise ValueError("Location registry cannot introduce a place in a future chapter.")
    for location in result.locations:
        if location.id not in registry or location.name != registry[location.id].name:
            raise ValueError("Chapter locations must resolve to the registered place identity/name.")
    for item in [*start.state, *end.state, *start.state_history, *end.state_history]:
        if item.scope == "location" and item.entity_id not in registry:
            raise ValueError("Location state must refer to a registered place.")
    for event in end.events:
        if not set(event.location_ids).issubset(registry):
            raise ValueError("Observed history references an unregistered place.")
    start_ids = {c.character_id for c in result.start_state.characters}
    if not main_ids.issubset(start_ids) or not start_ids.issubset(character_ids):
        raise ValueError("Starting state must retain the known cast, including every main character.")
    changed_blueprint = False
    if number == 1:
        if (previous_narrative is not None or result.previous_narrative_artifact_id
                or result.previous_state_hash):
            raise ValueError("First causal chapter cannot specify a predecessor artifact.")
        if start_ids != main_ids:
            raise ValueError("First chapter starts with main characters, before supporting cast is adopted.")
        if start != empty_memory(result.storyline_id, state=start.state, author_facts=start.author_facts):
            raise ValueError("Initial memory cannot contain events or planned developments as history.")
    elif (not result.previous_narrative_artifact_id
          or result.previous_state_hash != content_hash(result.start_state)):
        raise ValueError("Causal continuation must name its adopted predecessor and state hash.")
    if previous_narrative is not None:
        previous = NarrativeResult.model_validate(previous_narrative)
        if previous.workflow_version != 2 or previous.story_memory is None:
            raise ValueError("Legacy stories cannot be automatically converted into causal continuation.")
        if previous.chapter_number + 1 != number or previous.storyline_id != result.storyline_id:
            raise ValueError("Causal predecessor chapter or storyline does not match.")
        if previous.story_memory != start or previous.end_state != result.start_state:
            raise ValueError("Causal continuation must use the exact adopted predecessor memory/state.")
        current_supporting = {c.id: c for c in result.supporting_characters}
        if any(current_supporting.get(c.id) != c for c in previous.supporting_characters):
            raise ValueError("Previously adopted supporting identities/settings must be preserved.")
        old_registry = {location.id: location for location in previous.location_registry}
        if any(registry.get(key) != location for key, location in old_registry.items()):
            raise ValueError("Previously adopted location identities must be preserved.")
        if any(location.introduced_chapter != number for key, location in registry.items()
               if key not in old_registry):
            raise ValueError("New location identities must be introduced in the current chapter.")
        if len(previous.blueprint.chapters) != len(blueprint.chapters):
            raise ValueError("Causal continuation cannot change the approved chapter count.")
        changed_blueprint = previous.blueprint != blueprint
        if changed_blueprint and (
                blueprint.revision <= previous.blueprint.revision or not blueprint.revision_reason
                or blueprint.immutable_conditions != previous.blueprint.immutable_conditions
                or blueprint.ending_conditions != previous.blueprint.ending_conditions
                or blueprint.chapters[:number - 1] != previous.blueprint.chapters[:number - 1]):
            raise ValueError("Blueprint revisions may adjust the future with reasons, never adopted "
                             "history or fixed ending conditions.")

    if (len(result.scenes) != len(result.scene_intents)
            or len(result.scenes) != len(result.scene_extractions)):
        raise ValueError("Every scene requires exactly one intent and extraction.")
    _unique((scene.id for scene in result.scenes), "scene IDs")
    location_ids = {location.id for location in result.locations}
    replay = start
    for scene, scene_intent, extraction in zip(result.scenes, result.scene_intents,
                                              result.scene_extractions, strict=True):
        if (scene_intent.scene_id != scene.id or scene.plan.id != scene.id
                or scene_intent.chapter_number != number
                or scene_intent.character_ids != scene.plan.character_ids
                or scene_intent.location_id != scene.plan.location_id):
            raise ValueError("Scene intent must match the exact scene, cast and location.")
        if (not set(scene_intent.character_ids).issubset(character_ids)
                or scene_intent.location_id not in location_ids
                or scene_intent.viewpoint_character_id is not None
                and scene_intent.viewpoint_character_id not in character_ids):
            raise ValueError("Scene intent references an unregistered character or place.")
        if not set(scene_intent.inherited_event_ids).issubset({e.id for e in replay.events}):
            raise ValueError("Scene intent inherited an event that has not happened yet.")
        for event in extraction.events:
            if (not set(event.character_ids).issubset(character_ids)
                    or not set(event.location_ids).issubset(location_ids)):
                raise ValueError("Observed events reference an unregistered character or place.")
        for delta in extraction.state_deltas:
            if delta.scope == "character" and delta.entity_id not in character_ids:
                raise ValueError("A state change references an unregistered character.")
            if delta.scope == "location" and delta.entity_id not in registry:
                raise ValueError("A location state change references an unregistered place.")
        for update in extraction.knowledge_updates:
            if update.character_id not in character_ids:
                raise ValueError("Knowledge references an unregistered character.")
        replay = apply_scene_memory(replay, extraction, scenes=[scene],
            storyline_id=result.storyline_id, chapter_number=number)
    if replay != end:
        raise ValueError("Adopted memory is not the deterministic replay of verified scene extractions.")
    if (result.start_state != project_story_state(start, start_ids)
            or result.end_state != project_story_state(end, character_ids)):
        raise ValueError("Compatibility state must be an exact projection of causal memory.")
    review = result.continuity_review
    if (not review.passed or review.issues or review.checked_foreshadowing_indices
            or sorted(review.checked_character_ids) != sorted(character_ids)):
        raise ValueError("Compatibility continuity review has missing cast coverage or unresolved issues.")
    _validate_reviews(result, changed_blueprint=changed_blueprint)
