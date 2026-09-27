"""Scoped prompt memory keeps evidence addressable without replaying all history."""

import copy

import pytest

from packages.contracts.story_workflow import ChapterMemory, ReviewReport, SourceRef
from packages.narrative.story_ledger import memory_hash, text_hash
from services.worker.generation.causal_context import build_context, retrieve_context_evidence


def source(text, chapter, scene="s1", utterance="s1-u1"):
    ref = SourceRef(storyline_id="story", chapter_number=chapter, scene_id=scene,
        scene_revision=text_hash(f"scene {chapter}: {text}"), utterance_id=utterance,
        source_start=0, source_end=len(text), text_hash=text_hash(text))
    return {"ref": ref.model_dump(), "text": text, "speaker_id": "Hero"}


def memory():
    old = source("古い出来事の正確な原文。" * 500, 1)
    current = source("鍵を受け取り、今度は自分から旧図書室へ向かった。", 2)
    old_ref, current_ref = [old["ref"]], [current["ref"]]
    return ChapterMemory.model_validate({
        "storyline_id": "story", "chapter_number": 2, "summary": "鍵を受け取った。",
        "events": [
            {"id": "old-event", "description": "かつての事件。" * 100,
             "character_ids": ["Keeper"], "location_ids": ["hall"],
             "story_time": "初日", "evidence": old_ref},
            {"id": "new-event", "description": "鍵を受け取り旧図書室へ向かう決断をした。",
             "character_ids": ["Hero"], "location_ids": ["library"], "causes": ["old-event"],
             "story_time": "翌朝", "evidence": current_ref},
        ],
        "state": [{"scope": "character", "entity_id": "Hero", "key": "location",
                   "value": "旧図書室", "changed_chapter": 2, "evidence": current_ref}],
        "state_history": [{"id": "moved", "scope": "character", "entity_id": "Hero",
            "key": "location", "before": "廊下", "after": "旧図書室", "event_id": "new-event",
            "evidence": current_ref}],
        "knowledge": [{"id": "knows-key", "character_id": "Hero", "fact_id": "key-origin",
            "content": "鍵を受け取った。", "acquired_at": "翌朝", "event_ids": ["new-event"],
            "evidence": current_ref, "disclosed_chapter": 2}],
        "threads": [{"id": "old-promise", "question": "Keeperが古い手紙の送り主を調べる。",
            "character_ids": ["Keeper"], "event_ids": ["old-event"], "evidence": old_ref,
            "introduced_chapter": 1, "changed_chapter": 1}],
        "introductions": [{"id": "keeper-first", "kind": "character", "entity_ids": ["Keeper"],
            "description": "Keeperの初登場", "audience_character_ids": ["Keeper"], "evidence": old_ref}],
        "sources": [old, current],
        "author_facts": [{"id": "secret", "content": "送り主はKeeper。", "origin": "initial_canon",
            "authority_id": "approved", "authority_hash": "a" * 64, "established_at": "開始前",
            "known_by_character_ids": ["Keeper"], "disclosure_condition": "最後の章"}],
    })


def test_scene_keeps_direct_constraints_and_causal_index_without_all_ancestor_prose():
    original = memory()
    before = original.model_dump(mode="json")
    view = build_context(original, scope="scene", character_ids=["Hero"],
                         location_ids=["library"], event_ids=["new-event"])
    assert view["status"] == "ready"
    assert [event["id"] for event in view["events"]] == ["new-event"]
    assert view["events"][0]["causes"] == ["old-event"]
    assert {event["id"] for event in view["event_index"]} == {"old-event", "new-event"}
    assert [quote["text"] for quote in view["source_quotes"]] == [original.sources[1].text]
    assert view["state"][0]["value"] == "旧図書室"
    assert view["coverage"]["index_only_event_ids"] == ["old-event"]
    assert view["coverage"]["new_semantic_checks_performed"] is False
    assert view["author_secrets_not_shared_knowledge"] == []
    assert original.model_dump(mode="json") == before


def recurring_cast_memory(chapters):
    value = memory().model_dump()
    value.update(chapter_number=chapters, events=[], state=[], state_history=[],
                 knowledge=[], threads=[], introductions=[], sources=[])
    for chapter in range(1, chapters + 1):
        excerpt = source(f"第{chapter}章で二人が確かめた固有の事実。" * 400, chapter)
        refs = [excerpt["ref"]]
        event_id = f"event-{chapter}"
        value["events"].append({"id": event_id, "description": f"第{chapter}章の行動",
            "character_ids": ["Hero", "Keeper"], "location_ids": ["library"],
            "causes": [f"event-{chapter - 1}"] if chapter > 1 else [],
            "story_time": f"{chapter}日目", "evidence": refs})
        value["state"].append({"scope": "character", "entity_id": "Hero", "key": f"skill-{chapter}",
            "value": f"第{chapter}章で習得した技術", "changed_chapter": chapter, "evidence": refs})
        for character in ("Hero", "Keeper"):
            value["knowledge"].append({"id": f"knows-{character.lower()}-{chapter}", "character_id": character,
                "fact_id": f"fact-{chapter}", "content": f"第{chapter}章の事実を知っている。",
                "acquired_at": f"{chapter}日目", "event_ids": [event_id],
                "evidence": refs, "disclosed_chapter": chapter})
        value["threads"].append({"id": f"question-{chapter}", "question": f"第{chapter}章の謎",
            "character_ids": ["Hero", "Keeper"], "event_ids": [event_id], "evidence": refs,
            "introduced_chapter": chapter, "changed_chapter": chapter})
        value["introductions"].append({"id": f"introduction-{chapter}", "kind": "fact",
            "entity_ids": [f"fact-{chapter}"], "description": f"第{chapter}章の事実を既に紹介した。",
            "audience_character_ids": ["Hero", "Keeper"], "evidence": refs})
        value["sources"].append(excerpt)
    return ChapterMemory.model_validate(value)


@pytest.mark.parametrize("chapters", [3, 8])
def test_recurring_main_cast_keeps_adopted_values_without_replaying_all_old_prose(chapters):
    original = recurring_cast_memory(chapters)
    before = original.model_dump(mode="json")
    view = build_context(original, scope="scene", character_ids=["Hero", "Keeper"],
                         location_ids=["library"])
    assert view["status"] == "ready"
    assert [event["id"] for event in view["events"]] == [f"event-{chapters}"]
    assert [quote["text"] for quote in view["source_quotes"]] == [original.sources[-1].text]
    assert len(view["event_index"]) == chapters
    for key, records in (("state", original.state), ("knowledge", original.knowledge),
                         ("unresolved", original.threads), ("already_introduced", original.introductions)):
        # Only provenance representation changes. No adopted meaning is shortened.
        assert [{k: v for k, v in item.items() if k != "evidence"} for item in view[key]] == [
            record.model_dump(mode="json", exclude={"evidence"}) for record in records]
        assert all(item["evidence"] for item in view[key])
    old_alias = view["knowledge"][0]["evidence"]
    assert set(old_alias).isdisjoint(view["coverage"]["supplied_source_ids"])
    restored = retrieve_context_evidence(original, memory_hash=view["memory_hash"], source_ids=old_alias)
    assert restored["status"] == "ready"
    assert restored["source_quotes"][0]["text"] == original.sources[0].text
    event = retrieve_context_evidence(original, memory_hash=view["memory_hash"], record_ids=["event-1"])
    assert event["events"][0]["description"] == original.events[0].description
    assert event["source_quotes"] == restored["source_quotes"]
    assert original.model_dump(mode="json") == before


def test_explicit_inheritance_restores_old_full_event_and_unabridged_evidence():
    original = recurring_cast_memory(5)
    view = build_context(original, scope="scene", character_ids=["Hero", "Keeper"], event_ids=["event-2"])
    assert view["status"] == "ready"
    assert [event["id"] for event in view["events"]] == ["event-2", "event-5"]
    assert {quote["text"] for quote in view["source_quotes"]} == {
        original.sources[1].text, original.sources[-1].text}


def test_current_knowledge_keeps_other_characters_linked_event_in_lookup_index_only():
    value = memory().model_dump()
    value["events"][-1]["causes"] = []
    value["knowledge"][0].update(event_ids=["old-event"], evidence=[value["sources"][0]["ref"]],
                                 content="Keeperが以前調べた事件を知っている。")
    view = build_context(ChapterMemory.model_validate(value), scope="scene", character_ids=["Hero"])
    assert view["status"] == "ready"
    assert [event["id"] for event in view["events"]] == ["new-event"]
    assert view["coverage"]["index_only_event_ids"] == ["old-event"]
    assert [quote["chapter"] for quote in view["source_quotes"]] == [2]
    assert view["knowledge"][0]["event_ids"] == ["old-event"]


def test_latest_scene_citations_do_not_replay_older_scenes_or_older_chapters():
    value = memory().model_dump()
    latest = source("同じ章の次の場面では、本棚の裏に隠れていた手紙を取り出した。", 2,
                    scene="s2", utterance="s2-u1")
    value["sources"].append(latest)
    value["events"].append({"id": "latest-event", "description": "手紙を取り出した。",
        "character_ids": ["Hero"], "story_time": "翌朝の続き", "causes": ["new-event"],
        "evidence": [source["ref"] for source in value["sources"]]})
    original = ChapterMemory.model_validate(value)
    view = build_context(original, scope="scene", character_ids=["Hero"])
    assert view["status"] == "ready"
    assert [event["id"] for event in view["events"]] == ["latest-event"]
    assert len(view["events"][0]["evidence"]) == 3
    assert [quote["text"] for quote in view["source_quotes"]] == [latest["text"]]


def test_previous_scene_includes_adopted_evidence_not_cited_by_its_events():
    value = memory().model_dump()
    extra = source("鍵の送り主がKeeperだと気づいた。", 2, utterance="s1-u2")
    extra["ref"]["scene_revision"] = value["sources"][-1]["ref"]["scene_revision"]
    value["sources"].append(extra)
    value["knowledge"].append({"id": "knows-sender", "character_id": "Hero", "fact_id": "sender",
        "content": extra["text"], "acquired_at": "翌朝", "event_ids": ["new-event"],
        "evidence": [extra["ref"]], "disclosed_chapter": 2})
    original = ChapterMemory.model_validate(value)
    view = build_context(original, scope="scene", character_ids=["Hero"])
    assert view["status"] == "ready"
    assert {quote["text"] for quote in view["source_quotes"]} == {
        original.sources[1].text, extra["text"]}
    assert len(view["events"]) == 1


def test_deferred_missing_old_proof_remains_addressable_and_lookup_reports_the_failure():
    original = recurring_cast_memory(3)
    broken = original.model_copy(update={"sources": original.sources[1:]})
    view = build_context(broken, scope="scene", character_ids=["Hero", "Keeper"])
    assert view["status"] == "ready"
    assert view["knowledge"][0]["content"] == original.knowledge[0].content
    result = retrieve_context_evidence(broken, memory_hash=view["memory_hash"],
                                       source_ids=view["knowledge"][0]["evidence"])
    assert result["status"] == "insufficient_evidence"
    assert result["source_quotes"] == []
    required = build_context(broken, scope="scene", event_ids=["event-1"])
    assert required["status"] == "insufficient_evidence"


def test_ambiguous_preceding_scene_is_reported_instead_of_replaying_both_candidates():
    value = memory().model_dump()
    other = source("もう一つの場面の証拠。", 2, scene="other-scene")
    value["sources"].append(other)
    value["events"][-1]["evidence"].append(other["ref"])
    view = build_context(ChapterMemory.model_validate(value), scope="scene", character_ids=["Hero"])
    assert view["status"] == "insufficient_evidence"
    assert any("preceding adopted scene" in missing for missing in view["missing_information"])
    assert view["source_quotes"] == []


def test_index_excerpt_can_be_retrieved_as_complete_record_and_verbatim_long_source():
    original = memory()
    view = build_context(original, scope="scene", character_ids=["Hero"])
    old = next(item for item in view["event_index"] if item["id"] == "old-event")
    assert old["description_complete"] is False
    assert old["evidence_status"] == "index_only"
    restored = retrieve_context_evidence(original, memory_hash=view["memory_hash"],
                                          record_ids=[old["id"]])
    assert restored["status"] == "ready"
    assert restored["events"][0]["description"] == original.events[0].description
    assert restored["source_quotes"][0]["text"] == original.sources[0].text
    assert len(restored["source_quotes"][0]["text"]) > 5000
    by_alias = retrieve_context_evidence(original, memory_hash=view["memory_hash"],
                                         source_ids=old["evidence"])
    assert by_alias["source_quotes"] == restored["source_quotes"]


def test_plan_keeps_every_historical_event_index_and_unresolved_obligation():
    original = memory()
    view = build_context(original, scope="plan", character_ids=["Hero"], include_author_facts=True)
    assert [event["id"] for event in view["events"]] == ["new-event"]
    assert {event["id"] for event in view["event_index"]} == {"old-event", "new-event"}
    assert view["unresolved"][0]["id"] == "old-promise"
    assert view["already_introduced"][0]["id"] == "keeper-first"
    assert view["author_secrets_not_shared_knowledge"][0]["id"] == "secret"
    assert [quote["chapter"] for quote in view["source_quotes"]] == [2]


def test_extraction_uses_before_values_and_identity_index_without_old_raw_prose():
    view = build_context(memory(), scope="extraction", character_ids=["Hero"], location_ids=["library"])
    assert view["events"] == []
    assert view["source_quotes"] == []
    assert view["state"][0]["value"] == "旧図書室"
    assert view["knowledge"][0]["fact_id"] == "key-origin"
    assert {event["id"] for event in view["event_index"]} == {"old-event", "new-event"}
    assert view["thread_index"][0]["id"] == "old-promise"
    assert view["thread_index"][0]["evidence_status"] == "index_only"


def reviews():
    return [ReviewReport(scope="scene-continuity-s1", subject_hash="a" * 64, verdict="pass",
                checked_scene_ids=["s1"], checked_categories=["causality", "state"]),
            ReviewReport(scope="extraction-review-s1", subject_hash="b" * 64, verdict="pass",
                checked_scene_ids=["s1"], checked_categories=["support", "completeness"])]


def test_progress_records_current_chapter_evidence_and_prior_review_coverage():
    view = build_context(memory(), scope="progress", scene_reviews=reviews(), expected_scene_ids=["s1"])
    assert view["status"] == "ready"
    assert [event["id"] for event in view["events"]] == ["new-event"]
    assert view["state_deltas"][0]["event_id"] == "new-event"
    assert view["coverage"]["continuity_scene_ids"] == ["s1"]
    assert view["coverage"]["extraction_scene_ids"] == ["s1"]
    assert view["coverage"]["prior_reviews"][1]["subject_hash"] == "b" * 64
    assert view["coverage"]["new_semantic_checks_performed"] is False


@pytest.mark.parametrize("reports", [[], reviews()[:1], reviews()[1:]])
def test_progress_cannot_treat_partial_scene_reviews_as_full_coverage(reports):
    view = build_context(memory(), scope="progress", scene_reviews=reports, expected_scene_ids=["s1"])
    assert view["status"] == "insufficient_evidence"
    assert view["missing_information"]


def test_missing_required_source_is_explicit_and_never_replaced_by_event_description():
    original = memory()
    broken = original.model_copy(update={"sources": original.sources[:1]})
    view = build_context(broken, scope="scene", character_ids=["Hero"], event_ids=["new-event"])
    assert view["status"] == "insufficient_evidence"
    assert any("Unavailable source" in error for error in view["missing_information"])
    assert view["source_quotes"] == []


def test_corrupt_quotation_is_never_returned_as_evidence():
    original = memory()
    corrupt = original.sources[1].model_copy(update={"text": "捏造した原文"})
    broken = original.model_copy(update={"sources": [original.sources[0], corrupt]})
    view = build_context(broken, scope="scene", character_ids=["Hero"])
    assert view["status"] == "insufficient_evidence"
    assert "捏造した原文" not in str(view["source_quotes"])


def test_unknown_requested_events_are_rejected_before_generating_a_prompt():
    with pytest.raises(ValueError, match="Unknown required event IDs"):
        build_context(memory(), scope="scene", event_ids=["invented"])


def test_missing_causal_ancestor_is_reported_without_silently_dropping_its_link():
    original = memory()
    broken = original.model_copy(update={"events": original.events[1:]})
    view = build_context(broken, scope="scene", character_ids=["Hero"])
    assert view["status"] == "insufficient_evidence"
    assert "Unavailable causal event: old-event" in view["missing_information"]
    assert view["events"][0]["causes"] == ["old-event"]


def test_alias_lookup_rejects_a_changed_snapshot_and_reports_unknown_ids():
    original = memory()
    view = build_context(original, scope="scene", character_ids=["Hero"])
    changed = original.model_copy(update={"summary": "変わった履歴"})
    with pytest.raises(ValueError, match="memory_hash"):
        retrieve_context_evidence(changed, memory_hash=view["memory_hash"], source_ids=["q1"])
    result = retrieve_context_evidence(original, memory_hash=view["memory_hash"],
                                       source_ids=["q999"], record_ids=["missing"])
    assert result["status"] == "insufficient_evidence"
    assert len(result["missing_information"]) == 2


def test_span_only_and_same_utterance_in_other_chapters_have_unambiguous_aliases():
    original = memory()
    extra = source("章1の別の部分", 1, utterance=None)
    another = source("章1の更に別の部分", 1, utterance=None)
    expanded = original.model_copy(update={"sources": [*original.sources,
        original.sources[0].model_validate(extra), original.sources[0].model_validate(another)]})
    result = retrieve_context_evidence(expanded, memory_hash=memory_hash(expanded),
                                       source_ids=["q1", "q2", "q3", "q4"])
    assert result["status"] == "ready"
    assert len({item["id"] for item in result["source_quotes"]}) == 4
    assert {item["text"] for item in result["source_quotes"]} >= {extra["text"], another["text"]}
    first = build_context(expanded, scope="plan")
    reordered = expanded.model_copy(update={"sources": list(reversed(expanded.sources))})
    second = build_context(reordered, scope="plan")
    assert first["source_quotes"] == second["source_quotes"]


def test_retracted_knowledge_retains_its_status_in_current_view():
    original = memory()
    changed = copy.deepcopy(original.knowledge[0].model_dump())
    changed.update(id="correction", kind="retracted", supersedes_id="knows-key")
    newer = original.knowledge[0].model_validate(changed)
    updated = original.model_copy(update={"knowledge": [*original.knowledge, newer]})
    view = build_context(updated, scope="extraction", character_ids=["Hero"])
    assert len(view["knowledge"]) == 1
    assert view["knowledge"][0]["kind"] == "retracted"
    assert view["knowledge"][0]["supersedes_id"] == "knows-key"
