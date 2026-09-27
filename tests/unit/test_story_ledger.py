"""Regression checks for real-event memory, provenance, and chapter retrieval."""

import copy

import pytest
from pydantic import ValidationError

from packages.contracts.story_workflow import (
    AuthorFact,
    ChapterExtraction,
    ReviewReport,
    SourceRef,
    StateEntry,
    StoryBlueprint,
)
from packages.narrative.story_ledger import (
    adopt_author_facts,
    apply_chapter_memory,
    apply_scene_memory,
    content_hash,
    empty_memory,
    find_repeated_passages,
    memory_hash,
    select_context,
    source_catalog,
    source_excerpts,
    text_hash,
    validate_evidence,
)
from packages.narrative.validation import parse_scene_text


def scene(text="Hero: 明日の朝、旧図書室で必ず待つ。\nKeeper: 約束だ。", scene_id="s1"):
    return {"id": scene_id, "raw_text": text,
            "utterances": [u.model_dump() for u in parse_scene_text(
                text, scene_id, {"Hero", "Keeper"})]}


def extraction(scenes, chapter=1, event_id="c1_promise", **updates):
    refs = source_catalog(scenes, storyline_id="story", chapter_number=chapter)
    value = {"chapter_number": chapter, "summary": "翌朝の再会を約束した。",
             "events": [{"id": event_id, "description": "翌朝図書室で会う約束をした。",
                         "character_ids": ["Hero", "Keeper"], "location_ids": ["library"],
                         "story_time": "初日放課後", "evidence": [refs[0].model_dump()]}]}
    value.update(updates)
    return value


def apply(previous=None, value=None, scenes=None, chapter=1, **kwargs):
    scenes = scenes if scenes is not None else [scene()]
    previous = previous if previous is not None else empty_memory("story")
    value = value if value is not None else extraction(scenes, chapter)
    return apply_chapter_memory(previous, value, scenes=scenes, storyline_id="story",
                                chapter_number=chapter, **kwargs)


def delta(value, *, before=None, after="旧図書室", event_id="c1_promise", delta_id="c1_location"):
    return {"id": delta_id, "scope": "character", "entity_id": "Hero", "key": "location",
            "before": before, "after": after, "event_id": event_id,
            "evidence": value["events"][0]["evidence"]}


def test_source_reference_qualifies_ids_with_chapter_story_and_original_revision():
    scenes = [scene()]
    first = source_catalog(scenes, storyline_id="story", chapter_number=1)
    second = source_catalog(scenes, storyline_id="story", chapter_number=2)
    assert first[0].utterance_id == second[0].utterance_id
    assert first[0] != second[0]
    assert first[0].scene_revision == text_hash(scenes[0]["raw_text"])
    assert first[0].text_hash == text_hash(scenes[0]["utterances"][0]["display_text"])
    validated = validate_evidence(first, narratives=[
        {"storyline_id": "story", "chapter_number": 1, "scenes": scenes}])
    assert validated[0].text == "明日の朝、旧図書室で必ず待つ。"
    assert validated[0].speaker_id == "Hero"
    with pytest.raises(ValueError, match="available adopted source"):
        validate_evidence(second, sources=validated)


@pytest.mark.parametrize("damage,match", [
    ({"scene_revision": "a" * 64}, "stale scene"),
    ({"text_hash": "a" * 64}, "text hash"),
    ({"utterance_id": "s1_missing"}, "unknown or ambiguous utterance"),
    ({"source_end": 10000}, "outside the scene"),
])
def test_evidence_rejects_fabricated_or_stale_sources(damage, match):
    scenes = [scene()]
    ref = source_catalog(scenes, storyline_id="story", chapter_number=1)[0]
    damaged = SourceRef.model_validate({**ref.model_dump(), **damage})
    with pytest.raises(ValueError, match=match):
        validate_evidence([damaged], narratives=[
            {"storyline_id": "story", "chapter_number": 1, "scenes": scenes}])


def test_span_only_evidence_resolves_original_and_does_not_fabricate_a_speaker():
    scenes = [scene()]
    ref = source_catalog(scenes, storyline_id="story", chapter_number=1)[0]
    text = scenes[0]["raw_text"][ref.source_start:ref.source_start + 4]
    span = SourceRef.model_validate({**ref.model_dump(), "utterance_id": None,
        "source_end": ref.source_start + 4, "text_hash": text_hash(text)})
    excerpts = validate_evidence([span], narratives=[
        {"storyline_id": "story", "chapter_number": 1, "scenes": scenes}])
    assert excerpts[0].text == text and excerpts[0].speaker_id is None


def test_original_mapping_must_be_exact_and_unique():
    scenes = [scene()]
    scenes[0]["utterances"][0]["display_text"] = "原文にない約束"
    with pytest.raises(ValueError, match="original source span"):
        source_catalog(scenes, storyline_id="story", chapter_number=1)
    scenes = [scene(), scene()]
    with pytest.raises(ValueError, match="Duplicate scene"):
        source_catalog(scenes, storyline_id="story", chapter_number=1)


def test_delta_only_changes_named_state_and_preserves_initial_character_growth():
    original = empty_memory("story", state=[
        StateEntry(scope="character", entity_id="Hero", key="attitude", value="臆病"),
        StateEntry(scope="character", entity_id="Hero", key="possession", value="青いノート")])
    before = original.model_dump()
    scenes = [scene("Hero: 怖くても今度こそ私が先に行く。")]
    value = extraction(scenes)
    value["state_deltas"] = [{**delta(value, before="臆病", after="恐れながらも進む"),
                              "key": "attitude"}]
    memory = apply(original, value, scenes)
    assert {s.key: s.value for s in memory.state} == {
        "attitude": "恐れながらも進む", "possession": "青いノート"}
    assert original.model_dump() == before
    assert memory.previous_memory_hash == memory_hash(original)
    assert memory.state_history[0].before == "臆病"
    assert len(memory.sources) == 1
    assert memory_hash(memory) == memory_hash(apply(original, value, scenes))


def test_delta_rejects_reset_conflict_and_explicit_immutable_change():
    initial = empty_memory("story", state=[
        StateEntry(scope="character", entity_id="Hero", key="location", value="廊下")])
    value = extraction([scene()])
    value["state_deltas"] = [delta(value)]
    with pytest.raises(ValueError, match="before value"):
        apply(initial, value)
    value["state_deltas"][0]["before"] = "廊下"
    with pytest.raises(ValueError, match="immutable"):
        apply(initial, value, immutable_keys=[("character", "Hero", "location")])
    value["state_deltas"].append({**value["state_deltas"][0], "id": "other_delta"})
    with pytest.raises(ValueError, match="concurrent state updates"):
        apply(initial, value)


@pytest.mark.parametrize("assertion", ["reported", "believed"])
def test_report_or_belief_cannot_apply_its_claimed_physical_effect(assertion):
    initial = empty_memory("story", state=[StateEntry(scope="character", entity_id="Hero",
        key="physical_condition", value="腕を負傷している")])
    before = initial.model_dump()
    scenes = [scene("Keeper: 傷はもう治ったらしい。")]
    value = extraction(scenes)
    value["events"][0].update(assertion=assertion, description="主人公の傷は治ったらしい。")
    value["state_deltas"] = [{**delta(value, before="腕を負傷している", after="治癒済み"),
                              "key": "physical_condition"}]
    with pytest.raises(ValueError, match="current state change requires an observed event"):
        apply(initial, value, scenes)
    assert initial.model_dump() == before


@pytest.mark.parametrize("text,key,observed_change", [
    ("Hero: 傷は治ったと聞いた。", "last_statement", "傷が治ったという伝聞を話した"),
    ("Hero: 明日の朝に待つ。\nKeeper: 約束だ。", "agreement", "翌朝の再会に合意した"),
    ("NARRATOR: 主人公は不安を覚えた。", "emotion", "不安を感じている"),
])
def test_observed_speech_agreement_and_emotion_can_update_their_own_state(text, key, observed_change):
    scenes = [scene(text)]
    value = extraction(scenes)
    value["events"][0].update(assertion="observed", description=observed_change)
    value["state_deltas"] = [{**delta(value, after=observed_change), "key": key}]
    result = apply(value=value, scenes=scenes)
    assert [(state.key, state.value) for state in result.state] == [(key, observed_change)]
    assert all(state.key != "physical_condition" for state in result.state)


@pytest.mark.parametrize("assertion", ["reported", "believed"])
def test_reported_or_believed_events_can_be_saved_without_applying_physical_effects(assertion):
    initial = empty_memory("story", state=[StateEntry(scope="character", entity_id="Hero",
        key="physical_condition", value="腕を負傷している")])
    scenes = [scene("Keeper: 傷はもう治ったらしい。")]
    value = extraction(scenes)
    value["events"][0].update(assertion=assertion, description="主人公の傷は治ったらしい。")
    result = apply(initial, value, scenes)
    assert result.events[0].assertion == assertion
    assert result.state == initial.state and result.state_history == []
    assert result.sources[0].text == "傷はもう治ったらしい。"


def test_scene_candidates_accumulate_but_keep_one_exact_predecessor_hash():
    initial = empty_memory("story")
    first = apply(initial)
    scenes = [scene("Hero: 約束を果たす準備を始めよう。", "s2")]
    value = extraction(scenes, event_id="c1_prepare")
    value["events"][0]["causes"] = ["c1_promise"]
    next_scene = apply_scene_memory(first, value, scenes=scenes, storyline_id="story", chapter_number=1)
    assert next_scene.previous_memory_hash == memory_hash(initial)
    assert next_scene.revision == 2
    assert [e.id for e in next_scene.events] == ["c1_promise", "c1_prepare"]
    with pytest.raises(ValueError, match="preceding chapter"):
        apply(first, value, scenes)
    with pytest.raises(ValueError, match="already belongs"):
        apply_scene_memory(first, extraction([scene()]), scenes=[scene()], storyline_id="story",
                           chapter_number=1)


def test_append_cannot_replace_a_previously_adopted_scene_revision():
    previous = apply()
    scenes = [scene("Hero: 同じ場面の書き直し。")]
    with pytest.raises(ValueError, match="scene revision cannot be replaced"):
        apply_scene_memory(previous, extraction(scenes, event_id="c1_new"), scenes=scenes,
                           storyline_id="story", chapter_number=1)


@pytest.mark.parametrize("causes", [["future_event"], ["c1_promise"]])
def test_event_causality_rejects_unknown_or_self_cycle(causes):
    value = extraction([scene()])
    value["events"][0]["causes"] = causes
    with pytest.raises(ValueError, match="unknown ID or a causal cycle"):
        apply(value=value)


def test_only_current_source_evidence_can_add_a_new_observation():
    previous = apply()
    scenes = [scene("Hero: 今日は何も起こらなかった。")]
    value = extraction(scenes, chapter=2, event_id="c2_fake")
    value["events"][0]["evidence"] = [previous.events[0].evidence[0].model_dump()]
    with pytest.raises(ValueError, match="supplied scene text"):
        apply(previous, value, scenes, chapter=2)


@pytest.mark.parametrize("assertion", ["observed", "reported", "believed"])
def test_flashback_injury_updates_history_without_reinjuring_present_character(assertion):
    initial = empty_memory("story", state=[
        StateEntry(scope="character", entity_id="Hero", key="physical_condition", value="治癒済み")])
    scenes = [scene("Hero: あの日、私は腕を怪我した。")]
    value = extraction(scenes)
    value["events"][0].update(presentation="flashback", story_time="十年前", assertion=assertion)
    value["state_deltas"] = [{**delta(value, before="無傷", after="腕の怪我"),
        "key": "physical_condition", "time_scope": "historical", "effective_time": "十年前"}]
    memory = apply(initial, value, scenes)
    assert memory.state[0].value == "治癒済み"
    assert memory.state_history[0].after == "腕の怪我"
    value["state_deltas"][0].update(time_scope="current", before="治癒済み")
    with pytest.raises(ValueError, match="flashback event"):
        apply(initial, value, scenes)
    assert memory.knowledge == []


def test_knowledge_correction_requires_explicit_supersession_and_keeps_both_versions():
    first_scene = [scene("Hero: 犯人は門番だと思う。")]
    first = extraction(first_scene)
    first["knowledge_updates"] = [{"id": "c1_belief", "character_id": "Hero",
        "fact_id": "culprit", "content": "犯人は門番", "kind": "belief", "acquired_at": "初日",
        "event_ids": ["c1_promise"], "evidence": first["events"][0]["evidence"]}]
    previous = apply(value=first, scenes=first_scene)
    second_scene = [scene("Hero: 門番にはアリバイがあった。私の推測は間違いだった。")]
    second = extraction(second_scene, chapter=2, event_id="c2_disprove")
    second["knowledge_updates"] = [{"id": "c2_belief_retracted", "character_id": "Hero",
        "fact_id": "culprit", "content": "門番が犯人だという推測は誤り", "kind": "retracted",
        "acquired_at": "二日目", "event_ids": ["c2_disprove"], "supersedes_id": "c1_belief",
        "evidence": second["events"][0]["evidence"]}]
    memory = apply(previous, second, second_scene, chapter=2)
    assert len(memory.knowledge) == 2
    assert select_context(memory, character_ids=["Hero"]).knowledge[0].kind == "retracted"
    del second["knowledge_updates"][0]["supersedes_id"]
    with pytest.raises(ValueError, match="explicitly supersede"):
        apply(previous, second, second_scene, chapter=2)


def test_open_threads_only_resolve_after_evidenced_explicit_transition():
    value = extraction([scene()])
    value["thread_updates"] = [{"id": "meet_tomorrow", "question": "約束は果たされるか",
        "character_ids": ["Keeper"], "event_ids": ["c1_promise"],
        "evidence": value["events"][0]["evidence"]}]
    previous = apply(value=value)
    assert select_context(previous, character_ids=["SomeoneElse"]).threads[0].id == "meet_tomorrow"
    scenes = [scene("Hero: 約束通り、図書室まで来たよ。")]
    next_value = extraction(scenes, chapter=2, event_id="c2_meeting")
    update = {**value["thread_updates"][0], "status": "resolved", "expected_status": "open",
              "event_ids": ["c2_meeting"], "evidence": next_value["events"][0]["evidence"]}
    next_value["thread_updates"] = [update]
    memory = apply(previous, next_value, scenes, chapter=2)
    assert memory.threads[0].status == "resolved"
    assert len(memory.thread_history) == 2
    assert select_context(memory).threads == []
    value["thread_updates"][0]["status"] = "resolved"
    with pytest.raises(ValueError, match="new unresolved thread"):
        apply(value=value)


def test_retrieval_follows_causality_and_introductions_across_all_chapters():
    first = extraction([scene()])
    first["events"][0]["character_ids"] = ["Keeper"]
    first["events"][0]["location_ids"] = ["classroom"]
    first["introductions"] = [{"id": "library_rumor", "kind": "fact", "entity_ids": ["library"],
        "description": "図書室の噂を初めて聞いた。", "audience_character_ids": ["Hero"],
        "evidence": first["events"][0]["evidence"]}]
    previous = apply(value=first)
    scenes = [scene("Hero: あの噂を確かめるため、図書室へ行く。")]
    second = extraction(scenes, chapter=2, event_id="c2_investigate")
    second["events"][0]["causes"] = ["c1_promise"]
    second["events"][0]["character_ids"] = ["Hero"]
    memory = apply(previous, second, scenes, chapter=2)
    packet = select_context(memory, character_ids=["Hero"], location_ids=["library"])
    assert packet.status == "ready"
    assert [event.id for event in packet.events] == ["c1_promise", "c2_investigate"]
    assert packet.introductions[0].id == "library_rumor"
    assert len(packet.sources) == 2
    limited = select_context(memory, character_ids=["Hero"], max_events=1)
    assert limited.status == "insufficient_evidence"
    assert len(limited.events) == 2  # no silent loss of a required causal source
    missing = select_context(memory, event_ids=["unknown_fact"])
    assert missing.status == "insufficient_evidence" and "unknown_fact" in missing.missing_information[0]


def test_author_truth_is_separate_from_observed_events_and_hidden_by_default():
    secret = AuthorFact(id="secret_identity", content="門番は失踪した父親である。",
        origin="initial_canon", authority_id="approved-setting", authority_hash="a" * 64,
        established_at="本編開始前", known_by_character_ids=["Keeper"],
        disclosure_condition="最終章で本人が明かす")
    initial = empty_memory("story", author_facts=[secret])
    assert initial.events == [] and initial.knowledge == [] and initial.sources == []
    assert select_context(initial, character_ids=["Hero"]).author_facts == []
    assert select_context(initial, include_author_facts=True).author_facts == [secret]
    previous = apply(initial)
    author_event = AuthorFact(id="secret_departure", content="門番は約束を守るため出発した。",
        origin="author_event", authority_id="author-event-1", authority_hash="b" * 64,
        established_at="初日深夜", known_by_character_ids=["Keeper"],
        disclosure_condition="翌朝、本人が説明する", cause_event_ids=["c1_promise"])
    report = ReviewReport(scope="author_event", subject_hash=content_hash([author_event.model_dump()]),
                          verdict="pass")
    memory = adopt_author_facts(previous, [author_event], review=report,
                                authority_hashes={"author-event-1": "b" * 64})
    assert len(memory.author_facts) == 2 and len(memory.events) == 1
    with pytest.raises(ValueError, match="authority is absent"):
        adopt_author_facts(previous, [author_event], review=report, authority_hashes={})


def test_viewpoint_knowledge_does_not_include_other_characters_private_beliefs():
    value = extraction([scene()])
    value["knowledge_updates"] = [{"id": "keeper_private", "character_id": "Keeper",
        "fact_id": "father", "content": "主人公の父親の行方", "acquired_at": "今",
        "evidence": value["events"][0]["evidence"]}]
    memory = apply(value=value)
    packet = select_context(memory, character_ids=["Hero", "Keeper"], viewpoint_character_id="Hero")
    assert packet.knowledge == []


def test_review_cannot_pass_when_evidence_is_missing_or_errors_remain():
    report = {"scope": "chapter boundary", "subject_hash": "a" * 64,
              "verdict": "pass", "missing_information": ["前章での約束の原文"]}
    with pytest.raises(ValidationError, match="cannot pass"):
        ReviewReport.model_validate(report)
    report["verdict"] = "insufficient_evidence"
    assert ReviewReport.model_validate(report).verdict == "insufficient_evidence"
    report["missing_information"] = []
    with pytest.raises(ValidationError, match="identify missing evidence"):
        ReviewReport.model_validate(report)


def test_planned_ending_is_not_an_automatic_state_or_thread_update():
    blueprint = StoryBlueprint(central_question="約束を守れるか", ending_conditions=["再会する"],
        chapters=[{"number": 1, "question": "再会できるか", "unique_progress": ["再会を試みる"],
                   "ending_conditions": ["約束が解決する"]}])
    memory = apply()
    assert blueprint.chapters[0].ending_conditions == ["約束が解決する"]
    assert memory.state == [] and memory.threads == []
    with pytest.raises(ValidationError):
        ChapterExtraction.model_validate({**extraction([scene()]), "blueprint": blueprint.model_dump()})


def test_long_exact_repetition_is_a_candidate_and_short_refrains_are_not():
    repeated = "\n".join(f"Hero: {number}番目の、忘れられない夜の約束について話し合う。" for number in range(7))
    previous = [scene("Keeper: 序章。\n" + repeated)]
    current = [scene("Keeper: 二章。\n" + repeated + "\nHero: 今回は答えが違う。", "s3")]
    before = copy.deepcopy((previous, current))
    matches = find_repeated_passages(previous, current)
    assert len(matches) == 1 and len(matches[0].current_utterance_ids) == 7
    assert matches[0].previous_scene_id == "s1" and matches[0].current_scene_id == "s3"
    assert (previous, current) == before
    short = [scene("Hero: また明日。\nKeeper: 約束だ。")]
    assert find_repeated_passages(short, short) == []
    other_speaker = [scene(repeated.replace("Hero:", "Keeper:"))]
    assert find_repeated_passages(previous, other_speaker) == []


def test_corrupt_saved_excerpt_is_not_valid_context():
    memory = apply()
    damaged = memory.model_dump()
    damaged["sources"][0]["text"] = "実績を書き換えた"
    with pytest.raises(ValueError, match="hash is corrupt"):
        select_context(damaged)


def test_missing_source_requests_more_evidence_instead_of_passing_a_summary():
    memory = apply().model_dump()
    memory["sources"] = []
    packet = select_context(memory, character_ids=["Hero"])
    assert packet.status == "insufficient_evidence"
    assert packet.events and packet.sources == []
    assert "Missing source" in packet.missing_information[0]


def test_initial_memory_has_no_fake_future_prose_evidence():
    ref = source_catalog([scene()], storyline_id="story", chapter_number=1)[0]
    with pytest.raises(ValueError, match="has not happened"):
        empty_memory("story", state=[StateEntry(scope="character", entity_id="Hero",
            key="location", value="図書室", evidence=[ref])])


def test_stored_source_excerpts_reject_forged_text():
    sources = source_excerpts([scene()], storyline_id="story", chapter_number=1)
    corrupted = sources[0].model_dump()
    corrupted["text"] = "偽の原文"
    with pytest.raises(ValueError, match="hash is corrupt"):
        validate_evidence([sources[0].ref], sources=[corrupted])
