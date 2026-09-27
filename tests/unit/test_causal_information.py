"""Information observations become evidence-backed records without model-generated IDs."""

import copy
import json
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from packages.contracts.story_workflow import AuthorFact, ChapterExtraction
from packages.narrative.story_ledger import apply_scene_memory, empty_memory, source_catalog
from packages.narrative.validation import parse_scene_text
from services.worker.generation.causal_information import (
    InformationAnnotation,
    _validate,
    annotate_information,
    information_schema,
)
from tests.unit.test_m3_narrative import FakeLLM


def scene(text=None, scene_id="s1", cast=None):
    raw = text or (
        "Hero: この写真、提供者は三十年代と言っていたけど、裏付ける記録がない。\n"
        "keeper: たぶん、でいいんじゃないか。\n"
        "Hero: 断定して書くには足りない。\n"
        "NARRATOR: 二人は机の写真を見つめた。")
    cast = cast or ["Hero", "keeper"]
    return SimpleNamespace(id=scene_id, raw_text=raw,
        plan=SimpleNamespace(character_ids=cast, location_id="room"),
        utterances=parse_scene_text(raw, scene_id, set(cast)))


def primary(current):
    return {"summary": "写真の説明について話した。",
        "events": [{"id": f"c1-{current.id}-e1", "description": "写真の根拠について話した。",
            "character_ids": current.plan.character_ids, "location_ids": ["room"], "causes": [],
            "story_time": "この場面", "presentation": "current", "assertion": "observed",
            "evidence": [u.id for u in current.utterances[:2]]}],
        "state_deltas": [], "thread_updates": []}


def observation(current, **updates):
    value = {"fact_id": "photo_uncertainty",
        "content": "写真の年代を裏付ける記録がないと説明された。",
        "evidence_ids": [u.id for u in current.utterances[:2]],
        "event_ids": [f"c1-{current.id}-e1"],
        "knowledge_changes": [{"character_id": "keeper", "kind": "knowledge",
            "change": "learned", "acquired_at": "この場面の説明時"}],
        "reader_introduction": {"kind": "fact", "entity_ids": ["photo_uncertainty"],
            "description": "写真の年代に裏付けがないと説明された。", "audience_character_ids": ["keeper"]},
        "existing_record_ids": [],
        "reason": "相手が説明に応答したので、不足について説明を受けたことは原文で確認できる。"}
    value.update(updates)
    return value


def annotation(current, observations=None):
    return {"inspected_utterance_ids": [u.id for u in current.utterances],
            "observations": [observation(current)] if observations is None else observations}


class Run:
    def __init__(self, *responses, chapter=1):
        self.payload = {"chapter_number": chapter}
        self.llm = FakeLLM([{"content": json.dumps(value, ensure_ascii=False), "_finish_reason": "stop"}
                            for value in responses])
        self.nodes = []
        self.saved = []

    def node(self, stage, inputs, model, function):
        self.nodes.append((stage, inputs))
        value = function()
        self.saved.append(value)
        return value


def expand(raw, current, chapter=1):
    refs = {r.utterance_id: r.model_dump() for r in source_catalog(
        [current], storyline_id="story", chapter_number=chapter)}
    result = copy.deepcopy(raw)
    result["chapter_number"] = chapter
    for group in ("events", "state_deltas", "knowledge_updates", "thread_updates", "introductions"):
        for item in result[group]:
            item["evidence"] = [refs[uid] for uid in item["evidence"]]
    return ChapterExtraction.model_validate(result)


def apply(raw, current, memory=None):
    memory = memory or empty_memory("story")
    return apply_scene_memory(memory, expand(raw, current), scenes=[current],
                              storyline_id="story", chapter_number=1)


def first_memory():
    current = scene()
    result = annotate_information(Run(annotation(current)), empty_memory("story"), current, primary(current))
    return apply(result, current)


def validate(value, current, memory=None):
    _validate(InformationAnnotation.model_validate(value), memory=memory or empty_memory("story"),
              scene=current, primary=primary(current))


def test_independent_disclosure_is_added_without_modifying_primary_facts():
    current = scene()
    run = Run(annotation(current))
    original = primary(current)
    before = copy.deepcopy(original)
    result = annotate_information(run, empty_memory("story"), current, original)
    assert original == before
    for key in ("events", "state_deltas", "thread_updates", "summary"):
        assert result[key] == original[key]
    assert result["knowledge_updates"][0]["character_id"] == "keeper"
    assert result["knowledge_updates"][0]["supersedes_id"] is None
    assert result["introductions"][0]["reader_visible"]
    memory = apply(result, current)
    assert memory.knowledge[0].fact_id == "photo_uncertainty"
    assert memory.knowledge[0].evidence[0].utterance_id == "s1-u1"
    assert run.nodes[0][0] == "information-s1"
    assert run.llm.trace[-1]["observations"][0]["reason"]


def test_recall_and_new_recipient_are_independent_from_reader_introduction():
    memory = first_memory()
    current = scene("Hero: この写真には年代の裏付けがない。\nkeeper: さっき聞いたね。\n"
                    "guest: そうだったのですか。", "s2", ["Hero", "keeper", "guest"])
    recalled = observation(current, knowledge_changes=[], reader_introduction=None,
        existing_record_ids=[memory.knowledge[0].id, memory.introductions[0].id],
        reason="keeperは前の説明を再確認している。読者も既に説明を読んでいる。")
    learned = observation(current, knowledge_changes=[{"character_id": "guest", "kind": "knowledge",
        "change": "learned", "acquired_at": "今回の説明時"}], reader_introduction=None,
        evidence_ids=["s2-u1", "s2-u3"], reason="guestだけは今回初めて説明を受けて応答した。")
    run = Run(annotation(current, [recalled, learned]))
    result = annotate_information(run, memory, current, primary(current))
    assert [item["character_id"] for item in result["knowledge_updates"]] == ["guest"]
    assert result["introductions"] == []
    next_memory = apply(result, current, memory)
    assert len(next_memory.knowledge) == 2 and len(next_memory.introductions) == 1
    past = run.nodes[0][1]["inputs"]["previous_observed_information"]
    assert past["event_index"][0]["evidence_status"] == "index_only"
    assert "past_source_evidence" not in past and "source_quotes" not in past


def test_reader_only_flashback_does_not_disclose_private_history_to_characters():
    current = scene("NARRATOR: 彼女は昨夜、相手に黙って箱を動かしたことを思い出した。\nHero: 少し待って。")
    secret = AuthorFact(id="moved_box", content="Heroは昨夜箱を別の棚へ動かした。",
        origin="initial_canon", authority_id="approved", authority_hash="a" * 64,
        established_at="昨夜", known_by_character_ids=["Hero"], disclosure_condition="本人が話す時")
    memory = empty_memory("story", author_facts=[secret])
    row = observation(current, fact_id="moved_box", content="昨夜彼女が箱を移した。",
        evidence_ids=["s1-u1"], knowledge_changes=[], existing_record_ids=["moved_box"],
        reader_introduction={"kind": "fact", "entity_ids": ["moved_box"],
            "description": "昨夜の箱の移動が読者に示された。", "audience_character_ids": []},
        reason="本人には既知の回想で、相手へ発声せず読者だけが初めて知る。")
    result = annotate_information(Run(annotation(current, [row])), memory, current, primary(current))
    assert result["knowledge_updates"] == []
    assert result["introductions"][0]["audience_character_ids"] == []
    assert apply(result, current, memory).author_facts == memory.author_facts


def test_author_details_and_wrong_primary_candidates_are_not_copied_into_information_input():
    current = scene("Hero: 写真の年代を裏付ける資料はなかった。\nkeeper: 資料がないのか。")
    secret = AuthorFact(id="photo_uncertainty", content="写真の年代も人物名も不明。写る人物は梢かもしれない。",
        origin="initial_canon", authority_id="approved", authority_hash="a" * 64,
        established_at="昨夜", known_by_character_ids=["Hero"], disclosure_condition="本人が話す時")
    memory = empty_memory("story", author_facts=[secret])
    raw = primary(current)
    raw["knowledge_updates"] = [{"content": "候補だけにある架空の氏名"}]
    raw["introductions"] = [{"description": "候補だけにある架空の紹介"}]
    run = Run(annotation(current))
    result = annotate_information(run, memory, current, raw)
    inputs = run.nodes[0][1]["inputs"]
    assert "primary_information_candidates" not in inputs
    assert "候補だけ" not in json.dumps(inputs, ensure_ascii=False)
    assert set(inputs) == {"chapter_number", "scene_id", "scene_character_ids", "scene_location_id",
                          "previous_observed_information", "current_candidate_source", "current_event_index"}
    assert inputs["previous_observed_information"]["author_secrets_not_shared_knowledge"][0][
        "known_by_character_ids"] == ["Hero"]
    assert "作者事実の全文を、他人物が今回獲得したcontentへコピーしません" in run.nodes[0][1]["instruction"]
    assert "人物名" not in result["knowledge_updates"][0]["content"]
    assert result["knowledge_updates"][0]["character_id"] == "keeper"


@pytest.mark.parametrize("change", ["learned", "revised"])
def test_initially_known_fact_cannot_be_recorded_again_as_an_unchanged_acquisition(change):
    current = scene("Hero: 明日は九時開場だ。\nkeeper: 分かっている。")
    secret = AuthorFact(id="opening_time", content="明日は九時開場だ。", origin="initial_canon",
        authority_id="approved", authority_hash="a" * 64, established_at="前日",
        known_by_character_ids=["Hero", "keeper"], disclosure_condition="already_known")
    memory = empty_memory("story", author_facts=[secret])
    row = observation(current, fact_id="opening_time", content=secret.content,
        knowledge_changes=[{"character_id": "keeper", "kind": "knowledge", "change": change,
                            "acquired_at": "今回の確認時"}], reader_introduction=None)
    with pytest.raises(ValueError, match="known information|initial knowledge"):
        validate(annotation(current, [row]), current, memory)


def test_knowledge_correction_derives_supersedes_from_exact_recipient_fact_pair():
    memory = first_memory()
    current = scene("Hero: 日付入りの原板が見つかった。年代は三十二年と確認できた。\n"
                    "keeper: なら、記録なしという説明を直せるね。", "s2")
    row = observation(current, content="日付入り原板により写真の年代が三十二年と確認できた。",
        knowledge_changes=[{"character_id": "keeper", "kind": "knowledge", "change": "revised",
                            "acquired_at": "原板の説明時"}], reader_introduction=None,
        reason="裏付けなしという以前の知識を新しく発見された原板で訂正する。")
    result = annotate_information(Run(annotation(current, [row])), memory, current, primary(current))
    update = result["knowledge_updates"][0]
    assert update["supersedes_id"] == memory.knowledge[0].id
    assert update["id"] != update["supersedes_id"]
    assert update["evidence"] == ["s2-u1", "s2-u2"]
    assert len(apply(result, current, memory).knowledge) == 2


def test_multiple_recipients_and_introductions_have_unique_stable_ids_and_distinct_kinds():
    current = scene()
    row = observation(current, knowledge_changes=[
        {"character_id": "keeper", "kind": "uncertain", "change": "learned", "acquired_at": "説明時"},
        {"character_id": "Hero", "kind": "belief", "change": "learned", "acquired_at": "応答時"}])
    place = observation(current, fact_id=None, content="部屋が初めて描かれた。", knowledge_changes=[],
        reader_introduction={"kind": "location", "entity_ids": ["room"],
            "description": "写真を見るための机がある部屋。", "audience_character_ids": []})
    first = annotate_information(Run(annotation(current, [row, place])), empty_memory("story"), current, primary(current))
    second = annotate_information(Run(annotation(current, [place, row])), empty_memory("story"), current, primary(current))
    all_ids = [item["id"] for group in (first["knowledge_updates"], first["introductions"]) for item in group]
    assert len(all_ids) == len(set(all_ids)) == 4
    assert all(len(value) <= 64 for value in all_ids)
    assert {item["id"] for item in second["introductions"]} == {item["id"] for item in first["introductions"]}
    assert first["knowledge_updates"] == second["knowledge_updates"]
    assert [item["kind"] for item in first["knowledge_updates"]] == ["uncertain", "belief"]
    assert all(item["evidence"] == row["evidence_ids"] for item in first["knowledge_updates"])
    apply(first, current)


def test_knowledge_can_use_source_evidence_without_inventing_an_event():
    current = scene()
    row = observation(current, event_ids=[])
    result = annotate_information(Run(annotation(current, [row])), empty_memory("story"), current, primary(current))
    assert result["knowledge_updates"][0]["event_ids"] == []
    assert result["events"] == primary(current)["events"]
    apply(result, current)


@pytest.mark.parametrize("damage,match", [
    ("relearn", "known information"), ("unchanged", "Unchanged knowledge"),
    ("duplicate", "updated twice"), ("unknown_recipient", "current participant"),
    ("past_source", "current source"), ("invented_event", "existing current events"),
    ("missing_fact", "fact_id"), ("reintroduced", "another first introduction"),
])
def test_invalid_updates_are_rejected_before_merge(damage, match):
    memory = first_memory()
    current = scene("Hero: 年代の記録はまだない。\nkeeper: さっき聞いた。", "s2")
    row = observation(current, reader_introduction=None)
    if damage == "unchanged":
        row["knowledge_changes"][0]["change"] = "revised"
    elif damage == "duplicate":
        row["knowledge_changes"][0]["character_id"] = "Hero"
        row["knowledge_changes"].append(copy.deepcopy(row["knowledge_changes"][0]))
    elif damage == "unknown_recipient":
        row["knowledge_changes"][0]["character_id"] = "absent"
    elif damage == "past_source":
        row["evidence_ids"] = ["s1-u1"]
    elif damage == "invented_event":
        row["event_ids"] = ["new-event"]
    elif damage == "missing_fact":
        row["fact_id"] = None
    elif damage == "reintroduced":
        row["knowledge_changes"] = []
        row["reader_introduction"] = observation(current)["reader_introduction"]
    with pytest.raises(ValueError, match=match):
        validate(annotation(current, [row]), current, memory)


def test_no_change_requires_a_nonempty_audit_row_with_reason_and_complete_source_coverage():
    current = scene("Hero: 少し休もう。\nkeeper: そうしよう。")
    row = observation(current, fact_id=None, content="休憩の提案に同意した。", knowledge_changes=[],
        reader_introduction=None, reason="行動への同意で、新しい情報の説明や読者への初紹介はない。")
    value = annotation(current, [row])
    result = annotate_information(Run(value), empty_memory("story"), current, primary(current))
    assert result["knowledge_updates"] == [] and result["introductions"] == []
    with pytest.raises(ValidationError):
        InformationAnnotation.model_validate(annotation(current, []))
    row["reason"] = ""
    with pytest.raises(ValidationError):
        InformationAnnotation.model_validate(annotation(current, [row]))
    value["inspected_utterance_ids"].pop()
    row["reason"] = "確認済み。"
    with pytest.raises(ValueError, match="every current utterance"):
        validate(value, current)


def test_combined_validation_repairs_before_node_is_saved_and_rechecks_cache_hits():
    current = scene()
    bad = annotation(current)
    bad["observations"][0]["knowledge_changes"].append({"character_id": "Hero", "kind": "belief",
                                                      "change": "learned", "acquired_at": "今"})
    good = annotation(current)
    run = Run(bad, good)
    checked = []

    def validate_combined(value):
        checked.append(len(value["knowledge_updates"]))
        if checked[-1] > 1:
            raise ValueError("Combined knowledge limit exceeded.")
        apply(value, current)

    annotate_information(run, empty_memory("story"), current, primary(current), validate_combined=validate_combined)
    assert checked == [2, 1, 1]
    assert len(run.llm.calls) == 2 and len(run.saved) == 1
    cached = Run()
    cached.node = lambda *args: InformationAnnotation.model_validate(bad)
    with pytest.raises(ValueError, match="Combined knowledge limit"):
        annotate_information(cached, empty_memory("story"), current, primary(current),
                             validate_combined=validate_combined)
    assert not cached.llm.calls


def test_runtime_schema_has_no_new_record_ids_and_constrains_current_aliases():
    current = scene()
    schema = information_schema(empty_memory("story"), current, primary(current))
    definitions = schema["$defs"]
    for item in [schema, *definitions.values()]:
        assert set(item["required"]) == set(item["properties"])
        assert not {"id", "supersedes_id", "additional_events", "decisions"}.intersection(item["properties"])
    for name in ("KnowledgeObservation", "NonKnowledgeObservation"):
        properties = definitions[name]["properties"]
        assert properties["evidence_ids"]["items"]["enum"] == sorted(u.id for u in current.utterances)
        assert properties["event_ids"]["items"]["enum"] == ["c1-s1-e1"]
        assert properties["existing_record_ids"]["maxItems"] == 0
    assert definitions["KnowledgeChange"]["properties"]["character_id"]["enum"] == ["Hero", "keeper"]
    run = Run(annotation(current))
    annotate_information(run, empty_memory("story"), current, primary(current))
    assert run.llm.calls[0][1]["response_format"]["json_schema"]["schema"] == run.nodes[0][1]["schema"]


def test_an_introduction_with_a_real_knowledge_change_requires_its_own_fact_identifier():
    # r5 failed twice on a valid recipient with fact_id=null in a character introduction row.
    current = scene("keeper: この順に仮置きすれば流れが見える。\nHero: その配置はまだ決めていない。")
    row = observation(current, fact_id=None, content="相手が具体的な仮配置を提案した。",
        knowledge_changes=[{"character_id": "Hero", "kind": "knowledge", "change": "learned",
                            "acquired_at": "s1-u1"}],
        reader_introduction={"kind": "character", "entity_ids": ["keeper"],
            "description": "仮置きから作業を進める人物。", "audience_character_ids": []})
    before = copy.deepcopy(row)
    with pytest.raises(ValidationError):
        InformationAnnotation.model_validate(annotation(current, [row]))
    assert row == before
    good = copy.deepcopy(row)
    good["fact_id"] = "temporary_layout_proposal"
    run = Run(annotation(current, [row]), annotation(current, [good]))
    result = annotate_information(run, empty_memory("story"), current, primary(current))
    assert len(run.llm.calls) == 2 and len(run.saved) == 1
    update = result["knowledge_updates"][0]
    assert update["character_id"] == "Hero" and update["fact_id"] == good["fact_id"]
    assert update["content"] == row["content"] and update["evidence"] == row["evidence_ids"]
    assert result["introductions"][0]["entity_ids"] == ["keeper"]


def test_runtime_grammar_disallows_null_fact_with_nonempty_knowledge_changes():
    current = scene()
    schema = information_schema(empty_memory("story"), current, primary(current))
    branches = schema["properties"]["observations"]["items"]["anyOf"]
    assert {branch["$ref"] for branch in branches} == {
        "#/$defs/KnowledgeObservation", "#/$defs/NonKnowledgeObservation"}
    known = schema["$defs"]["KnowledgeObservation"]["properties"]
    no_change = schema["$defs"]["NonKnowledgeObservation"]["properties"]
    assert known["fact_id"]["type"] == "string" and "anyOf" not in known["fact_id"]
    assert known["knowledge_changes"]["minItems"] == 1
    assert no_change["knowledge_changes"]["maxItems"] == 0
    assert {item["type"] for item in no_change["fact_id"]["anyOf"]} == {"string", "null"}
    # No-change observations and reader-only introductions remain valid without fabricated fact IDs.
    value = annotation(current, [observation(current, fact_id=None, knowledge_changes=[])])
    assert InformationAnnotation.model_validate(value).observations[0].fact_id is None
