"""Previous scene prose is immutable evidence, never a shortened writing sample."""

import copy

import pytest

from packages.contracts.m3 import NarrativeScene, ScenePlan
from packages.narrative.story_ledger import (
    apply_chapter_memory,
    empty_memory,
    source_catalog,
    text_hash,
)
from packages.narrative.validation import parse_scene_text
from services.worker.generation.causal_context import (
    attach_preceding_scene,
    build_context,
    retrieve_context_evidence,
)


def scene(raw, scene_id="s1"):
    utterances = parse_scene_text(raw, scene_id, {"Hero", "Keeper"})
    return NarrativeScene(id=scene_id,
        plan=ScenePlan(id=scene_id, location_id="library", character_ids=["Hero", "Keeper"],
            objectives="受け取った手掛かりで行動する", start_state="手掛かりを受け取った",
            required_events=[{"id": "change", "description": "新しい行動を選ぶ"}],
            end_state="行動の結果を引き継ぐ", atmosphere="静かな相談"),
        raw_text=raw, utterances=utterances, directions=[],
        review={"passed": True, "issues": [], "events": [{"event_id": "change", "dramatized": True,
            "evidence_utterance_ids": [utterances[0].id], "reason": "行動を原文で確認した"}]})


def adopt(previous, value, chapter, *, add_span=False):
    refs = source_catalog([value], storyline_id="story", chapter_number=chapter)
    evidence = [refs[0]]
    if add_span:
        start, end = refs[1].source_start + 1, refs[1].source_end
        evidence.append(refs[1].model_copy(update={"utterance_id": None, "source_start": start,
            "source_end": end, "text_hash": text_hash(value.raw_text[start:end])}))
    return apply_chapter_memory(previous, {"chapter_number": chapter, "summary": "選択して行動した",
        "events": [{"id": f"event-{chapter}", "description": "次の行動を選んだ",
                    "character_ids": ["Hero"], "location_ids": ["library"],
                    "story_time": f"第{chapter}日", "evidence": evidence}]},
        scenes=[value], storyline_id="story", chapter_number=chapter)


def fixture():
    old = scene("Hero: 鍵を受け取った。\r\nKeeper: まず鍵の刻印を調べよう。\r\n")
    current = scene("Hero: 刻印は旧図書室を指している。\r\n\r\n"
                    "NARRATOR: 二人はすでに扉を開け、棚の前に立っていた。\r\n"
                    "Keeper: 今度は棚の裏を確かめよう。\r\n")
    memory = adopt(adopt(empty_memory("story"), old, 1), current, 2, add_span=True)
    packet = build_context(memory, scope="scene", character_ids=["Hero", "Keeper"], event_ids=["event-1"])
    return old, current, memory, packet


def test_full_preceding_source_keeps_speech_order_and_only_deduplicates_matching_provenance():
    _, current, memory, packet = fixture()
    preceding = {"chapter_number": 2, "scene": current.model_dump(mode="json")}
    before = copy.deepcopy((packet, memory.model_dump(mode="json"), preceding))
    attached = attach_preceding_scene(packet, memory, preceding)
    direct = attached["direct_preceding_scene"]
    assert direct["raw_text"] == current.raw_text
    assert direct["scene_revision"] == text_hash(current.raw_text)
    assert direct["chapter_number"] == 2 and direct["scene_id"] == "s1"
    assert direct["utterances"] == [{"id": u.id, "speaker": u.speaker_id,
        "source_start": u.source_start, "source_end": u.source_end} for u in current.utterances]
    # Even uncited dialogue at the end is supplied in its original position.
    assert direct["utterances"][-1]["id"] == "s1-u3"
    assert [quote["chapter"] for quote in attached["source_quotes"]] == [1]
    assert attached["source_quotes"] == [quote for quote in packet["source_quotes"] if quote["chapter"] == 1]
    pointers = attached["source_references_to_preceding_scene"]
    assert len(pointers) == 2
    assert {pointer["utterance_id"] for pointer in pointers.values()} == {"s1-u1", None}
    original_quotes = {quote["id"]: quote for quote in packet["source_quotes"]}
    for alias, pointer in pointers.items():
        exact = current.raw_text[pointer["source_start"]:pointer["source_end"]]
        assert exact == original_quotes[alias]["text"]
        assert text_hash(exact) == pointer["text_hash"]
        lookup = retrieve_context_evidence(memory, memory_hash=packet["memory_hash"], source_ids=[alias])
        assert lookup["status"] == "ready"
        assert lookup["source_quotes"] == [original_quotes[alias]]
    assert attached["coverage"]["supplied_source_ids"] == packet["coverage"]["supplied_source_ids"]
    assert attached["coverage"]["new_semantic_checks_performed"] is False
    assert attached["coverage"]["direct_preceding_scene"]["complete_raw_text_supplied"] is True
    assert (packet, memory.model_dump(mode="json"), preceding) == before
    assert attach_preceding_scene(attached, memory, preceding) == attached
    attached["events"][0]["description"] = "変更"
    assert packet["events"][0]["description"] != "変更"


def test_same_scene_id_from_another_chapter_cannot_replace_the_preceding_scene():
    old, _, memory, packet = fixture()
    with pytest.raises(ValueError, match="preceding adopted scene/revision"):
        attach_preceding_scene(packet, memory, {"chapter_number": 1, "scene": old.model_dump(mode="json")})


def test_chapter_identity_is_required_even_when_the_full_prose_is_identical():
    old, _, _, _ = fixture()
    memory = adopt(adopt(empty_memory("story"), old, 1), old, 2)
    packet = build_context(memory, scope="scene", event_ids=["event-1"])
    with pytest.raises(ValueError, match="preceding adopted scene/revision"):
        attach_preceding_scene(packet, memory, {"chapter_number": 1, "scene": old.model_dump(mode="json")})
    result = attach_preceding_scene(packet, memory, {"chapter_number": 2, "scene": old.model_dump(mode="json")})
    assert [quote["chapter"] for quote in result["source_quotes"]] == [1]


@pytest.mark.parametrize("change", ["raw", "display", "order", "speaker", "omitted"])
def test_modified_or_incomplete_preceding_source_is_rejected(change):
    _, current, memory, packet = fixture()
    value = current.model_dump(mode="json")
    if change == "raw":
        value["raw_text"] += "NARRATOR: 追加された未採用の結末。\n"
    elif change == "display":
        value["utterances"][1]["display_text"] = "違う文章"
    elif change == "order":
        value["utterances"] = list(reversed(value["utterances"]))
    elif change == "speaker":
        value["utterances"][0]["speaker_id"] = "Keeper"
    elif change == "omitted":
        value["utterances"].pop()
    with pytest.raises(ValueError):
        attach_preceding_scene(packet, memory, {"chapter_number": 2, "scene": value})


def test_stale_context_packet_and_corrupt_quote_are_rejected_before_deduplication():
    _, current, memory, packet = fixture()
    preceding = {"chapter_number": 2, "scene": current.model_dump(mode="json")}
    with pytest.raises(ValueError, match="memory_hash"):
        attach_preceding_scene({**packet, "memory_hash": "0" * 64}, memory, preceding)
    broken = copy.deepcopy(packet)
    next(quote for quote in broken["source_quotes"] if quote["chapter"] == 2)["text"] = "捏造"
    with pytest.raises(ValueError, match="quotation differs"):
        attach_preceding_scene(broken, memory, preceding)


def test_long_full_prose_is_not_cut_to_fit_a_nominal_context_limit():
    raw = "".join(f"Hero: 第{index}行。" + "引き継ぐ必要のある具体的な行動。" * 35 + "\n" for index in range(45))
    previous = scene(raw)
    memory = adopt(empty_memory("story"), previous, 1)
    packet = build_context(memory, scope="scene")
    result = attach_preceding_scene(packet, memory, {"chapter_number": 1, "scene": previous.model_dump(mode="json")})
    assert len(raw) > 16_384
    assert result["direct_preceding_scene"]["raw_text"] == raw
    assert len(result["direct_preceding_scene"]["utterances"]) == 45


def test_source_attachment_does_not_upgrade_an_insufficient_or_unreviewed_packet():
    _, current, memory, packet = fixture()
    packet.update(status="insufficient_evidence", missing_information=["未照合の矛盾を確認する必要がある"])
    result = attach_preceding_scene(packet, memory, {"chapter_number": 2, "scene": current.model_dump(mode="json")})
    assert result["status"] == packet["status"]
    assert result["missing_information"] == packet["missing_information"]
    assert result["coverage"]["new_semantic_checks_performed"] is False
    assert "extraction/context" in result["preceding_scene_instructions"]
    assert "今回の場面の文例ではありません" in result["preceding_scene_instructions"]
