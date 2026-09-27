"""A speaker typo must be corrected in the rejected source, never guessed away."""

import pytest

from packages.contracts.m3 import ScenePlan
from services.worker.generation.narrative import SceneSourceError, _scene_text, _source_grammar


class SourceLLM:
    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = 0
        self.calls, self.trace = [], []
        self.config = {"llm": {"model_id": "fake", "temperature": 0.5, "max_tokens": 3072}}
        self.payload = {}

    def chat(self, stage, messages, **options):
        self.requests += 1
        self.calls.append((stage, messages, options))
        return next(self.responses)


def plan():
    return ScenePlan(id="s3", location_id="gate", character_ids=["Hero", "support-1"],
        objectives="相談する", start_state="到着", end_state="協力", atmosphere="静か",
        required_events=[{"id": "ask", "description": "返事を受けて動く"}])


@pytest.mark.parametrize("speaker", ["support-t1", "main-character-from-previous-scene"])
def test_rejected_speaker_is_regenerated_with_exact_scene_cast(speaker):
    invalid = speaker + ": 開けて。\nNARRATOR: 手を伸ばす。"
    valid = "Hero: 開けて。\nNARRATOR: 手を伸ばす。\nsupport-1: わかった。"
    llm = SourceLLM([{"content": invalid}, {"content": valid}])
    raw, utterances = _scene_text(llm, "write", plan())
    assert raw == valid
    assert [u.speaker_id for u in utterances] == ["Hero", None, "support-1"]
    assert len(llm.calls) == 2
    assert "grammar" not in llm.calls[0][2]
    assert speaker not in llm.calls[1][2]["grammar"]
    assert '"support-1: "' in llm.calls[1][2]["grammar"]
    assert speaker in llm.calls[1][1][-1]["content"]  # actionable rejection feedback
    assert [t["raw_text"] for t in llm.trace if t["type"] == "scene_text"] == [invalid, valid]


def test_valid_source_keeps_original_request_and_bytes():
    original = "Hero:  空白を残す。\r\n\r\nNARRATOR: 頷いた。\r\n"
    llm = SourceLLM([{"content": original}])
    raw, _ = _scene_text(llm, "original prompt", plan())
    assert raw == original
    assert len(llm.calls) == 1
    assert llm.calls[0][1][-1]["content"] == "original prompt"
    assert llm.calls[0][2] == {"allow_truncated": True}


def test_repeated_bad_source_is_bounded_and_preserves_failed_checkpoint():
    llm = SourceLLM([{"content": "unknown: 一回目。"}, {"content": "unknown: 二回目。"}])
    llm.requests = 13
    with pytest.raises(SceneSourceError, match="Unknown scene speaker"):
        _scene_text(llm, "write", plan())
    assert len(llm.calls) == 2
    assert llm.failure_request == 14


def test_constrained_repair_continues_exact_anchor_without_losing_source():
    prefix = "Hero: 門の向こうに"
    full = prefix + "薬を届ける。\nNARRATOR: 袋を差し出した。"
    llm = SourceLLM([{"content": "wrong: 門を開けて。"},
                    {"content": prefix, "_finish_reason": "length"},
                    {"content": full, "_finish_reason": "stop"}])
    raw, mapped = _scene_text(llm, "write", plan())
    assert raw == full
    assert len(mapped) == 2
    assert llm.calls[2][2]["grammar"].startswith('root ::= "' + prefix + '" ')


@pytest.mark.parametrize("prefix", ["Hero", "Hero:", "Hero: ", "Hero: 続く", "Hero: 続く\n"])
def test_continuation_grammar_accepts_each_source_boundary(prefix):
    grammar = _source_grammar(plan(), prefix)
    assert grammar.startswith("root ::= ")
    assert '"support-1: "' in grammar

