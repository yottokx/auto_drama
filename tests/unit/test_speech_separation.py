from collections.abc import Mapping
from itertools import pairwise

import pytest

from packages.narrative.speech import parenthetical_candidates, separate_stage_directions
from packages.narrative.validation import parse_scene_text


def separate(raw, directions=(), **kwargs):
    return separate_stage_directions(raw, "s1", {"human", "dog"}, directions,
                                     {"human": "陽介", "dog": "ポチ"}, **kwargs)


def assert_source_coverage(raw, result):
    original = parse_scene_text(raw, "s1", {"human", "dog"})
    for utterance in original:
        segments = [segment for segment in result.segments
                    if segment.original_utterance_id == utterance.id]
        assert "".join(segment.text for segment in segments) == utterance.display_text
        assert segments[0].source_start == utterance.source_start
        assert segments[-1].source_end == utterance.source_end
        for left, right in pairwise(segments):
            assert left.source_end == right.source_start
    for segment in result.segments:
        assert raw[segment.source_start:segment.source_end] == segment.text
        if segment.utterance_id:
            assert result.raw_text[segment.normalized_source_start:segment.normalized_source_end] \
                == segment.display_text
        else:
            assert segment.kind == "delivery"
            assert segment.display_text == ""
            assert segment.normalized_source_start is None
            assert segment.normalized_source_end is None


def test_candidates_preserve_literal_parentheses_nested_pairs_and_positions():
    raw = 'human: 「(笑)」という表記だよ。（笑う(小さく)）(笑)\nNARRATOR: (地の文)'
    candidates = parenthetical_candidates(raw, "s1", {"human"})
    assert [item.id for item in candidates] == ["s1-u1-p1", "s1-u1-p2", "s1-u1-p3"]
    assert [item.text for item in candidates] == ["(笑)", "（笑う(小さく)）", "(笑)"]
    assert candidates[0].source_start != candidates[2].source_start
    assert all(raw[item.source_start:item.source_end] == item.text for item in candidates)
    assert all(item.speaker_id == "human" and item.utterance_id == "s1-u1" for item in candidates)
    result = separate(raw)
    assert result.raw_text == raw
    assert_source_coverage(raw, result)


@pytest.mark.parametrize("body", ["(未完", "（外側(内側)", "(不一致）", "（不一致)"])
def test_unbalanced_parentheses_are_not_silently_repaired(body):
    raw = "human: " + body
    assert parenthetical_candidates(raw, "s1", {"human"}) == []
    assert separate(raw).raw_text == raw


def test_mixed_speech_and_whole_action_are_split_with_actor_attribution():
    raw = "human: おかえり。（笑顔になる）待っていたよ。\ndog: （しっぽを振る）"
    result = separate(raw, ["s1-u1-p1", "s1-u2-p1"])
    assert result.raw_text == (
        "human: おかえり。\nNARRATOR: 陽介：（笑顔になる）\nhuman: 待っていたよ。\n"
        "NARRATOR: ポチ：（しっぽを振る）"
    )
    assert result.utterance_map == {"s1-u1": ["s1-u1", "s1-u2", "s1-u3"], "s1-u2": ["s1-u4"]}
    assert result.segments[1].kind == "narration"
    assert result.segments[1].speaker_id is None
    assert result.segments[1].original_speaker_id == "human"
    assert_source_coverage(raw, result)


def test_natural_narration_replaces_visible_action_but_retains_original_trace():
    raw = "human: （笑顔で迎える）おかえり。\ndog: ワン！（しっぽを振る）"
    result = separate(raw, ["s1-u1-p1", "s1-u2-p1"], direction_narrations={
        "s1-u1-p1": "陽介は笑顔で迎えた。", "s1-u2-p1": "ポチはしっぽを振った。",
    })
    assert result.raw_text == (
        "NARRATOR: 陽介は笑顔で迎えた。\nhuman: おかえり。\n"
        "dog: ワン！\nNARRATOR: ポチはしっぽを振った。"
    )
    assert result.segments[0].text == "（笑顔で迎える）"
    assert result.segments[0].display_text == "陽介は笑顔で迎えた。"
    assert_source_coverage(raw, result)


def test_blanks_crlf_whitespace_and_existing_narrator_lines_are_preserved():
    raw = "\r\nNARRATOR: (既存)\r\n  \r\nhuman:   (うなずく)  （座る）  \r\n\r\n"
    result = separate(raw, ["s1-u2-p1", "s1-u2-p2"])
    assert result.raw_text == (
        "\r\nNARRATOR: (既存)\r\n  \r\nNARRATOR:   陽介：(うなずく)  \r\n"
        "NARRATOR: 陽介：（座る）  \r\n\r\n"
    )
    assert len(parse_scene_text(result.raw_text, "s1", {"human"})) == 3
    assert_source_coverage(raw, result)


def test_names_are_flattened_and_unknown_or_blank_names_fall_back_to_id():
    raw = "human: (うなずく)\ndog: (座る)"
    result = separate_stage_directions(raw, "s1", {"human", "dog"},
                                      ["s1-u1-p1", "s1-u2-p1"], {"human": "陽\n介", "dog": " "})
    assert result.raw_text == "NARRATOR: 陽 介：(うなずく)\nNARRATOR: dog：(座る)"
    result = separate_stage_directions("human: (うなずく)", "s1", {"human"}, ["s1-u1-p1"], {})
    assert result.raw_text == "NARRATOR: human：(うなずく)"


def test_only_selected_repeated_parenthetical_occurrence_is_replaced():
    raw = 'human: 「(笑)」と書いた。(笑)おかしいね。'
    result = separate(raw, ["s1-u1-p2"])
    assert result.raw_text == 'human: 「(笑)」と書いた。\nNARRATOR: 陽介：(笑)\nhuman: おかしいね。'
    assert_source_coverage(raw, result)


def test_delivery_is_hidden_while_surrounding_dialogue_stays_in_one_utterance():
    raw = "human: （小声で）おかえり。（優しく）待っていたよ。\ndog: ワン！"
    result = separate(raw, delivery_candidate_ids=["s1-u1-p1", "s1-u1-p2"])
    assert result.raw_text == "human: おかえり。待っていたよ。\ndog: ワン！"
    assert result.utterance_map == {"s1-u1": ["s1-u1"], "s1-u2": ["s1-u2"]}
    assert [segment.kind for segment in result.segments] == [
        "delivery", "dialogue", "delivery", "dialogue", "dialogue",
    ]
    assert result.segments[1].utterance_id == result.segments[3].utterance_id == "s1-u1"
    assert_source_coverage(raw, result)


def test_delivery_and_narration_share_a_line_without_leaking_into_speech():
    raw = "human: （小声で）おかえり。（椅子を引く）  （穏やかに）座って。"
    result = separate(raw, ["s1-u1-p2"], delivery_candidate_ids=["s1-u1-p1", "s1-u1-p3"],
                      direction_narrations={"s1-u1-p2": "陽介は椅子を引いた。"})
    assert result.raw_text == "human: おかえり。\nNARRATOR: 陽介は椅子を引いた。\nhuman:   座って。"
    assert_source_coverage(raw, result)


@pytest.mark.parametrize("raw", ["human: （ため息）", "human:   (静かに)  ",
                               "human: (静かに)（ゆっくり）"])
def test_delivery_cannot_erase_the_entire_speech_line(raw):
    ids = [candidate.id for candidate in parenthetical_candidates(raw, "s1", {"human"})]
    with pytest.raises(ValueError, match="entire speech line"):
        separate(raw, delivery_candidate_ids=ids)


@pytest.mark.parametrize("directions, deliveries, message", [
    (["missing"], [], "Unknown"), (["s1-u1-p1", "s1-u1-p1"], [], "Duplicate"),
    ([], ["missing"], "Unknown"), ([], ["s1-u1-p1", "s1-u1-p1"], "Duplicate"),
    (["s1-u1-p1"], ["s1-u1-p1"], "both"), ("s1-u1-p1", [], "iterable"),
])
def test_invalid_selection_is_rejected(directions, deliveries, message):
    with pytest.raises((ValueError, TypeError), match=message):
        separate("human: (うなずく)はい。", directions, delivery_candidate_ids=deliveries)


@pytest.mark.parametrize("narration", ["", "  ", "一行目\n二行目", "一行目\r二行目", "声\x00", 
                                      "NARRATOR: 陽介は笑った。", "human: hi", "長" * 1001])
def test_invalid_narration_cannot_inject_source_structure(narration):
    with pytest.raises(ValueError, match="single-line"):
        separate("human: (うなずく)はい。", ["s1-u1-p1"],
                 direction_narrations={"s1-u1-p1": narration})


def test_unselected_narration_mapping_is_rejected():
    with pytest.raises(ValueError, match="unselected"):
        separate("human: (うなずく)はい。", direction_narrations={"s1-u1-p1": "陽介はうなずいた。"})


def test_duplicate_narration_mapping_is_rejected():
    class DuplicateMapping(Mapping):
        def __iter__(self):
            return iter(["s1-u1-p1", "s1-u1-p1"])

        def __len__(self):
            return 2

        def __getitem__(self, key):
            return "陽介はうなずいた。"

    with pytest.raises(ValueError, match="Duplicate"):
        separate("human: (うなずく)はい。", ["s1-u1-p1"], direction_narrations=DuplicateMapping())


def test_narration_can_contain_literal_parentheses():
    result = separate("human: （紙を見せる）", ["s1-u1-p1"],
                      direction_narrations={"s1-u1-p1": "陽介は『(仮)』と書かれた紙を見せた。"})
    assert result.raw_text == "NARRATOR: 陽介は『(仮)』と書かれた紙を見せた。"
