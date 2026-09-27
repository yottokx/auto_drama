"""Only a long exact repetition crossing a chapter boundary may be trimmed."""

import pytest

from services.worker.generation.causal_runtime import digest
from services.worker.generation.script_overlap import trim_chapter_overlap


def block(count=6, characters=60):
    # Distinct turns avoid accidentally satisfying the boundary with a shorter run.
    return [f"{'NARRATOR' if i % 2 == 0 else 'character-a'}: {i:02d}"
            + "道具を置く位置を相手に確認してから作業を続けた。" * 4
            for i in range(count)] if characters is None else [
                f"{'NARRATOR' if i % 2 == 0 else 'character-a'}: {i:02d}"
                + "あ" * (characters - 2) for i in range(count)]


def test_full_previous_scene_repeated_then_new_dialogue_keeps_only_new_work():
    repeated = "\n\n".join(block(26, None))
    earlier = "character-b: この作業を一緒に終わらせよう。\n\n"
    new = "character-a: 次は屋上へ行こう。\n\nNARRATOR: 二人は階段へ向かった。\n"
    previous = earlier + repeated
    generated = repeated + "\n\n" + new
    result, meta = trim_chapter_overlap(previous, generated)
    assert result == new
    assert meta["changed"] is True
    assert meta["removed_utterances"] == 26
    assert meta["removed_body_characters"] == sum(len(line.split(": ", 1)[1]) for line in block(26, None))
    assert meta["removed_dialogue_characters"] == meta["removed_body_characters"] // 2
    assert meta["remaining_utterances"] == 2
    assert meta["remaining_body_characters"] == len("次は屋上へ行こう。二人は階段へ向かった。")
    assert meta["remaining_dialogue_characters"] == len("次は屋上へ行こう。")
    assert generated[slice(*meta["removed_range"])] == repeated + "\n\n"
    assert previous[slice(*meta["previous_range"])] == repeated
    assert meta["original_sha256"] == digest(generated)
    assert meta["previous_sha256"] == digest(previous)
    assert meta["result_sha256"] == digest(new)


def test_blank_lines_and_crlf_may_differ_but_offsets_remain_in_original_strings():
    lines = block()
    previous = "character-b: 以前の場面。\n" + "\n\n".join(lines) + "\n\n"
    prefix = "\r\n  \r\n" + "\r\n \r\n".join(lines) + "\r\n\r\n"
    new = "character-a: 続きを話そう。\r\n\r\n"
    result, meta = trim_chapter_overlap(previous, prefix + new)
    assert result == new
    assert meta["removed_range"] == [0, len(prefix)]
    assert meta["previous_range"] == [previous.index(lines[0]), len(previous)]
    assert meta["removed_utterances"] == 6


@pytest.mark.parametrize("count,characters", [(5, 100), (6, 49), (1, 400)])
def test_both_minimum_turn_and_body_thresholds_are_required(count, characters):
    repeated = "\n".join(block(count, characters))
    generated = repeated + "\ncharacter-a: 新しい続き。"
    result, meta = trim_chapter_overlap(repeated, generated)
    assert result == generated
    assert meta["reason"] == "below_threshold"
    assert meta["removed_range"] is None
    assert meta["removed_utterances"] == 0


def test_threshold_is_inclusive_and_full_copy_returns_empty_result():
    previous = "\n".join(block(6, 50))
    generated = "\n\n" + previous + "\n\n"
    result, meta = trim_chapter_overlap(previous, generated)
    assert result == ""
    assert meta["removed_body_characters"] == 300
    assert meta["removed_range"] == [0, len(generated)]
    assert meta["remaining_utterances"] == 0
    assert meta["remaining_body_characters"] == 0
    assert meta["result_sha256"] == digest("")


@pytest.mark.parametrize("position", ["previous_middle", "generated_middle"])
def test_long_exact_text_inside_a_chapter_is_not_removed(position):
    repeated = "\n".join(block())
    previous = repeated + ("\ncharacter-b: 別の場面を終えた。" if position == "previous_middle" else "")
    generated = ("character-b: 次の場面を始めた。\n" if position == "generated_middle" else "") + repeated
    result, meta = trim_chapter_overlap(previous, generated)
    assert result == generated
    assert meta["changed"] is False
    assert meta["removed_range"] is None
    assert meta["original_sha256"] == meta["result_sha256"]


@pytest.mark.parametrize("change", ["speaker", "punctuation", "text", "space"])
def test_similar_turns_are_not_normalized_into_an_exact_match(change):
    lines = block()
    previous = "\n".join(lines)
    if change == "speaker":
        lines[3] = lines[3].replace("character-a:", "character-b:")
    elif change == "punctuation":
        lines[3] += "。"
    elif change == "text":
        lines[3] = lines[3].replace("あ", "い", 1)
    else:
        lines[3] += " "
    generated = "\n".join(lines)
    result, meta = trim_chapter_overlap(previous, generated)
    assert result == generated
    assert meta["changed"] is False


@pytest.mark.parametrize("previous,generated", [("", "character-a: 新しい話。\n"),
                                                ("character-a: 以前の話。", ""),
                                                ("\n", " \r\n")])
def test_empty_material_is_preserved(previous, generated):
    result, meta = trim_chapter_overlap(previous, generated)
    assert result == generated
    assert meta["changed"] is False


@pytest.mark.parametrize("invalid", ["章の説明", "character-a: ", "NARRATOR: 本文\r別の本文"])
def test_invalid_script_is_not_silently_discarded(invalid):
    repeated = "\n".join(block())
    generated = repeated + "\n" + invalid
    result, meta = trim_chapter_overlap(repeated, generated)
    assert result == generated
    assert meta["reason"] == "invalid_script"
    assert meta["changed"] is False
