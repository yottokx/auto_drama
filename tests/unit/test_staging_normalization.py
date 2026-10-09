"""Repeated generated effects cannot create long unresponsive pauses."""

from copy import deepcopy

import pytest

from packages.narrative.staging import MAX_PAUSE_MS_PER_UTTERANCE, normalize_directions


def direction(identifier, *, kind="pause", utterance="line", timing="after", duration=500,
              **fields):
    return {"id": identifier, "kind": kind, "utterance_id": utterance,
            "timing": timing, "duration_ms": duration, **fields}


def test_real_repeated_after_pauses_keep_first_and_separate_start_pause():
    source = [direction("start", timing="start", duration=300)] + [
        direction(f"pause-{index}") for index in range(49)]

    result, report = normalize_directions(source)

    assert result == [source[0], source[1]]
    assert report["input_count"] == 50
    assert report["output_count"] == 2
    assert report["removed_count"] == report["duplicate_count"] == 48
    assert report["merge_count"] == report["clipped_count"] == 0
    assert report["policy"] == "direction-normalization"
    assert report["version"] == 1
    assert report["original_directions"] == source
    assert report["changes"][0]["source_index"] == 2
    assert report["changes"][0]["retained_source_index"] == 1


def test_mixed_pause_durations_keep_longest_and_first_id():
    source = [direction("first", duration=500), direction("second", duration=800),
              direction("third", duration=300)]

    result, report = normalize_directions(source)

    assert result == [{**source[0], "duration_ms": 800}]
    assert report["merge_count"] == report["removed_count"] == 2
    assert report["changes"][0]["normalized_duration_ms"] == 800
    assert report["changes"][1]["retained_direction_id"] == "first"


@pytest.mark.parametrize("kind,first_fields,second_fields", [
    ("focus", {"character_id": "ren"}, {"character_id": "aoi"}),
    ("position", {"character_id": "ren", "position": "left"},
     {"character_id": "ren", "position": "right"}),
    ("background", {"asset_id": "classroom"}, {"asset_id": "schoolyard"}),
    ("pause", {"character_id": "ren"}, {"character_id": "aoi"}),
])
def test_changed_target_or_asset_is_preserved(kind, first_fields, second_fields):
    source = [direction("first", kind=kind, **first_fields),
              direction("second", kind=kind, **second_fields)]

    result, report = normalize_directions(source)

    assert result == source
    assert report["changes"] == []
    assert "original_directions" not in report


def test_exact_duplicate_action_ignores_id_but_not_duration():
    source = [direction("first", kind="blackout", duration=8000),
              direction("duplicate", kind="blackout", duration=8000),
              direction("different", kind="blackout", duration=9000)]

    result, report = normalize_directions(source)

    assert result == [source[0], source[2]]
    assert report["duplicate_count"] == 1
    assert report["clipped_count"] == 0


def test_action_separator_preserves_intentional_nonconsecutive_pauses():
    source = [direction("first"), direction("action", kind="focus", character_id="ren"),
              direction("second")]

    result, report = normalize_directions(source)

    assert result == source
    assert report["changes"] == []


def test_interleaved_anchors_follow_compiler_grouping_and_preserve_source_order():
    source = [direction("first"), direction("other", utterance="other"),
              direction("start", timing="start"), direction("duplicate"),
              direction("merged", duration=800), direction("other-duplicate", utterance="other")]

    result, report = normalize_directions(source)

    assert result == [{**source[0], "duration_ms": 800}, source[1], source[2]]
    assert report["removed_count"] == 3
    assert report["duplicate_count"] == 2
    assert report["merge_count"] == 1


def test_pause_budget_uses_execution_timing_order_and_each_utterance_gets_own_budget():
    source = [direction("after", duration=800),
              direction("other", utterance="other", duration=1500),
              direction("start", timing="start", duration=800),
              direction("before", timing="before", duration=800)]

    result, report = normalize_directions(source)

    assert result == [source[1], {**source[2], "duration_ms": 700}, source[3]]
    assert report["clipped_count"] == 2
    assert report["removed_count"] == 1
    assert [change["action"] for change in report["changes"]] == ["pause_clip", "pause_drop"]
    assert report["max_pause_ms_per_utterance"] == MAX_PAUSE_MS_PER_UTTERANCE == 1500


def test_budget_clips_last_fitting_pause_within_group_and_drops_remaining_pauses():
    source = [direction("first", duration=900),
              direction("action-1", kind="focus", character_id="ren"),
              direction("second", duration=900),
              direction("action-2", kind="focus", character_id="aoi"),
              direction("third", duration=200)]

    result, report = normalize_directions(source)

    assert result == [source[0], source[1], {**source[2], "duration_ms": 600}, source[3]]
    assert report["clipped_count"] == 2
    assert report["removed_count"] == 1


def test_no_mutation_and_idempotence_even_when_dropped_pause_exposes_duplicates():
    source = [direction("first", timing="before", duration=9000),
              direction("action-1", kind="focus", character_id="ren"),
              direction("dropped", duration=800),
              direction("action-2", kind="focus", character_id="ren")]
    original = deepcopy(source)

    result, report = normalize_directions(source)
    repeated, repeated_report = normalize_directions(result)

    assert source == original
    assert result == [{**source[0], "duration_ms": 1500}, source[1]]
    assert repeated == result
    assert repeated_report["changes"] == []
    assert repeated_report["removed_count"] == repeated_report["clipped_count"] == 0
    assert report["removed_count"] == 2
    result[0]["duration_ms"] = 1
    report["original_directions"][0]["duration_ms"] = 2
    assert source == original


def test_empty_directions_have_stable_unchanged_report():
    result, report = normalize_directions([])

    assert result == []
    assert report["input_count"] == report["output_count"] == report["removed_count"] == 0
    assert report["changes"] == []
    assert "original_directions" not in report
