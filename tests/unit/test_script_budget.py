"""Scene reservations follow intended volume and measured output, not spare input."""

from copy import deepcopy

import pytest

from services.worker.generation.context_budget import ContextPolicy
from services.worker.generation.script_budget import (
    SceneSize,
    SceneTokenPolicy,
    scene_output_budget,
)
from services.worker.generation.script_cast import ScriptOptions
from services.worker.generation.script_chapter_plan import (
    ChapterScriptPlan,
    FirstChapterPlan,
    project_plan,
)


def size(body=1000, dialogue=500):
    return SceneSize(length_weight=1, body_characters=body, dialogue_characters=dialogue)


def test_large_scene_gets_more_output_without_reducing_its_body_target():
    policy = SceneTokenPolicy()
    small = scene_output_budget(size(), ["aoi", "ren"], policy, 16384)
    entire_chapter = scene_output_budget(size(4000, 2000), ["aoi", "ren"], policy, 16384)
    assert 2048 <= small["max_tokens"] <= 4096
    assert entire_chapter["max_tokens"] > small["max_tokens"]
    assert entire_chapter["scene_size"]["body_characters"] == 4000
    assert entire_chapter["scene_size"]["dialogue_characters"] == 2000
    assert not entire_chapter["limited_by_profile"]


def test_long_speaker_labels_reserve_more_than_short_labels_for_same_body():
    short = scene_output_budget(size(1000, 1000), ["aoi"], SceneTokenPolicy(), 16384)
    long = scene_output_budget(size(1000, 1000), ["character-" + "a" * 60], SceneTokenPolicy(), 16384)
    assert long["estimated_label_characters"] > short["estimated_label_characters"]
    assert long["max_tokens"] > short["max_tokens"]
    assert long["scene_size"] == short["scene_size"]


def test_measured_samples_increase_reservation_and_are_recorded_without_changing_inputs():
    policy = SceneTokenPolicy()
    target = size()
    samples = [{"step": "c001-s1-text", "body_characters": 1000, "completion_tokens": 4200},
               {"step": "c001-s2-text", "body_characters": 0, "completion_tokens": 10000}]
    before = deepcopy(samples)
    original_size = target.model_dump()
    baseline = scene_output_budget(target, ["aoi"], policy, 16384)
    observed = scene_output_budget(target, ["aoi"], policy, 16384, samples)
    assert observed["max_tokens"] > baseline["max_tokens"]
    assert observed["observed_tokens_per_body_character"] == 4.2
    assert observed["samples"] == before
    assert samples == before and target.model_dump() == original_size
    assert observed["policy"] == policy.model_dump()
    assert observed == scene_output_budget(target, ["aoi"], policy, 16384, deepcopy(before))


def test_short_observed_output_does_not_shrink_the_intended_volume_estimate():
    args = (size(4000, 2000), ["aoi"], SceneTokenPolicy(), 16384)
    baseline = scene_output_budget(*args)
    measured = scene_output_budget(*args, samples=[{"body_characters": 1000, "completion_tokens": 100}])
    assert measured["max_tokens"] == baseline["max_tokens"]
    assert measured["scene_size"]["body_characters"] == 4000


def test_profile_cap_is_explicit_without_rewriting_target_or_recommendation():
    target = size(8000, 4000)
    unrestricted = scene_output_budget(target, ["aoi"], SceneTokenPolicy(), 32768)
    capped = scene_output_budget(target, ["aoi"], SceneTokenPolicy(), 4096)
    assert capped["max_tokens"] == capped["profile_output_limit"] == 4096
    assert capped["limited_by_profile"]
    assert capped["recommended_tokens"] == unrestricted["recommended_tokens"]
    assert capped["scene_size"] == unrestricted["scene_size"] == target.model_dump()
    assert unrestricted["max_tokens"] > 8192


@pytest.mark.parametrize("limit", [0, -1, True, 4.5])
def test_invalid_profile_allowance_is_not_silently_accepted(limit):
    with pytest.raises(ValueError, match="allowance"):
        scene_output_budget(size(), ["aoi"], SceneTokenPolicy(), limit)


def test_input_capacity_does_not_shrink_output_or_intended_story_volume():
    reservation = scene_output_budget(size(4000, 2000), ["aoi"], SceneTokenPolicy(), 16384)
    before = deepcopy(reservation)
    results = []
    for capacity in (16384, 32768, 65536):
        policy = ContextPolicy(capacity, capacity, capacity, 512, False)
        results.append(policy.budget(14000, reservation["max_tokens"], capacity))
    assert [row["fits"] for row in results] == [False, True, True]
    assert {row["output_tokens"] for row in results} == {reservation["max_tokens"]}
    assert reservation == before
    assert reservation["scene_size"]["body_characters"] == 4000


def test_projected_scene_sizes_are_durable_and_public_scene_contract_stays_clean():
    draft = FirstChapterPlan.model_validate({"new_characters": [],
        "locations": [{"id": "hall", "name": "廊下", "description": "学校の廊下", "time_of_day": "夕方",
                       "atmosphere": "静かな放課後", "image_prompt": "empty school hallway"}],
        "scenes": [{"id": f"scene-{n}", "location_id": "hall", "character_ids": ["aoi"],
                    "objectives": "短い目的", "start_state": "開始", "end_state": "終了", "atmosphere": "静か",
                    "required_events": [{"id": f"event-{n}", "description": "話す"}], "length_weight": weight}
                   for n, weight in [(1, 1), (2, 3)]]})
    before = draft.model_dump()
    plan = project_plan(draft, ScriptOptions())
    assert plan.scene_sizes["scene-1"].body_characters == 1000
    assert plan.scene_sizes["scene-2"].body_characters == 3000
    assert sum(row.dialogue_characters for row in plan.scene_sizes.values()) == 2000
    assert all("length_weight" not in row.model_dump() for row in plan.scenes)
    assert all("body_characters" not in row.model_dump() for row in plan.scenes)
    assert "本文約1000文字、うち台詞約500文字" in plan.scenes[0].objectives
    assert ChapterScriptPlan.model_validate_json(plan.model_dump_json()) == plan
    assert draft.model_dump() == before
