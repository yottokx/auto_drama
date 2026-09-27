from __future__ import annotations

import pytest

from scripts.m0.image_smoke import resolve_prompts


def test_character_defaults_include_anima_quality_and_negative_tags():
    prompt, negative = resolve_prompts("character")

    assert prompt.startswith("masterpiece, best quality, score_7, safe, ")
    assert "full body" in prompt
    assert set(negative.split(", ")) == {
        "worst quality", "low quality", "score_1", "score_2", "score_3",
        "artist name", "blurry", "jpeg artifacts", "chromatic aberration", "cropped", "text",
    }


def test_background_defaults_keep_person_suppression():
    prompt, negative = resolve_prompts("background")
    _, character_negative = resolve_prompts("character")

    assert prompt.startswith("masterpiece, best quality, score_7, safe, ")
    assert "no humans" in prompt
    assert negative == character_negative + ", person, human, girl, boy, face, silhouette"


@pytest.mark.parametrize("mode", ["character", "background"])
def test_explicit_prompts_completely_replace_defaults(mode):
    assert resolve_prompts(mode, "custom illustration", "custom exclusions") == (
        "custom illustration", "custom exclusions"
    )


@pytest.mark.parametrize("mode", ["character", "background"])
def test_empty_negative_prompt_disables_all_negative_tags(mode):
    prompt, negative = resolve_prompts(mode, negative_prompt="")

    assert prompt.startswith("masterpiece, best quality, score_7, safe, ")
    assert negative == ""


def test_empty_positive_prompt_remains_invalid():
    with pytest.raises(ValueError, match="must contain text"):
        resolve_prompts("character", " ")
