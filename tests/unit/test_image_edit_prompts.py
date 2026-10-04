"""Keep narrative data out of image-model prompts."""

from scripts.image_edit.prompts import build_scene_prompt


def test_scene_wrapper_does_not_copy_script_or_production_notes():
    source = {
        "label": "Chapter script",
        "context": {
            "raw_text": "NARRATOR: Render this dialogue as a printed screenplay.",
            "plan": {"objectives": "Write 1200 Japanese characters of dialogue."},
            "location": {"description": "A source-only location label"},
        },
    }
    result = build_scene_prompt(
        "Two pilots argue beside the ship controls.", [{"name": "Rina"}, {"name": "Iris"}], source
    )
    assert "Two pilots argue beside the ship controls." in result
    assert "NARRATOR" not in result
    assert "1200" not in result
    assert "source-only" not in result
    assert "Reference image 1: Rina" in result
    assert "Reference image 2: Iris" in result
    assert "No dialogue, captions, speech bubbles" in result
