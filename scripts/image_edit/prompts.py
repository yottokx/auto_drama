"""Image-only instructions; narrative source must pass through the scene LLM."""

from __future__ import annotations

PORTRAIT_PRESETS = {
    "笑顔": "Change only the character's facial expression to a warm, natural smile.",
    "怒り": "Change only the character's facial expression to anger, with furrowed brows.",
    "悲しみ": "Change only the character's facial expression to sadness, with downcast eyes.",
    "驚き": "Change only the character's facial expression to surprise, with widened eyes.",
    "腕を組む": "Change the character's pose to standing with their arms crossed.",
    "手を振る": "Change the character's pose to waving one hand in greeting.",
    "冬服": "Change the character's outfit to a winter coat and scarf suitable for their design.",
    "フォーマル": "Change the character's outfit to elegant formal attire suitable for their design.",
}


def build_portrait_prompt(instruction: str, transparent: bool = True, preserve: bool = True) -> str:
    parts = [instruction.strip()]
    if preserve:
        parts.append(
            "Keep the same character identity, facial features, hairstyle, hair and eye "
            "colors, body proportions, and illustration style as reference image 1. "
            "Keep all attributes and details unchanged except the requested edits."
        )
    parts.append("Show a single character with no cropped head or limbs.")
    if transparent:
        parts.append(
            "This is an RGBA image with transparency. The image has an alpha channel "
            "and the background is transparent."
        )
    return "\n".join(value for value in parts if value)


def build_scene_prompt(instruction: str, references: list[dict], scene: dict | None = None) -> str:
    # Compatibility with callers passing scene: source belongs only to the LLM,
    # never append script, dialogue, objectives or scene labels to an image prompt.
    mapping = "\n".join(
        f"Reference image {index}: {row.get('name') or f'Character {index}'}."
        for index, row in enumerate(references, 1)
    )
    parts = [
        mapping,
        (
            "Create one coherent illustration using the referenced characters. Preserve each "
            "character's individual face, hairstyle, hair and eye colors, clothing details, "
            "body proportions, and illustration style. Do not mix features between characters."
        ),
        instruction.strip(),
        (
            "Depict a single visual moment as a full scene illustration. Do not render the prompt "
            "or character names as text. No dialogue, captions, speech bubbles, page layout, "
            "readable lettering or watermarks. Screen graphics and props may use abstract shapes."
        ),
    ]
    return "\n".join(value for value in parts if value)
