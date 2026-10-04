"""Compile character-keyed visual directions to stable Qwen reference numbers.

The LLM interprets the scene, but never assigns image numbers. Both the subject
of each direction and the target of a gaze come from validated character IDs.
"""
from __future__ import annotations

import re

from scripts.audio.prompts import _clean_llm_content

VERSION = 2
_JAPANESE = re.compile(r"[\u3040-\u30ff\u3400-\u9fff]")
_REFERENCE = re.compile(r"\breference\b|\bimage\s*#?\s*\d+", re.IGNORECASE)


def _object(properties):
    return {"type": "object", "additionalProperties": False, "properties": properties,
        "required": list(properties)}


def schema(character_ids: list[str], image_ids: list[str]) -> dict:
    short = {"type": "string", "maxLength": 240}
    person = _object({"position": short, "action": short, "expression": short,
        "gaze_target": {"type": "string", "enum": [*character_ids, "scene", "camera", "none"]},
        "gaze_detail": short})
    item = _object({"id": {"type": "string", "enum": image_ids},
        "interpretation": {"type": "string", "minLength": 5, "maxLength": 600},
        "environment": {"type": "string", "maxLength": 500},
        "lighting": short, "framing": short,
        "characters": _object({cid: person for cid in character_ids})})
    return _object({"images": {"type": "array", "minItems": len(image_ids),
        "maxItems": len(image_ids), "items": item}})


SYSTEM = (
    "Interpret the selected scene as structured visual directions for a visual novel illustration. "
    "Return one image record per supplied ID: base first, then its variants, in exact order. "
    "Each interpretation is one short Japanese sentence. All other descriptive values are concise English. "
    "The characters object must have exactly the supplied character_id keys; these keys are identity, "
    "independent of who acts first or is on the left. NEVER return an image number or a 'Reference image' label. "
    "The application assigns all reference numbers. Match the Japanese names/dialogue to character_id keys. "
    "For each character, position describes ONLY that person's place in the frame; action describes ONLY "
    "that person's visible pose/gesture; expression describes ONLY that person's face. Write subjectless "
    "fragments, without character names, IDs, other people's actions, appearance changes or reference numbers. "
    "gaze_target is a supplied character_id for looking toward another person, camera for the viewer, "
    "scene for a prop or background object, or none. gaze_detail describes a prop or a nuance of the gaze; "
    "never name a person in it. Use gaze_target to identify another person, not translated names. "
    "environment, lighting and framing describe ONLY the place, light and camera, never assign a person's "
    "action, appearance or position. Preserve the local place and physical facts. Translate dialogue into "
    "visible behavior, never quote or render it. Choose one frozen moment, no montage, writing or speech bubbles. "
    "Each variant starts from the SAME base scene. Keep the base background, camera, character positions, "
    "clothing and body proportions. Change only its specified expressions, gazes and small gestures. "
    "Keep important faces above the lower message box. Do not invent extra people."
)


def _fragment(value, field, limit, character_ids):
    if not isinstance(value, str):
        raise TypeError(f"{field} must be English visual text.")
    text = value.strip()
    if (len(text) > limit or _JAPANESE.search(text) or _REFERENCE.search(text)
            or _clean_llm_content(text) != text or "```" in text
            or any(cid in text for cid in character_ids)):
        raise ValueError(f"{field} must be a concise English fragment without names, IDs or image numbers.")
    return text.rstrip(". ")


def compile_images(answer: dict, character_ids: list[str], image_ids: list[str]) -> list[dict]:
    """Reference subjects and cross-person gaze targets are rendered by code."""
    if not isinstance(answer, dict) or set(answer) != {"images"}:
        raise ValueError("Return exactly the structured images field.")
    images = answer["images"]
    if (not isinstance(images, list) or any(not isinstance(row, dict) for row in images)
            or [row.get("id") for row in images] != image_ids):
        raise ValueError("Provide the base and every variant in their exact supplied order.")
    compiled = []
    for index, row in enumerate(images):
        if set(row) != {"id", "interpretation", "environment", "lighting", "framing", "characters"}:
            raise ValueError("Each image requires structured scene fields and character-ID directions, not a prompt.")
        summary = row["interpretation"]
        if (not isinstance(summary, str) or not 5 <= len(summary.strip()) <= 600
                or not _JAPANESE.search(summary) or _clean_llm_content(summary) != summary):
            raise ValueError("Each image needs a short Japanese interpretation.")
        people = row["characters"]
        if not isinstance(people, dict) or set(people) != set(character_ids):
            raise ValueError("Character direction keys must exactly match the selected character IDs.")
        reference = {cid: number for number, cid in enumerate(character_ids, 2 if index else 1)}
        if index:
            parts = [("Edit Reference image 1, the complete base scene. Preserve its composition, background, "
                "camera, character positions, clothing and proportions. Change only the specified expressions, "
                "gazes and small gestures. The remaining reference images identify the individual characters.")]
        else:
            parts = ["Create one coherent full-scene illustration using the individual character references."]
        for field in ("environment", "lighting", "framing"):
            fragment = _fragment(row[field], field, 500 if field == "environment" else 240, character_ids)
            if fragment:
                parts.append(f"{field.capitalize()}: {fragment}.")
        for cid in character_ids:
            person = people[cid]
            if not isinstance(person, dict) or set(person) != {
                    "position", "action", "expression", "gaze_target", "gaze_detail"}:
                raise ValueError("Each character needs position, action, expression and structured gaze.")
            target = person["gaze_target"]
            if target not in {*character_ids, "scene", "camera", "none"} or target == cid:
                raise ValueError("Gaze targets must identify another supplied character, scene, camera or none.")
            fragments = {field: _fragment(person[field], field, 240, character_ids)
                for field in ("position", "action", "expression", "gaze_detail")}
            if not any(fragments[field] for field in ("position", "action", "expression")):
                raise ValueError("Every visible character requires a visual direction.")
            description = [f"Reference image {reference[cid]} character:"]
            for field in ("position", "action", "expression"):
                if fragments[field]:
                    description.append(f"{field} — {fragments[field]}.")
            if target in reference:
                description.append(f"Gaze directed toward the character from Reference image {reference[target]}.")
            elif target == "camera":
                description.append("Gaze directed toward the viewer.")
            elif target == "scene":
                description.append("Gaze directed toward a scene object.")
            if fragments["gaze_detail"]:
                description.append(f"Gaze detail: {fragments['gaze_detail']}.")
            parts.append(" ".join(description))
        parts.append("Preserve each referenced person's distinct face, hairstyle, hair and eye colors, "
            "clothing details, body proportions and illustration style. Do not swap or blend identities. "
            "Keep faces above the lower message area. No text, lettering, captions, speech bubbles, "
            "watermarks, extra people or montage.")
        prompt = "\n".join(parts)
        if len(prompt) > 10000:
            raise ValueError("Compiled CG visual directions exceed the prompt limit.")
        compiled.append({"id": row["id"], "interpretation": summary.strip(), "prompt": prompt})
    return compiled
