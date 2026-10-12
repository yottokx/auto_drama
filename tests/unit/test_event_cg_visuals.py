"""Image-number identity is code-owned, including base-CG offset and gaze."""
from __future__ import annotations

import copy

import pytest

from services.worker.generation import event_cg_visuals as visuals

LINA = "character-1"
IRIS = "character-9845b9c6-bb2e-4105-be81-4bc9cd973bd0"
IDS = [LINA, IRIS]
IMAGES = ["cg_001", "cg_001_v01"]


def directions():
    people = {
        # Intentionally put Iris first in the generated JSON. Array/object order
        # is not authority for the fixed character-reference manifest.
        IRIS: {"position": "standing on the right", "action": "holding a holographic terminal",
            "expression": "stern frustration", "gaze_target": LINA, "gaze_detail": "a direct accusing stare"},
        LINA: {"position": "seated in the pilot chair on the left", "action": "gripping the control stick",
            "expression": "cold anger", "gaze_target": IRIS, "gaze_detail": "narrowed eyes"},
    }
    return {"images": [{"id": identifier, "interpretation": "二人がコックピットで対立する瞬間。",
        "environment": "a worn cockpit on a spaceship", "lighting": "dim blue and orange light",
        "framing": "medium shot above the lower message box", "characters": copy.deepcopy(people)}
        for identifier in IMAGES]}


def test_character_keys_preserve_identity_despite_llm_order_and_shifted_variant_references():
    result = visuals.compile_images(directions(), IDS, IMAGES)
    base, variant = (image["prompt"] for image in result)
    assert "Reference image 1 character: position — seated in the pilot chair" in base
    assert "Reference image 2 character: position — standing on the right" in base
    assert "Reference image 2 character: position — seated in the pilot chair" in variant
    assert "Reference image 3 character: position — standing on the right" in variant
    assert "Reference image 1 character:" not in variant
    assert "Edit Reference image 1, the complete base scene." in variant
    pilot = next(line for line in variant.splitlines() if line.startswith("Reference image 2 character:"))
    officer = next(line for line in variant.splitlines() if line.startswith("Reference image 3 character:"))
    assert "toward the character from Reference image 3" in pilot
    assert "toward the character from Reference image 2" in officer
    assert LINA not in variant and IRIS not in variant
    assert "Lina" not in variant and "Iris" not in variant


def test_schema_requires_exact_ids_and_does_not_ask_model_for_reference_numbers():
    schema = visuals.schema(IDS, IMAGES)
    people = schema["properties"]["images"]["items"]["properties"]["characters"]
    assert people["required"] == IDS and people["additionalProperties"] is False
    assert "reference_index" not in str(schema)
    assert people["properties"][IRIS]["properties"]["gaze_target"]["enum"] == [*IDS, "scene", "camera", "none"]


@pytest.mark.parametrize("field,value", [
    ("action", "Reference image 2 (Iris) points at Reference image 3 (Lina)"),
    ("position", "next to image 3"),
    ("expression", "the face of " + LINA),
    ("gaze_target", "Lina"),
    ("gaze_target", IRIS),
])
def test_model_cannot_reassign_reference_numbers_or_use_freeform_gaze_identity(field, value):
    answer = directions()
    answer["images"][1]["characters"][IRIS][field] = value
    with pytest.raises(ValueError):
        visuals.compile_images(answer, IDS, IMAGES)


@pytest.mark.parametrize("kind", ["missing", "unknown", "legacy"])
def test_missing_foreign_or_old_freeform_identity_is_rejected(kind):
    answer = directions()
    if kind == "missing":
        del answer["images"][1]["characters"][LINA]
    elif kind == "unknown":
        answer["images"][1]["characters"]["other"] = answer["images"][1]["characters"].pop(LINA)
    else:
        answer["images"][1] = {"id": IMAGES[1], "interpretation": "混同した旧指示。",
            "prompt": "Reference image 2 (Iris) faces Reference image 3 (Lina)."}
    with pytest.raises(ValueError):
        visuals.compile_images(answer, IDS, IMAGES)



@pytest.mark.parametrize("row", [None, "invalid", 17, []])
def test_non_object_image_records_raise_repairable_validation_error(row):
    with pytest.raises(ValueError, match="exact supplied order"):
        visuals.compile_images({"images": [row]}, IDS, IMAGES)


def test_fixed_text_sets_one_drawn_style_and_scale_for_the_whole_picture():
    base, variant = (image["prompt"] for image in visuals.compile_images(directions(), IDS, IMAGES))
    head = base.splitlines()[0]
    assert head.startswith("Create one coherent full-scene illustration using the individual character references.")
    # Stated positively: the image model ignores "not a photograph".
    assert "hand-painted anime background art" in head and "sharp focus" in head
    assert "photograph" not in base
    assert "Environment, as hand-painted anime background art: a worn cockpit on a spaceship." in base
    assert "Lighting: dim blue and orange light." in base
    for prompt in (base, variant):
        assert "proportionally smaller, at one consistent scale" in prompt
    # A variant edits the finished base picture, so it does not restate the style.
    assert "Art style of the whole picture" not in variant.splitlines()[0]


@pytest.mark.parametrize("environment,crowd", [
    ("a worn cockpit on a spaceship", False),
    ("a quiet street with greenery", False),
    ("a station square, bustling with people in the background", True),
    ("station ticket gate area with crowds", True),
    ("a classroom after school with a few students", True),
])
def test_background_people_are_named_only_when_the_place_has_them(environment, crowd):
    """Mentioning background people, even conditionally, makes the image model add some."""
    answer = directions()
    for image in answer["images"]:
        image["environment"] = environment
    for image in visuals.compile_images(answer, IDS, IMAGES):
        tail = image["prompt"].splitlines()[-1]
        assert ("background people" in tail) is crowd
        assert tail.endswith("watermarks, further main characters or montage." if crowd
                             else "watermarks, extra people or montage.")


def test_writer_states_whom_a_held_object_is_turned_toward_not_what_the_viewer_sees():
    assert "whom its screen or pages are turned toward" in visuals.SYSTEM
    assert "Never state what the viewer of the picture sees of it" in visuals.SYSTEM
    assert "never add an object for it" in visuals.SYSTEM
    assert "without lens effects such as blur" in visuals.SYSTEM
    # Both instruction sets are versioned apart from every earlier one.
    assert len({visuals.VERSION, visuals.STAGING_VERSION, 2, 3}) == 4
