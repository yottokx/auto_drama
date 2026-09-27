"""Publication keeps its source, staging, portraits and asset references unchanged."""

import hashlib

import pytest

from packages.contracts.m3 import NarrativeResult, PortraitSetting
from packages.narrative.script_conversion import narrative_to_script, stable_id
from packages.narrative.validation import script_character_id
from packages.tyrano_export import compile_bundle, demo_content
from packages.tyrano_export.compiler import canonical_json
from tests.unit.test_m3_narrative import narrative_fixture


def conversion_inputs(*, audio=True):
    value, snapshot = narrative_fixture()
    for row, name, height in zip(snapshot["characters"], ("勇者", "門番"), (174.0, 162.0), strict=True):
        row["id"] = row["result"]["id"]
        row["result"].update(name=name, body_type="humanoid", height_cm=height)
    value["scenes"][0]["utterances"][0].update(voice_emotion="calm", delivery="小声で")
    for kind, cid, position, duration in (("focus", "Hero", None, 0),
                                        ("position", "keeper", "center", 200),
                                        ("pause", None, None, 350),
                                        ("blackout", None, None, 400),
                                        ("exit", "keeper", None, 150)):
        value["scenes"][0]["directions"].append({
            "id": "s1-" + kind, "utterance_id": "s1-u4", "kind": kind,
            "timing": "after", "character_id": cid, "position": position,
            "duration_ms": duration})
    narrative = NarrativeResult.model_validate(value)
    _, images = demo_content()
    references, assets = {}, {}
    items = [("m3_image", "Hero", "character", images["aki_sprite"]),
             ("m3_image", "keeper", "character", images["ren_sprite"]),
             ("m3_background", "gate", "background", images["station"])]
    if audio:
        items.extend(("m3_voice_clone", utterance.id, "audio", b"saved-audio-" + utterance.id.encode())
                     for utterance in narrative.scenes[0].utterances if utterance.speaker_id)
    for requirement, target, kind, data in items:
        aid = stable_id(kind, target)
        references[requirement, target] = {
            "id": aid, "kind": kind, "artifact_id": "fixture-" + aid,
            "filename": aid + (".wav" if kind == "audio" else ".png"),
            "sha256": hashlib.sha256(data).hexdigest()}
        assets[aid] = data
    return narrative, snapshot, references, assets


def test_publication_conversion_matches_the_pre_extraction_output():
    narrative, snapshot, references, assets = conversion_inputs()
    script = narrative_to_script(narrative, snapshot, references, script_id="chapter-fixture")
    # Fingerprints recorded from the original M3Service._publish conversion.
    assert hashlib.sha256(canonical_json(script.model_dump(mode="json"))).hexdigest() == (
        "e9e35ce081fab8825ff395d5aadf844122ff9aa66c9cab1dc6999fe6b5fe25e1")
    assert hashlib.sha256(compile_bundle(script, assets)).hexdigest() == (
        "a02f1e95c78bc0a24686b9b9dbf3961c9dac0703fcbda52b4ec35a81e4e2f6c8")


def test_source_fields_and_all_existing_stage_directions_survive_conversion():
    narrative, snapshot, references, _ = conversion_inputs()
    script = narrative_to_script(narrative, snapshot, references, script_id="chapter-fixture")
    original = narrative.scenes[0].utterances
    assert [row.id for row in script.utterances] == [row.id for row in original]
    for actual, source in zip(script.utterances, original, strict=True):
        assert actual.speaker_id == (script_character_id(source.speaker_id) if source.speaker_id else None)
        assert actual.display_text == source.display_text
        assert actual.spoken_text == source.spoken_text
        assert actual.voice_emotion == source.voice_emotion
        assert actual.delivery == source.delivery
    # Automatic scene background/reset comes before the preserved authored directions.
    assert [row.kind for row in script.directions[:3]] == ["background", "exit", "exit"]
    authored = script.directions[3:]
    assert [row.id for row in authored] == [row.id for row in narrative.scenes[0].directions]
    assert {row.kind for row in authored} == {"enter", "exit", "focus", "position", "pause", "blackout"}
    assert all(row.audio_asset_id is not None for row in script.utterances if row.speaker_id)


def test_audio_omission_is_explicit_and_does_not_strip_other_fields():
    narrative, snapshot, references, _ = conversion_inputs(audio=False)
    with pytest.raises(KeyError):
        narrative_to_script(narrative, snapshot, references, script_id="chapter-fixture")
    script = narrative_to_script(narrative, snapshot, references, script_id="chapter-fixture",
                                 allow_missing_audio=True)
    assert all(row.audio_asset_id is None for row in script.utterances)
    assert script.utterances[0].spoken_text and script.utterances[0].delivery == "小声で"
    assert len(script.directions) == len(narrative.scenes[0].directions) + 3


@pytest.mark.parametrize("same_image", [False, True])
def test_portrait_body_bounds_remain_tied_to_the_chosen_image(same_image):
    narrative, snapshot, references, _ = conversion_inputs()
    image_id = references["m3_image", "Hero"]["artifact_id"]
    settings = {"Hero": PortraitSetting(
        character_id="Hero", image_artifact_id=image_id if same_image else "old-image",
        framing="upper_body", height_cm=180.0,
        body_bounds={"left": 0.1, "top": 0.1, "right": 0.9, "bottom": 0.9})}
    script = narrative_to_script(narrative, snapshot, references, script_id="chapter-fixture",
                                 portrait_settings=settings)
    hero = script.characters[0]
    assert hero.framing == "upper_body" and hero.height_cm == 180.0
    assert (hero.body_bounds is not None) == same_image
