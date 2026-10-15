"""A voice heard through a phone or the like, or remembered: no portrait on stage, a framed bust instead."""

import copy
import hashlib
import io
import json
import zipfile

import pytest
from pydantic import ValidationError

from packages.contracts.m3 import GenerationEnvelope, PortraitFace, ScenePlan
from packages.contracts.script import RemoteVoice, Script
from packages.narrative import parse_scene_text, validate_narrative
from packages.tyrano_export import compile_bundle, compile_scenario, validate_bundle
from packages.tyrano_export.compiler import player_overlay, remote_frames
from packages.tyrano_export.demo import demo_content
from services.coordinator.m3_bundle import validate_bundle as validate_result
from services.worker import __main__ as worker_main
from services.worker import generation
from services.worker.generation import narrative, portrait_face
from tests.unit.test_m3_narrative import FakeLLM, narrative_fixture
from tests.unit.test_script_continuation_run import chapter_plan

PHONE = {"voice_source": "remote", "voice_channel": "phone"}


def test_a_remote_voice_names_its_channel_and_a_remembered_one_none():
    assert RemoteVoice(source="remote", channel="radio").model_dump() == {"source": "remote", "channel": "radio"}
    assert RemoteVoice(source="recollection", expression="sad").model_dump() == {
        "source": "recollection", "expression": "sad"}
    with pytest.raises(ValidationError):
        RemoteVoice(source="remote")
    with pytest.raises(ValidationError):
        RemoteVoice(source="recollection", channel="phone")
    script, _ = demo_content()
    data = script.model_dump(mode="json")
    data["utterances"][0]["remote"] = {"source": "remote", "channel": "phone"}
    with pytest.raises(ValidationError, match="only a speaker's line"):
        Script.model_validate(data)


def scene(raw):
    plan = ScenePlan.model_validate(chapter_plan("s1")["scenes"][0])
    return plan, parse_scene_text(raw, plan.id, set(plan.character_ids))


def answer(utterances, voices=None, directions=()):
    voices = voices or {}
    return {"emotions": [{"utterance_id": row.id, "inner_emotion": "落ち着き", "voice_emotion": "neutral",
                          "delivery": "", "expression": "smile" if row.speaker_id else "neutral",
                          **voices.get(row.id, {})} for row in utterances],
            "directions": list(directions)}


CALL = "aoi: 着いたよ。どこにいる？\nNARRATOR: 葵は電話を耳に当てた。\nren: 時計の下だよ。\naoi: 見えた。\n"
ARRIVAL = ("ren: もしもし、今どこ？\nNARRATOR: 蓮の声が電話から聞こえる。\naoi: 改札の前。\n"
           "NARRATOR: 背後から足音が近づいた。\nren: 見つけた。\n")


def test_staging_asks_every_line_where_its_voice_comes_from():
    plan, utterances = scene(CALL)
    reply = answer(utterances, {utterances[2].id: PHONE})
    llm = FakeLLM([{"content": json.dumps(reply, ensure_ascii=False)}])
    staged = narrative._staging(llm, "context", plan, utterances)
    ((_, messages), extra), = llm.calls
    emotion = extra["response_format"]["json_schema"]["schema"]["$defs"]["EmotionAnnotation"]
    assert {"voice_source", "voice_channel"} <= set(emotion["required"])
    assert narrative.VOICE_SOURCE_INSTRUCTION in messages[-1]["content"]
    assert [row.voice_source for row in staged.emotions] == ["present", "present", "remote", "present"]


def test_a_remote_answer_must_name_its_channel():
    plan, utterances = scene(CALL)
    bad = answer(utterances, {utterances[2].id: {"voice_source": "remote", "voice_channel": "none"}})
    good = answer(utterances, {utterances[2].id: PHONE})
    llm = FakeLLM([{"content": json.dumps(bad, ensure_ascii=False)},
                   {"content": json.dumps(good, ensure_ascii=False)}])
    narrative._staging(llm, "context", plan, utterances)
    # The first answer is rejected and asked again, as any malformed staging is.
    assert len(llm.calls) == 2 and "channel" in llm.calls[1][0][1][-1]["content"]


def test_someone_only_heard_never_comes_on_stage_nor_lights_up():
    plan, utterances = scene(CALL)
    remote = utterances[2].id
    staged = narrative.Staging.model_validate(answer(utterances, {remote: PHONE}, [
        {"utterance_id": remote, "kind": "focus", "character_id": "ren", "timing": "start", "duration_ms": 0}]))
    directions = narrative._directions(plan, utterances, staged)
    entrances = [(row.character_id, row.utterance_id, row.position) for row in directions if row.kind == "enter"]
    assert entrances == [("aoi", utterances[0].id, "center")]
    assert not [row for row in directions if row.kind == "focus"]
    # The caller's smile belongs to the frame; nothing on stage changes face for it.
    assert ("ren", remote) not in {(row.character_id, row.utterance_id) for row in directions
                                   if row.kind == "expression"}
    lines = [row.model_copy(update={"remote": narrative.remote_voice(row, emotion)})
             for row, emotion in zip(utterances, staged.emotions, strict=True)]
    assert lines[2].remote == RemoteVoice(source="remote", channel="phone", expression="smile")
    assert all(line.remote is None for index, line in enumerate(lines) if index != 2)


def test_someone_heard_first_comes_on_at_their_first_line_in_person():
    plan, utterances = scene(ARRIVAL)
    staged = narrative.Staging.model_validate(answer(utterances, {utterances[0].id: PHONE}))
    entrances = [(row.character_id, row.utterance_id, row.position, row.duration_ms)
                 for row in narrative._directions(plan, utterances, staged) if row.kind == "enter"]
    assert entrances == [("aoi", utterances[0].id, "left", 0),
                         ("ren", utterances[4].id, "right", narrative.ARRIVAL_MS)]


def test_narration_is_never_a_remote_voice():
    _, utterances = scene(CALL)
    emotion = narrative.EmotionAnnotation(utterance_id=utterances[1].id, inner_emotion="-", voice_emotion="neutral",
                                          delivery="", voice_source="remote", voice_channel="telepathy")
    assert narrative.remote_voice(utterances[1], emotion) is None


def test_an_answer_saved_before_remote_voices_reads_and_dumps_as_it_did():
    _, utterances = scene(CALL)
    old = {"emotions": [{"utterance_id": row.id, "inner_emotion": "落ち着き", "voice_emotion": "neutral",
                         "delivery": ""} for row in utterances], "directions": []}
    assert narrative.Staging.model_validate(old).model_dump() == old


def test_a_chapter_may_leave_off_stage_only_people_it_never_shows():
    value, snapshot = narrative_fixture()
    keeper_lines = {row["id"] for row in value["scenes"][0]["utterances"] if row["speaker_id"] == "keeper"}
    value["scenes"][0]["directions"] = [row for row in value["scenes"][0]["directions"]
                                        if row["character_id"] != "keeper"]
    with pytest.raises(ValueError, match="entrance"):
        validate_narrative(copy.deepcopy(value), snapshot)
    for row in value["scenes"][0]["utterances"]:
        if row["id"] in keeper_lines:
            row["remote"] = {"source": "remote", "channel": "radio"}
    assert validate_narrative(value, snapshot).scenes[0].utterances[2].remote.channel == "radio"


def remote_script(*, ren_on_stage=False, face=True, expression=None):
    script, assets = demo_content()
    data = script.model_dump(mode="json")
    if not ren_on_stage:
        data["directions"] = [row for row in data["directions"] if row.get("character_id") != "ren"]
    ren = next(row for row in data["characters"] if row["id"] == "ren")
    if face:
        ren["face"] = {"left": 8, "top": 4, "right": 20, "bottom": 16}
    if expression:
        picture = assets["ren_sprite"]
        data["assets"].append({"id": "ren_sad", "kind": "character", "artifact_id": "fixture",
                               "filename": "ren_sad.png", "sha256": hashlib.sha256(picture).hexdigest()})
        assets["ren_sad"] = picture
        ren["expressions"] = [{"expression": expression, "image_asset_id": "ren_sad"}]
    for row in data["utterances"]:
        if row["speaker_id"] == "ren":
            row["remote"] = {"source": "remote", "channel": "phone", **({"expression": expression} if expression else {})}
    return Script.model_validate(data), assets


def test_a_remote_line_lights_no_one_and_its_frame_is_cut_around_the_face():
    script, assets = remote_script()
    scenario = compile_scenario(script, assets)
    start = scenario.index("*utterance_line_002")
    assert '[chara_ptext name=""]' in scenario[start:scenario.index('[ad_say id="line_002"]')]
    assert '[chara_ptext name="ad_aki"]' in scenario[scenario.index("*utterance_line_003"):]
    frames = remote_frames(script, assets)["lines"]
    assert set(frames) == {"line_002", "line_004"}
    # From 0.45 face heights above the face to one below the chin, 3:4 around the face's center.
    assert frames["line_002"] == {"speaker": "ren", "source": "remote", "channel": "phone",
                                  "storage": "ren_sprite.png", "width": 28, "height": 42,
                                  "crop": [3, -1, 22, 29], "on_stage": False}


def test_the_frame_shows_the_lines_face_and_knows_who_is_on_stage():
    script, assets = remote_script(ren_on_stage=True, expression="sad")
    frame = remote_frames(script, assets)["lines"]["line_002"]
    assert frame["storage"] == "ren_sad.png" and frame["on_stage"] is True


def test_without_a_known_face_the_frame_is_cut_from_the_silhouette():
    script, assets = remote_script(face=False)
    _, _, width, height = remote_frames(script, assets)["lines"]["line_002"]["crop"]
    assert width > 0 and abs(width / height - 3 / 4) < 0.05


def test_only_a_chapter_with_remote_voices_carries_frames():
    script, assets = demo_content()
    assert "data/others/auto_drama_remote.json" not in player_overlay(script, assets, b"{}")
    script, assets = remote_script()
    bundle = compile_bundle(script, assets)
    with zipfile.ZipFile(io.BytesIO(bundle)) as archive:
        frames = json.loads(archive.read("data/others/auto_drama_remote.json"))
        assert "auto_drama_voice_effects.js" in archive.read("index.html").decode()
    assert set(frames["lines"]) == {"line_002", "line_004"}
    validate_bundle(bundle, script, assets)


def face_job(tmp_path, portrait):
    (tmp_path / "portrait.png").write_bytes(portrait)
    return {"id": "job", "kind": "m3_portrait_face",
            "payload": {"schema_version": 1, "seed": 1,
                        "portrait": {"artifact_id": "art", "sha256": hashlib.sha256(portrait).hexdigest()}}}


def test_finding_a_face_unloads_no_model_and_answers_in_the_m3_contract(tmp_path, monkeypatch):
    _, assets = demo_content()
    portrait = assets["ren_sprite"]

    def refuse(_kind):
        raise AssertionError("a session was prepared for a model-free job")

    from services.worker.generation import event_cg_session, music_session
    monkeypatch.setattr(music_session, "prepare_job", refuse)
    monkeypatch.setattr(event_cg_session, "prepare_job", refuse)
    monkeypatch.setattr(portrait_face, "detect", lambda path, config: [8, 4, 20, 16])
    monkeypatch.setattr(portrait_face.pipeline, "load_config", dict)
    bundle = generation.generate_job(face_job(tmp_path, portrait), tmp_path)
    envelope, _files = validate_result(bundle, "m3_portrait_face")
    assert PortraitFace.model_validate(envelope["result"]).face.model_dump() == {
        "left": 8, "top": 4, "right": 20, "bottom": 16}
    for prepare in ("prepare_job",):
        monkeypatch.setattr(worker_main, prepare, refuse)
    worker_main.prepare_media_job("m3_portrait_face")


def test_a_portrait_without_a_face_is_answered_with_none(tmp_path, monkeypatch):
    _, assets = demo_content()

    def missing(path, config):
        raise portrait_face.FaceNotFound("none")

    monkeypatch.setattr(portrait_face, "detect", missing)
    monkeypatch.setattr(portrait_face.pipeline, "load_config", dict)
    bundle = portrait_face.generate_job(face_job(tmp_path, assets["aki_sprite"]), tmp_path)
    envelope, _ = validate_result(bundle, "m3_portrait_face")
    assert envelope["result"]["face"] is None and envelope["result"]["width"] == 28


def test_a_face_outside_its_portrait_is_refused():
    with pytest.raises(ValidationError, match="outside"):
        PortraitFace(portrait_sha256="a" * 64, width=10, height=10,
                     face={"left": 1, "top": 1, "right": 11, "bottom": 5})
    with pytest.raises(ValidationError):
        GenerationEnvelope(schema_version=1, kind="m3_unknown", result={}, provenance={}, trace=[])
