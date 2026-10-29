"""Version 3 chapter music: music in every scene, silence only as moments inside one."""
from __future__ import annotations

import json

import pytest

from packages.contracts.music import MusicPlan
from services.worker.generation import music_pipeline as music
from tests.unit.test_music_worker import answer


def context():
    def lines(scene, count):
        return [{"id": f"{scene}-u{number}", "speaker_id": "hero" if number % 2 else None,
                 "display_text": f"{scene}の{number}行目"} for number in range(1, count + 1)]

    return {"approval_snapshot": {"world": {"result": {"title": "凪の島", "genre": "ドラマ"}},
                                  "characters": [{"id": "hero", "result": {"name": "海斗"}}]},
            "world": {"title": "凪の島", "genre": "ドラマ"}, "chapter_number": 1,
            "overall_plot": {"chapters": [{"number": 1, "role": "祖父の声が届く"}]},
            "narrative": {"title": "嵐の夜", "chapter_number": 1, "locations": [], "supporting_characters": [],
                          "scenes": [{"id": "s1", "plan": {"atmosphere": "不安"}, "raw_text": "", "utterances": lines("s1", 4)},
                                     {"id": "s2", "plan": {"atmosphere": "静寂"}, "raw_text": "", "utterances": lines("s2", 5)}]}}


class RecordingLLM:
    def __init__(self, answers):
        self.answers, self.calls = iter(answers), []

    def structured(self, stage, messages, schema):
        self.calls.append({"stage": stage, "messages": json.loads(json.dumps(messages)), "schema": schema})
        return next(self.answers)


def decisions(*actions, drops=()):
    return {"scenes": [{"scene_id": scene, "action": action, "transition": {"visual": "dissolve"},
                        "reason": "同じ流れを保ちます。"} for scene, action in zip(("s1", "s2"), actions, strict=True)],
            "music_drops": [{"kind": kind, "start_before": start, "resume_before": resume, "reason": "静けさを受け止める"}
                            for kind, start, resume in drops]}


def test_every_line_is_material_and_the_answer_offers_no_silent_scene():
    llm = RecordingLLM([decisions("play", "continue"), answer()])
    result = music.plan_music({"context": context(), "planning_version": 3}, llm)
    call = llm.calls[0]
    source = json.loads(call["messages"][1]["content"])
    assert [line["id"] for line in source["scenes_in_order"][1]["lines"]] == [f"s2-u{n}" for n in range(1, 6)]
    assert source["scenes_in_order"][0]["lines"][0] == {"id": "s1-u1", "speaker": "海斗", "text": "s1の1行目"}
    assert source["scenes_in_order"][0]["lines"][1]["speaker"] == "地の文"
    assert call["schema"]["properties"]["scenes"]["items"]["properties"]["action"]["enum"] == ["play", "continue"]
    drops = call["schema"]["properties"]["music_drops"]
    # No count reaches an external model as a target, and a scene's first line belongs to its entrance.
    assert "maxItems" not in drops and "minItems" not in drops
    starts = drops["items"]["properties"]["start_before"]["enum"]
    assert "s1-u1" not in starts and "s2-u1" not in starts and "s2-u2" in starts
    assert drops["items"]["properties"]["resume_before"]["enum"][-1] == "scene_end"
    assert result["planning_version"] == 3 and "music_drops" not in result
    assert [row.get("action", "play") for row in result["scenes"]] == ["play", "continue"]


def test_first_scene_continue_starts_the_track():
    llm = RecordingLLM([decisions("continue", "continue"), answer()])
    result = music.plan_music({"context": context(), "planning_version": 3}, llm)
    assert result["scenes"][0].get("action", "play") == "play" and result["scenes"][0]["prompt"]


def test_drops_resume_inside_their_scene_or_hold_to_its_end_and_overlaps_merge():
    llm = RecordingLLM([decisions("play", "continue", drops=[
        ("silence_in_text", "s2-u2", "s2-u4"), ("loss", "s2-u3", "s2-u5"),  # overlap: one stretch
        ("premise_overturned", "s1-u3", "s2-u3"),  # resume in another scene: to the scene's end
    ]), answer()])
    result = music.plan_music({"context": context(), "planning_version": 3}, llm)
    assert result["music_drops"] == [
        {"kind": "premise_overturned", "start_utterance_id": "s1-u3", "resume_utterance_id": None, "reason": "静けさを受け止める"},
        {"kind": "silence_in_text", "start_utterance_id": "s2-u2", "resume_utterance_id": "s2-u5", "reason": "静けさを受け止める"},
    ]
    MusicPlan.model_validate(result).check_drops([[f"s1-u{n}" for n in range(1, 5)], [f"s2-u{n}" for n in range(1, 6)]])


def test_a_resume_before_the_start_holds_to_the_scene_end():
    llm = RecordingLLM([decisions("play", "play", drops=[("loss", "s2-u4", "s2-u2")]), answer(), answer()])
    result = music.plan_music({"context": context(), "planning_version": 3}, llm)
    assert result["music_drops"][0]["resume_utterance_id"] is None


def test_version_two_keeps_its_old_answer():
    llm = RecordingLLM([{"scenes": decisions("play", "stop")["scenes"]}, answer()])
    result = music.plan_music({"context": context(), "planning_version": 2}, llm)
    assert "music_drops" not in llm.calls[0]["schema"]["properties"]
    assert result["scenes"][1]["action"] == "stop"


SCENES = [["s1-u1", "s1-u2", "s1-u3"], ["s2-u1", "s2-u2", "s2-u3", "s2-u4"]]


def plan(*drops, version=3):
    scenes = [{"scene_id": "s1", "prompt": "Genre: x.", "transition": {}, "reason": "理由"},
              {"scene_id": "s2", "action": "continue", "transition": {}, "reason": "理由"}]
    return MusicPlan.model_validate({"planning_version": version, "scenes": scenes, "music_drops": [
        {"kind": "loss", "start_utterance_id": start, "resume_utterance_id": resume} for start, resume in drops]})


def test_plan_contract_keeps_music_in_every_scene_and_drops_only_in_version_three():
    with pytest.raises(ValueError, match="keeps music in every scene"):
        MusicPlan.model_validate({"planning_version": 3, "scenes": [
            {"scene_id": "s1", "action": "stop", "transition": {}, "reason": "理由"}]})
    with pytest.raises(ValueError, match="version 3"):
        plan(("s1-u2", None), version=2)


@pytest.mark.parametrize("drops", [
    [("s1-u1", None)],  # a scene's first line
    [("s1-u2", "s2-u2")],  # resumes in another scene
    [("s1-u3", "s1-u2")],  # resumes before it starts
    [("s2-u2", None), ("s2-u3", None)],  # overlaps a drop held to the scene's end
    [("s2-u3", None), ("s2-u2", "s2-u3")],  # out of order
    [("missing", None)],
    [("s2-u2", "s2-u3"), ("s2-u3", None)],  # starts on the line the last one resumes at
])
def test_check_drops_rejects_what_the_player_cannot_run(drops):
    with pytest.raises(ValueError):
        plan(*drops).check_drops(SCENES)


def test_check_drops_accepts_separate_stretches():
    plan(("s1-u2", "s1-u3"), ("s2-u2", "s2-u3"), ("s2-u4", None)).check_drops(SCENES)
