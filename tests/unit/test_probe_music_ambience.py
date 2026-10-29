"""The BGM and ambience annotation probe: its inputs, answer shapes and the plan it resolves."""
import pytest

from scripts.story import probe_music_ambience as probe
from services.worker.generation.schemas import validate_schema


def payload():
    lines = {
        "s1": [("s1-u1", None, "雨が窓を叩いている。"), ("s1-u2", "character-1", "……濡れちゃった")],
        "s2": [("s2-u1", None, "部屋は静かだった。"), ("s2-u2", "side-1", "知ってたの？"),
               ("s2-u3", "character-1", "うん")],
        "s3": [("s3-u1", None, "翌朝。")],
    }
    scenes = [{"id": scene, "raw_text": "本文", "plan": {"location_id": f"loc-{scene}", "atmosphere": "静か"},
               "utterances": [{"id": line, "speaker_id": speaker, "display_text": text}
                              for line, speaker, text in rows]}
              for scene, rows in lines.items()]
    return {"approval_snapshot": {"characters": [{"id": "character-1", "result": {"name": "灯"}}]},
            "context": {"chapter_number": 1, "world": {"title": "雨の街", "genre": "日常", "mood": "穏やか"},
                        "overall_plot": {"chapters": [{"number": 1, "role": "出会い", "summary": "要約"}]},
                        "narrative": {"title": "第一章", "supporting_characters": [{"id": "side-1", "name": "凪"}],
                                      "locations": [{"id": f"loc-{scene}", "name": f"場所{scene}",
                                                     "description": "説明", "time_of_day": "夜",
                                                     "atmosphere": "雨"} for scene in lines],
                                      "scenes": scenes}}}


def chapter():
    return probe.material(payload())


def test_material_carries_every_line_once_with_names():
    value = chapter()
    first, second = value["scenes"][0], value["scenes"][1]
    assert [line["speaker"] for line in first["lines"]] == ["地の文", "灯"]
    assert second["lines"][1] == {"id": "s2-u2", "speaker": "凪", "text": "知ってたの？"}
    assert first["location"] == {"name": "場所s1", "description": "説明", "time_of_day": "夜", "atmosphere": "雨"}
    assert "raw_text" not in str(value)
    assert value["chapter"] == {"number": 1, "title": "第一章", "role": "出会い", "summary": "要約"}


def test_audible_answer_names_each_scene_and_cites_its_own_lines():
    schema = probe.audible_schema(chapter())
    scenes = schema["properties"]["scenes"]
    assert scenes["required"] == ["s1", "s2", "s3"]
    assert scenes["properties"]["s1"]["properties"]["evidence"]["enum"] == ["none", "location", "s1-u1", "s1-u2"]
    answer = {"scenes": {"s1": {"evidence": "s1-u1", "source": "weather", "sound": "窓を叩く雨"},
                         "s2": {"evidence": "none", "source": "none", "sound": ""},
                         "s3": {"evidence": "location", "source": "none", "sound": ""}}}
    validate_schema(answer, schema)
    with pytest.raises(ValueError):
        validate_schema({"scenes": {**answer["scenes"], "s2": {**answer["scenes"]["s2"], "evidence": "s1-u1"}}},
                        schema)


def test_v2_names_the_sounds_part_and_cites_only_a_line():
    schema = probe.audible_schema(chapter(), "v2")
    item = schema["properties"]["scenes"]["properties"]["s1"]
    assert list(item["properties"]) == ["source", "role", "evidence", "sound"]
    assert item["properties"]["role"]["enum"] == ["none", "noticed", "recurring", "carries"]
    assert item["properties"]["evidence"]["enum"] == ["none", "s1-u1", "s1-u2"]
    v1 = probe.audible_schema(chapter())["properties"]["scenes"]["properties"]["s1"]
    assert list(v1["properties"]) == ["evidence", "source", "sound"]
    sounds_ = probe.audible({"scenes": {
        "s1": {"source": "weather", "role": "noticed", "evidence": "s1-u1", "sound": "雨"},
        "s2": {"source": "crowd", "role": "none", "evidence": "s2-u1", "sound": "ざわめき"}}})
    assert (sounds_["s1"]["audible"], sounds_["s1"]["contradiction"]) == (True, False)
    assert (sounds_["s2"]["audible"], sounds_["s2"]["contradiction"]) == (False, True)


def test_v2_drops_name_their_turning_point_first():
    drop = probe.music_schema(chapter(), ASSETS, "v2")["properties"]["music_drops"]["items"]
    assert list(drop["properties"]) == ["kind", "start_before", "resume_before", "reason"]
    assert drop["properties"]["kind"]["enum"] == ["revelation", "loss", "decisive_choice", "silence_in_text"]
    assert "kind" not in probe.music_schema(chapter(), ASSETS)["properties"]["music_drops"]["items"]["properties"]
    v3 = probe.music_schema(chapter(), ASSETS, "v3")["properties"]["music_drops"]["items"]["properties"]["kind"]
    assert v3["enum"] == ["silence_in_text", "loss", "premise_overturned"]
    assert probe.audible_schema(chapter(), "v3") == probe.audible_schema(chapter(), "v2")


def test_a_sound_needs_both_a_source_and_evidence():
    sounds = probe.audible({"scenes": {
        "s1": {"evidence": "s1-u1", "source": "weather", "sound": "雨"},
        "s2": {"evidence": "none", "source": "crowd", "sound": "ざわめき"},
        "s3": {"evidence": "none", "source": "none", "sound": ""}}})
    assert [sounds[scene]["audible"] for scene in ("s1", "s2", "s3")] == [True, False, False]
    assert [sounds[scene]["contradiction"] for scene in ("s1", "s2", "s3")] == [False, True, False]


def sounds(**chosen):
    rows = {scene: {"evidence": "none", "source": "none", "sound": ""} for scene in ("s1", "s2", "s3")}
    rows.update(chosen)
    return probe.audible({"scenes": rows})


def test_only_scenes_with_candidates_of_their_kind_are_matched(monkeypatch):
    monkeypatch.setattr(probe, "LIBRARY", [{"id": "rain_light", "source": "weather", "description": "しとしと降る雨"}])
    heard = sounds(s1={"evidence": "s1-u1", "source": "weather", "sound": "窓を叩く雨"},
                   s2={"evidence": "location", "source": "fire", "sound": "暖炉"})
    source, schema = probe.match_request(chapter(), heard)
    assert [row["scene_id"] for row in source["scenes"]] == ["s1"]
    assert source["scenes"][0]["evidence"]["text"] == "雨が窓を叩いている。"
    assert schema["properties"]["scenes"]["properties"]["s1"]["properties"]["asset"]["enum"] == [
        "rain_light", "no_match"]
    assets = probe.matched(chapter(), heard, {"scenes": {"s1": {"asset": "rain_light"}}})
    assert assets == {"s1": {"asset": "rain_light", "status": "matched"},
                      "s2": {"asset": None, "status": "no_candidates"},
                      "s3": {"asset": None, "status": "none"}}
    assert probe.match_request(chapter(), sounds()) == (None, None)


ASSETS = {"s1": {"asset": "rain_light", "status": "matched"}, "s2": {"asset": None, "status": "none"},
          "s3": {"asset": None, "status": "no_match"}}


def scene(arrangement="bgm", new_track=True):
    return {"arrangement": arrangement, "new_track": new_track, "transition": "dissolve", "reason": "理由"}


def test_a_scene_without_a_matched_sound_can_only_have_music():
    schema = probe.music_schema(chapter(), ASSETS)
    scenes = schema["properties"]["scenes"]["properties"]
    assert scenes["s1"]["properties"]["arrangement"]["enum"] == ["bgm", "bgm_and_ambience", "ambience_only"]
    assert scenes["s2"]["properties"]["arrangement"]["enum"] == ["bgm"]
    assert scenes["s3"]["properties"]["arrangement"]["enum"] == ["bgm"]
    assert schema["properties"]["music_drops"]["maxItems"] == 3
    assert "scene_end" in schema["properties"]["music_drops"]["items"]["properties"]["resume_before"]["enum"]
    answer = {"scenes": {"s1": scene("ambience_only"), "s2": scene(), "s3": scene()}, "music_drops": []}
    validate_schema(answer, schema)
    with pytest.raises(ValueError):
        validate_schema({**answer, "scenes": {**answer["scenes"], "s2": scene("ambience_only")}}, schema)


def test_music_source_names_the_matched_sound():
    value = probe.music_source(chapter(), ASSETS)
    assert [row["environmental_sound"] for row in value["scenes"]] == ["しとしと降る雨", "none", "none"]


def test_tracks_follow_new_track_and_return_after_an_ambience_only_scene():
    plan = probe.resolve(chapter(), {**ASSETS, "s2": {"asset": "rain_light", "status": "matched"}}, {
        "scenes": {"s1": scene("bgm_and_ambience", new_track=False), "s2": scene("ambience_only"),
                   "s3": scene(new_track=False)}, "music_drops": []})
    rows = plan["scenes"]
    assert [row["track"] for row in rows] == [1, None, 1]
    assert [row["ambience"] for row in rows] == ["rain_light", "rain_light", None]
    assert not any(row["silent"] for row in rows)


def test_drops_take_their_scene_from_the_start_line_and_resume_inside_it():
    plan = probe.resolve(chapter(), {**ASSETS, "s1": {"asset": "rain_light", "status": "matched"}}, {
        "scenes": {"s1": scene("ambience_only"), "s2": scene(), "s3": scene()},
        "music_drops": [{"start_before": "s2-u2", "resume_before": "s2-u3", "reason": "告白"},
                        {"start_before": "s2-u3", "resume_before": "s2-u1", "reason": "戻る位置が前"},
                        {"start_before": "s1-u2", "resume_before": "s3-u1", "reason": "環境音だけ"}]})
    first, second, third = plan["drops"]
    assert (first["scene_id"], first["resume"], first["resume_moved"]) == ("s2", "s2-u3", False)
    assert (second["resume"], second["resume_moved"]) == ("scene_end", True)
    assert (third["scene_id"], third["resume"], third["no_music"]) == ("s1", "scene_end", True)


def result(drops, *, audible_s1=True, arrangement="bgm_and_ambience", key="k1"):
    heard = sounds(s1={"evidence": "s1-u1", "source": "weather", "sound": "雨"}) if audible_s1 else sounds()
    assets = {"s1": {"asset": "rain_light", "status": "matched"} if audible_s1 else {"asset": None, "status": "none"},
              "s2": {"asset": None, "status": "none"}, "s3": {"asset": None, "status": "none"}}
    answer = {"scenes": {"s1": scene(arrangement if audible_s1 else "bgm"), "s2": scene(), "s3": scene()},
              "music_drops": [{"start_before": line, "resume_before": "scene_end", "reason": "間"} for line in drops]}
    return {"key": key, "title": "雨の街", "material": chapter(), "audible": heard, "assets": assets,
            "plan": probe.resolve(chapter(), assets, answer)}


def test_summary_counts_sounds_drops_and_missing_library_sounds():
    missing = result([])
    missing["audible"]["s2"] = {"evidence": "location", "source": "fire", "sound": "暖炉", "audible": True,
                                "contradiction": False}
    missing["assets"]["s2"] = {"asset": None, "status": "no_match"}
    summary = probe.summarize([result(["s2-u2"]), missing, {"key": "k3", "error": "x", "step": "music"}])
    assert summary["silent_scenes"] == 0
    assert summary["audible_scenes"] == 3
    assert summary["sources"] == {"weather": 2, "fire": 1}
    assert summary["drops_per_chapter"] == {0: 1, 1: 1}
    assert summary["failures"] == {"music": 1}
    assert summary["missing_sounds"] == [{"chapter": "k1", "title": "雨の街", "scene_id": "s2", "source": "fire",
                                          "sound": "暖炉", "status": "no_match"}]


def test_agreement_counts_drops_within_two_lines():
    counts = probe.agreement([result(["s1-u1", "s1-u2"])], [result(["s2-u3"], audible_s1=False)])
    assert counts["scenes"] == 3
    assert counts["same_audible"] == 2
    assert counts["same_arrangement"] == 2
    # s1-u1 and s1-u2 are four and three lines before s2-u3: neither is near.
    assert (counts["drops"], counts["drops_other"], counts["drops_near"]) == (2, 1, 0)
    assert probe.agreement([result(["s1-u1"])], [result(["s2-u1"])])["drops_near"] == 1


class FakeLLM:
    def __init__(self, answers):
        self.answers, self.stages, self.trace = list(answers), [], []

    def structured(self, stage, messages, schema):
        self.stages.append(stage)
        self.trace.append({"type": "llm_generation", "usage": {"prompt_tokens": 10}})
        answer = self.answers.pop(0)
        validate_schema(answer, schema)
        return answer


def case():
    return {"key": "k1", "title": "雨の街", "chapter": 1, "job_id": "job", "payload": payload(), "current": None}


def test_annotate_runs_three_steps_and_skips_matching_without_a_sound():
    quiet = {"scenes": {scene_id: {"evidence": "none", "source": "none", "sound": ""} for scene_id in ("s1", "s2", "s3")}}
    music = {"scenes": {"s1": scene(), "s2": scene(new_track=False), "s3": scene()}, "music_drops": []}
    llm = FakeLLM([quiet, music])
    row = probe.annotate(llm, case())
    assert llm.stages == ["music-ambience-audible", "music-ambience-music"]
    assert [value["track"] for value in row["plan"]["scenes"]] == [1, 1, 2]
    assert row["usage"]["audible"] == {"prompt_tokens": 10}
    rainy = {"scenes": {**quiet["scenes"], "s1": {"evidence": "s1-u1", "source": "weather", "sound": "雨"}}}
    llm = FakeLLM([rainy, {"scenes": {"s1": {"asset": "rain_light"}}},
                   {**music, "scenes": {**music["scenes"], "s1": scene("ambience_only")}}])
    row = probe.annotate(llm, case())
    assert llm.stages == ["music-ambience-audible", "music-ambience-match", "music-ambience-music"]
    assert row["plan"]["scenes"][0] == {**row["plan"]["scenes"][0], "arrangement": "ambience_only",
                                        "ambience": "rain_light", "silent": False}


def test_annotate_v2_uses_its_own_instructions_and_counts_kinds():
    class Recording(FakeLLM):
        def structured(self, stage, messages, schema):
            self.systems = [*getattr(self, "systems", []), messages[0]["content"]]
            return super().structured(stage, messages, schema)

    quiet = {"scenes": {scene_id: {"source": "none", "role": "none", "evidence": "none", "sound": ""}
                        for scene_id in ("s1", "s2", "s3")}}
    music = {"scenes": {"s1": scene(), "s2": scene(), "s3": scene()},
             "music_drops": [{"kind": "revelation", "start_before": "s2-u2", "resume_before": "scene_end",
                              "reason": "秘密が明かされる"}]}
    llm = Recording([quiet, music])
    row = probe.annotate(llm, case(), "v2")
    assert llm.systems == [probe.AUDIBLE_SYSTEM_V2, probe.MUSIC_SYSTEM_V2]
    assert row["criteria"] == "v2"
    assert probe.summarize([{**row}])["drop_kinds"] == {"revelation": 1}


def test_a_failed_step_is_recorded_and_the_chapter_is_kept():
    llm = FakeLLM([{"scenes": {}}])
    row = probe.annotate(llm, case())
    assert row["step"] == "audible"
    assert "error" in row and "plan" not in row


def test_pages_render_from_results(tmp_path):
    rows = [{**result(["s2-u2"]), "chapter": 1, "current": {"scenes": [{"scene_id": "s1", "action": "stop"}]}}]
    summary = probe.report(tmp_path, {"a": rows, "b": rows})
    assert summary["current_stops"] == (1, 1)
    assert "BGMを止める" in (tmp_path / "k1.html").read_text(encoding="utf-8")
    assert "雨の街" in (tmp_path / "index.html").read_text(encoding="utf-8")
