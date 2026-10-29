"""Ambience as spans over a chapter's lines: where a sound begins and ends, and what plays in each scene."""
import pytest

from scripts.story import probe_music_ambience as base
from scripts.story import probe_music_ambience_spans as probe
from services.worker.generation.schemas import validate_schema
from tests.unit.test_probe_music_ambience import FakeLLM, payload


def chapter(shared=True):
    """s1 and s2 share a place when shared; s3 is elsewhere. Lines: s1-u1 s1-u2 | s2-u1 s2-u2 s2-u3 | s3-u1."""
    value = base.material(payload())
    if shared:
        value["scenes"][1]["location"] = dict(value["scenes"][0]["location"])
    return value


def span(evidence, begins="scene_start", ends="location_change", sound="雨", source="weather"):
    return {"source": source, "role": "carries", "sound": sound, "evidence": evidence,
            "begins_before": begins, "ends_before": ends}


def test_a_sound_already_there_begins_with_its_scene_however_late_it_is_mentioned():
    (rain,) = probe.resolve_spans(chapter(), {"spans": [span("s2-u3")]})
    # Mentioned on the third line of s2: it plays from s2's first line until the place changes at s3.
    assert (rain["start"], rain["end"], rain["onset"]) == (2, 5, False)
    (whole,) = probe.resolve_spans(chapter(), {"spans": [span("s1-u2")]})
    assert (whole["start"], whole["end"]) == (0, 5)
    (elsewhere,) = probe.resolve_spans(chapter(shared=False), {"spans": [span("s1-u2")]})
    assert elsewhere["end"] == 2


def test_only_a_line_that_tells_the_beginning_starts_a_sound_mid_scene():
    (rain,) = probe.resolve_spans(chapter(), {"spans": [span("s2-u3", begins="s2-u2", ends="s2-u3")]})
    assert (rain["start"], rain["end"], rain["onset"], rain["end_moved"]) == (3, 4, True, False)
    (backwards,) = probe.resolve_spans(chapter(), {"spans": [span("s2-u3", begins="s2-u2", ends="s1-u1")]})
    assert (backwards["end"], backwards["end_moved"]) == (5, True)
    (long,) = probe.resolve_spans(chapter(), {"spans": [span("s1-u1", ends="chapter_end")]})
    assert long["end"] == 6


def test_a_later_span_replaces_the_one_before_it():
    storm, rain = probe.resolve_spans(chapter(), {"spans": [
        span("s2-u1", begins="s2-u2", sound="小雨"), span("s1-u1", sound="嵐", ends="chapter_end")]})
    assert (storm["sound"], storm["start"], storm["end"], storm["cut"]) == ("嵐", 0, 3, True)
    assert (rain["sound"], rain["start"], rain["end"], rain["cut"]) == ("小雨", 3, 5, False)


def test_v5_asks_how_long_a_sound_lasts_and_how_close_its_match_is():
    assert probe.SPANS_SYSTEM_V5 != probe.SPANS_SYSTEM and "duration is throughout" in probe.SPANS_SYSTEM_V5
    item = probe.spans_schema(chapter(), "v5")["properties"]["spans"]["items"]["properties"]
    assert list(item)[:5] == ["source", "role", "sound", "duration", "evidence"]
    assert "duration" not in probe.spans_schema(chapter())["properties"]["spans"]["items"]["properties"]
    spans = probe.resolve_spans(chapter(), {"spans": [{**span("s1-u1"), "duration": "throughout"}]}, "v5")
    _, schema = probe.match_request(chapter(), spans, "v5")
    assert schema["properties"]["spans"]["properties"]["a1"]["required"] == ["asset", "fit"]


def test_v6_asks_whether_a_sound_goes_on_by_itself():
    assert "keeps_sounding when" in probe.SPANS_SYSTEM_V6 and "or moment when" not in probe.SPANS_SYSTEM_V6
    item = probe.spans_schema(chapter(), "v6")["properties"]["spans"]["items"]["properties"]
    assert item["duration"]["enum"] == ["keeps_sounding", "ends_with_its_action"]
    pen, rain = probe.resolve_spans(chapter(), {"spans": [
        {**span("s1-u1", sound="ペン", source="machine", ends="chapter_end"), "duration": "ends_with_its_action"},
        {**span("s2-u1", begins="s2-u1"), "duration": "keeps_sounding"}]}, "v6")
    assert (pen["excluded"], rain["excluded"]) == ("excluded_moment", None)
    summary = probe.summarize([result(probe.matched([pen, rain], {"spans": {"a2": {"asset": "rain_light",
                                                                                    "fit": "same"}}}))])
    assert summary["moment_but_lasting"] == 1


def test_v5_neither_plays_nor_lets_a_moment_or_a_crowd_cut_another_sound():
    spans = probe.resolve_spans(chapter(), {"spans": [
        {**span("s1-u1", ends="chapter_end"), "duration": "throughout"},
        {**span("s2-u2", begins="s2-u2", sound="叫び声", source="creatures"), "duration": "moment"},
        {**span("s3-u1", begins="s3-u1", sound="歓声", source="crowd"), "duration": "throughout"}]}, "v5")
    rain, cry, crowd = spans
    assert (rain["end"], rain["cut"]) == (6, False)
    assert (cry["excluded"], crowd["excluded"]) == ("excluded_moment", "excluded_source")
    source, _ = probe.match_request(chapter(), spans, "v5")
    assert [row["span_id"] for row in source["spans"]] == ["a1"]
    result = probe.matched(spans, {"spans": {"a1": {"asset": "rain_light", "fit": "same"}}})
    assert [row["status"] for row in result] == ["matched", "excluded_moment", "excluded_source"]
    assert probe.coverage(chapter(), result) == ["rain_light"] * 6


def test_v5_plays_only_a_match_to_the_same_sound():
    spans = probe.resolve_spans(chapter(), {"spans": [{**span("s1-u1"), "duration": "throughout"}]}, "v5")
    loose = probe.matched(spans, {"spans": {"a1": {"asset": "machinery", "fit": "similar"}}})
    assert (loose[0]["asset"], loose[0]["candidate"], loose[0]["status"]) == (None, "machinery", "loose_match")
    summary = probe.summarize([result(loose)])
    assert summary["loose_matches"][0]["candidate"] == "machinery"
    assert summary["lines_with_ambience"] == 0


def test_every_library_sound_is_a_candidate_for_every_span():
    spans = probe.resolve_spans(chapter(), {"spans": [span("s1-u1", source="water")]})
    source, schema = probe.match_request(chapter(), spans)
    assert [row["id"] for row in source["library"]] == [row["id"] for row in base.LIBRARY]
    assert source["spans"][0]["evidence"]["id"] == "s1-u1"
    enum = schema["properties"]["spans"]["properties"]["a1"]["properties"]["asset"]["enum"]
    assert "rain_light" in enum and enum[-1] == "no_match"
    assert probe.match_request(chapter(), []) == (None, None)
    assert probe.matched(spans, {"spans": {"a1": {"asset": "no_match"}}})[0]["status"] == "no_match"


def rainy(answer_spans, assets):
    spans = probe.resolve_spans(chapter(), {"spans": answer_spans})
    return probe.matched(spans, {"spans": {span_["id"]: {"asset": asset} for span_, asset in zip(
        sorted(spans, key=lambda value: value["id"]), assets, strict=True)}})


def test_only_a_scene_covered_from_first_to_last_line_may_drop_its_music():
    spans = rainy([span("s1-u1"), span("s2-u2", begins="s2-u2", ends="s2-u3", sound="雷雨")],
                  ["rain_light", "storm"])
    sounds = probe.scene_sounds(chapter(), spans)
    # The storm replaces the rain at s2-u2 and stops before s2-u3; the rain does not come back.
    assert [sound["covered"] for sound in sounds] == [True, False, False]
    assert sounds[1]["sounds"][0] == {"sound": "しとしと降る雨", "from": "scene_start", "to": "s2-u2"}
    schema = probe.music_schema(chapter(), sounds)
    scenes = schema["properties"]["scenes"]["properties"]
    assert scenes["s1"]["properties"]["arrangement"]["enum"] == ["bgm", "ambience_only"]
    assert scenes["s3"]["properties"]["arrangement"]["enum"] == ["bgm"]
    kind = schema["properties"]["music_drops"]["items"]["properties"]["kind"]
    assert kind["enum"] == list(base.DROP_KINDS_V3)
    answer = {"scenes": {"s1": scene("ambience_only"), "s2": scene(new_track=False), "s3": scene()},
              "music_drops": []}
    validate_schema(answer, schema)
    with pytest.raises(ValueError):
        validate_schema({**answer, "scenes": {**answer["scenes"], "s3": scene("ambience_only")}}, schema)
    plan = probe.resolve(chapter(), sounds, answer)
    assert [row["track"] for row in plan["scenes"]] == [None, 1, 2]
    assert [row["has_ambience"] for row in plan["scenes"]] == [True, True, False]
    assert not any(row["silent"] for row in plan["scenes"])


def scene(arrangement="bgm", new_track=True):
    return {"arrangement": arrangement, "new_track": new_track, "transition": "none", "reason": "理由"}


def result(spans, drops=(), key="k1"):
    sounds = probe.scene_sounds(chapter(), spans)
    answer = {"scenes": {"s1": scene(), "s2": scene(), "s3": scene()},
              "music_drops": [{"kind": "loss", "start_before": line, "resume_before": "scene_end", "reason": "別れ"}
                              for line in drops]}
    return {"key": key, "title": "雨の街", "chapter": 1, "material": chapter(), "spans": spans, "sounds": sounds,
            "plan": probe.resolve(chapter(), sounds, answer), "current": None}


def test_a_sound_that_stops_between_two_scenes_in_one_place_is_reported():
    stops = rainy([span("s1-u1", ends="s2-u1")], ["rain_light"])
    assert probe.boundary_changes(result(stops)) == [
        {"from": "s1", "to": "s2", "place": "場所s1", "before": "rain_light", "after": None}]
    assert probe.boundary_changes(result(rainy([span("s1-u1")], ["rain_light"]))) == []


def test_summary_and_agreement_count_lines_spans_and_drops():
    first = result(rainy([span("s1-u1"), span("s3-u1", source="crowd", sound="歓声")], ["rain_light", "no_match"]),
                   drops=["s2-u2"])
    second = result(rainy([span("s1-u1", ends="s2-u1")], ["rain_light"]), drops=["s2-u3"])
    summary = probe.summarize([first, {"key": "k2", "error": "x", "step": "spans"}])
    assert (summary["spans"], summary["status_matched"], summary["status_no_match"]) == (2, 1, 1)
    assert (summary["lines"], summary["lines_with_ambience"], summary["scenes_with_ambience"]) == (6, 5, 2)
    assert summary["failures"] == {"spans": 1}
    assert summary["missing_sounds"][0]["sound"] == "歓声"
    agree = probe.agreement([first], [second])
    # Rain on lines 0-4 against rain on lines 0-1: lines 0, 1 and the silent line 5 agree.
    assert (agree["lines"], agree["same_line_sound"]) == (6, 3)
    assert (agree["drops"], agree["drops_near"]) == (1, 1)


def test_annotate_runs_spans_match_and_music():
    music = {"scenes": {"s1": scene(), "s2": scene(new_track=False), "s3": scene()}, "music_drops": []}
    llm = FakeLLM([{"spans": [span("s1-u2")]}, {"spans": {"a1": {"asset": "rain_light"}}}, music])
    row = probe.annotate(llm, {"key": "k1", "title": "雨の街", "chapter": 1, "job_id": "j", "payload": payload(),
                               "current": None})
    assert llm.stages == ["music-ambience-spans", "music-ambience-match", "music-ambience-music"]
    # The fixture's places all differ, so the rain ends where s2 begins.
    assert (row["spans"][0]["start"], row["spans"][0]["end"], row["spans"][0]["asset"]) == (0, 2, "rain_light")
    assert [value["has_ambience"] for value in row["plan"]["scenes"]] == [True, False, False]
    llm = FakeLLM([{"spans": []}, music])
    assert probe.annotate(llm, {"key": "k1", "title": "t", "chapter": 1, "job_id": "j", "payload": payload(),
                                "current": None})["spans"] == []
    assert llm.stages == ["music-ambience-spans", "music-ambience-music"]


def test_pages_render_from_results(tmp_path):
    rows = [result(rainy([span("s2-u2", begins="s2-u2")], ["storm"]), drops=["s2-u3"])]
    summary = probe.report(tmp_path, {"a": rows, "b": rows})
    page = (tmp_path / "k1.html").read_text(encoding="utf-8")
    assert "本文が始まりを書いている" in page and "BGMを止める" in page
    assert summary["arms"]["a"]["onset_starts"] == 1
