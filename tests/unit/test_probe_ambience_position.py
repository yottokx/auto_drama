"""Where an ambience sound is heard from: the play rule, scoring, agreement and the report."""
from scripts.audio import probe_ambience_position as probe
from tests.unit.test_probe_music_ambience import FakeLLM


def chapters():
    keys = {(key, scene_id) for key, scene_id, *_ in probe.CASES}
    rows = {}
    for key, scene_id in sorted(keys):
        scene = {"scene_id": scene_id, "location": {"name": "部屋", "description": "", "time_of_day": "", "atmosphere": ""},
                 "lines": [{"id": f"{scene_id}-u1", "speaker": "地の文", "text": "雨音が響く。"}]}
        chapter = rows.setdefault(key, {"key": key, "title": "作品", "chapter": 1, "material": {"scenes": []}})
        chapter["material"]["scenes"].append(scene)
    return rows


def answer(position, shown, line="none"):
    return {"where": "雨は窓の外、聞き手は部屋の中。", "position": position, "shown": shown, "line": line}


def test_cases_use_known_values_and_their_plays_follow_the_rule():
    for _, _, _, positions, shown, plays in probe.CASES:
        assert set(positions) <= set(probe.POSITIONS) and set(shown) <= set(probe.SHOWN)
        assert probe.play(positions[0], shown[0]) == plays[0]


def test_play_rule():
    assert probe.play("same_space", "present") == "as_is"
    assert probe.play("same_space", "not_shown") == "none"
    assert probe.play("beyond_wall", "heard") == "muffled"
    assert probe.play("beyond_wall", "present") == "none"
    assert probe.play("inside_device", "heard") == "none"
    assert probe.play(None, None) is None


def test_classify_sends_the_scene_and_scores_each_judgment():
    rooms = chapters()
    llm = FakeLLM([answer("beyond_wall", "heard", "s1-u1"), answer("same_space", "heard")])
    right = probe.classify(llm, rooms, probe.CASES[1])
    loud = probe.classify(llm, rooms, probe.CASES[1])
    assert llm.stages == ["ambience-position"] * 2
    assert (right["position_ok"], right["shown_ok"], right["play_ok"], right["play"]) == (True, True, True, "muffled")
    assert (loud["position_ok"], loud["play_ok"], loud["play"]) == (False, False, "as_is")


def test_v2_asks_for_the_words_before_shown_and_heard_needs_them_in_the_line():
    assert list(probe.schema(["s1-u1"], "v2")["properties"]) == ["where", "position", "line", "sound_words", "shown"]
    assert "sound_words" not in probe.schema(["s1-u1"])["properties"]
    assert "sound_words" in probe.SYSTEM_V2 and probe.SYSTEM_V2.startswith(probe.SYSTEM.split(" Then choose")[0])

    def words(text, line="s1-u1"):
        return {**answer("beyond_wall", "heard", line), "sound_words": text}

    llm = FakeLLM([words("「雨音が響く」"), words("雨が降る"), words("", "s2-u1")])
    quoted = probe.classify(llm, chapters(), probe.CASES[1], "v2")
    missing = probe.classify(llm, chapters(), probe.CASES[1], "v2")
    empty = probe.classify(llm, chapters(), probe.CASES[6], "v2")
    assert (quoted["words_found"], quoted["shown"], quoted["play"]) == (True, "heard", "muffled")
    assert (missing["words_found"], missing["answered_shown"], missing["shown"], missing["play"]) == (
        False, "heard", "present", "none")
    assert (empty["shown"], empty["play"], empty["play_ok"]) == ("present", "none", True)


def test_a_failed_request_is_kept_as_a_miss():
    class Broken:
        def structured(self, stage, messages, schema):
            raise ValueError("ambience-position: Invalid JSON output.")

    row = probe.classify(Broken(), chapters(), probe.CASES[0])
    assert row["play"] is None and not row["play_ok"] and row["error"]


def test_summary_agreement_and_page():
    rooms = chapters()
    rows = [probe.judge(case, answer(case[3][0], case[4][0])) for case in probe.CASES]
    other = [dict(row) for row in rows]
    other[2] = probe.judge(probe.CASES[2], answer("beyond_wall", "heard"))
    summary = probe.summarize(other)
    assert summary["play"] == len(rows) - 1 and len(summary["misses"]) == 1
    assert probe.agreement({"a": rows, "b": other}) == {"same_answer": len(rows) - 1, "same_play": len(rows) - 1,
                                                        "cases": len(rows)}
    page = probe.page({"a": rows, "b": other}, rooms)
    assert page.count("<tr>") == len(rows) + 1 and 'class="ng"' in page
