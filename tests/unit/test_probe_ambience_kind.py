"""Telling an ambience sound's kind from its description: scoring, the two-way view and the report."""
from scripts.audio import probe_ambience_kind as probe
from tests.unit.test_probe_music_ambience import FakeLLM


def answer(kind):
    return {"over_a_minute": "雨が途切れず降り続く。", "kind": kind}


def test_cases_name_only_known_kinds_and_are_distinct():
    assert all(acceptable and set(acceptable) <= set(probe.KINDS) for _, acceptable in probe.CASES)
    assert len({description for description, _ in probe.CASES}) == len(probe.CASES)


def test_classify_scores_the_expected_an_acceptable_and_the_two_way_answer():
    llm = FakeLLM([answer("continuous"), answer("occasional"), answer("swell"), answer("ticks")])
    exact = probe.classify(llm, "しとしと降る雨", ("continuous",))
    other = probe.classify(llm, "朝の森の鳥のさえずり", ("continuous", "occasional"))
    lasting = probe.classify(llm, "しとしと降る雨", ("continuous",))
    wrong = probe.classify(llm, "車のクラクション", ("occasional",))
    assert llm.stages == ["ambience-kind"] * 4
    assert (exact["expected"], exact["accepted"], exact["lasting_ok"]) == (True, True, True)
    assert (other["expected"], other["accepted"]) == (False, True)
    assert (lasting["accepted"], lasting["lasting_ok"]) == (False, True)
    assert (wrong["accepted"], wrong["lasting_ok"]) == (False, False)


def test_a_failed_request_is_kept_as_a_miss():
    class Broken:
        def structured(self, stage, messages, schema):
            raise ValueError("ambience-kind: Invalid JSON output.")

    row = probe.classify(Broken(), "霧雨", ("continuous",))
    assert row["kind"] is None and not row["accepted"] and not row["lasting_ok"] and row["error"]


def test_summary_agreement_and_page():
    rows = [probe.judge(description, acceptable, answer(acceptable[0])) for description, acceptable in probe.CASES]
    other = [dict(row) for row in rows]
    other[0] = probe.judge(probe.CASES[0][0], probe.CASES[0][1], answer("swell"))
    summary = probe.summarize(other)
    assert summary["expected"] == summary["accepted"] == len(rows) - 1
    assert summary["lasting_ok"] == len(rows)
    assert summary["misses"] == ["しとしと降る雨: swell（可: continuous）"]
    assert probe.agreement({"a": rows, "b": other})["same_kind"] == len(rows) - 1
    page = probe.page({"a": rows, "b": other})
    assert page.count("<tr>") == len(rows) + 1 and 'class="ng"' in page


def test_v2_adds_only_the_rule_for_a_sound_that_keeps_sounding_underneath():
    assert probe.KIND_SYSTEM_V2.startswith(probe.KIND_SYSTEM)
    assert "keeps sounding underneath" in probe.KIND_SYSTEM_V2[len(probe.KIND_SYSTEM):]

    class Recording(FakeLLM):
        def structured(self, stage, messages, schema):
            self.system = messages[0]["content"]
            return super().structured(stage, messages, schema)

    llm = Recording([answer("continuous")])
    probe.classify(llm, "焚き火のはぜる音", ("continuous",), "v2")
    assert llm.system == probe.KIND_SYSTEM_V2
