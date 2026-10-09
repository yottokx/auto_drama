"""Version 2 planning tests use fake LLM replies and never load generation models."""
from __future__ import annotations

import copy
import json

import pytest

from packages.contracts.event_cg import image_spans
from services.worker.generation import event_cg_pipeline as cg
from services.worker.generation.cancellation import GenerationCancelled


class FakeLLM:
    def __init__(self, answers):
        self.answers, self.messages, self.trace = iter(answers), [], []

    def structured(self, stage, messages, schema):
        self.messages.append((stage, copy.deepcopy(messages), schema))
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return copy.deepcopy(answer)


def payload():
    return {"schema_version": 1, "seed": 1,
        "policy": {"planning_version": 2, "max_cgs": 2, "max_variants_per_cg": 3},
        "chapter_budget": 2, "references": [{"character_id": "alice", "name": "アリス", "outfit_id": "default"}],
        "context": {"source_sha256": "a" * 64, "narrative": {"scenes": [{"id": "scene1",
            "plan": {"character_ids": ["alice"]}, "utterances": [
                {"id": f"u{i}", "speaker_id": "alice", "display_text": f"台詞{i}"} for i in range(1, 31)]}]}}}


def staging(start, last, change=""):
    return {"visual_state": "本を手にしたまま窓際で相手の話を聞く。",
        "evidence_utterance_ids": [f"u{start}"], "safe_last_utterance_id": f"u{last}",
        "reason": "相手の話を聞く姿勢がこの区間で変わらないため。", "change": change}


def candidate(start=1, last=15, variants=(9,), safe_last=None):
    return {"scene_id": "scene1", "start_utterance_id": f"u{start}", "last_utterance_id": f"u{last}",
        "character_ids": ["alice"], "interpretation": "打ち明け話を受け止める会話のまとまり。",
        "composition": "窓際の椅子に座った人物を横から映し、本を手に持つ。",
        "end_reason": "会話が終わり本を閉じる直前まで。", "end_evidence_utterance_id": f"u{last}",
        "staging": staging(start, safe_last or last),
        "variants": [{"start_utterance_id": f"u{index}", "interpretation": "告白を受け入れ微笑む転機。",
            "staging": staging(index, last, "こわばった表情から受容の微笑みに変わる。")}
            for index in variants]}


def prompts(*ids):
    return {"images": [{"id": identifier, "interpretation": "窓際で本を開く人物。",
        "environment": "a quiet library at dusk", "lighting": "soft blue light from the window",
        "framing": "a medium shot", "characters": {"alice": {"position": "beside the window",
            "action": "holding an open book", "expression": "a restrained smile",
            "gaze_target": "scene", "gaze_detail": "at the open book"}}} for identifier in ids]}


def test_staged_selection_preserves_reasons_and_passes_separate_image_intervals():
    source = payload()
    original = copy.deepcopy(source)
    choice = candidate()
    llm = FakeLLM([{"cgs": [choice]}, prompts("cg_001", "cg_001_v01")])
    result = cg.plan_cgs(source, llm)
    assert source == original
    assert result["planning_version"] == 2
    assert len(llm.messages) == 2  # No new routine review or annotation request.
    selection_source = llm.messages[0][1][1]["content"]
    assert selection_source == json.dumps(json.loads(selection_source), ensure_ascii=False, separators=(",", ":"))
    selection_system = llm.messages[0][1][0]["content"]
    assert "Ground EVERY concrete pose, physical contact and visible prop in actual utterance text" in selection_system
    assert "NOT evidence that an action has already happened" in selection_system
    assert "stop at the FIRST contradictory action" in selection_system
    selected = result["cgs"][0]
    assert selected["interpretation"] == choice["interpretation"]
    assert selected["variants"][0]["interpretation"] == choice["variants"][0]["interpretation"]
    assert [span["utterance_count"] for span in image_spans(selected, source["context"]["narrative"])] == [8, 7]
    visual_source = json.loads(llm.messages[1][1][1]["content"])
    base, variant = visual_source["image_intervals"]
    assert [row["id"] for row in base["display_utterances"]] == [f"u{i}" for i in range(1, 9)]
    assert base["future_states_do_not_depict"][0]["id"] == "cg_001_v01"
    assert base["following_utterance_for_boundary_only"]["id"] == "u9"
    assert variant["evidence"][0]["id"] == "u9"
    assert variant["future_states_do_not_depict"] == []
    assert visual_source["common_composition"] == choice["composition"]
    assert "Never move a later embrace" in llm.messages[1][1][0]["content"]
    assert llm.trace[-1]["type"] == "event_cg_image_intervals"
    assert llm.trace[-2]["version"] == 3


def test_six_lines_split_three_ways_are_repaired_to_one_safe_image():
    source = payload()
    bad = {"cgs": [candidate(last=6, variants=(4, 6))]}
    llm = FakeLLM([bad, bad, prompts("cg_001")])
    result = cg.plan_cgs(source, llm)
    assert len(llm.messages) == 3
    correction = llm.messages[1][1][-1]["content"]
    assert "cg_001: 3 utterances" in correction
    assert "cg_001_v01: 2 utterances" in correction
    assert "cg_001_v02: 1 utterances" in correction
    assert len(result["cgs"]) == 1
    assert result["cgs"][0]["variants"] == []
    assert result["cgs"][0]["end_utterance_id"] == "u7"
    assert {note["variant_id"] for note in result["planning_notes"]} >= {"cg_001_v01", "cg_001_v02"}


def test_repair_can_fix_anchors_without_new_draw_calls():
    bad = {"cgs": [candidate(last=15, variants=(3, 5))]}
    fixed = {"cgs": [candidate(last=15, variants=(9,))]}
    llm = FakeLLM([bad, fixed, prompts("cg_001", "cg_001_v01")])
    result = cg.plan_cgs(payload(), llm)
    assert len(llm.messages) == 3
    assert result["planning_notes"] == []
    assert result["cgs"][0]["variants"][0]["start_utterance_id"] == "u9"


def test_one_repair_reports_end_boundary_and_all_future_evidence_together():
    choice = candidate()
    choice["end_evidence_utterance_id"] = "u13"
    choice["staging"]["evidence_utterance_ids"] = ["u1", "u3"]
    choice["variants"][0]["staging"]["evidence_utterance_ids"] = ["u9", "u11"]
    fixed = candidate(start=3, variants=(11,))
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": [fixed]}, prompts("cg_001", "cg_001_v01")])
    result = cg.plan_cgs(payload(), llm)
    correction = llm.messages[1][1][-1]["content"]
    assert "end_evidence_utterance_id=u13 is invalid; allowed IDs are u15 through u30 within this scene" in correction
    assert "cg_001: evidence_utterance_id=u3 is AFTER this image's start_utterance_id=u1" in correction
    assert "cg_001_v01: evidence_utterance_id=u11 is AFTER this image's start_utterance_id=u9" in correction
    assert "move the image start to u11 or later" in correction
    assert "Do not merely delete or substitute the evidence" in correction
    assert "Correct EVERY listed boundary/evidence error" in correction
    assert len(result["cgs"]) == 1
    assert result["cgs"][0]["variants"][0]["start_utterance_id"] == "u11"
    assert len(llm.messages) == 3


def test_boundary_diagnostics_include_safe_range_and_real_chapter_end():
    choice = candidate(last=30, variants=(20,), safe_last=15)
    choice["end_evidence_utterance_id"] = "u28"
    choice["variants"][0]["staging"]["evidence_utterance_ids"] = ["u20", "missing", "u25"]
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": []}])
    result = cg.plan_cgs(payload(), llm)
    correction = llm.messages[1][1][-1]["content"]
    assert "allowed IDs are u30." in correction
    assert "safe_last_utterance_id=u15 must be at or after its assigned last utterance u19" in correction
    assert "at or before CG last utterance u30" in correction
    assert "evidence_utterance_id=missing does not exist" in correction
    assert "evidence_utterance_id=u25 is AFTER" in correction
    assert result["cgs"] == []


def test_future_evidence_diagnosis_measures_duration_after_latest_required_state():
    choice = candidate(start=5, last=13, variants=())
    choice["staging"]["evidence_utterance_ids"] = ["u6", "u10"]
    fixed = candidate(start=10, last=20, variants=())
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": [fixed]}, prompts("cg_001")])
    result = cg.plan_cgs(payload(), llm)
    correction = llm.messages[1][1][-1]["content"]
    assert "moving start to the latest supporting utterance u10 leaves 4 utterances" in correction
    assert "before the next image/end boundary u14" in correction
    assert "below the minimum of 5; moving only the start does not repair the plan" in correction
    assert "SAME visible state stays true for at least 5 utterances, or choose another CG" in correction
    assert "Inner monologue" in llm.messages[0][1][0]["content"]
    assert result["cgs"][0]["start_utterance_id"] == "u10"
    assert result["cgs"][0]["end_utterance_id"] == "u21"
    assert len(llm.messages) == 3


def test_bad_candidate_does_not_remove_a_valid_other_candidate():
    bad = candidate(start=20, last=25, variants=())
    bad["character_ids"] = ["intruder"]
    answer = {"cgs": [candidate(), bad]}
    llm = FakeLLM([answer, answer, prompts("cg_001", "cg_001_v01")])
    result = cg.plan_cgs(payload(), llm)
    assert len(result["cgs"]) == 1
    assert result["omission_reason"] == ""
    assert any(note["cg_id"] == "cg_002" for note in result["planning_notes"])


def test_malformed_repair_retains_initial_valid_candidate():
    answer = {"cgs": [candidate(), {"scene_id": "bad"}]}
    llm = FakeLLM([answer, {"invalid": True}, prompts("cg_001", "cg_001_v01")])
    result = cg.plan_cgs(payload(), llm)
    assert len(result["cgs"]) == 1
    assert any("初回" in note["reason"] for note in result["planning_notes"])


def test_empty_repair_does_not_discard_a_different_valid_candidate():
    answer = {"cgs": [candidate(), {"scene_id": "bad"}]}
    llm = FakeLLM([answer, {"cgs": []}, prompts("cg_001", "cg_001_v01")])
    result = cg.plan_cgs(payload(), llm)
    assert len(result["cgs"]) == 1


def test_empty_repair_after_invalid_candidates_records_why_zero_remain():
    choice = candidate(variants=())
    choice["end_evidence_utterance_id"] = "u2"
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": []}])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"] == []
    assert "end_evidence_utterance_id=u2" in result["omission_reason"]
    assert result["planning_notes"]
    assert len(llm.messages) == 2


def test_repair_keeps_a_missing_valid_candidate_beside_a_new_nonoverlapping_one():
    answer = {"cgs": [candidate(), {"scene_id": "bad"}]}
    repair = {"cgs": [candidate(start=20, last=29, variants=())]}
    llm = FakeLLM([answer, repair, prompts("cg_001"), prompts("cg_002", "cg_002_v01")])
    result = cg.plan_cgs(payload(), llm)
    assert len(result["cgs"]) == 2
    assert {item["start_utterance_id"] for item in result["cgs"]} == {"u1", "u20"}


def test_zero_variant_allowance_can_select_a_long_single_image():
    source = payload()
    source["policy"]["max_variants_per_cg"] = 0
    llm = FakeLLM([{"cgs": [candidate(last=20, variants=())]}, prompts("cg_001")])
    result = cg.plan_cgs(source, llm)
    assert result["cgs"][0]["variants"] == []
    assert llm.messages[0][2]["properties"]["cgs"]["items"]["properties"]["variants"]["maxItems"] == 0


def test_short_variant_gap_ends_at_safe_boundary_without_resuming_later():
    choice = candidate(last=20, variants=(6, 8), safe_last=5)
    choice["variants"][0]["staging"]["safe_last_utterance_id"] = "u7"
    answer = {"cgs": [choice]}
    llm = FakeLLM([answer, answer, prompts("cg_001")])
    result = cg.plan_cgs(payload(), llm)
    selected = result["cgs"][0]
    assert selected["variants"] == []
    assert selected["end_utterance_id"] == "u6"
    assert selected["end_evidence_utterance_id"] == "u20"
    assert selected["end_reason"] == choice["end_reason"]
    assert selected["staging"]["safe_end_utterance_id"] == "u6"
    assert any("短縮" in note["reason"] for note in result["planning_notes"])


def test_short_base_is_not_extended_beyond_its_safe_boundary():
    answer = {"cgs": [candidate(last=15, variants=(4,), safe_last=3)]}
    llm = FakeLLM([answer, answer])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"] == [] and result["omission_reason"]
    assert len(llm.messages) == 2


def test_future_evidence_delays_base_instead_of_showing_earlier_state():
    choice = candidate(variants=())
    choice["staging"]["evidence_utterance_ids"] = ["u7"]
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": [choice]}, prompts("cg_001")])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"][0]["start_utterance_id"] == "u7"
    assert result["cgs"][0]["staging"]["evidence_utterance_ids"] == ["u7"]
    assert result["planning_notes"]


def test_recovery_trims_base_to_evidence_and_clamps_safe_end_without_new_claims():
    source = payload()
    source["context"]["narrative"]["scenes"][0]["utterances"].extend(
        {"id": f"u{i}", "speaker_id": "alice", "display_text": f"台詞{i}"} for i in range(31, 36))
    choice = candidate(start=23, last=32, variants=(), safe_last=33)
    choice["staging"]["evidence_utterance_ids"] = ["u24"]
    choice["end_evidence_utterance_id"] = "u34"
    original = copy.deepcopy(choice)
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": [choice]}, prompts("cg_001")])
    result = cg.plan_cgs(source, llm)
    selected = result["cgs"][0]
    assert choice == original
    assert selected["start_utterance_id"] == "u24"
    assert selected["end_utterance_id"] == "u33"
    assert selected["staging"]["safe_end_utterance_id"] == "u33"
    assert selected["staging"]["evidence_utterance_ids"] == ["u24"]
    assert selected["end_evidence_utterance_id"] == "u34"
    assert selected["end_reason"] == choice["end_reason"]
    assert image_spans(selected, source["context"]["narrative"])[0]["utterance_count"] == 9
    assert any("u23 から u24" in note["reason"] for note in result["planning_notes"])
    assert any("u34 から u33" in note["reason"] for note in result["planning_notes"])
    assert len(llm.messages) == 3


def test_trim_and_clamp_do_not_salvage_less_than_five_utterances():
    choice = candidate(start=23, last=27, variants=(), safe_last=29)
    choice["staging"]["evidence_utterance_ids"] = ["u24"]
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": [choice]}])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"] == [] and result["omission_reason"]
    assert len(llm.messages) == 2


@pytest.mark.parametrize("evidence", ["u1b", "missing"])
def test_base_trim_never_uses_another_scene_or_unknown_evidence(evidence):
    source = payload()
    source["context"]["narrative"]["scenes"].append({"id": "scene2", "plan": {"character_ids": ["alice"]},
        "utterances": [{"id": "u1b", "speaker_id": "alice", "display_text": "次の場面"}]})
    choice = candidate(variants=(), safe_last=20)
    choice["staging"]["evidence_utterance_ids"] = ["u1", evidence]
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": [choice]}])
    assert cg.plan_cgs(source, llm)["cgs"] == []


def test_clamping_overlong_safe_end_keeps_an_already_valid_start():
    choice = candidate(start=5, last=15, variants=(), safe_last=20)
    choice["staging"]["evidence_utterance_ids"] = ["u2", "u5"]
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": [choice]}, prompts("cg_001")])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"][0]["start_utterance_id"] == "u5"
    assert result["cgs"][0]["staging"]["safe_end_utterance_id"] == "u16"
    assert not any("表示開始" in note["reason"].split("診断:")[0] for note in result["planning_notes"])


def test_recovery_drops_a_variant_with_future_evidence_without_moving_its_start():
    choice = candidate(last=15, variants=(6,), safe_last=5)
    choice["variants"][0]["staging"]["evidence_utterance_ids"] = ["u7"]
    llm = FakeLLM([{"cgs": [choice]}, {"cgs": [choice]}, prompts("cg_001")])
    result = cg.plan_cgs(payload(), llm)
    selected = result["cgs"][0]
    assert selected["variants"] == []
    assert selected["start_utterance_id"] == "u1"
    assert selected["end_utterance_id"] == "u6"


@pytest.mark.parametrize("last", [5, 30])
def test_single_image_can_be_shorter_than_target_or_reach_chapter_end(last):
    answer = {"cgs": [candidate(last=last, variants=())]}
    llm = FakeLLM([answer, prompts("cg_001")])
    result = cg.plan_cgs(payload(), llm)
    assert len(result["cgs"]) == 1
    assert result["cgs"][0]["end_utterance_id"] == (None if last == 30 else "u6")


def test_scene_fallback_retains_rules_and_separate_candidate_ids():
    source = payload()
    scene2 = copy.deepcopy(source["context"]["narrative"]["scenes"][0])
    scene2["id"] = "scene2"
    for row in scene2["utterances"]:
        row["id"] += "b"
    source["context"]["narrative"]["scenes"].append(scene2)
    second = candidate(variants=())
    second["scene_id"] = "scene2"
    for key in ("start_utterance_id", "last_utterance_id", "end_evidence_utterance_id"):
        second[key] += "b"
    second["staging"]["safe_last_utterance_id"] += "b"
    second["staging"]["evidence_utterance_ids"] = ["u1b"]
    llm = FakeLLM([cg.ContextBudgetError("too long"), {"cgs": [candidate(variants=())]}, {"cgs": [second]},
        prompts("select_scene_1_cg_001"), prompts("select_scene_2_cg_001")])
    result = cg.plan_cgs(source, llm)
    assert len(result["cgs"]) == 2
    assert len(llm.messages) == 5
    for message in llm.messages[1:3]:
        assert "12-30" in message[1][0]["content"]
        assert json.loads(message[1][1]["content"])["minimum_utterances_per_image"] == 5


def test_invalid_chapter_choice_keeps_valid_scene_candidates_within_allowance(monkeypatch):
    source = payload()
    source["policy"]["max_cgs"] = 1
    source["chapter_budget"] = 1
    second_scene = copy.deepcopy(source["context"]["narrative"]["scenes"][0])
    second_scene["id"] = "scene2"
    source["context"]["narrative"]["scenes"].append(second_scene)
    first = cg.event_cg_staging.normalize(candidate(variants=()), 1, source)
    second = copy.deepcopy(first)
    second["id"] = "cg_002"

    def select(_payload, _llm, scenes, stage, _notes):
        if stage == "select-chapter":
            raise cg.ContextBudgetError("long chapter")
        return [first if scenes[0]["id"] == "scene1" else second]

    monkeypatch.setattr(cg, "_select", select)
    notes = []
    llm = FakeLLM([{"selected_ids": ["missing"]}, {"selected_ids": ["missing"]}])
    selected = cg._select_with_context_limit(source, llm, notes)
    assert [item["id"] for item in selected] == ["cg_001"]
    assert len(llm.messages) == 2
    assert notes[0]["cg_id"] == "cg_001"


def test_context_overflow_in_one_scene_keeps_another_scenes_valid_candidate():
    source = payload()
    second_scene = copy.deepcopy(source["context"]["narrative"]["scenes"][0])
    second_scene["id"] = "scene2"
    for row in second_scene["utterances"]:
        row["id"] += "b"
    source["context"]["narrative"]["scenes"].append(second_scene)
    llm = FakeLLM([cg.ContextBudgetError("chapter too long"), {"cgs": [candidate(variants=())]},
        cg.ContextBudgetError("scene2 too long"), prompts("select_scene_1_cg_001")])
    result = cg.plan_cgs(source, llm)
    assert len(result["cgs"]) == 1
    assert result["cgs"][0]["id"] == "select_scene_1_cg_001"
    assert result["omission_reason"] == ""
    assert len(llm.messages) == 4
    assert any("scene2 too long" in note["reason"] for note in result["planning_notes"])


def test_zero_budget_and_optional_zero_selection_keep_new_version():
    source = payload()
    source["chapter_budget"] = 0
    result = cg.plan_cgs(source, None)
    assert result["cgs"] == [] and result["planning_version"] == 2
    llm = FakeLLM([{"cgs": []}])
    result = cg.plan_cgs(payload(), llm)
    assert result["cgs"] == [] and result["omission_reason"] == ""
    assert len(llm.messages) == 1


@pytest.mark.parametrize("error", [RuntimeError("transport failed"), GenerationCancelled()])
def test_transport_and_cancel_errors_are_not_converted_to_selection_omissions(error):
    with pytest.raises(type(error)):
        cg.plan_cgs(payload(), FakeLLM([error]))
