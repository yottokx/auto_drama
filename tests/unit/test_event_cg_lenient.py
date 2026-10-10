"""The lenient CG planner fits an answer to the script instead of sending it back."""

from packages.contracts.event_cg import LENIENT_RULES_VERSION, EventCgPolicy
from services.worker.generation.event_cg_lenient import fit, schema, select


def payload(max_cgs=3, variants=0):
    def scene(identifier, count, cast, **extra):
        return {"id": identifier, "plan": {"character_ids": cast, **extra.pop("plan", {})},
                "utterances": [{"id": f"{identifier}-u{n}"} for n in range(1, count + 1)], **extra}

    return {"policy": EventCgPolicy(planning_version=LENIENT_RULES_VERSION, max_cgs=max_cgs,
                                    max_variants_per_cg=variants).model_dump(),
            "chapter_budget": max_cgs,
            "references": [{"character_id": "alice"}, {"character_id": "bob"}],
            "context": {"source_sha256": "a" * 64, "narrative": {"scenes": [
                scene("s1", 6, ["alice", "bob"]),
                scene("s2", 8, ["alice"], plan={"outfits": [{"character_id": "alice", "outfit_id": "default",
                                                              "change_to": "armor"}]},
                      directions=[{"kind": "outfit", "utterance_id": "s2-u3", "character_id": "alice",
                                   "outfit_id": "armor"}])]}}}


def cg(start, last, people=("alice",), **extra):
    return {"start_utterance_id": start, "last_utterance_id": last, "character_ids": list(people),
            "interpretation": "見せ場。", "composition": "二人が向かい合う。", **extra}


def test_short_or_misplaced_ends_are_kept_and_shaped_not_rejected():
    cgs, notes = fit({"cgs": [cg("s1-u2", "s1-u2", ("bob", "alice")), cg("s1-u5", "s2-u4")]}, payload(), 3)
    # A single line is allowed; an end in another scene means "to the end of this scene".
    assert [(row["start_utterance_id"], row["end_utterance_id"]) for row in cgs] == [("s1-u2", "s1-u3"), ("s1-u5", "s2-u1")]
    assert cgs[0]["character_ids"] == ["alice", "bob"] and notes == []


def test_overlap_and_change_of_look_are_trimmed():
    cgs, notes = fit({"cgs": [cg("s1-u1", "s1-u4"), cg("s1-u3", "s1-u6"), cg("s2-u2", "s2-u8")]}, payload(), 3)
    assert [(row["start_utterance_id"], row["end_utterance_id"]) for row in cgs] == [
        ("s1-u1", "s1-u5"), ("s1-u5", "s2-u1"), ("s2-u3", None)]
    assert [note["cg_id"] for note in notes] == ["cg_002", "cg_003"]


def test_unknown_people_fall_back_to_the_scene_cast_and_extra_cgs_are_dropped():
    cgs, _ = fit({"cgs": [cg("s2-u4", "s2-u8", ("bob",)), cg("s1-u1", "s1-u6")]}, payload(max_cgs=1), 1)
    assert [(row["start_utterance_id"], row["character_ids"]) for row in cgs] == [("s1-u1", ["alice"])]


def test_variants_outside_the_cg_are_dropped_silently():
    answer = {"cgs": [cg("s1-u1", "s1-u6", variants=[
        {"start_utterance_id": "s1-u1", "interpretation": "同時", "change": "微笑む"},
        {"start_utterance_id": "s1-u4", "interpretation": "後半", "change": "涙ぐむ"}])]}
    cgs, _ = fit(answer, payload(variants=2), 3)
    assert [(row["id"], row["start_utterance_id"], row["interpretation"]) for row in cgs[0]["variants"]] == [
        ("cg_001_v01", "s1-u4", "涙ぐむ")]


def test_planner_is_asked_once_and_told_about_looks_only_when_present():
    class Planner:
        def __init__(self):
            self.calls, self.trace = [], []

        def structured(self, stage, messages, wanted):
            self.calls.append((stage, messages[0]["content"], wanted))
            return {"cgs": [cg("s1-u1", "s1-u6")]}

    value = payload()
    plain = {"references": value["references"], "scenes": [
        {"id": "s1", "utterances": [{"id": f"s1-u{n}"} for n in range(1, 7)]}]}
    llm = Planner()
    cgs, _ = select(value, llm, "select-chapter", plain)
    assert len(llm.calls) == 1 and cgs[0]["end_utterance_id"] == "s2-u1"
    assert "alternate looks" not in llm.calls[0][1] and "variants" not in schema(3, 0, ["a"], ["alice"])["properties"]["cgs"]["items"]["properties"]
    select(value, llm, "select-chapter", {**plain, "scenes": [{**plain["scenes"][0], "looks_at_scene_start": {"alice": "鎧"}}]})
    assert "alternate looks" in llm.calls[1][1]
