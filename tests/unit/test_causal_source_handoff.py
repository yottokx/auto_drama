"""The next scene must receive the preceding dialogue, including unextracted lines."""

import copy

from services.worker.generation import causal_narrative as causal
from tests.unit.test_causal_narrative import Harness


def test_source_handoff_crosses_scene_and_chapter_boundaries(monkeypatch):
    harness = Harness(monkeypatch)
    original = harness.write
    supplied = []

    def write(run, setting, memory, proposal, names, cast):
        supplied.append(copy.deepcopy(setting.get("preceding_scene")))
        return original(run, setting, memory, proposal, names, cast)

    monkeypatch.setattr(causal, "_write_scene", write)
    first = harness.chapter()
    second = harness.chapter(first)
    assert second.chapter_number == 2
    assert supplied[0] is None
    assert supplied[1]["chapter_number"] == 1
    assert supplied[1]["scene"]["raw_text"] == first.scenes[0].raw_text
    assert supplied[2]["chapter_number"] == 1
    assert supplied[2]["scene"]["raw_text"] == first.scenes[-1].raw_text
    for stage in ("chapter-intent", "scene-sequence", "scene-continuity-s1"):
        inputs = [value for name, value in harness.calls if name == stage][-1]
        actual = inputs["actual_memory"]
        assert actual["direct_preceding_scene"]["raw_text"] == first.scenes[-1].raw_text
        assert actual["direct_preceding_scene"]["scene_id"] == first.scenes[-1].id
        assert actual["source_references_to_preceding_scene"]
    adapted = next(value for name, value in harness.calls if name == "adapt-plan-s2")
    assert adapted["actual_memory"]["direct_preceding_scene"]["raw_text"] == first.scenes[0].raw_text
