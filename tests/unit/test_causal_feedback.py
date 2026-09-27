"""A repair must see the original evidence that led to the rejection."""

import json

import pytest

from services.worker.generation import causal_narrative as causal
from services.worker.generation.causal_runtime import CausalRun, encoded
from tests.unit.test_causal_narrative import evidence_memory, verdict
from tests.unit.test_m3_narrative import FakeLLM


def test_retrieved_original_and_concrete_issue_are_preserved_for_repair():
    memory = evidence_memory()
    failure = {"content": json.dumps({
        "verdict": "fail", "checked_categories": ["knowledge"],
        "rationale": "以前の発言と一致しない", "missing_information": [],
        "issues": [{"code": "repeated-discovery", "description": "協力は既に合意している。",
                    "repair_scope": "scene", "evidence_ids": []}]}, ensure_ascii=False)}
    llm = FakeLLM([verdict(event_ids=["c1-cooperation"]), failure])
    with (CausalRun(llm, {"storyline_id": "story", "chapter_number": 2}) as run,
          pytest.raises(causal.GateError) as rejected):
        causal._gate(run, "knowledge-review", {}, "原文を照合する。", ["knowledge"],
                     subject={}, memory=memory)
    feedback = rejected.value.feedback()
    assert feedback["review"]["issues"][0]["description"] == "協力は既に合意している。"
    assert len(feedback["retrieved_evidence"]) == 1
    supplement = feedback["retrieved_evidence"][0]
    assert supplement["status"] == "ready"
    assert supplement["events"][0]["id"] == "c1-cooperation"
    assert {quote["text"] for quote in supplement["source_quotes"]} == {source.text for source in memory.sources}
    assert memory.sources[0].text in encoded(feedback)
