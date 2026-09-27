import pytest

from services.worker.generation import narrative
from services.worker.generation.causal_narrative import GateVerdict
from tests.unit.test_m3_narrative import FakeLLM


@pytest.mark.parametrize("system", [None, "原文の根拠を照合する検査担当です。"])
def test_structured_repair_retains_the_selected_inspector_role(system):
    response = GateVerdict(verdict="pass", checked_categories=["knowledge"], rationale="原文を照合した")
    llm = FakeLLM([{"content": "{}"}, {"content": response.model_dump_json()}])
    narrative._structured(llm, "fact-comparison-s1", "同じ原文を検査する", GateVerdict, system=system)
    assert len(llm.calls) == 2
    expected = narrative.SYSTEM if system is None else system
    for call in llm.calls:
        assert call[0][1][0] == {"role": "system", "content": expected}


@pytest.mark.parametrize("purpose,profile", [
    ("fact-comparison-s1-b1", "continuity_review"),
    ("state-comparison-s1", "continuity_review"),
    ("state-observation-s1", "story_extraction"),
])
def test_inspector_stages_respect_existing_purpose_profiles(purpose, profile):
    llm = FakeLLM([])
    llm.payload.update(story_workflow_version=2, profiles={profile: {"max_tokens": 2200}})
    narrative._purpose(llm, purpose)
    assert llm.profile["max_tokens"] == 2200
    assert llm.trace[-1]["profile_purpose"] == profile
