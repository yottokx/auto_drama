from __future__ import annotations

import copy
import json
from contextlib import nullcontext

import pytest

from services.worker.generation import m3_pipeline
from services.worker.generation.llm import LocalLLM
from services.worker.generation.narrative import StructuredGenerationError, generate_narrative
from tests.unit.test_m3_narrative import FakeLLM, narrative_fixture, narrative_responses


@pytest.mark.parametrize("defect", [
    "unknown_character", "duplicate_character", "unknown_location", "unused_location",
    "duplicate_location", "duplicate_scene", "duplicate_event", "long_scene_id",
])
def test_plan_is_repaired_before_writing_any_prose(defect):
    result, snapshot = narrative_fixture()
    responses = narrative_responses(result)
    bad = json.loads(responses[2]["content"])
    scene = bad["scenes"][0]
    if defect == "unknown_character":
        scene["character_ids"][1] = "keeper-typo"
    elif defect == "duplicate_character":
        scene["character_ids"] = ["Hero", "Hero"]
    elif defect == "unknown_location":
        scene["location_id"] = "missing"
    elif defect == "unused_location":
        bad["locations"].append({**bad["locations"][0], "id": "unused"})
    elif defect == "duplicate_location":
        bad["locations"].append(copy.deepcopy(bad["locations"][0]))
    elif defect == "duplicate_scene":
        bad["scenes"].append(copy.deepcopy(scene))
    elif defect == "duplicate_event":
        scene["required_events"].append(copy.deepcopy(scene["required_events"][0]))
    else:
        scene["id"] = "s" * 51
    responses.insert(2, {"content": json.dumps(bad)})
    llm = FakeLLM(responses)

    generated = generate_narrative({"approval_snapshot": snapshot}, llm)

    assert generated["scenes"][0]["raw_text"] == result["scenes"][0]["raw_text"]
    assert [call[0][0] for call in llm.calls[:5]] == [
        "story_outline", "supporting_character", "scene_plan", "scene_plan", "scene-text-s1"]
    rejected = [item for item in llm.trace if item["type"] == "structured_rejected"]
    assert len(rejected) == 1 and rejected[0]["purpose"] == "scene_plan"
    assert rejected[0]["reason"] in llm.calls[3][0][1][-1]["content"]


def test_long_character_id_must_be_selected_exactly_including_supporting_cast():
    from tests.integration.test_m2 import CHARACTER

    result, snapshot = narrative_fixture()
    exact_id = "character-80715cfc-9cc9-4100-bd39-6cbfdc8c6928"
    result = json.loads(json.dumps(result).replace("keeper", exact_id))
    snapshot["characters"] = [{"result": {"id": "Hero"}}]
    support = {**copy.deepcopy(CHARACTER), "id": exact_id}
    responses = narrative_responses(result)
    responses[1] = {"content": json.dumps({"characters": [support]})}
    bad = json.loads(responses[2]["content"])
    bad["scenes"][0]["character_ids"][1] = "character-80715cfc-9cc9-4100-bd39-6cbfdc8cge8"
    responses.insert(2, {"content": json.dumps(bad)})
    llm = FakeLLM(responses)

    generated = generate_narrative({"approval_snapshot": snapshot}, llm)

    schema = llm.calls[2][1]["response_format"]["json_schema"]["schema"]
    assert schema["$defs"]["ScenePlan"]["properties"]["character_ids"]["items"]["enum"] == [
        "Hero", exact_id]
    assert generated["scenes"][0]["plan"]["character_ids"] == ["Hero", exact_id]


def test_failed_plan_job_retry_reuses_outline_but_resamples_plan(monkeypatch, tmp_path):
    """Exercise real disk caches and the job boundary, with only model HTTP mocked."""
    result, snapshot = narrative_fixture()
    good = narrative_responses(result)
    bad = json.loads(good[2]["content"])
    bad["scenes"][0]["character_ids"][1] = "keeper-typo"
    messages = iter([*good[:2], {"content": json.dumps(bad)}, {"content": json.dumps(bad)}])
    calls = []

    class CachedLLM(LocalLLM):
        def __enter__(self):
            self.output.mkdir(parents=True, exist_ok=True)
            # A previous stage's failure must not redirect this plan's retry.
            self.failure_request = 99
            return self

        def __exit__(self, *args):
            return False

        def request(self, path, request):
            calls.append(request)
            return {"choices": [{"finish_reason": "stop", "message": next(messages)}]}

    monkeypatch.setattr(m3_pipeline, "LocalLLM", CachedLLM)
    monkeypatch.setattr(m3_pipeline, "gpu_lock", lambda *args: nullcontext())
    job = {"kind": "m3_narrative", "payload": {
        "schema_version": 1, "seed": 123, "approval_snapshot": snapshot}}
    with pytest.raises(StructuredGenerationError, match="unknown characters"):
        m3_pipeline.generate_job(job, tmp_path)
    assert len(calls) == 4
    assert not (tmp_path / "result.zip").exists()
    llm_path = tmp_path / "llm"
    retry = json.loads((llm_path / "retry-state.json").read_text())
    assert retry["segments"] == [{"from_request": 3, "salt": 104729}]
    saved = {p.name: p.read_bytes() for p in llm_path.glob("0[12]-*.json")}

    calls.clear()
    messages = iter(good[2:])
    bundle = m3_pipeline.generate_job(job, tmp_path)
    assert len(calls) == 4  # New plan, source, staging, review; no outline/cast regeneration.
    assert calls[0]["response_format"]["json_schema"]["name"] == "scene_plan"
    assert calls[0]["seed"] == 123 + 3 + 104729
    assert {p.name: p.read_bytes() for p in llm_path.glob("0[12]-*.json")} == saved
    assert len(list(llm_path.glob("03-scene_plan-*.json"))) == 2
    assert m3_pipeline.generate_job(job, tmp_path) == bundle
    assert len(calls) == 4
