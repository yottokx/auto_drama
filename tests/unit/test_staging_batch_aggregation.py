"""Batch annotation limits must not reject a valid assembled scene."""

import copy
import json

import pytest

from packages.contracts.m3 import ScenePlan
from packages.narrative import parse_scene_text
from services.worker.generation import narrative
from services.worker.generation.draft_story import DraftExecutionError
from tests.unit.test_m3_narrative import FakeLLM
from tests.unit.test_script_continuation_run import FIRST, chapter_plan, outline, read_json
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import generate, job
from tests.unit.test_script_production import runtime as _runtime

runtime = _runtime
app_runtime = _app_runtime


def scene_annotations():
    raw = FIRST * 10 + "aoi: これで配置を決められる。\n"
    plan = ScenePlan.model_validate(chapter_plan("s1")["scenes"][0])
    utterances = parse_scene_text(raw, plan.id, set(plan.character_ids))
    batches = []
    for start, count in zip(range(0, len(utterances), 10), (4, 50, 1, 1), strict=True):
        batch = utterances[start:start + 10]
        # Repeated annotations are deliberately retained; deduplication would
        # be a separate semantic policy, not the batch/aggregate limit fix.
        ids = ([batch[0].id] * 25 + [batch[1].id] * 24 + [batch[2].id]
               if count == 50 else [batch[index % len(batch)].id for index in range(count)])
        batches.append({
            "emotions": [{"utterance_id": row.id, "inner_emotion": "落ち着き",
                          "voice_emotion": "calm" if row.speaker_id else "neutral", "delivery": ""}
                         for row in batch],
            "directions": [{"utterance_id": identifier, "kind": "pause", "character_id": "",
                            "timing": "start", "duration_ms": 500} for identifier in ids],
        })
    return raw, plan, utterances, batches


def fake_llm(responses):
    return FakeLLM([{"content": json.dumps(value, ensure_ascii=False)} for value in responses])


def test_valid_batches_keep_all_annotations_and_the_per_request_limit():
    raw, plan, utterances, batches = scene_annotations()
    llm = fake_llm(batches)

    result = narrative._staging(llm, "context", plan, utterances)

    assert len(llm.calls) == 4
    assert len(result.emotions) == 31
    assert len(result.directions) == 56
    assert result.model_dump() == {
        field: [value for batch in batches for value in batch[field]]
        for field in ("emotions", "directions")
    }
    for _, extra in llm.calls:
        schema = extra["response_format"]["json_schema"]["schema"]
        assert schema["title"] == "Staging"
        assert schema["properties"]["directions"]["maxItems"] == 50
    assert len(narrative._directions(plan, utterances, result)) == 58  # Includes two entrances.
    assert [raw[row.source_start:row.source_end] for row in utterances] == [
        row.display_text for row in utterances]


def test_single_batch_over_50_is_still_rejected():
    _, plan, utterances, batches = scene_annotations()
    invalid = copy.deepcopy(batches[0])
    invalid["directions"] = [invalid["directions"][0]] * 51
    llm = fake_llm([invalid, invalid])

    with pytest.raises(narrative.StructuredGenerationError, match="at most 50"):
        narrative._staging(llm, "context", plan, utterances[:10])

    assert len(llm.calls) == 2


@pytest.mark.parametrize("invalid_reference", ["utterance", "character", "emotion_order"])
def test_batch_reference_checks_are_preserved(invalid_reference):
    _, plan, utterances, batches = scene_annotations()
    invalid = copy.deepcopy(batches[0])
    if invalid_reference == "utterance":
        invalid["directions"][0]["utterance_id"] = "unknown-utterance"
        message = "unknown utterance"
    elif invalid_reference == "character":
        invalid["directions"][0].update(kind="focus", character_id="unknown-character")
        message = "unknown character"
    else:
        invalid["emotions"].reverse()
        message = "preserve every source ID in order"
    llm = fake_llm([invalid, invalid])

    with pytest.raises(ValueError, match=message):
        narrative._staging(llm, "context", plan, utterances[:10])

    assert len(llm.calls) == 2


def test_explicit_retry_of_old_aggregate_failure_reuses_source_and_valid_batches(
        app_runtime, monkeypatch, tmp_path):
    calls, control = app_runtime
    raw, _, _, batches = scene_annotations()
    current_aggregate = narrative.SceneStaging
    monkeypatch.setattr(narrative, "SceneStaging", narrative.Staging)
    control["responses"] = [outline(), chapter_plan("s1"), raw, *batches]
    request = job()

    with pytest.raises(ValueError, match="at most 50"):
        generate(request, tmp_path)

    assert len(calls) == 7
    state_path = tmp_path / "script/draft-state.json"
    before = read_json(state_path)
    source = tmp_path / "script/sources/c001-s1.raw.txt"
    original_source = source.read_bytes()
    saved_requests = {path: path.read_bytes()
                      for path in (tmp_path / "script/requests").glob("*")}
    staging_keys = sorted((key for key in before["steps"] if "-staging-" in key),
                          key=lambda key: before["steps"][key]["attempts"][0]["request"])
    assert len(staging_keys) == 4
    rejected = before["steps"][staging_keys[-1]]["attempts"][0]
    assert "at most 50" in rejected["validation_error"]
    assert json.loads(rejected["reply"]["content"]) == batches[-1]

    monkeypatch.setattr(narrative, "SceneStaging", current_aggregate)
    with pytest.raises(DraftExecutionError, match="at most 50"):
        generate(request, tmp_path)
    assert len(calls) == 7  # Existing rejection remains until the explicit UI retry.

    request["retry_generation"] = 1
    control["responses"] = [batches[-1]]
    envelope = generate(request, tmp_path)

    assert [call["purpose"] for call in calls[7:]] == ["script-staging"]
    scene = envelope["result"]["scenes"][0]
    assert scene["raw_text"] == raw
    assert len(scene["utterances"]) == 31
    assert len(scene["directions"]) == 58
    assert [row["utterance_id"] for row in scene["directions"][2:]] == [
        row["utterance_id"] for batch in batches for row in batch["directions"]]
    after = read_json(state_path)
    for key in ("outline", "plan-001", "c001-s1-text", *staging_keys[:-1]):
        assert after["steps"][key] == before["steps"][key]
    assert after["steps"][staging_keys[-1]]["attempts"][0] == rejected
    assert len(after["steps"][staging_keys[-1]]["attempts"]) == 2
    assert source.read_bytes() == original_source
    assert all(path.read_bytes() == content for path, content in saved_requests.items())
    assert after["request_ordinal"] == len(calls) == 8
    assert read_json(tmp_path / "script/report.json")["metrics"]["charged_tokens"] == 8 * 530
    assert generate(request, tmp_path) == envelope
    assert len(calls) == 8
