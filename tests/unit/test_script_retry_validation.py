"""Explicit retries regenerate rejected stages while retaining successful source."""

import copy

import pytest

from packages.narrative.speech import parenthetical_candidates
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.draft_story import DraftExecutionError
from tests.unit.test_script_continuation_run import (
    FIRST,
    chapter_plan,
    outline,
    read_json,
    staging,
)
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import generate, job
from tests.unit.test_script_production import runtime as _runtime

runtime = _runtime
app_runtime = _app_runtime


@pytest.mark.parametrize("purpose", ["speech", "staging"])
@pytest.mark.parametrize("rejection", ["schema", "references"])
def test_explicit_retry_replaces_rejected_annotations_only(
        app_runtime, tmp_path, purpose, rejection):
    calls, control = app_runtime
    raw = FIRST.replace("年は", "（仮の）年は") if purpose == "speech" else FIRST
    if purpose == "speech":
        candidate = parenthetical_candidates(raw, "s1", {"aoi", "ren"})[0]
        valid = {"annotations": [{"candidate_id": candidate.id, "kind": "spoken",
                                   "narration": "", "delivery": ""}]}
        invalid = copy.deepcopy(valid)
        if rejection == "schema":
            invalid = {"annotations": []}
        else:
            invalid["annotations"][0]["candidate_id"] = "unknown-candidate"
        fresh_responses = [valid, staging(raw)]
    else:
        valid = staging(raw)
        invalid = copy.deepcopy(valid)
        if rejection == "schema":
            invalid = {"emotions": "not-an-array", "directions": []}
        else:
            invalid["emotions"][0]["utterance_id"] = "unknown-utterance"
        fresh_responses = [valid]
    control["responses"] = [outline(), chapter_plan("s1"), raw, invalid, invalid]
    request = job()
    with pytest.raises((RuntimeError, ValueError)):
        generate(request, tmp_path)
    assert len(calls) == 5
    state_path = tmp_path / "script/draft-state.json"
    before = read_json(state_path)
    source = tmp_path / "script/sources/c001-s1.raw.txt"
    original_source = source.read_bytes()
    saved_requests = {path: path.read_bytes()
                      for path in (tmp_path / "script/requests").glob("*")}
    with pytest.raises((RuntimeError, ValueError)):
        generate(request, tmp_path)
    assert len(calls) == 5

    request["retry_generation"] = 1
    control["responses"] = fresh_responses
    envelope = generate(request, tmp_path)
    after = read_json(state_path)
    expected = ["script-speech", "script-staging"] if purpose == "speech" else ["script-staging"]
    assert [call["purpose"] for call in calls[5:]] == expected
    assert envelope["result"]["scenes"][0]["raw_text"] == raw
    assert source.read_bytes() == original_source
    for key in ("outline", "plan-001", "c001-s1-text"):
        assert after["steps"][key] == before["steps"][key]
    assert all(path.read_bytes() == content for path, content in saved_requests.items())
    assert after["request_ordinal"] == len(calls)
    assert read_json(tmp_path / "script/report.json")["metrics"]["charged_tokens"] == len(calls) * 530
    assert generate(request, tmp_path) == envelope
    assert len(calls) == 5 + len(expected)


def test_explicit_retry_replaces_unknown_writer_speaker(app_runtime, tmp_path):
    calls, control = app_runtime
    invalid = FIRST.replace("aoi:", "unknown-character:", 1)
    control["responses"] = [outline(), chapter_plan("s1"), invalid]
    request = job()
    with pytest.raises(ValueError, match="Unknown scene speaker"):
        generate(request, tmp_path)
    before = read_json(tmp_path / "script/draft-state.json")
    with pytest.raises((RuntimeError, ValueError)):
        generate(request, tmp_path)
    assert len(calls) == 3

    request["retry_generation"] = 1
    control["responses"] = [FIRST, staging(FIRST)]
    envelope = generate(request, tmp_path)
    after = read_json(tmp_path / "script/draft-state.json")
    assert [call["purpose"] for call in calls[3:]] == ["script-scene", "script-staging"]
    assert envelope["result"]["scenes"][0]["raw_text"] == FIRST
    for key in ("outline", "plan-001"):
        assert after["steps"][key] == before["steps"][key]
    assert after["steps"]["c001-s1-text"]["attempts"][0] == before["steps"]["c001-s1-text"]["attempts"][0]
    assert after["steps"]["c001-s1-text"]["attempts"][0]["reply"]["content"] == invalid


def test_explicit_retry_replaces_bad_continuation_only(app_runtime, tmp_path):
    calls, control = app_runtime
    prefix = FIRST.splitlines(keepends=True)[0]
    control["responses"] = [outline(), chapter_plan("s1"),
                            {"content": prefix, "_finish_reason": "length"},
                            "NARRATOR: This continuation omitted its source anchor."]
    request = job()
    with pytest.raises(DraftExecutionError, match="source anchor"):
        generate(request, tmp_path)
    before = read_json(tmp_path / "script/draft-state.json")
    with pytest.raises((RuntimeError, ValueError)):
        generate(request, tmp_path)
    assert len(calls) == 4

    request["retry_generation"] = 1
    control["responses"] = [FIRST, staging(FIRST)]
    envelope = generate(request, tmp_path)
    after = read_json(tmp_path / "script/draft-state.json")
    assert [call["purpose"] for call in calls[4:]] == ["script-scene", "script-staging"]
    assert envelope["result"]["scenes"][0]["raw_text"] == FIRST
    for key in ("outline", "plan-001", "c001-s1-text"):
        assert after["steps"][key] == before["steps"][key]
    assert len(after["steps"]["c001-s1-continue"]["attempts"]) == 2
    assert after["steps"]["c001-s1-continue"]["attempts"][0] == before["steps"]["c001-s1-continue"]["attempts"][0]


def test_explicit_retry_after_two_uncached_interruptions(app_runtime, tmp_path):
    calls, control = app_runtime
    control["responses"] = [outline(), chapter_plan("s1"),
                            GenerationCancelled("first interruption"),
                            GenerationCancelled("second interruption")]
    request = job()
    for _ in range(2):
        with pytest.raises(GenerationCancelled):
            generate(request, tmp_path)
    assert len(calls) == 4
    with pytest.raises(DraftExecutionError, match="retry exhausted"):
        generate(request, tmp_path)
    assert len(calls) == 4
    before = read_json(tmp_path / "script/draft-state.json")

    request["retry_generation"] = 1
    control["responses"] = [FIRST, staging(FIRST)]
    envelope = generate(request, tmp_path)
    after = read_json(tmp_path / "script/draft-state.json")
    assert [call["purpose"] for call in calls[4:]] == ["script-scene", "script-staging"]
    assert envelope["result"]["scenes"][0]["raw_text"] == FIRST
    assert after["steps"]["c001-s1-text"]["attempts"][:2] == before["steps"]["c001-s1-text"]["attempts"]
    assert after["request_ordinal"] == 6
    assert read_json(tmp_path / "script/report.json")["metrics"]["charged_tokens"] == 6 * 530
