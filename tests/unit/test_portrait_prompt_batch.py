"""Portrait conversion completes as one LLM phase before the image phase."""

import copy
import io
import json
from contextlib import nullcontext
from types import SimpleNamespace
from zipfile import ZipFile

import pytest

from services.worker.generation import generate_job, pipeline
from tests.unit.test_image_prompt_retry import _job, _response
from tests.unit.test_image_prompt_retry import cached_runtime as cached_runtime  # noqa: PLC0414


def jobs():
    first = _job("m3_image")
    first["payload"]["storyline_id"] = "series"
    second = copy.deepcopy(first)
    second.update(id="image-job-2")
    second["payload"].update(seed=123, character_id="guide")
    second["payload"]["character_result"].update(id="guide", name="案内係", appearance="赤い上着の案内係。")
    batch = [{key: job["payload"][key] for key in ("character_id", "character_result")} for job in (first, second)]
    for job in (first, second):
        job["payload"]["portrait_prompt_batch"] = copy.deepcopy(batch)
    return first, second


def envelope(data):
    with ZipFile(io.BytesIO(data)) as archive:
        return json.loads(archive.read("result.json"))


def test_all_portraits_translate_before_first_image_and_other_jobs_need_no_llm(tmp_path, cached_runtime):
    runtime = cached_runtime
    runtime.responses.extend([_response(), _response()])
    first, second = jobs()
    first_result = envelope(generate_job(first, tmp_path / "first"))
    assert len(runtime.requests) == 2 and runtime.entered == runtime.exited == 1
    assert len(runtime.image_prompts) == 1
    second_result = envelope(generate_job(second, tmp_path / "second"))
    assert len(runtime.requests) == 2 and runtime.entered == 1
    assert len(runtime.image_prompts) == 2
    for result, identifier in ((first_result, "receptionist"), (second_result, "guide")):
        conversion = next(row for row in result["trace"] if row["type"] == "image_prompt")
        assert conversion["character_id"] == identifier
    assert first_result["provenance"]["llm"]["seed"] == second_result["provenance"]["llm"]["seed"]


def test_changed_batch_source_cannot_reuse_old_prompt(tmp_path, cached_runtime):
    runtime = cached_runtime
    runtime.responses.extend([_response(), _response()])
    first, _ = jobs()
    generate_job(first, tmp_path / "first")
    changed = copy.deepcopy(first)
    changed["payload"]["character_result"]["appearance"] += "金色の帽子。"
    changed["payload"]["portrait_prompt_batch"][0]["character_result"] = copy.deepcopy(changed["payload"]["character_result"])
    runtime.responses.extend([_response(), _response()])
    generate_job(changed, tmp_path / "changed")
    assert len(runtime.requests) == 3 and runtime.entered == 2


def test_failed_batch_never_renders_and_retry_preserves_valid_cached_conversion(tmp_path, cached_runtime):
    runtime = cached_runtime
    runtime.responses.extend([_response(), _response(invalid=True), _response(invalid=True)])
    first, _ = jobs()
    with pytest.raises(ValueError, match="invalid or excessive"):
        generate_job(first, tmp_path / "first")
    assert not runtime.image_prompts
    runtime.responses.append(_response())
    generate_job(first, tmp_path / "first")
    assert len(runtime.requests) == 4  # First person's successful conversion is replayed.
    assert len(runtime.image_prompts) == 1


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("retake_batch", [False, True])
def test_fresh_retake_seed_reuses_conversion_without_releasing_image_model(
    tmp_path, cached_runtime, monkeypatch, batched, retake_batch,
):
    runtime = cached_runtime
    first = jobs()[0] if batched else _job("m2_image")
    runtime.responses.extend([_response()] * (2 if batched else 1))
    releases = []
    monkeypatch.setattr(pipeline.image_session, "current_session", lambda: SimpleNamespace(
        close=lambda: releases.append("image"), gpu_scope=lambda lease: nullcontext(),
    ))
    generate_job(first, tmp_path / "first")
    load_count = runtime.entered
    releases.clear()
    retake = copy.deepcopy(first)
    retake["id"] = "fresh-retake"
    retake["payload"].pop("portrait_prompt_batch", None)
    if retake_batch:
        retake["payload"]["portrait_prompt_batch"] = [{
            "character_id": retake["payload"]["character_id"],
            "character_result": copy.deepcopy(retake["payload"]["character_result"]),
        }]
    retake["payload"]["seed"] += 1
    first_result = envelope(generate_job(retake, tmp_path / "retake"))
    assert runtime.entered == load_count
    assert not releases
    assert runtime.image_prompts[0] == runtime.image_prompts[1]
    assert first_result["provenance"]["seed"] == retake["payload"]["seed"]


@pytest.mark.parametrize("change", ["appearance", "instruction", "profile"])
def test_retake_conversion_cache_tracks_source_instruction_and_model_profile(
    tmp_path, cached_runtime, change,
):
    runtime = cached_runtime
    runtime.responses.extend([_response(), _response()])
    first = _job("m2_image")
    generate_job(first, tmp_path / "first")
    retake = copy.deepcopy(first)
    retake["id"] = "changed-retake"
    if change == "appearance":
        retake["payload"]["character_result"]["appearance"] += "金色の帽子。"
    elif change == "instruction":
        retake["payload"]["instruction"] = "右手を挙げる。"
    else:
        retake["payload"]["profile"] = {"temperature": 0.6}
    generate_job(retake, tmp_path / "retake")
    assert runtime.entered == 2 and len(runtime.requests) == 2
