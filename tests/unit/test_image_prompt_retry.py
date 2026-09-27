from __future__ import annotations

import copy
import io
import json
import zipfile
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from services.worker.generation import generate_job, pipeline
from services.worker.generation.llm import LocalLLM


def _features(*, invalid: bool = False) -> dict:
    return {
        "subject": "An elderly woman",
        "body": "Short stature",
        "skin": "",
        "hair": "Short gray hair",
        "eyes": "",
        "clothing": "A blue apron",
        "accessories": 'A name badge reading "受付"' if invalid else "A reception name badge",
        "other_features": "",
        "rendering": "",
    }


def _response(*, invalid: bool = False) -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(_features(invalid=invalid), ensure_ascii=False),
    }}]}


def _job(kind: str) -> dict:
    return {"id": "image-job", "kind": kind, "payload": {
        "schema_version": 1, "seed": 99, "character_id": "receptionist",
        "character_result": {
            "id": "receptionist", "name": "受付係", "age": "70歳", "gender": "女性",
            "appearance": "小柄な女性。短い白髪に青いエプロン。受付と書かれた名札。",
        },
        "instruction": "",
    }}


@pytest.fixture
def cached_runtime(tmp_path, monkeypatch):
    """Keep real LocalLLM request keys and disk cache; replace only external runtimes."""
    base_path = tmp_path / "llm-config.json"
    base_path.write_text(json.dumps({"model": {
        "revision": "fixture-revision", "publisher_sha256": "a" * 64,
    }}), encoding="utf-8")
    config = {
        "llm_config": str(base_path),
        "llm": {"provider": "local", "model_id": "fixture-model", "temperature": 0.7,
                "max_tokens": 1024, "max_tool_calls": 4},
        "gpu_lock_timeout_seconds": 1, "max_zip_bytes": 1_000_000,
    }
    runtime = SimpleNamespace(
        responses=[], requests=[], image_errors=[], image_prompts=[], entered=0, exited=0,
    )

    def enter(llm):
        runtime.entered += 1
        llm.output.mkdir(parents=True, exist_ok=True)
        return llm

    def exit_llm(llm, *args):
        runtime.exited += 1

    def request(llm, path, value=None, timeout=None):
        assert path == "/v1/chat/completions"
        runtime.requests.append(copy.deepcopy(value))
        assert runtime.responses, "Unexpected HTTP request instead of persisted cache reuse"
        response = runtime.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return copy.deepcopy(response)

    def image(payload, prompt, work, settings):
        assert runtime.entered == runtime.exited, "Image runtime started before Gemma exited"
        runtime.image_prompts.append(prompt)
        if runtime.image_errors:
            raise runtime.image_errors.pop(0)
        return b"fixture-image", {"alpha_range": [0, 255]}

    monkeypatch.setattr(pipeline, "load_config", lambda: copy.deepcopy(config))
    monkeypatch.setattr(pipeline, "gpu_lock", lambda *args: nullcontext())
    monkeypatch.setattr(pipeline, "generate_image", image)
    monkeypatch.setattr(LocalLLM, "__enter__", enter)
    monkeypatch.setattr(LocalLLM, "__exit__", exit_llm)
    monkeypatch.setattr(LocalLLM, "request", request)
    return runtime


@pytest.mark.parametrize("kind", ["m2_image", "m3_image"])
def test_rejected_conversion_retries_from_first_request_without_deleting_history(
    tmp_path, cached_runtime, kind,
):
    runtime = cached_runtime
    runtime.responses.extend([_response(invalid=True), _response(invalid=True)])
    work = tmp_path / "job"
    source = _job(kind)
    original = copy.deepcopy(source)

    with pytest.raises(ValueError, match="invalid or excessive English text"):
        generate_job(source, work)

    assert len(runtime.requests) == 2
    assert runtime.image_prompts == []
    assert not (work / "result.zip").exists()
    retry = json.loads((work / "llm/retry-state.json").read_text(encoding="utf-8"))
    assert retry["segments"][0]["from_request"] == 1
    assert retry["segments"][0]["salt"] != 0
    original_cache = {path: path.read_bytes() for path in (work / "llm").glob("*-image-prompt-*.json")}
    assert len(original_cache) == 2
    snapshot = (work / "job-request.json").read_bytes()

    runtime.responses.append(_response())
    result = generate_job(source, work)

    assert len(runtime.requests) == 3
    assert runtime.requests[2]["seed"] != runtime.requests[0]["seed"]
    assert runtime.requests[2]["messages"] == runtime.requests[0]["messages"]
    assert all(path.read_bytes() == content for path, content in original_cache.items())
    assert len(list((work / "llm").glob("*-image-prompt-*.json"))) == 3
    assert (work / "job-request.json").read_bytes() == snapshot
    assert source == original
    assert len(runtime.image_prompts) == 1
    with zipfile.ZipFile(io.BytesIO(result)) as archive:
        assert json.loads(archive.read("result.json"))["kind"] == kind
        assert archive.read("image.png") == b"fixture-image"


@pytest.mark.parametrize("kind", ["m2_image", "m3_image"])
@pytest.mark.parametrize("repair_needed", [False, True])
def test_image_runtime_failure_reuses_successful_conversion_including_its_repair(
    tmp_path, cached_runtime, kind, repair_needed,
):
    runtime = cached_runtime
    if repair_needed:
        runtime.responses.append(_response(invalid=True))
    runtime.responses.append(_response())
    # A renderer's ValueError must not be mistaken for a translation rejection.
    runtime.image_errors.append(ValueError("Image runtime rejected its output"))
    work = tmp_path / "job"
    source = _job(kind)

    with pytest.raises(ValueError, match="Image runtime rejected"):
        generate_job(source, work)

    assert not (work / "llm/retry-state.json").exists()
    assert not (work / "result.zip").exists()
    cache = {path: path.read_bytes() for path in (work / "llm").glob("*-image-prompt-*.json")}
    assert len(cache) == 1 + int(repair_needed)
    generate_job(source, work)

    assert len(runtime.requests) == len(cache)
    assert len(runtime.image_prompts) == 2
    assert runtime.image_prompts[0] == runtime.image_prompts[1]
    assert all(path.read_bytes() == content for path, content in cache.items())
    assert not (work / "llm/retry-state.json").exists()


@pytest.mark.parametrize("kind", ["m2_image", "m3_image"])
def test_completed_image_zip_is_reused_without_model_or_renderer(
    tmp_path, cached_runtime, kind,
):
    runtime = cached_runtime
    runtime.responses.append(_response())
    work = tmp_path / "job"
    source = _job(kind)
    first = generate_job(source, work)
    saved_zip = (work / "result.zip").read_bytes()

    assert generate_job(source, work) == first
    assert (work / "result.zip").read_bytes() == saved_zip
    assert runtime.entered == runtime.exited == 1
    assert len(runtime.requests) == len(runtime.image_prompts) == 1
    assert not (work / "llm/retry-state.json").exists()


@pytest.mark.parametrize("kind", ["m2_image", "m3_image"])
@pytest.mark.parametrize("error_type", [RuntimeError, TimeoutError])
def test_uncertain_llm_failure_keeps_seed_for_retry(
    tmp_path, cached_runtime, kind, error_type,
):
    runtime = cached_runtime
    runtime.responses.append(error_type("Local LLM connection failed"))
    work = tmp_path / "job"
    source = _job(kind)

    with pytest.raises(error_type, match="Local LLM connection failed"):
        generate_job(source, work)

    assert not (work / "llm/retry-state.json").exists()
    assert not list((work / "llm").glob("*-image-prompt-*.json"))
    assert runtime.image_prompts == []
    runtime.responses.append(_response())
    generate_job(source, work)

    assert len(runtime.requests) == 2
    assert runtime.requests[0] == runtime.requests[1]
    assert len(runtime.image_prompts) == 1
    assert not (work / "llm/retry-state.json").exists()
