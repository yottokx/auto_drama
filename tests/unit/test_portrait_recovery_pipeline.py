import io
import json
from zipfile import ZipFile

import pytest

from services.worker.generation import generate_job
from tests.unit.test_image_prompt_retry import _job, _response
from tests.unit.test_image_prompt_retry import (
    cached_runtime as cached_runtime,  # noqa: PLC0414 -- pytest fixture
)
from tests.unit.test_portrait_recovery import CHARACTER, FAILURE, diagnosis


@pytest.mark.parametrize("action", ["omit_portrait", "repair_prompt"])
def test_media_wrapper_recovers_only_after_failure_and_replays_result(tmp_path, cached_runtime, action):
    runtime = cached_runtime
    response = _response()
    features = json.loads(response["choices"][0]["message"]["content"])
    features.update(body="invisible", other_features="a directionless voice")
    response["choices"][0]["message"]["content"] = json.dumps(features)
    runtime.responses.extend([response, {"choices": [{"finish_reason": "stop", "message": {
        "role": "assistant", "content": json.dumps(diagnosis(action), ensure_ascii=False),
    }}]}])
    runtime.image_errors.append(FAILURE)
    job = _job("m3_image")
    job["payload"]["character_result"]["appearance"] = CHARACTER["appearance"]
    work = tmp_path / "job"
    data = generate_job(job, work)
    with ZipFile(io.BytesIO(data)) as archive:
        envelope = json.loads(archive.read("result.json"))
        assert envelope["kind"] == "m3_image"
        if action == "omit_portrait":
            assert archive.namelist() == ["result.json"]
            assert envelope["result"]["portrait"]["character_id"] == "receptionist"
        else:
            assert archive.read("image.png") == b"fixture-image"
            assert envelope["result"] == {}
    assert len(runtime.requests) == 2
    assert len(runtime.image_prompts) == (1 if action == "omit_portrait" else 2)
    # Cached completion never loads either model a second time.
    entered = runtime.entered
    assert generate_job(job, work) == data
    assert runtime.entered == entered
