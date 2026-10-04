import copy
import io
import json
from contextlib import nullcontext
from zipfile import ZipFile

import pytest

from services.worker.generation import generate_job, pipeline
from tests.unit.test_image_prompt_retry import _job, _response
from tests.unit.test_image_prompt_retry import cached_runtime as cached_runtime  # noqa: PLC0414
from tests.unit.test_m2_generation import job
from tests.unit.test_m2_generation import (
    recorded_voice_runtime as recorded_voice_runtime,  # noqa: PLC0414
)


@pytest.mark.parametrize("instruction", ["", "正面向きに手を見せる構図"])
def test_adjustment_portrait_uses_step3_conversion_with_isolated_retake_cache(
    tmp_path, cached_runtime, instruction,
):
    runtime = cached_runtime
    runtime.responses.extend([_response(), _response(), _response()])
    original = _job("m2_image")
    original["payload"]["instruction"] = instruction
    generate_job(original, tmp_path / "normal")
    adjusted = copy.deepcopy(original)
    adjusted["kind"] = "m3_image"
    adjusted["payload"]["adjustment"] = {"draft_id": "draft", "revision": 0, "purpose": "candidate"}
    before = copy.deepcopy(adjusted)
    first = generate_job(adjusted, tmp_path / "adjustment")
    assert adjusted == before
    assert len(runtime.requests) == 2  # separate cache entries keep each source trace intact
    assert runtime.requests[0]["messages"] == runtime.requests[1]["messages"]
    assert runtime.image_prompts[0] == runtime.image_prompts[1]
    with ZipFile(io.BytesIO(first)) as archive:
        envelope = json.loads(archive.read("result.json"))
    assert envelope["result"] == {} and envelope["provenance"]["purpose"] == "adjustment_portrait"
    assert generate_job(adjusted, tmp_path / "adjustment") == first
    assert len(runtime.requests) == 2
    next_request = copy.deepcopy(adjusted)
    next_request["payload"]["instruction"] = "手を振る"
    generate_job(next_request, tmp_path / "next")
    if instruction:
        assert instruction not in json.dumps(runtime.requests[-1]["messages"], ensure_ascii=False)
    assert original["payload"]["character_result"] == next_request["payload"]["character_result"]


def test_edited_appearance_uses_same_step3_conversion_and_repair(
    tmp_path, cached_runtime,
):
    runtime = cached_runtime
    runtime.responses.extend([_response(invalid=True), _response()] * 2)
    adjusted = _job("m3_image")
    adjusted["payload"].update(
        adjustment={"draft_id": "draft", "purpose": "candidate"},
        source_prompt="緑のコートを着た小柄な女性。",
        instruction="正面向きに、手を見せる構図で",
    )
    normal = _job("m2_image")
    normal["payload"]["character_result"]["appearance"] = adjusted["payload"]["source_prompt"]
    normal["payload"]["instruction"] = adjusted["payload"]["instruction"]
    before = copy.deepcopy(adjusted)
    generate_job(normal, tmp_path / "normal")
    result = _envelope(generate_job(adjusted, tmp_path / "adjusted"))

    assert adjusted == before
    assert len(runtime.requests) == 4
    for index in (0, 1):
        assert runtime.requests[index]["messages"] == runtime.requests[index + 2]["messages"]
    assert runtime.image_prompts[0] == runtime.image_prompts[1]
    assert pipeline.PORTRAIT_COMPOSITION in runtime.image_prompts[1]
    assert result["provenance"]["prompt_details"] == {
        "source": adjusted["payload"]["character_result"]["appearance"],
        "input": adjusted["payload"]["source_prompt"],
        "instruction": adjusted["payload"]["instruction"],
        "effective": runtime.image_prompts[1],
    }


def test_adjustment_reference_uses_exact_requested_transcript_without_editing_character(
    tmp_path, recorded_voice_runtime,
):
    payload = job("m2_voice")["payload"]
    payload.update(adjustment={"purpose": "candidate"}, reference_text="これは今回の確認用の声です。", instruction="少し低く")
    before = copy.deepcopy(payload)
    _, report = pipeline.generate_voice(payload, tmp_path, pipeline.load_config())
    assert payload == before
    assert report["reference_text"] == payload["reference_text"]
    assert recorded_voice_runtime[-1]["caption"].endswith("少し低く")
    payload["instruction"] = "落ち着いて"
    pipeline.generate_voice(payload, tmp_path, pipeline.load_config())
    assert "少し低く" not in recorded_voice_runtime[-1]["caption"]


def _envelope(bundle):
    with ZipFile(io.BytesIO(bundle)) as archive:
        return json.loads(archive.read("result.json"))


def test_direct_portrait_prompt_is_isolated_and_preserved_on_conversion_cache_hits(
    tmp_path, cached_runtime, monkeypatch,
):
    runtime = cached_runtime
    runtime.responses.extend([_response(), _response(), _response()])
    render = pipeline.generate_image

    def rendered_prompt(payload, prompt, work, config):
        image, report = render(payload, prompt, work, config)
        return image, {**report, "prompt": "quality, " + prompt}

    monkeypatch.setattr(pipeline, "generate_image", rendered_prompt)
    original = _job("m3_image")
    original["payload"]["adjustment"] = {"draft_id": "draft", "purpose": "candidate"}
    generate_job(original, tmp_path / "original")

    edited = copy.deepcopy(original)
    edited["payload"]["source_prompt"] = "赤いエプロンを着た、短い白髪の小柄な女性。"
    before = copy.deepcopy(edited)
    result = _envelope(generate_job(edited, tmp_path / "direct"))
    assert edited == before
    assert len(runtime.requests) == 2
    target, _ = json.JSONDecoder().raw_decode(runtime.requests[-1]["messages"][1]["content"])
    assert target["target_character"]["appearance"] == edited["payload"]["source_prompt"]
    assert target["retake_instruction"] == ""
    details = result["provenance"]["prompt_details"]
    assert details == {
        "source": original["payload"]["character_result"]["appearance"],
        "input": edited["payload"]["source_prompt"], "instruction": "",
        "effective": "quality, " + runtime.image_prompts[-1],
    }
    conversion = next(item for item in result["trace"] if item["type"] == "image_prompt")
    assert conversion["source_appearance"] == details["source"]
    assert conversion["input_appearance"] == details["input"]

    retake = copy.deepcopy(edited)
    retake["payload"]["seed"] += 1
    reused = _envelope(generate_job(retake, tmp_path / "retake"))
    assert len(runtime.requests) == 2
    assert reused["provenance"]["prompt_details"] == details

    revised = copy.deepcopy(edited)
    revised["payload"]["source_prompt"] = "緑のコートを着た小柄な女性。"
    revised["payload"]["instruction"] = "帽子をかぶる"
    generated = _envelope(generate_job(revised, tmp_path / "revised"))
    assert len(runtime.requests) == 3
    assert generated["provenance"]["prompt_details"]["input"] == revised["payload"]["source_prompt"]
    assert generated["provenance"]["prompt_details"]["instruction"] == "帽子をかぶる"
    assert edited["payload"]["source_prompt"] not in runtime.requests[-1]["messages"][1]["content"]


@pytest.mark.parametrize("instruction", ["", "少しゆっくり話す"])
def test_direct_voice_prompt_records_exact_input_and_effective_caption(
    tmp_path, recorded_voice_runtime, monkeypatch, instruction,
):
    monkeypatch.setattr(pipeline, "gpu_lock", lambda *args: nullcontext())
    request = job("m3_voice")
    request["payload"].update(
        adjustment={"draft_id": "draft", "purpose": "candidate"},
        source_prompt="明るく高めの声。", instruction=instruction,
        reference_text="編集した声で、この文章を読みます。",
    )
    before = copy.deepcopy(request)
    result = _envelope(generate_job(request, tmp_path / "voice"))
    assert request == before
    expected = request["payload"]["source_prompt"]
    if instruction:
        expected += "\n今回の演技・声の調整: " + instruction
    assert recorded_voice_runtime[-1]["caption"] == expected
    assert result["provenance"]["prompt_details"] == {
        "source": request["payload"]["character_result"]["voice"],
        "input": request["payload"]["source_prompt"],
        "instruction": instruction, "effective": expected,
    }
    assert result["provenance"]["voice"]["reference_text"] == request["payload"]["reference_text"]


def test_source_prompt_override_is_ignored_outside_adjustments(
    tmp_path, cached_runtime, recorded_voice_runtime,
):
    cached_runtime.responses.append(_response())
    portrait = _job("m2_image")
    portrait["payload"]["source_prompt"] = "調整以外で使用してはいけない外見。"
    result = _envelope(generate_job(portrait, tmp_path / "portrait"))
    target, _ = json.JSONDecoder().raw_decode(cached_runtime.requests[-1]["messages"][1]["content"])
    assert target["target_character"]["appearance"] == portrait["payload"]["character_result"]["appearance"]
    assert "prompt_details" not in result["provenance"]

    voice = job("m2_voice")["payload"]
    voice["source_prompt"] = "調整以外で使用してはいけない声。"
    # The portrait fixture replaces load_config with its minimal image settings.
    config = {"voice": {"python": "python", "reference_text": "既定の文章。", "num_steps": 20,
                        "model_precision": "bf16", "timeout_seconds": 1}}
    pipeline.generate_voice(voice, tmp_path, config)
    assert recorded_voice_runtime[-1]["caption"] == voice["character_result"]["voice"]
