"""Raw script constraints must not prevent the selected model from reasoning."""

import json

import pytest

from services.worker.generation import narrative, script_continuation as runner
from services.worker.generation.draft_story import DraftExecutionError
from services.worker.model_config import REGISTRY
from tests.unit.test_m3_source_repair import SourceLLM, plan
from tests.unit.test_script_continuation_run import (
    FIRST,
    SECOND,
    chapter_plan,
    outline,
    read_json,
    staging,
)
from tests.unit.test_script_output_budget_run import common_32k_input
from tests.unit.test_script_production import app_runtime as _app_runtime
from tests.unit.test_script_production import generate, job
from tests.unit.test_script_production import runtime as _runtime

runtime = _runtime
app_runtime = _app_runtime


@pytest.mark.parametrize("effort", ["none", "medium", "custom-effort"])
def test_writer_and_continuation_keep_reasoning_and_budget_without_raw_grammar(
        runtime, monkeypatch, tmp_path, effort):
    calls, control = runtime
    source = common_32k_input(monkeypatch, effort=effort)
    anchor = FIRST[-min(160, len(FIRST)):]
    reasoning = "The returned reasoning is metadata, not script source."
    control["responses"] = [outline(), chapter_plan("s1"),
        {"content": FIRST, "reasoning_content": reasoning, "_finish_reason": "length"},
        {"content": anchor + SECOND, "reasoning_content": reasoning, "_finish_reason": "stop"},
        staging(FIRST + SECOND)]

    report = runner.run_script_debug(source, tmp_path, chapter_limit=1)

    assert report["status"] == "chapter_limit_reached"
    state = read_json(tmp_path / "draft-state.json")
    frozen = state["scene_budgets"]["c001-s1"]["budget"]
    assert frozen["max_tokens"] == frozen["baseline_max_tokens"] + 8192
    writers = [call for call in calls if call["purpose"] == "script-scene"]
    assert len(writers) == 2
    for suffix, call in zip(("text", "continue"), writers, strict=True):
        key = "c001-s1-" + suffix
        saved = read_json(tmp_path / "requests" / f"{key}-1.json")
        attempt = state["steps"][key]["attempts"][0]
        assert ("grammar" in call["extra"]) is (effort == "none")
        assert ("grammar" in saved["request"]) is (effort == "none")
        assert saved["request"]["reasoning_effort"] == call["profile"]["reasoning_level"] == effort
        assert (saved["request"]["max_tokens"] == call["extra"]["max_tokens"]
                == saved["selection"]["budget"]["output_tokens"] == frozen["max_tokens"])
        assert saved["output_budget"] == attempt["output_budget"] == frozen
        assert attempt["reply"]["reasoning_content"] == reasoning
    assert state["chapters"][0]["text"] == FIRST + SECOND
    assert (tmp_path / "sources/c001-s1.raw.txt").read_text(encoding="utf-8") == FIRST + SECOND
    assert runner.run_script_debug(source, tmp_path, chapter_limit=1, resume=True)["status"] == "chapter_limit_reached"
    assert len(calls) == 5


@pytest.mark.parametrize("invalid", [
    FIRST.replace("aoi:", "unknown-character:", 1),
    FIRST.replace("aoi: ", "", 1),
])
def test_reasoning_writer_still_rejects_invalid_speaker_or_line(
        runtime, monkeypatch, tmp_path, invalid):
    calls, control = runtime
    source = common_32k_input(monkeypatch)
    control["responses"] = [outline(), chapter_plan("s1"), invalid]

    with pytest.raises(ValueError):
        runner.run_script_debug(source, tmp_path, chapter_limit=1)

    assert "grammar" not in calls[-1]["extra"]
    assert len(calls) == 3
    state = read_json(tmp_path / "draft-state.json")
    assert state["steps"]["c001-s1-text"]["attempts"][0]["validation_error"]
    assert not state["chapters"] and not state["scene_commits"]
    assert (tmp_path / "sources/c001-s1.raw.txt").read_text(encoding="utf-8") == invalid


def test_reasoning_continuation_still_requires_exact_source_anchor(
        runtime, monkeypatch, tmp_path):
    calls, control = runtime
    source = common_32k_input(monkeypatch)
    control["responses"] = [outline(), chapter_plan("s1"),
        {"content": FIRST, "_finish_reason": "length"}, SECOND]

    with pytest.raises(DraftExecutionError, match="source anchor"):
        runner.run_script_debug(source, tmp_path, chapter_limit=1)

    assert len(calls) == 4
    assert "grammar" not in calls[-1]["extra"]
    state = read_json(tmp_path / "draft-state.json")
    assert state["steps"]["c001-s1-continue"]["attempts"][0]["validation_error"]
    assert not state["chapters"] and not state["scene_commits"]
    assert (tmp_path / "sources/c001-s1.raw.txt").read_text(encoding="utf-8") == FIRST


@pytest.mark.parametrize("effort", ["none", "medium", "custom-effort"])
def test_legacy_source_repair_uses_reasoning_safe_options_for_both_chunks(effort):
    prefix = "Hero: 門の向こうに"
    complete = prefix + "薬を届ける。\nNARRATOR: 袋を差し出した。"
    llm = SourceLLM([{"content": "unknown: 門を開けて。"},
        {"content": prefix, "_finish_reason": "length"},
        {"content": complete, "_finish_reason": "stop"}])
    llm.payload["profile"] = {"common_settings_version": 1, "reasoning_level": effort}

    raw, utterances = narrative._scene_text(llm, "write", plan())

    assert raw == complete and len(utterances) == 2
    assert len(llm.calls) == 3
    assert "grammar" not in llm.calls[0][2]
    assert all(("grammar" in call[2]) is (effort == "none") for call in llm.calls[1:])
    assert llm.profile["reasoning_level"] == effort
    assert prefix in llm.calls[-1][1][-1]["content"]


def common_app_job(monkeypatch, tmp_path):
    """Use real registry/config selection without any dependency on local models."""
    worker = tmp_path / "worker"
    models = worker / "models"
    models.mkdir(parents=True)
    (models / "portable.gguf").write_bytes(b"GGUFtest")
    server = worker / "llama-server.exe"
    server.touch()
    registry = worker / REGISTRY
    registry.parent.mkdir(parents=True)
    registry.write_text(json.dumps({"schema_version": 1, "model_dirs": [str(models)],
                                   "server_executable": str(server)}), encoding="utf-8")
    monkeypatch.setattr(runner.pipeline, "ROOT", worker)
    request = job()
    request["payload"]["profile"] = {"common_settings_version": 1, "model_id": "portable",
        "temperature": 0.65, "top_p": 0.95, "reasoning_level": "medium", "context_size": 32768}
    return request


def test_explicit_retry_of_old_reasoning_grammar_failure_starts_fresh_scene(
        app_runtime, monkeypatch, tmp_path):
    calls, control = app_runtime
    request = common_app_job(monkeypatch, tmp_path)
    work = tmp_path / "production"
    current_options = narrative._source_options

    def old_options(_profile, scene_plan, raw=""):
        return {"grammar": narrative._source_grammar(scene_plan, raw)}

    monkeypatch.setattr(narrative, "_source_options", old_options)
    anchor = FIRST[-min(160, len(FIRST)):]
    control["responses"] = [outline(), chapter_plan("s1"),
        {"content": FIRST, "_finish_reason": "length"},
        {"content": anchor + SECOND, "_finish_reason": "length"}]
    with pytest.raises(DraftExecutionError, match="still truncated"):
        generate(request, work)
    assert len(calls) == 4
    before = read_json(work / "script/draft-state.json")
    saved = {path: path.read_bytes() for path in (work / "script/requests").glob("*")}
    original_request = (work / "job-request.json").read_bytes()
    original_manifest = (work / "script/experiment.json").read_bytes()

    monkeypatch.setattr(narrative, "_source_options", current_options)
    request["retry_generation"] = 1
    control["responses"] = [SECOND, staging(SECOND)]
    envelope = generate(request, work)

    assert envelope["result"]["scenes"][0]["raw_text"] == SECOND
    assert [call["purpose"] for call in calls[4:]] == ["script-scene", "script-staging"]
    assert "grammar" not in calls[4]["extra"]
    assert "接続文字列" not in calls[4]["messages"][0]["content"]
    after = read_json(work / "script/draft-state.json")
    assert after["scene_budgets"] == before["scene_budgets"]
    for key in ("outline", "plan-001"):
        assert after["steps"][key] == before["steps"][key]
    for key in ("c001-s1-text", "c001-s1-continue"):
        assert after["steps"][key]["attempts"][0] == before["steps"][key]["attempts"][0]
    assert after["steps"]["c001-s1-text"]["attempts"][-1]["reply"]["content"] == SECOND
    assert all(path.read_bytes() == content for path, content in saved.items())
    assert (work / "job-request.json").read_bytes() == original_request
    assert (work / "script/experiment.json").read_bytes() == original_manifest
    assert (work / "script/sources/c001-s1.raw.txt").read_text(encoding="utf-8") == SECOND
    assert after["request_ordinal"] == len(calls) == 6
    assert read_json(work / "script/report.json")["metrics"]["charged_tokens"] == 6 * 530
    assert generate(request, work) == envelope
    assert len(calls) == 6
