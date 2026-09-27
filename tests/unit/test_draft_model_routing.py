"""The draft experiment selects writer/planner models without semantic review gates."""

import copy
import json
import sys
from types import SimpleNamespace

import pytest

from scripts.story import check_text
from services.worker.generation import model_routing as routing
from services.worker.generation import text_debug
from services.worker.generation.pipeline import load_config
from services.worker.generation.workflow_version import generator_protocol

PAYLOAD = {"story_workflow_version": 2, "workflow_policy": "story_draft_v1",
           "execution_mode": "text_only"}


@pytest.mark.parametrize("purpose,category,model,context,reasoning,output", [
    ("draft-outline", "draft_outline", "Qwen", 32768, "low", 8192),
    ("draft-handoff", "draft_handoff", "Qwen", 32768, "none", 4096),
    ("draft-chapter", "draft_writer", "gemma", 16384, "none", 8192),
])
def test_draft_uses_the_expected_model_and_output_budget(
        purpose, category, model, context, reasoning, output):
    selected, profile = routing.resolve_purpose_profile(load_config(), PAYLOAD, purpose)
    assert selected == category
    assert profile["model_id"].startswith(model)
    assert (profile["context_size"], profile["reasoning_level"], profile["max_tokens"]) == (
        context, reasoning, output)


@pytest.mark.parametrize("payload", [{}, {"story_workflow_version": 2},
    {"story_workflow_version": 1, "workflow_policy": "story_draft_v1"},
    {"story_workflow_version": 2, "workflow_policy": "chapter_editor_v1"}])
def test_draft_purposes_do_not_change_other_workflow_routes(payload):
    config = load_config()
    category, profile = routing.resolve_purpose_profile(config, payload, "draft-outline")
    assert category == "draft-outline"
    assert profile["model_id"] == config["llm"]["model_id"]
    assert profile["max_tokens"] == config["llm"]["max_tokens"]


def test_draft_profiles_honor_per_purpose_overrides_and_model_context_settings():
    config = copy.deepcopy(load_config())
    config["model_routing"]["review"]["llm"]["context_size"] = 65536
    payload = {**PAYLOAD, "profile": {"max_tokens": 6144}, "profiles": {
        "draft_handoff": {"max_tokens": 2048},
        "draft-handoff": {"reasoning_level": "low", "max_tokens": 3072},
        "draft_writer": {"context_size": 24576},
        "draft-chapter": {"max_tokens": 7168},
    }}
    _, handoff = routing.resolve_purpose_profile(config, payload, "draft-handoff")
    assert (handoff["context_size"], handoff["max_tokens"], handoff["reasoning_level"]) == (
        65536, 3072, "low")
    _, writer = routing.resolve_purpose_profile(config, payload, "draft-chapter")
    assert (writer["context_size"], writer["max_tokens"]) == (24576, 7168)


def test_draft_default_outputs_respect_worker_allowances():
    config = copy.deepcopy(load_config())
    config["llm"]["max_output_tokens"] = 6144
    config["model_routing"]["review"]["llm"]["max_output_tokens"] = 2048
    assert routing.resolve_purpose_profile(config, PAYLOAD, "draft-chapter")[1]["max_tokens"] == 6144
    for purpose in ("draft-outline", "draft-handoff"):
        assert routing.resolve_purpose_profile(config, PAYLOAD, purpose)[1]["max_tokens"] == 2048


def test_draft_requires_both_routes_and_cannot_override_model_selection():
    config = copy.deepcopy(load_config())
    config.pop("model_routing")
    with pytest.raises(ValueError, match="Qwen route"):
        routing.resolve_purpose_profile(config, PAYLOAD, "draft-outline")
    payload = {**PAYLOAD, "profiles": {"draft-handoff": {"model_id": "different-model"}}}
    with pytest.raises(ValueError, match="configured model route"):
        routing.resolve_purpose_profile(load_config(), payload, "draft-handoff")


def test_draft_protocol_has_its_own_replay_boundary_and_rejects_legacy():
    draft = generator_protocol("causal", "story_draft_v1")
    assert draft["policy"] == "story_draft_v1"
    assert draft != generator_protocol("causal", "chapter_editor_v1")
    draft["implementation"] += 1
    assert draft != generator_protocol("causal", "story_draft_v1")
    with pytest.raises(ValueError, match="unsupported"):
        generator_protocol("legacy", "story_draft_v1")


def test_text_debug_dispatches_draft_directly_without_narrative_validation(monkeypatch, tmp_path):
    calls = []

    def run(value, output, **kwargs):
        calls.append((value, output, kwargs))
        return {"status": "draft_complete", "saved_chapter_count": 2}

    monkeypatch.setitem(sys.modules, "services.worker.generation.draft_story",
                        SimpleNamespace(run_draft_debug=run))
    monkeypatch.setattr(text_debug, "extract_input", lambda *_: pytest.fail("normal narrative path"))
    result = text_debug.run_text_debug({"input": "snapshot"}, tmp_path, policy="story_draft_v1",
                                      context_size=24576, chapter_limit=2, resume=True, seed=7)
    assert result["saved_chapter_count"] == 2
    assert calls == [({"input": "snapshot"}, tmp_path, {
        "context_size": 24576, "chapter_limit": 2, "resume": True, "seed": 7})]
    with pytest.raises(ValueError, match="causal text-only"):
        text_debug.run_text_debug({}, tmp_path, workflow="legacy", policy="story_draft_v1")


def test_cli_accepts_draft_policy_and_reports_saved_chapters(monkeypatch, tmp_path, capsys):
    source = tmp_path / "input.json"
    source.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_text", "--input", str(source),
        "--output-dir", str(tmp_path), "--policy", "story_draft_v1"])

    def run(_value, _output, **kwargs):
        assert kwargs["policy"] == "story_draft_v1"
        return {"status": "draft_complete", "saved_chapter_count": 3}

    monkeypatch.setattr(check_text, "run_text_debug", run)
    assert check_text.main() == 0
    assert json.loads(capsys.readouterr().out)["chapters"] == 3
