"""Script continuation defaults to Gemma throughout, without requiring Qwen."""

import json
import sys
from copy import deepcopy

import pytest

from scripts.story import check_text
from services.worker.generation.model_routing import resolve_purpose_profile
from services.worker.generation.pipeline import load_config
from services.worker.generation.workflow_version import generator_protocol

PAYLOAD = {"story_workflow_version": 2, "workflow_policy": "script_continuation_v1",
           "execution_mode": "text_only"}


@pytest.mark.parametrize("purpose,category,model,context,reasoning,output", [
    ("script-cast", "script_planning", "gemma", 16384, "none", 8192),
    ("script-outline", "script_planning", "gemma", 16384, "none", 8192),
    ("script-allocation", "script_planning", "gemma", 16384, "none", 2048),
    ("script-plan", "script_planning", "gemma", 16384, "none", 6144),
    ("script-handoff", "script_handoff", "gemma", 16384, "none", 4096),
    ("script-scene", "script_writer", "gemma", 16384, "none", 8192),
    ("script-speech", "script_speech", "gemma", 16384, "none", 8192),
    ("script-staging", "script_staging", "gemma", 16384, "none", 8192),
    ("speech_separation", "script_speech", "gemma", 16384, "none", 8192),
    ("staging", "script_staging", "gemma", 16384, "none", 8192),
    ("scene_text", "script_writer", "gemma", 16384, "none", 8192),
    ("story_outline", "script_planning", "gemma", 16384, "none", 8192),
    ("story_design", "script_planning", "gemma", 16384, "none", 8192),
    ("plot_outline", "script_planning", "gemma", 16384, "none", 8192),
    ("chapter_plan", "script_planning", "gemma", 16384, "none", 6144),
    ("supporting_character", "script_planning", "gemma", 16384, "none", 8192),
])
def test_script_stages_use_expected_models(purpose, category, model, context, reasoning, output):
    selected, profile = resolve_purpose_profile(load_config(), PAYLOAD, purpose)
    assert selected == category and profile["model_id"].startswith(model)
    assert (profile["context_size"], profile["reasoning_level"], profile["max_tokens"]) == (
        context, reasoning, output)


def test_script_stage_overrides_are_scalable_and_purpose_wins():
    config = deepcopy(load_config())
    config["llm"]["context_size"] = 32768
    payload = {**PAYLOAD, "profile": {"max_tokens": 6144}, "profiles": {
        "script_planning": {"max_tokens": 4096},
        "script-plan": {"max_tokens": 5120},
        "script_writer": {"context_size": 24576},
        "script-scene": {"max_tokens": 7168}}}
    planning = resolve_purpose_profile(config, payload, "script-plan")[1]
    writer = resolve_purpose_profile(config, payload, "script-scene")[1]
    assert (planning["context_size"], planning["max_tokens"]) == (32768, 5120)
    assert (writer["context_size"], writer["max_tokens"]) == (24576, 7168)


@pytest.mark.parametrize("purpose", [
    "script-cast",
    "script-outline", "script-plan", "script-handoff", "script-scene", "script-speech",
    "script-staging", "speech_separation", "staging", "scene_text", "story_outline",
    "story_design", "plot_outline", "chapter_plan", "supporting_character",
])
def test_script_works_without_a_review_route(purpose):
    config = deepcopy(load_config())
    config.pop("model_routing")
    _, profile = resolve_purpose_profile(config, PAYLOAD, purpose)
    assert profile["model_id"] == config["llm"]["model_id"]
    assert profile["reasoning_level"] == "none"


def test_script_cannot_change_model_route_with_profile_override():
    with pytest.raises(ValueError, match="configured model route"):
        resolve_purpose_profile(load_config(), {**PAYLOAD, "profiles": {
            "script-scene": {"model_id": "Qwen"}}}, "script-scene")


def test_script_defaults_obey_model_output_allowance():
    config = deepcopy(load_config())
    config["llm"]["max_output_tokens"] = 6144
    config["model_routing"]["review"]["llm"]["max_output_tokens"] = 3072
    assert resolve_purpose_profile(config, PAYLOAD, "script-outline")[1]["max_tokens"] == 6144
    assert resolve_purpose_profile(config, PAYLOAD, "script-scene")[1]["max_tokens"] == 6144


def test_explicit_logical_review_can_still_select_existing_review_route():
    # Routing capability remains; the script runner does not introduce this call.
    _, profile = resolve_purpose_profile(load_config(), PAYLOAD, "continuity_review")
    assert profile["model_id"].startswith("Qwen")


@pytest.mark.parametrize("payload", [{}, {"story_workflow_version": 2},
    {"story_workflow_version": 1, "workflow_policy": "script_continuation_v1"},
    {"story_workflow_version": 2, "workflow_policy": "chapter_editor_v1"},
    {"story_workflow_version": 2, "workflow_policy": "story_draft_v1"}])
def test_new_policy_does_not_change_other_routes(payload):
    config = load_config()
    category, profile = resolve_purpose_profile(config, payload, "script-outline")
    assert category == "script-outline"
    assert profile["model_id"] == config["llm"]["model_id"]
    assert profile["max_tokens"] == config["llm"]["max_tokens"]


def test_script_protocol_is_separate_and_rejects_legacy():
    protocol = generator_protocol("causal", "script_continuation_v1")
    assert protocol["policy"] == "script_continuation_v1"
    assert protocol != generator_protocol("causal", "story_draft_v1")
    protocol["implementation"] += 1
    assert protocol != generator_protocol("causal", "script_continuation_v1")
    with pytest.raises(ValueError, match="unsupported"):
        generator_protocol("legacy", "script_continuation_v1")


def test_cli_selects_script_policy_and_uses_saved_chapter_count(monkeypatch, tmp_path, capsys):
    path = tmp_path / "input.json"
    path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["check_text", "--input", str(path),
        "--output-dir", str(tmp_path), "--policy", "script_continuation_v1"])

    def run(value, output, **kwargs):
        assert kwargs["policy"] == "script_continuation_v1"
        return {"status": "script_complete", "saved_chapter_count": 1}

    monkeypatch.setattr(check_text, "run_text_debug", run)
    assert check_text.main() == 0
    assert json.loads(capsys.readouterr().out)["chapters"] == 1
