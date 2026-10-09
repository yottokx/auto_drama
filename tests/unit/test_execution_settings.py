"""Settings transitions preserve source identities, completed work and budgets."""
import copy
import json

import pytest

from services.worker.generation.causal_runtime import (
    CausalRun,
    PersistentUsage,
    ResourceBudgetError,
    digest,
)
from services.worker.generation.execution_settings import (
    bind_job_request,
    previous_executions,
    rebind_manifest,
    semantic_identity,
    source_config,
    source_payload,
)
from services.worker.generation.llm import write_json
from tests.unit.test_causal_runtime import FakeLLM, StageValue, forbidden, generated


def live(payload, revision=1):
    return {**copy.deepcopy(payload), "execution_settings_version": 1,
            "execution_settings_revision": revision}


def record(work, payload, config):
    write_json(work / "job-request.json", {"kind": "m3_narrative", "payload": payload,
        "generation_config": config, "fingerprint": digest({"kind": "m3_narrative", "payload": payload})})


def test_job_migration_keeps_source_and_prior_settings_while_refreshing_execution(tmp_path):
    original = {"seed": 23, "approval_snapshot": {"world": {"title": "same"}},
                "profile": {"context_size": 16384}}
    config = {"llm": {"context_size": 16384}, "image": {"width": 512}}
    record(tmp_path, original, config)
    changed = live(original)
    changed["profile"]["context_size"] = 32768
    updated = {**config, "llm": {"context_size": 32768}}
    selected, fingerprint, history = bind_job_request(tmp_path, "m3_narrative", changed, updated, 1)
    assert selected == updated
    assert fingerprint == digest({"kind": "m3_narrative", "payload": source_payload(original)})
    assert history[0]["payload"] == original
    assert history[0]["generation_config"] == config
    assert previous_executions(tmp_path, changed, updated) == history


@pytest.mark.parametrize("change", ["seed", "source", "image"])
def test_changed_non_llm_material_never_migrates_a_saved_request(tmp_path, change):
    payload = {"seed": 23, "approval_snapshot": {"world": {"title": "same"}}, "profile": {}}
    config = {"llm": {"context_size": 16384}, "image": {"width": 512}}
    record(tmp_path, payload, config)
    before = (tmp_path / "job-request.json").read_bytes()
    changed, updated = live(payload), copy.deepcopy(config)
    if change == "seed":
        changed["seed"] = 99
    elif change == "source":
        changed["approval_snapshot"]["world"]["title"] = "different"
    else:
        updated["image"]["width"] = 1024
    with pytest.raises(ValueError, match="another input snapshot"):
        bind_job_request(tmp_path, "m3_narrative", changed, updated, 1)
    assert (tmp_path / "job-request.json").read_bytes() == before


def test_automatic_retry_pins_settings_and_explicit_revision_is_required(tmp_path):
    payload = live({"seed": 1, "profile": {"context_size": 16384}}, 0)
    config = {"llm": {"context_size": 16384}}
    bind_job_request(tmp_path, "m3_narrative", payload, config)
    assert bind_job_request(tmp_path, "m3_narrative", payload,
                            {"llm": {"context_size": 32768}})[0] == config
    changed = copy.deepcopy(payload)
    changed["profile"]["context_size"] = 32768
    with pytest.raises(ValueError, match="explicit retry"):
        bind_job_request(tmp_path, "m3_narrative", changed, config)
    with pytest.raises(ValueError, match="match the explicit retry"):
        bind_job_request(tmp_path, "m3_narrative", live(payload, 2), config, 1)


def manifests():
    old = {"seed": 7, "approval_snapshot": {"title": "same"}, "profile": {"context_size": 16384},
           "generation_config": {"llm": {"context_size": 16384}, "image": {"width": 512}}}
    old["input_sha256"] = digest(old)
    current = live({key: value for key, value in old.items() if key != "input_sha256"})
    current["profile"]["context_size"] = 32768
    current["generation_config"]["llm"]["context_size"] = 32768
    current["input_sha256"] = digest(semantic_identity(current))
    return old, current


def test_manifest_rebind_preserves_completed_work_and_recovers_partial_migration(tmp_path):
    old, current = manifests()
    cast = {"plan": {"supporting_characters": []}, "input_sha256": old["input_sha256"],
            "sha256": digest({"supporting_characters": []})}
    steps = {"accepted": {"attempts": [{"status": "completed", "charged_tokens": 321,
                                         "profile": {"context_size": 16384}, "reply": "saved text"}]}}
    journal = {"input_sha256": old["input_sha256"], "cast_plan": cast,
               "request_ordinal": 4, "steps": steps, "sessions": [{"elapsed_seconds": 12.0}]}
    write_json(tmp_path / "draft-state.json", journal)
    write_json(tmp_path / "cast-plan.json", cast)
    migrated = rebind_manifest(tmp_path, old, current)
    # Retry with the old manifest after refs already migrated, simulating a crash.
    assert rebind_manifest(tmp_path, old, current) == migrated
    actual = json.loads((tmp_path / "draft-state.json").read_text(encoding="utf-8"))
    assert actual["steps"] == steps and actual["request_ordinal"] == 4
    assert actual["sessions"] == journal["sessions"]
    assert actual["cast_plan"] == json.loads((tmp_path / "cast-plan.json").read_text(encoding="utf-8"))
    assert actual["input_sha256"] == current["input_sha256"]
    assert actual["cast_plan"]["source_input_sha256"] == old["input_sha256"]
    assert migrated["execution_history"] == [old]


def test_manifest_rebind_rejects_unknown_journal_and_changed_source(tmp_path):
    old, current = manifests()
    write_json(tmp_path / "draft-state.json", {"input_sha256": "another-source", "steps": {}})
    before = (tmp_path / "draft-state.json").read_bytes()
    with pytest.raises(ValueError, match="another experiment"):
        rebind_manifest(tmp_path, old, current)
    assert (tmp_path / "draft-state.json").read_bytes() == before
    current["seed"] = 99
    current["input_sha256"] = digest(semantic_identity(current))
    with pytest.raises(ValueError, match="source inputs changed"):
        rebind_manifest(tmp_path, old, current)


def test_causal_legacy_retry_preserves_accepted_stage_and_measured_budgets(tmp_path):
    payload = {"seed": 7, "profile": {"context_size": 16384}}
    first = FakeLLM(tmp_path)
    first.output.mkdir()
    record(tmp_path, payload, first.config)
    original_runtime = first.runtime_identity()
    with CausalRun(first, payload) as run:
        run.node("accepted", {"source": "same"}, StageValue, lambda: generated(first))
        run.repair("accepted", "saved issue")
    write_json(first.output / "01-accepted-request.json", {"runtime": original_runtime})
    previous_state = json.loads((tmp_path / "causal-stages/budget.json").read_text(encoding="utf-8"))
    previous_usage = json.loads((tmp_path / "causal-stages/resources.json").read_text(encoding="utf-8"))
    changed = live(payload)
    changed["profile"]["context_size"] = 32768
    resumed = FakeLLM(tmp_path)
    resumed.config["llm"]["temperature"] = .9
    resumed.profile = {"temperature": .9}
    resumed.server_context_size = 32768
    bind_job_request(tmp_path, "m3_narrative", changed, resumed.config, 1)
    with CausalRun(resumed, changed) as run:
        assert run.node("accepted", {"source": "same"}, StageValue, forbidden).value == 1
        assert resumed.profile == {"temperature": .9}
        assert resumed.requests == 1
        assert run.state["calls"] == previous_state["calls"]
        assert run.state["repairs"] == previous_state["repairs"]
        assert run.state["issues"] == previous_state["issues"]
        assert run.usage.data["requests"] == previous_usage["requests"]
        assert run.usage.data["attempts"][:len(previous_usage["attempts"])] == previous_usage["attempts"]


def test_settings_migration_does_not_turn_unknown_usage_into_zero(tmp_path):
    payload = {"seed": 7, "profile": {"context_size": 16384}}
    config = {"llm": {"temperature": .6}}
    record(tmp_path, payload, config)
    original = PersistentUsage(tmp_path, payload, config)
    original.begin_request()
    before = copy.deepcopy(original.data["requests"])
    changed = live(payload)
    changed["profile"]["context_size"] = 32768
    updated = {"llm": {"temperature": .9}}
    bind_job_request(tmp_path, "m3_narrative", changed, updated, 1)
    migrated = PersistentUsage(tmp_path, changed, updated,
        previous_executions=previous_executions(tmp_path, changed, updated))
    assert migrated.data["requests"] == before
    with pytest.raises(ResourceBudgetError, match="unmeasured"):
        migrated.check()
    assert source_config(config) == source_config(updated)


@pytest.mark.parametrize("kind", ["m3_event_cg_budget", "m3_event_cg_plan"])
def test_legacy_cg_planner_without_config_retains_exact_identity_evidence(tmp_path, kind):
    payload = {"seed": 23, "context": {"chapter_number": 3}, "profile": {"context_size": 16384}}
    previous = {"kind": kind, "payload": payload, "fingerprint": digest({"kind": kind, "payload": payload}),
                "model_identity": {"generation": {"settings": {"context_size": 16384}, "base": {"model": "old"}}}}
    write_json(tmp_path / "job-request.json", previous)
    changed = live(payload)
    changed["profile"]["context_size"] = 32768
    config = {"llm": {"context_size": 32768}, "event_cg": {"model": "unchanged"}}
    actual, _, history = bind_job_request(tmp_path, kind, changed, config, 1)
    assert actual == config and history == [previous]
    assert "generation_config" not in history[0]
    assert previous_executions(tmp_path, changed, config) == []


def test_music_planner_registry_is_execution_settings_but_render_identity_is_source():
    old = {"llm": {"context_size": 16384}, "music_job_identity": {"generation": {"model": "old"}}}
    current = {"llm": {"context_size": 32768}, "music_job_identity": {"generation": {"model": "new"}}}
    assert source_config(old) == source_config(current)
    assert source_config({"music_job_identity": "render-model-a"}) != source_config(
        {"music_job_identity": "render-model-b"})
