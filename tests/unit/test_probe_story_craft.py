"""The diagnostic freezes six inputs and never repairs or resamples a failure."""

import copy
import json

import pytest

from scripts.story import probe_story_craft as probe
from services.worker.generation.script_cast import ScriptOptions
from services.worker.generation.script_plot import (
    CHAIN_INSTRUCTION,
    ChapterAllocation,
    StoryChain,
    StoryChainDraft,
    allocate_chain,
)
from tests.unit.test_script_continuation_run import allocation, chapter_plan, snapshot, story_chain


def test_six_conditions_share_pair_material_seed_schema_and_do_not_mutate_baseline(tmp_path):
    requests = tmp_path / "requests"
    requests.mkdir()
    for key, request in {
        "story-chain": {"seed": 3},
        "plan-001": {"seed": 5, "messages": [{"content": "OLD INSTRUCTION" + probe.SOURCE_RULES}]},
        "c001-s1-text": {"seed": 6, "max_tokens": 6144},
    }.items():
        probe.write_json(requests / f"{key}-1.json", {"request": request})
    setting = snapshot()
    setting["world"]["result"]["chapterCount"] = 3
    options = ScriptOptions().model_dump()
    manifest = {"script_options": options, "approved_chapter_count": 3}
    payload = {"approval_snapshot": setting, "script_options": options}
    plot = allocate_chain(StoryChain.model_validate(story_chain()), ChapterAllocation.model_validate(allocation()))
    state = {"cast_plan": {"plan": {"supporting_characters": [], "everyday_context": [], "connections": []}},
             "plot": plot.model_dump(), "plot_sha256": "saved-plot", "chapters": [{"number": 9, "text": "FUTURE_HISTORY"}],
             "notes": [{"number": 9, "text": "FUTURE_NOTE"}], "locations": {"future": {"name": "FUTURE_LOCATION"}},
             "steps": {"plan-001": {"attempts": [{"status": "completed", "reply": {"content": json.dumps(chapter_plan("s1", "s2"))}}]}}}
    original = copy.deepcopy(state)
    conditions = probe.prepare_conditions(tmp_path, manifest, state, payload, {})
    assert len(conditions) == 6 and state == original
    for left, right in zip(conditions[::2], conditions[1::2], strict=True):
        assert left.extra["seed"] == right.extra["seed"]
        assert "FUTURE_HISTORY" not in str(left.material()) + str(right.material())
        assert "FUTURE_NOTE" not in str(left.material()) + str(right.material())
    assert conditions[0].context == conditions[1].context
    assert conditions[2].context == conditions[3].context
    assert conditions[2].extra == conditions[3].extra
    assert conditions[2].system == "OLD INSTRUCTION"
    assert conditions[4].system == conditions[5].system and conditions[4].extra == conditions[5].extra
    left, right = copy.deepcopy(conditions[4].context), copy.deepcopy(conditions[5].context)
    assert left.pop("optional_example") is None
    assert right.pop("optional_example")["id"] == "script-exchange"
    assert left == right


def test_basis_ablation_keeps_all_other_schema_properties_and_setting_guard():
    schema = StoryChainDraft.model_json_schema()
    original = copy.deepcopy(schema)
    prompt, removed = probe.without_basis(CHAIN_INSTRUCTION, schema)
    assert schema == original
    assert "resolution_basis" not in prompt and "下書き" not in prompt
    assert "承認設定と矛盾する力や条件を便利な決着のために追加せず" in prompt
    assert "resolution_basis" not in removed["properties"]
    assert {key: value for key, value in schema["properties"].items() if key != "resolution_basis"} == removed["properties"]


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.requests, self.trace = reply, 0, []
        self.server_context_size, self.selected_route = 16384, "generation"
        self.profile = {"context_size": 16384, "reasoning_level": "none"}

    def select_purpose(self, purpose):
        pass

    def _ensure_runtime(self):
        pass

    def _chat_request(self, number, messages, extra):
        return {"messages": messages, **extra}, "frozen-request"

    def runtime_identity(self):
        return {"profile": self.profile}

    def chat(self, purpose, messages, **extra):
        self.requests += 1
        self.trace.append({"type": "llm_generation", "usage": {"completion_tokens": 30}})
        return self.reply


@pytest.mark.parametrize("reply", [
    {"content": "retained partial", "_finish_reason": "length"},
    {"content": "not JSON", "_finish_reason": "stop"},
])
def test_failed_response_is_saved_and_never_retried(monkeypatch, tmp_path, reply):
    for name in ("requests", "responses"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(probe, "fit_context", lambda *args, **kwargs: ([{"role": "user", "content": "same"}], {"budget": {}}))
    condition = probe.Condition("test", "script-plan", "system", {}, {"seed": 7}, json.loads, "single shot")
    llm = FakeLLM(reply)
    result = probe.run_condition(llm, tmp_path, condition)
    assert result["status"] == "failed" and result["dispatched"] and llm.requests == 1
    assert probe.read(tmp_path / "responses/test.json")["reply"] == reply
    assert (tmp_path / "responses/test.txt").read_text(encoding="utf-8") == reply["content"]


def test_dropped_example_is_not_counted_as_an_example_comparison(monkeypatch, tmp_path):
    for name in ("requests", "responses"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(probe, "fit_context", lambda *args, **kwargs: ([], {"budget": {}, "example": {"included": False}}))
    condition = probe.Condition("test", "script-scene", "system", {"optional_example": {"id": "test"}}, {}, json.loads, "single shot")
    llm = FakeLLM({"content": "should not be called", "_finish_reason": "stop"})
    result = probe.run_condition(llm, tmp_path, condition)
    assert result["status"] == "failed" and not result["dispatched"] and llm.requests == 0
    assert "omitted for capacity" in result["error"]
