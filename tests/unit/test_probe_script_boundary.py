"""The boundary diagnostic uses one frozen chapter and stops after at most two calls."""

import copy
import json

import pytest

from scripts.story import probe_script_boundary as probe
from services.worker.generation.llm import ContextBudgetError
from services.worker.generation.script_cast import ScriptOptions
from services.worker.generation.script_plot import ChapterAllocation, StoryChain, allocate_chain
from tests.unit.test_script_continuation_run import (
    FIRST,
    SECOND,
    allocation,
    chapter_plan,
    snapshot,
    story_chain,
)

PROFILES = {purpose: {"model_id": "gemma-test", "context_size": 16384, "reasoning_level": "none",
                      "max_tokens": tokens} for purpose, tokens in (("script-plan", 6144), ("script-scene", 8192))}


def source_data(text=FIRST):
    setting = snapshot()
    setting["world"]["result"]["chapterCount"] = 3
    options = ScriptOptions().model_dump()
    manifest = {"script_options": options, "approved_chapter_count": 3}
    payload = {"approval_snapshot": setting, "script_options": options}
    plot = allocate_chain(StoryChain.model_validate(story_chain()), ChapterAllocation.model_validate(allocation()))
    attempt = {"status": "completed", "purpose": "script-scene", "profile": PROFILES["script-scene"],
               "reply": {"content": text, "_finish_reason": "stop"}, "usage": {"completion_tokens": 100}}
    state = {"cast_plan": {"plan": {"supporting_characters": [], "everyday_context": [], "connections": []}},
             "plot": plot.model_dump(), "plot_sha256": "saved-plot", "chapters": [
                 {"number": 1, "text": text, "sha256": probe.digest(text), "scene_starts": [0],
                  "narrative": {"supporting_characters": [], "locations": chapter_plan("s1")["locations"]}},
                 {"number": 2, "text": "FUTURE_HISTORY"}],
             "notes": [{"number": 1, "text": "KEPT_NOTE"}, {"number": 2, "text": "FUTURE_NOTE"}],
             "locations": {"future": {"name": "FUTURE_LOCATION"}},
             "steps": {"c001-s1-text": {"attempts": [attempt]},
                       "c002-s1-text": {"attempts": [{**attempt, "usage": {"completion_tokens": 999999}}]}},
             "chapter_plans": {"2": {"plan": "FUTURE_PLAN"}},
             "scene_budgets": {"c002-s1": {"budget": "FUTURE_BUDGET"}},
             "scene_commits": {"c002-s1": {"text": "FUTURE_COMMIT"}}, "partial": "FUTURE_PARTIAL"}
    return manifest, state, payload


def prepared(text=FIRST):
    manifest, state, payload = source_data(text)
    reader, supporting = probe.prepare_reader(manifest, state, payload, {})
    stage = probe.prepare_plan(reader, supporting, 18, PROFILES["script-plan"])
    return reader, supporting, stage


def output_dir(tmp_path):
    for name in ("requests", "responses"):
        (tmp_path / name).mkdir()
    return tmp_path


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests, self._load_number, self.trace = 0, 1, []
        self.server_context_size, self.selected_route = 16384, "generation"

    def select_purpose(self, purpose):
        self.profile = PROFILES[purpose]

    def _ensure_runtime(self):
        pass

    def _chat_request(self, number, messages, extra):
        request = {"messages": messages, "max_tokens": self.profile["max_tokens"], **extra}
        return request, probe.digest(request)

    def check_context(self, stage, request):
        return {"prompt_tokens": 500, "output_tokens": request["max_tokens"], "margin_tokens": 512,
                "context_size": 16384, "fits": True}

    def runtime_identity(self):
        return {"profile": self.profile}

    def _output_limit(self, profile):
        return 8192

    def chat(self, purpose, messages, **extra):
        self.requests += 1
        self.trace.append({"type": "llm_generation", "usage": {"completion_tokens": 30}})
        response = self.responses.pop(0)
        if isinstance(response, dict) and "_finish_reason" in response:
            return response
        return {"content": response if isinstance(response, str) else json.dumps(response, ensure_ascii=False),
                "_finish_reason": "stop"}


def test_material_and_output_samples_use_only_chapter_one_without_mutating_the_baseline():
    manifest, state, payload = source_data()
    original = copy.deepcopy(state)
    reader, supporting = probe.prepare_reader(manifest, state, payload, {})
    stage = probe.prepare_plan(reader, supporting, 18, PROFILES["script-plan"])
    validated = stage.validate(json.dumps(chapter_plan("s1", "s2", continued=True)))
    scene = probe.prepare_scene(reader, supporting, validated["plan"], 19, PROFILES["script-scene"], 8192)
    assert state == original
    assert "FUTURE_" not in str(reader.state) + str(stage.material()) + str(scene.material())
    assert [row["number"] for row in stage.context["chapters"]] == [1]
    assert stage.context["chapters"][0]["text"] == FIRST
    assert stage.extra["seed"] == 18 and scene.extra["seed"] == 19
    assert [row["step"] for row in scene.output_estimate["samples"]] == ["c001-s1-text"]
    assert scene.output_estimate["samples"][0]["completion_tokens"] == 100
    assert scene.context["planned_connection"] == ""
    first_event = validated["plan"]["scenes"][0]["required_events"][0]["description"]
    sent_scene = json.loads(scene.context["brief"].split("今回の場面計画（未実施の予定）:\n", 1)[1]
                           .split("\n執筆範囲と後続場面の担当", 1)[0])
    assert first_event.startswith("【最初の新行動】")
    assert sent_scene["required_events"][0]["description"] == first_event
    assert scene.context["brief"].count("【最初の新行動】") == 1


def test_two_single_shots_keep_requests_seed_provenance_and_the_generated_plan(tmp_path):
    reader, supporting, plan = prepared()
    output_dir(tmp_path)
    llm = FakeLLM([chapter_plan("s1", "s2", continued=True), SECOND])
    rows = probe.execute(llm, tmp_path, reader, supporting, plan, 19, PROFILES["script-scene"], {})
    assert len(rows) == llm.requests == 2 and all(row["status"] == "completed" for row in rows)
    assert not llm.responses
    report = probe.read(tmp_path / "report.json")
    assert report["status"] == "complete" and report["generation_calls"] == 2
    writer = probe.read(tmp_path / "requests/chapter-002-opening.json")
    assert writer["request"]["seed"] == 19 and writer["selection"]["current_chapter"] == 2
    assert (tmp_path / "responses/chapter-002-opening.raw.txt").read_text(encoding="utf-8") == SECOND
    assert (tmp_path / "responses/chapter-002-opening.effective.txt").read_text(encoding="utf-8") == SECOND
    assert not rows[1]["overlap"]["changed"]
    assert not (tmp_path / "exports").exists()


@pytest.mark.parametrize("response", [
    {"content": "partial", "_finish_reason": "length"},
    {"content": "", "_finish_reason": "stop"},
    {"content": "not a plan", "_finish_reason": "stop"},
])
def test_plan_failure_saves_raw_and_never_dispatches_a_writer(tmp_path, response):
    reader, supporting, plan = prepared()
    output_dir(tmp_path)
    llm = FakeLLM([response, SECOND])
    rows = probe.execute(llm, tmp_path, reader, supporting, plan, 19, PROFILES["script-scene"], {})
    assert len(rows) == llm.requests == 1 and rows[0]["status"] == "failed"
    assert len(llm.responses) == 1
    assert (tmp_path / "responses/chapter-002-plan.raw.txt").read_text(encoding="utf-8") == response["content"]
    assert not (tmp_path / "writer-input.json").exists()


@pytest.mark.parametrize("new_text", [SECOND, ""])
def test_exact_copy_keeps_raw_and_metrics_and_stops_if_no_new_text_remains(tmp_path, new_text):
    prior = "\n".join("aoi: " + str(number) + "前章で既に話した内容。" * 8 for number in range(6))
    reader, supporting, plan = prepared(prior)
    output_dir(tmp_path)
    generated = prior + "\n" + new_text
    llm = FakeLLM([chapter_plan("s1", continued=True), generated])
    rows = probe.execute(llm, tmp_path, reader, supporting, plan, 19, PROFILES["script-scene"], {})
    row = rows[-1]
    assert llm.requests == 2 and row["overlap"]["removed_utterances"] == 6
    assert row["status"] == ("completed" if new_text else "failed")
    assert row["raw_metrics"]["body_characters"] > row["effective_metrics"]["body_characters"]
    assert (tmp_path / "responses/chapter-002-opening.raw.txt").read_text(encoding="utf-8") == generated
    assert (tmp_path / "responses/chapter-002-opening.effective.txt").read_text(encoding="utf-8") == new_text


def test_capacity_failure_stops_before_dispatch_and_preserves_budget(monkeypatch, tmp_path):
    reader, supporting, plan = prepared()
    output_dir(tmp_path)

    def blocked(*args, **kwargs):
        error = ContextBudgetError("request exceeds 16k")
        error.selection = {"budget": {"fits": False}, "example": {"included": False}}
        raise error

    monkeypatch.setattr(probe, "fit_context", blocked)
    llm = FakeLLM([chapter_plan("s1", continued=True)])
    rows = probe.execute(llm, tmp_path, reader, supporting, plan, 19, PROFILES["script-scene"], {})
    assert rows[0]["status"] == "failed" and not rows[0]["dispatched"] and llm.requests == 0
    assert rows[0]["selection"]["budget"]["fits"] is False


def test_optional_example_omission_is_allowed_like_production(monkeypatch, tmp_path):
    _, _, plan = prepared()
    output_dir(tmp_path)
    original = probe.fit_context

    def omit(*args, **kwargs):
        kwargs["optional_example"] = None
        messages, selection = original(*args, **kwargs)
        selection["example"] = {"included": False, "reason": "context_budget"}
        return messages, selection

    monkeypatch.setattr(probe, "fit_context", omit)
    llm = FakeLLM([chapter_plan("s1", continued=True)])
    row = probe.run_stage(llm, tmp_path, plan)
    assert row["status"] == "completed" and llm.requests == 1
    assert row["selection"]["example"]["included"] is False
