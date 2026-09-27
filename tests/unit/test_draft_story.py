"""Draft writing continues from saved prose without semantic acceptance gates."""

import copy
import json
from contextlib import nullcontext
from pathlib import Path

import pytest

from services.worker.generation import draft_story
from services.worker.generation.cancellation import GenerationCancelled
from services.worker.generation.llm import write_json
from services.worker.generation.model_routing import RoutedLLM

OUTLINE = "中心の問題は未確認の写真。第1章で空欄を選び、第2章で配置を変え、第3章で完成を決める。"
CHAPTERS = [
    "葵はゆず色の付箋を裏返した。『年を空欄にして、この写真を出そう』。蓮は手を止めた。",
    "蓮は前章で残した空欄の隣に、紫の小さな紙を置いた。葵は説明札を短く書き直した。",
    "二人は最後の写真を左へ移した。葵が紙の端を押さえ、蓮は展示室の灯りを消した。",
]
HANDOFFS = [
    "実際に起きたこと：葵が年を空欄にすると決めた。次章は配置を見直し、決定をやり直さない。",
    "実際に起きたこと：蓮が紫の紙を置き、葵が説明札を書き直した。最終章で作業を終える。",
]


def snapshot():
    return json.loads((Path(__file__).parents[1] / "fixtures" / "story-debug-relationship.json")
                      .read_text(encoding="utf-8"))


def read_report(output):
    return json.loads((output / "report.json").read_text(encoding="utf-8"))


def prompt_text(call):
    return "\n".join(message.get("content") or "" for message in call["messages"])


@pytest.fixture
def runtime(monkeypatch):
    control = {"responses": [OUTLINE, CHAPTERS[0], HANDOFFS[0], CHAPTERS[1],
                              HANDOFFS[1], CHAPTERS[2]], "before_chat": None,
               "unmeasured_errors": False}
    calls = []

    class FakeLLM(RoutedLLM):
        def select_purpose(self, purpose):
            super().select_purpose(purpose)
            self.purpose = purpose

        def _ensure_runtime(self):
            pass

        def check_context(self, stage, request):
            budget = self.context_budget(prompt_tokens=500, output_tokens=request["max_tokens"])
            self.trace.append({"type": "context_budget", "stage": stage, **budget})
            return budget

        def chat(self, stage, messages, *, allow_truncated=False, **extra):
            assert allow_truncated, "The runner must retain output that reaches the token limit."
            call = {"stage": stage, "purpose": self.purpose, "messages": copy.deepcopy(messages),
                    "profile": copy.deepcopy(self.profile), "extra": copy.deepcopy(extra)}
            calls.append(call)
            if control["before_chat"] is not None:
                control["before_chat"](call)
            assert control["responses"], f"Unexpected additional LLM call: {stage}"
            item = control["responses"].pop(0)
            self.requests += 1
            if not (isinstance(item, BaseException) and control["unmeasured_errors"]):
                self.trace.append({"type": "llm_generation", "stage": stage,
                    "request": self.requests, "cache_hit": False, "elapsed_seconds": 0.01,
                    "usage": {"prompt_tokens": 500, "completion_tokens": 20}})
            if isinstance(item, BaseException):
                raise item
            return {"content": item, "_finish_reason": "stop"} if isinstance(item, str) else item

        def __exit__(self, *args):
            write_json(self.output / "llm-metrics.json", self.trace)
            self._entered = False

    monkeypatch.setattr(draft_story, "RoutedLLM", FakeLLM)
    monkeypatch.setattr(draft_story, "gpu_lock", lambda *_args, **_kwargs: nullcontext())
    return calls, control


def test_three_chapters_use_six_requests_and_pass_real_prose_to_the_next_chapter(runtime, tmp_path):
    calls, _ = runtime
    report = draft_story.run_draft_debug(snapshot(), tmp_path)
    assert report["status"] == "draft_complete"
    assert report["saved_chapter_count"] == 3
    assert report["quality_acceptance"] == "not_evaluated"
    assert [call["purpose"] for call in calls] == [
        "draft-outline", "draft-chapter", "draft-handoff", "draft-chapter",
        "draft-handoff", "draft-chapter"]
    assert CHAPTERS[0] in prompt_text(calls[2])
    assert CHAPTERS[0] in prompt_text(calls[3])
    assert CHAPTERS[1] in prompt_text(calls[4])
    assert CHAPTERS[1] in prompt_text(calls[5])
    for number, content in enumerate(CHAPTERS, 1):
        assert content in (tmp_path / "chapters" / f"chapter-{number:03d}.md").read_text(encoding="utf-8")
    assert OUTLINE in (tmp_path / "outline.md").read_text(encoding="utf-8")
    assert HANDOFFS[0] in (tmp_path / "notes" / "chapter-001.md").read_text(encoding="utf-8")
    story = (tmp_path / "story.md").read_text(encoding="utf-8")
    assert all(content in story for content in CHAPTERS)
    assert story.index(CHAPTERS[0]) < story.index(CHAPTERS[1]) < story.index(CHAPTERS[2])


def test_free_text_with_inconsistencies_is_saved_without_reviews(runtime, tmp_path):
    calls, control = runtime
    prose = "時計は十九時四十分だった。次の瞬間、十九時半になった。<script>alert('本文')</script>"
    control["responses"] = ["見出しも配列もない自由な構成案。", prose]
    report = draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1)
    assert report["status"] == "chapter_limit_reached"
    assert report["saved_chapter_count"] == 1
    assert len(calls) == 2
    assert prose in (tmp_path / "story.md").read_text(encoding="utf-8")
    page = (tmp_path / "story.html").read_text(encoding="utf-8")
    assert "&lt;script&gt;" in page and "<script>" not in page


@pytest.mark.parametrize("chapter_count", [1, 4])
def test_approved_chapter_count_controls_the_loop(runtime, tmp_path, chapter_count):
    calls, control = runtime
    source = snapshot()
    source["world"]["result"]["chapterCount"] = chapter_count
    responses = [OUTLINE, "第1章の本文。"]
    for number in range(2, chapter_count + 1):
        responses.extend([f"第{number - 1}章の実績メモ。", f"第{number}章の本文。"])
    control["responses"] = responses
    report = draft_story.run_draft_debug(source, tmp_path)
    assert report["status"] == "draft_complete"
    assert report["saved_chapter_count"] == chapter_count and len(calls) == 2 * chapter_count


def test_empty_handoff_retries_once_then_continues_from_actual_prose(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [OUTLINE, CHAPTERS[0], "", "", CHAPTERS[1], HANDOFFS[1], CHAPTERS[2]]
    report = draft_story.run_draft_debug(snapshot(), tmp_path)
    assert report["status"] == "draft_complete" and report["saved_chapter_count"] == 3
    assert [call["purpose"] for call in calls[:5]] == [
        "draft-outline", "draft-chapter", "draft-handoff", "draft-handoff", "draft-chapter"]
    assert CHAPTERS[0] in prompt_text(calls[4])
    assert len(calls) == 7


def test_cancellation_then_resume_preserves_the_completed_chapter(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [OUTLINE, CHAPTERS[0], GenerationCancelled("cancelled for test")]
    with pytest.raises(GenerationCancelled):
        draft_story.run_draft_debug(snapshot(), tmp_path)
    saved = tmp_path / "chapters" / "chapter-001.md"
    before, changed = saved.read_bytes(), saved.stat().st_mtime_ns
    assert read_report(tmp_path)["status"] == "interrupted"
    assert CHAPTERS[0] in (tmp_path / "story.md").read_text(encoding="utf-8")
    first_calls = len(calls)
    control["responses"] = [HANDOFFS[0], CHAPTERS[1], HANDOFFS[1], CHAPTERS[2]]
    report = draft_story.run_draft_debug(snapshot(), tmp_path, resume=True)
    assert report["status"] == "draft_complete"
    assert saved.read_bytes() == before and saved.stat().st_mtime_ns == changed
    assert [call["purpose"] for call in calls[first_calls:]] == [
        "draft-handoff", "draft-chapter", "draft-handoff", "draft-chapter"]


def test_resume_keeps_an_exhausted_optional_handoff_in_fallback(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [OUTLINE, CHAPTERS[0], "", "", GenerationCancelled("writer cancelled")]
    with pytest.raises(GenerationCancelled):
        draft_story.run_draft_debug(snapshot(), tmp_path)
    assert [call["purpose"] for call in calls].count("draft-handoff") == 2
    first_calls = len(calls)
    control["responses"] = [CHAPTERS[1], HANDOFFS[1], CHAPTERS[2]]
    report = draft_story.run_draft_debug(snapshot(), tmp_path, resume=True)
    assert report["status"] == "draft_complete"
    assert [call["purpose"] for call in calls[first_calls:]] == [
        "draft-chapter", "draft-handoff", "draft-chapter"]
    assert CHAPTERS[0] in prompt_text(calls[first_calls])


def test_truncated_prose_is_saved_before_one_continuation(runtime, tmp_path):
    calls, control = runtime
    first, continuation = "葵はまだ白い札を持ち上げ、", "蓮の側へ置いた。二人は次の作業を決めた。"
    control["responses"] = [OUTLINE, {"content": first, "_finish_reason": "length"}, continuation]

    def before_chat(call):
        if len(calls) == 3:
            assert any(first in path.read_text(encoding="utf-8")
                       for path in tmp_path.rglob("*.md")), "A truncated draft must be saved before continuing."
            assert first in prompt_text(call)

    control["before_chat"] = before_chat
    report = draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1)
    assert report["saved_chapter_count"] == 1 and len(calls) == 3
    saved = (tmp_path / "chapters" / "chapter-001.md").read_text(encoding="utf-8")
    assert first in saved and continuation in saved
    assert saved.index(first) < saved.index(continuation)


def test_a_second_truncation_preserves_partial_text_and_stops(runtime, tmp_path):
    calls, control = runtime
    first, continuation = "最初の途中原稿。", "一度だけ続筆した途中原稿。"
    control["responses"] = [OUTLINE, {"content": first, "_finish_reason": "length"},
                              {"content": continuation, "_finish_reason": "length"}]
    with pytest.raises(RuntimeError):
        draft_story.run_draft_debug(snapshot(), tmp_path)
    report = read_report(tmp_path)
    assert report["status"] == "failed" and report["saved_chapter_count"] == 0
    story = (tmp_path / "story.md").read_text(encoding="utf-8")
    assert first in story and continuation in story
    assert len(calls) == 3
    with pytest.raises(RuntimeError):
        draft_story.run_draft_debug(snapshot(), tmp_path, resume=True)
    assert len(calls) == 3, "Resume must not grant a second continuation."


def test_failed_stage_retry_limit_survives_resume(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = ["", ""]
    with pytest.raises(RuntimeError):
        draft_story.run_draft_debug(snapshot(), tmp_path)
    assert len(calls) == 2 and read_report(tmp_path)["saved_chapter_count"] == 0
    control["responses"] = [OUTLINE]
    with pytest.raises(RuntimeError):
        draft_story.run_draft_debug(snapshot(), tmp_path, resume=True)
    assert len(calls) == 2


def test_story_token_budget_remains_spent_after_resume(runtime, tmp_path):
    calls, _ = runtime
    source = {"approval_snapshot": snapshot(), "workflow_limits": {"max_story_tokens": 1500},
              "profiles": {purpose: {"max_tokens": 256}
                           for purpose in ("draft_outline", "draft_handoff", "draft_writer")}}
    with pytest.raises(RuntimeError):
        draft_story.run_draft_debug(source, tmp_path)
    assert len(calls) == 2
    assert read_report(tmp_path)["saved_chapter_count"] == 1
    with pytest.raises(RuntimeError):
        draft_story.run_draft_debug(source, tmp_path, resume=True)
    assert len(calls) == 2
    assert CHAPTERS[0] in (tmp_path / "story.md").read_text(encoding="utf-8")


def test_chapter_limit_resume_writes_only_remaining_chapters(runtime, tmp_path):
    calls, _ = runtime
    report = draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1)
    assert report["status"] == "chapter_limit_reached" and len(calls) == 2
    saved = (tmp_path / "chapters" / "chapter-001.md").read_bytes()
    report = draft_story.run_draft_debug(snapshot(), tmp_path, resume=True)
    assert report["status"] == "draft_complete" and len(calls) == 6
    assert (tmp_path / "chapters" / "chapter-001.md").read_bytes() == saved
    report = draft_story.run_draft_debug(snapshot(), tmp_path, resume=True)
    assert report["status"] == "draft_complete" and len(calls) == 6


def test_changed_setting_or_saved_chapter_refuses_resume(runtime, tmp_path):
    calls, _ = runtime
    source = snapshot()
    draft_story.run_draft_debug(source, tmp_path, chapter_limit=1)
    changed = copy.deepcopy(source)
    changed["world"]["result"]["setting"] += "この実験では閉館時刻が変わる。"
    with pytest.raises(ValueError):
        draft_story.run_draft_debug(changed, tmp_path, resume=True)
    assert len(calls) == 2
    (tmp_path / "chapters" / "chapter-001.md").write_text("別の本文。", encoding="utf-8")
    with pytest.raises(ValueError):
        draft_story.run_draft_debug(source, tmp_path, resume=True)
    assert len(calls) == 2


@pytest.mark.parametrize("request_status", ["pending", "interrupted"])
def test_resume_recovers_an_uncommitted_request_from_its_saved_response(runtime, tmp_path, request_status):
    calls, control = runtime
    control["responses"] = [OUTLINE, GenerationCancelled("simulate writer interruption")]
    with pytest.raises(GenerationCancelled):
        draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1)

    state_path = tmp_path / "draft-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    step = state["steps"]["chapter-001"]
    attempt = step["attempts"][0]
    attempt["status"] = request_status
    for key in ("usage", "charged_tokens", "reply"):
        attempt.pop(key, None)
    step.pop("result", None)
    state["sessions"][-1]["status"] = "running"
    write_json(tmp_path / attempt["cache_file"], {
        "request_sha256": attempt["request_sha256"],
        "response": {"choices": [{"message": {"content": CHAPTERS[0]}, "finish_reason": "stop"}],
                     "usage": {"prompt_tokens": 500, "completion_tokens": 20}},
    })
    write_json(state_path, state)
    control["responses"] = []

    report = draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert report["status"] == "chapter_limit_reached" and report["saved_chapter_count"] == 1
    assert len(calls) == 2, "An already saved server response must not be generated again."
    assert (tmp_path / "chapters" / "chapter-001.md").read_text(encoding="utf-8") == CHAPTERS[0]
    recovered = json.loads(state_path.read_text(encoding="utf-8"))["steps"]["chapter-001"]
    assert recovered["attempts"][0]["status"] == "completed"
    assert recovered["attempts"][0]["charged_tokens"] == 520


@pytest.mark.parametrize("override", [
    {"profile": {"temperature": 0.25}},
    {"profiles": {"draft-handoff": {"reasoning_level": "low"}}},
])
def test_resume_refuses_changed_generation_profiles(runtime, tmp_path, override):
    calls, _ = runtime
    draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1)
    changed = {"approval_snapshot": snapshot(), **override}
    with pytest.raises(ValueError, match="changed"):
        draft_story.run_draft_debug(changed, tmp_path, resume=True)
    assert len(calls) == 2


def test_unmeasured_transport_failure_keeps_its_reservation_after_retry_and_resume(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [OUTLINE, OSError("connection lost without usage"), CHAPTERS[0]]
    control["unmeasured_errors"] = True
    report = draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1)
    assert report["saved_chapter_count"] == 1 and len(calls) == 3
    state = json.loads((tmp_path / "draft-state.json").read_text(encoding="utf-8"))
    failed, succeeded = state["steps"]["chapter-001"]["attempts"]
    assert failed["status"] == "failed" and "usage" not in failed
    assert failed["reserved_tokens"] == 500 + 8192
    assert succeeded["status"] == "completed"
    charged = report["metrics"]["charged_tokens"]
    assert charged == failed["reserved_tokens"] + 2 * 520
    assert report["metrics"]["unmeasured_requests"] == 1
    report = draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1, resume=True)
    assert len(calls) == 3 and report["metrics"]["charged_tokens"] == charged


def test_resume_refuses_missing_state_when_saved_prose_remains(runtime, tmp_path):
    calls, _ = runtime
    draft_story.run_draft_debug(snapshot(), tmp_path, chapter_limit=1)
    chapter_path = tmp_path / "chapters" / "chapter-001.md"
    saved = chapter_path.read_bytes()
    (tmp_path / "draft-state.json").unlink()
    with pytest.raises(ValueError):
        draft_story.run_draft_debug(snapshot(), tmp_path, resume=True)
    assert len(calls) == 2 and chapter_path.read_bytes() == saved
