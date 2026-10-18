"""Experiment-only scripts without staging keep their text and make no staging call."""

from services.worker.generation import script_continuation as runner
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    chapter_plan,
    outline,
    read_json,
    snapshot,
)
from tests.unit.test_script_continuation_run import runtime as runtime  # noqa: PLC0414


def test_skipped_staging_keeps_the_script_and_gives_each_line_a_neutral_voice(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = [outline(), chapter_plan("s1"), FIRST, HANDOFF,
                            chapter_plan("s1", continued=True), SECOND]
    source = {"approval_snapshot": snapshot(), "script_options": {"skip_staging": True}}
    report = runner.run_script_debug(source, tmp_path)
    assert report["status"] == "script_complete"
    assert "script-staging" not in [call["purpose"] for call in calls]
    for number, raw in enumerate((FIRST, SECOND), 1):
        exported = tmp_path / "exports" / f"chapter-{number:03d}"
        assert (exported / "sources" / "s1.txt").read_text(encoding="utf-8") == raw
        script = read_json(exported / "script.json")
        assert {row["voice_emotion"] for row in script["utterances"]} == {"neutral"}
        assert {row["kind"] for row in script["directions"]} <= {"background", "enter", "exit"}
