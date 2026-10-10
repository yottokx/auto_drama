"""A production without alternate outfits sends and saves exactly what it did before them."""

import hashlib
import json
import os
from pathlib import Path

from services.worker.generation import script_continuation as runner
from tests.unit.test_script_continuation_run import (
    FIRST,
    HANDOFF,
    SECOND,
    chapter_plan,
    empty_cast,
    staging,
    story_chain,
    story_chain_draft,
    three_chapter_snapshot,
)
from tests.unit.test_script_continuation_run import runtime as runtime  # noqa: PLC0414

BASELINE = Path(__file__).parents[1] / "fixtures" / "script-requests-without-outfits.json"
THIRD = ("aoi: 片付けの前に、入口から一度だけ見てみよう。\n"
         "NARRATOR: 蓮は照明を落とし、入口の側へ回った。\n"
         "ren: ここからでも札と写真は別々に読める。\n")


def _digest(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def _responses():
    chain = story_chain()
    chain["events"] = chain["events"][:3]
    presentation = {f"chapter_{n}": {"title": f"章{n}", "role": "結果を受けて進む。", "conversation_topics": [
        {"character_ids": ["aoi", "ren"], "topic": f"会話{n}", "exchange": "軽口を返す"}]} for n in (1, 2, 3)}
    presentation["foreshadowing"] = []
    return [empty_cast(), story_chain_draft(chain), presentation,
            chapter_plan("s1"), FIRST, staging(FIRST),
            HANDOFF, chapter_plan("s1", continued=True), SECOND, staging(SECOND),
            HANDOFF, chapter_plan("s1", continued=True), THIRD, staging(THIRD)]


def test_requests_and_saved_chapters_are_unchanged_without_outfits(runtime, tmp_path):
    calls, control = runtime
    control["responses"] = _responses()
    report = runner.run_script_debug(three_chapter_snapshot(), tmp_path)
    assert report["status"] == "script_complete"
    observed = {"requests": [{"purpose": call["purpose"],
                              "sha256": _digest({"messages": call["messages"], "extra": call["extra"]})}
                             for call in calls],
                "plot": _digest(json.loads((tmp_path / "plot.json").read_text(encoding="utf-8"))),
                "chapters": [{name: hashlib.sha256(
                    (tmp_path / "exports" / f"chapter-{number:03d}" / name).read_bytes()).hexdigest()
                    for name in ("narrative.json", "script.json")} for number in (1, 2, 3)]}
    if os.environ.get("AUTO_DRAMA_RECORD_OUTFIT_BASELINE") == "1":
        BASELINE.write_bytes((json.dumps(observed, ensure_ascii=False, indent=2) + "\n").encode())
    assert observed == json.loads(BASELINE.read_text(encoding="utf-8"))
