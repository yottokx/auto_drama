"""Does the STEP4 chapter allocation share the story out more evenly when it sees each chapter's size?

The allocation cuts the saved chain of events into three chapters at event ends. Most
saved plans give chapter 1 only the first event (two steps) and the last chapter the rest.
This replays each saved allocation request, with the model that made it, unchanged except for:

  current  the saved request as it is
  sized    the request lists every way to cut at event ends with each chapter's steps, and
           asks for chapters of similar size (10-minute episodes) at natural breaks
  step     a chapter may also end between two steps of one event (last_step instead of
           last_event), with the same size guidance and each step's number

Only the allocation step runs (STEP4); no chapter is written. Nothing in the product changes.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from services.worker.generation import pipeline
from services.worker.generation.llm import write_json
from services.worker.generation.script_plot import (
    SIZED_ALLOCATION_RULE,
    STEP_ALLOCATION_RULE,
    ChapterAllocation,
    StepChapterAllocation,
    StoryChain,
    allocate_chain,
    allocate_chain_by_steps,
    boundary_options,
    chapter_step_counts,
    numbered_chain_steps,
)
from services.worker.model_config import select_config

MARKER = "全3章へ配分する保存済みプロット（全て未実施）:\n"
EQUAL = "章数を増減せず、出来事を均等に三分割する必要もありません。"
EVENT_RULE = ("各章は連続した一つ以上の出来事を担当します。last_eventは入力のeventsの1始まりの位置です。"
              "全出来事を一度ずつ、抜け・重複・並べ替えなしで割り当てます。")
SUFFIX = re.compile(r"出来事は全(\d+)個です。last_eventはevent_numberを使い、3章の値を狭義昇順にし、第3章の値は必ず\d+にします。")
ARMS = ("current", "sized", "step")


def load_run(run: Path) -> dict:
    saved = json.loads((run / "requests" / "chapter-allocation-1.json").read_text(encoding="utf-8"))["request"]
    chain = StoryChain.model_validate_json((run / "story-chain.json").read_text(encoding="utf-8"))
    experiment = json.loads((run / "experiment.json").read_text(encoding="utf-8"))
    current = json.loads((run / "chapter-allocation.json").read_text(encoding="utf-8"))
    return {"run": run.name, "request": saved, "chain": chain, "profile": experiment["profile"],
            "saved_steps": chapter_step_counts(chain, [row["last_event"] for row in current["chapters"]])}


def _brief(user: str, value: dict) -> str:
    head, found, _ = user.partition(MARKER)
    if not found:
        raise ValueError("The saved allocation request has no plot section.")
    return head + MARKER + json.dumps(value, ensure_ascii=False)


def variants(row: dict) -> dict[str, dict]:
    request, chain = row["request"], row["chain"]
    system, user = request["messages"][0]["content"], request["messages"][-1]["content"]
    if EQUAL not in system or EVENT_RULE not in system or not SUFFIX.search(system):
        raise ValueError(f"{row['run']}: the saved instruction differs from the expected wording.")
    core = chain.core.model_dump(mode="json")
    events = [{"event_number": n, **event.model_dump(mode="json")} for n, event in enumerate(chain.events, 1)]
    total = sum(len(event.steps) for event in chain.events)

    sized = copy.deepcopy(request)
    sized["messages"][0]["content"] = system.replace(EQUAL, "章数を増減しません。").replace(
        EVENT_RULE, EVENT_RULE + SIZED_ALLOCATION_RULE)
    sized["messages"][-1]["content"] = _brief(user, {
        "core": core, "event_count": len(events), "total_steps": total, "events": events,
        "boundary_options": boundary_options(chain)})

    step = copy.deepcopy(request)
    count = SUFFIX.search(system).group(1)
    step["messages"][0]["content"] = SUFFIX.sub(
        f"手順は全{total}個です。last_stepはstep_numberを使い、3章の値を狭義昇順にし、第3章の値は必ず{total}にします。",
        system.replace(EQUAL, "章数を増減しません。").replace(EVENT_RULE, STEP_ALLOCATION_RULE))
    del count
    step["messages"][-1]["content"] = _brief(user, {
        "core": core, "event_count": len(events), "total_steps": total,
        "target_steps_per_chapter": round(total / 3, 1), "events": numbered_chain_steps(chain)})
    schema = copy.deepcopy(request["response_format"]["json_schema"]["schema"])
    boundary = schema["$defs"].pop("ChapterBoundary")
    fields = {key: value for key, value in boundary["properties"].items() if key != "last_event"}
    fields["last_step"] = {"type": "integer", "minimum": 1, "maximum": total, "title": "Last Step",
                           "description": "当章の最後に割り当てる手順の通し番号（1始まり）。"}
    boundary["properties"] = fields
    boundary["required"] = [key if key != "last_event" else "last_step" for key in boundary["required"]]
    boundary["title"] = "StepChapterBoundary"
    schema["$defs"]["StepChapterBoundary"] = boundary
    schema["properties"]["chapters"]["items"] = {"$ref": "#/$defs/StepChapterBoundary"}
    step["response_format"] = copy.deepcopy(request["response_format"])
    step["response_format"]["json_schema"]["schema"] = schema
    return {"current": copy.deepcopy(request), "sized": sized, "step": step}


def summary(arm: str, answer: dict, chain: StoryChain) -> dict:
    if arm == "step":
        allocation = StepChapterAllocation.model_validate(answer)
        plot = allocate_chain_by_steps(chain, allocation)
        ends = [row.last_step for row in allocation.chapters]
    else:
        allocation = ChapterAllocation.model_validate(answer)
        plot = allocate_chain(chain, allocation)
        ends = [row.last_event for row in allocation.chapters]
    return {"ends": ends,
            "steps": [sum(len(event.steps) for event in chapter.events) for chapter in plot.chapters],
            "events": [len(chapter.events) for chapter in plot.chapters],
            "split_events": sum(len(chapter.events) for chapter in plot.chapters) - len(chain.events),
            "chapters": [{"title": chapter.title, "role": chapter.role,
                          "first": f"{chapter.events[0].steps[0].character_id}: {chapter.events[0].steps[0].action}",
                          "last": f"{chapter.events[-1].steps[-1].character_id}: {chapter.events[-1].steps[-1].result}",
                          "topics": len(chapter.conversation_topics)} for chapter in plot.chapters]}


def report(results: list[dict], output: Path) -> None:
    lines = ["# 章への割り当て（STEP4）の分量", "",
             "保存済みの出来事の連鎖と章割りの依頼を、作ったモデルで再生成した結果。各章の手順数。", "",
             "| 作品 | モデル | 出来事 | 保存済み | current | sized | step |", "| --- | --- | ---: | --- | --- | --- | --- |"]
    for row in results:
        cells = []
        for arm in ARMS:
            value = row["arms"].get(arm, {})
            cells.append("失敗" if "error" in value else "/".join(map(str, value.get("summary", {}).get("steps", []))) or "-")
        lines.append(f"| {row['run']} | {row['model']} | {row['events']} | {'/'.join(map(str, row['saved_steps']))} | "
                     + " | ".join(cells) + " |")
    lines += ["", "## 章ごとの題名・最初の手順・最後の結果", ""]
    for row in results:
        lines.append(f"### {row['run']}（{row['model']}）")
        for arm in ARMS:
            value = row["arms"].get(arm, {})
            if "error" in value:
                lines.append(f"- {arm}: 失敗 {value['error'][:200]}")
                continue
            result = value.get("summary")
            if not result:
                continue
            lines.append(f"- {arm}: 区切り {result['ends']}、手順 {result['steps']}、分割された出来事 {result['split_events']}")
            for number, chapter in enumerate(result["chapters"], 1):
                lines.append(f"  - 第{number}章「{chapter['title']}」: {chapter['first']} … {chapter['last']}"
                             f"（会話材料{chapter['topics']}）")
        lines.append("")
    (output / "report.md").write_bytes(("\n".join(lines) + "\n").encode("utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("runs", nargs="+", type=Path, help="text experiment directories with a saved allocation")
    parser.add_argument("--dry", action="store_true")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = [load_run(path) for path in args.runs]
    for row in rows:
        variants(row)
        print(row["run"], row["profile"]["model_id"], "events", len(row["chain"].events), "saved", row["saved_steps"])
    if args.dry:
        return 0
    from services.worker.generation.model_routing import RoutedLLM
    from services.worker.generation.processes import gpu_lock

    saved = output / "results.json"
    results = json.loads(saved.read_text(encoding="utf-8")) if saved.is_file() else []
    by_model: dict[str, list[dict]] = {}
    for row in rows:
        by_model.setdefault(row["profile"]["model_id"], []).append(row)
    base = pipeline.load_config()
    for model, group in by_model.items():
        profile = group[0]["profile"]
        config = copy.deepcopy(select_config(base, {"profile": profile}, pipeline.ROOT))
        config["llm"]["total_timeout_seconds"] = 4 * 3600
        payload = {"seed": 20261018, "profile": profile, "profiles": {}, "story_workflow_version": 2,
                   "workflow_policy": "script_continuation_v1"}
        with (gpu_lock(pipeline.ROOT / "services/worker/cache/m2/gpu.lock", config["gpu_lock_timeout_seconds"]),
              RoutedLLM(pipeline.ROOT, config, payload, output / "llm" / model) as llm):
            for row in group:
                record = next((value for value in results if value["run"] == row["run"]), None)
                if record is None:
                    record = {"run": row["run"], "model": model, "events": len(row["chain"].events),
                              "saved_steps": row["saved_steps"], "arms": {}}
                    results.append(record)
                for arm, request in variants(row).items():
                    if "summary" in record["arms"].get(arm, {}):
                        continue
                    start = time.monotonic()
                    try:
                        llm.select_purpose("script-allocation")
                        reply = llm.chat("script-allocation", request["messages"], seed=request["seed"],
                                         response_format=request["response_format"],
                                         max_tokens=request["max_tokens"])
                        answer = json.loads(reply.get("content") or "")
                        record["arms"][arm] = {"summary": summary(arm, answer, row["chain"]), "answer": answer}
                    except (OSError, RuntimeError, ValueError, TypeError) as exc:
                        record["arms"][arm] = {"error": f"{type(exc).__name__}: {exc}"[:500]}
                    record["arms"][arm]["seconds"] = round(time.monotonic() - start, 1)
                    write_json(saved, results)
                    print(json.dumps({"run": row["run"], "arm": arm,
                                      "steps": record["arms"][arm].get("summary", {}).get("steps"),
                                      "error": record["arms"][arm].get("error")}, ensure_ascii=False), flush=True)
    report(results, output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
